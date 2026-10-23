//! YOLOv8-seg inference via ONNX Runtime: port of `ModelProcessor._process_single_image`
//! plus the ultralytics post-processing it relies on (letterbox, NMS, retina masks, masks.xy).

use crate::Progress;
use crate::geom;
use crate::project::{self, Polygon, Project};
use anyhow::{Context, Result, anyhow};
use image::{GrayImage, Luma, RgbImage, imageops::FilterType};
use imageproc::contours::{BorderType, find_contours};
use ort::{ep, session::Session, value::Tensor};
use serde_json::json;
use std::path::{Path, PathBuf};

const SIZE: usize = 800; // training imgsz, baked into the exported ONNX
const IOU: f32 = 0.7; // ultralytics predict default
const MAX_DET: usize = 300;

pub fn model_path() -> PathBuf {
    model_dir("yolo", "models_vision").join("BasicModelv8_v01.onnx")
}

/// PYPOTTERY_MODEL_CACHE/<sub>, else the suite's shared ../model_cache/<sub>, else ./<local>.
pub fn model_dir(sub: &str, local: &str) -> PathBuf {
    let shared = PathBuf::from("../model_cache");
    std::env::var_os("PYPOTTERY_MODEL_CACHE")
        .map(PathBuf::from)
        .or_else(|| shared.is_dir().then_some(shared))
        .map_or_else(|| local.into(), |c| c.join(sub))
}

pub struct Detector {
    session: Session,
}

/// ONNX Runtime allows one environment per process: build it once, share it between models.
pub fn session(path: &Path) -> Result<Session> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    let env = match ENV.get() {
        Some(e) => e,
        None => {
            let e = ort::init().build()?;
            ENV.get_or_init(|| e)
        }
    };
    Session::builder(env)?
        .with_execution_providers([ep::CoreML::default().build()])
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

impl Detector {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(anyhow!("Model not found: {} (run tools/export_onnx.py)", path.display()));
        }
        Ok(Self { session: session(path)? })
    }

    pub fn polygons(&mut self, img: &RgbImage, conf: f32) -> Result<Vec<Polygon>> {
        let (w0, h0) = (img.width() as f32, img.height() as f32);

        // letterbox to SIZE×SIZE, grey 114 padding, centred
        let gain = (SIZE as f32 / h0).min(SIZE as f32 / w0);
        let (nw, nh) = ((w0 * gain).round() as u32, (h0 * gain).round() as u32);
        let pad_x = ((SIZE as f32 - nw as f32) / 2.0 - 0.1).round();
        let pad_y = ((SIZE as f32 - nh as f32) / 2.0 - 0.1).round();
        let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);
        let plane = SIZE * SIZE;
        let mut input = vec![114.0 / 255.0; 3 * plane];
        for (x, y, p) in resized.enumerate_pixels() {
            let i = (y as usize + pad_y as usize) * SIZE + x as usize + pad_x as usize;
            for c in 0..3 {
                input[c * plane + i] = p[c] as f32 / 255.0;
            }
        }

        let outputs = self.session.run(ort::inputs!["images" => Tensor::from_array(([1, 3, SIZE, SIZE], input))?])?;
        let (pshape, preds) = outputs["output0"].try_extract_tensor::<f32>()?;
        let (mshape, protos) = outputs["output1"].try_extract_tensor::<f32>()?;
        let (rows, n) = (pshape[1] as usize, pshape[2] as usize); // 4 box + 1 class + 32 coefs
        let (nm, mh, mw) = (mshape[1] as usize, mshape[2] as usize, mshape[3] as usize);
        let at = |r: usize, i: usize| preds[r * n + i];

        // candidates over threshold → NMS, best first
        let mut cands: Vec<(f32, [f32; 4], usize)> = (0..n)
            .filter(|&i| at(4, i) > conf)
            .map(|i| {
                let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
                (at(4, i), [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], i)
            })
            .collect();
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut keep: Vec<(f32, [f32; 4], usize)> = vec![];
        for c in cands {
            if keep.len() < MAX_DET && keep.iter().all(|k| iou(&k.1, &c.1) <= IOU) {
                keep.push(c);
            }
        }

        // scale_masks: the proto area that maps onto the unpadded image
        let mgain = (mh as f32 / h0).min(mw as f32 / w0);
        let (mpw, mph) = ((mw as f32 - w0 * mgain) / 2.0, (mh as f32 - h0 * mgain) / 2.0);
        let (mtop, mleft) = ((mph - 0.1).round(), (mpw - 0.1).round());
        let (ch, cw) = (mh as f32 - (mph + 0.1).round() - mtop, mw as f32 - (mpw + 0.1).round() - mleft);

        let mut out = vec![];
        for (_, b, i) in keep {
            let coefs: Vec<f32> = (0..nm).map(|k| at(rows - nm + k, i)).collect();
            let logits: Vec<f32> = (0..mh * mw).map(|j| (0..nm).map(|k| coefs[k] * protos[k * mh * mw + j]).sum()).collect();
            // box back to original pixels (scale_boxes + clip)
            let bx = |v: f32, pad: f32, max: f32| ((v - pad) / gain).clamp(0.0, max);
            let (x1, y1, x2, y2) = (bx(b[0], pad_x, w0), bx(b[1], pad_y, h0), bx(b[2], pad_x, w0), bx(b[3], pad_y, h0));
            // retina mask inside the box: bilinear upsample (align_corners=False) of the proto crop, logit > 0
            let (ox, oy) = (x1.floor() as u32, y1.floor() as u32);
            let (bw, bh) = (x2.ceil() as u32 - ox, y2.ceil() as u32 - oy);
            if bw == 0 || bh == 0 {
                continue;
            }
            let sample = |x: f32, y: f32| -> f32 {
                let sx = ((x + 0.5) * cw / w0 - 0.5).max(0.0);
                let sy = ((y + 0.5) * ch / h0 - 0.5).max(0.0);
                let (x0, y0) = (sx.floor().min(cw - 1.0), sy.floor().min(ch - 1.0));
                let (x1, y1) = ((x0 + 1.0).min(cw - 1.0), (y0 + 1.0).min(ch - 1.0));
                let (fx, fy) = (sx - x0, sy - y0);
                let v = |x: f32, y: f32| logits[(y + mtop) as usize * mw + (x + mleft) as usize];
                (v(x0, y0) * (1.0 - fx) + v(x1, y0) * fx) * (1.0 - fy) + (v(x0, y1) * (1.0 - fx) + v(x1, y1) * fx) * fy
            };
            // 1px empty border so contours close on the box edge
            let mask = GrayImage::from_fn(bw + 2, bh + 2, |mx, my| {
                let (x, y) = (ox as f32 + mx as f32 - 1.0, oy as f32 + my as f32 - 1.0);
                let inside = mx > 0 && my > 0 && mx <= bw && my <= bh && x >= x1 && x < x2 && y >= y1 && y < y2;
                Luma([if inside && sample(x, y) > 0.0 { 255 } else { 0 }])
            });
            // ponytail: ultralytics stitches multi-part masks into one outline; we keep the largest part
            let contour = find_contours::<i32>(&mask)
                .into_iter()
                .filter(|c| c.border_type == BorderType::Outer && c.parent.is_none())
                .map(|c| c.points.iter().map(|p| [(p.x + ox as i32 - 1) as f64, (p.y + oy as i32 - 1) as f64]).collect::<Polygon>())
                .max_by(|a, b| geom::area(a).total_cmp(&geom::area(b)));
            if let Some(c) = contour.filter(|c| c.len() >= 3) {
                let poly = geom::simplify_closed(&c);
                if poly.len() >= 3 {
                    out.push(poly);
                }
            }
        }
        Ok(out)
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

/// `apply_model_to_project`: one `_polygons.json` per non-excluded image.
pub fn run_project(p: &mut Project, conf: f32, diagnostic: bool, prog: &Progress) -> Result<String> {
    prog.set(0, 1, "Loading model…");
    let mut det = Detector::load(&model_path())?;
    let mut images: Vec<String> = p.images().into_iter().filter(|n| !p.is_excluded(n)).collect();
    if diagnostic {
        images.truncate(25);
    }
    let total = images.len();
    let mut done = 0;
    for (i, name) in images.iter().enumerate() {
        if prog.cancelled() {
            break;
        }
        prog.set(i, total, name);
        let polys = image::open(p.dir("images").join(name)).map_err(anyhow::Error::from).and_then(|img| det.polygons(&img.to_rgb8(), conf));
        match polys {
            Ok(polys) => p.write_polygons(project::stem(name), &polys, 8)?,
            Err(e) => eprintln!("Error processing image {name}: {e:#}"),
        }
        done += 1;
    }
    let sidecars = project::list_files(&p.dir("masks"), |n| n.ends_with(project::POLYGONS_SUFFIX)).len();
    p.set_status("model_applied", json!(true));
    p.set_status("masks_extracted", json!(sidecars));
    p.save()?;
    Ok(format!("Model applied: {done} of {total} images processed (confidence {conf:.2})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(polys: &[Polygon], w: u32, h: u32) -> GrayImage {
        let mut m = GrayImage::new(w, h);
        for p in polys {
            geom::fill(&mut m, p, (0.0, 0.0), false);
        }
        m
    }

    /// Compares against polygons dumped from the Python/torch pipeline by tools/export_onnx.py.
    /// Needs the exported model and the project images, so it skips when they are missing.
    #[test]
    fn matches_python_golden() {
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
        let Ok(mut det) = Detector::load(&model_path()) else {
            eprintln!("skipped: no model at {} (set PYPOTTERY_MODEL_CACHE)", model_path().display());
            return;
        };
        for entry in std::fs::read_dir(golden).unwrap().flatten() {
            let g: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(entry.path()).unwrap()).unwrap();
            if g.get("polygons").is_none() {
                continue; // other goldens (classifier.json)
            }
            let Ok(img) = image::open(g["image"].as_str().unwrap()) else { continue };
            let img = img.to_rgb8();
            let want: Vec<Polygon> = serde_json::from_value(g["polygons"].clone()).unwrap();
            let got = det.polygons(&img, 0.5).unwrap();
            let (a, b) = (raster(&want, img.width(), img.height()), raster(&got, img.width(), img.height()));
            let (mut inter, mut union) = (0usize, 0usize);
            for (pa, pb) in a.pixels().zip(b.pixels()) {
                inter += (pa[0] != 0 && pb[0] != 0) as usize;
                union += (pa[0] != 0 || pb[0] != 0) as usize;
            }
            let iou = if union == 0 { 1.0 } else { inter as f64 / union as f64 };
            println!("{}: python {} / rust {} polygons, IoU {iou:.3}", entry.file_name().display(), want.len(), got.len());
            assert_eq!(want.len(), got.len(), "{}", entry.path().display());
            assert!(iou > 0.95, "{}: IoU {iou:.3}", entry.path().display());
        }
    }
}
