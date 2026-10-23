//! Card extraction: port of `MaskExtractor.extract_masks_from_project` and helpers (utils.py).

use crate::Progress;
use crate::geom;
use crate::project::{self, IMAGE_EXTS, POLYGONS_SUFFIX, Polygon, Project, VESSELS_SUFFIX};
use anyhow::{Result, bail};
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::region_labelling::{Connectivity, connected_components};
use serde_json::json;
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

const MARGIN: u32 = 50;

pub fn extract(p: &mut Project, expansion: Option<i64>, clean: bool, prog: &Progress) -> Result<String> {
    let (masks, cards) = (p.dir("masks"), p.dir("cards"));
    std::fs::create_dir_all(&cards)?;
    let mut bases: Vec<String> = project::list_files(&masks, |n| {
        n.ends_with(POLYGONS_SUFFIX) || n.ends_with(VESSELS_SUFFIX) || n.ends_with("_mask_layer.png")
    })
    .iter()
    .map(|n| n.trim_end_matches(POLYGONS_SUFFIX).trim_end_matches(VESSELS_SUFFIX).trim_end_matches("_mask_layer.png").to_string())
    .collect();
    bases.dedup();
    if bases.is_empty() {
        bail!("No mask files found. Apply a model first.");
    }

    // (file, mask stem, px_per_cm, bbox)
    let mut rows: Vec<(String, String, Option<f64>, [u32; 4])> = vec![];
    let mut seen = HashSet::new();
    let mut duplicates = 0;

    for (idx, base) in bases.iter().enumerate() {
        if prog.cancelled() {
            break;
        }
        prog.set(idx, bases.len(), base);
        let Some(img_path) = IMAGE_EXTS.iter().map(|e| p.dir("images").join(format!("{base}.{e}"))).find(|f| f.exists()) else {
            eprintln!("Original image not found for {base}");
            continue;
        };
        let img = image::open(&img_path)?.to_rgb8();
        let (polys, sidecar_exp) = p.read_polygons(base);
        let exp = expansion.unwrap_or(sidecar_exp).max(0) as u32;
        let scales = p.read_scales(base);

        for (i, poly) in polys.iter().enumerate() {
            let Some((mut card, bbox)) = crop_polygon(&img, poly, exp) else { continue };
            whiteout_nested(&mut card, bbox, i, &polys, exp);
            if clean {
                clean_card(&mut card);
            }
            let mut h = DefaultHasher::new();
            (card.dimensions(), card.as_raw()).hash(&mut h);
            if !seen.insert(h.finish()) {
                duplicates += 1;
                continue;
            }
            let stem = format!("{base}_mask_layer_{i}");
            let mut out = RgbImage::from_pixel(card.width() + 2 * MARGIN, card.height() + 2 * MARGIN, Rgb([255; 3]));
            image::imageops::replace(&mut out, &card, MARGIN as i64, MARGIN as i64);
            out.save(cards.join(format!("{stem}.png")))?;

            let centre = ((bbox[0] + bbox[2]) as f64 / 2.0, (bbox[1] + bbox[3]) as f64 / 2.0);
            rows.push((base.clone(), stem, project::assign_px_per_cm(&scales, centre), bbox));
        }
    }

    if !rows.is_empty() {
        write_csvs(&cards, &rows)?;
    }

    let count = project::list_files(&cards, |n| n.ends_with(".png")).len();
    p.set_status("cards_extracted", json!(count));
    p.save()?;
    if rows.is_empty() {
        bail!("No cards were extracted. Check if masks are properly drawn.");
    }
    let mut msg = format!("Extracted {} cards", rows.len());
    if duplicates > 0 {
        msg += &format!(" ({duplicates} exact duplicates skipped)");
    }
    Ok(msg)
}

/// `_save_metadata`: mask_info.csv (+ px_per_cm only when some card has a scale, as pandas does)
/// and mask_info_annots.csv with Python-tuple bboxes.
fn write_csvs(cards: &std::path::Path, rows: &[(String, String, Option<f64>, [u32; 4])]) -> Result<()> {
    let with_scale = rows.iter().any(|r| r.2.is_some());
    let mut info = csv::Writer::from_path(cards.join("mask_info.csv"))?;
    let mut annots = csv::Writer::from_path(cards.join("mask_info_annots.csv"))?;
    let mut header = vec!["file", "mask_file"];
    if with_scale {
        header.push("px_per_cm");
    }
    info.write_record(&header)?;
    annots.write_record(["bbox", "mask_file"])?;
    for (file, stem, ratio, b) in rows {
        let mut rec = vec![file.clone(), stem.clone()];
        if with_scale {
            rec.push(ratio.map(|r| format!("{r:?}")).unwrap_or_default());
        }
        info.write_record(rec)?;
        annots.write_record([format!("({}, {}, {}, {})", b[0], b[1], b[2], b[3]), format!("{stem}.png")])?;
    }
    info.flush()?;
    annots.flush()?;
    Ok(())
}

/// `extract_polygon_with_expansion`: crop the polygon's bbox (+pad), whiten everything outside the dilated polygon.
fn crop_polygon(img: &RgbImage, poly: &Polygon, pad: u32) -> Option<(RgbImage, [u32; 4])> {
    if poly.len() < 3 {
        return None;
    }
    let xs = poly.iter().map(|q| q[0] as i64);
    let ys = poly.iter().map(|q| q[1] as i64);
    let (x, y) = (xs.clone().min()?, ys.clone().min()?);
    let (w, h) = (xs.max()? - x + 1, ys.max()? - y + 1);
    let pad = pad as i64;
    let x1 = (x - pad).max(0) as u32;
    let y1 = (y - pad).max(0) as u32;
    let x2 = (x + w + pad).min(img.width() as i64).max(0) as u32;
    let y2 = (y + h + pad).min(img.height() as i64).max(0) as u32;
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let mut mask = GrayImage::new(x2 - x1, y2 - y1);
    geom::fill(&mut mask, poly, (x1 as f64, y1 as f64), false);
    let mask = geom::dilate(&mask, pad as u32);
    let card = RgbImage::from_fn(x2 - x1, y2 - y1, |cx, cy| {
        if mask.get_pixel(cx, cy)[0] == 0 { Rgb([255; 3]) } else { *img.get_pixel(x1 + cx, y1 + cy) }
    });
    Some((card, [x1, y1, x2, y2]))
}

/// `_whiteout_nested_polygons`: a strictly smaller polygon whose centroid lies inside
/// this one has its own card, so blank it here.
fn whiteout_nested(card: &mut RgbImage, bbox: [u32; 4], me: usize, polys: &[Polygon], pad: u32) {
    let outer = &polys[me];
    let outer_area = geom::area(outer);
    for (j, other) in polys.iter().enumerate() {
        if j == me || other.len() < 3 || geom::area(other) >= outer_area {
            continue;
        }
        let n = other.len() as f64;
        let (cx, cy) = (other.iter().map(|q| q[0]).sum::<f64>() / n, other.iter().map(|q| q[1]).sum::<f64>() / n);
        if !geom::contains(outer, cx, cy) {
            continue;
        }
        let mut hole = GrayImage::new(card.width(), card.height());
        geom::fill(&mut hole, other, (bbox[0] as f64, bbox[1] as f64), true);
        let hole = geom::dilate(&hole, pad);
        for (x, y, v) in hole.enumerate_pixels() {
            if v[0] != 0 {
                card.put_pixel(x, y, Rgb([255; 3]));
            }
        }
    }
}

/// `clean_card_deterministic`: drop ink components (numbers, section profiles, neighbours'
/// fragments) that mostly fall outside the main vessel's closed silhouette + convex hull.
pub fn clean_card(card: &mut RgbImage) {
    const INK: f32 = 220.0;
    const CLOSE_R: u32 = 25;
    let (w, h) = card.dimensions();
    // cv2.COLOR_BGR2GRAY applied to an RGB array, as in Python
    let ink = GrayImage::from_fn(w, h, |x, y| {
        let p = card.get_pixel(x, y);
        let g = (0.114 * p[0] as f32 + 0.587 * p[1] as f32 + 0.299 * p[2] as f32).round();
        Luma([if g < INK { 255 } else { 0 }])
    });
    let total = ink.pixels().filter(|p| p[0] != 0).count();
    if total < 50 {
        return;
    }
    let labels = connected_components(&ink, Connectivity::Eight, Luma([0u8]));
    let n = labels.pixels().map(|p| p[0]).max().unwrap_or(0) as usize;
    if n <= 1 {
        return;
    }
    let mut areas = vec![0usize; n + 1];
    for p in labels.pixels() {
        areas[p[0] as usize] += 1;
    }
    let main = (1..=n).max_by_key(|&i| areas[i]).unwrap();
    let main_mask = GrayImage::from_fn(w, h, |x, y| Luma([if labels.get_pixel(x, y)[0] as usize == main { 255 } else { 0 }]));
    let dist2 = imageproc::distance_transform::euclidean_squared_distance_transform(&main_mask);
    let mut min_d = vec![f64::INFINITY; n + 1];
    for (x, y, l) in labels.enumerate_pixels() {
        let l = l[0] as usize;
        if l != 0 {
            min_d[l] = min_d[l].min(dist2.get_pixel(x, y)[0].sqrt());
        }
    }
    // significant pieces close to the vessel (handles, a slightly detached elevation half)
    let core_label = |l: usize| l == main || (areas[l] as f64 > 0.05 * total as f64 && min_d[l] < 25.0);
    let core = GrayImage::from_fn(w, h, |x, y| {
        let l = labels.get_pixel(x, y)[0] as usize;
        Luma([if l != 0 && core_label(l) { 255 } else { 0 }])
    });
    let closed = geom::close(&core, CLOSE_R);
    if closed.pixels().all(|p| p[0] == 0) {
        return;
    }
    let filled = geom::fill_holes(&closed);
    let hull = geom::hull_mask(&closed);
    let keep = GrayImage::from_fn(w, h, |x, y| Luma([filled.get_pixel(x, y)[0].max(hull.get_pixel(x, y)[0])]));
    let keep = geom::dilate(&keep, 2);

    let mut overlap = vec![0usize; n + 1];
    for (x, y, l) in labels.enumerate_pixels() {
        if keep.get_pixel(x, y)[0] != 0 {
            overlap[l[0] as usize] += 1;
        }
    }
    let remove: Vec<bool> = (0..=n).map(|l| l != 0 && (overlap[l] as f64) < 0.3 * areas[l] as f64).collect();
    if !remove.iter().any(|&r| r) {
        return;
    }
    let removal = GrayImage::from_fn(w, h, |x, y| Luma([if remove[labels.get_pixel(x, y)[0] as usize] { 255 } else { 0 }]));
    // erase a margin too (anti-aliasing, JPEG ringing), but never inside the kept area
    let removal = geom::dilate(&removal, 10);
    for (x, y, v) in removal.enumerate_pixels() {
        if v[0] != 0 && keep.get_pixel(x, y)[0] == 0 {
            card.put_pixel(x, y, Rgb([255; 3]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_card_drops_stray_number_keeps_vessel() {
        let mut card = RgbImage::from_pixel(300, 300, Rgb([255; 3]));
        let black = Rgb([0u8; 3]);
        // vessel: thick ring centred at (100,150) with stippling inside
        for (x, y, p) in card.enumerate_pixels_mut() {
            let d = ((x as f64 - 100.0).powi(2) + (y as f64 - 150.0).powi(2)).sqrt();
            if (60.0..66.0).contains(&d) || (x % 9 == 0 && y % 9 == 0 && d < 50.0) {
                *p = black;
            }
        }
        // "inventory number": a blob far away
        for y in 20..40 { for x in 250..270 { card.put_pixel(x, y, black); } }
        clean_card(&mut card);
        assert_eq!(card.get_pixel(260, 30), &Rgb([255; 3]), "number not removed");
        assert_eq!(card.get_pixel(162, 150), &black, "vessel outline removed");
        assert_eq!(card.get_pixel(99, 144), &black, "inner stippling removed");
    }
}
