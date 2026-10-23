//! Post processing and export: port of `SecondStepProcessor` (utils.py) and the
//! /postprocess* and /export endpoints of app.py. Files stay compatible:
//! cards_modified/{classifications.csv, excluded_cards.json, merged_annotations.csv}.

use crate::Progress;
use crate::project::{self, Project};
use anyhow::{Context, Result, bail};
use image::{DynamicImage, GrayImage, imageops::FilterType};
use ort::{session::Session, value::Tensor};
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

const CARD_EXTS: [&str; 3] = ["png", "jpg", "jpeg"];
const DPI: f64 = 300.0; // pages are rasterized at this resolution (pdf.rs)

pub fn model_path() -> PathBuf {
    crate::detect::model_dir("classifier", "models_classifier").join("model_classifier.onnx")
}

/// One card as shown in the post-processing grid.
pub struct Card {
    pub name: String,
    pub size: [u32; 2],
    pub modified: bool,
}

/// Cards plus their classification and exclusion state, read from disk.
#[derive(Default)]
pub struct State {
    pub cards: Vec<Card>,
    pub types: HashMap<String, String>,
    pub excluded: BTreeSet<String>,
}

impl State {
    pub fn load(p: &Project) -> Self {
        let (cards_dir, modified_dir) = (p.dir("cards"), p.dir("cards_modified"));
        let cards = project::list_files(&cards_dir, |n| project::has_ext(n, &CARD_EXTS))
            .into_iter()
            .map(|name| {
                let modified = modified_dir.join(&name).exists();
                let src = if modified { modified_dir.join(&name) } else { cards_dir.join(&name) };
                let (w, h) = image::image_dimensions(&src).unwrap_or((0, 0));
                Card { name, size: [w, h], modified }
            })
            .collect();
        let types = read_csv(&modified_dir.join("classifications.csv"))
            .map(|(_, rows)| rows.into_iter().filter_map(|r| Some((r.get("filename")?.clone(), r.get("type")?.clone()))).collect())
            .unwrap_or_default();
        Self { cards, types, excluded: read_excluded(&modified_dir) }
    }

    pub fn card_type(&self, name: &str) -> &str {
        self.types.get(name).map_or("ENT", |t| t.as_str()) // Python defaults unclassified cards to ENT
    }
}

/// Path to the current version of a card (post-processed copy if any).
pub fn card_path(p: &Project, name: &str) -> PathBuf {
    let m = p.dir("cards_modified").join(name);
    if m.exists() { m } else { p.dir("cards").join(name) }
}

pub struct Classifier {
    session: Session,
}

impl Classifier {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            bail!("Classifier not found: {} (run tools/export_onnx.py)", path.display());
        }
        Ok(Self { session: crate::detect::session(path)? })
    }

    /// (type, position, rotation) like `SecondStepProcessor.process_image`; the heads output raw
    /// values that Python compares with 0.5 directly (no sigmoid).
    pub fn classify(&mut self, gray: &GrayImage) -> Result<(&'static str, &'static str, &'static str)> {
        let small = image::imageops::resize(gray, 224, 224, FilterType::Triangle);
        let input: Vec<f32> = small.pixels().map(|p| (p[0] as f32 / 255.0 - 0.485) / 0.229).collect();
        let out = self.session.run(ort::inputs!["image" => Tensor::from_array(([1, 1, 224, 224], input))?])?;
        let v = |k: &str| -> Result<f32> { Ok(out[k].try_extract_tensor::<f32>()?.1[0]) };
        Ok((
            if v("type")? < 0.5 { "ENT" } else { "FRAG" },
            if v("position")? < 0.5 { "BOTTOM" } else { "TOP" },
            if v("rotation")? < 0.5 { "LEFT" } else { "RIGHT" },
        ))
    }
}

/// "Process All Images": classify every card, auto-orient it, save to cards_modified.
pub fn process(p: &mut Project, flip_v: bool, flip_h: bool, prog: &Progress) -> Result<String> {
    prog.set(0, 1, "Loading classifier…");
    let mut clf = Classifier::load(&model_path())?;
    let (cards_dir, out_dir) = (p.dir("cards"), p.dir("cards_modified"));
    std::fs::create_dir_all(&out_dir)?;
    let cards = project::list_files(&cards_dir, |n| project::has_ext(n, &CARD_EXTS));
    if cards.is_empty() {
        bail!("No card images found: extract cards first.");
    }
    let mut rows = vec![];
    for (i, name) in cards.iter().enumerate() {
        if prog.cancelled() {
            break;
        }
        prog.set(i, cards.len(), format!("Classifying {name}"));
        let gray = match image::open(cards_dir.join(name)) {
            Ok(img) => img.to_luma8(),
            Err(e) => {
                eprintln!("Error processing {name}: {e}");
                continue;
            }
        };
        let (kind, pos, rot) = clf.classify(&gray)?;
        let mut out = gray;
        if flip_v && pos == "BOTTOM" {
            image::imageops::flip_vertical_in_place(&mut out);
        }
        if flip_h && rot == "LEFT" {
            image::imageops::flip_horizontal_in_place(&mut out);
        }
        out.save(out_dir.join(name))?;
        rows.push([name.clone(), kind.into(), pos.into(), rot.into()]);
    }
    if !rows.is_empty() {
        let mut w = csv::Writer::from_path(out_dir.join("classifications.csv"))?;
        w.write_record(["filename", "type", "position", "rotation"])?;
        for r in &rows {
            w.write_record(r)?;
        }
        w.flush()?;
    }
    Ok(format!("Classified {} cards", rows.len()))
}

pub fn flip(p: &Project, name: &str, vertical: bool) -> Result<()> {
    let mut img = image::open(card_path(p, name))?;
    img = if vertical { img.flipv() } else { img.fliph() };
    std::fs::create_dir_all(p.dir("cards_modified"))?;
    img.save(p.dir("cards_modified").join(name))?;
    Ok(())
}

/// Change ENT/FRAG for one card in classifications.csv (adding the row if the card was never classified).
pub fn set_type(p: &Project, name: &str, kind: &str) -> Result<()> {
    let path = p.dir("cards_modified").join("classifications.csv");
    let (mut header, mut rows) = read_csv(&path).unwrap_or_else(|| (["filename", "type", "position", "rotation"].map(String::from).to_vec(), vec![]));
    if !header.iter().any(|h| h == "type") {
        header.push("type".into());
    }
    match rows.iter_mut().find(|r| r.get("filename").map(String::as_str) == Some(name)) {
        Some(r) => {
            r.insert("type".into(), kind.into());
        }
        None => rows.push(HashMap::from([("filename".into(), name.into()), ("type".into(), kind.into())])),
    }
    std::fs::create_dir_all(p.dir("cards_modified"))?;
    write_csv(&path, &header, &rows)
}

fn read_excluded(dir: &Path) -> BTreeSet<String> {
    let Ok(text) = std::fs::read_to_string(dir.join("excluded_cards.json")) else { return BTreeSet::new() };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        Ok(v) => v["excluded"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect(),
        Err(_) => BTreeSet::new(),
    }
}

pub fn set_excluded(p: &Project, excluded: &BTreeSet<String>) -> Result<()> {
    let dir = p.dir("cards_modified");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("excluded_cards.json"), serde_json::to_string(excluded)?)?;
    Ok(())
}

type Rows = Vec<HashMap<String, String>>;

fn read_csv(path: &Path) -> Option<(Vec<String>, Rows)> {
    let mut r = csv::Reader::from_path(path).ok()?;
    let header: Vec<String> = r.headers().ok()?.iter().map(String::from).collect();
    let rows = r.records().flatten().map(|rec| header.iter().cloned().zip(rec.iter().map(String::from)).collect()).collect();
    Some((header, rows))
}

fn write_csv(path: &Path, header: &[String], rows: &[HashMap<String, String>]) -> Result<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record(header)?;
    for r in rows {
        w.write_record(header.iter().map(|h| r.get(h).map_or("", |v| v.as_str())))?;
    }
    w.flush()?;
    Ok(())
}

fn strip_ext(s: &str) -> &str {
    let lower = s.to_ascii_lowercase();
    for e in [".png", ".jpg", ".jpeg"] {
        if lower.ends_with(e) {
            return &s[..s.len() - e.len()];
        }
    }
    s
}

pub fn valid_acronym(a: &str) -> bool {
    !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Join mask_info.csv (+ any tabular columns) with classifications.csv on the card name,
/// saved as cards_modified/merged_annotations.csv like the Python export.
fn merge_annotations(p: &Project) -> Option<(Vec<String>, Rows)> {
    let (mut header, mut rows) = read_csv(&p.dir("cards").join("mask_info.csv"))?;
    let key = ["mask_file", "filename"].into_iter().find(|k| header.iter().any(|h| h == k))?.to_string();
    if let Some((cheader, crows)) = read_csv(&p.dir("cards_modified").join("classifications.csv")) {
        let by_name: HashMap<&str, &HashMap<String, String>> = crows.iter().filter_map(|r| Some((strip_ext(r.get("filename")?), r))).collect();
        // pandas suffixes clashing columns from the right side with "_class"
        let renamed: Vec<(String, String)> = cheader.iter().map(|c| (c.clone(), if header.contains(c) && c != "filename" { format!("{c}_class") } else { c.clone() })).collect();
        for (_, new) in &renamed {
            if !header.contains(new) {
                header.push(new.clone());
            }
        }
        for row in &mut rows {
            let base = strip_ext(row.get(&key).map_or("", |s| s.as_str())).to_string();
            if let Some(c) = by_name.get(base.as_str()) {
                for (old, new) in &renamed {
                    row.insert(new.clone(), c.get(old).cloned().unwrap_or_default());
                }
            }
        }
    }
    let _ = std::fs::create_dir_all(p.dir("cards_modified"));
    let _ = write_csv(&p.dir("cards_modified").join("merged_annotations.csv"), &header, &rows);
    Some((header, rows))
}

/// "Export Results": `ACR_1.png…` (300 DPI) + `ACR_metadata.csv` in one ZIP.
pub fn export(p: &mut Project, acronym: &str, zip_path: &Path, prog: &Progress) -> Result<String> {
    if !valid_acronym(acronym) {
        bail!("Acronym can only contain letters, numbers, and underscores");
    }
    let state = State::load(p);
    let cards: Vec<&Card> = state.cards.iter().filter(|c| !state.excluded.contains(&c.name)).collect();
    if cards.is_empty() {
        bail!("No cards to export");
    }
    let meta = merge_annotations(p);

    // metadata: id, type, then every other column alphabetically (as in Python)
    let drop = ["mask_file", "filename", "Filename", "filename_base", "file", "ID", "id"];
    let mut table: Vec<HashMap<String, String>> = vec![];
    for (i, c) in cards.iter().enumerate() {
        let ext = Path::new(&c.name).extension().and_then(|e| e.to_str()).unwrap_or("png");
        let mut row = HashMap::from([("id".to_string(), format!("{acronym}_{}.{ext}", i + 1))]);
        if let Some((header, rows)) = &meta {
            let key = ["mask_file", "filename", "Filename", "file"].into_iter().find(|k| header.iter().any(|h| h == k));
            let found = key.and_then(|k| rows.iter().find(|r| r.get(k).is_some_and(|v| v == &c.name || strip_ext(v) == strip_ext(&c.name))));
            if let Some(r) = found {
                for (k, v) in r {
                    if !drop.contains(&k.as_str()) {
                        row.insert(k.clone(), if v == "nan" || v == "None" { String::new() } else { v.clone() });
                    }
                }
            }
        }
        if !row.contains_key("type") && state.types.contains_key(&c.name) {
            row.insert("type".into(), state.types[&c.name].clone());
        }
        table.push(row);
    }
    let mut cols: BTreeSet<String> = table.iter().flat_map(|r| r.keys().cloned()).collect();
    cols.remove("id");
    let has_type = cols.remove("type");
    let header: Vec<String> = ["id".to_string()].into_iter().chain(has_type.then(|| "type".to_string())).chain(cols).collect();

    let file = std::fs::File::create(zip_path).with_context(|| format!("creating {}", zip_path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (i, c) in cards.iter().enumerate() {
        if prog.cancelled() {
            bail!("Export cancelled");
        }
        prog.set(i, cards.len(), format!("Adding {}", c.name));
        let src = card_path(p, &c.name);
        let ext = Path::new(&c.name).extension().and_then(|e| e.to_str()).unwrap_or("png");
        zip.start_file(format!("{acronym}_{}.{ext}", i + 1), opts)?;
        if ext.eq_ignore_ascii_case("png") {
            zip.write_all(&png_with_dpi(&image::open(&src)?)?)?;
        } else {
            // ponytail: legacy JPEG cards are copied as-is, without rewriting their DPI
            zip.write_all(&std::fs::read(&src)?)?;
        }
    }
    zip.start_file(format!("{acronym}_metadata.csv"), opts)?;
    let mut w = csv::Writer::from_writer(vec![]);
    w.write_record(&header)?;
    for r in &table {
        w.write_record(header.iter().map(|h| r.get(h).map_or("", |v| v.as_str())))?;
    }
    zip.write_all(&w.into_inner()?)?;
    zip.finish()?;
    let excluded = state.cards.len() - cards.len();
    Ok(format!("Exported {} cards to {} ({excluded} excluded)", cards.len(), zip_path.display()))
}

/// PNG with a pHYs chunk, so the files carry the 300 DPI the pages were rendered at.
fn png_with_dpi(img: &DynamicImage) -> Result<Vec<u8>> {
    let (color, data) = match img {
        DynamicImage::ImageLuma8(g) => (png::ColorType::Grayscale, g.as_raw().clone()),
        DynamicImage::ImageRgba8(_) | DynamicImage::ImageLumaA8(_) => (png::ColorType::Rgba, img.to_rgba8().into_raw()),
        _ => (png::ColorType::Rgb, img.to_rgb8().into_raw()),
    };
    let mut out = vec![];
    let mut enc = png::Encoder::new(&mut out, img.width(), img.height());
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    let ppm = (DPI / 0.0254).round() as u32;
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: ppm, yppu: ppm, unit: png::Unit::Meter }));
    enc.write_header()?.write_image_data(&data)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same labels as the torch model for cards dumped by tools/export_onnx.py.
    #[test]
    fn matches_python_classifier() {
        let Ok(mut clf) = Classifier::load(&model_path()) else {
            eprintln!("skipped: no classifier at {}", model_path().display());
            return;
        };
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/classifier.json");
        let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(golden).unwrap()).unwrap();
        for c in cases {
            let Ok(img) = image::open(c["image"].as_str().unwrap()) else { continue };
            let got = clf.classify(&img.to_luma8()).unwrap();
            let want = (c["type"].as_str().unwrap(), c["position"].as_str().unwrap(), c["rotation"].as_str().unwrap());
            assert_eq!(got, want, "{}", c["image"]);
        }
    }

    #[test]
    fn dpi_is_written() {
        let png = png_with_dpi(&DynamicImage::ImageLuma8(GrayImage::new(4, 4))).unwrap();
        let reader = png::Decoder::new(std::io::Cursor::new(png)).read_info().unwrap();
        let dims = reader.info().pixel_dims.unwrap();
        assert_eq!(dims.xppu, 11811);
    }
}
