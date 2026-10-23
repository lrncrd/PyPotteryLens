//! Project workspace on disk, format-compatible with the Python app
//! (project_manager.py + the *_polygons/_vessels/_scale.json sidecars in utils.py).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

pub const IMAGE_EXTS: [&str; 6] = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"];
pub const POLYGONS_SUFFIX: &str = "_polygons.json";
pub const VESSELS_SUFFIX: &str = "_vessels.json";
const SCALE_SUFFIX: &str = "_scale.json";
const FOLDERS: [&str; 7] = ["pdf_source", "images", "masks", "cards", "cards_modified", "exports", "models"];

pub type Polygon = Vec<[f64; 2]>;

// Unknown keys land in `extra` and are written back untouched, so the Python
// app never loses fields the Rust port doesn't know about.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Workflow {
    #[serde(default)]
    pub reviewed_images: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Settings {
    /// Python stores these as URLs (`/api/projects/<id>/image/<name>`); only the last segment matters.
    #[serde(default)]
    pub excluded_images: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Meta {
    pub project_id: String,
    pub project_name: String,
    #[serde(default)]
    pub last_modified: String,
    #[serde(default)]
    pub workflow_status: Workflow,
    #[serde(default)]
    pub settings: Settings,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone)]
pub struct Project {
    pub root: PathBuf,
    pub meta: Meta,
}

fn now() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

impl Project {
    pub fn open(root: &Path) -> Result<Self> {
        let file = root.join("project.json");
        let text = fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
        let meta = serde_json::from_str(&text).with_context(|| format!("parsing {}", file.display()))?;
        Ok(Self { root: root.to_path_buf(), meta })
    }

    pub fn save(&mut self) -> Result<()> {
        self.meta.last_modified = now();
        fs::write(self.root.join("project.json"), serde_json::to_string_pretty(&self.meta)?)?;
        Ok(())
    }

    pub fn dir(&self, sub: &str) -> PathBuf {
        self.root.join(sub)
    }

    pub fn status(&self, key: &str) -> Option<&Value> {
        self.meta.workflow_status.extra.get(key)
    }

    pub fn set_status(&mut self, key: &str, value: Value) {
        self.meta.workflow_status.extra.insert(key.into(), value);
    }

    /// Image file names in `images/`, natural-sorted (page_2 before page_10).
    pub fn images(&self) -> Vec<String> {
        list_files(&self.dir("images"), |n| has_ext(n, &IMAGE_EXTS))
    }

    pub fn is_excluded(&self, name: &str) -> bool {
        self.meta.settings.excluded_images.iter().any(|u| u.rsplit('/').next() == Some(name))
    }

    pub fn set_excluded(&mut self, name: &str, on: bool) {
        self.meta.settings.excluded_images.retain(|u| u.rsplit('/').next() != Some(name));
        if on {
            let url = format!("/api/projects/{}/image/{}", self.meta.project_id, name);
            self.meta.settings.excluded_images.push(url);
        }
    }

    pub fn is_reviewed(&self, name: &str) -> bool {
        self.meta.workflow_status.reviewed_images.iter().any(|n| n == name)
    }

    pub fn set_reviewed(&mut self, name: &str, on: bool) {
        let reviewed = &mut self.meta.workflow_status.reviewed_images;
        reviewed.retain(|n| n != name);
        if on {
            reviewed.push(name.into());
        }
        let n = reviewed.len();
        self.set_status("annotations_completed", json!(n));
    }

    /// Polygons + expansion for an image, like `read_polygons_sidecar`:
    /// falls back to the vessels sidecar; over-dense polygons get simplified and written back.
    pub fn read_polygons(&self, base: &str) -> (Vec<Polygon>, i64) {
        let masks = self.dir("masks");
        // ponytail: legacy *_mask_layer.png masks are not converted, the Python app still does that
        if let Some(v) = read_json(&masks.join(format!("{base}{POLYGONS_SUFFIX}"))) {
            let expansion = v["expansion_px"].as_i64().unwrap_or(8);
            let polys: Vec<Polygon> = serde_json::from_value(v["polygons"].clone()).unwrap_or_default();
            let dense = polys.iter().any(|p| p.len() > 25);
            let polys: Vec<Polygon> = polys
                .into_iter()
                .map(|p| {
                    if p.len() <= 25 {
                        return p;
                    }
                    let s = crate::geom::simplify_closed(&p);
                    if s.len() >= 3 { s } else { p }
                })
                .collect();
            if dense {
                let _ = self.write_polygons(base, &polys, expansion);
            }
            return (polys, expansion);
        }
        let vessels = read_json(&masks.join(format!("{base}{VESSELS_SUFFIX}")))
            .and_then(|v| serde_json::from_value(v["polygons"].clone()).ok())
            .unwrap_or_default();
        (vessels, 8)
    }

    pub fn write_polygons(&self, base: &str, polygons: &[Polygon], expansion: i64) -> Result<()> {
        let masks = self.dir("masks");
        fs::create_dir_all(&masks)?;
        let data = json!({ "image": base, "polygons": polygons, "expansion_px": expansion });
        fs::write(masks.join(format!("{base}{POLYGONS_SUFFIX}")), data.to_string())?;
        Ok(())
    }

    /// 1-based index of the project icon stamp (`"3.png"` → 3).
    pub fn icon(&self) -> usize {
        self.meta.extra.get("icon").and_then(|v| v.as_str()).and_then(|s| s.trim_end_matches(".png").parse().ok()).unwrap_or(1)
    }

    pub fn created(&self) -> &str {
        self.meta.extra.get("created_at").and_then(|v| v.as_str()).unwrap_or("")
    }

    pub fn delete(&self) -> Result<()> {
        // only ever remove a real project folder
        if !self.root.join("project.json").exists() {
            bail!("{} is not a project folder", self.root.display());
        }
        fs::remove_dir_all(&self.root)?;
        Ok(())
    }

    /// `write_scale_sidecar`: no scales = no file.
    pub fn write_scales(&self, base: &str, scales: &[Value]) -> Result<()> {
        let path = self.dir("masks").join(format!("{base}{SCALE_SUFFIX}"));
        if scales.is_empty() {
            if path.exists() {
                fs::remove_file(path)?;
            }
            return Ok(());
        }
        fs::write(path, json!({ "image": base, "scales": scales }).to_string())?;
        Ok(())
    }

    pub fn read_scales(&self, base: &str) -> Vec<Value> {
        read_json(&self.dir("masks").join(format!("{base}{SCALE_SUFFIX}")))
            .and_then(|v| v["scales"].as_array().cloned())
            .unwrap_or_default()
    }
}

/// All projects under `root`, most recently modified first.
pub fn list(root: &Path) -> Vec<Project> {
    let mut out: Vec<Project> = fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| Project::open(&e.path()).ok())
        .collect();
    out.sort_by(|a, b| b.meta.last_modified.cmp(&a.meta.last_modified));
    out
}

pub fn create(root: &Path, name: &str, description: &str, icon: usize) -> Result<Project> {
    let safe: String = name.chars().filter(|c| c.is_alphanumeric() || " -_".contains(*c)).collect();
    let safe = safe.trim().replace(' ', "_");
    if safe.is_empty() {
        bail!("Project name needs at least one letter or digit");
    }
    let id = format!("{safe}_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let root = root.join(&id);
    if root.exists() {
        bail!("Project already exists: {id}");
    }
    for f in FOLDERS {
        fs::create_dir_all(root.join(f))?;
    }
    let meta = json!({
        "project_id": id,
        "project_name": name,
        "description": description,
        "icon": format!("{icon}.png"),
        "created_at": now(),
        "last_modified": now(),
        "workflow_status": {
            "pdf_processed": false, "pdf_count": 0, "images_extracted": 0, "model_applied": false,
            "masks_extracted": 0, "annotations_completed": 0, "total_images": 0, "reviewed_images": []
        },
        "settings": { "model_file": null, "confidence_threshold": 0.5, "excluded_images": [] }
    });
    let mut p = Project { root, meta: serde_json::from_value(meta)? };
    p.save()?;
    Ok(p)
}

pub fn has_ext(name: &str, exts: &[&str]) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| exts.contains(&e.to_ascii_lowercase().as_str()))
}

pub fn stem(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(s, _)| s)
}

pub fn list_files(dir: &Path, keep: impl Fn(&str) -> bool) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| keep(n))
        .collect();
    names.sort_by(|a, b| natural_cmp(a, b));
    names
}

/// Natural sort: digit runs compare as numbers, the rest case-insensitively.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    fn chunks(s: &str) -> Vec<(bool, String)> {
        let mut out: Vec<(bool, String)> = vec![];
        for c in s.chars() {
            let d = c.is_ascii_digit();
            match out.last_mut() {
                Some((ld, buf)) if *ld == d => buf.push(c),
                _ => out.push((d, c.to_string())),
            }
        }
        out
    }
    for (x, y) in chunks(a).iter().zip(chunks(b).iter()) {
        let ord = match (x, y) {
            ((true, x), (true, y)) => {
                let (x, y) = (x.trim_start_matches('0'), y.trim_start_matches('0'));
                x.len().cmp(&y.len()).then_with(|| x.cmp(y))
            }
            ((_, x), (_, y)) => x.to_lowercase().cmp(&y.to_lowercase()),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    chunks(a).len().cmp(&chunks(b).len())
}

/// px/cm for a card centred at `c`, like `_assign_px_per_cm`: the smallest
/// zone containing it wins, then the first global scale.
/// px/cm of one scale entry: the calibrated value if any, else ruler length / real_cm (`computeScaleRatio`).
pub fn scale_ratio(s: &Value) -> Option<f64> {
    if let Some(cal) = s["calibrated_px_per_cm"].as_f64().filter(|v| *v > 0.0) {
        return Some(cal);
    }
    let p = |k: &str, i: usize| s[k][i].as_f64().unwrap_or(0.0);
    let dist = (p("p2", 0) - p("p1", 0)).hypot(p("p2", 1) - p("p1", 1));
    let cm = s["real_cm"].as_f64().unwrap_or(0.0);
    (cm > 0.0 && dist > 0.0).then(|| dist / cm)
}

pub fn scale_zone(s: &Value) -> Option<[f64; 4]> {
    let z = s["zone"].as_array().filter(|z| z.len() == 4)?;
    Some([0, 1, 2, 3].map(|i| z[i].as_f64().unwrap_or(0.0)))
}

pub fn assign_px_per_cm(scales: &[Value], c: (f64, f64)) -> Option<f64> {
    let ratio = |s: &Value| scale_ratio(s).map(|r| (r * 10.0).round() / 10.0);
    let zone = |s: &Value| scale_zone(s).map(|z| z.to_vec());
    let mut zoned: Vec<(f64, &Value)> = scales
        .iter()
        .filter_map(|s| {
            let z = zone(s)?;
            let inside = z[0].min(z[2]) <= c.0 && c.0 <= z[0].max(z[2]) && z[1].min(z[3]) <= c.1 && c.1 <= z[1].max(z[3]);
            inside.then(|| ((z[2] - z[0]).abs() * (z[3] - z[1]).abs(), s))
        })
        .collect();
    zoned.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let Some(r) = zoned.iter().find_map(|(_, s)| ratio(s)) {
        return Some(r);
    }
    scales.iter().find(|s| zone(s).is_none()).and_then(|s| ratio(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_sort() {
        let mut v = vec!["Osa_page_10a.jpg", "Osa_page_2b.jpg", "Osa_page_2a.jpg", "Osa_page_1a.jpg"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, ["Osa_page_1a.jpg", "Osa_page_2a.jpg", "Osa_page_2b.jpg", "Osa_page_10a.jpg"]);
    }

    #[test]
    fn px_per_cm_zone_beats_global() {
        let scales: Vec<Value> = serde_json::from_value(json!([
            { "p1": [0, 0], "p2": [100, 0], "real_cm": 5, "zone": null },
            { "p1": [0, 0], "p2": [30, 40], "real_cm": 1, "zone": [0, 0, 10, 10] }
        ]))
        .unwrap();
        assert_eq!(assign_px_per_cm(&scales, (5.0, 5.0)), Some(50.0));
        assert_eq!(assign_px_per_cm(&scales, (50.0, 50.0)), Some(20.0));
    }
}
