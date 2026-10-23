//! "Calibrate Scales" dialog: histogram + KDE of every scale's px/cm across the whole project,
//! with draggable centre lines; Apply snaps each scale to the nearest line
//! (`calibrated_px_per_cm`). Port of the calibration part of annotation-tab.js.

use crate::project::{self, Project};
use crate::ui::{self, Kind};
use eframe::egui::{self, Color32, FontId, Margin, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const PAD_X: f32 = 40.0;

/// Distinct, stable colour per index (golden-angle hue spread), like `scaleColor`.
pub fn color(i: usize, l: f32) -> Color32 {
    let h = (i as f32 * 137.508) % 360.0 / 360.0;
    let s = 0.75;
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |mut t: f32| {
        t = t.rem_euclid(1.0);
        let v = if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        };
        (v * 255.0).round() as u8
    };
    Color32::from_rgb(f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0))
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// Gaussian KDE, Silverman bandwidth.
fn kde(values: &[f64], grid: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = if var > 0.0 { var.sqrt() } else { (values[0].abs() * 0.01).max(0.01) };
    let bw = (1.06 * std * n.powf(-0.2)).max(1e-6);
    grid.iter().map(|x| values.iter().map(|v| (-0.5 * ((x - v) / bw).powi(2)).exp()).sum::<f64>() / (n * bw * (2.0 * std::f64::consts::PI).sqrt())).collect()
}

fn grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
}

pub struct Calib {
    pages: BTreeMap<String, Vec<Value>>,
    entries: Vec<(String, usize, f64)>, // page, scale index, px/cm
    lines: Vec<f64>,
    range: (f64, f64),
    drag: Option<usize>,
}

impl Calib {
    pub fn open(p: &Project, images: &[String]) -> Self {
        let mut pages = BTreeMap::new();
        let mut entries = vec![];
        for name in images {
            let base = project::stem(name);
            let scales = p.read_scales(base);
            for (i, s) in scales.iter().enumerate() {
                if let Some(r) = project::scale_ratio(s).filter(|r| *r > 0.0) {
                    entries.push((base.to_string(), i, r));
                }
            }
            if !scales.is_empty() {
                pages.insert(base.to_string(), scales);
            }
        }
        let mut c = Self { pages, entries, lines: vec![], range: (0.0, 1.0), drag: None };
        if c.entries.len() >= 2 {
            let v: Vec<f64> = c.entries.iter().map(|e| e.2).collect();
            let (min, max) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
            let span = (max - min).max(1e-6);
            c.range = (min - span * 0.15 - 0.01, max + span * 0.15 + 0.01);
            // start from the two highest KDE peaks
            let g = grid(c.range.0, c.range.1, 200);
            let d = kde(&v, &g);
            let top = d.iter().cloned().fold(0.0, f64::max);
            let mut peaks: Vec<(f64, f64)> = (1..199).filter(|&i| d[i] > d[i - 1] && d[i] >= d[i + 1] && d[i] > top * 0.15).map(|i| (g[i], d[i])).collect();
            peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
            c.lines = peaks.iter().take(2).map(|p| round1(p.0)).collect();
            c.lines.sort_by(f64::total_cmp);
        }
        c
    }

    /// Snap every scale to its nearest line and save the touched pages; returns how many.
    pub fn apply(&mut self, p: &Project) -> anyhow::Result<usize> {
        let mut changed = std::collections::BTreeSet::new();
        for (page, i, r) in &self.entries {
            let nearest = self.lines.iter().cloned().min_by(|a, b| (a - r).abs().total_cmp(&(b - r).abs())).unwrap_or(*r);
            let s = &mut self.pages.get_mut(page).unwrap()[*i];
            if s["calibrated_px_per_cm"].as_f64() != Some(nearest) {
                s["calibrated_px_per_cm"] = json!(nearest);
                changed.insert(page.clone());
            }
        }
        for page in &changed {
            p.write_scales(page, &self.pages[page])?;
        }
        Ok(changed.len())
    }

    /// Some(true) = Apply, Some(false) = closed.
    pub fn ui(&mut self, ctx: &egui::Context) -> Option<bool> {
        let mut out = None;
        let r = egui::Modal::new(egui::Id::new("calib")).frame(ui::card().inner_margin(Margin::same(24))).show(ctx, |ui| {
            ui.set_width(640.0);
            ui::section_title(ui, ui::I_CHART, "Calibrate Scales");
            let pages = self.entries.iter().map(|e| &e.0).collect::<std::collections::BTreeSet<_>>().len();
            let enough = self.entries.len() >= 2;
            ui.label(RichText::new(if enough { format!("{} scales across {pages} page(s).", self.entries.len()) } else { "Need at least 2 measured scales across the project to calibrate.".into() }).color(ui::DIM));
            let (rect, resp) = ui.allocate_exact_size(vec2(640.0, 240.0), Sense::click_and_drag());
            ui.painter().rect_filled(rect, 8.0, ui::BG_PAGE);
            if enough {
                self.histogram(ui, rect, &resp);
            }
            ui.label(ui::muted("Click empty space to add a centre line · drag a line to move it · double-click a line to remove it."));
            ui.horizontal_wrapped(|ui| {
                let mut remove = None;
                for (i, v) in self.lines.iter().enumerate() {
                    ui::chip(ui, &format!("{v:.1} px/cm"), color(i, 0.42), Color32::WHITE, true);
                    if ui.small_button(ui::I_X).clicked() {
                        remove = Some(i);
                    }
                }
                if let Some(i) = remove {
                    self.lines.remove(i);
                }
                if self.lines.is_empty() {
                    ui.label(ui::muted("No centre lines yet: click the histogram to add one."));
                }
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui::button(ui, Kind::Secondary, "", "Cancel", vec2(120.0, 38.0), true).clicked() {
                    out = Some(false);
                }
                if ui::button(ui, Kind::Primary, ui::I_CHECK2, "Apply Calibration", vec2(0.0, 38.0), enough && !self.lines.is_empty()).clicked() {
                    out = Some(true);
                }
            });
        });
        if out.is_none() && r.should_close() {
            out = Some(false);
        }
        out
    }

    fn histogram(&mut self, ui: &egui::Ui, rect: Rect, resp: &egui::Response) {
        let (lo, hi) = self.range;
        let (top, bottom) = (rect.top() + 26.0, rect.bottom() - 24.0);
        let plot_w = rect.width() - 2.0 * PAD_X;
        let to_x = |v: f64| rect.left() + PAD_X + ((v - lo) / (hi - lo)) as f32 * plot_w;
        let to_v = |x: f32| round1(lo + ((x - rect.left() - PAD_X) / plot_w) as f64 * (hi - lo));
        let near = |lines: &[f64], x: f32| lines.iter().enumerate().map(|(i, v)| (i, (to_x(*v) - x).abs())).filter(|d| d.1 < 10.0).min_by(|a, b| a.1.total_cmp(&b.1)).map(|d| d.0);

        if let Some(pos) = resp.interact_pointer_pos() {
            if resp.drag_started() || resp.clicked() {
                self.drag = near(&self.lines, pos.x);
                if self.drag.is_none() && !resp.double_clicked() {
                    self.lines.push(to_v(pos.x));
                    self.drag = Some(self.lines.len() - 1);
                }
            }
            if resp.dragged() && let Some(i) = self.drag {
                self.lines[i] = to_v(pos.x);
            }
            if resp.double_clicked() && let Some(i) = near(&self.lines, pos.x) {
                self.lines.remove(i);
                self.drag = None;
            }
        }
        if resp.drag_stopped() || resp.clicked() {
            self.drag = None;
            self.lines.sort_by(f64::total_cmp);
        }

        let painter = ui.painter();
        let values: Vec<f64> = self.entries.iter().map(|e| e.2).collect();
        const BINS: usize = 24;
        let mut bins = [0usize; BINS];
        for v in &values {
            bins[(((v - lo) / (hi - lo) * BINS as f64) as usize).min(BINS - 1)] += 1;
        }
        let max_bin = *bins.iter().max().unwrap_or(&1) as f32;
        let bw = plot_w / BINS as f32;
        for (i, n) in bins.iter().enumerate() {
            let h = *n as f32 / max_bin * (bottom - top);
            let r = Rect::from_min_max(pos2(rect.left() + PAD_X + i as f32 * bw, bottom - h), pos2(rect.left() + PAD_X + (i + 1) as f32 * bw - 1.0, bottom));
            painter.rect(r, 0.0, ui::PRIMARY.gamma_multiply(0.18), Stroke::new(1.0, ui::PRIMARY.gamma_multiply(0.4)), egui::StrokeKind::Inside);
        }
        let g = grid(lo, hi, 150);
        let d = kde(&values, &g);
        let dmax = d.iter().cloned().fold(1e-9, f64::max);
        let pts: Vec<Pos2> = g.iter().zip(&d).map(|(x, y)| pos2(to_x(*x), bottom - (*y / dmax) as f32 * (bottom - top))).collect();
        painter.add(Shape::line(pts, Stroke::new(1.5, Color32::from_rgb(0x0f, 0x17, 0x2a))));
        for (i, v) in self.lines.iter().enumerate() {
            let x = to_x(*v);
            painter.vline(x, top..=bottom, Stroke::new(2.0, color(i, 0.42)));
            painter.text(pos2(x, top - 8.0), egui::Align2::CENTER_CENTER, format!("{v:.1}"), FontId::new(11.0, ui::semibold()), color(i, 0.42));
        }
        let axis = FontId::proportional(11.0);
        painter.text(pos2(rect.left() + PAD_X, rect.bottom() - 6.0), egui::Align2::LEFT_BOTTOM, format!("{lo:.1} px/cm"), axis.clone(), ui::MUTED);
        painter.text(pos2(rect.right() - PAD_X, rect.bottom() - 6.0), egui::Align2::RIGHT_BOTTOM, format!("{hi:.1} px/cm"), axis, ui::MUTED);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kde_peaks_at_clusters() {
        let v = [10.0, 10.1, 9.9, 10.05, 20.0, 20.2, 19.9];
        let g = grid(5.0, 25.0, 201);
        let d = kde(&v, &g);
        assert!(d[50] > d[100] && d[150] > d[100]); // 10 and 20 beat 15
    }
}
