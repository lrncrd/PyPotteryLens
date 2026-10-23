//! Polygon review editor: port of static/js/annotation-tab.js (select / brush / polygon /
//! eraser / scale tools, vessels and scales panels, colorize, calibration).

use crate::geom;
use crate::project::{self, Polygon, Project};
use crate::scales;
use crate::ui::{self, Kind};
use eframe::egui::{self, Color32, FontId, Frame, Key, Margin, PointerButton, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind, TextureHandle, Vec2, pos2, vec2};
use image::{GrayImage, Luma};
use imageproc::contours::{BorderType, find_contours};
use serde_json::{Value, json};

const HIT: f32 = 8.0; // px on screen
const EDGE_HIT: f32 = 5.0;
// VESSEL_PALETTE of the web editor
const PALETTE: [Color32; 9] = [
    Color32::from_rgb(0x0d, 0x94, 0x88),
    Color32::from_rgb(0x7c, 0x3a, 0xed),
    Color32::from_rgb(0xec, 0x48, 0x99),
    Color32::from_rgb(0x10, 0xb9, 0x81),
    Color32::from_rgb(0x02, 0x84, 0xc7),
    Color32::from_rgb(0xd9, 0x77, 0x06),
    Color32::from_rgb(0xe1, 0x1d, 0x48),
    Color32::from_rgb(0x65, 0xa3, 0x0d),
    Color32::from_rgb(0x4f, 0x46, 0xe5),
];
const SELECTED: Color32 = Color32::from_rgb(0xea, 0x58, 0x0c);
const AMBER: Color32 = Color32::from_rgb(0xf5, 0x9e, 0x0b);
const RED: Color32 = Color32::from_rgb(0xdc, 0x26, 0x26);

#[derive(Clone, Copy, PartialEq, Default)]
enum Tool {
    #[default]
    Select,
    Brush,
    Polygon,
    Eraser,
    Scale,
}

/// Tool settings survive switching image or project.
struct Prefs {
    tool: Tool,
    brush: f32, // radius in screen px
    colorize: bool,
    hide_list: bool,
    vessels_open: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Self { tool: Tool::Select, brush: 20.0, colorize: true, hide_list: false, vessels_open: false }
    }
}

enum Drag {
    Vertex(usize, usize),
    Move(usize, Polygon, [f64; 2]),
    Pan,
    Stroke(Vec<[f64; 2]>),
    Erase(Pos2, bool),
    Zone([f64; 2]),
}

enum Confirm {
    Clear,
    Extract,
}

struct Loaded {
    name: String,
    tex: TextureHandle,
    size: Vec2,
}

#[derive(Default)]
pub struct Editor {
    images: Vec<String>,
    counts: Vec<usize>,
    index: usize,
    loaded: Option<Loaded>,
    polys: Vec<Polygon>,
    expansion: i64,
    undo: Vec<Vec<Polygon>>,
    selected: Option<usize>,
    drag: Option<Drag>,
    draft: Polygon,
    scales: Vec<Value>,
    ruler: Option<([f64; 2], Option<[f64; 2]>)>, // p1, p2 once placed (then the cm popup is open)
    ruler_cm: String,
    zone_for: Option<usize>,
    zoom: f32,
    pan: Vec2,
    fit: bool,
    fitted: Option<(f32, Vec2, Vec2)>, // zoom, pan, canvas size of the last fit
    canvas_rect: Option<Rect>,
    tools_h: f32,
    side_h: f32,
    msg: String,
    scroll_list: bool,
    prefs: Prefs,
    confirm: Option<Confirm>,
    calib: Option<scales::Calib>,
}

#[derive(Clone, Copy)]
struct View {
    rect: Rect,
    zoom: f32,
    pan: Vec2,
}

impl View {
    fn to_screen(&self, p: [f64; 2]) -> Pos2 {
        self.rect.min + self.pan + vec2(p[0] as f32, p[1] as f32) * self.zoom
    }
    fn to_img(&self, s: Pos2) -> [f64; 2] {
        let v = (s - self.rect.min - self.pan) / self.zoom;
        [(v.x as f64 * 10.0).round() / 10.0, (v.y as f64 * 10.0).round() / 10.0]
    }
}

impl Editor {
    /// Reload the image list (after opening a project or converting a PDF).
    pub fn reset(&mut self, p: &Project) {
        let index = self.index;
        let prefs = std::mem::take(&mut self.prefs);
        let images = p.images();
        let counts = images.iter().map(|n| p.read_polygons(project::stem(n)).0.len()).collect();
        *self = Self { images, counts, prefs, ..Default::default() };
        self.index = index.min(self.images.len().saturating_sub(1));
    }

    /// Expansion applied to every card at extraction (the slider value, like the web app).
    pub fn expansion(&self) -> i64 {
        self.expansion
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn images(&self) -> &[String] {
        &self.images
    }

    fn current(&self) -> &str {
        &self.images[self.index]
    }

    fn base(&self) -> String {
        project::stem(self.current()).to_string()
    }

    pub fn go(&mut self, i: usize) {
        self.index = i.min(self.images.len() - 1);
        self.loaded = None;
        self.scroll_list = true;
    }

    fn set_tool(&mut self, t: Tool) {
        if t != Tool::Polygon {
            self.draft.clear();
        }
        if t != Tool::Scale {
            self.ruler = None;
            self.zone_for = None;
        }
        self.drag = None;
        self.prefs.tool = t;
    }

    fn load(&mut self, ctx: &egui::Context, p: &Project) {
        let name = self.current().to_string();
        if self.loaded.as_ref().is_some_and(|l| l.name == name) {
            return;
        }
        // ponytail: decoded on the UI thread (~100 ms per 300 DPI page); move to a thread if it feels laggy
        let img = match image::open(p.dir("images").join(&name)) {
            Ok(img) => img.to_rgba8(),
            Err(e) => {
                self.msg = format!("Cannot open {name}: {e}");
                return;
            }
        };
        let size = vec2(img.width() as f32, img.height() as f32);
        let max = ctx.input(|i| i.max_texture_side) as u32;
        let img = if img.width() > max || img.height() > max {
            let s = max as f32 / img.width().max(img.height()) as f32;
            image::imageops::resize(&img, (img.width() as f32 * s) as u32, (img.height() as f32 * s) as u32, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
        let tex = ctx.load_texture(&name, color, egui::TextureOptions::LINEAR);
        let base = self.base();
        (self.polys, self.expansion) = p.read_polygons(&base);
        self.scales = p.read_scales(&base);
        self.loaded = Some(Loaded { name, tex, size });
        self.undo.clear();
        self.selected = None;
        self.drag = None;
        self.draft.clear();
        self.ruler = None;
        self.zone_for = None;
        self.fit = true;
        self.msg.clear();
    }

    fn save(&mut self, p: &Project) {
        if let Err(e) = p.write_polygons(&self.base(), &self.polys, self.expansion) {
            self.msg = format!("Save failed: {e:#}");
        }
        self.counts[self.index] = self.polys.len();
    }

    fn save_scales(&mut self, p: &Project) {
        if let Err(e) = p.write_scales(&self.base(), &self.scales) {
            self.msg = format!("Save failed: {e:#}");
        }
    }

    fn checkpoint(&mut self) {
        self.undo.push(self.polys.clone());
        if self.undo.len() > 50 {
            self.undo.remove(0);
        }
    }

    fn vertex_at(&self, view: &View, pos: Pos2) -> Option<(usize, usize)> {
        // the selected polygon wins over overlapping neighbours
        let order = self.selected.into_iter().chain(0..self.polys.len());
        for pi in order {
            for (vi, v) in self.polys[pi].iter().enumerate() {
                if view.to_screen(*v).distance(pos) < HIT {
                    return Some((pi, vi));
                }
            }
        }
        None
    }

    /// Edge of the selected polygon under `pos`: index of its first vertex.
    fn edge_at(&self, view: &View, pos: Pos2) -> Option<usize> {
        let poly = &self.polys[self.selected?];
        (0..poly.len())
            .map(|i| (i, seg_dist(pos, view.to_screen(poly[i]), view.to_screen(poly[(i + 1) % poly.len()]))))
            .filter(|e| e.1 < EDGE_HIT)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|e| e.0)
    }

    /// Smallest polygon under the point, so nested vessels stay selectable.
    fn polygon_at(&self, pt: [f64; 2]) -> Option<usize> {
        (0..self.polys.len()).filter(|&i| geom::contains(&self.polys[i], pt[0], pt[1])).min_by(|&a, &b| geom::area(&self.polys[a]).total_cmp(&geom::area(&self.polys[b])))
    }

    fn clamp(&self, q: [f64; 2]) -> [f64; 2] {
        let s = self.loaded.as_ref().map_or(vec2(1e9, 1e9), |l| l.size);
        [q[0].clamp(0.0, s.x as f64 - 1.0), q[1].clamp(0.0, s.y as f64 - 1.0)]
    }

    /// Vertex or edge handle of the selected polygon (edge click inserts a vertex), as a drag.
    fn grab_handle(&mut self, view: &View, o: Pos2) -> Option<Drag> {
        let si = self.selected?;
        if let Some(vi) = self.polys[si].iter().position(|v| view.to_screen(*v).distance(o) < HIT) {
            self.checkpoint();
            return Some(Drag::Vertex(si, vi));
        }
        let e = self.edge_at(view, o)?;
        self.checkpoint();
        self.polys[si].insert(e + 1, view.to_img(o));
        Some(Drag::Vertex(si, e + 1))
    }

    /// Remove every vertex within the brush radius of the segment a–b (screen px); polygons left
    /// with fewer than 3 vertices disappear. Like `eraseVerticesSweep`.
    fn erase(&mut self, view: &View, a: Pos2, b: Pos2) -> bool {
        let r = self.prefs.brush;
        let mut any = false;
        for pi in (0..self.polys.len()).rev() {
            let before = self.polys[pi].len();
            self.polys[pi].retain(|v| seg_dist(view.to_screen(*v), a, b) > r);
            if self.polys[pi].len() == before {
                continue;
            }
            any = true;
            if self.polys[pi].len() < 3 {
                self.polys.remove(pi);
                self.selected = match self.selected {
                    Some(s) if s == pi => None,
                    Some(s) if s > pi => Some(s - 1),
                    s => s,
                };
            }
        }
        any
    }

    /// Turn a round-brush stroke (image coords) into a polygon: merged with every polygon it
    /// touches, or a new vessel if it touches none. Like `stopDrawing` for the brush.
    fn commit_stroke(&mut self, p: &Project, pts: &[[f64; 2]]) {
        let Some(size) = self.loaded.as_ref().map(|l| l.size) else { return };
        let r = (self.prefs.brush / self.zoom) as f64;
        let bbox = |ps: &[[f64; 2]], pad: f64| ps.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, q| [b[0].min(q[0] - pad), b[1].min(q[1] - pad), b[2].max(q[0] + pad), b[3].max(q[1] + pad)]);
        let overlap = |a: [f64; 4], b: [f64; 4]| a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
        let sb = bbox(pts, r);
        let cand: Vec<usize> = (0..self.polys.len()).filter(|&i| overlap(bbox(&self.polys[i], 0.0), sb)).collect();
        let reg = cand.iter().fold(sb, |b, &i| {
            let q = bbox(&self.polys[i], 0.0);
            [b[0].min(q[0]), b[1].min(q[1]), b[2].max(q[2]), b[3].max(q[3])]
        });
        let reg = [reg[0].max(0.0), reg[1].max(0.0), reg[2].min(size.x as f64), reg[3].min(size.y as f64)];
        if reg[2] - reg[0] < 1.0 || reg[3] - reg[1] < 1.0 {
            return;
        }
        // raster at most 1500 px wide; 1 px empty border so contours close
        let k = (1500.0 / (reg[2] - reg[0]).max(reg[3] - reg[1])).min(1.0);
        let (mw, mh) = (((reg[2] - reg[0]) * k).ceil() as u32 + 2, ((reg[3] - reg[1]) * k).ceil() as u32 + 2);
        let to_m = |q: &[f64; 2]| [(q[0] - reg[0]) * k + 1.0, (q[1] - reg[1]) * k + 1.0];
        let mut stroke = GrayImage::new(mw, mh);
        let rm = (r * k).max(1.0);
        let mut prev: Option<[f64; 2]> = None;
        for q in pts.iter().map(to_m) {
            let a = prev.unwrap_or(q);
            let steps = ((q[0] - a[0]).hypot(q[1] - a[1]) / (rm / 3.0)).ceil().max(1.0) as usize;
            for s in 0..=steps {
                let t = s as f64 / steps as f64;
                let c = ((a[0] + (q[0] - a[0]) * t).round() as i32, (a[1] + (q[1] - a[1]) * t).round() as i32);
                imageproc::drawing::draw_filled_circle_mut(&mut stroke, c, rm.round() as i32, Luma([255]));
            }
            prev = Some(q);
        }
        clear_border(&mut stroke);
        if stroke.pixels().filter(|v| v[0] > 0).count() < 20 {
            return;
        }
        let mapped = |poly: &Polygon| poly.iter().map(to_m).collect::<Polygon>();
        let hits: Vec<usize> = cand
            .into_iter()
            .filter(|&i| {
                let mut m = GrayImage::new(mw, mh);
                geom::fill(&mut m, &mapped(&self.polys[i]), (0.0, 0.0), true);
                m.pixels().zip(stroke.pixels()).any(|(a, b)| a[0] > 0 && b[0] > 0)
            })
            .collect();
        let mut union = stroke;
        for &i in &hits {
            geom::fill(&mut union, &mapped(&self.polys[i]), (0.0, 0.0), true);
        }
        clear_border(&mut union);
        let contour = find_contours::<i32>(&union)
            .into_iter()
            .filter(|c| c.border_type == BorderType::Outer && c.parent.is_none())
            .map(|c| c.points.iter().map(|q| self.clamp([(q.x as f64 - 1.0) / k + reg[0], (q.y as f64 - 1.0) / k + reg[1]])).collect::<Polygon>())
            .max_by(|a, b| geom::area(a).total_cmp(&geom::area(b)));
        let Some(contour) = contour.filter(|c| c.len() >= 6) else { return };
        let poly = geom::simplify_eps(&contour, (2.0 / self.zoom as f64).clamp(1.0, 8.0));
        if poly.len() < 3 {
            return;
        }
        self.checkpoint();
        match hits.first() {
            None => {
                self.polys.push(poly);
                self.selected = Some(self.polys.len() - 1);
            }
            Some(&t) => {
                self.polys[t] = poly;
                for &i in hits[1..].iter().rev() {
                    self.polys.remove(i);
                }
                self.selected = Some(t);
                if hits.len() > 1 {
                    self.msg = format!("Merged with vessel #{}", t + 1);
                }
            }
        }
        self.save(p);
    }

    /// Which scale each polygon gets at extraction: centre of the expanded bbox, smallest zone
    /// containing it, else the global scale (`computeScaleAssignment`). (centre, scale, conflict)
    fn assignment(&self) -> Vec<([f64; 2], Option<usize>, bool)> {
        let s = self.loaded.as_ref().map_or(vec2(1e9, 1e9), |l| l.size);
        let pad = self.expansion.max(0) as f64;
        let global = self.scales.iter().position(|s| project::scale_zone(s).is_none());
        self.polys
            .iter()
            .map(|poly| {
                let b = poly.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, q| [b[0].min(q[0]), b[1].min(q[1]), b[2].max(q[0]), b[3].max(q[1])]);
                let (x1, y1, x2, y2) = ((b[0] - pad).max(0.0), (b[1] - pad).max(0.0), (b[2] + 1.0 + pad).min(s.x as f64), (b[3] + 1.0 + pad).min(s.y as f64));
                let c = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
                let mut m: Vec<(f64, usize)> = self
                    .scales
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| {
                        let z = project::scale_zone(s)?;
                        let inside = z[0].min(z[2]) <= c[0] && c[0] <= z[0].max(z[2]) && z[1].min(z[3]) <= c[1] && c[1] <= z[1].max(z[3]);
                        inside.then(|| ((z[2] - z[0]).abs() * (z[3] - z[1]).abs(), i))
                    })
                    .collect();
                m.sort_by(|a, b| a.0.total_cmp(&b.0));
                (c, m.first().map(|x| x.1).or(global), m.len() > 1)
            })
            .collect()
    }

    fn color(&self, i: usize) -> Color32 {
        if self.prefs.colorize { PALETTE[i % PALETTE.len()] } else { ui::TEAL }
    }

    /// Returns true when the extraction was confirmed.
    pub fn ui(&mut self, ui: &mut egui::Ui, p: &mut Project, clean: &mut bool) -> bool {
        if self.images.is_empty() {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new(ui::I_IMAGES).size(40.0).color(ui::MUTED));
                ui.label(RichText::new("No images yet").font(FontId::new(18.0, ui::semibold())));
                ui.label(ui::muted("Convert a PDF in step 1 first."));
            });
            return false;
        }
        let mut goto = None;
        // explicit sizes: nested panels inside the content frame overflow its margins
        let size = ui.available_size();
        let gap = 20.0;
        let left_w = if self.prefs.hide_list { 0.0 } else { 330.0 };
        let right_w = size.x - left_w - if left_w > 0.0 { gap } else { 0.0 };
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            if left_w > 0.0 {
                ui.allocate_ui_with_layout(vec2(left_w, size.y), egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.spacing_mut().item_spacing.y = 10.0;
                    let list_h = (size.y - self.side_h).max(160.0);
                    ui.allocate_ui(vec2(left_w, list_h), |ui| self.image_list(ui, p, &mut goto, left_w));
                    let side = ui.vertical(|ui| self.side_panels(ui, p, left_w));
                    let h = side.response.rect.height() + 10.0;
                    if (h - self.side_h).abs() > 0.5 {
                        self.side_h = h;
                        ui.ctx().request_repaint();
                    }
                });
            }
            ui.allocate_ui_with_layout(vec2(right_w, size.y), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.spacing_mut().item_spacing.y = 12.0;
                self.title_bar(ui, p, right_w);
                // the toolbar's height is measured each frame; the canvas takes what is left
                let canvas_h = (ui.available_height() - 2.0 * 12.0 - self.tools_h).max(200.0);
                ui.allocate_ui(vec2(right_w, canvas_h), |ui| {
                    Frame::new().fill(ui::CANVAS_BG).stroke(Stroke::new(1.0, ui::BORDER)).corner_radius(14).show(ui, |ui| {
                        ui.set_width(right_w - 2.0);
                        ui.set_height(canvas_h - 2.0);
                        self.canvas(ui, p, &mut goto);
                    });
                });
                let tools = ui.vertical(|ui| {
                    self.toolbar(ui, p, &mut goto, right_w);
                    ui::card().inner_margin(Margin::same(14)).show(ui, |ui| {
                        ui.set_width(right_w - 30.0);
                        ui.horizontal(|ui| {
                            if ui::button(ui, Kind::Secondary, ui::I_CHART, "Calibrate Scales", vec2(0.0, 46.0), true)
                                .on_hover_text("Review every scale measured across the whole project on one histogram, and snap them to consistent values")
                                .clicked()
                            {
                                self.calib = Some(scales::Calib::open(p, &self.images));
                            }
                            let w = ui.available_width();
                            if ui::button(ui, Kind::Primary, ui::I_CROP, "Extract Cards", vec2(w, 46.0), true).clicked() {
                                self.confirm = Some(Confirm::Extract);
                            }
                        });
                    });
                });
                let h = tools.response.rect.height();
                if (h - self.tools_h).abs() > 0.5 {
                    self.tools_h = h;
                    ui.ctx().request_repaint();
                }
            });
        });

        if let Some(i) = goto {
            self.go(i);
        }
        self.modals(ui.ctx(), p, clean)
    }

    fn modals(&mut self, ctx: &egui::Context, p: &mut Project, clean: &mut bool) -> bool {
        let mut extract = false;
        if let Some(c) = self.calib.as_mut()
            && let Some(apply) = c.ui(ctx)
        {
            if apply {
                match c.apply(p) {
                    Ok(n) => self.msg = format!("Calibrated {n} page(s)."),
                    Err(e) => self.msg = format!("Calibration failed: {e:#}"),
                }
                self.scales = p.read_scales(&self.base());
            }
            self.calib = None;
        }
        let Some(c) = &self.confirm else { return false };
        let (mut ok, mut cancel) = (false, false);
        let is_extract = matches!(c, Confirm::Extract);
        let r = egui::Modal::new(egui::Id::new("editor_confirm")).frame(ui::card().inner_margin(Margin::same(26))).show(ctx, |ui| {
            ui.set_width(460.0);
            if is_extract {
                ui.label(RichText::new(format!("{}  Extract Cards from Masks", ui::I_CROP)).font(FontId::new(19.0, ui::semibold())).color(ui::TEXT));
                ui.label(RichText::new("Crop drawing cards from all annotated masks in the active project").color(ui::DIM));
                Frame::new().fill(ui::PEACH).stroke(Stroke::new(1.0, ui::PRIMARY_BORDER)).corner_radius(10).inner_margin(12).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new(format!("{}  Overwrite Warning", ui::I_WARN)).font(FontId::new(14.0, ui::semibold())).color(ui::PRIMARY));
                    ui.label(RichText::new("Any previously extracted cards in this project will be re-generated and overwritten.").size(13.5).color(ui::DIM));
                });
                ui.horizontal(|ui| {
                    ui::toggle(ui, clean, "");
                    ui.vertical(|ui| {
                        ui.label(RichText::new(format!("{}  Clean Artifacts & Numbers", ui::I_STARS)).font(FontId::new(14.0, ui::semibold())).color(ui::TEXT));
                        ui.label(RichText::new("Deterministically removes isolated inventory numbers, section cuts, and stray fragments outside the vessel.").size(13.0).color(ui::MUTED));
                    });
                });
                ui.label(ui::muted(format!("Expansion: {} px on every card", self.expansion)));
            } else {
                ui.label(RichText::new(format!("{}  Clear Annotations", ui::I_ERASER)).font(FontId::new(19.0, ui::semibold())).color(ui::TEXT));
                ui.label(RichText::new("Remove all drawn pottery contours on this image? You can redraw them anytime before extracting cards (Undo also works).").color(ui::DIM));
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                cancel = ui::button(ui, Kind::Secondary, ui::I_X, "Cancel", vec2(120.0, 40.0), true).clicked();
                ok = if is_extract {
                    ui::button(ui, Kind::Primary, ui::I_CROP, "Extract Cards", vec2(180.0, 40.0), true).clicked()
                } else {
                    ui::button(ui, Kind::Danger, ui::I_TRASH, "Clear Canvas", vec2(160.0, 40.0), true).clicked()
                };
            });
        });
        if ok {
            if is_extract {
                extract = true;
            } else {
                self.checkpoint();
                self.polys.clear();
                self.selected = None;
                self.save(p);
            }
            self.confirm = None;
        } else if cancel || r.should_close() {
            self.confirm = None;
        }
        extract
    }

    fn title_bar(&mut self, ui: &mut egui::Ui, p: &mut Project, width: f32) {
        ui::card().inner_margin(Margin::symmetric(16, 8)).show(ui, |ui| {
            ui.set_width(width - 34.0);
            ui.horizontal(|ui| {
                ui.set_height(28.0);
                let (icon, text) = if self.prefs.hide_list { (ui::I_SIDEBAR_INSET, "Show Images") } else { (ui::I_SIDEBAR, "Images List") };
                if ui::button(ui, Kind::Secondary, icon, text, vec2(0.0, 28.0), true).clicked() {
                    self.prefs.hide_list = !self.prefs.hide_list;
                }
                ui.add_space(6.0);
                let name = self.current().to_string();
                ui.label(RichText::new(format!("{}/{}  –  {name}", self.index + 1, self.images.len())).font(FontId::monospace(14.0)).color(ui::TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut excluded = p.is_excluded(&name);
                    let mut reviewed = p.is_reviewed(&name);
                    let x = ui::toggle(ui, &mut excluded, "");
                    ui.label(RichText::new("Excluded").size(13.5).color(ui::DIM));
                    ui.add_space(12.0);
                    let r = ui::toggle(ui, &mut reviewed, "");
                    ui.label(RichText::new("Reviewed").size(13.5).color(ui::DIM));
                    if r.changed() || x.changed() {
                        p.set_reviewed(&name, reviewed);
                        p.set_excluded(&name, excluded);
                        if let Err(e) = p.save() {
                            self.msg = format!("Save failed: {e:#}");
                        }
                    }
                });
            });
        });
    }

    fn image_list(&mut self, ui: &mut egui::Ui, p: &Project, goto: &mut Option<usize>, width: f32) {
        ui.spacing_mut().item_spacing.y = 2.0;
        ui::card().inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
            ui.set_width(width - 30.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(ui::I_IMAGES).color(ui::TEXT));
                ui.label(RichText::new(format!("Images ({})", self.images.len())).font(FontId::new(15.0, ui::semibold())).color(ui::TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let reviewed = p.meta.workflow_status.reviewed_images.len();
                    ui.label(ui::muted(format!("{reviewed} reviewed")));
                });
            });
        });
        Frame::new().fill(egui::Color32::WHITE).stroke(Stroke::new(1.0, ui::BORDER)).corner_radius(14).show(ui, |ui| {
            ui.set_width(width - 2.0);
            ui.set_height(ui.available_height() - 2.0);
            const ROW: f32 = 46.0;
            ui.spacing_mut().item_spacing.y = 0.0;
            let mut area = egui::ScrollArea::vertical().auto_shrink(false);
            if std::mem::take(&mut self.scroll_list) {
                area = area.vertical_scroll_offset((self.index as f32 * ROW - ui.available_height() / 2.0).max(0.0));
            }
            area.show_rows(ui, ROW, self.images.len(), |ui, range| {
                for i in range {
                    let name = &self.images[i];
                    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), ROW), Sense::click());
                    let active = i == self.index;
                    let painter = ui.painter();
                    if active {
                        painter.rect_filled(rect, 0.0, Color32::from_rgb(0xfb, 0xf3, 0xee));
                        painter.rect_filled(Rect::from_min_size(rect.min, vec2(3.0, ROW)), 0.0, ui::PRIMARY);
                    } else if resp.hovered() {
                        painter.rect_filled(rect, 0.0, ui::BG_PAGE);
                    }
                    painter.hline(rect.x_range(), rect.bottom() - 0.5, Stroke::new(1.0, ui::BORDER));
                    let excluded = p.is_excluded(name);
                    let (icon, color) = if excluded {
                        (ui::I_EYE_SLASH, ui::MUTED)
                    } else if p.is_reviewed(name) {
                        (ui::I_CHECK_FILL, ui::TEAL)
                    } else {
                        (ui::I_CIRCLE, Color32::from_rgb(0xc8, 0xc2, 0xba))
                    };
                    let y = rect.center().y;
                    painter.text(pos2(rect.left() + 22.0, y), egui::Align2::CENTER_CENTER, icon, FontId::proportional(15.0), color);
                    painter.text(pos2(rect.left() + 42.0, y), egui::Align2::LEFT_CENTER, name, FontId::proportional(14.5), if excluded { ui::MUTED } else { ui::TEXT });
                    if self.counts[i] > 0 {
                        painter.text(pos2(rect.right() - 14.0, y), egui::Align2::RIGHT_CENTER, format!("{} {}", ui::I_PEN, self.counts[i]), FontId::proportional(13.0), ui::DIM);
                    }
                    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        *goto = Some(i);
                    }
                }
            });
        });
    }

    /// "Vessels & Polygons" and "Scale calibrations" panels.
    fn side_panels(&mut self, ui: &mut egui::Ui, p: &Project, width: f32) {
        let assignment = if self.scales.is_empty() { vec![] } else { self.assignment() };
        let small = vec2(0.0, 26.0);
        if !self.polys.is_empty() {
            ui::card().inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
                ui.set_width(width - 30.0);
                let head = ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}  Vessels & Polygons ({})", ui::I_VESSELS, self.polys.len())).font(FontId::new(14.5, ui::semibold())).color(ui::TEXT));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(if self.prefs.vessels_open { ui::I_CHEVRON_DOWN } else { ui::I_RIGHT }).color(ui::MUTED));
                    });
                });
                if head.response.interact(Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    self.prefs.vessels_open = !self.prefs.vessels_open;
                }
                ui.label(ui::muted("Each polygon becomes a separate card on extraction."));
                if !self.prefs.vessels_open {
                    return;
                }
                let (mut simplify, mut delete) = (None, None);
                egui::ScrollArea::vertical().max_height(200.0).id_salt("vessels").show(ui, |ui| {
                    for i in 0..self.polys.len() {
                        let sel = self.selected == Some(i);
                        ui.horizontal(|ui| {
                            let (bar, _) = ui.allocate_exact_size(vec2(4.0, 24.0), Sense::hover());
                            ui.painter().rect_filled(bar, 2.0, if sel { SELECTED } else { self.color(i) });
                            let label = RichText::new(format!("#{} — {} pts", i + 1, self.polys[i].len())).size(13.0).color(if sel { SELECTED } else { ui::TEXT });
                            if ui.add(egui::Label::new(label).sense(Sense::click())).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                                self.selected = if sel { None } else { Some(i) };
                            }
                            if let Some((_, z, conflict)) = assignment.get(i) {
                                zone_chip(ui, *z, *conflict);
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui::button(ui, Kind::Danger, ui::I_TRASH, "", vec2(28.0, 26.0), true).on_hover_text("Delete").clicked() {
                                    delete = Some(i);
                                }
                                if ui::button(ui, Kind::Secondary, ui::I_BEZIER, "Simplify", small, self.polys[i].len() > 4).clicked() {
                                    simplify = Some(i);
                                }
                            });
                        });
                    }
                });
                if let Some(i) = simplify {
                    let s = geom::simplify_eps(&self.polys[i], 8.0);
                    if s.len() >= 3 && s.len() < self.polys[i].len() {
                        self.checkpoint();
                        self.msg = format!("Simplified vessel #{}: {} to {} points", i + 1, self.polys[i].len(), s.len());
                        self.polys[i] = s;
                        self.save(p);
                    }
                }
                if let Some(i) = delete {
                    self.selected = Some(i);
                    self.delete_selected(p);
                }
            });
        }
        if !self.scales.is_empty() {
            ui::card().inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
                ui.set_width(width - 30.0);
                ui.label(RichText::new(format!("{}  Scale calibrations ({})", ui::I_RULERS, self.scales.len())).font(FontId::new(14.5, ui::semibold())).color(ui::TEXT));
                ui.label(ui::muted("The ratio fills px_per_cm when you Extract Cards. Use Zone for multi-scale pages."));
                let (mut zone, mut delete) = (None, None);
                for (i, s) in self.scales.iter().enumerate() {
                    let n = assignment.iter().filter(|a| !a.2 && a.1 == Some(i)).count();
                    ui.horizontal(|ui| {
                        zone_chip(ui, Some(i), false);
                        let ratio = project::scale_ratio(s).map_or("—".into(), |r| format!("{r:.1} px/cm"));
                        let kind = if project::scale_zone(s).is_some() { "zone" } else { "global" };
                        let cal = if s["calibrated_px_per_cm"].as_f64().is_some_and(|v| v > 0.0) { format!(" {}", ui::I_CHECK_FILL) } else { String::new() };
                        ui.label(RichText::new(format!("{}cm · {ratio}{cal}", s["real_cm"])).size(12.5).color(ui::DIM))
                            .on_hover_text(format!("{kind} scale · {n} vessel(s) use it{}", if cal.is_empty() { "" } else { " · calibrated" }));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui::button(ui, Kind::Danger, ui::I_TRASH, "", vec2(28.0, 26.0), true).clicked() {
                                delete = Some(i);
                            }
                            if ui::button(ui, if self.zone_for == Some(i) { Kind::Primary } else { Kind::Secondary }, ui::I_GEO, "Zone", small, true)
                                .on_hover_text("Drag a rectangle on the canvas to limit this scale to a page region")
                                .clicked()
                            {
                                zone = Some(i);
                            }
                        });
                    });
                }
                if let Some(i) = zone {
                    self.set_tool(Tool::Scale);
                    self.zone_for = Some(i);
                    self.msg = "Drag a rectangle on the canvas to define the zone for this scale".into();
                }
                if let Some(i) = delete {
                    self.scales.remove(i);
                    self.save_scales(p);
                }
            });
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, p: &Project, goto: &mut Option<usize>, width: f32) {
        let small = vec2(36.0, 32.0);
        let h = vec2(0.0, 32.0);
        ui::card().inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
            ui.set_width(width - 34.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if ui::button(ui, Kind::Secondary, ui::I_ZOOM_OUT, "", small, true).clicked() {
                    self.zoom_by(1.0 / 1.25);
                }
                ui.label(RichText::new(format!("{:>3.0}%", self.zoom * 100.0)).font(FontId::monospace(13.0)).color(ui::DIM));
                if ui::button(ui, Kind::Secondary, ui::I_ZOOM_IN, "", small, true).clicked() {
                    self.zoom_by(1.25);
                }
                if ui::button(ui, Kind::Secondary, ui::I_EXPAND, "Fit", h, true).clicked() {
                    self.fit = true;
                }
                ui.add_space(16.0);
                ui.label(RichText::new(format!("{}  Size", ui::I_CIRCLE)).size(13.5).color(ui::DIM));
                ui::slider(ui, &mut self.prefs.brush, 5.0..=100.0, 100.0).on_hover_text("Brush / eraser radius");
                self.prefs.brush = self.prefs.brush.round();
                ui::chip(ui, &format!("{}px", self.prefs.brush), ui::DIM, ui::BG_PAGE, true);
                ui.add_space(12.0);
                ui.label(RichText::new(format!("{}  Expansion", ui::I_EXPAND)).size(13.5).color(ui::DIM));
                let mut exp = self.expansion as f32;
                let r = ui::slider(ui, &mut exp, 0.0..=100.0, 110.0).on_hover_text("Padding around every vessel at extraction (keeps rims and handles)");
                if r.changed() {
                    self.expansion = exp.round() as i64;
                }
                if r.drag_stopped() || r.clicked() {
                    self.save(p);
                }
                ui::chip(ui, &format!("{}px", self.expansion), ui::PRIMARY, ui::PEACH, true);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Secondary, "", &format!("Next  {}", ui::I_RIGHT), h, self.index + 1 < self.images.len()).clicked() {
                        *goto = Some(self.index + 1);
                    }
                    if ui::button(ui, Kind::Secondary, ui::I_LEFT, "Prev", h, self.index > 0).clicked() {
                        *goto = Some(self.index.saturating_sub(1));
                    }
                });
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let tools = [
                    (Tool::Select, ui::I_CURSOR, "Select", "Select (V): click a vessel to select it, drag it to move it. Drag a vertex to move it, click an edge to add a vertex, Alt/Shift+click a vertex to delete it."),
                    (Tool::Brush, ui::I_BRUSH, "Brush", "Round Brush (B): paint with a circular cursor; becomes a clean polygon on release, merged with the vessels it touches."),
                    (Tool::Polygon, ui::I_PEN, "Polygon", "Polygon (P): click to place vertices; click the first point, double-click or Enter to close."),
                    (Tool::Eraser, ui::I_ERASER_LINE, "Eraser", "Eraser (E): sweep over vertices to erase them."),
                    (Tool::Scale, ui::I_RULERS, "Scale", "Scale: click two points along the graphical bar, then enter its length in cm."),
                ];
                for (t, icon, label, tip) in tools {
                    let kind = if self.prefs.tool == t { Kind::Primary } else { Kind::Secondary };
                    if ui::button(ui, kind, icon, label, h, true).on_hover_text(tip).clicked() {
                        self.set_tool(t);
                    }
                }
                let kind = if self.prefs.colorize { Kind::TealOutline } else { Kind::Secondary };
                if ui::button(ui, kind, ui::I_PALETTE, "Colorize", h, true).on_hover_text("View each vessel with a distinct colour").clicked() {
                    self.prefs.colorize = !self.prefs.colorize;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Danger, ui::I_TRASH, "Clear", h, !self.polys.is_empty()).on_hover_text("Remove all polygons on this image").clicked() {
                        self.confirm = Some(Confirm::Clear);
                    }
                    if ui::button(ui, Kind::Danger, ui::I_X, "Delete", h, self.selected.is_some()).clicked() {
                        self.delete_selected(p);
                    }
                    if ui::button(ui, Kind::Secondary, ui::I_UNDO, "Undo", h, !self.undo.is_empty()).clicked() {
                        self.undo_last(p);
                    }
                });
            });
            let hint = match self.prefs.tool {
                Tool::Select => "Click: select · drag a vessel: move · drag a vertex: move · click an edge: add vertex · Alt/Shift+click or right-click a vertex: remove",
                Tool::Brush => "Paint over a drawing: it becomes a polygon, merged with the vessels it touches · click a vessel: select · drag its vertices to edit",
                Tool::Polygon => "Click to add points · click the first point, double-click or Enter to close · Backspace: remove last point · Esc: cancel",
                Tool::Eraser => "Sweep over vertices to remove them · a polygon left with less than 3 points is deleted",
                Tool::Scale => "Click both ends of the graphical scale bar, then type its length in cm · Zone (in the scales panel) limits a scale to part of the page",
            };
            ui.label(ui::muted(format!("{hint} · Scroll: pan · Pinch / ⌘+scroll: zoom · Right-drag: pan · ⌘Z: undo · arrow keys: image")));
            if !self.msg.is_empty() {
                ui.label(RichText::new(&self.msg).size(13.0).color(if self.msg.contains("failed") || self.msg.starts_with("Cannot") { ui::DANGER } else { ui::TEAL }));
            }
        });
    }

    /// Zoom about the canvas centre (toolbar buttons).
    fn zoom_by(&mut self, f: f32) {
        let Some(rect) = self.canvas_rect else { return };
        let c = rect.center();
        let anchor = (c - rect.min - self.pan) / self.zoom;
        self.zoom = (self.zoom * f).clamp(0.02, 20.0);
        self.pan = c - rect.min - anchor * self.zoom;
    }

    fn delete_selected(&mut self, p: &Project) {
        if let Some(i) = self.selected.take() {
            self.checkpoint();
            self.polys.remove(i);
            self.save(p);
        }
    }

    fn undo_last(&mut self, p: &Project) {
        if let Some(prev) = self.undo.pop() {
            self.polys = prev;
            self.selected = None;
            self.drag = None;
            self.save(p);
        }
    }

    fn finish_draft(&mut self, p: &Project) {
        let d = std::mem::take(&mut self.draft);
        if d.len() >= 3 {
            self.checkpoint();
            self.polys.push(d);
            self.selected = Some(self.polys.len() - 1);
            self.save(p);
        }
    }

    fn keyboard(&mut self, ui: &egui::Ui, p: &Project, goto: &mut Option<usize>) {
        if ui.memory(|m| m.focused().is_some()) {
            return;
        }
        let k = |key: Key| ui.input(|i| i.key_pressed(key) && !i.modifiers.command);
        if ui.input(|i| i.modifiers.command && i.key_pressed(Key::Z)) {
            self.undo_last(p);
            return;
        }
        if k(Key::ArrowLeft) {
            *goto = Some(self.index.saturating_sub(1));
        }
        if k(Key::ArrowRight) {
            *goto = Some(self.index + 1);
        }
        for (key, t) in [(Key::V, Tool::Select), (Key::B, Tool::Brush), (Key::P, Tool::Polygon), (Key::E, Tool::Eraser)] {
            if k(key) {
                self.set_tool(t);
            }
        }
        let (esc, enter, del, bksp) = (k(Key::Escape), k(Key::Enter), k(Key::Delete), k(Key::Backspace));
        match self.prefs.tool {
            Tool::Polygon if !self.draft.is_empty() => {
                if bksp {
                    self.draft.pop();
                }
                if esc {
                    self.draft.clear();
                } else if enter {
                    self.finish_draft(p);
                }
            }
            Tool::Scale if esc => {
                self.ruler = None;
                self.zone_for = None;
            }
            _ => {
                if esc {
                    self.selected = None;
                }
                if del || bksp {
                    self.delete_selected(p);
                }
            }
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui, p: &mut Project, goto: &mut Option<usize>) {
        self.load(ui.ctx(), p);
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let Some(size) = self.loaded.as_ref().map(|l| l.size) else { return };
        let rect = resp.rect;
        self.canvas_rect = Some(rect);
        // refit when the canvas resizes, unless the user has zoomed or panned since
        let untouched = self.fitted.is_some_and(|(z, pan, s)| z == self.zoom && pan == self.pan && s != rect.size());
        if self.fit || untouched {
            self.zoom = ((rect.width() - 24.0) / size.x).min((rect.height() - 24.0) / size.y);
            self.pan = (rect.size() - size * self.zoom) / 2.0;
            self.fit = false;
            self.fitted = Some((self.zoom, self.pan, rect.size()));
        }
        // scroll pans, pinch / ⌘+scroll zooms about the cursor (like the web canvas)
        if let Some(hover) = resp.hover_pos() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            if zoom != 1.0 {
                let anchor = (hover - rect.min - self.pan) / self.zoom;
                self.zoom = (self.zoom * zoom).clamp(0.02, 20.0);
                self.pan = hover - rect.min - anchor * self.zoom;
            } else {
                self.pan += scroll;
            }
        }
        if resp.dragged_by(PointerButton::Secondary) || resp.dragged_by(PointerButton::Middle) {
            self.pan += resp.drag_delta();
        }
        let view = View { rect, zoom: self.zoom, pan: self.pan };
        let pointer = resp.interact_pointer_pos().or(resp.hover_pos());
        self.keyboard(ui, p, goto);
        self.pointer(ui, &resp, &view, pointer, p);
        self.paint(ui, &painter, &view, size, resp.hover_pos());
        self.ruler_popup(ui.ctx(), &view, p);
    }

    fn pointer(&mut self, ui: &egui::Ui, resp: &egui::Response, view: &View, pointer: Option<Pos2>, p: &Project) {
        let mods = ui.input(|i| i.modifiers);
        let tool = self.prefs.tool;
        if resp.drag_started_by(PointerButton::Primary)
            && let Some(o) = ui.input(|i| i.pointer.press_origin())
        {
            self.drag = Some(match tool {
                Tool::Select => match self.grab_handle(view, o) {
                    Some(d) => d,
                    None => match self.vertex_at(view, o) {
                        Some((pi, vi)) => {
                            self.checkpoint();
                            self.selected = Some(pi);
                            Drag::Vertex(pi, vi)
                        }
                        None => match self.polygon_at(view.to_img(o)) {
                            Some(pi) => {
                                self.checkpoint();
                                self.selected = Some(pi);
                                Drag::Move(pi, self.polys[pi].clone(), view.to_img(o))
                            }
                            None => Drag::Pan,
                        },
                    },
                },
                Tool::Brush => self.grab_handle(view, o).unwrap_or_else(|| Drag::Stroke(vec![view.to_img(o)])),
                Tool::Eraser => {
                    self.checkpoint();
                    let any = self.erase(view, o, o);
                    Drag::Erase(o, any)
                }
                Tool::Scale if self.zone_for.is_some() => Drag::Zone(view.to_img(o)),
                Tool::Polygon | Tool::Scale => Drag::Pan,
            });
        }
        if resp.dragged_by(PointerButton::Primary) && let Some(pos) = pointer {
            let img = self.clamp(view.to_img(pos));
            match &mut self.drag {
                Some(Drag::Vertex(pi, vi)) => self.polys[*pi][*vi] = img,
                Some(Drag::Move(pi, orig, start)) => {
                    let (pi, orig, start) = (*pi, orig.clone(), *start);
                    let b = orig.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, q| [b[0].min(q[0]), b[1].min(q[1]), b[2].max(q[0]), b[3].max(q[1])]);
                    let s = self.loaded.as_ref().map_or(vec2(1e9, 1e9), |l| l.size);
                    let to = view.to_img(pos);
                    let dx = (to[0] - start[0]).clamp(-b[0], s.x as f64 - 1.0 - b[2]);
                    let dy = (to[1] - start[1]).clamp(-b[1], s.y as f64 - 1.0 - b[3]);
                    self.polys[pi] = orig.iter().map(|q| [((q[0] + dx) * 10.0).round() / 10.0, ((q[1] + dy) * 10.0).round() / 10.0]).collect();
                }
                Some(Drag::Pan) => self.pan += resp.drag_delta(),
                Some(Drag::Stroke(pts)) => {
                    if pts.last().is_none_or(|l| view.to_screen(*l).distance(pos) > 2.0) {
                        pts.push(view.to_img(pos));
                    }
                }
                Some(Drag::Erase(prev, _)) => {
                    let a = *prev;
                    let hit = self.erase(view, a, pos);
                    if let Some(Drag::Erase(prev, any)) = &mut self.drag {
                        *prev = pos;
                        *any |= hit;
                    }
                }
                Some(Drag::Zone(_)) | None => {}
            }
        }
        if resp.drag_stopped_by(PointerButton::Primary) {
            match self.drag.take() {
                Some(Drag::Vertex(..)) | Some(Drag::Move(..)) => self.save(p),
                Some(Drag::Stroke(pts)) => self.commit_stroke(p, &pts),
                Some(Drag::Erase(_, any)) => {
                    if any {
                        self.save(p);
                    } else {
                        self.undo.pop();
                    }
                }
                Some(Drag::Zone(start)) => {
                    if let (Some(pos), Some(i)) = (pointer, self.zone_for.take()) {
                        let end = self.clamp(view.to_img(pos));
                        if (end[0] - start[0]).abs() >= 5.0 && (end[1] - start[1]).abs() >= 5.0 && i < self.scales.len() {
                            self.scales[i]["zone"] = json!([start[0].min(end[0]), start[1].min(end[1]), start[0].max(end[0]), start[1].max(end[1])]);
                            self.save_scales(p);
                            self.msg.clear();
                        }
                    }
                }
                Some(Drag::Pan) | None => {}
            }
        }

        let Some(pos) = pointer else { return };
        if resp.secondary_clicked() && let Some((pi, vi)) = self.vertex_at(view, pos) {
            if self.polys[pi].len() > 3 {
                self.checkpoint();
                self.polys[pi].remove(vi);
                self.save(p);
            }
            return;
        }
        if resp.double_clicked() && tool == Tool::Polygon {
            self.finish_draft(p);
            return;
        }
        if !resp.clicked() {
            return;
        }
        let img = view.to_img(pos);
        match tool {
            Tool::Select => {
                if (mods.alt || mods.shift)
                    && let Some((pi, vi)) = self.vertex_at(view, pos)
                {
                    if self.polys[pi].len() > 3 {
                        self.checkpoint();
                        self.polys[pi].remove(vi);
                        self.save(p);
                    }
                } else if let (Some(si), Some(e)) = (self.selected, self.edge_at(view, pos)) {
                    self.checkpoint();
                    self.polys[si].insert(e + 1, img);
                    self.save(p);
                } else {
                    self.selected = self.polygon_at(img);
                }
            }
            Tool::Brush => {
                let hit = self.polygon_at(img);
                if hit.is_some() || self.selected.is_some() {
                    self.selected = hit;
                } else {
                    self.commit_stroke(p, &[img]);
                }
            }
            Tool::Eraser => {
                self.checkpoint();
                if self.erase(view, pos, pos) {
                    self.save(p);
                } else {
                    self.undo.pop();
                }
            }
            Tool::Polygon => {
                let d = &self.draft;
                if d.len() >= 3 && view.to_screen(d[0]).distance(pos) < 12.0 {
                    self.finish_draft(p);
                } else if d.last().is_none_or(|l| view.to_screen(*l).distance(pos) >= 5.0) {
                    self.draft.push(self.clamp(img));
                }
            }
            Tool::Scale if self.zone_for.is_none() => match self.ruler {
                None => self.ruler = Some((img, None)),
                Some((p1, None)) if view.to_screen(p1).distance(pos) >= 5.0 => {
                    self.ruler = Some((p1, Some(img)));
                    self.ruler_cm.clear();
                }
                _ => {}
            },
            Tool::Scale => {}
        }
    }

    /// "Distance (cm)" popup next to a freshly drawn ruler.
    fn ruler_popup(&mut self, ctx: &egui::Context, view: &View, p: &Project) {
        let Some((p1, Some(p2))) = self.ruler else { return };
        let mid = view.to_screen([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]) + vec2(-120.0, 16.0);
        let (mut ok, mut cancel) = (false, false);
        egui::Area::new(egui::Id::new("ruler_popup")).fixed_pos(mid).order(egui::Order::Foreground).show(ctx, |ui| {
            ui::card().inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}  Distance (cm):", ui::I_RULERS)).size(13.5).color(ui::TEXT));
                    let e = ui.add(egui::TextEdit::singleline(&mut self.ruler_cm).hint_text("e.g. 5").desired_width(70.0));
                    if !e.has_focus() && !e.lost_focus() {
                        e.request_focus();
                    }
                    ok = ui::button(ui, Kind::Primary, ui::I_CHECK2, "", vec2(32.0, 28.0), true).clicked() || ui.input(|i| i.key_pressed(Key::Enter));
                    cancel = ui::button(ui, Kind::Secondary, ui::I_X, "", vec2(32.0, 28.0), true).clicked() || ui.input(|i| i.key_pressed(Key::Escape));
                });
            });
        });
        if ok && let Ok(cm) = self.ruler_cm.trim().replace(',', ".").parse::<f64>()
            && cm > 0.0
        {
            self.scales.push(json!({ "p1": p1, "p2": p2, "real_cm": cm, "zone": null }));
            self.save_scales(p);
            self.ruler = None;
        } else if cancel {
            self.ruler = None;
        }
    }

    fn paint(&self, ui: &egui::Ui, painter: &egui::Painter, view: &View, size: Vec2, hover: Option<Pos2>) {
        let loaded = self.loaded.as_ref().unwrap();
        let img_rect = Rect::from_min_size(view.to_screen([0.0, 0.0]), size * view.zoom);
        painter.rect_filled(img_rect.translate(vec2(0.0, 4.0)).expand(2.0), 2.0, Color32::from_black_alpha(18));
        painter.image(loaded.tex.id(), img_rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);

        for (i, poly) in self.polys.iter().enumerate() {
            let selected = self.selected == Some(i);
            let color = if selected { SELECTED } else { self.color(i) };
            let pts: Vec<Pos2> = poly.iter().map(|v| view.to_screen(*v)).collect();
            // live preview of the dilated card boundary
            if self.expansion > 0 {
                let w = (self.expansion as f32 * view.zoom * 2.0).max(3.0);
                painter.add(Shape::closed_line(pts.clone(), Stroke::new(w, color.gamma_multiply(if selected { 0.4 } else { 0.35 }))));
            }
            let mut mesh = egui::Mesh::default();
            let fill = color.gamma_multiply(if selected { 0.25 } else { 0.22 });
            for pt in &pts {
                mesh.colored_vertex(*pt, fill);
            }
            for t in geom::triangulate(poly) {
                mesh.add_triangle(t[0] as u32, t[1] as u32, t[2] as u32);
            }
            painter.add(Shape::mesh(mesh));
            painter.add(Shape::closed_line(pts.clone(), Stroke::new(if selected { 2.5 } else { 2.0 }, color)));
            if selected {
                let dragged = match self.drag {
                    Some(Drag::Vertex(pi, vi)) if pi == i => Some(vi),
                    _ => None,
                };
                for (k, pt) in pts.iter().enumerate() {
                    let on = dragged == Some(k);
                    painter.circle(*pt, if on { 4.5 } else { 3.5 }, if on { SELECTED } else { Color32::WHITE }, Stroke::new(1.5, SELECTED));
                }
            }
            if let Some(top) = pts.iter().min_by(|a, b| a.y.total_cmp(&b.y)) {
                let galley = painter.layout_no_wrap(format!("#{}", i + 1), FontId::new(11.0, ui::semibold()), Color32::WHITE);
                let r = Rect::from_center_size(*top - vec2(0.0, 13.0), galley.size() + vec2(10.0, 6.0));
                painter.rect(r, 4.0, color, Stroke::NONE, StrokeKind::Inside);
                painter.galley(r.center() - galley.size() / 2.0, galley, Color32::WHITE);
            }
        }

        // scales: rulers, zones, per-vessel assignment
        for (i, s) in self.scales.iter().enumerate() {
            let pt = |k: &str| [s[k][0].as_f64().unwrap_or(0.0), s[k][1].as_f64().unwrap_or(0.0)];
            let (a, b) = (view.to_screen(pt("p1")), view.to_screen(pt("p2")));
            if a.distance(b) >= 1.0 {
                let st = Stroke::new(2.0, AMBER);
                painter.line_segment([a, b], st);
                let n = vec2(b.y - a.y, a.x - b.x).normalized() * 6.0;
                painter.line_segment([a - n, a + n], st);
                painter.line_segment([b - n, b + n], st);
                let ratio = project::scale_ratio(s).map_or(String::new(), |r| format!(" · {r:.1} px/cm"));
                painter.text(pos2((a.x + b.x) / 2.0 + 4.0, (a.y + b.y) / 2.0 - 4.0), egui::Align2::LEFT_BOTTOM, format!("Z{} · {}cm{ratio}", i + 1, s["real_cm"]), FontId::new(11.0, ui::semibold()), AMBER);
            }
            if let Some(z) = project::scale_zone(s) {
                let c = scales::color(i, 0.45);
                let r = Rect::from_two_pos(view.to_screen([z[0], z[1]]), view.to_screen([z[2], z[3]]));
                painter.rect_filled(r, 0.0, c.gamma_multiply(0.25));
                dashed_rect(painter, r, c);
                painter.text(r.min + vec2(6.0, 5.0), egui::Align2::LEFT_TOP, format!("Z{}", i + 1), FontId::new(12.0, ui::semibold()), c);
            }
        }
        if !self.scales.is_empty() {
            for (c, z, conflict) in self.assignment() {
                let (text, fill) = match (conflict, z) {
                    (true, _) => ("!".to_string(), RED),
                    (false, Some(z)) => (format!("Z{}", z + 1), scales::color(z, 0.45)),
                    (false, None) => ("?".to_string(), Color32::from_rgba_unmultiplied(100, 116, 139, 217)),
                };
                let galley = painter.layout_no_wrap(text, FontId::new(11.0, ui::semibold()), Color32::WHITE);
                let r = Rect::from_center_size(view.to_screen(c), vec2((galley.size().x + 12.0).max(20.0), 20.0));
                painter.rect(r, 10.0, fill, Stroke::new(1.5, Color32::WHITE), StrokeKind::Outside);
                painter.galley(r.center() - galley.size() / 2.0, galley, Color32::WHITE);
            }
        }

        // drafts
        if !self.draft.is_empty() {
            let mut pts: Vec<Pos2> = self.draft.iter().map(|v| view.to_screen(*v)).collect();
            if let Some(h) = hover {
                pts.push(h);
            }
            painter.add(Shape::line(pts, Stroke::new(2.0, SELECTED)));
            for (k, v) in self.draft.iter().enumerate() {
                painter.circle_filled(view.to_screen(*v), if k == 0 { 6.0 } else { 4.0 }, if k == 0 { Color32::from_rgb(0x16, 0xa3, 0x4a) } else { SELECTED });
            }
        }
        if let Some((p1, p2)) = self.ruler {
            let a = view.to_screen(p1);
            match (p2, hover) {
                (Some(p2), _) => {
                    painter.line_segment([a, view.to_screen(p2)], Stroke::new(2.0, AMBER));
                }
                (None, Some(h)) => painter.extend(Shape::dashed_line(&[a, h], Stroke::new(1.5, AMBER), 5.0, 3.0)),
                _ => {}
            }
            painter.circle_filled(a, 4.0, AMBER);
        }
        if let (Some(Drag::Zone(start)), Some(h)) = (&self.drag, hover) {
            let r = Rect::from_two_pos(view.to_screen(*start), h);
            painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(254, 243, 199, 64));
            dashed_rect(painter, r, AMBER);
        }
        if let Some(Drag::Stroke(pts)) = &self.drag {
            let r = self.prefs.brush;
            let pts: Vec<Pos2> = pts.iter().map(|v| view.to_screen(*v)).collect();
            let col = Color32::from_rgba_unmultiplied(239, 68, 68, 150);
            for pt in &pts {
                painter.circle_filled(*pt, r, col);
            }
            painter.add(Shape::line(pts, Stroke::new(r * 2.0, col)));
        }
        // brush / eraser footprint
        if let (Tool::Brush | Tool::Eraser, Some(h)) = (self.prefs.tool, hover) {
            let r = self.prefs.brush;
            if self.prefs.tool == Tool::Brush {
                painter.circle(h, r, Color32::from_rgba_unmultiplied(255, 0, 0, 64), Stroke::new(1.5, RED));
            } else {
                painter.circle_filled(h, r, Color32::from_white_alpha(90));
                let ring: Vec<Pos2> = (0..=48).map(|k| h + vec2((k as f32 / 48.0 * std::f32::consts::TAU).cos(), (k as f32 / 48.0 * std::f32::consts::TAU).sin()) * r).collect();
                painter.extend(Shape::dashed_line(&ring, Stroke::new(1.5, Color32::from_rgb(0x1e, 0x29, 0x3b)), 6.0, 4.0));
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
    }
}

fn zone_chip(ui: &mut egui::Ui, z: Option<usize>, conflict: bool) {
    let (text, fill, tip) = match (conflict, z) {
        (true, _) => ("!".to_string(), RED, "More than one zone contains this vessel: the smallest zone is used"),
        (false, Some(z)) => (format!("Z{}", z + 1), scales::color(z, 0.45), "Scale used for this vessel"),
        (false, None) => ("?".to_string(), ui::MUTED, "No scale applies to this vessel"),
    };
    ui::chip(ui, &text, Color32::WHITE, fill, false).on_hover_text(tip);
}

fn dashed_rect(painter: &egui::Painter, r: Rect, c: Color32) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    painter.extend(Shape::dashed_line(&pts, Stroke::new(1.5, c), 5.0, 3.0));
}

fn clear_border(m: &mut GrayImage) {
    let (w, h) = m.dimensions();
    for x in 0..w {
        m.put_pixel(x, 0, Luma([0]));
        m.put_pixel(x, h - 1, Luma([0]));
    }
    for y in 0..h {
        m.put_pixel(0, y, Luma([0]));
        m.put_pixel(w - 1, y, Luma([0]));
    }
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_stroke_becomes_and_merges_polygons() {
        let root = std::env::temp_dir().join(format!("lens_brush_{}", std::process::id()));
        let p = project::create(&root, "B", "", 1).unwrap();
        let ctx = egui::Context::default();
        let tex = ctx.load_texture("t", egui::ColorImage::filled([4, 4], Color32::WHITE), Default::default());
        let mut e = Editor { images: vec!["x.jpg".into()], counts: vec![0], zoom: 1.0, ..Default::default() };
        e.loaded = Some(Loaded { name: "x.jpg".into(), tex, size: vec2(1000.0, 1000.0) });
        // a horizontal stroke: one new vessel ~ 200 × 40
        e.commit_stroke(&p, &[[100.0, 100.0], [300.0, 100.0]]);
        assert_eq!(e.polys.len(), 1);
        let a = geom::area(&e.polys[0]);
        assert!((7000.0..10000.0).contains(&a), "area {a}");
        // a far stroke: second vessel; then a stroke bridging both merges them
        e.commit_stroke(&p, &[[100.0, 300.0], [300.0, 300.0]]);
        assert_eq!(e.polys.len(), 2);
        e.commit_stroke(&p, &[[200.0, 100.0], [200.0, 300.0]]);
        assert_eq!(e.polys.len(), 1);
        assert!(geom::contains(&e.polys[0], 200.0, 200.0));
        // eraser: sweeping over every vertex deletes the vessel
        let view = View { rect: Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 1000.0)), zoom: 1.0, pan: Vec2::ZERO };
        e.prefs.brush = 2000.0;
        assert!(e.erase(&view, pos2(0.0, 0.0), pos2(1.0, 1.0)));
        assert!(e.polys.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
