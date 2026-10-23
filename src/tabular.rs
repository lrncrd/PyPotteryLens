//! Tabular metadata: cards/mask_info.csv holds one row per card (`file`, `mask_file`, then any
//! user/AI columns); mask_info_annots.csv gives each card's bbox on its page. Port of /tabular/*.

use crate::project::{self, IMAGE_EXTS, Project};
use crate::ui::{self, Kind};
use crate::ai;
use anyhow::{Result, bail};
use eframe::egui::{self, Color32, FontId, Frame, Margin, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle, Vec2, vec2};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct Annot {
    pub page: String,
    pub id: String,
    pub bbox: [i64; 4],
}

pub struct Tabular {
    pub header: Vec<String>,
    pub rows: Vec<HashMap<String, String>>,
    pub annots: Vec<Annot>,
    pub pages: Vec<String>,
}

/// `Osa_page_0b_mask_layer_3.png` → `3`
pub fn id_of(mask_file: &str) -> String {
    mask_file.split_once("layer_").map_or("0".into(), |(_, id)| id.trim_end_matches(".png").to_string())
}

pub fn page_file(p: &Project, page: &str) -> Option<PathBuf> {
    IMAGE_EXTS.iter().map(|e| p.dir("images").join(format!("{page}.{e}"))).find(|f| f.exists())
}

impl Tabular {
    pub fn load(p: &Project) -> Result<Self> {
        let cards = p.dir("cards");
        let (info, annots) = (cards.join("mask_info.csv"), cards.join("mask_info_annots.csv"));
        if !info.exists() || !annots.exists() {
            bail!("Annotation CSV files not found: extract cards first");
        }
        let mut r = csv::Reader::from_path(&info)?;
        let header: Vec<String> = r.headers()?.iter().map(String::from).collect();
        let rows = r.records().flatten().map(|rec| header.iter().cloned().zip(rec.iter().map(String::from)).collect()).collect();

        let mut r = csv::Reader::from_path(&annots)?;
        let annots: Vec<Annot> = r
            .records()
            .flatten()
            .filter_map(|rec| {
                let mask_file = rec.get(1)?;
                let (page, _) = mask_file.split_once("_mask_layer_")?;
                let c: Vec<i64> = rec.get(0)?.trim_matches(|c| c == '(' || c == ')').split(',').filter_map(|v| v.trim().parse().ok()).collect();
                (c.len() == 4).then(|| Annot { page: page.into(), id: id_of(mask_file), bbox: [c[0], c[1], c[2], c[3]] })
            })
            .collect();
        let mut pages: Vec<String> = annots.iter().map(|a| a.page.clone()).collect();
        pages.sort_by(|a, b| project::natural_cmp(a, b));
        pages.dedup();
        Ok(Self { header, rows, annots, pages })
    }

    pub fn save(&self, p: &Project) -> Result<()> {
        let mut w = csv::Writer::from_path(p.dir("cards").join("mask_info.csv"))?;
        w.write_record(&self.header)?;
        for r in &self.rows {
            w.write_record(self.header.iter().map(|h| r.get(h).map_or("", |v| v.as_str())))?;
        }
        w.flush()?;
        Ok(())
    }

    /// Columns shown in the table: ID, then everything but the internal file columns.
    pub fn columns(&self) -> Vec<String> {
        std::iter::once("ID".to_string()).chain(self.header.iter().filter(|h| *h != "file" && *h != "mask_file").cloned()).collect()
    }

    pub fn page_rows(&self, page: &str) -> Vec<usize> {
        (0..self.rows.len()).filter(|&i| self.rows[i].get("file").map(String::as_str) == Some(page)).collect()
    }

    pub fn page_annots(&self, page: &str) -> impl Iterator<Item = &Annot> {
        self.annots.iter().filter(move |a| a.page == page)
    }

    /// Row of card `id` on `page` (mask_file with or without extension).
    pub fn find_row(&self, page: &str, id: &str) -> Option<usize> {
        let want = format!("{page}_mask_layer_{id}");
        self.rows.iter().position(|r| r.get("mask_file").is_some_and(|m| m.trim_end_matches(".png") == want))
    }

    pub fn add_column(&mut self, name: &str) {
        if !self.header.iter().any(|h| h == name) && name != "ID" {
            self.header.push(name.into());
        }
    }

    pub fn set(&mut self, row: usize, col: &str, value: String) {
        self.add_column(col);
        self.rows[row].insert(col.into(), value);
    }

    /// Clear every value of a page except the internal columns (the ID stays, like the web app).
    pub fn clear_page(&mut self, page: &str) {
        for i in self.page_rows(page) {
            for (k, v) in self.rows[i].iter_mut() {
                if k != "file" && k != "mask_file" {
                    v.clear();
                }
            }
        }
    }

    /// Pages in `pages` where any AI column already has a value (to warn before a batch overwrites).
    pub fn already_filled(&self, pages: &[String]) -> usize {
        let cols = ["page", "plate", "figure", "number"];
        pages
            .iter()
            .filter(|pg| self.page_rows(pg).iter().any(|&i| cols.iter().any(|c| self.rows[i].get(*c).is_some_and(|v| !v.trim().is_empty()))))
            .count()
    }
}

// ---------------------------------------------------------------- tab "4. Tabular Information"

enum Confirm {
    Clear,
    Fill,
    Batch(Vec<String>, usize),
}

#[derive(PartialEq)]
enum Drawer {
    Backend,
    Prompt,
}

pub struct View {
    table: Result<Tabular, String>,
    page: usize,
    show_boxes: bool,
    new_col: String,
    fill_col: String,
    fill_val: String,
    pub ai: ai::Settings,
    drawer: Option<Drawer>,
    batch: (usize, usize), // 1-based, inclusive
    confirm: Option<Confirm>,
    tex: Option<(String, TextureHandle, Vec2)>, // page, texture, full-res size
    hover: Option<String>,
    dirty: bool,
    msg: String,
    hide_list: bool,
    magnify: Option<egui::Pos2>, // zoom 2× about this point of the preview (fraction of the image)
    editing: Option<(usize, egui::Pos2)>, // row of the card whose box was clicked, popup position
}

impl Default for View {
    fn default() -> Self {
        Self {
            table: Err(String::new()),
            page: 0,
            show_boxes: true,
            new_col: String::new(),
            fill_col: String::new(),
            fill_val: String::new(),
            ai: ai::load_settings(),
            drawer: None,
            batch: (1, 1),
            confirm: None,
            tex: None,
            hover: None,
            dirty: false,
            msg: String::new(),
            hide_list: false,
            magnify: None,
            editing: None,
        }
    }
}

const BLUE: Color32 = Color32::from_rgb(0x25, 0x63, 0xeb);
const ORANGE: Color32 = Color32::from_rgb(0xf9, 0x73, 0x16);

impl View {
    /// (Re)load from disk; keeps the current page so a finished AI job lands where it started.
    pub fn reset(&mut self, p: &Project) {
        self.table = Tabular::load(p).map_err(|e| e.to_string());
        let n = self.table.as_ref().map_or(0, |t| t.pages.len());
        self.page = self.page.min(n.saturating_sub(1));
        self.batch = (1, n.max(1));
        self.dirty = false;
    }

    fn save(&mut self, p: &Project) {
        if let Ok(t) = &self.table {
            match t.save(p) {
                Ok(()) => self.dirty = false,
                Err(e) => self.msg = format!("Save failed: {e:#}"),
            }
        }
    }

    /// Returns the pages to run the AI on when the user asks for it.
    pub fn ui(&mut self, ui: &mut egui::Ui, p: &mut Project) -> Option<Vec<String>> {
        ui::page_heading(ui, ui::I_TABLE, "Tabular Information", "View, edit, and transcribe archaeological metadata associated with detected ceramic drawings");
        let pages = match &self.table {
            Ok(t) if !t.pages.is_empty() => t.pages.clone(),
            _ => {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.label(RichText::new(ui::I_TABLE).size(36.0).color(ui::MUTED));
                    ui.label(ui::muted("No card metadata yet: review the polygons in step 3, then Extract Cards."));
                });
                return None;
            }
        };
        let page = pages[self.page].clone();
        let mut run = None;

        // toolbar
        ui::sand_card().inner_margin(Margin::symmetric(16, 10)).show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut go = None;
                let (icon, text) = if self.hide_list { (ui::I_SIDEBAR_INSET, "Show Plates") } else { (ui::I_SIDEBAR, "Plates List") };
                if ui::button(ui, Kind::Secondary, icon, text, vec2(0.0, 34.0), true).clicked() {
                    self.hide_list = !self.hide_list;
                }
                ui.add_space(8.0);
                if ui::button(ui, Kind::Secondary, ui::I_LEFT, "Prev", vec2(0.0, 34.0), self.page > 0).clicked() {
                    go = Some(self.page - 1);
                }
                ui::chip(ui, &format!("Plate {} of {}", self.page + 1, pages.len()), ui::PRIMARY, ui::PEACH, false);
                if ui::button(ui, Kind::Secondary, ui::I_RIGHT, "Next", vec2(0.0, 34.0), self.page + 1 < pages.len()).clicked() {
                    go = Some(self.page + 1);
                }
                ui::chip(ui, &page, ui::DIM, Color32::WHITE, true);
                if let Some(i) = go {
                    self.go(p, i);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui::toggle(ui, &mut self.show_boxes, "Show boxes");
                    ui.add_space(8.0);
                    if p.is_reviewed(&page) {
                        ui::button(ui, Kind::Teal, ui::I_CHECK_FILL, "Reviewed", vec2(0.0, 34.0), true);
                    } else if ui::button(ui, Kind::Secondary, ui::I_EYE, "Mark as Reviewed", vec2(0.0, 34.0), true).clicked() {
                        p.set_reviewed(&page, true);
                        if let Err(e) = p.save() {
                            self.msg = format!("Save failed: {e:#}");
                        }
                    }
                    if !self.msg.is_empty() {
                        ui.label(RichText::new(&self.msg).size(13.0).color(ui::DANGER));
                    }
                });
            });
        });
        ui.add_space(14.0);

        let h = ui.available_height();
        let list_w = if self.hide_list { 0.0 } else { 240.0 };
        let left_w = ((ui.available_width() - list_w) * 0.36).max(300.0);
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 20.0;
            if list_w > 0.0 {
                self.page_list(ui, p, &pages, list_w, h - 4.0);
            }
            Frame::new().fill(ui::CANVAS_BG).stroke(Stroke::new(1.0, ui::BORDER)).corner_radius(12).show(ui, |ui| {
                let (rect, resp) = ui.allocate_exact_size(vec2(left_w, h - 4.0), Sense::click());
                self.preview(ui, p, &page, rect, &resp);
            });
            ui.vertical(|ui| {
                run = self.controls(ui, &page, &pages);
                ui.add_space(10.0);
                self.grid(ui, p, &page);
            });
        });
        self.card_editor(ui.ctx(), p);
        self.modals(ui.ctx(), p, &page, &mut run);
        if run.is_some() && self.dirty {
            self.save(p); // the worker reads the CSV from disk
        }
        run
    }

    fn go(&mut self, p: &Project, i: usize) {
        if self.dirty {
            self.save(p);
        }
        self.page = i;
        self.magnify = None;
        self.editing = None;
    }

    /// "Project Plates" sidebar with the reviewed status of every page.
    fn page_list(&mut self, ui: &mut egui::Ui, p: &Project, pages: &[String], w: f32, h: f32) {
        let mut go = None;
        Frame::new().fill(Color32::WHITE).stroke(Stroke::new(1.0, ui::BORDER)).corner_radius(12).show(ui, |ui| ui.vertical(|ui| {
            ui.set_width(w - 2.0);
            ui.set_height(h);
            ui.spacing_mut().item_spacing.y = 0.0;
            let (head, _) = ui.allocate_exact_size(vec2(w - 2.0, 40.0), Sense::hover());
            ui.painter().text(head.left_center() + vec2(14.0, 0.0), egui::Align2::LEFT_CENTER, format!("{}  Project Plates", ui::I_IMAGES), FontId::new(14.5, ui::semibold()), ui::TEXT);
            ui.painter().hline(head.x_range(), head.bottom(), Stroke::new(1.0, ui::BORDER));
            const ROW: f32 = 38.0;
            egui::ScrollArea::vertical().id_salt("plates").auto_shrink(false).show_rows(ui, ROW, pages.len(), |ui, range| {
                for i in range {
                    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), ROW), Sense::click());
                    let painter = ui.painter();
                    if i == self.page {
                        painter.rect_filled(rect, 0.0, Color32::from_rgb(0xfb, 0xf3, 0xee));
                        painter.rect_filled(Rect::from_min_size(rect.min, vec2(3.0, ROW)), 0.0, ui::PRIMARY);
                    } else if resp.hovered() {
                        painter.rect_filled(rect, 0.0, ui::BG_PAGE);
                    }
                    painter.hline(rect.x_range(), rect.bottom() - 0.5, Stroke::new(1.0, ui::BORDER));
                    let galley = painter.layout(pages[i].clone(), FontId::proportional(13.0), ui::TEXT, rect.width() - 50.0);
                    painter.galley(egui::pos2(rect.left() + 14.0, rect.center().y - galley.size().y / 2.0), galley, ui::TEXT);
                    let (icon, color) = if p.is_reviewed(&pages[i]) { (ui::I_CHECK_FILL, ui::TEAL) } else { (ui::I_CIRCLE, Color32::from_rgb(0xc8, 0xc2, 0xba)) };
                    painter.text(rect.right_center() - vec2(16.0, 0.0), egui::Align2::CENTER_CENTER, icon, FontId::proportional(14.0), color);
                    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        go = Some(i);
                    }
                }
            });
        }));
        if let Some(i) = go {
            self.go(p, i);
        }
    }

    /// Popup with every field of the card whose box was clicked (`showBboxEditor`).
    fn card_editor(&mut self, ctx: &egui::Context, p: &Project) {
        let Some((row, at)) = self.editing else { return };
        let Ok(t) = &mut self.table else { return };
        if row >= t.rows.len() {
            self.editing = None;
            return;
        }
        let cols: Vec<String> = t.columns().into_iter().skip(1).collect();
        let id = id_of(t.rows[row].get("mask_file").map_or("", String::as_str));
        let (mut close, mut changed) = (false, false);
        let area = egui::Area::new(egui::Id::new("card_editor")).fixed_pos(at + vec2(12.0, -20.0)).order(egui::Order::Foreground).constrain(true).show(ctx, |ui| {
            ui::card().inner_margin(Margin::same(14)).show(ui, |ui| {
                ui.set_width(260.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}  ID: {id}", ui::I_BBOX)).font(FontId::new(14.5, ui::semibold())).color(ui::PRIMARY));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        close |= ui::button(ui, Kind::Secondary, ui::I_X, "", vec2(28.0, 26.0), true).clicked();
                    });
                });
                if cols.is_empty() {
                    ui.label(ui::muted("No columns yet: add one or run Extract References."));
                }
                for c in &cols {
                    ui.label(RichText::new(c).size(12.5).color(ui::DIM));
                    let v = t.rows[row].entry(c.clone()).or_default();
                    changed |= ui.add(egui::TextEdit::singleline(v).desired_width(f32::INFINITY).margin(Margin::symmetric(8, 5))).changed();
                }
                ui.add_space(4.0);
                let w = ui.available_width();
                close |= ui::button(ui, Kind::Primary, ui::I_CHECK2, "Save", vec2(w, 34.0), true).clicked();
            });
        });
        self.dirty |= changed;
        let outside = ctx.input(|i| i.pointer.primary_pressed() && i.pointer.interact_pos().is_some_and(|q| !area.response.rect.contains(q)));
        if close || outside || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.editing = None;
            if self.dirty {
                self.save(p);
            }
        }
    }

    fn preview(&mut self, ui: &mut egui::Ui, p: &Project, page: &str, rect: Rect, resp: &egui::Response) {
        if self.tex.as_ref().is_none_or(|t| t.0 != page) {
            // ponytail: decoded on the UI thread, like the editor
            self.tex = page_file(p, page).and_then(|f| image::open(f).ok()).map(|img| {
                let full = vec2(img.width() as f32, img.height() as f32);
                let img = img.thumbnail(1800, 1800).to_rgba8();
                let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
                (page.to_string(), ui.ctx().load_texture(page, color, egui::TextureOptions::LINEAR), full)
            });
        }
        let Some((_, tex, full)) = &self.tex else {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Image not found", FontId::proportional(14.0), ui::MUTED);
            return;
        };
        let area = rect.shrink(12.0);
        let mut k = (area.width() / full.x).min(area.height() / full.y);
        let mut img_rect = Rect::from_center_size(area.center(), *full * k);
        // click on the picture: inspect it at 2× about that point (`setupMagnifyingGlass`)
        if let Some(f) = self.magnify {
            let m = img_rect.min + img_rect.size() * f.to_vec2();
            img_rect = Rect::from_min_max(m + (img_rect.min - m) * 2.0, m + (img_rect.max - m) * 2.0);
            k *= 2.0;
        }
        let painter = ui.painter_at(rect);
        painter.image(tex.id(), img_rect, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        let hint = if self.magnify.is_some() { "Click to zoom out" } else { "Click a box to edit its card · click elsewhere to zoom" };
        painter.text(rect.right_bottom() - vec2(10.0, 8.0), egui::Align2::RIGHT_BOTTOM, format!("{}  {hint}", ui::I_ZOOM_IN), FontId::proportional(11.5), ui::MUTED);
        let Ok(t) = &self.table else { return };
        let boxes: Vec<(String, Rect)> = t
            .page_annots(page)
            .map(|a| {
                let [x1, y1, x2, y2] = a.bbox.map(|v| v as f32 * k);
                (a.id.clone(), Rect::from_min_max(img_rect.min + vec2(x1, y1), img_rect.min + vec2(x2, y2)))
            })
            .collect();
        let under = |q: egui::Pos2| boxes.iter().filter(|b| b.1.contains(q)).min_by(|a, b| a.1.area().total_cmp(&b.1.area())).map(|b| b.0.clone());
        if self.show_boxes && let Some(q) = resp.hover_pos() {
            if let Some(id) = under(q) {
                self.hover = Some(id);
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            } else {
                ui.ctx().set_cursor_icon(if self.magnify.is_some() { egui::CursorIcon::ZoomOut } else { egui::CursorIcon::ZoomIn });
            }
        }
        if resp.clicked() && let Some(q) = resp.interact_pointer_pos() {
            match under(q).filter(|_| self.show_boxes).and_then(|id| t.find_row(page, &id)) {
                Some(row) => self.editing = Some((row, q)),
                None if self.magnify.is_some() => self.magnify = None,
                None => {
                    let f = (q - img_rect.min) / img_rect.size();
                    self.magnify = Some(egui::pos2(f.x.clamp(0.0, 1.0), f.y.clamp(0.0, 1.0)));
                }
            }
        }
        if !self.show_boxes {
            return;
        }
        for (id, r) in &boxes {
            let r = *r;
            let hov = self.hover.as_deref() == Some(id.as_str()) || self.editing.is_some_and(|(row, _)| t.rows.get(row).and_then(|x| x.get("mask_file")).is_some_and(|m| id_of(m) == *id));
            let color = if hov { ORANGE } else { BLUE };
            if hov {
                painter.rect_filled(r, 0.0, ORANGE.gamma_multiply(0.18));
            }
            painter.rect_stroke(r, 0.0, Stroke::new(if hov { 2.5 } else { 1.5 }, color), StrokeKind::Outside);
            let galley = painter.layout_no_wrap(format!("ID: {id}"), FontId::new(11.5, ui::semibold()), Color32::WHITE);
            let tag = Rect::from_min_size(r.left_top() - vec2(0.0, galley.size().y + 4.0), galley.size() + vec2(8.0, 4.0));
            painter.rect_filled(tag, 2.0, color);
            painter.galley(tag.min + vec2(4.0, 2.0), galley, Color32::WHITE);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui, page: &str, pages: &[String]) -> Option<Vec<String>> {
        let mut run = None;
        let Ok(t) = &mut self.table else { return None };
        let cols: Vec<String> = t.columns().into_iter().skip(1).collect();
        if !cols.contains(&self.fill_col) {
            self.fill_col = cols.first().cloned().unwrap_or_default();
        }
        let field = |ui: &mut egui::Ui, s: &mut String, hint: &str, w: f32| ui.add(egui::TextEdit::singleline(s).hint_text(hint).desired_width(w).margin(Margin::symmetric(10, 7)));
        let h = vec2(0.0, 34.0);
        ui::card().inner_margin(Margin::same(16)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let enter = field(ui, &mut self.new_col, "New column name", 170.0).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let name = self.new_col.trim().to_string();
                let ok = !name.is_empty() && name != "ID" && !t.header.contains(&name);
                if ui::button(ui, Kind::Secondary, ui::I_PLUS, "Add Column", h, ok).clicked() || enter && ok {
                    t.add_column(&name);
                    self.new_col.clear();
                    self.dirty = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Danger, ui::I_ERASER, "Clear Page Data", h, true).clicked() {
                        self.confirm = Some(Confirm::Clear);
                    }
                });
            });
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("fill_col").selected_text(&self.fill_col).width(150.0).show_ui(ui, |ui| {
                    for c in &cols {
                        ui.selectable_value(&mut self.fill_col, c.clone(), c);
                    }
                });
                field(ui, &mut self.fill_val, "Value", 140.0);
                if ui::button(ui, Kind::Secondary, ui::I_FILL, "Fill Column", h, !self.fill_col.is_empty()).clicked() {
                    self.confirm = Some(Confirm::Fill);
                }
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(RichText::new(ui::I_STARS).size(16.0).color(ui::TEAL));
                ui.label(RichText::new("AI transcription").font(FontId::new(15.0, ui::semibold())).color(ui::TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Secondary, ui::I_CHAT, "Prompt", vec2(0.0, 30.0), true).clicked() {
                        self.drawer = Some(Drawer::Prompt);
                    }
                    let key = if self.ai.api_key.trim().is_empty() { "Backend (no key)" } else { "Backend" };
                    if ui::button(ui, Kind::Secondary, ui::I_CLOUD, key, vec2(0.0, 30.0), true).on_hover_text(format!("OpenRouter model: {}", self.ai.model)).clicked() {
                        self.drawer = Some(Drawer::Backend);
                    }
                });
            });
            ui.horizontal(|ui| {
                if ui::button(ui, Kind::Teal, ui::I_STARS, "Extract References", h, true)
                    .on_hover_text("Read page, plate, figure and catalogue number of every drawing on this page")
                    .clicked()
                {
                    run = Some(vec![page.to_string()]);
                }
                ui.add_space(14.0);
                let n = pages.len();
                ui.label(RichText::new("Pages").size(13.5).color(ui::DIM));
                ui.add(egui::DragValue::new(&mut self.batch.0).range(1..=n));
                ui.label(RichText::new("to").size(13.5).color(ui::DIM));
                ui.add(egui::DragValue::new(&mut self.batch.1).range(self.batch.0..=n));
                if ui::button(ui, Kind::TealOutline, ui::I_COLLECTION, "Batch Extract", h, true).clicked() {
                    let sel = pages[self.batch.0 - 1..self.batch.1.max(self.batch.0)].to_vec();
                    match t.already_filled(&sel) {
                        0 => run = Some(sel),
                        k => self.confirm = Some(Confirm::Batch(sel, k)),
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui::toggle(ui, &mut self.ai.numbers_from_crops, "Use crops").on_hover_text("Re-read each inventory number from a zoomed crop: one extra request per drawing, more accurate").changed() {
                    ai::save_settings(&self.ai);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(ui::muted(format!("Model: {}", self.ai.model)));
                });
            });
        });
        run
    }

    fn grid(&mut self, ui: &mut egui::Ui, p: &Project, page: &str) {
        let Ok(t) = &mut self.table else { return };
        let cols = t.columns();
        let rows = t.page_rows(page);
        let (mut changed, mut lost, mut hover) = (false, false, None);
        ui::card().inner_margin(Margin::same(12)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(ui.available_height() - 4.0);
            egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                egui::Grid::new("tabular_grid").striped(true).spacing(vec2(6.0, 4.0)).show(ui, |ui| {
                    for c in &cols {
                        ui.label(RichText::new(c).font(FontId::new(13.0, ui::semibold())).color(ui::TEXT));
                    }
                    ui.end_row();
                    for &r in &rows {
                        let id = id_of(t.rows[r].get("mask_file").map_or("", String::as_str));
                        let l = ui.label(RichText::new(&id).font(FontId::monospace(13.0)).color(ui::PRIMARY));
                        let mut row_hov = l.hovered();
                        for c in &cols[1..] {
                            let v = t.rows[r].entry(c.clone()).or_default();
                            let e = ui.add(egui::TextEdit::singleline(v).desired_width(110.0).margin(Margin::symmetric(6, 4)));
                            changed |= e.changed();
                            lost |= e.lost_focus();
                            row_hov |= e.hovered() || e.has_focus();
                        }
                        if row_hov {
                            hover = Some(id);
                        }
                        ui.end_row();
                    }
                });
                if rows.is_empty() {
                    ui.label(ui::muted("No cards on this page."));
                }
            });
        });
        self.hover = hover;
        self.dirty |= changed;
        if lost && self.dirty {
            self.save(p);
        }
    }

    fn modals(&mut self, ctx: &egui::Context, p: &Project, page: &str, run: &mut Option<Vec<String>>) {
        if let Some(c) = &self.confirm {
            let Ok(t) = &self.table else { return };
            let n = t.page_rows(page).len();
            let (title, body, label) = match c {
                Confirm::Clear => ("Clear Page Data".to_string(), format!("Empty every value of the {n} card(s) on {page}? IDs are kept."), "Clear"),
                Confirm::Fill => ("Fill Column".into(), format!("Set “{}” to “{}” for all {n} row(s) on this page? Existing values are overwritten.", self.fill_col, self.fill_val), "Fill Column"),
                Confirm::Batch(sel, k) => ("Batch Extract".into(), format!("{k} of the {} selected pages already have AI references: they will be overwritten.", sel.len()), "Extract anyway"),
            };
            let (mut ok, mut cancel) = (false, false);
            let r = egui::Modal::new(egui::Id::new("tabular_confirm")).frame(ui::card().inner_margin(Margin::same(26))).show(ctx, |ui| {
                ui.set_width(420.0);
                ui.label(RichText::new(&title).font(FontId::new(19.0, ui::semibold())).color(ui::TEXT));
                ui.label(RichText::new(body).color(ui::DIM));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    cancel = ui::button(ui, Kind::Secondary, "", "Cancel", vec2(120.0, 38.0), true).clicked();
                    let kind = if matches!(c, Confirm::Clear) { Kind::Danger } else { Kind::Primary };
                    ok = ui::button(ui, kind, "", label, vec2(160.0, 38.0), true).clicked();
                });
            });
            if ok {
                let t = self.table.as_mut().unwrap();
                match self.confirm.take().unwrap() {
                    Confirm::Clear => t.clear_page(page),
                    Confirm::Fill => {
                        for r in t.page_rows(page) {
                            t.set(r, &self.fill_col, self.fill_val.clone());
                        }
                    }
                    Confirm::Batch(sel, _) => *run = Some(sel),
                }
                if run.is_none() {
                    self.save(p);
                }
            } else if cancel || r.should_close() {
                self.confirm = None;
            }
        }

        let Some(d) = &self.drawer else { return };
        let mut close = false;
        let r = egui::Modal::new(egui::Id::new("tabular_drawer")).frame(ui::card().inner_margin(Margin::same(26))).show(ctx, |ui| {
            ui.set_width(520.0);
            let s = &mut self.ai;
            if *d == Drawer::Backend {
                ui::section_title(ui, ui::I_CLOUD, "OpenRouter Backend");
                ui.label(RichText::new("Model").color(ui::TEXT));
                ui.add(egui::TextEdit::singleline(&mut s.model).hint_text(ai::DEFAULT_MODEL).desired_width(f32::INFINITY).margin(Margin::symmetric(10, 8)));
                ui.label(ui::muted("Any vision-capable model id from openrouter.ai/models"));
                ui.label(RichText::new("API key").color(ui::TEXT));
                ui.add(egui::TextEdit::singleline(&mut s.api_key).password(true).hint_text("sk-or-…").desired_width(f32::INFINITY).margin(Margin::symmetric(10, 8)));
                ui.label(ui::muted("Kept in memory only (or set OPENROUTER_API_KEY). Never written to disk."));
            } else {
                ui::section_title(ui, ui::I_CHAT, "Additional Prompt");
                ui.label(ui::muted("Extra context for the model. Write a field in [brackets] to get a column with exactly that name, e.g. “also read the [inventory] code”."));
                ui.add(egui::TextEdit::multiline(&mut s.prompt).desired_rows(8).desired_width(f32::INFINITY).margin(Margin::symmetric(10, 8)));
                let fields = ai::canonical_fields(&s.prompt);
                if !fields.is_empty() {
                    ui.label(ui::muted(format!("Extra columns: {}", fields.join(", "))));
                }
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if *d == Drawer::Prompt && ui::button(ui, Kind::Secondary, ui::I_UNDO, "Reset", vec2(0.0, 36.0), !s.prompt.is_empty()).clicked() {
                    s.prompt.clear();
                }
                if *d == Drawer::Backend && ui::button(ui, Kind::Secondary, ui::I_UNDO, "Default model", vec2(0.0, 36.0), s.model != ai::DEFAULT_MODEL).clicked() {
                    s.model = ai::DEFAULT_MODEL.into();
                }
                close = ui::button(ui, Kind::Primary, ui::I_CHECK2, "Done", vec2(120.0, 36.0), true).clicked();
            });
        });
        if close || r.should_close() {
            self.ai.model = self.ai.model.trim().to_string();
            if self.ai.model.is_empty() {
                self.ai.model = ai::DEFAULT_MODEL.into();
            }
            ai::save_settings(&self.ai);
            self.drawer = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_from_mask_files() {
        assert_eq!(id_of("Osa_page_0b_mask_layer_3"), "3");
        assert_eq!(id_of("Osa_page_0b_mask_layer_12.png"), "12");
        assert_eq!(id_of("weird"), "0");
    }

    #[test]
    fn csv_roundtrip_and_edits() {
        let root = std::env::temp_dir().join(format!("lens_tab_{}", std::process::id()));
        let p = project::create(&root, "T", "", 1).unwrap();
        let cards = p.dir("cards");
        let info = "file,mask_file,Notes\nP_page_1,P_page_1_mask_layer_0,\"a, b\"\nP_page_1,P_page_1_mask_layer_1,\nP_page_2,P_page_2_mask_layer_0,x\n";
        std::fs::write(cards.join("mask_info.csv"), info).unwrap();
        std::fs::write(cards.join("mask_info_annots.csv"), "bbox,mask_file\n\"(1, 2, 3, 4)\",P_page_1_mask_layer_0\n\"(5, 6, 7, 8)\",P_page_1_mask_layer_1\n\"(0, 0, 9, 9)\",P_page_2_mask_layer_0\n").unwrap();
        let mut t = Tabular::load(&p).unwrap();
        t.save(&p).unwrap();
        assert_eq!(std::fs::read_to_string(cards.join("mask_info.csv")).unwrap(), info);
        assert_eq!(t.pages, ["P_page_1", "P_page_2"]);
        assert_eq!(t.columns(), ["ID", "Notes"]);
        assert_eq!(t.find_row("P_page_1", "1"), Some(1));
        t.set(0, "number", "7".into());
        assert_eq!(t.already_filled(&t.pages.clone()), 1);
        t.clear_page("P_page_1");
        assert_eq!(t.rows[0]["Notes"], "");
        assert_eq!(t.rows[2]["Notes"], "x");
        let _ = std::fs::remove_dir_all(&root);
    }
}
