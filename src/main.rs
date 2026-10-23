//! PyPotteryLens, native (egui + wgpu). Works on the same `projects/` folder as the Python app.
//!
//!     PYPOTTERY_MODEL_CACHE=../model_cache cargo run --release [-- <projects dir>]
//!
//! Needs libpdfium in `lib/` (https://github.com/bblanchon/pdfium-binaries) and the ONNX model
//! exported by tools/export_onnx.py.

mod ai;
mod cards;
mod detect;
mod editor;
mod geom;
mod pdf;
mod post;
mod project;
mod scales;
mod tabular;
mod thumbs;
mod ui;

use eframe::egui::{self, Color32, CornerRadius, FontId, Frame, Margin, Rect, RichText, Sense, Shape, Stroke, StrokeKind, TextureHandle, pos2, vec2};
use project::Project;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use ui::Kind;

/// Shared between the UI and a worker thread: progress and a cancel flag.
#[derive(Clone, Default)]
pub struct Progress {
    state: Arc<Mutex<(usize, usize, String)>>,
    cancel: Arc<AtomicBool>,
}

impl Progress {
    pub fn set(&self, done: usize, total: usize, msg: impl Into<String>) {
        *self.state.lock().unwrap() = (done, total, msg.into());
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

struct Job {
    progress: Progress,
    handle: JoinHandle<anyhow::Result<String>>,
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Projects,
    Pdf,
    Detect,
    Editor,
    Tabular,
    Post,
}

const TABS: [(Tab, &str, &str); 6] = [
    (Tab::Projects, ui::I_FOLDER, "Project Manager"),
    (Tab::Pdf, ui::I_PDF, "1. PDF Document Processing"),
    (Tab::Detect, ui::I_CPU, "2. Apply Model"),
    (Tab::Editor, ui::I_PENCIL, "3. Review Annotations & Masks"),
    (Tab::Tabular, ui::I_TABLE, "4. Tabular Information"),
    (Tab::Post, ui::I_GRID, "5. Post Processing"),
];

struct Assets {
    logo: TextureHandle,
    icons: Vec<TextureHandle>, // project icon stamps 1..=6
}

struct App {
    root: PathBuf,
    projects: Vec<Project>,
    new_name: String,
    new_desc: String,
    new_icon: usize,
    confirm_delete: Option<usize>,
    project: Option<Project>,
    tab: Tab,
    editor: editor::Editor,
    post: post::State,
    tabular: tabular::View,
    auto_v: bool,
    auto_h: bool,
    export_acronym: Option<String>,
    lightbox: Option<(String, TextureHandle)>,
    info: bool,
    thumbs: thumbs::Thumbs,
    assets: Assets,
    job: Option<Job>,
    status: String,
    split: bool,
    conf: f32,
    diagnostic: bool,
    clean: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext, root: PathBuf) -> Self {
        let mut app = Self::blank(cc, root);
        // reopen the project of the last session, like the web app's localStorage
        let last = ai::read_setting("last_project").and_then(|v| v.as_str().map(String::from));
        if let Some(p) = last.and_then(|id| app.projects.iter().find(|p| p.meta.project_id == id).cloned()) {
            app.open(p);
        }
        app
    }

    fn blank(cc: &eframe::CreationContext, root: PathBuf) -> Self {
        let ctx = &cc.egui_ctx;
        ui::install(ctx);
        let icons = [
            &include_bytes!("../static/imgs/icons/1.png")[..],
            include_bytes!("../static/imgs/icons/2.png"),
            include_bytes!("../static/imgs/icons/3.png"),
            include_bytes!("../static/imgs/icons/4.png"),
            include_bytes!("../static/imgs/icons/5.png"),
            include_bytes!("../static/imgs/icons/6.png"),
        ];
        let assets = Assets {
            logo: ui::load_png(ctx, "logo", include_bytes!("../static/imgs/LogoLens.png")),
            icons: icons.iter().enumerate().map(|(i, b)| ui::load_png(ctx, &format!("icon{i}"), b)).collect(),
        };
        Self {
            projects: project::list(&root),
            root,
            new_name: String::new(),
            new_desc: String::new(),
            new_icon: 1,
            confirm_delete: None,
            project: None,
            tab: Tab::Projects,
            editor: Default::default(),
            post: Default::default(),
            tabular: Default::default(),
            auto_v: true,
            auto_h: true,
            export_acronym: None,
            lightbox: None,
            info: false,
            thumbs: thumbs::Thumbs::new(ctx),
            assets,
            job: None,
            status: String::new(),
            split: false,
            conf: 0.5,
            diagnostic: false,
            clean: true,
        }
    }

    fn open(&mut self, p: Project) {
        let id = p.meta.project_id.clone();
        if !cfg!(test) && self.project.as_ref().is_none_or(|a| a.meta.project_id != id) {
            ai::update_settings(|m| {
                m.insert("last_project".into(), serde_json::json!(id));
            });
        }
        self.editor.reset(&p);
        self.post = post::State::load(&p);
        self.tabular.reset(&p);
        self.project = Some(p);
    }

    fn spawn(&mut self, f: impl FnOnce(&mut Project, &Progress) -> anyhow::Result<String> + Send + 'static) {
        let Some(mut p) = self.project.clone() else { return };
        let progress = Progress::default();
        let prog = progress.clone();
        let handle = std::thread::spawn(move || f(&mut p, &prog));
        self.job = Some(Job { progress, handle });
    }

    fn start_pdf(&mut self, path: PathBuf) {
        let split = self.split;
        self.spawn(move |p, prog| pdf::convert(p, &path, split, prog));
    }

    fn poll_job(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        if !job.handle.is_finished() {
            ctx.request_repaint_after(Duration::from_millis(100));
            return;
        }
        let job = self.job.take().unwrap();
        self.status = match job.handle.join() {
            Ok(Ok(msg)) => msg,
            Ok(Err(e)) => format!("Error: {e:#}"),
            Err(_) => "Error: the worker crashed".into(),
        };
        // the worker wrote project.json and files; reload everything from disk
        if let Some(p) = self.project.as_ref().and_then(|p| Project::open(&p.root).ok()) {
            self.open(p);
        }
        self.projects = project::list(&self.root);
        self.thumbs.clear();
    }

    fn header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::Image::new(&self.assets.logo).fit_to_exact_size(vec2(56.0, 56.0)));
            ui.add_space(8.0);
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.label(RichText::new("PyPottery").font(FontId::new(30.0, ui::xbold())).color(ui::TEXT));
            ui.label(RichText::new("Lens").font(FontId::new(30.0, ui::xbold())).color(ui::PRIMARY));
            ui.add_space(4.0);
            ui::chip(ui, &format!("v{}", include_str!("../VERSION").trim()), ui::PRIMARY, ui.visuals().widgets.hovered.weak_bg_fill, true);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui::button(ui, Kind::Secondary, ui::I_INFO, "Info", vec2(0.0, 36.0), true).on_hover_text("App Information").clicked() {
                    self.info = true;
                }
                ui.add_space(8.0);
                let (label, name, fill, stroke, fg) = match &self.project {
                    Some(p) => ("PROJECT:", p.meta.project_name.as_str(), ui::PEACH, ui::PRIMARY_BORDER, ui::PRIMARY),
                    None => ("PROJECT:", "No project selected", ui::BG_PAGE, ui::BORDER, ui::MUTED),
                };
                let text = format!("{}   {label}  {name}", ui::I_FOLDER);
                let galley = ui.painter().layout_no_wrap(text, FontId::new(13.5, ui::semibold()), fg);
                let (rect, resp) = ui.allocate_exact_size(galley.size() + vec2(32.0, 18.0), Sense::click());
                let stroke = if resp.hovered() { ui::PRIMARY } else { stroke };
                ui.painter().rect(rect, 99.0, fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
                ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
                if resp.on_hover_text("Active project (click to manage projects)").on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    self.tab = Tab::Projects;
                }
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.set_height(30.0);
            if let Some(job) = &self.job {
                let (done, total, msg) = job.progress.state.lock().unwrap().clone();
                ui.label(RichText::new(ui::I_HOURGLASS).color(ui::PRIMARY));
                ui.label(RichText::new(&msg).color(ui::TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Secondary, ui::I_X, "Cancel", vec2(0.0, 30.0), true).clicked() {
                        job.progress.cancel.store(true, Ordering::Relaxed);
                    }
                    ui.label(RichText::new(format!("{done} / {total}")).font(FontId::monospace(13.0)).color(ui::DIM));
                    ui::progress(ui, if total > 0 { done as f32 / total as f32 } else { 0.0 }, 360.0);
                });
            } else {
                let error = self.status.starts_with("Error");
                let (icon, color) = if error { (ui::I_WARN, ui::DANGER) } else { (ui::I_CHECK_FILL, ui::TEAL) };
                ui.label(RichText::new(icon).color(color));
                ui.label(RichText::new(&self.status).color(if error { ui::DANGER } else { ui::TEXT }));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Secondary, ui::I_X, "", vec2(30.0, 30.0), true).clicked() {
                        self.status.clear();
                    }
                });
            }
        });
    }

    // ---------------------------------------------------------------- Project Manager

    fn projects_ui(&mut self, ui: &mut egui::Ui) {
        ui::page_heading(ui, ui::I_FOLDER, "Project Manager", "Create, configure, and activate your archaeological ceramic documentation projects");
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(360.0);
                ui::sand_card().show(ui, |ui| self.new_project_form(ui));
            });
            ui.add_space(24.0);
            ui.vertical(|ui| self.repository(ui));
        });
    }

    fn new_project_form(&mut self, ui: &mut egui::Ui) {
        ui::section_title(ui, ui::I_FOLDER_PLUS, "New Project Dossier");
        ui.separator();
        let label = |ui: &mut egui::Ui, text: &str, extra: RichText| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(RichText::new(text).size(15.0).color(ui::TEXT));
                ui.label(extra);
            });
        };
        label(ui, "Project Name", RichText::new("*").color(ui::PRIMARY));
        ui.add(egui::TextEdit::singleline(&mut self.new_name).hint_text("e.g. Scavi_Capena_2024").desired_width(f32::INFINITY).margin(Margin::symmetric(12, 10)));
        label(ui, "Description", ui::muted("(optional)"));
        ui.add(egui::TextEdit::multiline(&mut self.new_desc).hint_text("Site context, trench, or ceramic assemblage notes…").desired_rows(3).desired_width(f32::INFINITY).margin(Margin::symmetric(12, 10)));
        label(ui, "Project Icon Stamp", RichText::new(""));
        Frame::new().fill(Color32::WHITE).stroke(Stroke::new(1.0, ui::BORDER)).corner_radius(10).inner_margin(8).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(8.0, 8.0);
                for (i, tex) in self.assets.icons.iter().enumerate() {
                    let (rect, resp) = ui.allocate_exact_size(vec2(66.0, 66.0), Sense::click());
                    let sel = self.new_icon == i + 1;
                    let (fill, stroke) = if sel { (ui::PEACH, ui::PRIMARY) } else if resp.hovered() { (ui::BG_PAGE, ui::PRIMARY_BORDER) } else { (Color32::WHITE, ui::BORDER) };
                    ui.painter().rect(rect, 8.0, fill, Stroke::new(if sel { 2.0 } else { 1.0 }, stroke), StrokeKind::Inside);
                    egui::Image::new(tex).paint_at(ui, Rect::from_center_size(rect.center(), vec2(44.0, 44.0)));
                    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        self.new_icon = i + 1;
                    }
                }
            });
        });
        ui.add_space(6.0);
        let w = ui.available_width();
        if ui::button(ui, Kind::Primary, ui::I_FOLDER_PLUS, "Create Project", vec2(w, 46.0), !self.new_name.trim().is_empty()).clicked() {
            match project::create(&self.root, &self.new_name, &self.new_desc, self.new_icon) {
                Ok(p) => {
                    self.new_name.clear();
                    self.new_desc.clear();
                    self.projects = project::list(&self.root);
                    self.open(p);
                    self.tab = Tab::Pdf;
                }
                Err(e) => self.status = format!("Error: {e:#}"),
            }
        }
    }

    fn repository(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui::section_title(ui, ui::I_ARCHIVE, "Project Repository");
                ui.label(RichText::new("Select a project card to activate it across all tabs").size(13.5).color(ui::DIM));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui::button(ui, Kind::Secondary, ui::I_REFRESH, "Refresh", vec2(0.0, 34.0), true).clicked() {
                    self.projects = project::list(&self.root);
                }
            });
        });
        ui.separator();
        if self.projects.is_empty() {
            ui.label(ui::muted(format!("No projects in {} yet.", self.root.display())));
            return;
        }
        let (mut select, mut delete) = (None, None);
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            let gap = 20.0;
            let w = (ui.available_width() - gap) / 2.0;
            for (row, pair) in self.projects.chunks(2).enumerate() {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = gap;
                    for (k, p) in pair.iter().enumerate() {
                        let i = row * 2 + k;
                        let active = self.project.as_ref().is_some_and(|a| a.meta.project_id == p.meta.project_id);
                        let icon = &self.assets.icons[(p.icon().clamp(1, 6)) - 1];
                        match project_card(ui, p, icon, active, w) {
                            Some(true) => select = Some(i),
                            Some(false) => delete = Some(i),
                            None => {}
                        }
                    }
                });
                ui.add_space(gap - 10.0);
            }
        });
        if let Some(i) = select {
            self.open(self.projects[i].clone());
            self.tab = if self.editor.image_count() > 0 { Tab::Editor } else { Tab::Pdf };
        }
        if delete.is_some() {
            self.confirm_delete = delete;
        }
    }

    fn delete_modal(&mut self, ctx: &egui::Context) {
        let Some(i) = self.confirm_delete else { return };
        let name = self.projects[i].meta.project_name.clone();
        let mut close = false;
        egui::Modal::new(egui::Id::new("confirm_delete")).frame(ui::card().inner_margin(Margin::same(26))).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.label(RichText::new(format!("{}  Delete “{name}”?", ui::I_TRASH)).font(FontId::new(19.0, ui::semibold())).color(ui::TEXT));
            ui.label(RichText::new("The project folder, its images, masks and cards are removed from disk. This cannot be undone.").color(ui::DIM));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui::button(ui, Kind::Secondary, "", "Cancel", vec2(120.0, 38.0), true).clicked() {
                    close = true;
                }
                if ui::button(ui, Kind::Danger, ui::I_TRASH, "Delete project", vec2(160.0, 38.0), true).clicked() {
                    let p = self.projects[i].clone();
                    match p.delete() {
                        Ok(()) => {
                            if self.project.as_ref().is_some_and(|a| a.root == p.root) {
                                self.project = None;
                            }
                            self.status = format!("Deleted project {name}");
                        }
                        Err(e) => self.status = format!("Error: {e:#}"),
                    }
                    self.projects = project::list(&self.root);
                    close = true;
                }
            });
        });
        if close || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.confirm_delete = None;
        }
    }

    // ---------------------------------------------------------------- 1. PDF

    fn pdf_ui(&mut self, ui: &mut egui::Ui) {
        ui::page_heading(ui, ui::I_PDF, "PDF Document Processing", "Extract high-resolution plate illustrations and sherd drawings directly from published monograph PDFs");
        let Some(p) = &self.project else { return };
        let pdfs = project::list_files(&p.dir("pdf_source"), |n| project::has_ext(n, &["pdf"]));
        let images = self.editor.image_count();
        ui.vertical_centered(|ui| {
            ui.set_max_width(860.0);
            let zone = Frame::new().fill(Color32::from_rgb(0xf8, 0xfa, 0xfc)).corner_radius(14).inner_margin(Margin::symmetric(40, 44)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(64.0, 64.0), Sense::hover());
                    ui.painter().circle_filled(r.center(), 32.0, Color32::from_rgb(0xe2, 0xe8, 0xf0));
                    ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, ui::I_CLOUD_UP, FontId::proportional(26.0), Color32::from_rgb(0x64, 0x74, 0x8b));
                    ui.add_space(6.0);
                    let slate = Color32::from_rgb(0x47, 0x55, 0x69);
                    if let Some(name) = pdfs.first() {
                        ui.label(RichText::new(format!("{}  PDF Already Ingested", ui::I_FILE_LOCK)).font(FontId::new(21.0, ui::semibold())).color(slate));
                        ui.label(RichText::new(format!("Active: “{name}”. A project can only contain 1 PDF.")).size(15.0).color(Color32::from_rgb(0x64, 0x74, 0x8b)));
                        ui.add_space(8.0);
                        ui::button(ui, Kind::Primary, ui::I_LOCK, "PDF Already Uploaded", vec2(240.0, 44.0), false);
                    } else {
                        ui.label(RichText::new("Upload a Monograph PDF").font(FontId::new(21.0, ui::semibold())).color(slate));
                        ui.label(RichText::new("Choose a PDF or drop it onto this window.").size(15.0).color(Color32::from_rgb(0x64, 0x74, 0x8b)));
                        ui.add_space(8.0);
                        if ui::button(ui, Kind::Primary, ui::I_PDF, "Choose PDF…", vec2(220.0, 44.0), true).clicked()
                            && let Some(path) = rfd::FileDialog::new().add_filter("PDF", &["pdf"]).pick_file()
                        {
                            self.start_pdf(path);
                        }
                    }
                    ui.add_space(10.0);
                    ui.label(ui::muted("Accepts standard .pdf documents · Pages rendered at 300 DPI"));
                });
            });
            ui::dashed_rect(ui.painter(), zone.response.rect, Color32::from_rgb(0xcb, 0xd5, 0xe1));
            ui.add_space(18.0);
            ui::sand_card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui::toggle(ui, &mut self.split, "");
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("{}  Split Scanned Spreads", ui::I_SPLIT)).font(FontId::new(15.0, ui::semibold())).color(ui::TEXT));
                            ui::chip(ui, &format!("{}  Hover for visual comparison", ui::I_INFO), ui::PRIMARY, ui::PEACH, false).on_hover_ui(spread_guide);
                        });
                        ui.label(RichText::new("Automatically bisect two-page book spreads into separate left and right plates").size(13.5).color(ui::DIM));
                    });
                });
            });
            ui.add_space(12.0);
            ui::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}  INGESTION STATUS", ui::I_IMAGES)).size(12.5).color(ui::MUTED).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (text, fg, fill) = if images > 0 { (format!("{images} plates extracted"), ui::TEAL, ui::TEAL_SOFT) } else { ("No plates yet".into(), ui::MUTED, ui::BG_PAGE) };
                        ui::chip(ui, &text, fg, fill, false);
                    });
                });
            });
        });
    }

    // ---------------------------------------------------------------- 2. Apply Model

    fn detect_ui(&mut self, ui: &mut egui::Ui) {
        let total = self.editor.image_count();
        let excluded = self.editor.images().iter().filter(|n| self.project.as_ref().is_some_and(|p| p.is_excluded(n))).count();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(400.0);
                ui::card().show(ui, |ui| {
                    ui.set_min_height(ui.available_height() - 40.0);
                    ui::section_title(ui, ui::I_CPU, "Model Selection");
                    ui.label(RichText::new("Active Vision Model").color(ui::TEXT));
                    let path = detect::model_path();
                    Frame::new().stroke(Stroke::new(1.0, ui::BORDER)).corner_radius(8).inner_margin(Margin::symmetric(12, 9)).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(ui::I_CPU).color(ui::PRIMARY));
                            ui.label(RichText::new(path.file_name().unwrap_or_default().to_string_lossy()).color(ui::TEXT));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if path.exists() { ui::chip(ui, "Ready", ui::TEAL, ui::TEAL_SOFT, false) } else { ui::chip(ui, "Missing", ui::DANGER, Color32::from_rgb(0xff, 0xf1, 0xf2), false) }
                                    .on_hover_text(path.display().to_string());
                            });
                        });
                    });
                    ui.label(ui::muted("YOLOv8 segmentation network trained on archaeological profiles (ONNX)"));
                    ui.separator();
                    ui::section_title(ui, ui::I_SLIDERS, "Model Parameters");
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Confidence Threshold").color(ui::TEXT));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui::chip(ui, &format!("{:.2}", self.conf), ui::PRIMARY, ui::PEACH, true);
                        });
                    });
                    let w = ui.available_width();
                    if ui::slider(ui, &mut self.conf, 0.05..=0.95, w).changed() {
                        self.conf = (self.conf * 100.0).round() / 100.0;
                    }
                    ui.label(ui::muted("Lower values detect more objects but may increase false positives"));
                    ui.separator();
                    ui::section_title(ui, ui::I_GEAR, "Execution Mode");
                    if option_card(ui, !self.diagnostic, ui::I_COLLECTION, "Full Dataset", &format!("Process all {} selected images in project", total - excluded)) {
                        self.diagnostic = false;
                    }
                    if option_card(ui, self.diagnostic, ui::I_BOLT, "Diagnostic Test", "Process first 25 images only for quick testing") {
                        self.diagnostic = true;
                    }
                    ui.separator();
                    let w = ui.available_width();
                    if ui::button(ui, Kind::Primary, ui::I_PLAY, "Apply Model to Project", vec2(w, 48.0), total > 0).clicked() {
                        let (conf, diag) = (self.conf, self.diagnostic);
                        self.spawn(move |p, prog| detect::run_project(p, conf, diag, prog));
                    }
                    ui.label(ui::muted("Replaces the polygons of every processed image."));
                });
            });
            ui.add_space(24.0);
            ui.vertical(|ui| self.figures(ui, total, excluded));
        });
    }

    /// Page grid: click a page to exclude it from detection.
    fn figures(&mut self, ui: &mut egui::Ui, total: usize, excluded: usize) {
        let Some(p) = &mut self.project else { return };
        ui::sand_card().inner_margin(Margin::symmetric(16, 10)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{}  Project Figures", ui::I_IMAGES)).font(FontId::new(16.0, ui::semibold())).color(ui::TEXT));
                ui::chip(ui, &format!("{total} figures"), ui::DIM, Color32::WHITE, true);
                ui::chip(ui, &format!("{} to process", total - excluded), ui::TEAL, ui::TEAL_SOFT, true);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let all = ui::button(ui, Kind::Secondary, ui::I_CHECK_SQUARE, "Select All", vec2(0.0, 32.0), excluded > 0).clicked();
                    let none = ui::button(ui, Kind::Secondary, ui::I_DASH_SQUARE, "Deselect All", vec2(0.0, 32.0), excluded < total).clicked();
                    if all || none {
                        for n in self.editor.images() {
                            p.set_excluded(n, none);
                        }
                        let _ = p.save();
                    }
                });
            });
        });
        ui.add_space(14.0);
        let dir = p.dir("images");
        let open = ui::card().inner_margin(Margin::same(18)).show(ui, |ui| {
            let gap = 16.0;
            // at least 170px wide, stretched to fill the row
            let cols = (((ui.available_width() + gap) / (170.0 + gap)) as usize).max(1);
            let tw = (ui.available_width() + gap) / cols as f32 - gap;
            let th = tw * 1.3;
            let images = self.editor.images();
            let (mut toggled, mut open) = (None, None);
            egui::ScrollArea::vertical().auto_shrink(false).show_rows(ui, th + gap, images.len().div_ceil(cols), |ui, range| {
                for r in range {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = gap;
                        for name in images.iter().skip(r * cols).take(cols) {
                            let off = p.is_excluded(name);
                            let (rect, resp) = ui.allocate_exact_size(vec2(tw, th), Sense::click());
                            let hov = resp.hovered();
                            let painter = ui.painter();
                            painter.rect(rect, 8.0, if off { ui::BG_PAGE } else { Color32::WHITE }, Stroke::new(if hov { 2.0 } else { 1.0 }, if hov { ui::PRIMARY } else { ui::BORDER }), StrokeKind::Inside);
                            if let Some(tex) = self.thumbs.get(&dir.join(name)) {
                                let area = Rect::from_min_max(rect.min + vec2(8.0, 8.0), rect.max - vec2(8.0, 26.0));
                                let s = tex.size_vec2();
                                let k = (area.width() / s.x).min(area.height() / s.y);
                                let tint = if off { Color32::from_white_alpha(70) } else { Color32::WHITE };
                                painter.image(tex.id(), Rect::from_center_size(area.center(), s * k), Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), tint);
                            }
                            if off {
                                let badge = Rect::from_min_size(rect.min + vec2(8.0, 8.0), vec2(72.0, 20.0));
                                painter.rect_filled(badge, 4.0, Color32::from_rgb(0xd6, 0xcb, 0xbe));
                                painter.text(badge.center(), egui::Align2::CENTER_CENTER, "EXCLUDED", FontId::new(10.5, ui::semibold()), Color32::WHITE);
                            }
                            painter.text(pos2(rect.center().x, rect.bottom() - 13.0), egui::Align2::CENTER_CENTER, project::stem(name), FontId::proportional(11.5), ui::DIM);
                            // corner button: exclude / include; the picture itself opens full size
                            let br = Rect::from_min_size(pos2(rect.right() - 32.0, rect.top() + 8.0), vec2(24.0, 24.0));
                            let b = ui.interact(br, egui::Id::new(("excl", name)), Sense::click()).on_hover_text(if off { "Include in processing" } else { "Exclude from processing" });
                            if hov || b.hovered() || off {
                                let fill = if off { ui::TEAL } else if b.hovered() { ui::DANGER } else { Color32::from_white_alpha(235) };
                                let painter = ui.painter();
                                painter.circle(br.center(), 12.0, fill, Stroke::new(1.0, ui::BORDER));
                                painter.text(br.center(), egui::Align2::CENTER_CENTER, if off { ui::I_CHECK2 } else { ui::I_X }, FontId::proportional(12.0), if off || b.hovered() { Color32::WHITE } else { ui::TEXT });
                            }
                            if b.clicked() {
                                toggled = Some(name.clone());
                            } else if resp.on_hover_text(format!("{name} (click to view full size)")).on_hover_cursor(egui::CursorIcon::ZoomIn).clicked() {
                                open = Some(name.clone());
                            }
                        }
                    });
                }
            });
            if let Some(n) = toggled {
                let off = p.is_excluded(&n);
                p.set_excluded(&n, !off);
                let _ = p.save();
            }
            open
        });
        if let Some(n) = open.inner {
            self.open_lightbox(ui.ctx(), &n, &dir.join(&n));
        }
    }

    fn open_lightbox(&mut self, ctx: &egui::Context, name: &str, path: &std::path::Path) {
        match image::open(path) {
            Ok(img) => {
                // the GPU texture limit; the lightbox shows it at most at screen size anyway
                let img = img.thumbnail(4096, 4096).to_rgba8();
                let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
                self.lightbox = Some((name.to_string(), ctx.load_texture(name, color, egui::TextureOptions::LINEAR)));
            }
            Err(e) => self.status = format!("Error: {e}"),
        }
    }

    // ---------------------------------------------------------------- 4. Cards

    fn post_ui(&mut self, ui: &mut egui::Ui) {
        ui::page_heading(ui, ui::I_GRID, "Post Processing", "Review and classify extracted cards from current project");
        let n = self.post.cards.len();
        let frag = self.post.cards.iter().filter(|c| self.post.card_type(&c.name) == "FRAG").count();
        ui::card().inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{}  Cards: {n}", ui::I_IMAGE)).font(FontId::new(16.0, ui::semibold())).color(ui::TEXT));
                ui.add_space(12.0);
                ui::toggle(ui, &mut self.auto_v, "Auto V-flip").on_hover_text("Turn upside-down profiles the right way up");
                ui.add_space(6.0);
                ui::toggle(ui, &mut self.auto_h, "Auto H-flip").on_hover_text("Put the section on the same side for every card");
                ui.add_space(12.0);
                ui::chip(ui, &format!("ENT {}", n - frag), ui::TEAL, ui::TEAL_SOFT, false);
                ui::chip(ui, &format!("FRAG {frag}"), ui::PRIMARY, ui::PEACH, false);
                if !self.post.excluded.is_empty() {
                    ui::chip(ui, &format!("{} excluded", self.post.excluded.len()), ui::DANGER, Color32::from_rgb(0xff, 0xf1, 0xf2), false);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui::button(ui, Kind::Primary, ui::I_BOX_OUT, "Export Results", vec2(0.0, 40.0), n > 0).clicked() {
                        self.export_acronym = Some(String::new());
                    }
                    if ui::button(ui, Kind::Primary, ui::I_MAGIC, "Process All Images", vec2(0.0, 40.0), n > 0).clicked() {
                        let (v, h) = (self.auto_v, self.auto_h);
                        self.spawn(move |p, prog| post::process(p, v, h, prog));
                    }
                });
            });
        });
        ui.add_space(14.0);
        let Some(p) = &self.project else { return };
        let mut action = None;
        ui::card().inner_margin(Margin::same(18)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            if n == 0 {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.label(RichText::new(ui::I_GRID).size(36.0).color(ui::MUTED));
                    ui.label(ui::muted("No cards yet: review the polygons in step 3, then Extract Cards."));
                    ui.add_space(40.0);
                });
                return;
            }
            action = card_grid(ui, p, &self.post, &mut self.thumbs);
        });
        if let Some(a) = action {
            self.post_action(ui.ctx(), a);
        }
    }

    fn post_action(&mut self, ctx: &egui::Context, a: PostAction) {
        let Some(p) = &self.project else { return };
        let res = match &a {
            PostAction::Flip(name, v) => post::flip(p, name, *v).map(|()| {
                if let Some(c) = self.post.cards.iter_mut().find(|c| &c.name == name) {
                    c.modified = true;
                }
                self.thumbs.forget(&post::card_path(p, name));
            }),
            PostAction::Type(name) => {
                let new = if self.post.card_type(name) == "FRAG" { "ENT" } else { "FRAG" };
                post::set_type(p, name, new).map(|()| {
                    self.post.types.insert(name.clone(), new.into());
                })
            }
            PostAction::Exclude(name) => {
                if !self.post.excluded.remove(name) {
                    self.post.excluded.insert(name.clone());
                }
                post::set_excluded(p, &self.post.excluded)
            }
            PostAction::Open(name) => {
                let path = post::card_path(p, name);
                self.open_lightbox(ctx, name, &path);
                Ok(())
            }
        };
        if let Err(e) = res {
            self.status = format!("Error: {e:#}");
        }
    }

    fn info_modal(&mut self, ctx: &egui::Context) {
        if !self.info {
            return;
        }
        let version = include_str!("../VERSION").trim();
        let citation = format!(
            "This research utilized PyPotteryLens (version {version}) for automated computer-vision extraction of ceramic figures from monograph publications. Developed by Lorenzo Cardarelli (https://github.com/lrncrd/PyPotteryLens)."
        );
        let mut close = false;
        let r = egui::Modal::new(egui::Id::new("info")).frame(ui::card().inner_margin(Margin::same(26))).show(ctx, |ui| {
            ui.set_width(520.0);
            ui.horizontal(|ui| {
                ui.add(egui::Image::new(&self.assets.logo).fit_to_exact_size(vec2(56.0, 56.0)));
                ui.vertical(|ui| {
                    ui.label(RichText::new("PyPotteryLens").font(FontId::new(22.0, ui::xbold())).color(ui::TEXT));
                    ui.label(RichText::new(format!("Version {version} • By Lorenzo Cardarelli")).color(ui::DIM));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    close = ui::button(ui, Kind::Secondary, ui::I_X, "", vec2(30.0, 30.0), true).clicked();
                });
            });
            ui.label(RichText::new("Computer-vision extraction and classification of ceramic figures directly from published monograph PDFs.").color(ui::TEXT));
            ui.horizontal(|ui| {
                let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
                ui::chip(ui, &format!("{}  {cores} Cores ({} {})", ui::I_CPU, std::env::consts::OS, std::env::consts::ARCH), ui::DIM, ui::BG_PAGE, false);
                let accel = if cfg!(target_os = "macos") { "Apple Silicon (CoreML)" } else { "CPU (ONNX Runtime)" };
                ui::chip(ui, accel, ui::TEAL, ui::TEAL_SOFT, false);
            });
            ui.add_space(4.0);
            ui::sand_card().inner_margin(14).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Citation").font(FontId::new(14.0, ui::semibold())).color(ui::TEXT));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui::button(ui, Kind::Secondary, ui::I_CHECK2, "Copy", vec2(0.0, 28.0), true).clicked() {
                            ui.ctx().copy_text(citation.clone());
                            self.status = "Citation copied".into();
                        }
                    });
                });
                ui.label(RichText::new(format!("“{citation}”")).size(13.0).italics().color(ui::DIM));
            });
            ui.horizontal(|ui| {
                for (label, url) in [
                    ("Documentation", "https://lrncrd.github.io/PyPottery/pypotterylens/usage.html"),
                    ("GitHub", "https://github.com/lrncrd/PyPotteryLens"),
                    ("Support", "https://ko-fi.com/lrncrd"),
                ] {
                    if ui::button(ui, Kind::Secondary, "", label, vec2(0.0, 32.0), true).on_hover_text(url).clicked() {
                        ui.ctx().open_url(egui::OpenUrl::new_tab(url));
                    }
                }
            });
        });
        if close || r.should_close() {
            self.info = false;
        }
    }

    fn lightbox_modal(&mut self, ctx: &egui::Context) {
        let Some((name, tex)) = &self.lightbox else { return };
        let screen = ctx.content_rect().size() * 0.8;
        let s = tex.size_vec2();
        let k = (screen.x / s.x).min(screen.y / s.y).min(2.0);
        let r = egui::Modal::new(egui::Id::new("lightbox")).frame(ui::card().inner_margin(Margin::same(14))).show(ctx, |ui| {
            let close = ui
                .horizontal(|ui| {
                    ui.label(RichText::new(name.as_str()).font(FontId::monospace(13.0)).color(ui::DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui::button(ui, Kind::Secondary, ui::I_X, "", vec2(30.0, 30.0), true).clicked()).inner
                })
                .inner;
            ui.add(egui::Image::new(tex).fit_to_exact_size(s * k));
            close
        });
        if r.inner || r.should_close() {
            self.lightbox = None;
        }
    }

    fn export_modal(&mut self, ctx: &egui::Context) {
        let Some(acronym) = &mut self.export_acronym else { return };
        let total = self.post.cards.len();
        let excluded = self.post.excluded.len();
        let (mut close, mut go) = (false, false);
        let r = egui::Modal::new(egui::Id::new("export")).frame(ui::card().inner_margin(Margin::same(26))).show(ctx, |ui| {
            ui.set_width(440.0);
            ui.label(RichText::new(format!("{}  Export Results", ui::I_BOX_OUT)).font(FontId::new(19.0, ui::semibold())).color(ui::TEXT));
            ui.label(RichText::new("Cards are renamed ACRONYM_1, ACRONYM_2 … and packed into a ZIP together with the metadata CSV.").color(ui::DIM));
            ui.add_space(4.0);
            ui.label(RichText::new("Acronym").color(ui::TEXT));
            let edit = ui.add(egui::TextEdit::singleline(acronym).hint_text("e.g. OSA").desired_width(f32::INFINITY).margin(Margin::symmetric(12, 10)));
            edit.request_focus();
            let valid = post::valid_acronym(acronym);
            if !acronym.is_empty() && !valid {
                ui.label(RichText::new("Letters, numbers and underscores only").size(13.0).color(ui::DANGER));
            }
            ui.horizontal(|ui| {
                ui::chip(ui, &format!("{} cards", total - excluded), ui::TEAL, ui::TEAL_SOFT, false);
                if excluded > 0 {
                    ui::chip(ui, &format!("{excluded} excluded"), ui::DANGER, Color32::from_rgb(0xff, 0xf1, 0xf2), false);
                }
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                close = ui::button(ui, Kind::Secondary, "", "Cancel", vec2(120.0, 40.0), true).clicked();
                let enter = valid && ui.input(|i| i.key_pressed(egui::Key::Enter));
                go = ui::button(ui, Kind::Primary, ui::I_BOX_OUT, "Export ZIP…", vec2(180.0, 40.0), valid).clicked() || enter;
            });
        });
        if go {
            let acr = acronym.clone();
            if let Some(path) = rfd::FileDialog::new().set_file_name(format!("{acr}.zip")).add_filter("ZIP", &["zip"]).save_file() {
                self.export_acronym = None;
                self.spawn(move |p, prog| post::export(p, &acr, &path, prog));
            }
        } else if close || r.should_close() {
            self.export_acronym = None;
        }
    }
}

enum PostAction {
    Flip(String, bool),
    Type(String),
    Exclude(String),
    Open(String),
}

/// Cards at their real relative size (130–300 px wide), border = ENT (teal) / FRAG (terracotta),
/// controls on hover, like `.pp-card` in the web app. Only visible rows are drawn.
fn card_grid(ui: &mut egui::Ui, p: &Project, state: &post::State, thumbs: &mut thumbs::Thumbs) -> Option<PostAction> {
    const GAP: f32 = 12.0;
    let max_dim = state.cards.iter().map(|c| c.size[0].max(c.size[1])).max().unwrap_or(1).max(1) as f32;
    let scale = 300.0 / max_dim;
    let width = ui.available_width();
    // pack into rows: (first, last, y, height)
    let mut rows: Vec<(usize, usize, f32, f32)> = vec![];
    let mut sizes = Vec::with_capacity(state.cards.len());
    let (mut x, mut y, mut row_h, mut start) = (0.0, 0.0, 0.0f32, 0);
    for (i, c) in state.cards.iter().enumerate() {
        let w = if c.size[0] > 0 { (c.size[0] as f32 * scale).round().clamp(130.0, 300.0) } else { 130.0 };
        let h = if c.size[0] > 0 { (w * c.size[1] as f32 / c.size[0] as f32).clamp(120.0, 320.0) } else { 130.0 };
        if x > 0.0 && x + w > width {
            rows.push((start, i, y, row_h));
            y += row_h + GAP;
            (x, row_h, start) = (0.0, 0.0, i);
        }
        sizes.push((x, w, h));
        x += w + GAP;
        row_h = row_h.max(h);
    }
    rows.push((start, state.cards.len(), y, row_h));
    let total_h = y + row_h;

    let mut action = None;
    egui::ScrollArea::vertical().auto_shrink(false).show_viewport(ui, |ui, view| {
        ui.set_height(total_h);
        let origin = ui.max_rect().min;
        let painter = ui.painter();
        for &(a, b, ry, rh) in rows.iter().filter(|r| r.2 + r.3 >= view.min.y && r.2 <= view.max.y) {
            for i in a..b {
                let c = &state.cards[i];
                let (cx, w, h) = sizes[i];
                let rect = Rect::from_min_size(origin + vec2(cx, ry), vec2(w, h));
                let _ = rh;
                let id = egui::Id::new(("pp", &c.name));
                let tile = ui.interact(rect, id, Sense::click());
                let frag = state.card_type(&c.name) == "FRAG";
                let excluded = state.excluded.contains(&c.name);
                let hovered = ui.rect_contains_pointer(rect);
                let border = if frag { ui::PRIMARY } else { ui::TEAL };
                let fade = |col: Color32| if excluded { col.gamma_multiply(0.4) } else { col };
                painter.rect(rect, 6.0, Color32::WHITE, Stroke::new(3.0, fade(border)), StrokeKind::Inside);
                if let Some(tex) = thumbs.get(&post::card_path(p, &c.name)) {
                    let area = rect.shrink(6.0);
                    let s = tex.size_vec2();
                    let k = (area.width() / s.x).min(area.height() / s.y);
                    painter.image(tex.id(), Rect::from_center_size(area.center(), s * k), Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), fade(Color32::WHITE));
                }
                if excluded {
                    let badge = Rect::from_min_size(rect.min + vec2(6.0, 6.0), vec2(66.0, 18.0));
                    painter.rect_filled(badge, 4.0, ui::DANGER);
                    painter.text(badge.center(), egui::Align2::CENTER_CENTER, "EXCLUDED", FontId::new(10.0, ui::semibold()), Color32::WHITE);
                }
                if !hovered {
                    if tile.clicked() {
                        action = Some(PostAction::Open(c.name.clone()));
                    }
                    continue;
                }
                // overlay: dark gradient + flip / exclude buttons + ENT|FRAG pill
                let ov = Rect::from_min_max(egui::pos2(rect.left() + 3.0, rect.bottom() - 74.0), rect.max - vec2(3.0, 3.0));
                let mut mesh = egui::Mesh::default();
                let (top, bottom) = (Color32::from_rgba_unmultiplied(15, 23, 42, 0), Color32::from_rgba_unmultiplied(15, 23, 42, 235));
                mesh.colored_vertex(ov.left_top(), top);
                mesh.colored_vertex(ov.right_top(), top);
                mesh.colored_vertex(ov.right_bottom(), bottom);
                mesh.colored_vertex(ov.left_bottom(), bottom);
                mesh.add_triangle(0, 1, 2);
                mesh.add_triangle(0, 2, 3);
                painter.add(egui::Shape::mesh(mesh));
                let btn_y = rect.bottom() - 58.0;
                let buttons = [(ui::I_ARROW_UD, "Flip vertical"), (ui::I_ARROW_LR, "Flip horizontal"), (ui::I_X, "Exclude / include in export")];
                for (k, (icon, tip)) in buttons.iter().enumerate() {
                    let r = Rect::from_min_size(egui::pos2(rect.center().x - 47.0 + k as f32 * 33.0, btn_y), vec2(28.0, 26.0));
                    let b = ui.interact(r, id.with(k), Sense::click()).on_hover_text(*tip);
                    let fill = if b.hovered() { if k == 2 { ui::DANGER } else { Color32::WHITE } } else { Color32::from_white_alpha(235) };
                    painter.rect_filled(r, 5.0, fill);
                    painter.text(r.center(), egui::Align2::CENTER_CENTER, *icon, FontId::proportional(13.0), if b.hovered() && k == 2 { Color32::WHITE } else { ui::TEXT });
                    if b.clicked() {
                        action = Some(match k {
                            0 => PostAction::Flip(c.name.clone(), true),
                            1 => PostAction::Flip(c.name.clone(), false),
                            _ => PostAction::Exclude(c.name.clone()),
                        });
                    }
                }
                let pill = Rect::from_center_size(egui::pos2(rect.center().x, rect.bottom() - 17.0), vec2(92.0, 20.0));
                let pr = ui.interact(pill, id.with("type"), Sense::click()).on_hover_text("Toggle ENT / FRAG");
                let halves = [(Rect::from_min_max(pill.min, egui::pos2(pill.center().x, pill.max.y)), "ENT", !frag, ui::TEAL), (Rect::from_min_max(egui::pos2(pill.center().x, pill.min.y), pill.max), "FRAG", frag, ui::PRIMARY)];
                for (j, (r, label, on, color)) in halves.into_iter().enumerate() {
                    let radius = if j == 0 { egui::CornerRadius { nw: 10, sw: 10, ne: 0, se: 0 } } else { egui::CornerRadius { nw: 0, sw: 0, ne: 10, se: 10 } };
                    painter.rect_filled(r, radius, if on { color } else { Color32::from_white_alpha(218) });
                    painter.text(r.center(), egui::Align2::CENTER_CENTER, label, FontId::new(10.5, ui::semibold()), if on { Color32::WHITE } else { Color32::from_rgb(0x64, 0x74, 0x8b) });
                }
                if pr.clicked() {
                    action = Some(PostAction::Type(c.name.clone()));
                } else if tile.clicked() && action.is_none() {
                    action = Some(PostAction::Open(c.name.clone()));
                }
            }
        }
    });
    action
}

/// Project card of the repository grid. Some(true) = Select, Some(false) = Delete.
fn project_card(ui: &mut egui::Ui, p: &Project, icon: &TextureHandle, active: bool, width: f32) -> Option<bool> {
    let mut action = None;
    let stroke = if active { Stroke::new(2.0, ui::PRIMARY) } else { Stroke::new(1.0, ui::BORDER) };
    ui::card().stroke(stroke).show(ui, |ui| ui.vertical(|ui| {
        let inner = width - 44.0; // 2 × 20 margin + stroke
        ui.set_width(inner);
        ui.spacing_mut().item_spacing.x = 10.0; // the grid row sets 20 between cards
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::hover());
            ui.painter().rect_filled(r, 10.0, ui::BG_SURFACE);
            egui::Image::new(icon).paint_at(ui, Rect::from_center_size(r.center(), vec2(30.0, 30.0)));
            ui.label(RichText::new(&p.meta.project_name).font(FontId::new(18.0, ui::semibold())).color(ui::TEXT));
            if active {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui::chip(ui, "ACTIVE", ui::PRIMARY, ui::PEACH, false);
                });
            }
        });
        ui.separator();
        let n = |k: &str| p.status(k).and_then(|v| v.as_u64()).unwrap_or(0);
        let pdf = p.status("pdf_processed").and_then(|v| v.as_bool()).unwrap_or(false);
        let w = (inner - 10.0) / 2.0;
        let stat = |ui: &mut egui::Ui, icon: &str, label: &str, value: Option<u64>, ok: bool| {
            let (rect, _) = ui.allocate_exact_size(vec2(w, 40.0), Sense::hover());
            let painter = ui.painter();
            painter.rect(rect, 8.0, ui::BG_PAGE, Stroke::new(1.0, ui::BORDER), StrokeKind::Inside);
            painter.text(rect.left_center() + vec2(12.0, 0.0), egui::Align2::LEFT_CENTER, format!("{icon}  {label}"), FontId::proportional(13.0), ui::DIM);
            let (text, color) = match value {
                Some(v) => (v.to_string(), ui::PRIMARY),
                None if ok => (ui::I_CHECK_FILL.to_string(), ui::TEAL),
                None => (ui::I_CIRCLE.to_string(), ui::MUTED),
            };
            painter.text(rect.right_center() - vec2(12.0, 0.0), egui::Align2::RIGHT_CENTER, text, FontId::new(14.0, ui::semibold()), color);
        };
        ui.horizontal(|ui| {
            stat(ui, ui::I_PDF, "PDF:", None, pdf);
            stat(ui, ui::I_IMAGES, "Images:", Some(n("images_extracted")), true);
        });
        ui.horizontal(|ui| {
            stat(ui, ui::I_BBOX, "Masks:", Some(n("masks_extracted")), true);
            stat(ui, ui::I_CHECK2, "Annotations:", Some(p.meta.workflow_status.reviewed_images.len() as u64), true);
        });
        let date = |s: &str| s.get(..10).map(|d| format!("{}/{}/{}", &d[8..10], &d[5..7], &d[..4])).unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label(ui::muted(format!("Created: {}", date(p.created()))));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(ui::muted(format!("Modified: {}", date(&p.meta.last_modified))));
            });
        });
        ui.horizontal(|ui| {
            if ui::button(ui, Kind::Primary, ui::I_FOLDER, if active { "Selected" } else { "Select" }, vec2(w, 36.0), true).clicked() {
                action = Some(true);
            }
            if ui::button(ui, Kind::Danger, ui::I_TRASH, "Delete", vec2(w, 36.0), true).clicked() {
                action = Some(false);
            }
        });
    }));
    action
}

/// "PDF Page Structure Guide" popover: single page vs scanned spread.
fn spread_guide(ui: &mut egui::Ui) {
    ui.set_width(440.0);
    ui.label(RichText::new("PDF Page Structure Guide").font(FontId::new(14.5, ui::semibold())).color(ui::TEXT));
    ui.columns(2, |cols| {
        for (k, ui) in cols.iter_mut().enumerate() {
            let (rect, _) = ui.allocate_exact_size(vec2(200.0, 120.0), Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(rect, 8.0, ui::BG_PAGE);
            // dummy sheet(s): header bar, figure box, text lines
            let sheets = if k == 0 { 1 } else { 2 };
            let sw = 64.0;
            let x0 = rect.center().x - sw * sheets as f32 / 2.0;
            for i in 0..sheets {
                let r = Rect::from_min_size(pos2(x0 + i as f32 * sw, rect.top() + 12.0), vec2(sw - 2.0, 96.0));
                painter.rect(r, 2.0, Color32::WHITE, Stroke::new(1.0, ui::BORDER), StrokeKind::Inside);
                painter.rect_filled(Rect::from_min_size(r.min + vec2(8.0, 8.0), vec2(r.width() - 16.0, 5.0)), 1.0, ui::BORDER);
                let fig = Rect::from_min_size(r.min + vec2(10.0, 20.0), vec2(r.width() - 20.0, 36.0));
                painter.rect_filled(fig, 3.0, ui::PEACH);
                painter.text(fig.center(), egui::Align2::CENTER_CENTER, ui::I_IMAGE, FontId::proportional(14.0), ui::PRIMARY);
                for l in 0..4 {
                    painter.hline(r.left() + 8.0..=r.right() - 8.0, r.top() + 66.0 + l as f32 * 7.0, Stroke::new(2.0, ui::BORDER));
                }
            }
            if sheets == 2 {
                let x = rect.center().x - 1.0;
                painter.extend(Shape::dashed_line(&[pos2(x, rect.top() + 6.0), pos2(x, rect.bottom() - 6.0)], Stroke::new(1.5, ui::PRIMARY), 4.0, 3.0));
            }
            let (tag, desc) = if k == 0 {
                ("Single Page", "1 sheet = 1 page. Standard digital monograph PDF or single-page crop. Leave toggle OFF.")
            } else {
                ("Scanned Spread", "1 sheet = 2 facing pages. Open book on scanner bed. PyPottery bisects the spine into 2 individual plates. Turn toggle ON.")
            };
            ui.label(RichText::new(tag).font(FontId::new(13.0, ui::semibold())).color(if k == 0 { ui::TEAL } else { ui::PRIMARY }));
            ui.label(RichText::new(desc).size(12.5).color(ui::DIM));
        }
    });
}

/// Radio-style option card (`.execution-mode-option`).
fn option_card(ui: &mut egui::Ui, selected: bool, icon: &str, title: &str, desc: &str) -> bool {
    let (fill, stroke) = if selected { (ui::PEACH, ui::PRIMARY) } else { (Color32::WHITE, ui::BORDER) };
    let resp = Frame::new().fill(fill).stroke(Stroke::new(1.0, stroke)).corner_radius(10).inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(vec2(36.0, 36.0), Sense::hover());
            ui.painter().rect_filled(r, 8.0, if selected { ui::PRIMARY } else { ui::BG_SURFACE });
            ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, icon, FontId::proportional(17.0), if selected { Color32::WHITE } else { ui::MUTED });
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.label(RichText::new(title).font(FontId::new(15.0, ui::semibold())).color(if selected { ui::PRIMARY_HOVER } else { ui::TEXT }));
                ui.label(RichText::new(desc).size(13.0).color(ui::MUTED));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (icon, color) = if selected { (ui::I_CHECK_FILL, ui::PRIMARY) } else { (ui::I_CIRCLE, Color32::from_rgb(0xc8, 0xc2, 0xba)) };
                ui.label(RichText::new(icon).size(18.0).color(color));
            });
        });
    });
    resp.response.interact(Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked()
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job(&ctx);
        self.thumbs.begin_frame(&ctx);
        ui::paint_grid(ui.painter(), ui.max_rect());

        // drop a PDF anywhere on the window
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if let Some(path) = dropped.into_iter().find(|p| project::has_ext(&p.to_string_lossy(), &["pdf"]))
            && self.job.is_none()
            && self.project.as_ref().is_some_and(|p| !pdf::has_pdf(p))
        {
            self.tab = Tab::Pdf;
            self.start_pdf(path);
        }

        egui::Panel::top("header").frame(Frame::new().fill(Color32::WHITE).inner_margin(Margin::symmetric(52, 16))).show(ui, |ui| self.header(ui));
        if self.job.is_some() || !self.status.is_empty() {
            egui::Panel::bottom("status").frame(Frame::new().fill(Color32::WHITE).inner_margin(Margin::symmetric(52, 8))).show(ui, |ui| self.status_bar(ui));
        }

        let busy = self.job.is_some();
        egui::CentralPanel::default().frame(Frame::NONE.inner_margin(Margin { left: 52, right: 52, top: 26, bottom: 24 })).show(ui, |ui| {
            let active = TABS.iter().position(|t| t.0 == self.tab).unwrap_or(0);
            let labels: Vec<(&str, &str)> = TABS.iter().map(|t| (t.1, t.2)).collect();
            let has_project = self.project.is_some();
            if let Some(i) = ui::tab_strip(ui, &labels, active, &|i| i == 0 || has_project) {
                self.tab = TABS[i].0;
            }
            ui.add_space(-ui.spacing().item_spacing.y - 1.0);
            Frame::new()
                .fill(Color32::WHITE)
                .stroke(Stroke::new(1.0, ui::BORDER))
                .corner_radius(CornerRadius { nw: 0, ne: 0, sw: 16, se: 16 })
                .inner_margin(Margin::same(32))
                .show(ui, |ui| {
                    ui.set_min_size(ui.available_size());
                    ui.add_enabled_ui(!busy, |ui| match self.tab {
                        Tab::Projects => self.projects_ui(ui),
                        Tab::Pdf => self.pdf_ui(ui),
                        Tab::Detect => self.detect_ui(ui),
                        Tab::Post => self.post_ui(ui),
                        Tab::Tabular => {
                            if let Some(p) = &mut self.project
                                && let Some(pages) = self.tabular.ui(ui, p)
                            {
                                let s = self.tabular.ai.clone();
                                self.spawn(move |p, prog| ai::extract(p, pages, s, prog));
                            }
                        }
                        Tab::Editor => {
                            if let Some(p) = &mut self.project
                                && self.editor.ui(ui, p, &mut self.clean)
                            {
                                let (clean, exp) = (self.clean, self.editor.expansion());
                                self.spawn(move |p, prog| cards::extract(p, Some(exp), clean, prog));
                            }
                        }
                    });
                });
        });
        self.delete_modal(&ctx);
        self.lightbox_modal(&ctx);
        self.info_modal(&ctx);
        self.export_modal(&ctx);
    }
}

fn main() -> eframe::Result {
    let root = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| "projects".into());
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 920.0]).with_min_inner_size([1100.0, 700.0]).with_title("PyPotteryLens"),
        ..Default::default()
    };
    eframe::run_native("PyPotteryLens", options, Box::new(|cc| Ok(Box::new(App::new(cc, root)))))
}

/// Off-screen screenshots for checking the look without opening a window:
///     SHOT_TAB=0 SHOT_PROJECT=projects/Osa_… SHOT_OUT=/tmp/x.png cargo test --release shot -- --ignored
#[cfg(test)]
mod shot {
    use super::*;

    #[test]
    #[ignore]
    fn shot() {
        let env = |k: &str| std::env::var(k).ok();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1440.0, 920.0)).wgpu().build_eframe(|cc| App::blank(cc, "projects".into()));
        if let Some(dir) = env("SHOT_PROJECT") {
            let p = Project::open(std::path::Path::new(&dir)).unwrap();
            h.state_mut().open(p);
        }
        if let Some(i) = env("SHOT_IMAGE").and_then(|i| i.parse().ok()) {
            h.state_mut().editor.go(i);
        }
        h.state_mut().info = env("SHOT_INFO").is_some();
        h.state_mut().tab = TABS[env("SHOT_TAB").and_then(|t| t.parse().ok()).unwrap_or(0)].0;
        for _ in 0..20 {
            h.step();
            std::thread::sleep(Duration::from_millis(50)); // let thumbnails decode
        }
        h.render().unwrap().save(env("SHOT_OUT").unwrap_or("shot.png".into())).unwrap();
    }
}
