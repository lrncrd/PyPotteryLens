//! Look & feel of the web app (static/css/style.css): tokens, fonts, and the few widgets it uses.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Frame, Margin, Pos2, Rect, Response, RichText, Sense, Shadow, Stroke,
    StrokeKind, TextStyle, Ui, Vec2, pos2, vec2,
};
use std::sync::Arc;

pub const PRIMARY: Color32 = Color32::from_rgb(0xc2, 0x41, 0x0c);
pub const PRIMARY_HOVER: Color32 = Color32::from_rgb(0x9a, 0x34, 0x12);
pub const PRIMARY_SOFT: Color32 = Color32::from_rgb(0xfb, 0xf0, 0xea); // 8% terracotta on white
pub const PRIMARY_BORDER: Color32 = Color32::from_rgb(0xf0, 0xcf, 0xc0); // 20%
pub const PEACH: Color32 = Color32::from_rgb(0xff, 0xf7, 0xed);
pub const TEAL: Color32 = Color32::from_rgb(0x0d, 0x94, 0x88);
pub const TEAL_SOFT: Color32 = Color32::from_rgb(0xe9, 0xf6, 0xf5);
pub const BG_PAGE: Color32 = Color32::from_rgb(0xfb, 0xf9, 0xf5);
pub const BG_SURFACE: Color32 = Color32::from_rgb(0xf5, 0xef, 0xe6);
pub const CANVAS_BG: Color32 = Color32::from_rgb(0xf3, 0xed, 0xe4);
pub const BORDER: Color32 = Color32::from_rgb(0xe8, 0xe3, 0xd8);
pub const TEXT: Color32 = Color32::from_rgb(0x1c, 0x19, 0x17);
pub const DIM: Color32 = Color32::from_rgb(0x57, 0x53, 0x4e);
pub const MUTED: Color32 = Color32::from_rgb(0x78, 0x71, 0x6c);
pub const DANGER: Color32 = Color32::from_rgb(0xe1, 0x1d, 0x48);
pub const DISABLED: Color32 = Color32::from_rgb(0xa8, 0xb5, 0xc4);

// Bootstrap Icons codepoints (static/vendor/bootstrap-icons)
pub const I_FOLDER: &str = "\u{f3d8}";
pub const I_FOLDER_PLUS: &str = "\u{f3d3}";
pub const I_ARCHIVE: &str = "\u{f10d}";
pub const I_PDF: &str = "\u{f63e}";
pub const I_CPU: &str = "\u{f2d6}";
pub const I_PENCIL: &str = "\u{f4ca}";
pub const I_GRID: &str = "\u{f3f9}";
pub const I_REFRESH: &str = "\u{f116}";
pub const I_CHECK_FILL: &str = "\u{f26a}";
pub const I_CIRCLE: &str = "\u{f28a}";
pub const I_IMAGES: &str = "\u{f42b}";
pub const I_IMAGE: &str = "\u{f42a}";
pub const I_BBOX: &str = "\u{f1b6}";
pub const I_CHECK2: &str = "\u{f270}";
pub const I_TRASH: &str = "\u{f78b}";
pub const I_CLOUD_UP: &str = "\u{f297}";
pub const I_LOCK: &str = "\u{f47a}";
pub const I_SPLIT: &str = "\u{f460}";
pub const I_PLAY: &str = "\u{f4f2}";
pub const I_BOLT: &str = "\u{f46d}";
pub const I_COLLECTION: &str = "\u{f2cb}";
pub const I_SLIDERS: &str = "\u{f56b}";
pub const I_GEAR: &str = "\u{f3e5}";
pub const I_ZOOM_IN: &str = "\u{f62c}";
pub const I_ZOOM_OUT: &str = "\u{f62d}";
pub const I_EXPAND: &str = "\u{f14a}";
pub const I_LEFT: &str = "\u{f284}";
pub const I_RIGHT: &str = "\u{f285}";
pub const I_CURSOR: &str = "\u{f2e3}";
pub const I_PEN: &str = "\u{f604}";
pub const I_CROP: &str = "\u{f2dd}";
pub const I_EYE_SLASH: &str = "\u{f340}";
pub const I_CHECK_SQUARE: &str = "\u{f271}";
pub const I_DASH_SQUARE: &str = "\u{f2e9}";
pub const I_FILE_LOCK: &str = "\u{f36f}";
pub const I_UNDO: &str = "\u{f117}";
pub const I_X: &str = "\u{f659}";
pub const I_ARROW_UD: &str = "\u{f127}";
pub const I_ARROW_LR: &str = "\u{f12b}";
pub const I_BOX_OUT: &str = "\u{f1c5}";
pub const I_MAGIC: &str = "\u{f675}";
pub const I_TABLE: &str = "\u{f5aa}";
pub const I_PLUS: &str = "\u{f64d}";
pub const I_FILL: &str = "\u{f112}";
pub const I_ERASER: &str = "\u{f330}";
pub const I_STARS: &str = "\u{f589}";
pub const I_CLOUD: &str = "\u{f2c1}";
pub const I_CHAT: &str = "\u{f252}";
pub const I_EYE: &str = "\u{f341}";
pub const I_WARN: &str = "\u{f332}";
pub const I_CHART: &str = "\u{f17d}";
pub const I_BRUSH: &str = "\u{f1d8}";
pub const I_ERASER_LINE: &str = "\u{f331}";
pub const I_PALETTE: &str = "\u{f4b1}";
pub const I_RULERS: &str = "\u{f523}";
pub const I_SIDEBAR: &str = "\u{f45f}";
pub const I_SIDEBAR_INSET: &str = "\u{f45d}";
pub const I_GEO: &str = "\u{f3e8}";
pub const I_BEZIER: &str = "\u{f18c}";
pub const I_VESSELS: &str = "\u{f1b5}";
pub const I_CHEVRON_DOWN: &str = "\u{f282}";
pub const I_INFO: &str = "\u{f431}";
pub const I_HOURGLASS: &str = "\u{f41f}";

pub fn semibold() -> FontFamily {
    FontFamily::Name("semibold".into())
}
pub fn xbold() -> FontFamily {
    FontFamily::Name("xbold".into())
}

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let add = |fonts: &mut FontDefinitions, name: &str, bytes: &'static [u8]| {
        fonts.font_data.insert(name.into(), Arc::new(FontData::from_static(bytes)));
    };
    add(&mut fonts, "jakarta", include_bytes!("../assets/fonts/PlusJakartaSans-Regular.ttf"));
    add(&mut fonts, "jakarta-semibold", include_bytes!("../assets/fonts/PlusJakartaSans-SemiBold.ttf"));
    add(&mut fonts, "jakarta-xbold", include_bytes!("../assets/fonts/PlusJakartaSans-ExtraBold.ttf"));
    add(&mut fonts, "jetbrains", include_bytes!("../assets/fonts/JetBrainsMono-Medium.ttf"));
    add(&mut fonts, "bi", include_bytes!("../assets/fonts/bootstrap-icons.ttf"));
    // egui's own fonts stay as fallbacks for glyphs outside the latin subsets
    let fallback = fonts.families[&FontFamily::Proportional].clone();
    let family = |main: &str| [vec![main.to_string(), "bi".into()], fallback.clone()].concat();
    fonts.families.insert(FontFamily::Proportional, family("jakarta"));
    fonts.families.insert(semibold(), family("jakarta-semibold"));
    fonts.families.insert(xbold(), family("jakarta-xbold"));
    fonts.families.insert(FontFamily::Monospace, [vec!["jetbrains".to_string(), "bi".into()], fallback].concat());
    ctx.set_fonts(fonts);

    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(12.5)),
            (TextStyle::Body, FontId::proportional(14.5)),
            (TextStyle::Button, FontId::new(14.0, semibold())),
            (TextStyle::Heading, FontId::new(26.0, xbold())),
            (TextStyle::Monospace, FontId::monospace(13.0)),
        ]
        .into();
        let s = &mut style.spacing;
        s.item_spacing = vec2(10.0, 10.0);
        s.button_padding = vec2(14.0, 7.0);
        s.interact_size.y = 34.0;
        s.text_edit_width = 300.0;

        let v = &mut style.visuals;
        *v = egui::Visuals::light();
        v.override_text_color = None;
        v.panel_fill = BG_PAGE;
        v.window_fill = Color32::WHITE;
        v.window_stroke = Stroke::new(1.0, BORDER);
        v.window_corner_radius = CornerRadius::same(18);
        v.window_shadow = Shadow { offset: [0, 18], blur: 48, spread: 0, color: Color32::from_black_alpha(36) };
        v.extreme_bg_color = Color32::WHITE;
        v.faint_bg_color = BG_SURFACE;
        v.hyperlink_color = PRIMARY;
        v.selection.bg_fill = Color32::from_rgb(0xff, 0xed, 0xd5);
        v.selection.stroke = Stroke::new(1.0, PRIMARY_HOVER);
        let r = CornerRadius::same(8);
        let w = &mut v.widgets;
        w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        w.noninteractive.corner_radius = r;
        w.inactive.weak_bg_fill = BG_SURFACE;
        w.inactive.bg_fill = Color32::WHITE;
        w.inactive.bg_stroke = Stroke::new(1.0, BORDER);
        w.inactive.fg_stroke = Stroke::new(1.0, DIM);
        w.inactive.corner_radius = r;
        w.hovered.weak_bg_fill = PRIMARY_SOFT;
        w.hovered.bg_fill = Color32::WHITE;
        w.hovered.bg_stroke = Stroke::new(1.0, PRIMARY);
        w.hovered.fg_stroke = Stroke::new(1.0, PRIMARY);
        w.hovered.corner_radius = r;
        w.hovered.expansion = 0.0;
        w.active.weak_bg_fill = PRIMARY_BORDER;
        w.active.bg_fill = Color32::WHITE;
        w.active.bg_stroke = Stroke::new(1.0, PRIMARY);
        w.active.fg_stroke = Stroke::new(1.0, PRIMARY_HOVER);
        w.active.corner_radius = r;
        w.active.expansion = 0.0;
        w.open = w.active;
    });
}

pub fn load_png(ctx: &egui::Context, name: &str, bytes: &[u8]) -> egui::TextureHandle {
    let img = image::load_from_memory(bytes).expect("bundled image").to_rgba8();
    let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
    ctx.load_texture(name, color, egui::TextureOptions::LINEAR)
}

/// Archival graph-paper background of the web app body.
pub fn paint_grid(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect, 0.0, BG_PAGE);
    for (step, alpha) in [(14.0, 9), (70.0, 18)] {
        let stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(194, 65, 12, alpha));
        let mut x = rect.left();
        while x < rect.right() {
            painter.vline(x, rect.y_range(), stroke);
            x += step;
        }
        let mut y = rect.top();
        while y < rect.bottom() {
            painter.hline(rect.x_range(), y, stroke);
            y += step;
        }
    }
}

/// White rounded card (`.card` / `.panel` in the CSS).
pub fn card() -> Frame {
    Frame::new()
        .fill(Color32::WHITE)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(14)
        .inner_margin(Margin::same(20))
        .shadow(Shadow { offset: [0, 4], blur: 18, spread: 0, color: Color32::from_rgba_unmultiplied(120, 53, 15, 10) })
}

/// Sand-tinted card used for side forms and toolbars.
pub fn sand_card() -> Frame {
    Frame::new().fill(Color32::from_rgb(0xfa, 0xf6, 0xf0)).stroke(Stroke::new(1.0, BORDER)).corner_radius(14).inner_margin(Margin::same(20))
}

pub fn icon_text(icon: &str, text: &str) -> String {
    match (icon.is_empty(), text.is_empty()) {
        (true, _) => text.to_string(),
        (_, true) => icon.to_string(),
        _ => format!("{icon}  {text}"),
    }
}

/// Tab heading: icon + big title, subtitle, divider.
pub fn page_heading(ui: &mut Ui, icon: &str, title: &str, subtitle: &str) {
    ui.label(RichText::new(icon_text(icon, title)).font(FontId::new(26.0, xbold())).color(TEXT));
    ui.label(RichText::new(subtitle).size(15.0).color(DIM));
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(8.0);
}

pub fn section_title(ui: &mut Ui, icon: &str, title: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(icon).size(17.0).color(PRIMARY));
        ui.label(RichText::new(title).font(FontId::new(17.0, semibold())).color(TEXT));
    });
}

pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).size(12.5).color(MUTED)
}

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Primary,
    Secondary,
    Danger,
    Teal,
    TealOutline,
}

/// Painted button so hover states follow the CSS (egui's `Button::fill` has no hover variant).
pub fn button(ui: &mut Ui, kind: Kind, icon: &str, text: &str, size: Vec2, enabled: bool) -> Response {
    let font = FontId::new(if size.y >= 44.0 { 15.5 } else { 14.0 }, semibold());
    let galley = ui.painter().layout_no_wrap(icon_text(icon, text), font, Color32::PLACEHOLDER);
    let size = vec2(size.x.max(galley.size().x + 28.0), size.y);
    let (rect, resp) = ui.allocate_exact_size(size, if enabled { Sense::click() } else { Sense::hover() });
    let hov = enabled && resp.hovered();
    let (fill, stroke, fg) = match (kind, enabled) {
        (Kind::Primary, false) => (DISABLED, DISABLED, Color32::WHITE),
        (_, false) => (BG_PAGE, BORDER, MUTED.gamma_multiply(0.55)),
        (Kind::Primary, _) => (if hov { PRIMARY_HOVER } else { PRIMARY }, Color32::TRANSPARENT, Color32::WHITE),
        (Kind::Secondary, _) => (if hov { PRIMARY_SOFT } else { BG_SURFACE }, if hov { PRIMARY } else { BORDER }, if hov { PRIMARY } else { DIM }),
        (Kind::Danger, _) => (if hov { Color32::from_rgb(0xff, 0xf1, 0xf2) } else { Color32::WHITE }, if hov { DANGER } else { BORDER }, DANGER),
        (Kind::Teal, _) => (if hov { Color32::from_rgb(0x0f, 0x76, 0x6e) } else { TEAL }, Color32::TRANSPARENT, Color32::WHITE),
        (Kind::TealOutline, _) => (if hov { Color32::from_rgb(0xd5, 0xef, 0xec) } else { TEAL_SOFT }, TEAL.gamma_multiply(0.5), Color32::from_rgb(0x0f, 0x76, 0x6e)),
    };
    let p = ui.painter();
    p.rect(rect, 8.0, fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
    p.galley(rect.center() - galley.size() / 2.0, galley, fg);
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// Pill badge (`.version-badge`, figure counters, value readouts).
pub fn chip(ui: &mut Ui, text: &str, fg: Color32, fill: Color32, mono: bool) -> Response {
    let font = if mono { FontId::monospace(12.0) } else { FontId::new(12.0, semibold()) };
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, fg);
    let (rect, resp) = ui.allocate_exact_size(galley.size() + vec2(16.0, 8.0), Sense::hover());
    ui.painter().rect(rect, 99.0, fill, Stroke::new(1.0, fg.gamma_multiply(0.25)), StrokeKind::Inside);
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
    resp
}

/// iOS-style switch (`.toggle-switch`).
pub fn toggle(ui: &mut Ui, on: &mut bool, label: &str) -> Response {
    let (rect, mut resp) = ui.allocate_exact_size(vec2(40.0, 22.0), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool(resp.id, *on);
    let p = ui.painter();
    p.rect_filled(rect, 11.0, Color32::from_rgb(0xd6, 0xd3, 0xd1).lerp_to_gamma(PRIMARY, t));
    let x = egui::lerp(rect.left() + 11.0..=rect.right() - 11.0, t);
    p.circle(pos2(x, rect.center().y), 8.0, Color32::WHITE, Stroke::NONE);
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if label.is_empty() {
        return resp;
    }
    let l = ui.label(RichText::new(label).size(13.5).color(DIM)).interact(Sense::click());
    if l.clicked() {
        *on = !*on;
        let mut r = resp | l;
        r.mark_changed();
        return r;
    }
    resp | l
}

/// Range slider with a sand rail and terracotta knob (`input[type=range]` in the CSS).
pub fn slider(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, width: f32) -> Response {
    let (rect, mut resp) = ui.allocate_exact_size(vec2(width, 22.0), Sense::click_and_drag());
    let (lo, hi) = (*range.start(), *range.end());
    if let Some(p) = resp.interact_pointer_pos() {
        let t = ((p.x - rect.left() - 9.0) / (rect.width() - 18.0)).clamp(0.0, 1.0);
        let v = lo + t * (hi - lo);
        if v != *value {
            *value = v;
            resp.mark_changed();
        }
    }
    let t = (*value - lo) / (hi - lo);
    let x = rect.left() + 9.0 + t * (rect.width() - 18.0);
    let p = ui.painter();
    let rail = Rect::from_center_size(rect.center(), vec2(rect.width(), 6.0));
    p.rect_filled(rail, 3.0, Color32::from_rgb(0xe7, 0xe0, 0xd5));
    p.circle(pos2(x, rect.center().y), 9.0, PRIMARY, Stroke::new(2.0, Color32::WHITE));
    resp.on_hover_cursor(egui::CursorIcon::Grab)
}

/// Folder-style tab strip; returns the clicked index.
pub fn tab_strip(ui: &mut Ui, tabs: &[(&str, &str)], active: usize, enabled: &dyn Fn(usize) -> bool) -> Option<usize> {
    let gap = 6.0;
    let w = (ui.available_width() - gap * (tabs.len() - 1) as f32) / tabs.len() as f32;
    let mut clicked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for (i, (icon, label)) in tabs.iter().enumerate() {
            let on = enabled(i);
            let (rect, resp) = ui.allocate_exact_size(vec2(w, 44.0), if on { Sense::click() } else { Sense::hover() });
            let is_active = i == active;
            let hov = on && resp.hovered();
            let p = ui.painter();
            let radius = CornerRadius { nw: 10, ne: 10, sw: 0, se: 0 };
            let fill = if is_active { Color32::WHITE } else if hov { Color32::from_rgb(0xfa, 0xf4, 0xec) } else { BG_SURFACE.gamma_multiply(0.7) };
            p.rect(rect, radius, fill, Stroke::new(1.0, BORDER), StrokeKind::Inside);
            if is_active {
                p.rect_filled(Rect::from_min_size(rect.min, vec2(rect.width(), 3.0)), CornerRadius { nw: 10, ne: 10, sw: 0, se: 0 }, PRIMARY);
                // merge into the content card below
                p.hline(rect.x_range().shrink(1.0), rect.bottom(), Stroke::new(2.0, Color32::WHITE));
            }
            let color = if is_active { PRIMARY } else if !on { MUTED.gamma_multiply(0.6) } else if hov { TEXT } else { DIM };
            let font = FontId::new(14.0, if is_active { semibold() } else { FontFamily::Proportional });
            p.text(rect.center() + vec2(0.0, 1.0), Align2::CENTER_CENTER, icon_text(icon, label), font, color);
            if resp.clicked() {
                clicked = Some(i);
            }
            if on && !is_active {
                resp.on_hover_cursor(egui::CursorIcon::PointingHand);
            }
        }
    });
    clicked
}

/// Dashed rounded-ish rectangle (the PDF drop zone).
pub fn dashed_rect(painter: &egui::Painter, rect: Rect, color: Color32) {
    let c = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom(), rect.left_top()];
    let pts: Vec<Pos2> = c.to_vec();
    painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(2.0, color), 8.0, 6.0));
}

/// Progress bar painted in the primary color.
pub fn progress(ui: &mut Ui, frac: f32, width: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(width, 8.0), Sense::hover());
    ui.painter().rect_filled(rect, 4.0, BG_SURFACE);
    let mut done = rect;
    done.set_width(rect.width() * frac.clamp(0.0, 1.0));
    ui.painter().rect_filled(done, 4.0, PRIMARY);
}
