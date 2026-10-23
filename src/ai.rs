//! AI extraction of bibliographic references through OpenRouter: port of
//! ai_extract_bibliographic(_batch), _read_numbers_from_crops and call_openrouter_ai (app.py).
//! The local Gemma backend is not ported.

use crate::Progress;
use crate::project::Project;
use crate::tabular::Tabular;
use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result, anyhow, bail};
use image::{Rgb, RgbImage, imageops::FilterType};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use serde_json::{Map, Value, json};
use std::time::Duration;

pub const DEFAULT_MODEL: &str = "deepseek/deepseek-v4-flash";
const ORANGE: Rgb<u8> = Rgb([255, 80, 0]);

#[derive(Clone)]
pub struct Settings {
    pub api_key: String,
    pub model: String,
    pub prompt: String,
    pub numbers_from_crops: bool,
}

fn settings_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(std::path::PathBuf::from(home).join(".pypotterylens").join("lens_native.json"))
}

/// Model, prompt and crop option persist across sessions; the API key never touches disk
/// (it can be preloaded from OPENROUTER_API_KEY).
pub fn load_settings() -> Settings {
    let v: Value = settings_path().and_then(|f| std::fs::read_to_string(f).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    Settings {
        api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
        model: v["openrouter_model"].as_str().unwrap_or(DEFAULT_MODEL).to_string(),
        prompt: v["ai_prompt"].as_str().unwrap_or_default().to_string(),
        numbers_from_crops: v["numbers_from_crops"].as_bool().unwrap_or(false),
    }
}

pub fn save_settings(s: &Settings) {
    update_settings(|m| {
        m.insert("openrouter_model".into(), json!(s.model));
        m.insert("ai_prompt".into(), json!(s.prompt));
        m.insert("numbers_from_crops".into(), json!(s.numbers_from_crops));
    });
}

/// App preferences file (~/.pypotterylens/lens_native.json), shared with the main window.
pub fn read_setting(key: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(settings_path()?).ok()?).ok()?;
    v.get(key).cloned()
}

pub fn update_settings(f: impl FnOnce(&mut Map<String, Value>)) {
    let Some(path) = settings_path() else { return };
    let mut m = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Map<String, Value>>(&t).ok()).unwrap_or_default();
    f(&mut m);
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(path, Value::Object(m).to_string());
}

/// Field names the user wrote in [brackets] become exact column names.
pub fn canonical_fields(prompt: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut rest = prompt;
    while let Some(start) = rest.find('[') {
        let after = &rest[start + 1..];
        let Some(end) = after.find(|c| c == ']' || c == '[') else { break };
        if after.as_bytes()[end] == b']' {
            let f = after[..end].trim().to_string();
            if !f.is_empty() && !out.contains(&f) {
                out.push(f);
            }
        }
        rest = &after[end..];
    }
    out
}

fn normalize_key(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

fn extra_fields_instruction(canonical: &[String]) -> String {
    if canonical.is_empty() {
        return "If the context above asks you to extract additional fields beyond the four standard ones, include them in each drawing's object using a concise snake_case key (e.g. 'material', 'ceramic_class').\n".into();
    }
    let keys = canonical.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ");
    format!(
        "In addition to the four standard fields, also extract these fields and use these EXACT JSON keys, spelled verbatim — do NOT translate, rename, pluralise or change the case: {keys}.\nUse the same key on every drawing; if a value is not present, use null.\n"
    )
}

fn page_prompt(n: usize, letters: &str, user: &str, canonical: &[String]) -> String {
    let mut p = format!(
        "This is a page from an archaeological publication about pottery.\n\
         There are {n} pottery drawings on this page. Each drawing is VISUALLY MARKED with an ORANGE BOUNDING BOX. \
         The orange letter (A, B, C...) at the top of each box is a software annotation only — it is NOT a number from the publication.\n\n\
         Your JSON response MUST use EXACTLY these letter keys: {letters}\n\
         Do NOT use numbers as keys. Each key must be one of the letters listed above.\n\n\
         For EACH drawing, extract:\n\
         - \"page\": the page number printed on the publication page. Look at the edges and corners of the full image. Same for all drawings.\n\
         - \"plate\": plate/table identifier (e.g. \"Tav. III\", \"Pl. 12\"). Usually at the top or bottom edge. Shared by all drawings.\n\
         - \"figure\": figure identifier for the plate (e.g. \"Fig. 3\", \"Abb. 5\"). Usually at top or bottom of the image.\n\
         - \"number\": the small catalogue number PRINTED IN THE ORIGINAL PUBLICATION near or inside the drawing inside the orange box. \
         It is a digit or short alphanumeric (e.g. \"1\", \"3\", \"14\", \"2a\", \"7b\") that appears as part of the publication layout, NOT the orange letter label.\n\n"
    );
    if !user.is_empty() {
        p += &format!("Additional context from the user:\n{user}\n\n");
    }
    p += &format!("Respond ONLY with a valid JSON object using EXACTLY these letter keys: {letters}\n");
    p += &extra_fields_instruction(canonical);
    p += "Example for two drawings labelled A and B:\n\
          {\"A\": {\"page\": \"45\", \"plate\": \"Tav. III\", \"figure\": \"Fig. 5\", \"number\": \"1\"}, \
          \"B\": {\"page\": \"45\", \"plate\": \"Tav. III\", \"figure\": \"Fig. 5\", \"number\": \"2a\"}}\n\
          If a value is not found, use null.";
    p
}

const NUMBER_PROMPT: &str = "This is a cropped detail from an archaeological pottery catalogue. It shows one pottery drawing and the area around it. Find the small catalogue/inventory NUMBER printed next to or under the drawing — it is a digit or short alphanumeric such as '1', '14', '2a', '7b'. Reply with ONLY that number and nothing else. If there is no number, reply 'null'.";

/// Tolerant JSON extraction from an LLM reply (`_parse_ai_json`): code fences, smart quotes,
/// trailing commas, and replies truncated mid-object (keeps the complete drawings).
pub fn parse_ai_json(raw: &str) -> Result<Map<String, Value>> {
    let mut text = raw.trim().to_string();
    if text.is_empty() {
        bail!("empty response");
    }
    if text.starts_with("```") {
        text = text.splitn(2, '\n').nth(1).unwrap_or("").trim_end().trim_end_matches("```").trim().to_string();
    }
    let text = text.replace(['“', '”'], "\"").replace('’', "'");
    let start = text.find('{').ok_or_else(|| anyhow!("no JSON object found in response"))?;
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    let (mut end, mut last_inner) = (None, None);
    for (i, ch) in text[start..].char_indices().map(|(i, c)| (i + start, c)) {
        if in_str {
            match (esc, ch) {
                (true, _) => esc = false,
                (_, '\\') => esc = true,
                (_, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 1 {
                    last_inner = Some(i);
                } else if depth == 0 {
                    end = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let candidate = &text[start..end.unwrap_or(text.len())];
    let strip_commas = |s: &str| {
        let mut out = String::with_capacity(s.len());
        let chars: Vec<char> = s.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            if c == ',' && chars[i + 1..].iter().find(|c| !c.is_whitespace()).is_some_and(|n| *n == '}' || *n == ']') {
                continue;
            }
            out.push(c);
        }
        out
    };
    for attempt in [candidate.to_string(), strip_commas(candidate)] {
        if let Ok(Value::Object(m)) = serde_json::from_str(&attempt) {
            return Ok(m);
        }
    }
    if let Some(li) = last_inner {
        let trimmed = format!("{}}}", text[start..=li].trim_end().trim_end_matches(','));
        if let Ok(Value::Object(m)) = serde_json::from_str(&strip_commas(&trimmed)) {
            return Ok(m);
        }
    }
    bail!("could not parse JSON from response")
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            out.push(if k <= c.len() { T[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// One vision request; retries once with a larger budget when the reply comes back empty
/// (reasoning models can spend all tokens thinking). Adds the usage to `tokens`.
fn openrouter(img: &RgbImage, prompt: &str, s: &Settings, max_tokens: u32, tokens: &mut u64) -> Result<String> {
    let mut png = vec![];
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    let data_url = format!("data:image/png;base64,{}", base64(&png));
    let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(300))).build().into();
    let call = |budget: u32| -> Result<Value> {
        let body = json!({
            "model": s.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": budget,
            "reasoning": {"enabled": false},
        });
        let mut resp = agent.post("https://openrouter.ai/api/v1/chat/completions").header("Authorization", format!("Bearer {}", s.api_key)).send_json(&body)?;
        let status = resp.status().as_u16();
        let text = resp.body_mut().with_config().limit(50 * 1024 * 1024).read_to_string()?;
        if status != 200 {
            if status == 404 || text.to_lowercase().contains("image input") || text.contains("No endpoints") {
                bail!("Model '{}' does not support image/vision input on OpenRouter: choose a vision-capable model in Backend.", s.model);
            }
            bail!("OpenRouter error {status}: {}", text.chars().take(300).collect::<String>());
        }
        Ok(serde_json::from_str(&text)?)
    };
    let content = |v: &Value| v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    let mut v = call(max_tokens)?;
    if content(&v).trim().is_empty() {
        v = call((max_tokens * 4).clamp(300, 8192))?;
    }
    let text = content(&v);
    if text.trim().is_empty() {
        bail!(
            "OpenRouter returned an empty response twice in a row for model '{}' (finish_reason={}). Try a non-reasoning vision model, or a page with fewer drawings.",
            s.model,
            v["choices"][0]["finish_reason"]
        );
    }
    *tokens += v["usage"]["total_tokens"].as_u64().unwrap_or(0);
    Ok(text)
}

/// Orange boxes with letter tags, as `_annotate_image_with_bboxes` draws them for the model.
fn annotate(img: &mut RgbImage, boxes: &[(String, [i64; 4])]) {
    let font = FontRef::try_from_slice(include_bytes!("../assets/fonts/PlusJakartaSans-ExtraBold.ttf")).expect("bundled font");
    let short = img.width().min(img.height()) as i64;
    let line = (short / 350).max(2);
    let size = (short / 55).max(14) as f32;
    let pad = line.max(3);
    for (label, [x1, y1, x2, y2]) in boxes {
        for k in 0..line {
            let (w, h) = ((x2 - x1 - 2 * k).max(1) as u32, (y2 - y1 - 2 * k).max(1) as u32);
            draw_hollow_rect_mut(img, Rect::at((x1 + k) as i32, (y1 + k) as i32).of_size(w, h), ORANGE);
        }
        let (tw, th) = text_size(PxScale::from(size), &font, label);
        let (tw, th) = (tw as i64, th as i64);
        let ly = if y1 - th - pad * 2 < 0 { y1 + pad } else { y1 - th - pad * 2 };
        draw_filled_rect_mut(img, Rect::at(*x1 as i32, ly as i32).of_size((tw + pad * 2) as u32, (th + pad * 2) as u32), ORANGE);
        draw_text_mut(img, Rgb([255, 255, 255]), (x1 + pad) as i32, (ly + pad) as i32, PxScale::from(size), &font, label);
    }
}

fn shrink(img: &RgbImage, max: u32) -> RgbImage {
    if img.width().max(img.height()) <= max {
        return img.clone();
    }
    let k = max as f64 / img.width().max(img.height()) as f64;
    image::imageops::resize(img, (img.width() as f64 * k).round() as u32, (img.height() as f64 * k).round() as u32, FilterType::Lanczos3)
}

fn page_image(p: &Project, page: &str) -> Result<RgbImage> {
    let path = crate::tabular::page_file(p, page).ok_or_else(|| anyhow!("Image not found: {page}"))?;
    Ok(image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8())
}

/// Fill page/plate/figure/number (+ bracketed extra fields) for every drawing of one page.
fn extract_page(t: &mut Tabular, p: &Project, page: &str, s: &Settings, tokens: &mut u64) -> Result<()> {
    let full = page_image(p, page)?;
    let mut img = shrink(&full, 2400);
    let k = img.width() as f64 / full.width() as f64;
    let annots: Vec<(String, [i64; 4])> = t.page_annots(page).map(|a| (a.id.clone(), a.bbox)).collect();
    if annots.is_empty() {
        bail!("No annotations for: {page}");
    }
    // letters instead of IDs, so the model can't mistake our label for the publication's number
    let letter = |i: usize| if i < 26 { ((b'A' + i as u8) as char).to_string() } else { format!("Z{i}") };
    let boxes: Vec<(String, [i64; 4])> = annots.iter().enumerate().map(|(i, (_, b))| (letter(i), b.map(|v| (v as f64 * k) as i64))).collect();
    annotate(&mut img, &boxes);
    let letters = boxes.iter().map(|b| b.0.clone()).collect::<Vec<_>>().join(", ");
    let canonical = canonical_fields(&s.prompt);
    let prompt = page_prompt(boxes.len(), &letters, s.prompt.trim(), &canonical);
    let max_tokens = (boxes.len() as u32 * 70 + 256).clamp(512, 4096);
    let raw = openrouter(&img, &prompt, s, max_tokens, tokens)?;
    let result = parse_ai_json(&raw).map_err(|e| anyhow!("Model did not return valid JSON ({e}). Response: {}", raw.chars().take(300).collect::<String>()))?;

    let lookup: std::collections::HashMap<String, &String> = canonical.iter().map(|c| (normalize_key(c), c)).collect();
    for c in &canonical {
        t.add_column(c);
    }
    for (i, (id, _)) in annots.iter().enumerate() {
        let Some(Value::Object(values)) = result.get(&letter(i)) else { continue };
        let Some(row) = t.find_row(page, id) else { continue };
        for (key, val) in values {
            let col = lookup.get(&normalize_key(key)).map_or(key.clone(), |c| (*c).clone());
            let val = match val {
                Value::Null => continue,
                Value::String(s) => s.clone(),
                v => v.to_string(),
            };
            t.set(row, &col, val);
        }
    }
    if s.numbers_from_crops {
        numbers_from_crops(t, &full, page, &annots, s, tokens);
    }
    Ok(())
}

/// Re-read each drawing's small inventory number from a zoomed crop of the full page.
fn numbers_from_crops(t: &mut Tabular, full: &RgbImage, page: &str, annots: &[(String, [i64; 4])], s: &Settings, tokens: &mut u64) {
    let (w, h) = (full.width() as i64, full.height() as i64);
    t.add_column("number");
    for (id, [x1, y1, x2, y2]) in annots {
        let (bw, bh) = (x2 - x1, y2 - y1);
        let mx = (bw as f64 * 0.35) as i64 + 15;
        let (top, bottom) = ((bh as f64 * 0.25) as i64 + 15, (bh as f64 * 0.50) as i64 + 15); // numbers sit below
        let (cx1, cy1, cx2, cy2) = ((x1 - mx).max(0), (y1 - top).max(0), (x2 + mx).min(w), (y2 + bottom).min(h));
        if cx2 <= cx1 || cy2 <= cy1 {
            continue;
        }
        let mut crop = image::imageops::crop_imm(full, cx1 as u32, cy1 as u32, (cx2 - cx1) as u32, (cy2 - cy1) as u32).to_image();
        let longest = crop.width().max(crop.height());
        if longest < 768 {
            let f = 768.0 / longest as f64;
            crop = image::imageops::resize(&crop, (crop.width() as f64 * f) as u32, (crop.height() as f64 * f) as u32, FilterType::Lanczos3);
        }
        let Ok(raw) = openrouter(&crop, NUMBER_PROMPT, s, 20, tokens) else { continue };
        let value = raw.trim().trim_matches(|c| "\".'`".contains(c)).split_whitespace().next().unwrap_or("").to_string();
        if matches!(value.to_lowercase().as_str(), "" | "null" | "none" | "n/a") {
            continue;
        }
        if let Some(row) = t.find_row(page, id) {
            t.set(row, "number", value);
        }
    }
}

/// Run the extraction on `pages` and save mask_info.csv (also after errors or cancel).
pub fn extract(p: &mut Project, pages: Vec<String>, s: Settings, prog: &Progress) -> Result<String> {
    if s.api_key.trim().is_empty() {
        bail!("OpenRouter API key is required (Backend)");
    }
    let mut t = Tabular::load(p)?;
    for c in ["page", "plate", "figure", "number"] {
        t.add_column(c);
    }
    let (mut tokens, mut errors, mut done) = (0u64, vec![], 0);
    for (i, page) in pages.iter().enumerate() {
        if prog.cancelled() {
            break;
        }
        prog.set(i, pages.len(), format!("AI: {page} · {tokens} tokens"));
        match extract_page(&mut t, p, page, &s, &mut tokens) {
            Ok(()) => done += 1,
            Err(e) if pages.len() == 1 || e.to_string().contains("vision") => {
                t.save(p)?;
                return Err(e);
            }
            Err(e) => errors.push(format!("{page}: {e:#}")),
        }
    }
    t.save(p)?;
    let mut msg = format!("AI extraction: {done} of {} pages · {tokens} tokens", pages.len());
    if !errors.is_empty() {
        msg += &format!(" · {} errors (first: {})", errors.len(), errors[0]);
    }
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_from_messy_replies() {
        let fenced = "```json\n{\"A\": {\"page\": \"45\", \"number\": \"1\"},}\n```";
        assert_eq!(parse_ai_json(fenced).unwrap()["A"]["page"], "45");
        let truncated = "{\"A\": {\"number\": \"1\"}, \"B\": {\"number\": \"2\"}, \"C\": {\"numb";
        let m = parse_ai_json(truncated).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m["B"]["number"], "2");
        assert!(parse_ai_json("sorry, no").is_err());
    }

    #[test]
    fn bracket_fields() {
        assert_eq!(canonical_fields("Add also [Scale] and [ceramic class], not [Scale] twice"), ["Scale", "ceramic class"]);
    }

    #[test]
    fn base64_matches_rfc() {
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
    }
}
