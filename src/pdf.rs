//! PDF → page JPEGs: port of `PDFProcessor.process_pdf_to_folder` (PyMuPDF → pdfium).

use crate::Progress;
use crate::project::{self, Project};
use anyhow::{Context, Result, anyhow, bail};
use image::{RgbImage, codecs::jpeg::JpegEncoder};
use pdfium_render::prelude::*;
use serde_json::json;
use std::path::Path;

const DPI: f32 = 300.0;

/// libpdfium is looked up in `lib/` next to the executable, then in `./lib`, then system-wide.
fn pdfium() -> Result<Pdfium> {
    let exe_lib = std::env::current_exe()?.parent().map(|d| d.join("lib")).unwrap_or_default();
    let bindings = Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(&exe_lib))
        .or_else(|_| Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("lib/")))
        .or_else(|_| Pdfium::bind_to_system_library())
        .map_err(|e| anyhow!("libpdfium not found ({e}); put it in lib/"))?;
    Ok(Pdfium::new(bindings))
}

pub fn has_pdf(p: &Project) -> bool {
    !project::list_files(&p.dir("pdf_source"), |n| project::has_ext(n, &["pdf"])).is_empty()
}

pub fn convert(p: &mut Project, pdf: &Path, split: bool, prog: &Progress) -> Result<String> {
    if has_pdf(p) {
        bail!("This project already has a PDF");
    }
    let dest = p.dir("pdf_source").join(pdf.file_name().context("not a file")?);
    std::fs::create_dir_all(p.dir("pdf_source"))?;
    std::fs::copy(pdf, &dest)?;

    let pdfium = pdfium()?;
    let doc = pdfium.load_pdf_from_file(&dest, None).map_err(|e| anyhow!("opening PDF: {e}"))?;
    let out = p.dir("images");
    std::fs::create_dir_all(&out)?;
    // named after the project, not the PDF, like the Python app
    let base = p.meta.project_name.clone();
    let pages = doc.pages();
    let total = pages.len() as usize;
    let config = PdfRenderConfig::new().scale_page_by_factor(DPI / 72.0);
    for (i, page) in pages.iter().enumerate() {
        if prog.cancelled() {
            break;
        }
        prog.set(i, total, format!("Page {}", i + 1));
        let img = page.render_with_config(&config)?.as_image()?.to_rgb8();
        if split {
            let mid = img.width() / 2;
            save_jpg(&image::imageops::crop_imm(&img, 0, 0, mid, img.height()).to_image(), &out.join(format!("{base}_page_{i}a.jpg")))?;
            save_jpg(&image::imageops::crop_imm(&img, mid, 0, img.width() - mid, img.height()).to_image(), &out.join(format!("{base}_page_{i}b.jpg")))?;
        } else {
            save_jpg(&img, &out.join(format!("{base}_page_{i}.jpg")))?;
        }
    }

    let count = p.images().len();
    p.set_status("pdf_processed", json!(true));
    p.set_status("pdf_count", json!(1));
    p.set_status("images_extracted", json!(count));
    p.set_status("total_images", json!(count));
    p.save()?;
    Ok(format!("PDF processed: {count} images extracted"))
}

fn save_jpg(img: &RgbImage, path: &Path) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    JpegEncoder::new_with_quality(file, 75).encode_image(img)?; // PIL's default quality
    Ok(())
}
