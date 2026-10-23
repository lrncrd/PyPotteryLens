//! Thumbnails for the page and card grids, decoded off the UI thread.

use eframe::egui::{self, ColorImage, TextureHandle};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

const SIDE: u32 = 220;
const KEEP: usize = 600; // textures kept on the GPU; least recently drawn are dropped beyond this

type Queue = Arc<(Mutex<Vec<PathBuf>>, Condvar)>;

pub struct Thumbs {
    queue: Queue,
    done: Receiver<(PathBuf, Option<ColorImage>)>,
    map: HashMap<PathBuf, (Option<TextureHandle>, u64)>, // None = pending or failed
    frame: u64,
}

impl Thumbs {
    pub fn new(ctx: &egui::Context) -> Self {
        let queue: Queue = Default::default();
        let (tx, done) = channel();
        for _ in 0..4 {
            spawn_worker(queue.clone(), tx.clone(), ctx.clone());
        }
        Self { queue, done, map: HashMap::new(), frame: 0 }
    }

    /// Call once per frame before `get`.
    pub fn begin_frame(&mut self, ctx: &egui::Context) {
        self.frame += 1;
        while let Ok((path, img)) = self.done.try_recv() {
            let tex = img.map(|img| ctx.load_texture(path.display().to_string(), img, egui::TextureOptions::LINEAR));
            self.map.insert(path, (tex, self.frame));
        }
        if self.map.len() > KEEP {
            let mut by_age: Vec<(PathBuf, u64)> = self.map.iter().map(|(p, (_, f))| (p.clone(), *f)).collect();
            by_age.sort_by_key(|(_, f)| *f);
            for (p, _) in by_age.into_iter().take(self.map.len() - KEEP) {
                self.map.remove(&p);
            }
        }
    }

    pub fn get(&mut self, path: &Path) -> Option<TextureHandle> {
        if let Some((tex, used)) = self.map.get_mut(path) {
            *used = self.frame;
            return tex.clone();
        }
        self.map.insert(path.to_path_buf(), (None, self.frame));
        let (lock, cv) = &*self.queue;
        lock.lock().unwrap().push(path.to_path_buf());
        cv.notify_one();
        None
    }

    /// Drop one thumbnail (its file changed).
    pub fn forget(&mut self, path: &Path) {
        self.map.remove(path);
    }

    /// Forget everything (files were rewritten by a job).
    pub fn clear(&mut self) {
        self.map.clear();
        self.queue.0.lock().unwrap().clear();
    }
}

fn spawn_worker(queue: Queue, tx: Sender<(PathBuf, Option<ColorImage>)>, ctx: egui::Context) {
    std::thread::spawn(move || {
        loop {
            let path = {
                let (lock, cv) = &*queue;
                let mut q = cv.wait_while(lock.lock().unwrap(), |q| q.is_empty()).unwrap();
                q.pop().unwrap() // newest first: what's on screen now
            };
            let img = image::open(&path).ok().map(|i| {
                let t = i.thumbnail(SIDE, SIDE).to_rgba8();
                ColorImage::from_rgba_unmultiplied([t.width() as usize, t.height() as usize], t.as_raw())
            });
            if tx.send((path, img)).is_err() {
                return;
            }
            ctx.request_repaint();
        }
    });
}
