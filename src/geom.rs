//! Raster/vector helpers standing in for the OpenCV calls of utils.py.

use crate::project::Polygon;
use image::{GrayImage, Luma};
use imageproc::geometry::approximate_polygon_dp;
use imageproc::point::Point;

pub fn perimeter(pts: &[[f64; 2]]) -> f64 {
    (0..pts.len()).map(|i| dist(pts[i], pts[(i + 1) % pts.len()])).sum()
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

pub fn area(pts: &[[f64; 2]]) -> f64 {
    let s: f64 = (0..pts.len()).map(|i| {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum();
    s.abs() / 2.0
}

pub fn contains(pts: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut j = pts.len().wrapping_sub(1);
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[j]);
        if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// `cv2.approxPolyDP(pts, max(3.5, 0.007 * perimeter), closed=True)`, rounded to 0.1 px.
/// The closed curve is split at the vertex farthest from the first and each half simplified.
pub fn simplify_closed(pts: &[[f64; 2]]) -> Polygon {
    simplify_eps(pts, (0.007 * perimeter(pts)).max(3.5))
}

/// Closed-curve Douglas-Peucker with a fixed tolerance (the editor's brush and "Simplify").
pub fn simplify_eps(pts: &[[f64; 2]], eps: f64) -> Polygon {
    let p: Vec<Point<f64>> = pts.iter().map(|q| Point::new(q[0], q[1])).collect();
    let far = (1..pts.len()).max_by(|&a, &b| dist(pts[0], pts[a]).total_cmp(&dist(pts[0], pts[b])));
    let Some(far) = far else { return pts.to_vec() };
    let mut out = approximate_polygon_dp(&p[..=far], eps, false);
    let mut back = p[far..].to_vec();
    back.push(p[0]);
    let back = approximate_polygon_dp(&back, eps, false);
    out.pop();
    out.extend_from_slice(&back[..back.len() - 1]);
    out.iter().map(|q| [(q.x * 10.0).round() / 10.0, (q.y * 10.0).round() / 10.0]).collect()
}

/// Fill a polygon (shifted by -`origin`, coordinates truncated like `np.int32`) with 255.
pub fn fill(mask: &mut GrayImage, pts: &[[f64; 2]], origin: (f64, f64), round: bool) {
    let cvt = |v: f64| if round { v.round() as i32 } else { v as i32 };
    let mut poly: Vec<Point<i32>> = vec![];
    for q in pts {
        let p = Point::new(cvt(q[0] - origin.0), cvt(q[1] - origin.1));
        if poly.last() != Some(&p) {
            poly.push(p);
        }
    }
    while poly.len() > 1 && poly.first() == poly.last() {
        poly.pop();
    }
    if poly.len() >= 3 {
        imageproc::drawing::draw_polygon_mut(mask, &poly, Luma([255]));
    }
}

/// Dilation with exactly `cv2.getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1))`: row dy spans
/// ±round(√(r²−dy²)). A plain L2 disk is a pixel short on the diagonals, enough to lose a
/// 1px vessel outline lying on the edge of an expanded polygon.
pub fn dilate(mask: &GrayImage, r: u32) -> GrayImage {
    if r == 0 {
        return mask.clone();
    }
    let (w, h) = (mask.width() as i64, mask.height() as i64);
    let r = r as i64;
    let half: Vec<i64> = (-r..=r).map(|dy| (((r * r - dy * dy) as f64).sqrt()).round() as i64).collect();
    // per-row prefix sums: "any foreground in [a, b]" in O(1)
    let pre: Vec<Vec<u32>> = (0..h)
        .map(|y| {
            let mut acc = 0;
            std::iter::once(0).chain((0..w).map(|x| { acc += (mask.get_pixel(x as u32, y as u32)[0] != 0) as u32; acc })).collect()
        })
        .collect();
    GrayImage::from_fn(w as u32, h as u32, |x, y| {
        let (x, y) = (x as i64, y as i64);
        let hit = (-r..=r).any(|dy| {
            let sy = y + dy;
            let d = half[(dy + r) as usize];
            (0..h).contains(&sy) && {
                let row = &pre[sy as usize];
                row[(x + d + 1).min(w) as usize] > row[(x - d).max(0) as usize]
            }
        });
        Luma([if hit { 255 } else { 0 }])
    })
}

/// `cv2.morphologyEx(MORPH_CLOSE)` with the same ellipse; outside the image never erodes, as in cv2.
pub fn close(mask: &GrayImage, r: u32) -> GrayImage {
    let invert = |m: &GrayImage| GrayImage::from_fn(m.width(), m.height(), |x, y| Luma([255 - m.get_pixel(x, y)[0]]));
    invert(&dilate(&invert(&dilate(mask, r)), r))
}

/// Fill holes: background not 4-connected to the border becomes foreground
/// (what drawing external contours filled does in OpenCV).
pub fn fill_holes(mask: &GrayImage) -> GrayImage {
    let (w, h) = mask.dimensions();
    let mut outside = vec![false; (w * h) as usize];
    let mut stack: Vec<(u32, u32)> = vec![];
    for x in 0..w {
        stack.extend([(x, 0), (x, h - 1)]);
    }
    for y in 0..h {
        stack.extend([(0, y), (w - 1, y)]);
    }
    while let Some((x, y)) = stack.pop() {
        let i = (y * w + x) as usize;
        if outside[i] || mask.get_pixel(x, y)[0] != 0 {
            continue;
        }
        outside[i] = true;
        if x > 0 { stack.push((x - 1, y)); }
        if y > 0 { stack.push((x, y - 1)); }
        if x + 1 < w { stack.push((x + 1, y)); }
        if y + 1 < h { stack.push((x, y + 1)); }
    }
    GrayImage::from_fn(w, h, |x, y| Luma([if outside[(y * w + x) as usize] { 0 } else { 255 }]))
}

/// Convex hull of the foreground, filled.
pub fn hull_mask(mask: &GrayImage) -> GrayImage {
    let (w, h) = mask.dimensions();
    // row extremes are enough: every hull vertex is a leftmost or rightmost pixel of its row
    let mut pts: Vec<Point<i32>> = vec![];
    for y in 0..h {
        let row = (0..w).filter(|&x| mask.get_pixel(x, y)[0] != 0);
        let (mut lo, mut hi) = (None, None);
        for x in row {
            lo = lo.or(Some(x));
            hi = Some(x);
        }
        if let (Some(lo), Some(hi)) = (lo, hi) {
            pts.extend([Point::new(lo as i32, y as i32), Point::new(hi as i32, y as i32)]);
        }
    }
    let mut out = GrayImage::new(w, h);
    let hull: Vec<[f64; 2]> = imageproc::geometry::convex_hull(pts).iter().map(|p| [p.x as f64, p.y as f64]).collect();
    fill(&mut out, &hull, (0.0, 0.0), false);
    out
}

/// Ear-clipping triangulation of a simple polygon (egui only fills convex paths).
/// O(n³), fine for the ~20-vertex simplified outlines.
pub fn triangulate(pts: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let orient = (0..pts.len()).map(|i| { let (a, b) = (pts[i], pts[(i + 1) % pts.len()]); a[0] * b[1] - b[0] * a[1] }).sum::<f64>().signum();
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    let mut out = vec![];
    while idx.len() > 3 {
        let m = idx.len();
        let ear = (0..m).find(|&i| {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            if cross(pts[a], pts[b], pts[c]) * orient <= 0.0 {
                return false; // reflex or degenerate
            }
            !idx.iter().any(|&j| {
                j != a && j != b && j != c && {
                    let p = pts[j];
                    let (d1, d2, d3) = (cross(pts[a], pts[b], p), cross(pts[b], pts[c], p), cross(pts[c], pts[a], p));
                    d1 * orient >= 0.0 && d2 * orient >= 0.0 && d3 * orient >= 0.0
                }
            })
        });
        let Some(i) = ear else { break }; // self-intersecting: give up on the rest
        out.push([idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]]);
        idx.remove(i);
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulate_concave() {
        let l = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [4.0, 4.0], [4.0, 10.0], [0.0, 10.0]];
        let tris = triangulate(&l);
        assert_eq!(tris.len(), 4);
        let sum: f64 = tris.iter().map(|t| area(&[l[t[0]], l[t[1]], l[t[2]]])).sum();
        assert!((sum - area(&l)).abs() < 1e-9);
    }

    #[test]
    fn simplify_dense_square() {
        let mut pts = vec![];
        for i in 0..100 { pts.push([i as f64, 0.0]); }
        for i in 0..100 { pts.push([100.0, i as f64]); }
        for i in 0..100 { pts.push([100.0 - i as f64, 100.0]); }
        for i in 0..100 { pts.push([0.0, 100.0 - i as f64]); }
        let s = simplify_closed(&pts);
        assert_eq!(s.len(), 4, "{s:?}");
        assert!((area(&s) - 10000.0).abs() < 1.0);
    }

    #[test]
    fn dilate_matches_cv2_ellipse() {
        // cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)).sum() == 213
        let mut m = GrayImage::new(41, 41);
        m.put_pixel(20, 20, Luma([255]));
        let d = dilate(&m, 8);
        assert_eq!(d.pixels().filter(|p| p[0] != 0).count(), 213);
        assert_eq!(d.get_pixel(20 + 7, 20 + 4)[0], 255); // diagonal reach an L2 disk misses
    }

    #[test]
    fn holes_get_filled() {
        let mut ring = GrayImage::new(20, 20);
        for y in 5..15 { for x in 5..15 {
            if x == 5 || x == 14 || y == 5 || y == 14 { ring.put_pixel(x, y, Luma([255])); }
        }}
        let filled = fill_holes(&ring);
        assert_eq!(filled.get_pixel(10, 10)[0], 255);
        assert_eq!(filled.get_pixel(2, 2)[0], 0);
    }
}
