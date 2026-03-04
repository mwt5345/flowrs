use burn::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn tensor_to_points<B: Backend>(tensor: Tensor<B, 2>) -> Vec<[f32; 2]> {
    let dims = tensor.dims();
    let data = tensor.to_data();
    let values: Vec<f32> = data.to_vec().unwrap();
    let mut points = Vec::with_capacity(dims[0]);
    for i in 0..dims[0] {
        points.push([values[i * 2], values[i * 2 + 1]]);
    }
    points
}

pub fn write_csv(path: &Path, points: &[[f32; 2]]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "x,y")?;
    for p in points {
        writeln!(f, "{},{}", p[0], p[1])?;
    }
    Ok(())
}

pub fn scatter_svg(points: &[[f32; 2]], w: u32, h: u32, title: &str) -> String {
    // Find bounds
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for p in points {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    // Add padding
    let pad_x = (max_x - min_x) * 0.1;
    let pad_y = (max_y - min_y) * 0.1;
    min_x -= pad_x;
    max_x += pad_x;
    min_y -= pad_y;
    max_y += pad_y;

    let margin = 40.0_f32;
    let plot_w = w as f32 - 2.0 * margin;
    let plot_h = h as f32 - 2.0 * margin;

    let map_x = |x: f32| margin + (x - min_x) / (max_x - min_x) * plot_w;
    let map_y = |y: f32| margin + (1.0 - (y - min_y) / (max_y - min_y)) * plot_h;

    let mid_x = w as f32 / 2.0;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">\n\
         <rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n\
         <text x=\"{mid_x}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\">{title}</text>\n\
         <rect x=\"{margin}\" y=\"{margin}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"none\" stroke=\"#ccc\"/>\n"
    );

    for p in points {
        let cx = map_x(p[0]);
        let cy = map_y(p[1]);
        svg.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"1.5\" fill=\"#4285f4\" opacity=\"0.6\"/>\n"
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

/// Render a filled contour (heatmap) plot of a 2D density grid as SVG.
/// `density[iy * nx + ix]` is the density at grid point (ix, iy).
/// `bounds` = (min_x, max_x, min_y, max_y).
pub fn contour_svg(
    density: &[f32],
    nx: usize,
    ny: usize,
    bounds: (f32, f32, f32, f32),
    w: u32,
    h: u32,
    title: &str,
    training_points: Option<&[[f32; 2]]>,
) -> String {
    let (min_x, max_x, min_y, max_y) = bounds;
    let margin = 40.0_f32;
    let plot_w = w as f32 - 2.0 * margin;
    let plot_h = h as f32 - 2.0 * margin;

    let map_x = |x: f32| margin + (x - min_x) / (max_x - min_x) * plot_w;
    let map_y = |y: f32| margin + (1.0 - (y - min_y) / (max_y - min_y)) * plot_h;

    // Find density range (ignore zeros/negatives for log scale)
    let max_d = density.iter().cloned().fold(0.0_f32, f32::max);
    if max_d <= 0.0 {
        return String::from("<svg></svg>");
    }

    let mid_x = w as f32 / 2.0;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">\n\
         <rect width=\"{w}\" height=\"{h}\" fill=\"#1a1a2e\"/>\n\
         <text x=\"{mid_x}\" y=\"24\" text-anchor=\"middle\" font-size=\"14\" \
         font-family=\"sans-serif\" fill=\"white\">{title}</text>\n"
    );

    // Draw filled cells as rectangles
    let cell_w = (max_x - min_x) / (nx - 1) as f32;
    let cell_h = (max_y - min_y) / (ny - 1) as f32;

    for iy in 0..ny - 1 {
        for ix in 0..nx - 1 {
            // Average density of 4 corners
            let d00 = density[iy * nx + ix];
            let d10 = density[iy * nx + ix + 1];
            let d01 = density[(iy + 1) * nx + ix];
            let d11 = density[(iy + 1) * nx + ix + 1];
            let avg = (d00 + d10 + d01 + d11) / 4.0;

            if avg <= 0.0 {
                continue;
            }

            let t = (avg / max_d).sqrt(); // sqrt for better visual spread

            // Viridis-inspired colormap: dark purple -> teal -> yellow
            let (r, g, b) = viridis_color(t);

            let x1 = map_x(min_x + ix as f32 * cell_w);
            let y1 = map_y(min_y + (iy + 1) as f32 * cell_h);
            let rw = map_x(min_x + (ix + 1) as f32 * cell_w) - x1;
            let rh = map_y(min_y + iy as f32 * cell_h) - y1;

            svg.push_str(&format!(
                "<rect x=\"{x1:.1}\" y=\"{y1:.1}\" width=\"{rw:.1}\" height=\"{rh:.1}\" \
                 fill=\"rgb({r},{g},{b})\" />\n"
            ));
        }
    }

    // Draw contour lines using marching squares
    let levels = compute_contour_levels(density, 10);
    for (li, &level) in levels.iter().enumerate() {
        let brightness = 150 + (105.0 * li as f32 / levels.len() as f32) as u32;
        let segments = marching_squares(density, nx, ny, level);
        for (x1f, y1f, x2f, y2f) in &segments {
            let sx1 = map_x(min_x + x1f * cell_w);
            let sy1 = map_y(min_y + y1f * cell_h);
            let sx2 = map_x(min_x + x2f * cell_w);
            let sy2 = map_y(min_y + y2f * cell_h);
            svg.push_str(&format!(
                "<line x1=\"{sx1:.1}\" y1=\"{sy1:.1}\" x2=\"{sx2:.1}\" y2=\"{sy2:.1}\" \
                 stroke=\"rgba({brightness},{brightness},{brightness},0.5)\" stroke-width=\"0.8\" />\n"
            ));
        }
    }

    // Overlay training data points if provided
    if let Some(pts) = training_points {
        for p in pts {
            let cx = map_x(p[0]);
            let cy = map_y(p[1]);
            if cx >= margin && cx <= margin + plot_w && cy >= margin && cy <= margin + plot_h {
                svg.push_str(&format!(
                    "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"0.8\" fill=\"white\" opacity=\"0.15\"/>\n"
                ));
            }
        }
    }

    // Border
    svg.push_str(&format!(
        "<rect x=\"{margin}\" y=\"{margin}\" width=\"{plot_w}\" height=\"{plot_h}\" \
         fill=\"none\" stroke=\"#666\"/>\n"
    ));

    svg.push_str("</svg>\n");
    svg
}

fn viridis_color(t: f32) -> (u8, u8, u8) {
    // Simplified viridis: dark purple -> blue -> teal -> green -> yellow
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (
            68.0 + s * (33.0 - 68.0),
            1.0 + s * (102.0 - 1.0),
            84.0 + s * (172.0 - 84.0),
        )
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (
            33.0 + s * (32.0 - 33.0),
            102.0 + s * (165.0 - 102.0),
            172.0 + s * (168.0 - 172.0),
        )
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (
            32.0 + s * (128.0 - 32.0),
            165.0 + s * (205.0 - 165.0),
            168.0 + s * (80.0 - 168.0),
        )
    } else {
        let s = (t - 0.75) / 0.25;
        (
            128.0 + s * (253.0 - 128.0),
            205.0 + s * (231.0 - 205.0),
            80.0 + s * (37.0 - 80.0),
        )
    };
    (r as u8, g as u8, b as u8)
}

fn compute_contour_levels(density: &[f32], n: usize) -> Vec<f32> {
    let max_d = density.iter().cloned().fold(0.0_f32, f32::max);
    if max_d <= 0.0 {
        return vec![];
    }
    (1..=n).map(|i| max_d * i as f32 / (n + 1) as f32).collect()
}

/// Marching squares: returns line segments (x1, y1, x2, y2) in grid coordinates.
fn marching_squares(
    density: &[f32],
    nx: usize,
    ny: usize,
    level: f32,
) -> Vec<(f32, f32, f32, f32)> {
    let mut segments = Vec::new();

    for iy in 0..ny - 1 {
        for ix in 0..nx - 1 {
            let d00 = density[iy * nx + ix]; // bottom-left
            let d10 = density[iy * nx + ix + 1]; // bottom-right
            let d01 = density[(iy + 1) * nx + ix]; // top-left
            let d11 = density[(iy + 1) * nx + ix + 1]; // top-right

            let b00 = d00 >= level;
            let b10 = d10 >= level;
            let b01 = d01 >= level;
            let b11 = d11 >= level;

            let case = (b00 as u8) | ((b10 as u8) << 1) | ((b01 as u8) << 2) | ((b11 as u8) << 3);

            if case == 0 || case == 15 {
                continue;
            }

            let x = ix as f32;
            let y = iy as f32;

            // Interpolation helpers
            let lerp = |a: f32, b: f32| -> f32 {
                if (b - a).abs() < 1e-10 {
                    0.5
                } else {
                    (level - a) / (b - a)
                }
            };

            // Edge midpoints with interpolation
            let bottom = (x + lerp(d00, d10), y); // bottom edge
            let top = (x + lerp(d01, d11), y + 1.0); // top edge
            let left = (x, y + lerp(d00, d01)); // left edge
            let right = (x + 1.0, y + lerp(d10, d11)); // right edge

            let add = |segs: &mut Vec<(f32, f32, f32, f32)>, a: (f32, f32), b: (f32, f32)| {
                segs.push((a.0, a.1, b.0, b.1));
            };

            match case {
                1 => add(&mut segments, bottom, left),
                2 => add(&mut segments, bottom, right),
                3 => add(&mut segments, left, right),
                4 => add(&mut segments, left, top),
                5 => add(&mut segments, bottom, top),
                6 => {
                    add(&mut segments, bottom, left);
                    add(&mut segments, right, top);
                }
                7 => add(&mut segments, right, top),
                8 => add(&mut segments, right, top),
                9 => {
                    add(&mut segments, bottom, right);
                    add(&mut segments, left, top);
                }
                10 => add(&mut segments, bottom, top),
                11 => add(&mut segments, left, top),
                12 => add(&mut segments, left, right),
                13 => add(&mut segments, bottom, right),
                14 => add(&mut segments, bottom, left),
                _ => {}
            }
        }
    }

    segments
}
