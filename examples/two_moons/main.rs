#![recursion_limit = "8192"]

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::path::Path;

use flowrs::{Flow, MafConfig, NsfConfig, RealNvpConfig};

mod data;
mod viz;

use data::{ToyDataset, generate_dataset, sample_batch};
use viz::{contour_svg, scatter_svg, tensor_to_points, write_csv};

#[derive(Clone, Copy, Debug)]
enum FlowModel {
    Maf,
    Nsf,
    RealNvp,
}

/// Evaluate log_prob on a 2D grid and return density values.
fn eval_density_grid<B: Backend>(
    model: &impl Flow<B>,
    nx: usize,
    ny: usize,
    bounds: (f32, f32, f32, f32),
    device: &B::Device,
) -> Vec<f32> {
    let (min_x, max_x, min_y, max_y) = bounds;
    let mut grid_points = Vec::with_capacity(nx * ny * 2);
    for iy in 0..ny {
        let y = min_y + (max_y - min_y) * iy as f32 / (ny - 1) as f32;
        for ix in 0..nx {
            let x = min_x + (max_x - min_x) * ix as f32 / (nx - 1) as f32;
            grid_points.push(x);
            grid_points.push(y);
        }
    }

    let n = nx * ny;
    let chunk_size = 4096;
    let mut all_densities = Vec::with_capacity(n);

    for start in (0..n).step_by(chunk_size) {
        let end = (start + chunk_size).min(n);
        let batch_len = end - start;
        let flat: Vec<f32> = grid_points[start * 2..end * 2].to_vec();
        let tensor = Tensor::<B, 2>::from_floats(TensorData::new(flat, [batch_len, 2]), device);
        let log_probs = model.log_prob(tensor);
        let data = log_probs.to_data();
        let lp: Vec<f32> = data.to_vec().unwrap();
        for &v in &lp {
            all_densities.push(v.exp());
        }
    }

    all_densities
}

fn visualize<B: Backend>(
    model: &impl Flow<B>,
    all_data: &[[f32; 2]],
    device: &B::Device,
    prefix: &str,
    title: &str,
) {
    println!("Generating {} visualizations...", prefix);

    // 1. Scatter plot of samples
    let n_samples = 5000;
    let z = Tensor::<B, 2>::random(
        [n_samples, 2],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let samples = model.inverse(z);
    let sample_points = tensor_to_points(samples);

    let csv_path = format!("{}_samples.csv", prefix);
    let svg_path = format!("{}_samples.svg", prefix);
    write_csv(Path::new(&csv_path), &sample_points).expect("Failed to write CSV");
    let svg = scatter_svg(&sample_points, 600, 400, &format!("{} Samples", title));
    std::fs::write(&svg_path, &svg).expect("Failed to write SVG");

    // 2. Density contour plot
    let grid_res = 200;
    let bounds = (-1.5_f32, 2.5, -1.0, 1.5);
    println!("Evaluating density on {grid_res}x{grid_res} grid...");
    let density = eval_density_grid(model, grid_res, grid_res, bounds, device);

    let svg_contour = contour_svg(
        &density,
        grid_res,
        grid_res,
        bounds,
        800,
        600,
        &format!("{} Density", title),
        Some(all_data),
    );
    std::fs::write(format!("{}_density.svg", prefix), &svg_contour)
        .expect("Failed to write density SVG");

    let svg_clean = contour_svg(
        &density,
        grid_res,
        grid_res,
        bounds,
        800,
        600,
        &format!("{} Density", title),
        None,
    );
    std::fs::write(format!("{}_density_clean.svg", prefix), &svg_clean)
        .expect("Failed to write clean density SVG");

    println!("  {}_samples.csv / .svg", prefix);
    println!("  {}_density.svg / {}_density_clean.svg", prefix, prefix);
}

fn train_flow<B: AutodiffBackend, M>(
    mut model: M,
    train_data: &[[f32; 2]],
    val_data: &[[f32; 2]],
    all_data: &[[f32; 2]],
    device: &B::Device,
    rng: &mut StdRng,
    prefix: &str,
    title: &str,
    num_epochs: usize,
    lr: f64,
) where
    M: AutodiffModule<B> + Flow<B>,
    M::InnerModule: Flow<<B as AutodiffBackend>::InnerBackend>,
{
    let batch_size = 512;
    let batches_per_epoch = 20;
    let eval_every = 50;
    let patience = 200;

    let mut optim = AdamConfig::new().init();
    let mut best_val_nll = f32::MAX;
    let mut epochs_without_improvement = 0usize;

    for epoch in 1..=num_epochs {
        let mut epoch_loss_sum = 0.0_f32;

        for _ in 0..batches_per_epoch {
            let batch = sample_batch::<B>(train_data, batch_size, device, rng);
            let log_prob = model.log_prob(batch);
            let loss = -log_prob.mean();

            let loss_val: f32 = loss.clone().into_scalar().elem();
            epoch_loss_sum += loss_val;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            let progress = (epoch - 1) as f64 / num_epochs as f64;
            let effective_lr = lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            model = optim.step(effective_lr, model, grads);
        }

        let train_nll = epoch_loss_sum / batches_per_epoch as f32;

        if epoch % eval_every == 0 || epoch == 1 {
            let model_valid = model.valid();
            let val_batch = sample_batch::<<B as AutodiffBackend>::InnerBackend>(
                val_data,
                val_data.len().min(2048),
                device,
                rng,
            );
            let val_log_prob = model_valid.log_prob(val_batch);
            let val_nll: f32 = (-val_log_prob.mean()).into_scalar().elem();

            let improved = if val_nll < best_val_nll {
                best_val_nll = val_nll;
                epochs_without_improvement = 0;
                " *"
            } else {
                epochs_without_improvement += eval_every;
                ""
            };

            let progress = (epoch - 1) as f64 / num_epochs as f64;
            let current_lr = lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            println!(
                "[{prefix}] Epoch {epoch:>4}/{num_epochs}  train: {train_nll:.4}  val: {val_nll:.4}  best_val: {best_val_nll:.4}  lr: {current_lr:.2e}{improved}"
            );

            if epochs_without_improvement >= patience {
                println!("[{prefix}] Early stopping: no val improvement for {patience} epochs");
                break;
            }
        }
    }

    let model_valid = model.valid();
    visualize(&model_valid, all_data, device, prefix, title);
}

fn train<B: AutodiffBackend>(device: B::Device) {
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);
    let n_data = 30_000;

    let mut all_data = generate_dataset(ToyDataset::TwoMoons, n_data, 0.05, seed);
    all_data.shuffle(&mut rng);
    let n_train = (n_data as f32 * 0.8) as usize;
    let train_data = &all_data[..n_train];
    let val_data = &all_data[n_train..];
    println!(
        "Dataset: {} train, {} val",
        train_data.len(),
        val_data.len()
    );

    // Determine which model(s) to train from FLOW_MODEL env var
    let flow_models: Vec<FlowModel> = match std::env::var("FLOW_MODEL").as_deref() {
        Ok("maf") => vec![FlowModel::Maf],
        Ok("nsf") => vec![FlowModel::Nsf],
        Ok("realnvp") => vec![FlowModel::RealNvp],
        Ok("all") => vec![FlowModel::Maf, FlowModel::Nsf, FlowModel::RealNvp],
        _ => vec![FlowModel::Maf], // default
    };

    for flow_model in &flow_models {
        println!("\n=== Training {:?} ===", flow_model);
        // Reset RNG for each model so results are comparable
        let mut model_rng = StdRng::seed_from_u64(seed);

        match flow_model {
            FlowModel::Maf => {
                let config = MafConfig::new(2, 10, vec![256, 256]).with_seed(seed);
                let model = config.init::<B>(&device);
                train_flow::<B, _>(
                    model,
                    train_data,
                    val_data,
                    &all_data,
                    &device,
                    &mut model_rng,
                    "maf",
                    "MAF (Two Moons)",
                    2000,
                    5e-4,
                );
            }
            FlowModel::Nsf => {
                let config = NsfConfig::new(2, 8, vec![128, 128])
                    .with_num_bins(8)
                    .with_tail_bound(3.0);
                let model = config.init::<B>(&device);
                train_flow::<B, _>(
                    model,
                    train_data,
                    val_data,
                    &all_data,
                    &device,
                    &mut model_rng,
                    "nsf",
                    "NSF (Two Moons)",
                    2000,
                    5e-4,
                );
            }
            FlowModel::RealNvp => {
                let config = RealNvpConfig::new(2, 8, vec![128, 128]);
                let model = config.init::<B>(&device);
                train_flow::<B, _>(
                    model,
                    train_data,
                    val_data,
                    &all_data,
                    &device,
                    &mut model_rng,
                    "realnvp",
                    "RealNVP (Two Moons)",
                    2000,
                    5e-4,
                );
            }
        }
    }

    // Write training data reference
    write_csv(Path::new("training_data.csv"), &all_data).expect("Failed to write training CSV");
    let svg_train = scatter_svg(&all_data, 600, 400, "Training Data (Two Moons)");
    std::fs::write("training_data.svg", &svg_train).expect("Failed to write training SVG");

    println!("\nDone!");
}

fn main() {
    env_logger::init();

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        let device = Default::default();
        log::info!("Using WGPU backend");
        train::<Autodiff<Wgpu>>(device);
    }

    #[cfg(feature = "cuda")]
    {
        use burn::backend::{Autodiff, Cuda};
        let device = Default::default();
        log::info!("Using CUDA backend");
        train::<Autodiff<Cuda>>(device);
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::{Autodiff, NdArray};
        let device = Default::default();
        log::info!("Using NdArray backend");
        train::<Autodiff<NdArray>>(device);
    }
}
