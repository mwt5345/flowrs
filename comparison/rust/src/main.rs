use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use flowrs::{Flow, MafConfig, NsfConfig, RealNvpConfig};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

// ─── Config ───

#[derive(Deserialize)]
struct Config {
    training: TrainingConfig,
    data: DataConfig,
    maf: MafCfg,
    nsf: NsfCfg,
    realnvp: RealNvpCfg,
}

#[derive(Deserialize)]
struct TrainingConfig {
    batch_size: usize,
    num_epochs: usize,
    batches_per_epoch: usize,
    learning_rate: f64,
    seed: u64,
    n_train: usize,
    n_val: usize,
}

#[derive(Deserialize)]
struct DataConfig {
    noise: f32,
}

#[derive(Deserialize)]
struct MafCfg {
    num_layers: usize,
    hidden_sizes: Vec<usize>,
}

#[derive(Deserialize)]
struct NsfCfg {
    num_layers: usize,
    hidden_sizes: Vec<usize>,
    num_bins: usize,
    tail_bound: f32,
}

#[derive(Deserialize)]
struct RealNvpCfg {
    num_layers: usize,
    hidden_sizes: Vec<usize>,
}

// ─── Results ───

#[derive(Serialize)]
struct BenchResult {
    model: String,
    framework: String,
    train_throughput_samples_per_sec: f64,
    forward_ms_per_batch: f64,
    inverse_ms_per_batch: f64,
    final_val_nll: f64,
    train_time_sec: f64,
}

// ─── Data generation (matching Python's two_moons_data.py) ───

fn generate_two_moons(n_samples: usize, noise: f32, seed: u64) -> Vec<[f32; 2]> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    let n_half = n_samples / 2;
    let n_other = n_samples - n_half;
    let mut data = Vec::with_capacity(n_samples);

    // Upper moon
    for i in 0..n_half {
        let theta = std::f32::consts::PI * i as f32 / (n_half - 1).max(1) as f32;
        let x = theta.cos() + rng.r#gen::<f32>() * noise * 2.0 - noise;
        let y = theta.sin() + rng.r#gen::<f32>() * noise * 2.0 - noise;
        data.push([x, y]);
    }

    // Lower moon
    for i in 0..n_other {
        let theta = std::f32::consts::PI * i as f32 / (n_other - 1).max(1) as f32;
        let x = 1.0 - theta.cos() + rng.r#gen::<f32>() * noise * 2.0 - noise;
        let y = -theta.sin() + 0.5 + rng.r#gen::<f32>() * noise * 2.0 - noise;
        data.push([x, y]);
    }

    data.shuffle(&mut rng);
    data
}

fn sample_batch<B: Backend>(
    data: &[[f32; 2]],
    batch_size: usize,
    device: &B::Device,
    rng: &mut StdRng,
) -> Tensor<B, 2> {
    use rand::Rng;
    let n = data.len();
    let mut flat = Vec::with_capacity(batch_size * 2);
    for _ in 0..batch_size {
        let idx = rng.gen_range(0..n);
        flat.push(data[idx][0]);
        flat.push(data[idx][1]);
    }
    Tensor::from_floats(TensorData::new(flat, [batch_size, 2]), device)
}

// ─── Training + benchmarking ───

fn train_and_benchmark<B: AutodiffBackend, M>(
    mut model: M,
    train_data: &[[f32; 2]],
    val_data: &[[f32; 2]],
    cfg: &Config,
    model_name: &str,
    device: &B::Device,
) -> BenchResult
where
    M: AutodiffModule<B> + Flow<B>,
    M::InnerModule: Flow<<B as AutodiffBackend>::InnerBackend>,
{
    let mut rng = StdRng::seed_from_u64(cfg.training.seed);
    let batch_size = cfg.training.batch_size;

    let mut optim = AdamConfig::new().init();

    // Training
    let t_start = Instant::now();
    let mut total_samples: usize = 0;

    for _epoch in 1..=cfg.training.num_epochs {
        for _ in 0..cfg.training.batches_per_epoch {
            let batch = sample_batch::<B>(train_data, batch_size, device, &mut rng);
            let log_prob = model.log_prob(batch);
            let loss = -log_prob.mean();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.training.learning_rate, model, grads);
            total_samples += batch_size;
        }
    }

    let train_time = t_start.elapsed().as_secs_f64();
    let train_throughput = total_samples as f64 / train_time;

    let model_valid = model.valid();

    // Validation NLL
    let val_flat: Vec<f32> = val_data.iter().flat_map(|p| p.iter().copied()).collect();
    let val_tensor = Tensor::<<B as AutodiffBackend>::InnerBackend, 2>::from_floats(
        TensorData::new(val_flat, [val_data.len(), 2]),
        device,
    );
    let val_log_prob = model_valid.log_prob(val_tensor);
    let val_nll: f64 = (-val_log_prob.mean()).into_scalar().elem::<f32>() as f64;

    // Forward timing
    let n_timing = 100;
    // warmup
    for _ in 0..10 {
        let x = Tensor::<<B as AutodiffBackend>::InnerBackend, 2>::random(
            [batch_size, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let _ = model_valid.log_prob(x);
    }
    let t0 = Instant::now();
    for _ in 0..n_timing {
        let x = Tensor::<<B as AutodiffBackend>::InnerBackend, 2>::random(
            [batch_size, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let _ = model_valid.log_prob(x);
    }
    let forward_ms = t0.elapsed().as_secs_f64() / n_timing as f64 * 1000.0;

    // Inverse timing
    for _ in 0..10 {
        let z = Tensor::<<B as AutodiffBackend>::InnerBackend, 2>::random(
            [batch_size, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let _ = model_valid.inverse(z);
    }
    let t0 = Instant::now();
    for _ in 0..n_timing {
        let z = Tensor::<<B as AutodiffBackend>::InnerBackend, 2>::random(
            [batch_size, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let _ = model_valid.inverse(z);
    }
    let inverse_ms = t0.elapsed().as_secs_f64() / n_timing as f64 * 1000.0;

    println!("  Train throughput: {train_throughput:.0} samples/sec");
    println!("  Forward: {forward_ms:.3} ms/batch");
    println!("  Inverse: {inverse_ms:.3} ms/batch");
    println!("  Val NLL: {val_nll:.4}");
    println!("  Train time: {train_time:.2} sec");

    BenchResult {
        model: model_name.to_string(),
        framework: "flowrs".to_string(),
        train_throughput_samples_per_sec: (train_throughput * 10.0).round() / 10.0,
        forward_ms_per_batch: (forward_ms * 1000.0).round() / 1000.0,
        inverse_ms_per_batch: (inverse_ms * 1000.0).round() / 1000.0,
        final_val_nll: (val_nll * 10000.0).round() / 10000.0,
        train_time_sec: (train_time * 100.0).round() / 100.0,
    }
}

fn run<B: AutodiffBackend>(device: B::Device) {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config.toml");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", config_path.display()));
    let cfg: Config = toml::from_str(&config_str).expect("Failed to parse config.toml");

    let n_total = cfg.training.n_train + cfg.training.n_val;
    let data = generate_two_moons(n_total, cfg.data.noise, cfg.training.seed);
    let train_data = &data[..cfg.training.n_train];
    let val_data = &data[cfg.training.n_train..];

    println!(
        "Dataset: {} train, {} val",
        train_data.len(),
        val_data.len()
    );

    let mut results = Vec::new();

    // MAF
    println!("\n=== MAF ===");
    let maf = MafConfig::new(2, cfg.maf.num_layers, cfg.maf.hidden_sizes.clone())
        .with_seed(cfg.training.seed)
        .init::<B>(&device);
    results.push(train_and_benchmark::<B, _>(
        maf,
        train_data,
        val_data,
        &cfg,
        "MAF",
        &device,
    ));

    // NSF
    println!("\n=== NSF ===");
    let nsf = NsfConfig::new(2, cfg.nsf.num_layers, cfg.nsf.hidden_sizes.clone())
        .with_num_bins(cfg.nsf.num_bins)
        .with_tail_bound(cfg.nsf.tail_bound)
        .init::<B>(&device);
    results.push(train_and_benchmark::<B, _>(
        nsf,
        train_data,
        val_data,
        &cfg,
        "NSF",
        &device,
    ));

    // RealNVP
    println!("\n=== RealNVP ===");
    let nvp = RealNvpConfig::new(2, cfg.realnvp.num_layers, cfg.realnvp.hidden_sizes.clone())
        .init::<B>(&device);
    results.push(train_and_benchmark::<B, _>(
        nvp,
        train_data,
        val_data,
        &cfg,
        "RealNVP",
        &device,
    ));

    let out_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rust_results.json");
    let json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    std::fs::write(&out_path, json).expect("Failed to write results JSON");
    println!("\nResults written to {}", out_path.display());
}

fn main() {
    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        println!("Backend: WGPU");
        run::<Autodiff<Wgpu>>(Default::default());
    }

    #[cfg(feature = "cuda")]
    {
        use burn::backend::{Autodiff, Cuda};
        println!("Backend: CUDA");
        run::<Autodiff<Cuda>>(Default::default());
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::{Autodiff, NdArray};
        println!("Backend: NdArray (CPU)");
        run::<Autodiff<NdArray>>(Default::default());
    }
}
