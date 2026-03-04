<p align="center">
  <img src="assets/flowrs-logo.png?raw=true" alt="flowrs logo" width="140"/>
</p>


<p align="center">
  Normalizing flows in Rust, built on <a href="https://burn.dev">Burn</a>.
</p>

<p align="center">
  <a href="#flow-types">Flow Types</a> &middot;
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#examples">Examples</a>
</p>

---

**flowrs** is a normalizing flow library for density estimation and generative modeling.
It provides composable flow architectures with exact log-likelihood computation,
efficient sampling, and invertible transforms.

All models share a unified API:

```rust
let (z, log_det) = model.forward(x);  // encode
let x = model.inverse(z);             // decode / sample
let log_p = model.log_prob(x);        // exact log-likelihood
```

## Flow Types

| Model | Description | Paper |
|-------|-------------|-------|
| **MAF** | Masked Autoregressive Flow | [Papamakarios et al., 2017](https://arxiv.org/abs/1705.07057) |
| **NSF** | Neural Spline Flow (rational quadratic) | [Durkan et al., 2019](https://arxiv.org/abs/1906.04032) |
| **RealNVP** | Affine coupling layers | [Dinh et al., 2017](https://arxiv.org/abs/1605.08803) |

### Building Blocks

Individual components can be composed into custom architectures:

- **ActNorm** -- learnable scale and shift (data-dependent initialization)
- **LULinear** -- invertible linear layer via LU decomposition
- **AffineCoupling** -- affine coupling transform
- **SplineCoupling** -- rational quadratic spline coupling transform
- **MADE** -- Masked Autoregressive Distribution Estimator

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
flowrs = { git = "https://github.com/mwt5345/flowrs" }
burn = { version = "0.20", features = ["autodiff"] }
```

### Backend Features

flowrs supports multiple Burn backends via feature flags:

| Feature | Backend | Use case |
|---------|---------|----------|
| `wgpu` (default) | WebGPU | GPU acceleration, cross-platform |
| `cuda` | CUDA | NVIDIA GPUs |
| `ndarray` | NdArray | CPU-only, no GPU dependencies |

```toml
# CPU-only
flowrs = { git = "https://github.com/mwt5345/flowrs", default-features = false, features = ["ndarray"] }
```

## Quick Start

### Neural Spline Flow (recommended)

```rust
use burn::prelude::*;
use burn::backend::{Autodiff, NdArray};
use flowrs::NsfConfig;

type B = Autodiff<NdArray>;

let device = Default::default();

// 2D data, 6 flow layers, [128, 128] hidden units, 8 spline bins
let model: flowrs::Nsf<B> = NsfConfig::new(2, 6, vec![128, 128])
    .with_num_bins(8)
    .with_tail_bound(3.0)
    .init::<B>(&device);

// Compute log-likelihood
let data: Tensor<B, 2> = Tensor::random([1000, 2], Distribution::Normal(0.0, 1.0), &device);
let log_prob = model.log_prob(data);

// Generate samples (use inference backend)
use burn::module::AutodiffModule;
let model_infer: flowrs::Nsf<NdArray> = model.valid();
let z = Tensor::<NdArray, 2>::random([500, 2], Distribution::Normal(0.0, 1.0), &device);
let samples = model_infer.inverse(z);
```

### Masked Autoregressive Flow

```rust
use flowrs::MafConfig;

let model: flowrs::Maf<B> = MafConfig::new(2, 8, vec![256, 256])
    .with_seed(42)
    .init::<B>(&device);
```

### RealNVP

```rust
use flowrs::RealNvpConfig;

let model: flowrs::RealNvp<B> = RealNvpConfig::new(2, 8, vec![128, 128])
    .init::<B>(&device);
```

### Training Loop

```rust
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

let mut optim = AdamConfig::new().init();

for epoch in 1..=500 {
    let batch: Tensor<B, 2> = /* your data */;
    let loss = -model.log_prob(batch).mean();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    model = optim.step(1e-3, model, grads);
}
```

## Examples

### Two Moons CLI

Train MAF, NSF, or RealNVP on the two-moons dataset and output SVG visualizations:

```bash
# Train all models
FLOW_MODEL=all cargo run --release --example two_moons

# Train a specific model
FLOW_MODEL=nsf cargo run --release --example two_moons
```

### Jupyter Notebook

An interactive [evcxr](https://github.com/evcxr/evcxr) Jupyter notebook is provided in [`notebooks/two_moons_nsf.ipynb`](notebooks/two_moons_nsf.ipynb), demonstrating NSF training with inline plots via [plotpy](https://github.com/cpmech/plotpy).

## Architecture

```
src/
├── lib.rs             # Public API and re-exports
├── maf.rs             # Masked Autoregressive Flow
├── made.rs            # MADE autoregressive network
├── masked_linear.rs   # Masked linear layer
├── nsf.rs             # Neural Spline Flow (composite)
├── realnvp.rs         # RealNVP (composite)
├── coupling.rs        # Affine and spline coupling layers
├── spline.rs          # Rational quadratic spline transforms
├── actnorm.rs         # Activation normalization
├── lu_linear.rs       # LU-decomposed invertible linear
├── mlp.rs             # MLP conditioner network
└── flow.rs            # Shared utilities
```

## License

MIT
