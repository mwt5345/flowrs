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
  <a href="#conditional-flows">Conditional Flows</a> &middot;
  <a href="#python-bindings">Python Bindings</a> &middot;
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

## Conditional Flows

All flow types support conditional density estimation $p(x \mid \text{context})$ via an optional context vector. This is useful for learning distributions that depend on external parameters (e.g., physical parameters, class labels).

```rust
use flowrs::MafConfig;

// 3D target, 5D context vector, 4 flow layers
let model: flowrs::Maf<B> = MafConfig::new(3, 4, vec![64, 64])
    .with_d_context(Some(5))
    .with_seed(42)
    .init::<B>(&device);

let x: Tensor<B, 2> = /* [batch, 3] target data */;
let ctx: Tensor<B, 2> = /* [batch, 5] conditioning variables */;

// Conditional log-likelihood
let log_p = model.log_prob_conditional(x.clone(), Some(ctx.clone()));

// Conditional forward / inverse
let (z, log_det) = model.forward_conditional(x, Some(ctx.clone()));
let x_reconstructed = model.inverse_conditional(z, Some(ctx));
```

Context is injected additively at each hidden layer of the MADE network. When no context is provided (`None`), the model behaves as an unconditional flow — existing code is fully backward compatible.

## Python Bindings

**pyflowrs** provides Python bindings via [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs/), giving you access to all flow architectures from Python/Jupyter with an `nflows`-style API.

### Installation

```bash
cd pyflowrs
pip install maturin
maturin develop --release
```

### Usage

```python
import numpy as np
import pyflowrs

# Unconditional flow
flow = pyflowrs.MAF(d_input=2, num_flows=6, hidden_sizes=[128, 128])
history = flow.fit(X_train, num_steps=10000, batch_size=64, lr=3e-4)

log_p = flow.log_prob(X_test)         # exact log-likelihood
samples = flow.sample(1000)           # generate samples
```

### Conditional flows

```python
# Conditional: p(x | context)
flow = pyflowrs.MAF(d_input=3, num_flows=4, hidden_sizes=[64, 64], d_context=5)

history = flow.fit(
    X_train, y=context_train,
    x_val=X_val, y_val=context_val,
    num_steps=20000,
    batch_size=64,
    lr=3e-4,
    noise_std=0.1,       # input noise regularization
    weight_decay=1e-3,   # L2 regularization
    patience=5000,
)

# Condition on a specific context
ctx = np.array([[0.96, 0.67, 0.022, 0.31, 0.81]], dtype=np.float32)
samples = flow.sample(5000, context=ctx)
log_p = flow.log_prob(X_test, context=ctx_test)
```

All three architectures (`MAF`, `NSF`, `RealNVP`) share the same Python API.

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

## Benchmarks

### flowrs vs nflows (Python/PyTorch)

Head-to-head on the **two-moons** density estimation task (CPU, batch 512).
Both frameworks use the same architecture per model — see [`comparison/config.toml`](comparison/config.toml) for full details.

#### Throughput & latency

| Model | Metric | flowrs | nflows | Speedup |
|-------|--------|-------:|-------:|--------:|
| **MAF** | Train (samples/sec) | 12,799 | 4,683 | **2.7x** |
| | Forward (ms/batch) | 9.0 | 25.9 | **2.9x** |
| | Inverse (ms/batch) | 18.9 | 54.3 | **2.9x** |
| **NSF** | Train (samples/sec) | 17,709 | 11,804 | **1.5x** |
| | Forward (ms/batch) | 8.4 | 13.2 | **1.6x** |
| | Inverse (ms/batch) | 8.0 | 29.1 | **3.6x** |
| **RealNVP** | Train (samples/sec) | 37,179 | 15,969 | **2.3x** |
| | Forward (ms/batch) | 3.5 | 8.6 | **2.5x** |
| | Inverse (ms/batch) | 3.3 | 14.9 | **4.5x** |

#### Validation NLL (lower is better)

| Model | flowrs | nflows |
|-------|-------:|-------:|
| MAF | **0.27** | 0.33 |
| NSF | **-0.25** | 0.37 |
| RealNVP | **-0.25** | 0.32 |

> Measured on an ARM CPU (NdArray backend) with 100 epochs, Adam lr=5e-4.
> Reproduce with `cd comparison && python python/benchmark_nflows.py && cd rust && cargo run --release --features ndarray --no-default-features && cd .. && python compare_results.py`.

### Criterion micro-benchmarks

```bash
cargo bench --features ndarray --no-default-features
```

Benchmarks forward, inverse, and log_prob for all three architectures, plus batch-size scaling for NSF. Reports are written to `target/criterion/`.

## Architecture

```
src/
├── lib.rs             # Public API and re-exports
├── maf.rs             # Masked Autoregressive Flow (+ conditional)
├── made.rs            # MADE autoregressive network (+ context injection)
├── masked_linear.rs   # Masked linear layer
├── nsf.rs             # Neural Spline Flow (composite)
├── realnvp.rs         # RealNVP (composite)
├── coupling.rs        # Affine and spline coupling layers
├── spline.rs          # Rational quadratic spline transforms
├── actnorm.rs         # Activation normalization
├── lu_linear.rs       # LU-decomposed invertible linear
├── mlp.rs             # MLP conditioner network
└── flow.rs            # Shared utilities
pyflowrs/
├── src/lib.rs         # PyO3 bindings (MAF, NSF, RealNVP)
├── Cargo.toml
└── pyproject.toml     # maturin build config
```

## License

MIT
