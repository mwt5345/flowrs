# flowrs vs nflows Performance Comparison

Head-to-head comparison of [flowrs](../) (Rust/Burn) against
[nflows](https://github.com/bayesiains/nflows) (Python/PyTorch) on the
two-moons density estimation task.

## Setup

### Python (nflows)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Rust (flowrs)

```bash
cd rust
cargo build --release --features ndarray --no-default-features
```

## Running

```bash
# Python benchmark → python_results.json
python python/benchmark_nflows.py

# Rust benchmark → rust_results.json
cd rust && cargo run --release --features ndarray --no-default-features
cd ..

# Compare
python compare_results.py
```

## Metrics

| Metric | Description |
|--------|-------------|
| `train_throughput_samples_per_sec` | Training samples processed per second |
| `forward_ms_per_batch` | Forward / log_prob inference latency (ms) |
| `inverse_ms_per_batch` | Inverse / sampling inference latency (ms) |
| `final_val_nll` | Validation negative log-likelihood at end of training |

## Fairness Notes

- Both frameworks use the **same** architecture (layers, hidden sizes, bins)
  as defined in `config.toml`.
- Both use Adam with the same learning rate and batch size.
- Rust uses the NdArray (CPU) backend for a fair single-threaded comparison
  against PyTorch CPU. Set `FLOW_BACKEND=wgpu` or use `--features wgpu` for
  GPU benchmarks (compare against PyTorch CUDA accordingly).
- Timings exclude data loading and initialization; only forward/backward
  passes and inference are timed.
- nflows uses its own spline implementation; flowrs uses a from-scratch
  implementation. Minor numerical differences are expected.
