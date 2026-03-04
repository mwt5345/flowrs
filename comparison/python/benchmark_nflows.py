"""Benchmark nflows on two-moons: train, time forward/inverse, output JSON."""

import json
import time
from pathlib import Path

import numpy as np
import toml
import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
    ReversePermutation,
)

from two_moons_data import generate_two_moons

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_config():
    return toml.load(CONFIG_PATH)


def build_maf(cfg, d=2):
    """Build MAF matching flowrs architecture."""
    maf_cfg = cfg["maf"]
    transforms = []
    for i in range(maf_cfg["num_layers"]):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=d,
                hidden_features=maf_cfg["hidden_sizes"][0],
                num_blocks=len(maf_cfg["hidden_sizes"]),
            )
        )
        if i % 2 == 1:
            transforms.append(ReversePermutation(features=d))
        else:
            transforms.append(RandomPermutation(features=d))
    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([d])
    return Flow(transform, base_dist)


def build_nsf(cfg, d=2):
    """Build NSF matching flowrs architecture."""
    nsf_cfg = cfg["nsf"]
    transforms = []
    for i in range(nsf_cfg["num_layers"]):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=d,
                hidden_features=nsf_cfg["hidden_sizes"][0],
                num_blocks=len(nsf_cfg["hidden_sizes"]),
                num_bins=nsf_cfg["num_bins"],
                tails="linear",
                tail_bound=nsf_cfg["tail_bound"],
            )
        )
        if i % 2 == 1:
            transforms.append(ReversePermutation(features=d))
        else:
            transforms.append(RandomPermutation(features=d))
    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([d])
    return Flow(transform, base_dist)


def build_realnvp(cfg, d=2):
    """Build RealNVP matching flowrs architecture."""
    nvp_cfg = cfg["realnvp"]
    transforms = []
    for i in range(nvp_cfg["num_layers"]):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=d,
                hidden_features=nvp_cfg["hidden_sizes"][0],
                num_blocks=len(nvp_cfg["hidden_sizes"]),
            )
        )
        if i % 2 == 1:
            transforms.append(ReversePermutation(features=d))
        else:
            transforms.append(RandomPermutation(features=d))
    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([d])
    return Flow(transform, base_dist)


def train_and_benchmark(model, train_data, val_data, cfg, model_name):
    """Train model and collect metrics."""
    tcfg = cfg["training"]
    batch_size = tcfg["batch_size"]
    num_epochs = tcfg["num_epochs"]
    batches_per_epoch = tcfg["batches_per_epoch"]
    lr = tcfg["learning_rate"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)

    # Training
    model.train()
    t_start = time.perf_counter()
    total_samples = 0

    for epoch in range(1, num_epochs + 1):
        for _ in range(batches_per_epoch):
            idx = torch.randint(0, len(train_tensor), (batch_size,))
            batch = train_tensor[idx]
            optimizer.zero_grad()
            loss = -model.log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            total_samples += batch_size

    t_train = time.perf_counter() - t_start
    train_throughput = total_samples / t_train

    # Validation NLL
    model.eval()
    with torch.no_grad():
        val_log_prob = model.log_prob(val_tensor)
        val_nll = -val_log_prob.mean().item()

    # Forward (log_prob) timing
    test_batch = train_tensor[:batch_size]
    n_timing_runs = 100
    with torch.no_grad():
        # warmup
        for _ in range(10):
            model.log_prob(test_batch)

        t0 = time.perf_counter()
        for _ in range(n_timing_runs):
            model.log_prob(test_batch)
        forward_ms = (time.perf_counter() - t0) / n_timing_runs * 1000

    # Inverse (sample) timing
    with torch.no_grad():
        # warmup
        for _ in range(10):
            model.sample(batch_size)

        t0 = time.perf_counter()
        for _ in range(n_timing_runs):
            model.sample(batch_size)
        inverse_ms = (time.perf_counter() - t0) / n_timing_runs * 1000

    return {
        "model": model_name,
        "framework": "nflows",
        "train_throughput_samples_per_sec": round(train_throughput, 1),
        "forward_ms_per_batch": round(forward_ms, 3),
        "inverse_ms_per_batch": round(inverse_ms, 3),
        "final_val_nll": round(val_nll, 4),
        "train_time_sec": round(t_train, 2),
    }


def main():
    cfg = load_config()
    tcfg = cfg["training"]

    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    n_total = tcfg["n_train"] + tcfg["n_val"]
    data = generate_two_moons(n_total, noise=cfg["data"]["noise"], seed=tcfg["seed"])
    train_data = data[: tcfg["n_train"]]
    val_data = data[tcfg["n_train"] :]

    print(f"Dataset: {len(train_data)} train, {len(val_data)} val")

    results = []

    builders = [
        ("MAF", build_maf),
        ("NSF", build_nsf),
        ("RealNVP", build_realnvp),
    ]

    for name, builder in builders:
        print(f"\n=== {name} ===")
        model = builder(cfg)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        result = train_and_benchmark(model, train_data, val_data, cfg, name)
        result["param_count"] = param_count
        results.append(result)

        print(f"  Train throughput: {result['train_throughput_samples_per_sec']:.0f} samples/sec")
        print(f"  Forward: {result['forward_ms_per_batch']:.3f} ms/batch")
        print(f"  Inverse: {result['inverse_ms_per_batch']:.3f} ms/batch")
        print(f"  Val NLL: {result['final_val_nll']:.4f}")

    out_path = Path(__file__).parent.parent / "python_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
