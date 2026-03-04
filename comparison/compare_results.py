"""Compare flowrs and nflows benchmark results side-by-side."""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_results(path: Path) -> dict:
    """Load results JSON and index by model name."""
    with open(path) as f:
        data = json.load(f)
    return {r["model"]: r for r in data}


def fmt_speedup(rust_val, python_val, lower_is_better=True):
    """Format a speedup ratio string."""
    if lower_is_better:
        ratio = python_val / rust_val if rust_val > 0 else float("inf")
    else:
        ratio = rust_val / python_val if python_val > 0 else float("inf")

    if ratio >= 1.0:
        return f"{ratio:.1f}x faster (Rust)"
    else:
        return f"{1/ratio:.1f}x faster (Python)"


def main():
    rust_path = SCRIPT_DIR / "rust_results.json"
    python_path = SCRIPT_DIR / "python_results.json"

    if not rust_path.exists():
        print(f"Missing {rust_path}. Run the Rust benchmark first.")
        sys.exit(1)
    if not python_path.exists():
        print(f"Missing {python_path}. Run the Python benchmark first.")
        sys.exit(1)

    rust = load_results(rust_path)
    python = load_results(python_path)

    models = sorted(set(rust.keys()) & set(python.keys()))
    if not models:
        print("No matching models found between results.")
        sys.exit(1)

    # Header
    print("=" * 80)
    print("flowrs (Rust) vs nflows (Python) — Two-Moons Benchmark Comparison")
    print("=" * 80)

    for model in models:
        r = rust[model]
        p = python[model]

        print(f"\n--- {model} ---")
        print(f"{'Metric':<35} {'flowrs':>12} {'nflows':>12}   {'Comparison'}")
        print("-" * 80)

        # Training throughput (higher is better)
        rt = r["train_throughput_samples_per_sec"]
        pt = p["train_throughput_samples_per_sec"]
        cmp = fmt_speedup(1 / rt, 1 / pt, lower_is_better=True)
        print(f"{'Train throughput (samples/sec)':<35} {rt:>12.0f} {pt:>12.0f}   {cmp}")

        # Forward latency (lower is better)
        rf = r["forward_ms_per_batch"]
        pf = p["forward_ms_per_batch"]
        cmp = fmt_speedup(rf, pf)
        print(f"{'Forward latency (ms/batch)':<35} {rf:>12.3f} {pf:>12.3f}   {cmp}")

        # Inverse latency (lower is better)
        ri = r["inverse_ms_per_batch"]
        pi = p["inverse_ms_per_batch"]
        cmp = fmt_speedup(ri, pi)
        print(f"{'Inverse latency (ms/batch)':<35} {ri:>12.3f} {pi:>12.3f}   {cmp}")

        # Val NLL (lower is better, but small differences don't matter much)
        rn = r["final_val_nll"]
        pn = p["final_val_nll"]
        diff = rn - pn
        direction = "(lower is better)" if diff > 0 else "(Rust is better)" if diff < 0 else "(equal)"
        print(f"{'Val NLL':<35} {rn:>12.4f} {pn:>12.4f}   delta={diff:+.4f} {direction}")

        # Train time
        rtt = r.get("train_time_sec", 0)
        ptt = p.get("train_time_sec", 0)
        if rtt > 0 and ptt > 0:
            cmp = fmt_speedup(rtt, ptt)
            print(f"{'Train time (sec)':<35} {rtt:>12.2f} {ptt:>12.2f}   {cmp}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
