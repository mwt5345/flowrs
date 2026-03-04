"""Two-moons dataset generation matching flowrs's data.rs implementation."""

import numpy as np


def generate_two_moons(n_samples: int, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    """Generate two-moons dataset.

    Args:
        n_samples: Total number of samples.
        noise: Standard deviation of Gaussian noise.
        seed: Random seed.

    Returns:
        Array of shape [n_samples, 2].
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    # Upper moon
    theta_upper = np.linspace(0, np.pi, n_half)
    x_upper = np.cos(theta_upper)
    y_upper = np.sin(theta_upper)

    # Lower moon (shifted)
    theta_lower = np.linspace(0, np.pi, n_samples - n_half)
    x_lower = 1.0 - np.cos(theta_lower)
    y_lower = -np.sin(theta_lower) + 0.5

    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])

    data = np.stack([x, y], axis=1).astype(np.float32)
    data += rng.normal(0, noise, size=data.shape).astype(np.float32)

    # Shuffle
    rng.shuffle(data)
    return data
