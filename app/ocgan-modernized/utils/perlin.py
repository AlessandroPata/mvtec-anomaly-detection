"""
Perlin noise generation for DRAEM-style synthetic anomaly masks.

Generates smooth, organic-looking anomaly masks by combining Perlin noise
at multiple octaves, producing more realistic synthetic defects than simple
CutPaste rectangles.
"""
from __future__ import annotations

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def generate_perlin_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single-octave Perlin noise field in [0, 1]."""
    h, w = shape
    delta_y, delta_x = res[0] / h, res[1] / w

    d_y = np.arange(h).reshape(-1, 1) * delta_y  # (h,1)
    d_x = np.arange(w).reshape(1, -1) * delta_x  # (1,w)

    # Grid cell coordinates (integer parts)
    y0 = d_y.astype(int)  # (h,1)
    x0 = d_x.astype(int)  # (1,w)

    # Fractional parts and fade curves
    fy = _fade(d_y - y0)  # (h,1)
    fx = _fade(d_x - x0)  # (1,w)

    # Random gradient angles on (res+1) grid
    period_y = res[0] + 1
    period_x = res[1] + 1
    angles = 2.0 * np.pi * rng.random((period_y, period_x))
    grad_y = np.sin(angles)
    grad_x = np.cos(angles)

    # Broadcast integer cell indices to (h,w) for fancy indexing
    Y0 = np.broadcast_to(y0 % period_y, (h, w))
    X0 = np.broadcast_to(x0 % period_x, (h, w))
    Y1 = np.broadcast_to((y0 + 1) % period_y, (h, w))
    X1 = np.broadcast_to((x0 + 1) % period_x, (h, w))

    dy_frac = d_y - y0  # (h,1), broadcasts with (h,w) via ops below
    dx_frac = d_x - x0  # (1,w)

    n00 = grad_y[Y0, X0] * dy_frac + grad_x[Y0, X0] * dx_frac
    n10 = grad_y[Y1, X0] * (dy_frac - 1) + grad_x[Y1, X0] * dx_frac
    n01 = grad_y[Y0, X1] * dy_frac + grad_x[Y0, X1] * (dx_frac - 1)
    n11 = grad_y[Y1, X1] * (dy_frac - 1) + grad_x[Y1, X1] * (dx_frac - 1)

    x1_lerp = _lerp(n00, n10, fy)
    x2_lerp = _lerp(n01, n11, fy)
    noise = _lerp(x1_lerp, x2_lerp, fx)

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def generate_fractal_perlin_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
    octaves: list[int] | None = None,
    threshold: float = 0.5,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.25,
) -> np.ndarray:
    """
    Generate a binary anomaly mask using multi-octave Perlin noise.

    Args:
        h, w: output mask dimensions
        rng: numpy random generator
        octaves: list of grid resolutions to sum (e.g. [2, 4, 8])
        threshold: binarization threshold (adjusted to hit target area)
        min_area_ratio: minimum fraction of image covered by the mask
        max_area_ratio: maximum fraction of image covered by the mask

    Returns:
        Binary mask of shape (h, w) with values in {0, 1}
    """
    if octaves is None:
        octaves = [2, 4, 8]

    noise = np.zeros((h, w), dtype=np.float64)
    for res in octaves:
        noise += generate_perlin_noise_2d((h, w), (res, res), rng)

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    # Adaptive thresholding to hit target area
    target_ratio = rng.uniform(min_area_ratio, max_area_ratio)
    target_pixels = int(target_ratio * h * w)

    # Find threshold that gives approximately the right area
    sorted_vals = np.sort(noise.ravel())[::-1]
    if target_pixels < len(sorted_vals):
        adaptive_threshold = sorted_vals[target_pixels]
    else:
        adaptive_threshold = sorted_vals[-1]

    mask = (noise >= adaptive_threshold).astype(np.float32)
    return mask


def generate_anomaly_texture(
    h: int,
    w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random color texture for anomaly overlay.

    Returns:
        Texture array of shape (h, w, 3) with values in [0, 1]
    """
    # Smooth random color patches using low-frequency Perlin noise
    texture = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        res = rng.integers(2, 6)
        noise = generate_perlin_noise_2d((h, w), (res, res), rng)
        texture[:, :, c] = noise.astype(np.float32)

    # Add some high-frequency detail
    detail = rng.random((h, w, 3)).astype(np.float32) * 0.3
    texture = np.clip(texture * 0.7 + detail, 0, 1)

    return texture


def create_perlin_anomaly(
    image: np.ndarray,
    rng: np.random.Generator,
    beta_range: tuple[float, float] = (0.2, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Perlin-noise-based synthetic anomaly to an image.

    Args:
        image: input image of shape (H, W, 3), values in [0, 1]
        rng: numpy random generator
        beta_range: blending range (0 = all texture, 1 = all original)

    Returns:
        (anomalous_image, binary_mask) both (H, W, ...) shaped
    """
    h, w = image.shape[:2]
    mask = generate_fractal_perlin_mask(h, w, rng)
    texture = generate_anomaly_texture(h, w, rng)

    beta = rng.uniform(*beta_range)
    mask_3d = mask[:, :, np.newaxis]

    anomalous = image * (1 - mask_3d) + (beta * image + (1 - beta) * texture) * mask_3d
    anomalous = np.clip(anomalous, 0, 1).astype(np.float32)

    return anomalous, mask
