#!/usr/bin/env python3
"""Check that the accelerated glass_blur is exactly equivalent to legacy code."""

import numpy as np
from PIL import Image
from skimage.filters import gaussian

from utils.imagecorruptions import corruptions as corruption_impl


SETTINGS = [
    (0.7, 1, 2),
    (0.9, 2, 1),
    (1.0, 2, 3),
    (1.1, 3, 2),
    (1.5, 4, 2),
]


def legacy_glass_blur(image, severity):
    sigma, max_delta, iterations = SETTINGS[severity - 1]
    x = np.uint8(
        gaussian(
            np.array(image) / 255.0,
            sigma=sigma,
            channel_axis=-1,
        ) * 255
    )
    x = corruption_impl._glass_blur_reference_shuffle(
        x,
        max_delta=max_delta,
        iterations=iterations,
    )
    return np.clip(
        gaussian(x / 255.0, sigma=sigma, channel_axis=-1),
        0,
        1,
    ) * 255


def rng_states_equal(lhs, rhs):
    return (
        lhs[0] == rhs[0]
        and np.array_equal(lhs[1], rhs[1])
        and lhs[2] == rhs[2]
        and lhs[3] == rhs[3]
        and lhs[4] == rhs[4]
    )


def main():
    rng = np.random.default_rng(20260918)
    image = rng.integers(
        0,
        256,
        size=(64, 67, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(image)

    seeds = (0, 1, 42, 2026)

    for severity in range(1, 6):
        for seed in seeds:
            np.random.seed(seed)
            reference = legacy_glass_blur(image, severity)
            reference_state = np.random.get_state()

            np.random.seed(seed)
            accelerated = corruption_impl.glass_blur(image, severity)
            accelerated_state = np.random.get_state()

            if not np.array_equal(reference, accelerated):
                diff = np.abs(reference - accelerated)
                raise AssertionError(
                    f"Output mismatch: severity={severity}, seed={seed}, "
                    f"max_abs_diff={diff.max()}"
                )

            if not rng_states_equal(reference_state, accelerated_state):
                raise AssertionError(
                    f"RNG-state mismatch: severity={severity}, seed={seed}"
                )

    if not corruption_impl._validate_glass_blur_fast_path():
        raise RuntimeError(
            "Exact fast path did not validate. Check that numba==0.58.1 is installed."
        )

    print(
        "PASS: accelerated glass_blur is exactly equal to the legacy "
        "implementation for all 5 severities and tested seeds, including "
        "the final NumPy RNG state."
    )


if __name__ == "__main__":
    main()
