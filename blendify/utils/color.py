"""Stateless color conversion utilities used across blendify."""

import numpy as np


def rgb_to_uint8(color) -> list:
    """Convert an RGB color in [0, 1] (or already 0-255) to a 0-255 int list.

    Heuristic: if every channel is <= 1.0, the color is assumed to be in [0, 1] and
    scaled by 255; otherwise it's assumed to already be in 0-255 and just rounded/cast.
    Used to build the ``{id: [R, G, B]}`` mapping for segmentation masks and similar.

    Args:
        color: RGB triple in [0, 1] or 0-255.

    Returns:
        list of three 0-255 ints.
    """
    arr = np.asarray(color, dtype=np.float64)[:3]
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return [int(round(v)) for v in arr]
