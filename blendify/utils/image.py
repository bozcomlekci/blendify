"""Stateless image utility functions used across blendify."""

import numpy as np

from ..internal.types import Vector3d


def blend_with_background(img: np.ndarray, bkg_color: Vector3d = (1., 1., 1.)) -> np.ndarray:
    """Blend the RGBA image with a uniform colored background, return RGB image.

    Args:
        img: RGBA foreground image.
        bkg_color: RGB uniform background color (default is white).

    Returns:
        np.ndarray: RGB image blended with the background.
    """
    bkg_color = np.array(bkg_color)
    if img.dtype == np.uint8:
        bkg_color_uint8 = (bkg_color * 255).astype(np.uint8)
        alpha = img[:, :, 3:4].astype(np.int32)
        img_with_bkg = ((img[:, :, :3] * alpha
                         + bkg_color_uint8[None, None, :] * (255 - alpha)) // 255).astype(np.uint8)
    else:
        alpha = img[:, :, 3:4]
        img_with_bkg = (img[:, :, :3] * alpha + bkg_color[None, None, :] * (1. - alpha))
    return img_with_bkg


def rgba_to_labels(image: np.ndarray, id_to_color: dict) -> np.ndarray:
    """Convert an RGB(A) image to an integer-label map using an ``{id: [R, G, B]}``
    mapping.

    The image is assumed to use exact 8-bit colors from the mapping (the segmentation
    mask render pipeline guarantees this). Matching is done via a 1-D code lookup
    (``r << 16 | g << 8 | b``) over the distinct colors actually present, so the cost
    is O(H*W) for the scan plus O(unique * palette) for the lookup. Unrecognized pixels
    fall back to id 0.

    Args:
        image: Rendered image as an (H, W, 3) or (H, W, 4) array in 0-255.
        id_to_color: Mapping ``{id: [R, G, B]}`` in 0-255 (id 0 = background).

    Returns:
        (H, W) integer label map.
    """
    rgb = image[..., :3].astype(np.int64)
    codes = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    code_to_id = {(c[0] << 16) | (c[1] << 8) | c[2]: int(idx)
                  for idx, c in id_to_color.items()}
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    mapped = np.array([code_to_id.get(int(code), 0) for code in unique_codes],
                      dtype=np.int32)
    return mapped[inverse].reshape(codes.shape)
