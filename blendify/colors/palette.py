"""The :class:`PaletteColors` member of the ``Colors`` family.

Unlike a single :class:`~blendify.colors.base.Colors` container (one object's coloring
information), a palette is an **ordered sequence of colors**. :class:`PaletteColors`
therefore subclasses :data:`~blendify.colors.base.ColorsList`
(``Sequence[Colors]``): its elements are real :class:`~blendify.colors.UniformColors`
instances, so it structurally satisfies the codebase's "list of Colors" contract while
owning the color <-> id <-> tag bookkeeping used by segmentation mask rendering
(id 0 is the background, id ``i + 1`` maps to element ``i``).
"""

from typing import List, Tuple, Union

import numpy as np

from .base import ColorsList
from .common import UniformColors
from ..internal.types import Vector3d
from ..utils.color import rgb_to_uint8


Color = Union[Tuple[float, float, float], List[float]]


class PaletteColors(ColorsList):
    """An ordered palette of distinct colors with color <-> id <-> tag bookkeeping.

    Subclasses ``ColorsList`` (= ``Sequence[Colors]``): elements are
    :class:`UniformColors`, and the ``Sequence`` ABC supplies ``__iter__``,
    ``__contains__``, ``index`` and ``count`` from the implemented ``__getitem__`` /
    ``__len__``.

    The palette assigns 1-based ids in registration order (id 0 is reserved for the
    ``background_color``) and deduplicates: registering an already-known color returns
    its existing id. Tags (e.g. renderable names) can be labeled against the palette,
    populating :attr:`tag_to_id` and :attr:`tag_to_color`.

    Distinct color generation is exposed as :meth:`generate`: colors are sampled on a
    regular RGB grid, snapped to exact 8-bit values (``k/255``) so they render without
    rounding, and kept ``min_distance`` apart per channel so they map back to ids
    unambiguously after 8-bit rendering.
    """

    def __init__(self, colors: List[Color] = None, background_color: Vector3d = (0.0, 0.0, 0.0)):
        """Args:
            colors: optional initial colors (RGB in [0, 1]) registered in order.
            background_color: RGB color reserved for id 0, in [0, 1].
        """
        self.background_color = tuple(background_color)
        self._rgb: list = []      # rounded (r, g, b) tuples in [0, 1]; id i+1 -> _rgb[i]
        self._items: list = []    # parallel UniformColors elements (the Sequence content)
        self._index: dict = {}    # rounded color key -> 1-based id
        self.tag_to_id: dict = {}
        self.tag_to_color: dict = {}
        for color in (colors or []):
            self.add(color)

    # ----------------------------------------------------------------- Sequence protocol
    def __getitem__(self, index) -> UniformColors:
        """0-based access to the palette elements; the element for id ``k`` is ``self[k - 1]``."""
        return self._items[index]

    def __len__(self) -> int:
        """Number of registered colors (the background is not counted)."""
        return len(self._items)

    @property
    def colors(self) -> list:
        """The registered colors in id order (ids 1..N), as (r, g, b) tuples in [0, 1]."""
        return list(self._rgb)

    # ----------------------------------------------------------------- id bookkeeping
    def add(self, color: Color) -> int:
        """Register ``color`` in the palette and return its 1-based id.

        Deduplicates: if the color is already registered, the existing id is returned.
        New colors are stored as :class:`UniformColors` elements.
        """
        key = tuple(round(float(c), 6) for c in color[:3])
        if key not in self._index:
            self._rgb.append(key)
            self._items.append(UniformColors(key))
            self._index[key] = len(self._rgb)
        return self._index[key]

    def label(self, tag, color: Color) -> int:
        """Label ``tag`` with ``color``: register the color (see :meth:`add`) and record
        ``tag_to_id[tag]`` and ``tag_to_color[tag]`` (0-255). Returns the assigned id."""
        mask_id = self.add(color)
        self.tag_to_id[tag] = mask_id
        self.tag_to_color[tag] = rgb_to_uint8(color)
        return mask_id

    @property
    def id_to_color(self) -> dict:
        """``{id: [R, G, B]}`` (0-255) for the background (id 0) and every registered color."""
        out = {0: rgb_to_uint8(self.background_color)}
        for i, color in enumerate(self._rgb):
            out[i + 1] = rgb_to_uint8(color)
        return out

    # ----------------------------------------------------------------- color generation
    @classmethod
    def generate(
            cls, n_colors: int, min_distance: int = 8, exclude: List[Color] = None
    ) -> List[Tuple[float, float, float]]:
        """Generate ``n_colors`` visually distinct RGB tuples in [0, 1].

        Colors are drawn from a regular RGB grid (see :meth:`_distinct_grid`), snapped to
        exact 8-bit values (``k/255``) so they render without rounding, and kept at least
        ``min_distance`` per channel apart from each other and from every color in
        ``exclude`` (e.g. the background), so they map back to ids unambiguously after
        8-bit rendering.

        Args:
            n_colors: Number of colors to generate.
            min_distance: Minimum per-channel separation (0-255) between any two colors.
            exclude: Colors to stay away from (e.g. the background), in [0, 1] or 0-255.

        Returns:
            List of distinct RGB tuples in range [0, 1].
        """
        if n_colors <= 0:
            return []

        chosen = []
        # Seed `used` with excluded colors so generated colors keep their distance.
        used = [np.round(np.asarray(c, dtype=np.float64)[:3] * (255.0 if max(c[:3]) <= 1.0 else 1.0))
                for c in (exclude or [])]
        # Generate a grid fine enough to yield enough distinct candidates after dedup
        for c in cls._distinct_grid(max(n_colors * 3, 64)):
            if len(chosen) >= n_colors:
                break
            key = np.round(np.array(c) * 255.0)
            if any(np.max(np.abs(key - u)) < min_distance for u in used):
                continue
            used.append(key)
            # Snap to k/255 so the color renders without the +/-1 quantization a non-
            # representable value (e.g. 1/6) would incur.
            chosen.append(tuple(key / 255.0))
        return chosen[:n_colors]

    @staticmethod
    def _distinct_grid(n_colors: int) -> List[Tuple[float, float, float]]:
        """Generate up to ``n_colors`` evenly spaced RGB colors on a 3D grid (no black).

        Backs :meth:`generate`.
        """
        if n_colors <= 0:
            return []
        splits = max(int(np.ceil((n_colors + 1) ** (1.0 / 3.0))), 2)
        denom = splits - 1

        colors = []
        for r in range(splits):
            for g in range(splits):
                for b in range(splits):
                    color = (r / denom, g / denom, b / denom)
                    if color != (0.0, 0.0, 0.0):  # reserved for background
                        colors.append(color)
        return colors[:n_colors]
