"""Render-time contexts for blendify's :class:`~blendify.scene.Scene`.

This module defines the :class:`Context` base — a Singleton context manager that
specific render workflows extend — and :class:`MaskContext`, the segmentation mask
context. :meth:`Scene.render` selects a context based on its arguments; the chosen
context configures Cycles and per-renderable state on ``__enter__``, the actual render
runs inside the ``with`` block, and ``__exit__`` restores everything.

Adapted from blender_datagen (https://github.com/Bozcomlekci/blender_datagen).
"""

from typing import Sequence

import bpy

from ..colors import PaletteColors, UniformColors
from ..internal import Singleton
from ..internal.types import Vector3d
from ..materials import EmissionMaterial


class Context(metaclass=Singleton):
    """Base render-time context for the :class:`~blendify.scene.Scene`.

    Singleton (one instance per concrete subclass). Subclasses provide the workflow that
    ``Scene.render`` selects based on its arguments; the base is a no-op context manager
    so subclasses can override only what they need.
    """

    @classmethod
    def instance(cls):
        """Return the cached singleton, creating it on first call.

        :class:`Singleton`'s ``__call__`` raises on the second construction, so callers
        that want "get-or-create" semantics must go through this method instead of
        ``cls()``.
        """
        if cls not in cls._instances:
            cls()
        return cls._instances[cls]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MaskContext(Context):
    """Render-time context for segmentation mask rendering.

    A singleton (per :class:`Context`). The first construction sets up empty state; each
    use is preceded by :meth:`configure` and wrapped in a ``with`` block:

    .. code-block:: python

        ctx = MaskContext.instance().configure(mode="instance")
        with ctx:
            image = scene.render(samples=1, aa_filter_width=0, use_denoiser=False)
        # ctx.palette (a PaletteColors) holds the color <-> id <-> tag bookkeeping:
        # ctx.palette.tag_to_id / .tag_to_color / .id_to_color

    ``__enter__`` snapshots Cycles render settings, applies flat/unlit settings, collects
    maskable renderables (meshes/NURBS/curves through their material slots; point clouds
    through the polymorphic ``update_material``/``update_colors``), snapshots their
    state, paints them with flat :class:`EmissionMaterial`, and writes the
    ``category_id`` custom property. ``__exit__`` restores material slots, point-cloud
    specs, deduplicated emission materials, and the original Cycles settings.

    Reference:
        blender_datagen/utils/blender_utils.py (blenderInit, mask_render branch),
        blender_datagen/utils/bprocutil.py (colorize_object).
    """

    _COLORABLE_TYPES = {'MESH', 'SURFACE', 'CURVE', 'FONT', 'META'}

    def __init__(self):
        # Per-render parameters (set via configure)
        self.mode = "instance"
        self.categories = None
        self.colors = None
        self.background_color = (0.0, 0.0, 0.0)
        self.assign_ids = True
        # Outputs / internal state
        self._reset_state()

    def _reset_state(self):
        # Color <-> id <-> tag bookkeeping for this render (public output)
        self.palette = PaletteColors(background_color=self.background_color)
        # Internal collections / snapshots
        self._items: list = []
        self._material_snapshot: dict = {}
        self._pc_snapshot: dict = {}
        self._color_materials: dict = {}
        self._saved_cycles: dict = {}
        self._entered = False

    def configure(self, mode: str = "instance", *, categories: dict = None,
                  colors: Sequence[Vector3d] = None,
                  background_color: Vector3d = (0.0, 0.0, 0.0),
                  assign_ids: bool = True):
        """Set parameters for the next mask render. Returns ``self`` so callers can
        chain ``with MaskContext.instance().configure(...) as ctx: ...``.
        """
        mode = mode.lower()
        if mode not in ("instance", "silhouette", "semantic"):
            raise ValueError(
                f"Unknown mask mode '{mode}', expected instance/silhouette/semantic"
            )
        if mode == "semantic" and not categories:
            raise ValueError(
                "semantic mode requires `categories`: a dict mapping renderable tag -> "
                "category label"
            )
        self.mode = mode
        self.categories = categories
        self.colors = colors
        self.background_color = background_color
        self.assign_ids = assign_ids
        # Clear state so a previously-used singleton starts fresh
        self._reset_state()
        return self

    # ------------------------------------------------------------------ context manager
    def __enter__(self):
        self._collect()
        self._enter_cycles_settings()
        self._entered = True
        try:
            for tag, kind, handle in self._items:
                self._snapshot(kind, handle)
            self._apply_mode()
            if self.assign_ids:
                for tag, kind, handle in self._items:
                    if tag in self.palette.tag_to_id:
                        bobj = handle if kind == "slot" else handle._blender_object
                        bobj["category_id"] = self.palette.tag_to_id[tag]
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._entered:
            return False
        # Restore slot objects (reassign original material datablocks)
        for obj, (mats, active) in self._material_snapshot.items():
            if mats:
                for i, m in enumerate(mats):
                    obj.data.materials[i] = m
            else:
                obj.data.materials.clear()
            obj.active_material = active
        self._material_snapshot.clear()
        # Restore point clouds from their retained material/colors spec
        for pc, (saved_material, saved_colors, saved_strength) in self._pc_snapshot.items():
            pc.particle_emission_strength = saved_strength
            if saved_material is not None:
                pc.update_material(saved_material)
            if saved_colors is not None:
                pc.update_colors(saved_colors)
        self._pc_snapshot.clear()
        # Remove the (deduplicated) temporary slot emission materials, now unassigned
        for mat in self._color_materials.values():
            try:
                bpy.data.materials.remove(mat)
            except (ReferenceError, RuntimeError):
                pass
        self._color_materials.clear()
        self._exit_cycles_settings()
        self._entered = False
        return False

    # ------------------------------------------------------------------ Cycles settings
    def _enter_cycles_settings(self):
        """Snapshot the Cycles render settings and apply the flat/unlit configuration."""
        scene = bpy.context.scene
        cycles = scene.cycles
        self._saved_cycles = {
            "engine": scene.render.engine,
            "samples": cycles.samples,
            "use_denoising": cycles.use_denoising,
            "use_adaptive_sampling": cycles.use_adaptive_sampling,
            "filter_width": cycles.filter_width,
            "max_bounces": cycles.max_bounces,
            "diffuse_bounces": cycles.diffuse_bounces,
            "glossy_bounces": cycles.glossy_bounces,
            "transmission_bounces": cycles.transmission_bounces,
            "volume_bounces": cycles.volume_bounces,
            "transparent_max_bounces": cycles.transparent_max_bounces,
            "view_transform": scene.view_settings.view_transform,
            "dither_intensity": scene.render.dither_intensity,
            "film_transparent": scene.render.film_transparent,
            "world_use_nodes": scene.world.use_nodes,
            "world_color": tuple(scene.world.color),
            "color_depth": scene.render.image_settings.color_depth,
            "color_mode": scene.render.image_settings.color_mode,
        }
        scene.render.engine = "CYCLES"
        cycles.samples = 1
        cycles.use_denoising = False
        cycles.use_adaptive_sampling = False
        cycles.filter_width = 0.0
        cycles.max_bounces = 0
        cycles.diffuse_bounces = 0
        cycles.glossy_bounces = 0
        cycles.transmission_bounces = 0
        cycles.volume_bounces = 0
        cycles.transparent_max_bounces = 0
        # Raw view transform + dither off => exact 8-bit colors (no off-by-one edges)
        scene.view_settings.view_transform = "Raw"
        scene.render.dither_intensity = 0.0
        # Opaque flat-color world (background is a solid id-0 color)
        scene.render.film_transparent = False
        scene.world.use_nodes = False
        scene.world.color = self.background_color
        # 8-bit RGBA PNG: exact integer colors for RGB->id post-processing
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"

    def _exit_cycles_settings(self):
        """Restore the Cycles render settings snapshotted in :meth:`_enter_cycles_settings`."""
        if not self._saved_cycles:
            return
        scene = bpy.context.scene
        cycles = scene.cycles
        s = self._saved_cycles
        scene.render.engine = s["engine"]
        cycles.samples = s["samples"]
        cycles.use_denoising = s["use_denoising"]
        cycles.use_adaptive_sampling = s["use_adaptive_sampling"]
        cycles.filter_width = s["filter_width"]
        cycles.max_bounces = s["max_bounces"]
        cycles.diffuse_bounces = s["diffuse_bounces"]
        cycles.glossy_bounces = s["glossy_bounces"]
        cycles.transmission_bounces = s["transmission_bounces"]
        cycles.volume_bounces = s["volume_bounces"]
        cycles.transparent_max_bounces = s["transparent_max_bounces"]
        scene.view_settings.view_transform = s["view_transform"]
        scene.render.dither_intensity = s["dither_intensity"]
        scene.render.film_transparent = s["film_transparent"]
        scene.world.use_nodes = s["world_use_nodes"]
        scene.world.color = s["world_color"]
        scene.render.image_settings.color_depth = s["color_depth"]
        scene.render.image_settings.color_mode = s["color_mode"]
        self._saved_cycles = {}

    # ------------------------------------------------------------------ private: setup
    def _collect(self):
        from blendify import scene  # the Scene singleton instance
        from ..renderables import PointCloud
        for tag, renderable in scene.renderables._renderables.items():
            blender_object = getattr(renderable, "_blender_object", None)
            if getattr(blender_object, "type", None) in self._COLORABLE_TYPES \
                    and hasattr(blender_object.data, "materials"):
                self._items.append((tag, "slot", blender_object))
            elif isinstance(renderable, PointCloud):
                self._items.append((tag, "pc", renderable))
        if not self._items:
            raise RuntimeError("No colorable renderables found to render a mask for")

    def _snapshot(self, kind: str, handle):
        if kind == "slot":
            self._material_snapshot[handle] = (
                [s.material for s in handle.material_slots], handle.active_material
            )
        else:
            self._pc_snapshot[handle] = (
                handle._material, handle._colors, handle.particle_emission_strength
            )

    # ------------------------------------------------------------------ private: paint
    def _emission_material(self, color):
        """Return a deduplicated emission Blender material for ``color`` (slot path)."""
        key = tuple(round(float(c), 6) for c in color[:3])
        material = self._color_materials.get(key)
        if material is None:
            material = (
                EmissionMaterial(color=color, strength=1.0)
                .create_material(name="segmentation")
                .blender_material
            )
            self._color_materials[key] = material
        return material

    @staticmethod
    def _assign_slot(obj, material):
        if len(obj.material_slots) > 0:
            for i in range(len(obj.material_slots)):
                obj.data.materials[i] = material
        else:
            obj.data.materials.append(material)
        obj.active_material = material

    def _paint(self, kind: str, handle, color):
        if kind == "slot":
            self._assign_slot(handle, self._emission_material(color))
        else:  # PointCloud: polymorphic, restored from retained spec on __exit__
            handle.particle_emission_strength = 0
            handle.update_material(EmissionMaterial(color=color))
            handle.update_colors(UniformColors(color))

    # ------------------------------------------------------------------ private: mode
    def _apply_mode(self):
        if self.mode == "instance":
            self._apply_instance()
        elif self.mode == "silhouette":
            self._apply_silhouette()
        else:
            self._apply_semantic()

    def _apply_instance(self):
        if self.colors is not None:
            if len(self.colors) < len(self._items):
                raise ValueError(
                    f"Need at least {len(self._items)} colors, got {len(self.colors)}"
                )
            obj_colors = list(self.colors)
        else:
            obj_colors = PaletteColors.generate(
                len(self._items), exclude=[self.background_color]
            )
        for (tag, kind, handle), c in zip(self._items, obj_colors):
            self.palette.label(tag, c)
            self._paint(kind, handle, c)

    def _apply_silhouette(self):
        white = (1.0, 1.0, 1.0)
        for tag, kind, handle in self._items:
            self.palette.label(tag, white)
            self._paint(kind, handle, white)

    def _apply_semantic(self):
        unique_categories = sorted(set(self.categories.values()), key=lambda c: str(c))
        category_to_color = dict(zip(
            unique_categories,
            PaletteColors.generate(
                len(unique_categories), exclude=[self.background_color]
            ),
        ))
        # Reserve ids in category order so they are deterministic.
        for category in unique_categories:
            self.palette.add(category_to_color[category])
        for tag, kind, handle in self._items:
            category = self.categories.get(tag)
            if category is None:
                self._paint(kind, handle, self.background_color)
            else:
                c = category_to_color[category]
                self.palette.label(tag, c)
                self._paint(kind, handle, c)
