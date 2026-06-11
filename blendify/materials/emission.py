from typing import Union

import bpy

from .base import Material, material_property, MaterialInstance
from ..internal.types import Vector3d, Vector4d


class EmissionMaterial(Material):
    """A pure emission material: renders a flat, lighting-independent color.

    Unaffected by scene lighting, shadows or reflections, so an object using it appears
    as a solid color. Used for segmentation mask rendering (see
    :meth:`blendify.scene.Scene.render_mask`) and for emissive objects in general.

    Like the other :class:`~blendify.materials.base.Material` subclasses, the color can be
    driven through the ``"Color"`` input by a :class:`~blendify.colors.base.Colors` instance,
    or baked in directly via the ``color`` argument.
    """

    def __init__(self, color: Union[Vector3d, Vector4d] = (1.0, 1.0, 1.0), strength: float = 1.0):
        """Create the emission material container.

        Args:
            color: emission color in RGB or RGBA, values in [0, 1].
            strength: emission strength (default 1.0).
        """
        super().__init__()
        self.color, self._color = material_property("color"), tuple(color)
        self.strength, self._strength = material_property("strength"), strength

    def create_material(self, name: str = "object_material") -> MaterialInstance:
        """Create the Blender emission material.

        Args:
            name (str): a unique material name for Blender

        Returns:
            MaterialInstance: the Blender material and its ``"Color"`` input socket
        """
        object_material = bpy.data.materials.new(name=name)
        object_material.use_nodes = True
        object_material.cycles.sample_as_light = False
        nodes = object_material.node_tree.nodes
        links = object_material.node_tree.links
        nodes.clear()

        emission_node = nodes.new("ShaderNodeEmission")
        output_node = nodes.new("ShaderNodeOutputMaterial")
        links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

        emission_node.inputs["Color"].default_value[:3] = self._color[:3]
        if len(self._color) == 4:
            emission_node.inputs["Color"].default_value[3] = self._color[3]
        emission_node.inputs["Strength"].default_value = self._strength

        return MaterialInstance(blender_material=object_material,
                                inputs={"Color": emission_node.inputs["Color"]})
