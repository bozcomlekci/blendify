"""Example 09: Segmentation mask rendering on a 3D scene with many mixed shapes.

Builds a grid of 100 randomly-shaped 3D objects (cubes, spheres, cylinders, ellipsoids)
plus a point cloud (101 renderables total) viewed in perspective, and renders both a normal
lit image and three segmentation masks of the same scene:

  * instance   - a unique id (and color) per renderable (101 ids; exercises the palette
                 overflow path, since the fixed palette holds 20 colors);
  * semantic   - one id per shape category (cube/sphere/cylinder/ellipsoid/pointcloud), so
                 all renderables of the same kind share a color;
  * silhouette - all renderables as a single binary foreground mask.

The final lit ``scene.render()`` also confirms mask rendering fully restores the scene.

Masks are rendered with ``debug=True`` so the integer-id label maps and id<->color
mappings are written too; the default pipeline (``debug=False``) would only write the
(already exact) flat-color RGB masks.

Usage:
    python examples/09_mask_rendering.py --cpu

By default renderings are written to the gitignored ``output/`` directory.
"""
import argparse
import os

import numpy as np

from blendify import scene
from blendify.colors import UniformColors, VertexColors
from blendify.materials import PrincipledBSDFMaterial

SHAPES = ("cube", "sphere", "cylinder", "ellipsoid")


def add_shape(shape, material, color, x, y, tag):
    """Add one 3D primitive of the given type, resting on the z=0 ground plane."""
    if shape == "cube":
        scene.renderables.add_cube_mesh(0.9, material, color, translation=(x, y, 0.45), tag=tag)
    elif shape == "sphere":
        scene.renderables.add_sphere_nurbs(0.5, material, color, translation=(x, y, 0.5), tag=tag)
    elif shape == "cylinder":
        scene.renderables.add_cylinder_mesh(0.45, 1.1, material, color, translation=(x, y, 0.55), tag=tag)
    else:  # ellipsoid
        scene.renderables.add_ellipsoid_nurbs((0.5, 0.5, 0.75), material, color, translation=(x, y, 0.75), tag=tag)


def build_scene(resolution, grid=10, spacing=1.6, seed=0):
    """Build a ``grid`` x ``grid`` array of mixed 3D shapes plus a point cloud.

    Each renderable gets a distinct instance id and is tagged with its kind (shape name or
    ``"pointcloud"``) for the shared semantic id/color. Returns the per-tag categories.
    """
    extent = (grid - 1) * spacing
    # Perspective camera looking across the grid from the front, slightly elevated
    scene.set_perspective_camera(
        resolution, fov_x=np.deg2rad(55),
        rotation_mode="look_at", rotation=(0, 0, 0.5),
        translation=(0, -extent * 1.25, extent * 0.85),
    )

    material = PrincipledBSDFMaterial()
    rng = np.random.default_rng(seed)
    categories = {}
    for row in range(grid):
        for col in range(grid):
            shape = SHAPES[rng.integers(len(SHAPES))]
            x = col * spacing - extent / 2
            y = row * spacing - extent / 2
            color = UniformColors(tuple(rng.uniform(0.15, 0.9, 3)))
            tag = f"{shape}_{row:02d}_{col:02d}"
            add_shape(shape, material, color, x, y, tag)
            categories[tag] = shape

    # A point cloud hovering above the grid (one renderable -> one mask id). Only a single
    # point cloud is added: vanilla blendify cannot host two (their particle objects clash).
    ball = rng.uniform(-1, 1, (4000, 3))
    ball = ball[np.linalg.norm(ball, axis=1) <= 1.0] * 2.0 + np.array([0.0, 0.0, extent * 0.45 + 2.0])
    scene.renderables.add_pointcloud(
        ball, material, VertexColors(rng.uniform(0.2, 0.9, (len(ball), 3))),
        point_size=0.06, tag="cloud",
    )
    categories["cloud"] = "pointcloud"

    scene.lights.add_sun(strength=5.0, translation=(4, -6, 12))
    return categories


def main(args):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    resolution = (args.resolution, args.resolution)
    categories = build_scene(resolution, grid=args.grid)
    n_objects = len(categories)

    # Normal lit render of the 3D scene
    render_path = os.path.join(os.path.dirname(args.output) or ".", "09_render.png")
    scene.render(render_path, use_gpu=not args.cpu, samples=args.n_samples)

    # Masks via scene.render(mask=...) -- a single render entry point on Scene that
    # drives the blendify.settings.MaskContext context manager. debug=True also writes the *_ids / .json
    # artifacts; the default pipeline (debug=False) only writes the flat-color RGB mask.
    _, instance_info = scene.render(
        f"{args.output}_instance.png", mask="instance", debug=True, use_gpu=not args.cpu
    )
    print(f"Instance: {n_objects} objects -> {len(np.unique(instance_info['mask'])) - 1} ids in mask")

    scene.render(f"{args.output}_silhouette.png", mask="silhouette",
                 debug=True, use_gpu=not args.cpu)

    _, semantic_info = scene.render(
        f"{args.output}_semantic.png", mask="semantic", categories=categories,
        debug=True, use_gpu=not args.cpu,
    )
    n_categories = len(set(categories.values()))
    print(f"Semantic: {n_categories} shape categories -> {len(np.unique(semantic_info['mask'])) - 1} ids in mask")
    print(f"Wrote {render_path} and {args.output}_{{instance,silhouette,semantic}}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blendify example 09: mask rendering")
    parser.add_argument("-o", "--output", type=str, default="./output/09_mask",
                        help="output path prefix (default: ./output/09_mask)")
    parser.add_argument("-g", "--grid", type=int, default=10,
                        help="grid side length; total objects = grid^2 (default: 10 -> 100)")
    parser.add_argument("-n", "--n-samples", type=int, default=64,
                        help="number of samples for the final lit render")
    parser.add_argument("-r", "--resolution", type=int, default=512,
                        help="square render resolution (default: 512)")
    parser.add_argument("--cpu", action="store_true", help="render on CPU instead of GPU")
    main(parser.parse_args())
