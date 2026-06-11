"""Small disk-IO helpers used by blendify utilities."""

import json
from pathlib import Path
from typing import Union


def save_json(path: Union[str, Path], data: dict, indent: int = 4) -> None:
    """Serialize ``data`` to ``path`` as JSON, creating parent directories if needed.

    Args:
        path: Target file path; parent directories are created if they don't exist.
        data: A JSON-serializable mapping.
        indent: Indentation for the output JSON (default 4 for human readability).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=indent)


def save_mapping(
        path: Union[str, Path], maskint_to_color: dict, tag_to_id: dict,
        extra: dict = None,
) -> None:
    """Write the segmentation mask sidecar JSON.

    Mask-specific dict shape (``{"maskint_to_color": ..., "tag_to_id": ...}``) on top
    of :func:`save_json`. Used by ``Scene._render_mask`` in debug mode.

    Args:
        path: Target ``.mask_mapping.json`` path.
        maskint_to_color: ``{id: [R, G, B]}`` in 0-255.
        tag_to_id: ``{renderable_tag: id}``.
        extra: Optional additional fields merged into the saved metadata.
    """
    metadata = {
        "maskint_to_color": {int(k): v for k, v in maskint_to_color.items()},
        "tag_to_id": dict(tag_to_id),
    }
    if extra:
        metadata.update(extra)
    save_json(path, metadata)
