"""Choose the yaw (0/90/180/270 degrees) that best presents a module to the judge's front cameras.

The judge compares four front views (30 degrees left/right of +Z, front from below, front from above) with the
reference photo. When the photo shows an object's side (a car in profile, a building's long face), a model built
facing +Z shows the wrong side to those cameras. This scorer compares silhouettes: the object mask of the photo
(pixels that differ from its border color) against the module's rendered masks at each yaw (rendered on a magenta
background so the model cuts out cleanly), after cropping both to their bounding boxes. Only numpy and Pillow are
used, and everything is deterministic, so replay reproduces the choice.
"""

from __future__ import annotations

import io
import re
from typing import Protocol

import numpy as np
from PIL import Image
from witness_client import RenderOptions

YAWS = (0, 90, 180, 270)
MASK_SIZE = 64
FRONT_THETAS = (30.0, 330.0, 0.0, 0.0)
FRONT_PHIS = (0.0, 0.0, 15.0, -30.0)
KEY = (255, 0, 255)


class Renderer(Protocol):
    def render(self, source: str, options: RenderOptions) -> bytes: ...


def _normalized(mask: np.ndarray) -> np.ndarray | None:
    """Crop a boolean mask to its bounding box and fit it, aspect preserved, into a centered square."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    crop = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    height, width = crop.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top: top + height, left: left + width] = crop.astype(np.uint8) * 255
    resized = Image.fromarray(square).resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def reference_mask(image: bytes) -> np.ndarray | None:
    """Object mask of the photo: pixels far from the median border color."""
    rgb = np.asarray(Image.open(io.BytesIO(image)).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                     dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))
    mask = distance > 40.0
    if mask.mean() < 0.01 or mask.mean() > 0.95:
        return None
    return _normalized(mask)


def _render_masks(renderer: Renderer, source: str) -> list[np.ndarray | None]:
    """Masks of the four front views from one grid render on a magenta background."""
    options = RenderOptions(thetas=FRONT_THETAS, phis=FRONT_PHIS, img_size=128, gap=0, bg_color="ff00ff",
                            lighting="neutral")
    grid = np.asarray(Image.open(io.BytesIO(renderer.render(source, options))).convert("RGB"), dtype=np.float32)
    masks = []
    for row in range(2):
        for col in range(2):
            tile = grid[row * 128: (row + 1) * 128, col * 128: (col + 1) * 128]
            distance = np.sqrt(((tile - np.array(KEY, dtype=np.float32)) ** 2).sum(axis=2))
            masks.append(_normalized(distance > 60.0))
    return masks


def rotated(source: str, yaw: int) -> str:
    """The module with its asset rotated about Y before the unit-cube fit (expects the miner404 wrapper)."""
    if yaw == 0:
        return source
    call = "root.add(__asset404(THREE));"
    replacement = (f"const __yaw404 = new THREE.Group(); __yaw404.rotation.y = {yaw} * Math.PI / 180; "
                   "__yaw404.add(__asset404(THREE)); root.add(__yaw404);")
    return source.replace(call, replacement, 1)


def stood_up(source: str, pitch: int) -> str:
    """The module with its asset turned about X before the fit, so a face pointing up turns toward the cameras."""
    call = "root.add(__asset404(THREE));"
    replacement = (f"const __pose404 = new THREE.Group(); __pose404.rotation.x = {pitch} * Math.PI / 180; "
                   "__pose404.add(__asset404(THREE)); root.add(__pose404);")
    return source.replace(call, replacement, 1)


def is_flat(source: str, renderer: Renderer) -> bool:
    """True when the object lies flat: its height is under half its width or depth, from front and side renders."""
    options = RenderOptions(thetas=(0.0, 90.0), phis=(0.0, 0.0), img_size=128, gap=0, bg_color="ff00ff",
                            lighting="neutral")
    grid = np.asarray(Image.open(io.BytesIO(renderer.render(source, options))).convert("RGB"), dtype=np.float32)
    distance = np.sqrt(((grid - np.array(KEY, dtype=np.float32)) ** 2).sum(axis=2)) > 60.0
    front, side = distance[:, :128], distance[:, 128:]
    if not front.any() or not side.any():
        return False
    height = np.ptp(np.flatnonzero(front.any(axis=1))) + 1
    width = np.ptp(np.flatnonzero(front.any(axis=0))) + 1
    depth = np.ptp(np.flatnonzero(side.any(axis=0))) + 1
    return bool(height < 0.5 * max(width, depth))


def silhouette_scores(image: bytes, source: str, renderer: Renderer) -> dict[int, float] | None:
    """Mean soft IoU between the photo mask and the four front-view masks, per yaw; None if not measurable."""
    target = reference_mask(image)
    if target is None:
        return None
    scores: dict[int, float] = {}
    for yaw in YAWS:
        values = []
        for mask in _render_masks(renderer, rotated(source, yaw)):
            if mask is None:
                continue
            intersection = np.minimum(mask, target).sum()
            union = np.maximum(mask, target).sum()
            values.append(float(intersection / union) if union else 0.0)
        scores[yaw] = round(sum(values) / len(values), 4) if values else 0.0
    return scores


def best_yaw(image: bytes, source: str, renderer: Renderer, margin: float = 0.04) -> int:
    """Rotate a quarter turn only when that side's silhouettes match clearly better than the model as built.

    Never 180: front and back silhouettes are nearly identical, so the mask cannot tell them apart, and the judge
    almost never prefers a back view."""
    scores = silhouette_scores(image, source, renderer)
    if not scores:
        return 0
    yaw = max((90, 270), key=lambda y: (scores[y], -y))
    return yaw if scores[yaw] - scores[0] >= margin else 0


def turned(source: str, yaw: int, pitch: int, roll: int) -> str:
    """The module with its asset turned about Y, then X, then Z before the fit (expects the miner404 wrapper)."""
    if not (yaw or pitch or roll):
        return source
    call = "root.add(__asset404(THREE));"
    replacement = ("const __turn404 = new THREE.Group(); __turn404.rotation.order = 'YXZ'; "
                   f"__turn404.rotation.set({pitch} * Math.PI / 180, {yaw} * Math.PI / 180, {roll} * Math.PI / 180); "
                   "__turn404.add(__asset404(THREE)); root.add(__turn404);")
    return source.replace(call, replacement, 1)


_MODULE_EXPORT = re.compile(r"export\s+default\s+function\s*(?:[A-Za-z_$][\w$]*\s*)?\(")


def turned_module(raw: str, yaw: int, pitch: int, roll: int) -> str:
    """A model's own module with a pose group around its object, in code the next rewrite can read and keep."""
    match = _MODULE_EXPORT.search(raw)
    if match is None:
        return raw
    inner = raw[: match.start()] + "function __posedObject(" + raw[match.end():]
    return inner + f"""

// The pose was checked against the target: keep this group unless the target clearly shows otherwise.
export default function generate(THREE) {{
  const pose = new THREE.Group();
  pose.rotation.order = 'YXZ';
  pose.rotation.set({pitch} * Math.PI / 180, {yaw} * Math.PI / 180, {roll} * Math.PI / 180);
  pose.add(__posedObject(THREE));
  return pose;
}}
"""
