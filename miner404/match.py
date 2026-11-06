"""How closely a module's four judge views match the photo, from pixels alone: free, deterministic, replayable.

Numpy and Pillow only.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from witness_client import RenderOptions, RenderRejected

from .orient import FRONT_PHIS, FRONT_THETAS, Renderer

VIEWS = ("30 degrees left", "30 degrees right", "from below", "from above")
TILE = 512  # the size the weights were fitted at
SIDE = 64  # crops are compared at this size, so fine detail does not decide the alignment
OVERLAY_SIDE = 256
WEIGHTS = {"ncc": 2.322, "pix_err": -0.0567, "aspect_err": -3.1964, "hist_sim": 2.3108, "contrast": 0.084, "fill": 9.7765}
LUMA = np.array([299, 587, 114], dtype=np.int64)  # thousandths
BINS = 6
# Pose search: small renders, one axis at a time. Quarter turns replace orient.best_yaw; a half turn is never offered,
# because front and back silhouettes look alike and the judge almost never prefers a back view.
SEARCH_RENDER = RenderOptions(thetas=FRONT_THETAS, phis=FRONT_PHIS, img_size=256, gap=0, bg_color="ffffff",
                              lighting="neutral")
SEARCH_AXES = {"yaw": (-90, -30, -15, 15, 30, 90), "pitch": (-15, 15, 30), "roll": (-15, 15)}
SEARCH_MARGIN = 0.15  # score an axis must gain to move at all: most modules are already posed about right
GRADE_MARGIN = 0.05


@dataclass(frozen=True)
class Appearance:
    crop: np.ndarray  # SIDE x SIDE RGB on white, uint8, object centred in a square
    mask: np.ndarray  # SIDE x SIDE, uint8
    aspect: float
    fill: float
    contrast: float
    hist: np.ndarray  # counts, int64


def _appearance(rgb: np.ndarray, mask: np.ndarray) -> Appearance | None:
    """Everything is integer arithmetic ending in one IEEE division or square root, so every machine gets the same bits:
    these numbers go into request bodies, and replay must reproduce them exactly."""
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    pixels = rgb[mask].astype(np.int64)
    if rows.size == 0 or len(pixels) * 1000 < mask.size:
        return None
    crop = rgb[rows[0]: rows[-1] + 1, columns[0]: columns[-1] + 1].copy()
    inside = mask[rows[0]: rows[-1] + 1, columns[0]: columns[-1] + 1]
    crop[~inside] = 255
    height, width = crop.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    square[top: top + height, left: left + width] = crop
    shape = np.zeros((side, side), dtype=np.uint8)
    shape[top: top + height, left: left + width] = inside.astype(np.uint8) * 255
    luma = pixels @ LUMA
    count, total = len(luma), int(luma.sum())
    contrast = math.sqrt(count * int((luma * luma).sum()) - total * total) / count / 1000
    sampled = pixels[:: max(1, count // 20000)] * BINS // 256
    hist = np.bincount((sampled[:, 0] * BINS + sampled[:, 1]) * BINS + sampled[:, 2], minlength=BINS ** 3)
    return Appearance(
        np.asarray(Image.fromarray(square).resize((SIDE, SIDE), Image.Resampling.BILINEAR)),
        np.asarray(Image.fromarray(shape).resize((SIDE, SIDE), Image.Resampling.BILINEAR)),
        width / height, len(pixels) / mask.size, contrast, hist.astype(np.int64))


def photo_appearance(image: bytes) -> Appearance | None:
    """The photo's object: pixels far from its median border color. None when the object cannot be separated."""
    photo = Image.open(io.BytesIO(image)).convert("RGB").resize((TILE, TILE), Image.Resampling.BILINEAR)
    rgb = np.asarray(photo)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    mask = ((rgb.astype(np.float64) - np.median(border, axis=0)) ** 2).sum(axis=2) > 1600.0  # exact: multiples of 0.25
    if not 0.01 < mask.mean() < 0.95:
        return None
    return _appearance(rgb, mask)


def view_appearances(grid: bytes) -> list[Appearance | None]:
    """The four front views of a gap-free 2x2 render on white."""
    image = Image.open(io.BytesIO(grid)).convert("RGB")
    width, height = image.width // 2, image.height // 2
    views = []
    for row in range(2):
        for column in range(2):
            tile = image.crop((column * width, row * height, (column + 1) * width, (row + 1) * height))
            rgb = np.asarray(tile.resize((TILE, TILE), Image.Resampling.BILINEAR))
            views.append(_appearance(rgb, (rgb < 250).any(axis=2)))
    return views


def _metrics(view: Appearance, photo: Appearance) -> dict[str, float]:
    a, b = (view.crop.astype(np.int64) @ LUMA).ravel(), (photo.crop.astype(np.int64) @ LUMA).ravel()
    count, sum_a, sum_b = a.size, int(a.sum()), int(b.sum())
    spread = (count * int((a * a).sum()) - sum_a * sum_a) * (count * int((b * b).sum()) - sum_b * sum_b)
    ncc = (count * int((a * b).sum()) - sum_a * sum_b) / math.sqrt(spread) if spread > 0 else 0.0
    errors = np.abs(view.crop.astype(np.int64) - photo.crop.astype(np.int64))
    view_total, photo_total = int(view.hist.sum()), int(photo.hist.sum())
    shared = int(np.minimum(view.hist * photo_total, photo.hist * view_total).sum())
    return {"ncc": ncc, "pix_err": int(errors.sum()) / errors.size, "aspect_err": abs(math.log(view.aspect / photo.aspect)),
            "hist_sim": shared / (view_total * photo_total), "contrast": view.contrast, "fill": view.fill}


def view_score(view: Appearance | None, photo: Appearance) -> float:
    if view is None:
        return 0.0
    return sum(WEIGHTS[name] * value for name, value in _metrics(view, photo).items())


def grid_score(grid: bytes, photo: Appearance) -> float:
    """Mean score of the four views: the same rule of thumb the fit found best for whole duels. Rounded, so a last-bit
    difference in a logarithm cannot change a comparison."""
    return round(sum(view_score(view, photo) for view in view_appearances(grid)) / 4, 6)


def feedback(grid: bytes, photo: Appearance) -> tuple[str, bytes]:
    """Measurements for a rewrite, and an overlay of the photo's outline on the closest view's outline."""
    views = view_appearances(grid)
    pairs = [(name, view) for name, view in zip(VIEWS, views, strict=True) if view is not None]
    if not pairs:
        return "", b""

    def overlap(view: Appearance) -> float:
        union = int(np.maximum(view.mask, photo.mask).astype(np.int64).sum())
        return int(np.minimum(view.mask, photo.mask).astype(np.int64).sum()) / union if union else 0.0

    closest_name, closest = max(pairs, key=lambda pair: round(view_score(pair[1], photo), 6))
    lines = [
        f"- width / height of the object: photo {photo.aspect:.2f}; "
        + ", ".join(f"{name} {view.aspect:.2f}" for name, view in pairs),
        "- outline overlap with the photo (1.00 = identical): "
        + ", ".join(f"{name} {overlap(view):.2f}" for name, view in pairs),
        f"- contrast of the object: photo {photo.contrast:.0f}; "
        + ", ".join(f"{name} {view.contrast:.0f}" for name, view in pairs),
        f"- closest view to the photo: {closest_name}; the overlay image compares it with the photo",
    ]
    canvas = np.full((SIDE, SIDE, 3), 255, dtype=np.uint8)
    canvas[photo.mask >= 128] = (224, 64, 64)
    canvas[closest.mask >= 128] = (64, 96, 224)
    canvas[(closest.mask >= 128) & (photo.mask >= 128)] = (128, 128, 128)
    overlay = Image.fromarray(canvas).resize((OVERLAY_SIDE, OVERLAY_SIDE), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    overlay.save(buffer, "PNG")
    return "\n".join(lines), buffer.getvalue()


def search_pose(source: str, photo: Appearance, renderer: Renderer,
                turned: Callable[[str, int, int, int], str]) -> tuple[tuple[int, int, int], float, float]:
    """The turn (yaw, pitch, roll in degrees) that best matches the photo, one axis at a time.

    Returns the turn, its score and the score as built. A view the renderer rejects simply scores nothing.
    """

    def scored(yaw: int, pitch: int, roll: int) -> float:
        try:
            return grid_score(renderer.render(turned(source, yaw, pitch, roll), SEARCH_RENDER), photo)
        except RenderRejected:  # a turn that fails to render is simply not chosen
            return float("-inf")

    start = scored(0, 0, 0)
    best, pose = start, {"yaw": 0, "pitch": 0, "roll": 0}
    for axis, values in SEARCH_AXES.items():
        current = best
        for value in values:
            trial = {**pose, axis: value}
            score = scored(trial["yaw"], trial["pitch"], trial["roll"])
            if score > best and score >= current + SEARCH_MARGIN:
                best, pose = score, trial
    return (pose["yaw"], pose["pitch"], pose["roll"]), best, start


GRADE_GAMMAS = (0.85, 1.15, 1.3, 1.5)  # above 1 deepens tones and raises contrast; the photo decides how far
GRADE_SATURATIONS = (1.15, 1.3)


def search_grade(source: str, photo: Appearance, renderer: Renderer, call: str) -> tuple[tuple[float, float], float, float]:
    """The tone curve (gamma, saturation) that best matches the photo, one value at a time. `call` is the wrapper's
    inert grading call. Returns the values, their score and the score as built."""

    def graded(gamma: float, saturation: float) -> str:
        return source.replace(call, f"__grade404(THREE, root, {gamma}, {saturation});", 1)

    def scored(gamma: float, saturation: float) -> float:
        try:
            return grid_score(renderer.render(graded(gamma, saturation), SEARCH_RENDER), photo)
        except RenderRejected:
            return float("-inf")

    start = best = scored(1, 1)
    gamma, saturation = 1.0, 1.0
    for value in GRADE_GAMMAS:
        score = scored(value, 1)
        if score > best + GRADE_MARGIN:
            best, gamma = score, value
    for value in GRADE_SATURATIONS:
        score = scored(gamma, value)
        if score > best + GRADE_MARGIN:
            best, saturation = score, value
    return (gamma, saturation), best, start
