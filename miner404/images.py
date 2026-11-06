"""Encode and crop images for OpenAI image inputs. Numpy and Pillow only, and deterministic for replay."""

import base64
import io

import numpy as np
from PIL import Image

CLOSEUP_SIDE = 1024
VIEW_MIN, VIEW_MAX = 512, 768
SHEET_TILE = 384


def png_data_url(image: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def data_url(image: bytes) -> str:
    kind = "jpeg" if image.startswith(b"\xff\xd8") else "png"
    return f"data:image/{kind};base64," + base64.b64encode(image).decode("ascii")


def to_jpeg(image: Image.Image | bytes) -> bytes:
    """JPEG, smaller than PNG, so many images fit the 8 MiB request limit; no chroma subsampling, to keep fine colour."""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=90, subsampling=0)
    return buffer.getvalue()


def zoomed_views(grid: bytes, columns: int = 2, rows: int = 2) -> list[bytes]:
    """Each tile of a gap-free render grid on white, cropped close to the object at the same scale for every tile.

    The object fills only a small part of the judge's frame, so this shows the model its own details at the judge's
    resolution instead of a small picture of them.
    """
    image = Image.open(io.BytesIO(grid)).convert("RGB")
    pixels = np.asarray(image)
    width, height = image.width // columns, image.height // rows
    boxes: list[tuple[int, int, int, int] | None] = []
    for row in range(rows):
        for column in range(columns):
            top, left = row * height, column * width
            mask = (pixels[top:top + height, left:left + width] < 250).any(axis=2)
            ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
            boxes.append((left + int(xs[0]), top + int(ys[0]), left + int(xs[-1]) + 1, top + int(ys[-1]) + 1)
                         if ys.size else None)
    sides = [max(box[2] - box[0], box[3] - box[1]) for box in boxes if box is not None]
    side = min(min(width, height), int(max(sides, default=min(width, height)) * 1.16) + 2)
    output = min(VIEW_MAX, max(VIEW_MIN, side))
    views = []
    for index, box in enumerate(boxes):
        row, column = divmod(index, columns)
        tile_left, tile_top = column * width, row * height
        center_x = (box[0] + box[2]) // 2 if box else tile_left + width // 2
        center_y = (box[1] + box[3]) // 2 if box else tile_top + height // 2
        left = min(max(center_x - side // 2, tile_left), tile_left + width - side)
        top = min(max(center_y - side // 2, tile_top), tile_top + height - side)
        crop = image.crop((left, top, left + side, top + side))
        if side != output:
            crop = crop.resize((output, output), Image.Resampling.LANCZOS)
        views.append(to_jpeg(crop))
    return views


def view_sheet(views: list[bytes]) -> bytes:
    """A 2x2 sheet of close-up views for comparing many candidates in one request."""
    gap = 6
    sheet = Image.new("RGB", (2 * SHEET_TILE + gap, 2 * SHEET_TILE + gap), (200, 200, 200))
    for index, view in enumerate(views):
        row, column = divmod(index, 2)
        tile = Image.open(io.BytesIO(view)).convert("RGB").resize((SHEET_TILE, SHEET_TILE), Image.Resampling.LANCZOS)
        sheet.paste(tile, (column * (SHEET_TILE + gap), row * (SHEET_TILE + gap)))
    return to_jpeg(sheet)


def object_closeup(image: bytes) -> bytes | None:
    """The photo cropped to its object, or None when the object already fills most of the photo or cannot be found."""
    photo = Image.open(io.BytesIO(image)).convert("RGB")
    small = np.asarray(photo.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    mask = np.sqrt(((small - np.median(border, axis=0)) ** 2).sum(axis=2)) > 30.0
    if not 0.002 < mask.mean() < 0.9:
        return None
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    pad = 0.08 * max(int(xs[-1] - xs[0]), int(ys[-1] - ys[0]))
    left, right = max(0.0, xs[0] - pad) / 256, min(256.0, xs[-1] + 1 + pad) / 256
    top, bottom = max(0.0, ys[0] - pad) / 256, min(256.0, ys[-1] + 1 + pad) / 256
    if (right - left) * (bottom - top) > 0.6:
        return None
    crop = photo.crop((int(left * photo.width), int(top * photo.height),
                       int(right * photo.width), int(bottom * photo.height)))
    scale = CLOSEUP_SIDE / max(crop.size)
    return to_jpeg(crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                               Image.Resampling.LANCZOS))


def judge_tiles(grid: bytes, columns: int = 2, rows: int = 2) -> list[bytes]:
    """wren-plus N2: the tiles of a gap-free render grid as they are (PNG, no crop), i.e. the judge's own stage-1 views."""
    image = Image.open(io.BytesIO(grid)).convert("RGB")
    width, height = image.width // columns, image.height // rows
    tiles = []
    for index in range(columns * rows):
        row, column = divmod(index, columns)
        buffer = io.BytesIO()
        image.crop((column * width, row * height, (column + 1) * width, (row + 1) * height)).save(buffer, "PNG")
        tiles.append(buffer.getvalue())
    return tiles
