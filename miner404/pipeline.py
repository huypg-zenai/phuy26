"""Image to Three.js: candidates narrowed by selection and rewritten against their own renders.

Per prompt (the plan is fixed per round; see StagePlan and versions.py):
  1. Drafts: independent modules generated concurrently, then cut down by the selector, so a bad draw is dropped early.
  2. Rewrites: each survivor is rewritten from the photo, a close-up of the object, its own renders at the judge's
     resolution and its code. After each round the selector keeps the best of parents and children; a parent must beat
     a child by a clear margin to stay.
  3. Selection: Sol scores candidates twice, in opposite presentation orders, on the judge's four front views and the
     judge's second-stage grid (wren-plus: rendered in neutral light, as 404's judge since 05/11, render.py:236).
  4. Pose: a quarter turn (photo silhouette) and, for flat objects, standing up are offered to the selector, and win
     only by a clear margin.
Every module is wrapped (99% unit-cube fit, small tilt for flat objects, draw-call merge, texture shrink) and validated
with the renderer; a rejected one gets one repair call.
The prompt never exceeds the call or credit limits: every stage reserves its worst case before dispatching.
A failed model call stops the prompt (see CallFailed).

Determinism: every budget decision is made on the main thread from settled usage before a stage dispatches; calls
inside a stage run concurrently but are gathered in a fixed order; renders run sequentially in candidate order;
random choices are seeded from the round seed and the image hash.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Unpack, cast

from openai import APIConnectionError, APIError, APITimeoutError
from openai.types.responses import Response, ResponseInputParam
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from witness_client import ExchangePending, RenderOptions, RenderRejected, RenderUnavailable, WitnessError

from . import match, orient, patch, prompts
from .budget import Budget, worst_case
from .fallback import fallback_module
from .images import data_url, judge_tiles, object_closeup, png_data_url, to_jpeg, view_sheet, zoomed_views
from .orient import Renderer

SOL = "gpt-6.1-sol"
ASTRA = "gpt-6-astra"
GENERATE_EFFORT = "high"
SELECT_EFFORT = "medium"
GENERATE_MAX_OUTPUT = 32768
SELECT_MAX_OUTPUT = 8192
GARBAGE_PENALTY = 2.0  # added when at least three side views look like garbage (the judge's side guard)
ASTRA_MAX_OUTPUT = 22000  # five times Sol per token, and the budget guard reserves the whole cap before each call
EFFORT_MAX_OUTPUT = {"xhigh": 44000, "max": 64000}  # reasoning counts against the cap
GRID_WEIGHT = 0.25  # the grid only decides duels the front views leave close
PARENT_MARGIN = 0.25  # mean penalty points a parent must lead by
POSE_PITCHES = (45, 90)  # for a photo from above, and for a face shown head-on
POSE_MARGIN = 0.5  # mean penalty points a new pose must gain; the selector is noisy and most poses are already right
JOIN_ATTEMPTS = 6  # a repeated request joins a capture that outlasted the client's wait, without paying again
RENDER_WAITS = (5, 10, 20, 40, 60, 60)  # the Pod has one render slot and a two-minute queue limit
RENDER_TIMEOUT_RETRIES = 2  # the renderer answers HTTP 500 when a module misses its 30-second deadline
# Request ids are fixed per plan step, so a retried prompt reuses its saved calls.
DRAFT_ID = 1
REDRAFT_ID = 50
REWRITE_BASE = 100  # step i, survivor s, rewrite m: 100 * (i + 1) + 10 * s + m
REPAIR_OFFSET = 5000
SELECT_ID = 9000  # plus 10 * (step + 1); the mirrored call takes the next id
FINAL_SELECT_ID = 9800
POSE_ID = 9900
BRIEF_ID = 9990
# wren-plus N2: the second gate of every keep-1 step = 404's stage-1 judge prompt on the judge's own front views, child vs
# parent, both orders x 4 views = 8 Sol-low calls (ids DUEL_ID + 10 * step + k); the child stays only if its summed
# penalty is LOWER than its parent's (a tie or an unusable reply keeps the parent).
DUEL_ID = 9400
FINAL_DUEL_ID = 9700  # beat-all L3: final duel, ids 9700-9707
DUEL_EFFORT = "low"
DUEL_MAX_OUTPUT = 4096
BRIEF_MAX_OUTPUT = 4096
MEASURED_ASTRA_MAX_OUTPUT = 18000  # a lower cap reserves less, so late rewrites are not refused
MATCH_WEIGHT = 0.5  # penalty points per unit of match score
POSE_SEARCH_GAIN = 0.5  # match score a turn must gain to be offered to the selector at all

# The judge's Stage 1 front views and Stage 4 side views (shared/subnet_common/render.py), white, neutral light.
FRONT = RenderOptions(thetas=(30, 330, 0, 0), phis=(0, 0, 15, -30), img_size=384, bg_color="ffffff",
                      lighting="neutral")
SIDES = RenderOptions(thetas=(90, 180, 270, 0), phis=(0, 0, 0, -90), img_size=256, bg_color="ffffff",
                      lighting="neutral")
# The judge's own pixels. A module that renders here within the deadline should also pass the final save's renders.
FRONT_JUDGE = RenderOptions(thetas=(30, 330, 0, 0), phis=(0, 0, 15, -30), img_size=1024, gap=0, bg_color="ffffff",
                            lighting="neutral")
# wren-plus: 404's judge renders its stage-2 grid with lighting=neutral since 05/11 (shared/subnet_common/render.py:236,
# commit d41cc70); the renderer's default is still "studio", so ask for neutral to see what the judge sees.
GRID_VIEW = RenderOptions(lighting="neutral")

# Calls the model's function directly: the validator rejects passing THREE through an aliased callee.
# Over the 200 draw-call limit, meshes sharing a material are baked into one mesh (same look, fewer draws);
# over the 4 MB texture-data limit, the largest textures are halved until the module fits.
WRAPPER = """
function __drawCalls404(root) {
  let count = 0;
  root.traverse(function (node) { if (node.isMesh || node.isLine || node.isPoints) count += 1; });
  return count;
}

function __mergeByMaterial404(THREE, root) {
  root.updateMatrixWorld(true);
  const byMaterial = new Map();
  const groups = [];
  root.traverse(function (node) {
    if (!node.isMesh || node.isInstancedMesh || !node.visible || Array.isArray(node.material)) return;
    const geometry = node.geometry;
    if (!geometry || !geometry.attributes.position) return;
    if (geometry.morphAttributes && Object.keys(geometry.morphAttributes).length) return;
    const names = Object.keys(geometry.attributes).sort();
    const key = names.map(function (n) { return n + ':' + geometry.attributes[n].itemSize; }).join(',');
    if (!byMaterial.has(node.material)) byMaterial.set(node.material, new Map());
    const byLayout = byMaterial.get(node.material);
    if (!byLayout.has(key)) { const group = { names: names, meshes: [] }; byLayout.set(key, group); groups.push(group); }
    byLayout.get(key).meshes.push(node);
  });
  const vector = new THREE.Vector3();
  const normalMatrix = new THREE.Matrix3();
  const merged = [];
  for (const group of groups) {
    const meshes = group.meshes;
    if (meshes.length < 2) continue;
    let vertexCount = 0;
    let indexCount = 0;
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      vertexCount += geometry.attributes.position.count;
      indexCount += geometry.index ? geometry.index.count : geometry.attributes.position.count;
    }
    const arrays = {};
    for (const name of group.names) arrays[name] = new Float32Array(vertexCount * meshes[0].geometry.attributes[name].itemSize);
    const index = vertexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
    let vertexOffset = 0;
    let indexOffset = 0;
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      const matrix = mesh.matrixWorld;
      normalMatrix.getNormalMatrix(matrix);
      const count = geometry.attributes.position.count;
      for (const name of group.names) {
        const attribute = geometry.attributes[name];
        const size = attribute.itemSize;
        const out = arrays[name];
        for (let i = 0; i < count; i++) {
          const base = (vertexOffset + i) * size;
          if (name === 'position' || name === 'normal') {
            vector.fromBufferAttribute(attribute, i);
            if (name === 'position') vector.applyMatrix4(matrix); else vector.applyNormalMatrix(normalMatrix);
            out[base] = vector.x; out[base + 1] = vector.y; out[base + 2] = vector.z;
          } else {
            out[base] = attribute.getX(i);
            if (size > 1) out[base + 1] = attribute.getY(i);
            if (size > 2) out[base + 2] = attribute.getZ(i);
            if (size > 3) out[base + 3] = attribute.getW(i);
          }
        }
      }
      const flip = matrix.determinant() < 0;
      const triangles = geometry.index ? geometry.index.count : count;
      for (let i = 0; i < triangles; i++) {
        const source = flip ? i - (i % 3) + [0, 2, 1][i % 3] : i;
        index[indexOffset + i] = (geometry.index ? geometry.index.getX(source) : source) + vertexOffset;
      }
      vertexOffset += count;
      indexOffset += triangles;
    }
    const combined = new THREE.BufferGeometry();
    for (const name of group.names) {
      combined.setAttribute(name, new THREE.BufferAttribute(arrays[name], meshes[0].geometry.attributes[name].itemSize));
    }
    combined.setIndex(new THREE.BufferAttribute(index, 1));
    for (const mesh of meshes) mesh.parent.remove(mesh);
    merged.push(new THREE.Mesh(combined, meshes[0].material));
  }
  for (const mesh of merged) root.add(mesh);
}

function __textures404(root) {
  const found = new Map();
  root.traverse(function (node) {
    const materials = node.material ? [].concat(node.material) : [];
    for (const material of materials) {
      for (const key of Object.keys(material)) {
        const value = material[key];
        if (value && value.isDataTexture && value.image && value.image.data) {
          if (!found.has(value)) found.set(value, []);
          found.get(value).push([material, key]);
        }
      }
    }
  });
  return found;
}

function __shrinkTextures404(THREE, root, limit) {
  for (let pass = 0; pass < 8; pass++) {
    const found = __textures404(root);
    let total = 0;
    let largest = null;
    for (const texture of found.keys()) {
      total += texture.image.data.byteLength;
      if (!largest || texture.image.data.byteLength > largest.image.data.byteLength) largest = texture;
    }
    if (total <= limit || !largest || largest.image.width < 16 || largest.image.height < 16) return;
    const width = largest.image.width;
    const height = largest.image.height;
    const data = largest.image.data;
    const channels = Math.round(data.length / (width * height));
    const halfWidth = Math.floor(width / 2);
    const halfHeight = Math.floor(height / 2);
    const size = halfWidth * halfHeight * channels;
    const out = data instanceof Float32Array ? new Float32Array(size)
      : data instanceof Uint16Array ? new Uint16Array(size) : new Uint8Array(size);
    for (let y = 0; y < halfHeight; y++) {
      for (let x = 0; x < halfWidth; x++) {
        for (let c = 0; c < channels; c++) {
          const a = ((2 * y) * width + 2 * x) * channels + c;
          const b = a + channels;
          const d = a + width * channels;
          out[(y * halfWidth + x) * channels + c] = (data[a] + data[b] + data[d] + data[d + channels]) / 4;
        }
      }
    }
    const smaller = new THREE.DataTexture(out, halfWidth, halfHeight, largest.format, largest.type);
    smaller.wrapS = largest.wrapS;
    smaller.wrapT = largest.wrapT;
    smaller.magFilter = largest.magFilter;
    smaller.minFilter = largest.minFilter;
    smaller.generateMipmaps = largest.generateMipmaps;
    smaller.colorSpace = largest.colorSpace;
    smaller.flipY = largest.flipY;
    smaller.repeat.copy(largest.repeat);
    smaller.offset.copy(largest.offset);
    smaller.needsUpdate = true;
    for (const [material, key] of found.get(largest)) material[key] = smaller;
  }
}

export default function generate(THREE) {
  const root = new THREE.Group();
  root.add(__asset404(THREE));
  if (__drawCalls404(root) > 200) __mergeByMaterial404(THREE, root);
  __shrinkTextures404(THREE, root, 4100000);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (extent > 0 && Number.isFinite(extent)) {
    const center = box.getCenter(new THREE.Vector3());
    const scale = 0.95 / extent;
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  }
  return root;
}
"""
# Flat objects otherwise show the front cameras a thin edge. Applied before the fit, so the fit includes the tilt.
_TILT = """  const flat = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3());
  if (flat.y < 0.5 * Math.max(flat.x, flat.z)) root.rotation.x = 15 * Math.PI / 180;
"""
_FIT = "  const box = new THREE.Box3().setFromObject(root);\n"


_GLOW_FUNCTION = """
function __lift404(root, amount) {
  const seen = new Set();
  root.traverse(function (node) {
    for (const material of (node.material ? [].concat(node.material) : [])) {
      if (seen.has(material) || !material.isMeshStandardMaterial || material.vertexColors) continue;
      seen.add(material);
      material.emissive.multiplyScalar(material.emissiveIntensity).add(material.color.clone().multiplyScalar(amount));
      material.emissiveIntensity = 1;
      if (material.map && !material.emissiveMap) material.emissiveMap = material.map;
      material.color.multiplyScalar(1 - 1.6 * amount);
    }
  });
}
"""
_EXPORT_LINE = "export default function generate(THREE) {\n"
# The validator's box is not per-vertex: it wraps each part's own box after rotating it, so a rotated part inflates the box
# and the fit then shrinks the whole object (an apple turned 40 degrees renders at half the area). Moving every transform
# into the vertices makes that box snug, so the same model renders larger and still passes.
_OUTLINE_FUNCTION = """
function __bake404(THREE, root) {
  root.updateMatrixWorld(true);
  let vertices = 0;
  root.traverse(function (node) {
    if (node.geometry && node.geometry.attributes.position && !node.isInstancedMesh) vertices += node.geometry.attributes.position.count;
  });
  if (vertices > 200000) return;
  const instance = new THREE.Matrix4();
  const offset = new THREE.Matrix4();
  const places = new Map();
  root.traverse(function (node) {
    const isShape = node.isMesh || node.isLine || node.isPoints;
    if (!isShape || !node.geometry || !node.geometry.attributes.position) return;
    // Each part keeps its own position: three.js sorts transparent parts by it, so glass still draws in the same order.
    const place = new THREE.Vector3().setFromMatrixPosition(node.matrixWorld);
    places.set(node, place);
    offset.makeTranslation(-place.x, -place.y, -place.z);
    if (node.isInstancedMesh) {
      for (let i = 0; i < node.count; i++) {
        node.getMatrixAt(i, instance);
        node.setMatrixAt(i, instance.premultiply(node.matrixWorld).premultiply(offset));
      }
      node.instanceMatrix.needsUpdate = true;
      node.boundingBox = null;
      node.boundingSphere = null;
      return;
    }
    const geometry = node.geometry.clone();
    geometry.applyMatrix4(node.matrixWorld);
    geometry.applyMatrix4(offset);
    if (node.isMesh && node.matrixWorld.determinant() < 0) {
      const count = geometry.index ? geometry.index.count : geometry.attributes.position.count;
      const order = new Uint32Array(count - (count % 3));
      for (let i = 0; i < order.length; i++) {
        const corner = i - (i % 3) + [0, 2, 1][i % 3];
        order[i] = geometry.index ? geometry.index.getX(corner) : corner;
      }
      geometry.setIndex(new THREE.BufferAttribute(order, 1));
    }
    node.geometry = geometry;
  });
  // wren-plus fix: every baked part moves directly under root. Left under a baked parent (mesh inside a mesh) it would
  // get the parent's place added a second time. A part under a hidden group stays hidden; updateMatrix() also covers
  // parts built with matrixAutoUpdate = false, whose matrix would otherwise keep the old transform on top of the bake.
  const shapes = [];
  root.traverse(function (node) { if (places.has(node)) shapes.push(node); });
  for (const node of shapes) {
    for (let up = node.parent; up; up = up.parent) if (!up.visible) node.visible = false;
    if (node.parent !== root) { node.parent.remove(node); root.add(node); }
  }
  root.traverse(function (node) {
    const place = places.get(node);
    if (place) node.position.copy(place); else node.position.set(0, 0, 0);
    node.quaternion.identity();
    node.scale.set(1, 1, 1);
    node.updateMatrix();
  });
  root.updateMatrixWorld(true);
}
"""
# Tone grading: every color, color texture and vertex color goes through one curve, the same in linear and sRGB values
# because a power curve survives the conversion. match.search_grade picks the values against the photo, with free renders.
_GRADE_FUNCTION = """
function __grade404(THREE, root, gamma, saturation) {
  if (gamma === 1 && saturation === 1) return;
  const curve = function (r, g, b) {
    const R = Math.pow(Math.max(r, 0), gamma);
    const G = Math.pow(Math.max(g, 0), gamma);
    const B = Math.pow(Math.max(b, 0), gamma);
    const luma = 0.2126 * R + 0.7152 * G + 0.0722 * B;
    return [luma + (R - luma) * saturation, luma + (G - luma) * saturation, luma + (B - luma) * saturation].map(function (v) {
      return Math.min(1, Math.max(0, v));
    });
  };
  const done = new Set();
  const gradeAttribute = function (attribute) {
    if (!attribute || done.has(attribute)) return;
    done.add(attribute);
    for (let i = 0; i < attribute.count; i++) {
      const c = curve(attribute.getX(i), attribute.getY(i), attribute.getZ(i));
      attribute.setXYZ(i, c[0], c[1], c[2]);
    }
    attribute.needsUpdate = true;
  };
  const gradeTexture = function (texture) {
    if (!texture || !texture.isDataTexture || done.has(texture) || !texture.image || !(texture.image.data instanceof Uint8Array)) return;
    done.add(texture);
    const data = texture.image.data;
    const channels = Math.round(data.length / (texture.image.width * texture.image.height));
    if (channels < 3) return;
    for (let i = 0; i + 2 < data.length; i += channels) {
      const c = curve(data[i] / 255, data[i + 1] / 255, data[i + 2] / 255);
      data[i] = Math.round(c[0] * 255);
      data[i + 1] = Math.round(c[1] * 255);
      data[i + 2] = Math.round(c[2] * 255);
    }
    texture.needsUpdate = true;
  };
  root.traverse(function (node) {
    for (const material of (node.material ? [].concat(node.material) : [])) {
      if (!done.has(material)) {
        done.add(material);
        if (material.color) {
          const c = curve(material.color.r, material.color.g, material.color.b);
          material.color.setRGB(c[0], c[1], c[2]);
        }
        gradeTexture(material.map);
      }
      if (material.vertexColors && node.geometry) gradeAttribute(node.geometry.attributes.color);
    }
    if (node.isInstancedMesh && node.instanceColor) gradeAttribute(node.instanceColor);
  });
}
"""
GRADE_CALL = "__grade404(THREE, root, 1, 1);"
_OUTLINE_TILT = """  __grade404(THREE, root, 1, 1);
  const flat = new THREE.Box3().setFromObject(root, true).getSize(new THREE.Vector3());
  if (flat.y < 0.5 * Math.max(flat.x, flat.z)) root.rotation.x = 15 * Math.PI / 180;
  __bake404(THREE, root);
"""


def chain_wrapper(fill: float, glow: float = 0.0, outline_fit: bool = False) -> str:
    """The fit and tilt wrapper. `glow` makes part of each color independent of the light, so it survives the grid's dim
    light, and darkens the base color to keep the bright views about the same. `outline_fit` fits the object's real outline."""
    wrapper = WRAPPER.replace(_FIT, _TILT + _FIT, 1).replace("0.95 / extent", f"{fill} / extent", 1)
    if outline_fit:
        wrapper = wrapper.replace(_TILT, _OUTLINE_TILT, 1).replace(_EXPORT_LINE, _OUTLINE_FUNCTION + _GRADE_FUNCTION + "\n" + _EXPORT_LINE, 1)
    if glow:
        wrapper = wrapper.replace(_TILT, f"  __lift404(root, {glow});\n" + _TILT, 1)
        wrapper = wrapper.replace(_EXPORT_LINE, _GLOW_FUNCTION + "\n" + _EXPORT_LINE, 1)
    return wrapper


_EXPORT = re.compile(r"export\s+default\s+function\s*(?:[A-Za-z_$][\w$]*\s*)?\(")


class ModelCalls(Protocol):
    def create(self, *, request_id: int = 0, **params: Unpack[ResponseCreateParamsBase]) -> Response: ...


class PatientRenderer:
    """Waits out a busy renderer. A module that keeps missing the render deadline counts as a rejected module, so the
    model can make it lighter; otherwise its prompt could never finish, and a missing prompt loses its duels."""

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    def render(self, source: str, options: RenderOptions) -> bytes:
        waits = iter(RENDER_WAITS)
        timeouts = 0
        while True:
            try:
                return self._renderer.render(source, options)
            except RenderRejected as rejection:
                # A module near the 5-second limit can pass on one run and fail on the next; replay must not flip.
                timeouts += 1
                if "TIMEOUT_EXCEEDED" not in rejection.detail_json or timeouts > RENDER_TIMEOUT_RETRIES:
                    raise
            except RenderUnavailable as error:
                response = getattr(error.__cause__, "response", None)
                if getattr(response, "status_code", None) == 500:
                    timeouts += 1
                    if timeouts > RENDER_TIMEOUT_RETRIES:
                        detail = "RENDER_TIMEOUT: the renderer could not finish this module in time. Make it lighter: "
                        raise RenderRejected(json.dumps(
                            {"error": detail + "fewer meshes, fewer vertices, lower segment counts."})) from error
                wait = next(waits, None)
                if wait is None:
                    raise
                time.sleep(wait)


class CallFailed(Exception):
    """A model call ended without a saved response: an HTTP or stream error (rate and usage limits included), or a reply
    without usage. Replay returns saved responses only, so code that carried on without this call could not replay.
    The prompt stops here; the runner tries it again, reusing saved calls and retrying this one under the same id."""


@dataclass
class Candidate:
    name: str
    source: str
    front: bytes
    sides: bytes  # in staged plans, the judge's second-stage grid
    raw: str = ""  # the model's own module, before wrapping
    views: list[bytes] = field(default_factory=list)  # close-ups of the four front views
    parent: Candidate | None = None
    match: float = 0.0  # how closely the views match the photo, in measured plans
    measurements: str = ""
    overlay: bytes = b""
    judge_front: bytes = b""  # wren-plus N2: the FRONT_JUDGE render as it is (the duel's 4 views)


@dataclass
class Judgement:
    score: float
    fixes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Plan:
    """Fixed per-prompt call plan (the runtime forbids adapting to the clock, so it is chosen per round)."""

    drafts: tuple[tuple[str, str, str], ...]  # (model, reasoning effort, "base" | "handbook" instructions)
    draft_max_output: int = 16384
    group_size: int = 4  # candidates per judged group in the selection tournament
    orient: bool = True
    refine_rounds: int = 1
    refine_focuses: tuple[str, ...] = ("blind", "shape")
    refine_effort: str = "medium"


@dataclass(frozen=True)
class Draft:
    start: str  # "plain" | "proportions"
    model: str
    effort: str


@dataclass(frozen=True)
class Step:
    """One round of rewrites. With no rewrites at all, it only picks among each lineage's last two modules."""

    rewrites: tuple[tuple[str, str], ...] = ()  # (model, effort) of each rewrite of every survivor
    lineage: tuple[tuple[str, str], ...] = ()  # instead: one rewrite per survivor, by survivor position
    keep: int = 0  # survivors the selector keeps of parents and children; 0 = each lineage's newest module, unjudged
    focus: tuple[str, ...] = ()  # keys of prompts.REWRITE_FOCUS, by survivor position
    # beat-all L3: every lineage keeps its own best of parent and children (one panel judges them all), so the survivors
    # stay independent lineages to the end instead of collapsing into one.
    per_lineage: bool = False
    # beat-all L3 (idea I3): rewrites answer with SEARCH/REPLACE edits applied to the parent's module (patch.py, from
    # 3d-gen-sub); a reply whose edits all fail to apply is dropped (the parent stays), a whole-module reply is used as is.
    patch: bool = False


@dataclass(frozen=True)
class StagePlan:
    drafts: tuple[Draft, ...]
    steps: tuple[Step, ...]
    draft_keep: int = 0  # survivors of the draft selection; 0 = every valid draft goes on
    pose: bool = True
    fill: float = 0.99
    glow: float = 0.0  # see chain_wrapper
    # The photo brief, pixel measurements in rewrites and selection, and the pose search (see match.py).
    measured: bool = False
    # Fit the object's real outline, so rotated parts do not shrink it.
    outline_fit: bool = False
    # The same fit, applied only to the finished module: the chain runs without it.
    outline_fit_last: bool = False
    # beat-all L3: the final pick among the lineages = panel top two, then an 8-call stage-1 copy duel between them.
    final_duel: bool = False


@dataclass
class Call:
    request_id: int
    model: str
    instructions: str
    content: list[dict[str, Any]]
    effort: str
    max_output_tokens: int
    text: dict[str, Any] | None = None
    estimate: float = 0.0


def extract_source(text: str) -> str | None:
    """JavaScript from a model reply: the whole reply, or the last fenced block that exports generate."""
    text = text.strip()
    blocks = [b for b in re.findall(r"```(?:javascript|js|mjs)?[^\n]*\n([\s\S]*?)```", text) if _EXPORT.search(b)]
    if blocks:
        text = blocks[-1].strip()
    return text if _EXPORT.search(text) else None


def normalize(source: str, wrapper: str = WRAPPER) -> str:
    """Wrap the module so its asset is centered and scaled to fill the unit cube, whatever its scale."""
    match = _EXPORT.search(source)
    if match is None:
        return source
    inner = source[: match.start()] + "function __asset404(" + source[match.end():]
    return inner + "\n" + wrapper


class Pipeline:
    def __init__(self, image: bytes, seed: int, calls: ModelCalls, renderer: Renderer) -> None:
        self.image = image
        self.seed = seed
        self.calls = calls
        self.renderer = PatientRenderer(renderer)
        self.budget = Budget()
        self.patch_log: list[dict[str, Any]] = []  # beat-all L3: one entry per patch-edit reply
        self.target = {"type": "input_image", "image_url": png_data_url(image), "detail": "high"}
        self.digest = hashlib.sha256(image).hexdigest()
        self.best_so_far: str | None = None  # returned if a later stage fails unexpectedly
        self.wrapper = WRAPPER
        self.closeup: list[dict[str, Any]] = []
        self.instructions = prompts.GENERATE_CHAIN
        self.astra_cap = ASTRA_MAX_OUTPUT
        self.photo: match.Appearance | None = None
        self.brief = ""
        self.plan: StagePlan | None = None

    # ----- model calls -----

    def run_stage(self, pool: ThreadPoolExecutor, calls: list[Call]) -> list[Response | None]:
        """Reserve budget for every call (main thread, fixed order), then dispatch them concurrently."""
        admitted: list[Call | None] = []
        for call in calls:
            text_chars = len(call.instructions) + sum(len(c.get("text", "")) for c in call.content)
            images = sum(1 for c in call.content if c["type"] == "input_image")
            call.estimate = worst_case(call.model, text_chars, images, call.max_output_tokens)
            admitted.append(call if self.budget.reserve(call.estimate) else None)
        futures = [pool.submit(self._dispatch, call) if call is not None else None for call in admitted]
        return [future.result() if future is not None else None for future in futures]

    def _dispatch(self, call: Call) -> Response | None:
        usage = None
        params: dict[str, Any] = {
            "model": call.model,
            "store": False,
            "service_tier": "default",
            "reasoning": {"effort": call.effort},
            "max_output_tokens": call.max_output_tokens,
            "instructions": call.instructions,
            "input": cast(ResponseInputParam, [{"role": "user", "content": call.content}]),
        }
        if call.text is not None:
            params["text"] = call.text
        try:
            result = self._create(call.request_id, params)
            usage = result.usage
            if usage is None:
                raise CallFailed(f"call {call.request_id}: reply without usage")
            # A failed or incomplete reply with usage is a saved response, so replay returns it too.
            return result if result.status == "completed" else None
        except APIConnectionError:
            raise  # the witness reports delivery failures and timeouts this way; the runtime recovers the call
        except APIError as error:
            raise CallFailed(f"call {call.request_id}: {error}") from error
        finally:
            self.budget.settle(call.estimate, call.model, usage)

    def _create(self, request_id: int, params: dict[str, Any]) -> Response:
        """A slow capture can outlast the client's wait; repeating the request joins it without paying again."""
        for attempt in range(JOIN_ATTEMPTS):
            try:
                return self.calls.create(request_id=request_id, **params)
            except APITimeoutError:
                if attempt == JOIN_ATTEMPTS - 1:
                    raise
        raise AssertionError("unreachable")

    def generation_call(self, request_id: int, model: str, extra: list[dict[str, Any]], effort: str = GENERATE_EFFORT,
                        instructions: str = "base", max_output: int = GENERATE_MAX_OUTPUT) -> Call:
        content = [{"type": "input_text", "text": prompts.TARGET}, self.target, *extra]
        text = prompts.GENERATE_V2 if instructions == "handbook" else prompts.GENERATE
        return Call(request_id, model, text, content, effort, max_output)

    def module_call(self, request_id: int, model: str, effort: str, extra: list[dict[str, Any]],
                    start: str = "plain") -> Call:
        target = prompts.TARGET_PROPORTIONS if start == "proportions" else prompts.TARGET
        if self.brief and not extra:
            target = f"{target}\n\n{self.brief}"
        content = [{"type": "input_text", "text": target}, self.target, *self.closeup, *extra]
        cap = self.astra_cap if model == ASTRA else EFFORT_MAX_OUTPUT.get(effort, GENERATE_MAX_OUTPUT)
        return Call(request_id, model, self.instructions, content, effort, cap)

    def rewrite_call(self, request_id: int, model: str, effort: str, parent: Candidate, focus: str = "",
                     patch_edit: bool = False) -> Call:
        views = [{"type": "input_image", "image_url": data_url(view), "detail": "high"} for view in parent.views]
        if self.photo is not None or self.brief:
            overlay = [{"type": "input_image", "image_url": data_url(parent.overlay), "detail": "low"}] if parent.overlay else []
            text = prompts.REWRITE_MEASURED.format(source=parent.raw, focus=prompts.REWRITE_FOCUS[focus], brief=self.brief,
                                                 measurements=parent.measurements)
            return self.module_call(request_id, model, effort, [
                *views, {"type": "input_image", "image_url": data_url(parent.sides), "detail": "high"}, *overlay,
                {"type": "input_text", "text": text}])
        call = self.module_call(request_id, model, effort, [
            *views,
            {"type": "input_image", "image_url": data_url(parent.sides), "detail": "high"},
            {"type": "input_text", "text": (prompts.REWRITE_PATCH if patch_edit else prompts.REWRITE).format(
                source=parent.raw, focus=prompts.REWRITE_FOCUS[focus])},
        ])
        if patch_edit and self.instructions == prompts.GENERATE_CHAIN:
            call.instructions = prompts.GENERATE_CHAIN_PATCH
        return call

    # ----- rendering -----

    def render(self, name: str, source: str, front_options: RenderOptions = FRONT) -> Candidate | tuple[str, str]:
        """A rendered candidate, or (source, validation errors) when the module is rejected."""
        try:
            front = self.renderer.render(source, front_options)
        except RenderRejected as rejection:
            return source, rejection.detail_json
        return Candidate(name, source, front, self.renderer.render(source, SIDES))

    def oriented(self, candidate: Candidate) -> Candidate:
        """Turn the asset so the side matching the photo's silhouette faces the judge's front cameras."""
        try:
            yaw = orient.best_yaw(self.image, candidate.source, self.renderer)
        except RenderRejected:  # a turned copy should always validate; if not, keep the candidate as built
            return candidate
        if yaw == 0:
            return candidate
        outcome = self.render(candidate.name, orient.rotated(candidate.source, yaw))
        return outcome if isinstance(outcome, Candidate) else candidate

    def build(self, pool: ThreadPoolExecutor, named: list[tuple[str, Response | None]], repair_base: int,
              orient_candidates: bool = False) -> list[Candidate]:
        """Render replies; rejected modules get one repair, then those are rendered too."""
        candidates: list[Candidate] = []
        rejected: list[tuple[str, str, str]] = []
        for name, reply in named:
            source = extract_source(reply.output_text) if reply is not None else None
            if source is None:
                continue
            outcome = self.render(name, normalize(source))
            if isinstance(outcome, Candidate):
                candidates.append(self.oriented(outcome) if orient_candidates else outcome)
            else:
                rejected.append((name, *outcome))
        repairs = [
            self.generation_call(repair_base + i, SOL, [{"type": "input_text", "text": prompts.REPAIR.format(
                errors=errors, source=source)}])
            for i, (_, source, errors) in enumerate(rejected)
        ]
        for (name, _, _), reply in zip(rejected, self.run_stage(pool, repairs), strict=True):
            source = extract_source(reply.output_text) if reply is not None else None
            if source is not None:
                outcome = self.render(name + "r", normalize(source))
                if isinstance(outcome, Candidate):
                    candidates.append(self.oriented(outcome) if orient_candidates else outcome)
        return candidates

    def render_module(self, name: str, source: str, raw: str = "") -> Candidate | tuple[str, str]:
        """A rendered candidate, or (source, validation errors) when the module is rejected."""
        try:
            front = self.renderer.render(source, FRONT_JUDGE)
            views = zoomed_views(front)
            grid = to_jpeg(self.renderer.render(source, GRID_VIEW))
        except RenderRejected as rejection:
            return source, rejection.detail_json
        candidate = Candidate(name, source, view_sheet(views), grid, raw, views, judge_front=front)
        if self.photo is not None:
            candidate.match = match.grid_score(front, self.photo)
            candidate.measurements, candidate.overlay = match.feedback(front, self.photo)
        return candidate

    def modules(self, pool: ThreadPoolExecutor, calls: list[Call], names: list[str],
                bases: dict[int, str] | None = None) -> list[Candidate | None]:
        """Run calls as one stage and render the replies in order; rejected modules get one repair in a second stage.
        bases: call index -> the parent's module, for patch-edit calls (the reply's edits are applied to it)."""
        results: list[Candidate | None] = [None] * len(calls)
        rejected: list[tuple[int, str, str]] = []
        for index, reply in enumerate(self.run_stage(pool, calls)):
            if reply is None:
                raw = None
            elif bases is not None and index in bases:
                applied = patch.apply(bases[index], reply.output_text, extract_source)
                raw = applied.code if applied.code is not None and _EXPORT.search(applied.code) else None
                self.patch_log.append({"name": names[index], "applied": applied.applied, "skipped": len(applied.skipped),
                                       "full": applied.full, "used": raw is not None})
            else:
                raw = extract_source(reply.output_text)
            if raw is None:
                continue
            outcome = self.render_module(names[index], normalize(raw, self.wrapper), raw)
            if isinstance(outcome, Candidate):
                results[index] = outcome
            else:
                rejected.append((index, raw, outcome[1]))
        repairs = [
            self.module_call(calls[index].request_id + REPAIR_OFFSET, SOL, calls[index].effort, [
                {"type": "input_text", "text": prompts.REPAIR.format(errors=errors, source=raw)}])
            for index, raw, errors in rejected
        ]
        for (index, _, _), reply in zip(rejected, self.run_stage(pool, repairs), strict=True):
            raw = extract_source(reply.output_text) if reply is not None else None
            if raw is not None:
                outcome = self.render_module(names[index], normalize(raw, self.wrapper), raw)
                if isinstance(outcome, Candidate):
                    results[index] = outcome
        return results

    # ----- selection -----

    def judge_calls(self, candidates: list[Candidate], request_id: int, model: str = SOL, mirrored: bool = True,
                    instructions: str = prompts.SELECT) -> tuple[dict[str, Candidate], list[Call]]:
        ordered = candidates.copy()
        random.Random(f"{self.seed}:{self.digest}:{request_id}").shuffle(ordered)
        labels = {chr(65 + i): candidate for i, candidate in enumerate(ordered)}
        calls = []
        orders = (list(labels), list(reversed(labels))) if mirrored else (list(labels),)
        for offset, order in enumerate(orders):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target]
            for label in order:
                content += [
                    {"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"},
                    {"type": "input_image", "image_url": png_data_url(labels[label].front), "detail": "high"},
                    {"type": "input_text", "text": f"CANDIDATE {label} SIDE VIEWS:"},
                    {"type": "input_image", "image_url": png_data_url(labels[label].sides), "detail": "low"},
                ]
            calls.append(Call(request_id + offset, model, instructions, content, SELECT_EFFORT, SELECT_MAX_OUTPUT,
                              text=prompts.select_schema(list(labels))))
        return labels, calls

    @staticmethod
    def parse_judgement(labels: dict[str, Candidate], replies: list[Response | None]) -> dict[str, Judgement]:
        """Average each candidate's score over the replies; lower is better. Unscored candidates are absent."""
        scores: dict[str, list[float]] = {}
        fixes: dict[str, list[str]] = {}
        for reply in replies:
            if reply is None:
                continue
            try:
                entries = json.loads(reply.output_text)["candidates"]
            except (ValueError, KeyError, TypeError):
                continue
            seen = set()
            for entry in entries:
                label = entry.get("label")
                if label not in labels or label in seen:
                    continue
                seen.add(label)
                penalties = [entry[k] for k in ("front_left", "front_right", "front_below", "front_above")]
                score = sum(min(10, max(0, int(p))) for p in penalties) / 4
                if int(entry["garbage_sides"]) >= 3:
                    score += GARBAGE_PENALTY
                if "grid" in entry:
                    score += GRID_WEIGHT * min(10, max(0, int(entry["grid"])))
                scores.setdefault(label, []).append(score)
                fixes.setdefault(label, []).extend(str(f) for f in entry.get("fixes", [])[:6])
        return {
            labels[label].name: Judgement(sum(values) / len(values), list(dict.fromkeys(fixes[label]))[:8])
            for label, values in scores.items()
        }

    def judge(self, pool: ThreadPoolExecutor, candidates: list[Candidate], request_id: int, model: str = SOL,
              mirrored: bool = True, instructions: str = prompts.SELECT) -> dict[str, Judgement]:
        """Score candidates twice in opposite orders; lower is better. Missing scores mean no usable judgement."""
        labels, calls = self.judge_calls(candidates, request_id, model, mirrored, instructions)
        return self.parse_judgement(labels, self.run_stage(pool, calls))

    def tournament(self, pool: ThreadPoolExecutor, candidates: list[Candidate], group_size: int,
                   base: int) -> tuple[Candidate, Judgement | None]:
        """Judge groups of candidates (all groups of a round in one stage); group winners advance."""
        alive = candidates.copy()
        verdict: Judgement | None = None
        round_index = 0
        while len(alive) > 1:
            random.Random(f"{self.seed}:{self.digest}:t{round_index}").shuffle(alive)
            size = max(2, group_size)
            groups = [alive[i:i + size] for i in range(0, len(alive), size)]
            if len(groups) > 1 and len(groups[-1]) == 1:
                groups[-2].extend(groups.pop())  # never leave a lone candidate with a free pass
            stage: list[tuple[dict[str, Candidate], int]] = []
            calls: list[Call] = []
            for index, group in enumerate(groups):
                labels, group_calls = self.judge_calls(group, base + 100 * round_index + 2 * index)
                stage.append((labels, len(group_calls)))
                calls += group_calls
            replies = self.run_stage(pool, calls)
            winners: list[Candidate] = []
            offset = 0
            for (labels, count), group in zip(stage, groups, strict=True):
                judgements = self.parse_judgement(labels, replies[offset:offset + count])
                offset += count
                winner = self.best(group, judgements)
                winners.append(winner)
                verdict = judgements.get(winner.name)
            alive = winners
            round_index += 1
        return alive[0], verdict

    def scores(self, pool: ThreadPoolExecutor, candidates: list[Candidate], request_id: int,
               instructions: str = prompts.SELECT_CLOSE) -> dict[str, Judgement]:
        """Score candidates twice in opposite orders on the front views and the grid; lower is better."""
        ordered = candidates.copy()
        random.Random(f"{self.seed}:{self.digest}:{request_id}").shuffle(ordered)
        labels = {chr(65 + i): candidate for i, candidate in enumerate(ordered)}
        calls = []
        for offset, order in enumerate((list(labels), list(reversed(labels)))):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target, *self.closeup]
            if self.brief:
                content.append({"type": "input_text", "text": self.brief})
            for label in order:
                content += [
                    {"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"},
                    {"type": "input_image", "image_url": data_url(labels[label].front), "detail": "high"},
                    {"type": "input_text", "text": f"CANDIDATE {label} GRID:"},
                    {"type": "input_image", "image_url": data_url(labels[label].sides), "detail": "high"},
                ]
            calls.append(Call(request_id + offset, SOL, instructions, content, SELECT_EFFORT, SELECT_MAX_OUTPUT,
                              text=prompts.select_close_schema(list(labels))))
        return self.parse_judgement(labels, self.run_stage(pool, calls))

    def select(self, pool: ThreadPoolExecutor, candidates: list[Candidate], parents: set[str], keep: int,
               request_id: int) -> list[Candidate]:
        """The best `keep` candidates. Ties and unjudged candidates keep list order, so list children before parents."""
        if len(candidates) <= keep:
            return candidates
        judgements = self.scores(pool, candidates, request_id)

        def rank(item: tuple[int, Candidate]) -> tuple[float, int]:
            index, candidate = item
            judgement = judgements.get(candidate.name)
            # wren-plus: an unjudged parent (panel refused by the budget guard) ranks before an unjudged child
            score = judgement.score if judgement else (98.0 if candidate.name in parents else 99.0)
            return (score - MATCH_WEIGHT * candidate.match + (PARENT_MARGIN if candidate.name in parents else 0.0), index)

        return [candidate for _, candidate in sorted(enumerate(candidates), key=rank)[:keep]]

    # ----- run -----

    def run(self, plan: Plan) -> str:
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="miner404") as pool:
            draft_calls = [
                self.generation_call(1 + i, model, [], effort=effort, instructions=instructions,
                                     max_output=plan.draft_max_output)
                for i, (model, effort, instructions) in enumerate(plan.drafts)
            ]
            replies = self.run_stage(pool, draft_calls)
            candidates = self.build(pool, [(f"c{1 + i}", reply) for i, reply in enumerate(replies)],
                                    repair_base=500, orient_candidates=plan.orient)
            if not candidates:
                return fallback_module(self.image)
            self.best_so_far = candidates[0].source
            if len(candidates) == 1 or plan.group_size >= len(candidates):
                judgements = self.judge(pool, candidates, 1000)
                incumbent = self.best(candidates, judgements)
                verdict = judgements.get(incumbent.name)
            else:
                incumbent, verdict = self.tournament(pool, candidates, plan.group_size, base=1000)
            self.best_so_far = incumbent.source
            for round_index in range(1, plan.refine_rounds + 1):
                if verdict is None:
                    break
                base = 2000 + 100 * round_index
                calls = [
                    self.generation_call(base + i, SOL, [
                        {"type": "input_image", "image_url": png_data_url(incumbent.front), "detail": "high"},
                        {"type": "input_image", "image_url": png_data_url(incumbent.sides), "detail": "low"},
                        {"type": "input_text", "text": prompts.REFINE.format(
                            focus=prompts.REFINE_FOCUS["all" if focus == "blind" else focus],
                            fixes=("- (none listed)" if focus == "blind" else
                                   "\n".join(f"- {fix}" for fix in verdict.fixes) or "- (none listed)"),
                            source=incumbent.source)},
                    ], effort=plan.refine_effort)
                    for i, focus in enumerate(plan.refine_focuses)
                ]
                calls = self.affordable(calls)
                if not calls:
                    break
                replies = self.run_stage(pool, calls)
                children = self.build(pool, [(f"r{round_index}{c.request_id % 100}", r) for c, r in
                                             zip(calls, replies, strict=True)], repair_base=base + 50,
                                      orient_candidates=plan.orient)
                if not children:
                    continue
                judgements = self.judge(pool, [incumbent, *children], base + 80)
                best = self.best([incumbent, *children], judgements)
                if judgements.get(best.name) is not None:
                    incumbent, verdict = best, judgements[best.name]
                    self.best_so_far = incumbent.source
            return incumbent.source

    def run_stages(self, plan: StagePlan) -> str:
        self.plan = plan
        self.wrapper = chain_wrapper(plan.fill, plan.glow, plan.outline_fit or plan.measured)
        closeup = object_closeup(self.image)
        if closeup is not None:
            self.closeup = [{"type": "input_image", "image_url": data_url(closeup), "detail": "high"}]
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="miner404") as pool:
            if plan.measured:
                self.instructions = prompts.GENERATE_MEASURED
                self.astra_cap = MEASURED_ASTRA_MAX_OUTPUT
                self.photo = match.photo_appearance(self.image)
                self.brief = self.photo_brief(pool)
            calls = [self.module_call(DRAFT_ID + i, draft.model, draft.effort, [], draft.start)
                     for i, draft in enumerate(plan.drafts)]
            survivors = [c for c in self.modules(pool, calls, [f"d{i}" for i in range(len(calls))]) if c is not None]
            if not survivors:  # one more attempt on the cheaper model before giving up on the prompt
                calls = [self.module_call(REDRAFT_ID + i, SOL, "high", [], draft.start)
                         for i, draft in enumerate(plan.drafts[:3])]
                survivors = [c for c in self.modules(pool, calls, [f"e{i}" for i in range(len(calls))]) if c is not None]
            if not survivors:
                return fallback_module(self.image)
            self.best_so_far = survivors[0].source
            if plan.draft_keep:
                survivors = self.select(pool, survivors, set(), plan.draft_keep, SELECT_ID)
            searched = False
            for index, step in enumerate(plan.steps):
                survivors = self.advance(pool, survivors, step, index)
                self.best_so_far = survivors[0].source
                if plan.measured and len(survivors) == 1 and not searched:  # once only: the Pod has one render slot
                    survivors[0], searched = self.posed_by_search(pool, survivors[0]), True
            if plan.final_duel and len(survivors) > 1:
                winner = self.final_pick(pool, survivors)
            else:
                winner = self.select(pool, survivors, set(), 1, FINAL_SELECT_ID)[0]
            self.best_so_far = winner.source
            final = self.posed(pool, winner) if plan.pose and not plan.measured else winner
            return self.fitted_last(winner, plan) if plan.outline_fit_last and final is winner else final.source

    def fitted_last(self, winner: Candidate, plan: StagePlan) -> str:
        """The finished module with the outline fit, or as it is if the fit is rejected. A module the pose stage turned keeps
        its plain fit, because the turn is written around the wrapped source."""
        if not winner.raw:
            return winner.source
        source = normalize(winner.raw, chain_wrapper(plan.fill, plan.glow, True))
        try:
            self.renderer.render(source, FRONT_JUDGE)
        except RenderRejected:
            return winner.source
        return source

    def photo_brief(self, pool: ThreadPoolExecutor) -> str:
        """What the model must get right, written once from the photo and shown to every draft and rewrite."""
        call = Call(BRIEF_ID, SOL, prompts.BRIEF, [self.target, *self.closeup], SELECT_EFFORT, BRIEF_MAX_OUTPUT,
                    text=prompts.brief_schema())
        reply = self.run_stage(pool, [call])[0]
        try:
            brief = json.loads(reply.output_text) if reply is not None else None
            return prompts.BRIEF_BLOCK.format(
                main_object=brief["main_object"], items="; ".join(brief["items"]), features="; ".join(brief["features"]),
                pose=brief["pose"], true_colors="; ".join(brief["true_colors"]))
        except (TypeError, KeyError, ValueError):
            return ""

    def posed_by_search(self, pool: ThreadPoolExecutor, incumbent: Candidate) -> Candidate:
        """Offer the best turn to the selector, and keep it when the two scores together prefer it. The turn goes into the
        module's own code, so the rewrites that follow can see it and keep it."""
        if self.photo is None:
            return incumbent
        pose, best, start = match.search_pose(incumbent.source, self.photo, self.renderer, orient.turned)
        if best - start < POSE_SEARCH_GAIN or not any(pose):
            return incumbent
        raw = orient.turned_module(incumbent.raw, *pose)
        outcome = self.render_module(incumbent.name + "t", normalize(raw, self.wrapper), raw)
        if not isinstance(outcome, Candidate):
            return incumbent
        judgements = self.scores(pool, [incumbent, outcome], POSE_ID, prompts.SELECT_POSE)
        current, offered = judgements.get(incumbent.name), judgements.get(outcome.name)
        if current is None or offered is None:
            return incumbent
        now = current.score - MATCH_WEIGHT * incumbent.match
        return outcome if offered.score - MATCH_WEIGHT * outcome.match + PARENT_MARGIN <= now else incumbent

    def advance(self, pool: ThreadPoolExecutor, survivors: list[Candidate], step: Step, index: int) -> list[Candidate]:
        select_id = SELECT_ID + 10 * (index + 1)
        if not step.rewrites and not step.lineage:
            parents = [c.parent for c in survivors if c.parent is not None]
            return self.select(pool, [*survivors, *parents], {p.name for p in parents}, step.keep or 1, select_id)
        calls, names, owners = [], [], []
        bases: dict[int, str] = {}
        for position, parent in enumerate(survivors):
            rewrites = (step.lineage[position % len(step.lineage)],) if step.lineage else step.rewrites
            focus = step.focus[position % len(step.focus)] if step.focus else ""
            for number, (model, effort) in enumerate(rewrites):
                request_id = REWRITE_BASE * (index + 1) + 10 * position + number
                if step.patch:
                    bases[len(calls)] = parent.raw
                calls.append(self.rewrite_call(request_id, model, effort, parent, focus, step.patch))
                names.append(f"r{index}.{position}.{number}")
                owners.append(position)
        children = self.modules(pool, calls, names, bases if step.patch else None)
        for owner, child in zip(owners, children, strict=True):
            if child is not None:
                child.parent = survivors[owner]
        if step.per_lineage:
            return self.per_lineage(pool, survivors, children, owners, select_id)
        if step.keep:
            valid = [child for child in children if child is not None]
            kept = self.select(pool, [*valid, *survivors], {c.name for c in survivors}, step.keep, select_id)
            return self.gated(pool, kept, survivors, index) if step.keep == 1 else kept
        newest = list(survivors)
        for owner, child in zip(owners, children, strict=True):
            if child is not None:
                newest[owner] = child
        return newest

    def rank_key(self, judgements: dict[str, Judgement], candidate: Candidate, parents: set[str]) -> float:
        """select()'s ranking: lower is better; an unjudged parent ranks before an unjudged child."""
        judgement = judgements.get(candidate.name)
        score = judgement.score if judgement else (98.0 if candidate.name in parents else 99.0)
        return score - MATCH_WEIGHT * candidate.match + (PARENT_MARGIN if candidate.name in parents else 0.0)

    def per_lineage(self, pool: ThreadPoolExecutor, survivors: list[Candidate], children: list[Candidate | None],
                    owners: list[int], select_id: int) -> list[Candidate]:
        """beat-all L3: one panel over every parent and child; each lineage keeps the better of its parent and its own
        children (the parent needs no margin to stay; a child must beat it by PARENT_MARGIN). Lineages stay separate."""
        valid = [child for child in children if child is not None]
        if not valid:
            return survivors
        parents = {c.name for c in survivors}
        judgements = self.scores(pool, [*valid, *survivors], select_id)
        newest = []
        for position, parent in enumerate(survivors):
            own = [child for owner, child in zip(owners, children, strict=True) if owner == position and child is not None]
            ranked = sorted(enumerate([*own, parent]), key=lambda item: (self.rank_key(judgements, item[1], parents), item[0]))
            newest.append(ranked[0][1])
        return newest

    def final_pick(self, pool: ThreadPoolExecutor, survivors: list[Candidate]) -> Candidate:
        """beat-all L3: panel keeps the best two lineages; an 8-call Sol-low copy of 404's stage 1 decides between them
        when it fits the budget with the pose panel still left (else the panel's first). Tie -> the panel's first."""
        top = self.select(pool, survivors, set(), 2, FINAL_SELECT_ID)
        if len(top) < 2 or not top[0].judge_front or not top[1].judge_front:
            return top[0]
        calls = self.duel_calls(top[0], top[1], FINAL_DUEL_ID)
        estimates = [worst_case(c.model, len(c.instructions) + sum(len(p.get("text", "")) for p in c.content),
                                sum(1 for p in c.content if p["type"] == "input_image"), c.max_output_tokens)
                     for c in calls]
        panel = worst_case(SOL, len(prompts.SELECT_CLOSE) + len(prompts.SELECT_POSE) + len(self.brief) + 2000, 8,
                           SELECT_MAX_OUTPUT)
        if not self.budget.allows(estimates, keep_calls=2, keep_credits=panel):
            return top[0]
        first_penalty = second_penalty = 0.0
        for number, reply in enumerate(self.run_stage(pool, calls)):
            one, two = self.pair_penalties(reply)
            in_order = number < len(prompts.JUDGE_ANGLES)  # calls 0-3: top[0] is model 1; calls 4-7: swapped
            first_penalty += one if in_order else two
            second_penalty += two if in_order else one
        return top[1] if second_penalty < first_penalty else top[0]

    def later_calls(self, index: int) -> int:
        """Calls the plan still makes after step `index` when nothing is repaired: each later step's rewrites and its
        2-call selection, then the 2-call pose selection."""
        steps = self.plan.steps if self.plan is not None else ()
        calls, survivors = 2, (steps[index].keep or 1) if index < len(steps) else 1
        for step in steps[index + 1:]:
            calls += (1 if step.lineage else len(step.rewrites)) * survivors + 2
            survivors = step.keep or survivors
        return calls

    def duel_calls(self, first: Candidate, second: Candidate, request_id: int) -> list[Call]:
        """404's stage 1 for one pair: both orders x the 4 front views, one Sol-low call each."""
        tiles = {c.name: judge_tiles(c.judge_front) for c in (first, second)}
        calls = []
        for offset, (one, two) in enumerate(((first, second), (second, first))):
            for view, angle in enumerate(prompts.JUDGE_ANGLES):
                content: list[dict[str, Any]] = [
                    {"type": "input_text", "text": "Reference image (target object):"},
                    self.target,
                    {"type": "input_text", "text": "3D model 1:"},
                    {"type": "input_image", "image_url": png_data_url(tiles[one.name][view]), "detail": "high"},
                    {"type": "input_text", "text": "3D model 2:"},
                    {"type": "input_image", "image_url": png_data_url(tiles[two.name][view]), "detail": "high"},
                    {"type": "input_text", "text": prompts.JUDGE_USER.format(angle_desc=angle)},
                ]
                calls.append(Call(request_id + 4 * offset + view, SOL, prompts.JUDGE_SYSTEM, content, DUEL_EFFORT,
                                  DUEL_MAX_OUTPUT))
        return calls

    @staticmethod
    def pair_penalties(reply: Response | None) -> tuple[float, float]:
        """A judge reply -> (penalty_1, penalty_2); a missing or unusable reply -> (5, 5), which favours neither."""
        found = re.search(r"\{.*\}", reply.output_text if reply is not None else "", re.DOTALL)
        try:
            data = json.loads(found.group(0)) if found else {}
            return (min(max(float(data["penalty_1"]), 0.0), 10.0),  # the judge's 0-10 scale
                    min(max(float(data["penalty_2"]), 0.0), 10.0))
        except (ValueError, KeyError, TypeError):
            return 5.0, 5.0

    def gated(self, pool: ThreadPoolExecutor, kept: list[Candidate], survivors: list[Candidate],
              index: int) -> list[Candidate]:
        """wren-plus N2: the panel's pick of a keep-1 step must also beat its parent in an 8-call stage-1 duel, else the
        parent stays. Run only when the 8 calls fit the budget with every later planned call and the worst case of every
        later panel (and the pose panel) still left (recorded usage and reservations, so replay decides the same);
        otherwise the panel decides alone, as before."""
        winner = kept[0] if kept else None
        parent = winner.parent if winner is not None else None
        if (winner is None or parent is None or winner.name in {c.name for c in survivors}
                or not winner.judge_front or not parent.judge_front):
            return kept
        calls = self.duel_calls(winner, parent, DUEL_ID + 10 * index)
        estimates = [worst_case(c.model, len(c.instructions) + sum(len(p.get("text", "")) for p in c.content),
                                sum(1 for p in c.content if p["type"] == "input_image"), c.max_output_tokens)
                     for c in calls]
        panel = worst_case(SOL, len(prompts.SELECT_CLOSE) + len(prompts.SELECT_POSE) + len(self.brief) + 2000, 8,
                           SELECT_MAX_OUTPUT)
        later = self.later_calls(index)
        later_panels = 2 * (len(self.plan.steps) - index) if self.plan is not None else 2
        if not self.budget.allows(estimates, keep_calls=later, keep_credits=later_panels * panel):
            return kept
        child_penalty = parent_penalty = 0.0
        for number, reply in enumerate(self.run_stage(pool, calls)):
            first, second = self.pair_penalties(reply)
            child_first = number < len(prompts.JUDGE_ANGLES)
            child_penalty += first if child_first else second
            parent_penalty += second if child_first else first
        return kept if child_penalty < parent_penalty else [parent]

    def posed(self, pool: ThreadPoolExecutor, incumbent: Candidate) -> Candidate:
        """Offer other poses to the selector: the photo-matching quarter turn, and standing a flat object up."""
        try:
            sources = []
            yaw = orient.best_yaw(self.image, incumbent.source, self.renderer)
            if yaw:
                sources.append(orient.rotated(incumbent.source, yaw))
            if orient.is_flat(incumbent.source, self.renderer):
                sources += [orient.stood_up(incumbent.source, pitch) for pitch in POSE_PITCHES]
        except RenderRejected:  # a probe of a validated module should not fail; if it does, keep the module as built
            return incumbent
        variants = []
        for index, source in enumerate(sources):
            outcome = self.render_module(f"p{index}", source)
            if isinstance(outcome, Candidate):
                variants.append(outcome)
        if not variants:
            return incumbent
        judgements = self.scores(pool, [incumbent, *variants], POSE_ID, prompts.SELECT_POSE)
        best = self.best(variants, judgements)
        current, offered = judgements.get(incumbent.name), judgements.get(best.name)
        if current is not None and offered is not None and offered.score + POSE_MARGIN <= current.score:
            return best
        return incumbent

    def affordable(self, calls: list[Call]) -> list[Call]:
        """Drop refinements from the end until the round, its repairs and its two judge calls fit the budget."""
        judge_estimate = worst_case(SOL, 4000, 9, SELECT_MAX_OUTPUT)
        while calls:
            estimates = []
            for call in calls:
                text_chars = len(call.instructions) + sum(len(c.get("text", "")) for c in call.content)
                images = sum(1 for c in call.content if c["type"] == "input_image")
                estimates.append(worst_case(call.model, text_chars, images, call.max_output_tokens))
            if self.budget.allows(estimates, keep_calls=2 + len(calls), keep_credits=2 * judge_estimate):
                return calls
            calls = calls[:-1]
        return calls

    @staticmethod
    def best(candidates: list[Candidate], judgements: dict[str, Judgement]) -> Candidate:
        """Lowest score; unjudged candidates rank last; ties keep the earlier (incumbent first) candidate."""
        def key(item: tuple[int, Candidate]) -> tuple[float, int]:
            index, candidate = item
            judgement = judgements.get(candidate.name)
            return (judgement.score if judgement else 99.0, index)

        return min(enumerate(candidates), key=key)[1]


BASIC_PLAN = Plan(drafts=((SOL, "high", "base"),) * 9, draft_max_output=32768)


def run(image: bytes, seed: int, calls: ModelCalls, renderer: Renderer, plan: Plan | StagePlan) -> str | None:
    pipeline = Pipeline(image, seed, calls, renderer)
    try:
        return pipeline.run_stages(plan) if isinstance(plan, StagePlan) else pipeline.run(plan)
    except (APIConnectionError, CallFailed, WitnessError, ExchangePending, RenderUnavailable):
        raise  # the runner tries the prompt again, reusing saved calls
    except Exception:
        # A bug in a later stage must not cost the prompt; the outcome is deterministic, so replay matches.
        return pipeline.best_so_far or fallback_module(image)
