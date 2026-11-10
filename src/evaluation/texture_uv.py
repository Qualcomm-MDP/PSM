"""
texture_uv.py
UV mapping texture quality of the pipeline's model ("UV Mapping Texture Quality Metric").

One score for a textured UV map, computed automatically:

    Confidence = 1 - (alpha O + beta D + gamma F),  alpha + beta + gamma = 1, all >= 0

  O  obstruction ratio  how much of the texture that should hold photo texture doesn't:
                        O = 1 - |M_res ∩ M_gt| / |M_gt|
  D  distortion ratio   share of the surface area whose texture is visibly stretched: a
                        triangle's stretch is s = sigma1 / sigma2 of the Jacobian of its
                        surface -> texture map; it counts when s > s_max (default 1.5)
  F  realism            1 - exp(-FID / tau), FID between Inception-v3 features (2048-d
                        pool) of K random patches of the texture and of a reference

Default weights (the paper's, for building textures): alpha 0.35, beta 0.25, gamma 0.40.

How this pipeline's model is read (differences from the paper, which assumes a ground-truth
texture with the same UV layout; this pipeline has none):
  I_res      the buildings' texture atlas(es) in the .glb. The ground (terrain / NAIP
             aerial photo) is not evaluated.
  M_gt       the texels the mesh's UV triangles cover (rasterised charts).
  M_res      the texels in M_gt holding real photo texture: texturing records which
             texels came from a photo (not left grey because no photo saw them, and not
             painted over by obstruction removal) in <model>.texture_mask.png next to the
             .glb. Without that file (models from before it existed), every texel that
             isn't texturing's untextured grey counts.
  O          area-weighted: each triangle counts with its surface area times the share of
             its texels in M_res. The atlas gives every triangle its own texel density
             (the photo's resolution), so counting texels as the paper does would weigh
             close-up walls more. Triangles with no chart (texturing points untextured
             faces at one grey texel) count as fully missing. The paper's texel count is
             reported too (o_texels).
  D          stretch is measured in texels (the atlas is not square, so [0, 1] UVs would
             add a stretch the texture doesn't have). Triangles with no chart are left out
             (they are counted by O); zero-area triangles are left out; a chart triangle
             of zero area (its corners in a line) is infinitely stretched: distorted.
  FID        the reference ("ground truth" distribution) is the job's own street photos
             (the originals saved in images/), sampled the same way, unless a true
             ground-truth texture is given (--gt-texture), in which case the paper's
             definition is used as written (O, D and FID against that texture).
  tau        no calibration set exists yet: default TAU (100), recorded in the output.
             Calibrate it as the paper says: the median FID of good reference results.
If Inception-v3 can't be loaded (no torch / torchvision, or its weights can't be
downloaded), F is left out and the confidence is computed over O and D with their weights
renormalised; the result says "partial".

Run (from the repo root, in the project environment)
  poetry run python scripts/eval/eval_texture_uv.py --latest       # newest web app job
  poetry run python scripts/eval/eval_texture_uv.py output/jobs/<id>
  poetry run python scripts/eval/eval_texture_uv.py model.glb --reference photos/
  ... --gt-texture gt.png [--gt-mask m.png] [--res-mask m.png]   the paper's setting
Outputs (texture_eval/ next to the .glb): summary.md, summary.json, per_group.csv,
01_scorecard.png, 02_coverage.png, 03_stretch.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EVALUATION_DIR = "texture_eval"
MASK_SUFFIX = ".texture_mask.png"  # texturing's real-photo texels, next to the .glb
# What run writes, in reading order: the files the web app may serve.
OUTPUT_FILES = (
    "summary.md",
    "01_scorecard.png",
    "02_coverage.png",
    "03_stretch.png",
    "per_group.csv",
    "summary.json",
)

ALPHA, BETA, GAMMA = 0.35, 0.25, 0.40  # the paper's weights for building textures
S_MAX = 1.5  # stretch above which a triangle counts as distorted
TAU = 100.0  # FID scale: a starting value, to calibrate on this domain
K = 2048  # patches per texture for the FID
PATCH = 64  # patch side in texels (resized to Inception's 299)
MIN_FILL = 0.9  # a patch must lie this much inside its mask
SEED = 0
UNTEXTURED_GREY = 225  # texturing's colour for texels no photo covers
INCEPTION_SIZE = 299

_REPORTER: ContextVar = ContextVar("texture_eval_reporter", default=None)


class Unavailable(SystemExit):
    """The evaluation can't run on this input (no textured mesh): not a crash. A
    SystemExit, so the command line stops with the message."""


def _log(msg: str, warning: bool = False) -> None:
    reporter = _REPORTER.get()
    if reporter is None:
        print(msg)
        return
    for line in str(msg).strip("\n").splitlines():
        if line.strip():
            (reporter.warning if warning else reporter.info)(line)


# ----------------------------------------------------------------------------
# The score
# ----------------------------------------------------------------------------


def check_weights(alpha: float, beta: float, gamma: float) -> None:
    """alpha, beta, gamma must be >= 0 and sum to 1 (equation 1)."""
    w = (alpha, beta, gamma)
    if any(not math.isfinite(v) or v < 0 for v in w):
        raise ValueError(f"weights must be >= 0, got {w}")
    if abs(sum(w) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {sum(w):g}")


def confidence(
    terms: dict[str, float | None], weights: dict[str, float]
) -> tuple[float, bool, dict[str, float]]:
    """(Confidence, partial, the weights used). Terms that are None (not available) are
    left out and the others' weights renormalised to sum to 1; partial is then True."""
    have = {k: v for k, v in terms.items() if v is not None}
    total = sum(weights[k] for k in have)
    if not have or total <= 0:
        raise ValueError("no term with a positive weight is available")
    used = {k: weights[k] / total for k in have}
    score = 1.0 - sum(used[k] * have[k] for k in have)
    return float(min(1.0, max(0.0, score))), len(have) < len(terms), used


def f_term(fid: float, tau: float) -> float:
    """F = 1 - exp(-FID / tau) (equation 5)."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return float(1.0 - math.exp(-max(fid, 0.0) / tau))


def frechet_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6) -> float:
    """FID between two feature sets (n, d) (equation 4)."""
    from scipy import linalg

    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    s1 = np.atleast_2d(np.cov(f1, rowvar=False))
    s2 = np.atleast_2d(np.cov(f2, rowvar=False))
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():  # a singular product: nudge both diagonals
        off = np.eye(len(s1)) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = np.real(covmean)
    diff = mu1 - mu2
    return float(max(diff @ diff + np.trace(s1 + s2 - 2.0 * covmean), 0.0))


# ----------------------------------------------------------------------------
# Obstruction and distortion
# ----------------------------------------------------------------------------


def uv_to_texels(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """UVs (v up, as trimesh loads glTF) to texel coordinates (x right, y down)."""
    uv = np.asarray(uv, float)
    return np.c_[uv[:, 0] * width, (1.0 - uv[:, 1]) * height]


def chart_raster(tex: np.ndarray, faces: np.ndarray, shape: tuple[int, int]):
    """(face id per texel, -1 where no chart), texel centres inside each UV triangle."""
    import cv2

    ids = np.full(shape, -1, np.int32)
    shift = 4  # sub-texel precision: 1/16
    # a texel (i, j) covers [i, i+1); cv2 fills pixel centres, so move by half a texel
    pts = np.round((tex - 0.5) * (1 << shift)).astype(np.int32)
    for fi, f in enumerate(faces):
        cv2.fillPoly(ids, [pts[f].reshape(-1, 1, 2)], int(fi), shift=shift)
    return ids


def face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )


def no_chart(tex: np.ndarray, faces: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Triangles whose three UVs are one point (texturing's untextured faces)."""
    t = tex[faces]
    return np.ptp(t, axis=1).max(axis=1) < tol


def stretch(verts: np.ndarray, tex: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per triangle s = sigma1 / sigma2 of the 2x2 Jacobian of its surface -> texel map,
    in a local orthonormal frame of the triangle (inf when sigma2 is 0, nan when the
    triangle has no 3D area)."""
    P = verts[faces]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    n = np.cross(e1, e2)
    area2 = np.linalg.norm(n, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    ok = (area2 > 1e-12) & (l1 > 1e-12)
    s = np.full(len(faces), np.nan)
    if not ok.any():
        return s
    e1, e2, n, l1, area2 = e1[ok], e2[ok], n[ok], l1[ok], area2[ok]
    x = e1 / l1[:, None]
    y = np.cross(n / area2[:, None], x)
    Q = np.stack(
        [
            np.c_[l1, np.einsum("ij,ij->i", e2, x)],
            np.c_[np.zeros(len(l1)), np.einsum("ij,ij->i", e2, y)],
        ],
        axis=1,
    )  # columns: the edges in the local frame
    T = tex[faces[ok]]
    W = np.stack([T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]], axis=2)  # columns: UV edges
    J = W @ np.linalg.inv(Q)
    sv = np.linalg.svd(J, compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(
            sv[:, 1] > 1e-9 * np.maximum(sv[:, 0], 1e-30), sv[:, 0] / sv[:, 1], np.inf
        )
    s[ok] = r
    return s


def distortion(area: np.ndarray, s: np.ndarray, include: np.ndarray, s_max: float):
    """D (equation 3) over the included triangles with a 3D area, and the mask of the
    distorted ones."""
    use = include & np.isfinite(area) & (area > 0) & ~np.isnan(s)
    bad = use & (s > s_max)
    total = float(area[use].sum())
    return (float(area[bad].sum()) / total if total > 0 else None), bad, use


def coverage(
    ids: np.ndarray,
    valid: np.ndarray,
    tex: np.ndarray,
    faces: np.ndarray,
    charted: np.ndarray,
) -> np.ndarray:
    """Per triangle, the share of its chart's texels in M_res. A triangle too small to
    cover a texel centre takes the texel under its centroid; one with no chart: 0."""
    n = len(faces)
    inside = ids >= 0
    total = np.bincount(ids[inside], minlength=n).astype(float)
    good = np.bincount(ids[inside & valid], minlength=n).astype(float)
    small = charted & (total == 0)
    if small.any():
        c = tex[faces[small]].mean(axis=1)
        h, w = valid.shape
        x = np.clip(c[:, 0].astype(int), 0, w - 1)
        y = np.clip(c[:, 1].astype(int), 0, h - 1)
        total[small] = 1.0
        good[small] = valid[y, x]
    share = np.divide(good, total, out=np.zeros(n), where=total > 0)
    share[~charted] = 0.0
    return share


# ----------------------------------------------------------------------------
# FID: Inception-v3 features of random patches
# ----------------------------------------------------------------------------


def sample_patches(
    images: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    patch: int,
    rng: np.random.Generator,
    min_fill: float = MIN_FILL,
) -> np.ndarray:
    """k patches (k, patch, patch, 3) uint8 at random positions where at least min_fill
    of the patch is inside the image's mask, over (image RGB, mask bool) pairs (drawn
    with replacement; positions weighted by how many each image has)."""
    import cv2

    where = []
    for i, (img, mask) in enumerate(images):
        h, w = mask.shape
        if h < patch or w < patch:
            continue
        frac = cv2.boxFilter(
            mask.astype(np.float32),
            -1,
            (patch, patch),
            normalize=True,
            anchor=(0, 0),
            borderType=cv2.BORDER_CONSTANT,
        )[: h - patch + 1, : w - patch + 1]
        ys, xs = np.nonzero(frac >= min_fill - 1e-6)
        if len(ys):
            where.append((i, ys, xs))
    if not where:
        return np.zeros((0, patch, patch, 3), np.uint8)
    counts = np.array([len(ys) for _, ys, _ in where], float)
    pick = rng.choice(len(where), size=k, p=counts / counts.sum())
    out = np.empty((k, patch, patch, 3), np.uint8)
    for n, j in enumerate(pick):
        i, ys, xs = where[j]
        m = rng.integers(len(ys))
        y, x = ys[m], xs[m]
        out[n] = images[i][0][y : y + patch, x : x + patch, :3]
    return out


def feature_extractor(device: str | None = None):
    """A function: patches (n, p, p, 3) uint8 -> (n, 2048) Inception-v3 pool features
    (torchvision, ImageNet weights, downloaded once into TORCH_HOME). Raises when torch,
    torchvision or the weights aren't available."""
    import torch
    import torchvision

    from segmentation.model import select_device

    dev = select_device(device, allow_mps=False)
    weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(patches: np.ndarray, batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(patches), batch):
            x = torch.from_numpy(patches[i : i + batch]).to(dev)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(
                x,
                size=(INCEPTION_SIZE, INCEPTION_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            out.append(model((x - mean) / std).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)

    features.device = str(dev)  # type: ignore[attr-defined]
    return features


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


@dataclass
class TexturedPart:
    name: str
    vertices: np.ndarray  # (n, 3), node transform applied
    faces: np.ndarray
    uv: np.ndarray  # (n, 2), v up
    image: np.ndarray  # (H, W, 3) uint8 RGB


def load_textured_parts(glb: str | Path) -> list[TexturedPart]:
    """The buildings' textured meshes in the .glb (the ground, named "terrain" by the
    export, is left out), one per texture image."""
    import trimesh

    scene = trimesh.load(str(glb), force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname]
        if "terrain" in f"{node} {gname}".lower():
            continue
        if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
            continue
        vis = m.visual
        uv = getattr(vis, "uv", None)
        mat = getattr(vis, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if uv is None or img is None:
            continue
        verts = trimesh.transformations.transform_points(m.vertices, T)
        parts.append(
            TexturedPart(
                name=str(gname),
                vertices=np.asarray(verts, float),
                faces=np.asarray(m.faces, np.int64),
                uv=np.asarray(uv, float),
                image=np.asarray(img.convert("RGB"), np.uint8),
            )
        )
    return parts


def read_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """A binary mask image: its alpha channel when it has one, else non-zero pixels."""
    from PIL import Image

    im = Image.open(path)
    a = np.asarray(im.getchannel("A") if "A" in im.getbands() else im.convert("L"))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError(
            f"{path}: mask is {a.shape[1]}x{a.shape[0]}, texture {shape[1]}x{shape[0]}"
        )
    return a > 127


def valid_texels(image: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """ValidTexels: the alpha channel when given, else every texel that isn't
    background (the colour of the image's corners, e.g. texturing's grey)."""
    if alpha is not None:
        return alpha > 127
    corners = np.array(
        [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]], dtype=np.int32
    )
    vals, counts = np.unique(corners, axis=0, return_counts=True)
    bg = vals[counts.argmax()]
    return ~np.all(image.astype(np.int32) == bg, axis=2)


def reference_photos(folder: Path) -> list[Path]:
    """The job's saved street photos (images/*_original.jpg), else any jpg/png."""
    if not folder.is_dir():
        return []
    found = sorted(folder.glob("*_original.jpg"))
    return found or sorted(
        p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def find_model(path=None, latest=False, jobs_dir="output/jobs") -> Path:
    """The model .glb: a .glb, a folder holding model.glb, or the newest job's."""
    if latest:
        found = sorted(
            Path(jobs_dir).glob("*/model.glb"), key=lambda q: q.stat().st_mtime
        )
        if not found:
            raise Unavailable(f"no {jobs_dir}/*/model.glb: run the pipeline first")
        return found[-1]
    q = Path(path)
    q = q / "model.glb" if q.is_dir() else q
    if not q.exists():
        raise Unavailable(f"{q} not found")
    return q


# ----------------------------------------------------------------------------
# The evaluation
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model", nargs="?", help="a .glb, or a job folder holding model.glb")
    p.add_argument(
        "--latest", action="store_true", help="the newest output/jobs/*/model.glb"
    )
    p.add_argument(
        "--out", help="output folder (default: texture_eval/ next to the .glb)"
    )
    p.add_argument(
        "--alpha", type=float, default=ALPHA, help="weight of O (obstruction)"
    )
    p.add_argument("--beta", type=float, default=BETA, help="weight of D (distortion)")
    p.add_argument("--gamma", type=float, default=GAMMA, help="weight of F (realism)")
    p.add_argument(
        "--s-max", type=float, default=S_MAX, help="stretch counted as distorted"
    )
    p.add_argument("--tau", type=float, default=TAU, help="FID scale (calibrate it)")
    p.add_argument("--k", type=int, default=K, help="patches per texture for the FID")
    p.add_argument("--patch", type=int, default=PATCH, help="patch side in texels")
    p.add_argument("--seed", type=int, default=SEED, help="random seed for the patches")
    p.add_argument(
        "--device", default="auto", help="Inception's device: auto, cpu, cuda"
    )
    p.add_argument("--no-fid", action="store_true", help="skip the FID (O and D only)")
    p.add_argument(
        "--reference", help="folder of reference photos (default: the job's images/)"
    )
    p.add_argument(
        "--gt-texture",
        help="a ground-truth texture of the same size (the paper's setting)",
    )
    p.add_argument(
        "--gt-mask", help="its valid-texel mask (default: alpha, or non-background)"
    )
    p.add_argument(
        "--res-mask", help="the result's valid-texel mask (default: texturing's)"
    )
    return p


def settings(**overrides: Any) -> argparse.Namespace:
    """The command line's defaults with `overrides` (for the pipeline stage)."""
    a = build_parser().parse_args([])
    for k, v in overrides.items():
        if not hasattr(a, k):
            raise TypeError(f"unknown texture evaluation setting {k!r}")
        setattr(a, k, v)
    return a


def run(a: argparse.Namespace, glb: str | Path, out: str | Path) -> dict[str, Any]:
    """Evaluate `glb` with the settings `a`, writing every output to `out`. Returns the
    compact summary (web_summary), also in summary.json's "web" member."""
    t0 = time.perf_counter()
    check_weights(a.alpha, a.beta, a.gamma)
    if a.s_max < 1:
        raise ValueError("s_max must be >= 1")
    glb, out = Path(glb), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts = load_textured_parts(glb)
    if not parts:
        raise Unavailable(f"{glb.name} has no textured building mesh to evaluate")
    _log(
        f"[texture] {glb.name}: "
        + ", ".join(
            f"{p.name} ({len(p.faces):,} faces, {p.image.shape[1]}x{p.image.shape[0]} atlas)"
            for p in parts
        )
    )

    gt_mode = bool(a.gt_texture)
    if gt_mode and len(parts) != 1:
        raise Unavailable("--gt-texture needs a model with one texture atlas")
    mask_file = glb.parent / (glb.stem + MASK_SUFFIX)

    rows: list[dict[str, Any]] = []
    per_atlas = []
    res_images, gt_images = [], []
    s_all, a_all, use_all, bad_all = [], [], [], []
    area_tot = area_good = 0.0
    tex_gt = tex_hit = 0
    counts = {
        "faces": 0,
        "no_chart": 0,
        "zero_area": 0,
        "zero_uv_area": 0,
        "distorted": 0,
    }
    mask_source = None
    group_base = 0
    for part in parts:
        H, W = part.image.shape[:2]
        tex = uv_to_texels(part.uv, W, H)
        faces = part.faces
        area = face_areas(part.vertices, faces)
        charted = ~no_chart(tex, faces)
        ids = chart_raster(tex, faces, (H, W))

        # the masks
        if gt_mode:
            from PIL import Image

            gt_im = Image.open(a.gt_texture)
            gt_rgb = np.asarray(gt_im.convert("RGB"), np.uint8)
            if gt_rgb.shape[:2] != (H, W):
                raise Unavailable(
                    f"--gt-texture is {gt_rgb.shape[1]}x{gt_rgb.shape[0]}, the model's atlas {W}x{H}"
                )
            m_gt = (
                read_mask(a.gt_mask, (H, W))
                if a.gt_mask
                else valid_texels(
                    gt_rgb,
                    np.asarray(gt_im.getchannel("A"))
                    if "A" in gt_im.getbands()
                    else None,
                )
            )
            gt_images.append((gt_rgb, m_gt))
        else:
            m_gt = ids >= 0
        if a.res_mask:
            m_res = read_mask(a.res_mask, (H, W))
            mask_source = "given"
        elif len(parts) == 1 and mask_file.exists():
            m_res = read_mask(mask_file, (H, W))
            mask_source = "texturing"
        else:
            m_res = ~np.all(part.image == UNTEXTURED_GREY, axis=2)
            mask_source = "colour"
        if not gt_mode:
            m_res = m_res & m_gt
        res_images.append((part.image, m_res))

        # O
        tex_gt += int(m_gt.sum())
        tex_hit += int((m_res & m_gt).sum())
        share = coverage(ids, m_res, tex, faces, charted)
        has_area = area > 1e-12
        area_tot += float(area[has_area].sum())
        area_good += float((area * share)[has_area].sum())

        # D
        s = stretch(part.vertices, tex, faces)
        d_part, bad, use = distortion(area, s, charted, a.s_max)
        counts["faces"] += len(faces)
        counts["no_chart"] += int((~charted & has_area).sum())
        counts["zero_area"] += int((~has_area).sum())
        counts["zero_uv_area"] += int((charted & has_area & np.isinf(s)).sum())
        counts["distorted"] += int(bad.sum())
        s_all.append(s[use])
        a_all.append(area[use])
        use_all.append(use)
        bad_all.append(bad)

        # per group of connected triangles (about one building each)
        groups = face_groups(part.vertices, faces)
        for g in np.unique(groups):
            sel = (groups == g) & has_area
            ga = float(area[sel].sum())
            if ga <= 0:
                continue
            gu = sel & use
            ctr = (part.vertices[faces[sel]].mean(axis=1) * area[sel, None]).sum(0) / ga
            rows.append(
                {
                    "group": group_base + int(g),
                    "part": part.name,
                    "faces": int(sel.sum()),
                    "area_m2": round(ga, 2),
                    "x_m": round(float(ctr[0]), 1),
                    "y_m": round(float(-ctr[2]), 1),  # glTF: -z is north
                    "obstruction": round(
                        1.0 - float((area * share)[sel].sum()) / ga, 4
                    ),
                    "distortion": round(
                        float(area[gu & bad].sum()) / float(area[gu].sum()), 4
                    )
                    if area[gu].sum() > 0
                    else None,
                    "no_chart_faces": int((sel & ~charted).sum()),
                }
            )
        group_base += int(groups.max()) + 1 if len(groups) else 0
        per_atlas.append(
            {
                "part": part.name,
                "width": W,
                "height": H,
                "chart_texels": int(m_gt.sum()),
                "photo_texels": int((m_res & m_gt).sum()),
                "_cov": (part, m_gt, m_res),  # for the coverage figure, not saved
            }
        )

    if area_tot <= 0:
        raise Unavailable("the textured mesh has no surface area")
    o_area = 1.0 - area_good / area_tot
    o_texels = 1.0 - tex_hit / tex_gt if tex_gt else None
    o_term = o_texels if gt_mode else o_area
    s_cat, a_cat = np.concatenate(s_all), np.concatenate(a_all)
    d_term = (
        float(a_cat[s_cat > a.s_max].sum() / a_cat.sum()) if a_cat.sum() > 0 else None
    )
    _log(
        f"[texture] O (obstruction) {o_term:.3f}"
        + (
            f"  (by texels: {o_texels:.3f})"
            if o_texels is not None and not gt_mode
            else ""
        )
        + f"  ·  D (distortion) {'—' if d_term is None else f'{d_term:.3f}'}"
        f" (s > {a.s_max:g}: {counts['distorted']:,} of {sum(int(u.sum()) for u in use_all):,} triangles)"
    )
    if counts["no_chart"]:
        _log(
            f"[texture] {counts['no_chart']:,} triangles have no texture (no photo saw them): counted by O"
        )

    # F
    fid = f = None
    fid_reason = None
    ref_desc = None
    device = None
    if a.no_fid:
        fid_reason = "turned off (--no-fid)"
    else:
        rng = np.random.default_rng(a.seed)
        if gt_mode:
            ref = gt_images
            ref_desc = f"ground-truth texture {Path(a.gt_texture).name}"
        else:
            folder = Path(a.reference) if a.reference else glb.parent / "images"
            photos = reference_photos(folder)
            ref = []
            for p in photos:
                from PIL import Image

                rgb = np.asarray(Image.open(p).convert("RGB"), np.uint8)
                ref.append((rgb, np.ones(rgb.shape[:2], bool)))
            ref_desc = f"{len(ref)} street photo{'s' if len(ref) != 1 else ''} from {folder.name}/"
        p_res = sample_patches(res_images, a.k, a.patch, rng)
        p_ref = sample_patches(ref, a.k, a.patch, rng) if ref else np.zeros((0,))
        if not ref:
            fid_reason = "no reference photos (the job's images/ folder is empty; pass --reference)"
        elif len(p_res) < 2 or len(p_ref) < 2:
            fid_reason = f"no {a.patch}x{a.patch} patch fits inside the {'texture' if len(p_res) < 2 else 'reference'}'s mask"
        else:
            try:
                _log(
                    f"[texture] FID: Inception-v3 features of {a.k:,} patches each (texture, {ref_desc})"
                )
                extract = feature_extractor(a.device)
                device = getattr(extract, "device", None)
                f_res = extract(p_res)
                f_ref = extract(p_ref)
                fid = frechet_distance(
                    f_res.astype(np.float64), f_ref.astype(np.float64)
                )
                f = f_term(fid, a.tau)
            except Exception as exc:  # noqa: BLE001 - no torch, no weights: F is left out
                fid_reason = f"Inception-v3 not available ({type(exc).__name__}: {exc})"
    if fid_reason:
        _log(
            f"[texture] FID left out: {fid_reason}. The confidence is partial (O and D only).",
            warning=True,
        )
    else:
        _log(f"[texture] FID {fid:.1f}  ->  F = 1 - exp(-FID / {a.tau:g}) = {f:.3f}")

    weights = {"O": a.alpha, "D": a.beta, "F": a.gamma}
    terms = {"O": o_term, "D": d_term, "F": f}
    conf, partial, used = confidence(terms, weights)
    _log(
        f"[texture] Confidence {conf:.3f}"
        + (
            " (partial: "
            + ", ".join(k for k, v in terms.items() if v is None)
            + " left out)"
            if partial
            else ""
        )
    )

    summary = {
        "model": str(glb),
        "mode": "ground truth" if gt_mode else "pipeline",
        "confidence": round(conf, 4),
        "partial": partial,
        "terms": {k: None if v is None else round(v, 4) for k, v in terms.items()},
        "o_area": round(o_area, 4),
        "o_texels": None if o_texels is None else round(o_texels, 4),
        "fid": None if fid is None else round(fid, 3),
        "fid_reason": fid_reason,
        "tau": a.tau,
        "weights": weights,
        "weights_used": {k: round(v, 4) for k, v in used.items()},
        "s_max": a.s_max,
        "k": a.k,
        "patch": a.patch,
        "seed": a.seed,
        "device": device,
        "reference": ref_desc,
        "mask_source": mask_source,
        "counts": counts,
        "atlases": [
            {k: v for k, v in d.items() if not k.startswith("_")} for d in per_atlas
        ],
        "seconds": None,
    }
    write_outputs(out, summary, rows, per_atlas, s_cat, a_cat, a)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    web = web_summary(summary, rows, out)
    doc = {**summary, "web": web}
    (out / "summary.json").write_text(
        json.dumps(doc, indent=2, default=str), encoding="utf-8"
    )
    _log(f"[texture] wrote {out} in {summary['seconds']:.1f} s")
    return web


def face_groups(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """A group per set of triangles connected through shared corners (positions, since
    the textured mesh has its own vertices per triangle): about one building each."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(faces):
        return np.zeros(0, int)
    _, inv = np.unique(np.round(verts, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1)[faces]
    n_v, n_f = int(inv.max()) + 1, len(faces)
    rows = np.repeat(np.arange(n_f), 3)
    g = coo_matrix((np.ones(3 * n_f), (rows, f.reshape(-1))), shape=(n_f, n_v)).tocsr()
    # faces and vertices as one graph: faces 0..n_f-1, vertices n_f..
    from scipy.sparse import bmat

    adj = bmat([[None, g], [g.T, None]])
    _, labels = connected_components(adj, directed=False)
    face_labels = labels[:n_f]
    _, compact = np.unique(face_labels, return_inverse=True)
    return compact


def web_summary(
    summary: dict[str, Any], rows: list[dict[str, Any]], out: Path
) -> dict[str, Any]:
    """The headline results, small enough to keep on the job."""
    worst = sorted(
        (r for r in rows if r["area_m2"] >= 5),
        key=lambda r: (-(r["obstruction"] or 0), -(r["distortion"] or 0)),
    )[:5]
    return {
        "status": "done",
        "confidence": summary["confidence"],
        "partial": summary["partial"],
        "terms": summary["terms"],
        "o_texels": summary["o_texels"],
        "fid": summary["fid"],
        "fid_reason": summary["fid_reason"],
        "tau": summary["tau"],
        "weights": summary["weights"],
        "weights_used": summary["weights_used"],
        "s_max": summary["s_max"],
        "reference": summary["reference"],
        "mask_source": summary["mask_source"],
        "counts": summary["counts"],
        "groups": len(rows),
        "worst": worst,
        "seconds": summary["seconds"],
        "files": [n for n in OUTPUT_FILES if (out / n).exists() or n == "summary.json"],
    }


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BAR = "#2a78d6"  # one series: a single hue
MISSING = "#d03b3b"  # status: texels that should hold photo texture and don't


def write_outputs(out, summary, rows, per_atlas, s, area, a):
    with open(out / "per_group.csv", "w", newline="", encoding="utf-8") as fh:
        cols = [
            "group",
            "part",
            "faces",
            "area_m2",
            "x_m",
            "y_m",
            "obstruction",
            "distortion",
            "no_chart_faces",
        ]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["area_m2"]):
            w.writerow(r)
    write_markdown(out / "summary.md", summary, rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        make_plots(out, summary, per_atlas, s, area, a)
    except ImportError as exc:
        _log(f"[texture] no figures: {exc}", warning=True)


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def write_markdown(
    path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    t, c = summary["terms"], summary["counts"]
    wu = summary["weights_used"]
    L = [
        "# UV texture quality",
        "",
        f"**Confidence {summary['confidence']:.3f}**"
        + (" (partial: F left out, O and D reweighted)" if summary["partial"] else "")
        + f" = 1 − ({' + '.join(f'{wu[k]:.2f}·{k}' for k in wu)})",
        "",
        "| Term | Value | Weight | What it measures |",
        "|---|---|---|---|",
        f"| O obstruction | {_fmt(t['O'])} | {summary['weights']['O']:.2f} | surface without photo texture |",
        f"| D distortion | {_fmt(t['D'])} | {summary['weights']['D']:.2f} | surface stretched more than {summary['s_max']:g}× |",
        f"| F realism | {_fmt(t['F'])} | {summary['weights']['F']:.2f} | 1 − exp(−FID/τ), FID {_fmt(summary['fid'], 1)}, τ {summary['tau']:g} |",
        "",
        "## How it was measured",
        "",
        f"- Mode: {summary['mode']}; texel mask from {summary['mask_source']}",
        f"- O by surface area {_fmt(summary['o_area'])}; by texels (the paper's count) {_fmt(summary['o_texels'])}",
        f"- Triangles: {c['faces']:,}; without texture {c['no_chart']:,}; zero area {c['zero_area']:,}; "
        f"chart of zero area {c['zero_uv_area']:,}; distorted {c['distorted']:,}",
        f"- FID reference: {summary['reference'] or '—'}; {summary['k']:,} patches of {summary['patch']} texels, seed {summary['seed']}"
        + (f"; device {summary['device']}" if summary["device"] else ""),
    ]
    if summary["fid_reason"]:
        L.append(f"- FID left out: {summary['fid_reason']}")
    L.append(
        "- τ is a starting value: calibrate it as the median FID of good reference results"
    )
    worst = sorted(
        (r for r in rows if r["area_m2"] >= 5), key=lambda r: -(r["obstruction"] or 0)
    )[:10]
    if worst:
        L += [
            "",
            "## Groups with the most missing texture",
            "",
            "| Group | Area (m²) | Position (m, from the model centre) | O | D |",
            "|---|---|---|---|---|",
        ]
        for r in worst:
            L.append(
                f"| {r['group']} | {r['area_m2']:.0f} | {r['x_m']:+.0f} E, {r['y_m']:+.0f} N | "
                f"{r['obstruction']:.2f} | {_fmt(r['distortion'], 2)} |"
            )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _style(plt):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _head(fig, title, note):
    fig.text(
        0.02,
        0.975,
        title,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.02, 0.915, note, ha="left", va="top", fontsize=9.5, color=INK2)


def make_plots(out, summary, per_atlas, s, area, a):
    import matplotlib.pyplot as plt

    _style(plt)
    # 01: the three terms (penalties, lower is better)
    t = summary["terms"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.72, bottom=0.18)
    names = [("O", "Obstruction"), ("D", "Distortion"), ("F", "Realism (F)")]
    for i, (k, label) in enumerate(names):
        v = t[k]
        if v is None:
            ax.text(0.01, i, "not available", va="center", color=INK2, fontsize=9.5)
            continue
        ax.barh(i, v, height=0.5, color=BAR)
        ax.text(
            min(v, 1) + 0.01,
            i,
            f"{v:.2f} × {summary['weights_used'].get(k, 0):.2f}",
            va="center",
            color=INK,
            fontsize=9.5,
        )
    ax.set_yticks(range(3), [n for _, n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("penalty (0 best, 1 worst) × weight used")
    ax.grid(axis="y", visible=False)
    _head(
        fig,
        f"Confidence {summary['confidence']:.2f}"
        + (" (partial)" if summary["partial"] else ""),
        "Confidence = 1 − the weighted sum of the three penalties",
    )
    fig.savefig(out / "01_scorecard.png", dpi=150)
    plt.close(fig)

    # 02: where the texture is missing, on the atlas
    for d in per_atlas[:1]:
        part, m_gt, m_res = d["_cov"]
        img = part.image.astype(np.float32)
        step = max(1, int(math.ceil(img.shape[1] / 1600)))
        img, m_gt_s, m_res_s = (
            img[::step, ::step],
            m_gt[::step, ::step],
            m_res[::step, ::step],
        )
        shown = img.copy()
        shown[~m_gt_s] = 0.3 * shown[~m_gt_s] + 0.7 * 255  # outside the charts: faded
        miss = m_gt_s & ~m_res_s
        shown[miss] = np.array([0xD0, 0x3B, 0x3B], np.float32)
        h, w = shown.shape[:2]
        fig = plt.figure(figsize=(8, max(2.5, 8 * h / w + 1.0)))
        ax = fig.add_axes((0.02, 0.02, 0.96, 0.96 - 1.0 / max(2.5, 8 * h / w + 1.0)))
        ax.imshow(shown.astype(np.uint8), interpolation="nearest")
        ax.axis("off")
        _head(
            fig,
            f"Texture without photo: {100 * (1 - d['photo_texels'] / max(d['chart_texels'], 1)):.0f}% of the atlas's chart texels",
            "Red: texels that should hold photo texture and don't · faded: outside the UV charts",
        )
        fig.savefig(out / "02_coverage.png", dpi=150)
        plt.close(fig)

    # 03: stretch histogram (area-weighted)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.75, bottom=0.17)
    top = max(4.0, a.s_max * 1.5)
    sv = np.clip(np.where(np.isfinite(s), s, top), 1.0, top)
    if len(sv):
        bins = np.geomspace(1.0, top, 25)
        ax.hist(
            sv,
            bins=bins,
            weights=area / max(area.sum(), 1e-12) * 100,
            color=BAR,
            edgecolor=SURFACE,
            linewidth=1,
        )
    ax.set_xscale("log")
    ticks = [1, 1.25, a.s_max, 2, 3, top]
    ax.set_xticks(
        sorted(set(ticks)),
        [f"{v:g}" + ("+" if v == top else "") for v in sorted(set(ticks))],
    )
    ax.minorticks_off()
    ax.axvline(a.s_max, color=INK, linewidth=1.5, linestyle="--")
    ax.text(
        a.s_max * 1.02,
        ax.get_ylim()[1] * 0.92,
        f"s_max {a.s_max:g}",
        color=INK,
        fontsize=9,
    )
    ax.set_xlabel("stretch σ1/σ2 (1 = no stretch)")
    ax.set_ylabel("% of the surface")
    d_txt = (
        "—" if summary["terms"]["D"] is None else f"{100 * summary['terms']['D']:.1f}%"
    )
    _head(
        fig,
        f"Stretched texture: {d_txt} of the surface beyond s_max",
        "Each triangle's texture stretch, weighted by its area; right of the dashed line counts as distorted",
    )
    fig.savefig(out / "03_stretch.png", dpi=150)
    plt.close(fig)


def main() -> None:
    a = build_parser().parse_args()
    if not (a.model or a.latest):
        build_parser().error("give a model (.glb or job folder) or --latest")
    glb = find_model(a.model, a.latest)
    out = Path(a.out) if a.out else glb.parent / EVALUATION_DIR
    run(a, glb, out)


if __name__ == "__main__":
    main()
