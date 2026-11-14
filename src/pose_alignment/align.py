"""Stage 6 (align): ground-plane projection and alignment to the OpenStreetMap footprints.

Per leg and sequence model:
  1. Level with GeoCalib's gravity: each camera's down direction (GeoCalib, camera frame) turned
     into the model's frame by that camera's rotation; their robust mean is the model's down.
     Plan view: x right of travel, y along it, z up.
  2. Ground plane: the median height of the road points (segmentation) near the camera path.
  3. Building points: points labelled Building in most of their observations, seen in 3+
     frames, 2.5-40 m above the ground, thinned to one per 0.5 m cell in plan view.
  4. GPS placement: a 2D similarity of the levelled camera path onto the GPS track (Mapillary's
     computed positions, or per leg the measured GPS where they look broken: --gps auto) gives the
     scale, which is kept, and the starting heading and position (with weak priors in the fit).
  5. Fit: a global search over heading (+-15 deg, or +-1 deg around the facade heading) and
     position (+-40 m) for the placement whose building points lie closest to the OSM footprint
     edges (cameras inside buildings penalised), then a robust (Cauchy) least-squares refinement
     of point-to-edge distances. The scale stays the GPS placement's (x1.00 against it): the leg is
     turned and shifted onto the footprints but never stretched (fitting the scale too, the earlier
     method, let short or ambiguous legs shrink or grow onto the wrong walls). One placement per leg, or per piece when the leg
     was reconstructed with --split-weak: the leg is cut where reconstruct.py dropped weakly
     anchored frames (a gap the images do not hold together; a frame
     Mapillary skipped is not a cut), pieces under 8 cameras join a neighbour, and each 3D point
     goes with the piece whose cameras observe it most. stats.json reports each cut's aligned
     length next to the GPS distance (a yardstick only) and the model's neighbouring speed.
     Rig cameras (reconstruct.py's rigs: one model for all cameras of a leg) are placed together:
     the reference camera's fit uses the building points of every rig camera, levelled with its
     own gravity and placed from its own GPS path, and the other cameras get that transform
     instead of a fit of their own (roi_11's right-side camera, fitted alone, locked onto the
     wrong buildings 18-26 m off while the left cameras of the same model fitted to 0.9 m).
     Linked legs (reconstruct.py's leg overlap: a leg runs on into the next one's first frames):
     after every leg is fitted on its own, the fits of legs sharing frames are refined together,
     each still on its own building points, plus the condition that the shared cameras land in the
     same place from both legs (LINK_SIGMA). That lines the legs up end to end and carries their
     scales across (a leg whose walls leave it free along the street is held by its neighbours).
     All legs are placed in one ENU frame (the route's first point).
     summary.json records the procedure ("procedure": walls, heading, gps). The defaults, which
     run_pipeline.py and the web app use: --walls hybrid --heading facades --gps auto.
     --walls oriented: the building points are matched to footprint edges by
     direction. Each point's facade direction comes from its neighbours (plan view, a line through
     the points within 2 m); only points on such lines take part, each matches only walls within
     30 deg of its own direction, points up to 12 m from such a wall still count (a corner left in
     a gap between buildings costs its distance instead of being dropped), and facades across the
     street count 3x (only they hold the leg along the street; long frontages leave it free to
     slide). The position search reaches +-40 m around the GPS
     placement with a weak prior: roi_19 leg04's corners sat one building plot (~20 m) south of
     theirs, where the nearest-wall cost could not tell them from the true place.
     --heading facades: the heading from the walls first. The facade-line directions of the
     building points (plan, levelled model frame) and the directions of the OSM walls within 40 m
     of the GPS-placed path, each a histogram over 0-180 deg (1 deg kernel; walls weighted by
     length), are cross-correlated over rotations within +-20 deg of the GPS heading (so a street
     is not mistaken for its cross streets, 90 deg off); the best rotation is the heading. The fit
     then searches and refines the heading only within +-1 deg of it (position and the rest as
     before; leg linking too). Long facades fix this angle independently of where along the street
     the leg sits: roi_19 leg04's three reconstructions gave -1.0/-0.9/-0.8 deg from the facades
     while the free fit had turned them -1.5/-3.1/-6.1 deg. Used only when the facade directions
     give a clearly unique answer: at least 30 facade-line points, and the best rotation more than
     3 deg from the peak scoring at most 50% of it. Less distinct peaks (58-95%) were off by 3-5 deg
     on roi_27, roi_22 leg01 and roi_28 leg02 against the placements judged right, so there the
     heading stays free as before and the reliability block notes it. stats.json pieces:
     heading_from_facades.
     --heading-prior DEG: a loose prior holding each leg's heading to its GPS placement's (the
     direction of Mapillary's path): in the search 0.5 (dheading / DEG)^2 is added to the score (in
     metres of mean wall distance, like the position prior), and in the refinement and the leg
     linking a residual sqrt(n) dheading / DEG. Off by default (heading free within the +-15 deg
     search). roi_19 leg04 (SIFT, ~100 building points) turned 6 deg off with the focal held, its
     ends ~8 m sideways, and 1.5 vs 3.1 deg between two otherwise equal runs.
     --gps measured: the GPS placement (step 4: the scale, the starting heading and position,
     the priors) from each frame's measured GPS (Mapillary's "geometry")
     instead of Mapillary's computed positions ("computed_geometry", its own reconstruction; the
     default).
  6. Reliable when every check passes (stats.json "reliability": the values and the failed checks):
       - fit: the building points within 25 m of the path that have an OSM wall within 6 m lie
         within --max-error (2 m) of it (median). Far facades are triangulated much less precisely
         (roi_23 leg03: 0.35 m near the path, 5.5 m beyond 25 m), and points with no wall nearby
         are buildings missing from OSM or mislabelled points, not placement error;
       - coverage: at least 30 such nearby points, and at least half of the nearby building points
         have a wall within 6 m;
       - unambiguous along the street: sliding the placed leg along its path (+-40 m), the fit's own
         cost (direction-matched facade points when they exist, else nearest wall) has its best
         within 3 m of the placement and beats the best placement more than 6 m away by 10%
         (roi_19 leg04 had been placed one building plot, ~20 m, off: 1.55 vs 1.54);
       - at most 20% of the cameras inside buildings, and no camera behind an earlier one along
         the leg (a folded reconstruction; stats.json lists them as cameras_out_of_order).
     A rig's cameras share one placement and so one verdict, from the pooled points of all of
     them (the reference camera's). The old verdict (median of all building points within
     --max-error, inside, order) is kept as reliable_old.
Writes data/alignment/: <leg>_<folder>/{stats.json, transform.json, plan.png}, <folder>.json (aligned
cameras: lon, lat, height, heading, pitch, roll), summary.json; results/overview.png and
fit_<leg>_<folder>.png (plot_fit.py: the sparse cloud on the footprints, plan and 3D views).

Usage: python -m pose_alignment.align outputs/alignment/mapillary_roi_2
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from .geo_common import (
    MASK_BUILDING,
    MASK_ROAD,
    LocalFrame,
    RoiPaths,
    apply_2d,
    label_points,
    load_cameras,  # noqa: E402
    load_points,
    plan_onto,
    read_camera,
    segment_distances,
    similarity_2d,
    steps_against_track,
)
from .osm import WallMap, building_heights, fetch_buildings, side_of_path  # noqa: E402

OPT = {
    "min_track": 3,
    "min_height": 2.5,
    "max_height": 40.0,
    "class_fraction": 0.5,
    "point_grid": 0.5,
    "search_points": 800,
    "search_deg": 15.0,
    "search_m": 20.0,
    "shift_prior": 15.0,
    "match_dist": 6.0,
    "wall_scale": 0.75,
    "gps_sigma": 25.0,
    "margin": 60.0,
    "default_camera_height": 1.6,
    # scale vs the GPS placement's: searched range, prior std (log) and refinement freedom, with
    # building points on both sides of the street (scale observable) / on one side only
    "scale_sigma_both": 0.3,
    "scale_sigma_one": 0.1,
}


# labels() cache: (model, masks, value, points3D.txt mtime) -> label_points result
_LABELS: dict[tuple, tuple] = {}
# The scale is always the GPS placement's: the fit turns and shifts a leg onto the footprints but never
# stretches it (the earlier fit also fitted the scale to the walls; FIXED holds it where it starts)
FIXED = 1e-9  # bounds half-width (log scale) that holds the scale where it starts
# --walls: "hybrid" (default), "oriented" (see the docstring) or "nearest" (each building point to its
# nearest footprint edge)
WALLS = {"mode": "hybrid"}
# --gps: "computed" (Mapillary's computed positions), "measured" (the frames' measured GPS) or "auto"
# (per leg: computed unless gps_check finds them broken, then measured)
GPS = {"source": "auto"}
# --gps auto: a leg's computed positions count as broken when, for any of its cameras, their end-to-end
# length differs from the measured GPS's by more than length_tol, or when two cameras of a rig, over the
# time both cover (at least min_overlap_s), travel computed distances more than rig_tol apart. roi_11's
# Suzuki rig (three oblique fisheyes): measured / computed 0.86-1.53 per leg, and on one leg one camera
# 189 m against another's 127 m in the same 18 s; phone legs of six other ROIs: 0.91-1.13, single
# cameras (no rig check). Measured GPS for the whole ROI: roi_11 reliable 266 -> 478 (ALIKED + NN), but
# another ROI 81 -> 60, so per leg only where the check fails; with auto only roi_11's 5 rig legs switched
GPS_CHECK = {"length_tol": 0.15, "rig_tol": 0.10, "min_overlap_s": 5.0}
GPS_LEG = {}  # leg id -> {"source": "computed" | "measured", "reasons": [...]} (--gps auto)
GPS_MEASURED_FOLDERS = (
    set()
)  # sequences of the legs --gps auto placed by measured GPS (overview's x marks)
HEADING = {"prior_deg": None, "mode": "facades"}  # --heading-prior, --heading
FACADE_HDG = {
    "range_deg": 20.0,
    "hold_deg": 1.0,
    "kernel_deg": 1.0,
    "near_m": 40.0,
    "min_points": 30,
    "separation_deg": 3.0,
    "max_second": 0.5,
}


def facade_heading(dirs, cams_gps, ang0, A, B):
    """The model's heading from its walls (module docstring, --heading facades): (absolute angle in
    radians or None, info). dirs: facade-line directions of the building points in the levelled
    plan frame (NaN: none); cams_gps: the GPS-placed camera path (ENU); ang0: the GPS placement's
    heading (radians); A, B: footprint edges."""
    line = ~np.isnan(dirs[:, 0])
    info = {"facade_points": int(line.sum())}
    if line.sum() < FACADE_HDG["min_points"]:
        info["unconstrained"] = f"only {int(line.sum())} facade-line points"
        return None, info
    mid = (A + B) / 2
    from scipy.spatial import cKDTree

    near = cKDTree(cams_gps).query(mid)[0] < FACADE_HDG["near_m"]
    v = (B - A)[near]
    if len(v) < 2:
        info["unconstrained"] = "no OSM walls near the path"
        return None, info
    bins = np.arange(0.0, 180.0, 0.1)
    k = FACADE_HDG["kernel_deg"]

    def hist(ang, w):
        d = (bins[None, :] - ang[:, None] + 90.0) % 180.0 - 90.0
        return (w[:, None] * np.exp(-0.5 * (d / k) ** 2)).sum(axis=0)

    hm = hist(
        np.degrees(np.arctan2(dirs[line, 1], dirs[line, 0])) % 180.0,
        np.ones(int(line.sum())),
    )
    hw = hist(
        np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 180.0, np.linalg.norm(v, axis=1)
    )
    a0 = math.degrees(ang0)
    rots = np.arange(-FACADE_HDG["range_deg"], FACADE_HDG["range_deg"] + 1e-9, 0.1)
    score = np.array(
        [np.dot(np.roll(hm, int(round((a0 + r) * 10)) % len(bins)), hw) for r in rots]
    )
    i = int(np.argmax(score))
    far = np.abs(rots - rots[i]) > FACADE_HDG["separation_deg"]
    second = float(score[far].max() / score[i]) if far.any() and score[i] > 0 else 1.0
    info.update(
        deg_vs_gps=float(rots[i]),
        second_peak_ratio=second,
        second_peak_deg_vs_gps=float(rots[far][np.argmax(score[far])])
        if far.any()
        else None,
    )
    if second > FACADE_HDG["max_second"]:
        info["unconstrained"] = (
            f"no clear facade direction: {info['second_peak_deg_vs_gps']:+.1f} deg scores "
            f"{100 * second:.0f}% of {rots[i]:+.1f} deg"
        )
        return None, info
    return math.radians(a0 + rots[i]), info


# --walls hybrid: as oriented, plus the building points with no facade direction (blobs, clutter, corners),
# each scored by its plain distance to the nearest wall, capped at cap_m, at weight x a line point's.
# in_check: the along-street uniqueness check scores them the same way. max_move_m: the direction-matched
# fit places the leg first; the hybrid may then move its cameras' centre at most this far from there.
# Tested on 8 ROIs with ALIKED + NN (no reliable frame lost; roi_11 +59): an open lot's leg whose main
# cluster is a blob, not a line, moved 15 m onto its wall (fit 3.21 -> 1.74 m); a cap of 3 m let the blobs
# pull too little; without the move limit, legs on a uniform frontage slid 37-40 m (the search's edge)
HYBRID = {"weight": 0.5, "cap_m": 12.0, "in_check": True, "max_move_m": 20.0}
ORIENTED = {
    "radius_m": 2.0,
    "min_neighbours": 6,
    "line_ratio": 4.0,
    "tol_deg": 30.0,
    "cap_m": 12.0,
    "across_weight": 3.0,
    "min_points": 30,
    "bins": 12,
    "search_m": 40.0,
    "shift_prior": 40.0,
    "gps_sigma": 60.0,
}


def facade_dirs(X, radius):
    """Per point (plan coordinates X), the unit direction of the facade line through its neighbours
    within radius (principal axis), NaN where they do not form a line."""
    from scipy.spatial import cKDTree

    out = np.full((len(X), 2), np.nan)
    tree = cKDTree(X)
    for i, nb in enumerate(tree.query_ball_point(X, radius)):
        if len(nb) >= ORIENTED["min_neighbours"]:
            Y = X[nb] - X[nb].mean(axis=0)
            ev, evec = np.linalg.eigh(Y.T @ Y)
            if ev[1] >= ORIENTED["line_ratio"] * max(ev[0], 1e-12):
                out[i] = evec[:, 1]
    return out


def rotate_dirs(D, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    return D @ np.array([[c, -s_], [s_, c]]).T


class OrientedWalls:
    """Footprint edges by direction: per direction bin (0-180 deg), a distance raster of the edges
    within tol_deg of it, and the edges' directions for the refinement."""

    def __init__(self, A, B, lo, hi):
        self.A, self.B = A, B
        self.wdir = (B - A) / np.linalg.norm(B - A, axis=1)[:, None]
        n = ORIENTED["bins"]
        ang = np.mod(np.arctan2(self.wdir[:, 1], self.wdir[:, 0]), math.pi)
        self.maps = []
        for b in range(n):
            c = (b + 0.5) * math.pi / n
            diff = np.abs(np.mod(ang - c + math.pi / 2, math.pi) - math.pi / 2)
            keep = diff <= math.radians(ORIENTED["tol_deg"])
            self.maps.append(
                WallMap(A[keep], B[keep], [], lo, hi) if keep.any() else None
            )

    def bin_of(self, D):
        a = np.mod(np.arctan2(D[:, 1], D[:, 0]), math.pi)
        return np.minimum(
            (a / (math.pi / ORIENTED["bins"])).astype(int), ORIENTED["bins"] - 1
        )

    def distance(self, P, bins):
        """P (..., N, 2) placed points, bins (N,) their direction bins: distance to a wall of their direction."""
        out = np.full(P.shape[:-1], 1e3)
        for b in np.unique(bins):
            m = bins == b
            if self.maps[b] is not None:
                out[..., m] = self.maps[b].distance(P[..., m, :])
        return out

    def nearest(self, P, D):
        """Per point: (index of the nearest edge within tol_deg of direction D, its distance)."""
        d = segment_distances(P, self.A, self.B)
        ok = np.abs(D @ self.wdir.T) >= math.cos(math.radians(ORIENTED["tol_deg"]))
        d = np.where(ok, d, np.inf)
        k = np.argmin(d, axis=1)
        return k, d[np.arange(len(P)), k]


def oriented_setup(bpts, Lc, s_gps, D=None):
    """For --walls oriented: (line-point mask, their facade directions in the levelled plan frame,
    weights: across_weight for facades across the path), or None when too few points form lines.
    D: the points' directions when known (from the unthinned cloud), else found among bpts."""
    if D is None:
        D = facade_dirs(bpts, ORIENTED["radius_m"] / s_gps)
    line = ~np.isnan(D[:, 0])
    if line.sum() < ORIENTED["min_points"]:
        return None
    u = Lc[-1] - Lc[0]
    u = u / np.linalg.norm(u)
    across = np.abs(D[line] @ u) < math.cos(math.radians(45))
    return line, D[line], np.where(across, ORIENTED["across_weight"], 1.0)


def labels(model, mask_dir, value):
    """label_points, kept per model, masks and value: a rig's cameras are labelled again for the
    reference camera's pooled fit, and linked legs are placed twice."""
    key = (
        str(model),
        str(mask_dir),
        value,
        (Path(model) / "points3D.txt").stat().st_mtime_ns,
    )
    if key not in _LABELS:
        _LABELS[key] = label_points(model, mask_dir, value, OPT["class_fraction"])
    return _LABELS[key]


def level_with_gravity(names, C, R, gc):
    """(origin, basis rows [right, forward, up], gravity spread deg) from GeoCalib's per-frame
    gravity. R: world-from-camera rotations."""
    d = np.array(
        [R[i] @ -np.asarray(gc["frames"][n]["up_cam"]) for i, n in enumerate(names)]
    )
    keep = np.ones(len(d), bool)
    for _ in range(3):
        m = d[keep].mean(axis=0)
        m /= np.linalg.norm(m)
        ang = np.degrees(np.arccos(np.clip(d @ m, -1, 1)))
        keep = ang < max(3 * np.median(ang), 2.0)
    up = -m
    fwd = C[-1] - C[0]
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    return C[0], np.stack([right, fwd, up]), float(np.median(ang)), int((~keep).sum())


def heading_pitch_roll(R_enu):
    """Camera (world-from-camera rotation in east/north/up) -> heading (deg clockwise from north),
    pitch (deg, up positive), roll (deg)."""
    z, x = R_enu[:, 2], R_enu[:, 0]
    heading = math.degrees(math.atan2(z[0], z[1])) % 360
    pitch = math.degrees(math.asin(np.clip(z[2], -1, 1)))
    roll = math.degrees(math.asin(np.clip(x[2], -1, 1)))
    return heading, pitch, roll


def split_pieces(names, meta, dropped=(), min_cameras=8):
    """Camera index groups of a leg, cut only where frames were dropped as weakly anchored
    (reconstruct.py's dropped_frames: a gap the images do not hold together); a time gap alone
    (a frame Mapillary skipped) is not a cut. Pieces under min_cameras join their nearer neighbour."""
    t = np.array([meta[n]["captured_at_ms"] for n in names]) / 1000
    cuts = [
        i
        for i in range(1, len(names))
        if any(names[i - 1] < d < names[i] for d in dropped)
    ]
    pieces = [list(range(a, b)) for a, b in zip([0] + cuts, cuts + [len(names)])]
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for k, pc in enumerate(pieces):
            if len(pc) < min_cameras:
                if k == 0:
                    j = 1
                elif k == len(pieces) - 1:
                    j = k - 1
                else:  # the neighbour across the shorter time gap
                    j = (
                        k - 1
                        if t[pc[0]] - t[pieces[k - 1][-1]]
                        <= t[pieces[k + 1][0]] - t[pc[-1]]
                        else k + 1
                    )
                pieces[min(j, k)] = sorted(pieces[j] + pc)
                del pieces[max(j, k)]
                changed = True
                break
    return pieces


def point_pieces(model_txt, names, piece_of):
    """Per 3D point (points3D.txt order): the piece whose cameras observe it most."""
    raw = [
        row
        for row in (Path(model_txt) / "images.txt").read_text().splitlines()
        if not row.startswith("#")
    ]
    ids, i = {}, 0
    while i < len(raw):
        if raw[i].strip():
            q = raw[i].split()
            if Path(q[9]).name in piece_of:  # a folded leg's left-out cameras are not
                ids[int(q[0])] = piece_of[Path(q[9]).name]
            i += 2
        else:
            i += 1
    out = []
    for row in (Path(model_txt) / "points3D.txt").read_text().splitlines():
        if row and not row.startswith("#"):
            img = [int(v) for v in row.split()[8::2]]
            votes = np.bincount([ids[k] for k in img if k in ids], minlength=1)
            out.append(int(np.argmax(votes)))
    return np.array(out, int)


def fit_to_footprints(
    bpts,
    Lc,
    gps,
    A,
    B,
    wmap,
    both_sides,
    owalls=None,
    bdirs=None,
    info=None,
    dir_only=False,
):
    """One 2D similarity [angle, log scale, tx, ty] placing the building points bpts (levelled
    plan coordinates) closest to the footprint edges A->B: GPS-seeded global search, then a
    robust refinement with weak priors on the mean camera position and the GPS scale.
    Returns (params, params_gps_only)."""
    s_gps, ang0, t0 = similarity_2d(Lc, gps)
    params0 = np.array([ang0, math.log(s_gps), *t0])
    anchor = None
    if WALLS["mode"] == "hybrid" and not dir_only and owalls is not None:
        # HYBRID max_move_m: around the direction-matched fit
        prm_dir, _ = fit_to_footprints(
            bpts, Lc, gps, A, B, wmap, both_sides, owalls, bdirs, None, dir_only=True
        )
        anchor = apply_2d(prm_dir, Lc).mean(axis=0)
    a_fac = None
    if HEADING["mode"] == "facades":
        D_all = (
            bdirs
            if bdirs is not None
            else facade_dirs(bpts, ORIENTED["radius_m"] / s_gps)
        )
        a_fac, hinfo = facade_heading(D_all, apply_2d(params0, Lc), ang0, A, B)
        if info is not None:
            info["heading_from_facades"] = {**hinfo, "used": a_fac is not None}
    ori = oriented_setup(bpts, Lc, s_gps, bdirs) if owalls is not None else None
    free = None
    if (
        ori is not None
    ):  # line points only, matched by direction (see the module docstring)
        line, pdir, pw = ori
        if WALLS["mode"] == "hybrid" and not dir_only and (~line).any():
            free = bpts[~line]  # and the rest by plain distance (HYBRID)
        bpts = bpts[line]
    search_m = ORIENTED["search_m"] if ori is not None else OPT["search_m"]
    shift_prior = ORIENTED["shift_prior"] if ori is not None else OPT["shift_prior"]
    gps_sigma = ORIENTED["gps_sigma"] if ori is not None else OPT["gps_sigma"]
    scale_sigma = OPT["scale_sigma_both"] if both_sides else OPT["scale_sigma_one"]
    pick = (
        np.arange(len(bpts))
        if len(bpts) <= OPT["search_points"]
        else np.random.default_rng(0).choice(
            len(bpts), OPT["search_points"], replace=False
        )
    )
    search = bpts[pick]
    free_s = None
    if free is not None:
        free_s = (
            free
            if len(free) <= OPT["search_points"]
            else free[
                np.random.default_rng(1).choice(
                    len(free), OPT["search_points"], replace=False
                )
            ]
        )
    cam_mid, gps_mid = Lc.mean(axis=0), gps.mean(axis=0)
    offsets = np.arange(-search_m, search_m + 1e-9, 2.0)
    shift_cost = (
        0.5
        * (np.hypot(*np.meshgrid(offsets, offsets, indexing="ij")) / shift_prior) ** 2
    )
    best = (np.inf, None)
    dangs = (
        np.radians(
            np.arange(-FACADE_HDG["hold_deg"], FACADE_HDG["hold_deg"] + 1e-9, 0.25)
        )
        + (a_fac - params0[0])
        if a_fac is not None
        else np.radians(np.arange(-OPT["search_deg"], OPT["search_deg"] + 1e-9, 1.0))
    )
    for dang in dangs:
        ang = params0[0] + dang
        Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        for sc in [s_gps]:
            t = gps_mid - sc * Rm @ cam_mid
            prm = [ang, math.log(sc), *t]
            base, cams = apply_2d(prm, search), apply_2d(prm, Lc)
            if ori is not None:
                bins, wts = owalls.bin_of(rotate_dirs(pdir[pick], ang)), pw[pick]
            if free_s is not None:
                fbase = apply_2d(prm, free_s)
            for ix, dx in enumerate(offsets):
                shift = np.stack([np.full_like(offsets, dx), offsets], axis=1)[:, None]
                if ori is not None:
                    dd = np.minimum(
                        owalls.distance(base[None] + shift, bins), ORIENTED["cap_m"]
                    )
                    num, den = (dd * wts).sum(axis=1), wts.sum()
                    if free_s is not None:
                        num = num + HYBRID["weight"] * np.minimum(
                            wmap.distance(fbase[None] + shift), HYBRID["cap_m"]
                        ).sum(axis=1)
                        den = den + HYBRID["weight"] * len(free_s)
                    score = num / den
                else:
                    score = np.minimum(wmap.distance(base[None] + shift), 3.0).mean(
                        axis=1
                    )
                score += (
                    5.0 * wmap.inside(cams[None] + shift).mean(axis=1) + shift_cost[ix]
                )
                if anchor is not None:
                    moved = np.linalg.norm(
                        cams.mean(axis=0) + shift[:, 0] - anchor, axis=1
                    )
                    score = np.where(moved > HYBRID["max_move_m"], np.inf, score)
                if HEADING["prior_deg"]:
                    score += 0.5 * (math.degrees(dang) / HEADING["prior_deg"]) ** 2
                j = int(np.argmin(score))
                if score[j] < best[0]:
                    best = (
                        score[j],
                        np.array([ang, math.log(sc), t[0] + dx, t[1] + offsets[j]]),
                    )
    params = best[1]
    wdir = (B - A) / np.linalg.norm(B - A, axis=1)[:, None]
    wnrm = np.stack([-wdir[:, 1], wdir[:, 0]], axis=1)
    ds = FIXED
    lo_a, hi_a = (
        (
            a_fac - math.radians(FACADE_HDG["hold_deg"]),
            a_fac + math.radians(FACADE_HDG["hold_deg"]),
        )
        if a_fac is not None
        else (params[0] - math.radians(5), params[0] + math.radians(5))
    )
    lower = [lo_a, params[1] - ds, params[2] - 5, params[3] - 5]
    upper = [hi_a, params[1] + ds, params[2] + 5, params[3] + 5]
    for _ in range(10):
        if ori is not None:
            k, dk = owalls.nearest(apply_2d(params, bpts), rotate_dirs(pdir, params[0]))
            matched = dk < ORIENTED["cap_m"]
            extra = np.sqrt(pw[matched])
        else:
            d = segment_distances(apply_2d(params, bpts), A, B)
            k = np.argmin(d, axis=1)
            matched = d[np.arange(len(bpts)), k] < OPT["match_dist"]
            extra = 1.0
        S, Ad, Nd = bpts[matched], A[k[matched]], wnrm[k[matched]]
        sw = extra / np.sqrt(
            1.0
            + (
                np.abs(np.sum((apply_2d(params, S) - Ad) * Nd, axis=1))
                / OPT["wall_scale"]
            )
            ** 2
        )
        n = max(int(matched.sum()), 1)
        Sf = None
        if (
            free is not None
        ):  # the other points: their nearest wall, within HYBRID cap_m
            df = segment_distances(apply_2d(params, free), A, B)
            kf = np.argmin(df, axis=1)
            mf = df[np.arange(len(free)), kf] < HYBRID["cap_m"]
            Sf, Adf, Ndf = free[mf], A[kf[mf]], wnrm[kf[mf]]
            swf = math.sqrt(HYBRID["weight"]) / np.sqrt(
                1.0
                + (
                    np.abs(np.sum((apply_2d(params, Sf) - Adf) * Ndf, axis=1))
                    / OPT["wall_scale"]
                )
                ** 2
            )

        def residuals(prm):
            walls = (
                sw * np.sum((apply_2d(prm, S) - Ad) * Nd, axis=1) / OPT["wall_scale"]
            )
            if Sf is not None and len(Sf):
                walls = np.r_[
                    walls,
                    swf
                    * np.sum((apply_2d(prm, Sf) - Adf) * Ndf, axis=1)
                    / OPT["wall_scale"],
                ]
            g = math.sqrt(n) * (apply_2d(prm, Lc).mean(axis=0) - gps_mid) / gps_sigma
            sc = math.sqrt(n) * (prm[1] - params0[1]) / scale_sigma
            hd = (
                [
                    math.sqrt(n)
                    * math.degrees(prm[0] - params0[0])
                    / HEADING["prior_deg"]
                ]
                if HEADING["prior_deg"]
                else []
            )
            return np.r_[walls, g, sc, hd]

        params = least_squares(
            residuals,
            np.clip(params, lower, upper),
            bounds=(lower, upper),
            x_scale=[0.01, 0.01, 1, 1],
        ).x
    return params, params0


REL = {
    "near_m": 25.0,
    "explained_m": 6.0,
    "min_explained": 0.5,
    "min_points": 30,
    "scan_m": 40.0,
    "separation_m": 6.0,
    "best_within_m": 3.0,
    "margin": 0.10,
    "max_inside": 0.2,
}


def placement_checks(P, D, cams, A, B, wmap, owalls):
    """The values behind the reliability verdict (module docstring, step 6) for building points P
    as placed (plan, metres), their facade directions D in that frame (NaN: none) and the placed
    camera path."""
    from scipy.spatial import cKDTree

    d = segment_distances(P, A, B).min(axis=1)
    dpath = cKDTree(cams).query(P)[0]
    near, expl = dpath < REL["near_m"], d < REL["explained_m"]
    sel = near & expl
    out = {
        "near_points": int(sel.sum()),
        "near_explained_share": float(expl[near].mean()) if near.any() else 0.0,
        "near_median_m": float(np.median(d[sel])) if sel.any() else float("inf"),
        "far_median_m": float(np.median(d[~near])) if (~near).any() else None,
    }
    u = np.linalg.svd(cams - cams.mean(axis=0))[2][0]
    line = ~np.isnan(D[:, 0])
    ds = np.arange(-REL["scan_m"], REL["scan_m"] + 1e-9, 0.5)
    if line.sum() >= REL["min_points"]:
        across = np.abs(D[line] @ u) < math.cos(math.radians(45))
        w = np.where(across, ORIENTED["across_weight"], 1.0)
        Q = P[line][None] + ds[:, None, None] * u
        dd = np.minimum(owalls.distance(Q, owalls.bin_of(D[line])), ORIENTED["cap_m"])
        num, den = (dd * w).sum(axis=1), w.sum()
        out["along_cost"] = "direction-matched facade points"
        if WALLS["mode"] == "hybrid" and HYBRID["in_check"] and (~line).any():
            # scored as the fit scores them
            Qf = P[~line][None] + ds[:, None, None] * u
            num = num + HYBRID["weight"] * np.minimum(
                wmap.distance(Qf), HYBRID["cap_m"]
            ).sum(axis=1)
            den = den + HYBRID["weight"] * int((~line).sum())
            out["along_cost"] = (
                "direction-matched facade points + the others by nearest wall (hybrid)"
            )
        cost = num / den
    else:
        cost = np.minimum(wmap.distance(P[None] + ds[:, None, None] * u), 3.0).mean(
            axis=1
        )
        out["along_cost"] = "nearest wall"
    # ties (a uniform frontage scores the same everywhere) go to the shift nearest the placement
    i = int(np.argmin(cost + 1e-6 * np.abs(ds)))
    far = np.abs(ds - ds[i]) > REL["separation_m"]
    j = int(np.argmin(np.where(far, cost, np.inf)))
    out.update(
        along_best_shift_m=float(ds[i]),
        along_second_shift_m=float(ds[j]),
        along_margin=float((cost[j] - cost[i]) / max(cost[i], 0.05)),
    )
    return out


def verdict(chk, inside_share, backward, max_error):
    """Failed checks of placement_checks' values (empty: reliable)."""
    fails = []
    if chk["near_points"] < REL["min_points"]:
        fails.append(
            f"only {chk['near_points']} building points near the path with an OSM wall nearby"
        )
    if chk["near_explained_share"] < REL["min_explained"]:
        fails.append(
            f"{100 * (1 - chk['near_explained_share']):.0f}% of the nearby building points have no OSM wall"
        )
    if chk["near_median_m"] > max_error:
        fails.append(
            f"nearby building points {chk['near_median_m']:.2f} m from their walls"
        )
    if abs(chk["along_best_shift_m"]) > REL["best_within_m"]:
        fails.append(f"fits better {chk['along_best_shift_m']:+.1f} m along the street")
    elif chk["along_margin"] < REL["margin"]:
        fails.append(
            f"ambiguous along the street: {chk['along_second_shift_m']:+.1f} m fits almost as well "
            f"({100 * chk['along_margin']:.0f}% worse)"
        )
    if inside_share > REL["max_inside"]:
        fails.append(f"{100 * inside_share:.0f}% of the cameras inside buildings")
    if backward:
        fails.append(f"{len(backward)} cameras behind earlier ones")
    return fails


def rig_building_points(p, cam, origin, basis, s_gps):
    """A rig camera's usable building points (as align_camera's) in another camera's levelled
    plan frame (the same model world frame): (N, 2)."""
    model = p.reconstruction / cam["model"]
    _, C, _ = load_cameras(model)
    xyz, _, track = load_points(model)
    L, Lc = (xyz - origin) @ basis.T, (C - origin) @ basis.T
    masks = p.masks / cam["folder"]
    road, _ = labels(model, masks, MASK_ROAD)
    near = (
        np.min(np.linalg.norm(L[:, None, :2] - Lc[None, ::3, :2], axis=2), axis=1)
        * s_gps
        < 25
    )
    ground_z = (
        float(np.median(L[road & near, 2]))
        if (road & near).sum() >= 30
        else float(np.median(Lc[:, 2])) - OPT["default_camera_height"] / s_gps
    )
    building, _ = labels(model, masks, MASK_BUILDING)
    h_m = (L[:, 2] - ground_z) * s_gps
    sel = np.flatnonzero(
        building
        & (track >= OPT["min_track"])
        & (h_m > OPT["min_height"])
        & (h_m < OPT["max_height"])
    )
    _, first = np.unique(
        np.floor(L[sel, :2] / (OPT["point_grid"] / s_gps)).astype(np.int64),
        axis=0,
        return_index=True,
    )
    return L[sel[np.sort(first)], :2]


def gps_source(leg_id):
    """The GPS placement's source for a leg: --gps, or with --gps auto gps_check's choice."""
    if GPS["source"] == "auto":
        return GPS_LEG.get(leg_id, {}).get("source", "computed")
    return GPS["source"]


def gps_check(p, leg):
    """("computed" | "measured", reasons): measured when Mapillary's computed positions of the leg's
    frames look broken (see GPS_CHECK); judged from the positions alone, not the buildings."""
    reasons, tracks = [], {}
    for cam in leg["cameras"]:
        meta = p.frames(cam["folder"])
        fr = sorted(
            (
                meta[n]
                for n in cam.get("frame_files", [])
                if n in meta and meta[n].get("captured_at_ms")
            ),
            key=lambda m: m["captured_at_ms"],
        )
        if len(fr) < 3:
            continue
        enu = LocalFrame(fr[0]["lon"], fr[0]["lat"])
        C = np.array([enu.xy(m["lon"], m["lat"]) for m in fr])
        t = np.array([m["captured_at_ms"] for m in fr]) / 1000.0
        tracks[cam["folder"]] = (t, C)
        if all(m.get("gps") for m in fr):
            G = np.array([enu.xy(*m["gps"]) for m in fr])
            lc, lg = np.linalg.norm(C[-1] - C[0]), np.linalg.norm(G[-1] - G[0])
            if lc > 10 and abs(lg / lc - 1) > GPS_CHECK["length_tol"]:
                reasons.append(
                    f"{cam['folder']}: computed track {lc:.0f} m end to end, measured GPS {lg:.0f} m"
                )
    folders = list(tracks)
    for i, a in enumerate(folders):
        for b in folders[i + 1 :]:
            (ta, Ca), (tb, Cb) = tracks[a], tracks[b]
            t0, t1 = max(ta[0], tb[0]), min(ta[-1], tb[-1])
            if t1 - t0 < GPS_CHECK["min_overlap_s"]:
                continue
            ts = np.linspace(t0, t1, 20)
            la, lb = (
                np.linalg.norm(
                    np.diff(
                        np.c_[np.interp(ts, t, X[:, 0]), np.interp(ts, t, X[:, 1])],
                        axis=0,
                    ),
                    axis=1,
                ).sum()
                for t, X in ((ta, Ca), (tb, Cb))
            )
            if abs(la - lb) / max((la + lb) / 2, 1e-6) > GPS_CHECK["rig_tol"]:
                reasons.append(
                    f"rig cameras {a} and {b}: computed {la:.0f} vs {lb:.0f} m in the same {t1 - t0:.0f} s"
                )
    return ("measured" if reasons else "computed"), reasons


FOLD_KEEP_MIN = (
    8  # cameras the longest stretch between folds needs to be kept on its own
)


def backward_steps(Lc, gps=None):
    """Indices of cameras placed behind the one before: where the reconstruction folded. With gps
    (the cameras' GPS track, n x 2 m), a step back counts only where the track moves on
    (geo_common.steps_against_track): a capture that turns back on itself, a walker's U-turn on a
    campus path, goes back on the track too and is no fold. Without it, a step back along the leg
    (levelled y) of more than a third of a normal step."""
    if gps is not None and len(Lc) >= 3:
        return np.array(steps_against_track(plan_onto(Lc[:, :2], gps), gps), int)
    steps = np.diff(Lc[:, 1])
    return np.flatnonzero(steps < -np.median(np.abs(steps)) / 3) + 1


# A jump: the reconstruction put a camera far from the one before while the capture moved on a little
# (a step more than ratio x the GPS step between the same frames, and more than min_m, at the scale the
# steps suggest). Alignment_Pipeline's roi_30 line 3 (iPhone ultra-wide, campus) leg02: its first three
# frames, right after a turn, sat 12 m to the side of the rest (a 12.2 m step in 0.7 s, GPS 1.5 m) and
# turned the whole leg 10 deg; the fold check (steps back along the track) let the sideways jump through.
# Cut there, the leg fitted at +0.3 deg and linked with its neighbour.
JUMP = {"ratio": 4.0, "min_m": 3.0}
# re-placing the cameras a fold or jump left out (relocalize): RANSAC PnP inliers needed, how far the
# result may be from where the GPS step from the nearest kept camera puts it, and the reprojection px
RELOC = {"min_inliers": 20, "max_off_m": 4.0, "px": 4.0}


def jump_steps(Lc, gps):
    """Indices of the cameras (1 .. n-1) placed a jump away from the one before (JUMP)."""
    if gps is None or len(Lc) < 4:
        return np.zeros(0, int)
    m = np.linalg.norm(np.diff(Lc[:, :2], axis=0), axis=1)
    g = np.linalg.norm(np.diff(np.asarray(gps, float), axis=0), axis=1)
    ok = (m > 1e-9) & (g > 0.2)
    if ok.sum() < 3:
        return np.zeros(0, int)
    sc = float(np.median(g[ok] / m[ok]))
    return np.flatnonzero((m * sc > JUMP["min_m"]) & (m * sc > JUMP["ratio"] * g)) + 1


def fold_words(fold):
    """'folded at 0023, jumped at 0054' for a fold_trim record."""
    w = (
        [f"folded at {', '.join(n[6:10] for n in fold['folds_at'])}"]
        if fold.get("folds_at")
        else []
    ) + (
        [f"jumped at {', '.join(n[6:10] for n in fold['jumps_at'])}"]
        if fold.get("jumps_at")
        else []
    )
    return ", ".join(w) or "cut"


def fold_keep(Lc, gps=None):
    """(first, end) of the longest stretch of cameras between folds and jumps, or None (none, or
    no stretch of FOLD_KEEP_MIN cameras)."""
    bad = np.union1d(backward_steps(Lc, gps), jump_steps(Lc, gps)).astype(int)
    if not len(bad):
        return None
    cuts = [0, *bad.tolist(), len(Lc)]
    a, b = max(zip(cuts[:-1], cuts[1:]), key=lambda ab: ab[1] - ab[0])
    return (a, b) if b - a >= FOLD_KEEP_MIN else None


def relocalize(model_txt, frames, kept):
    """World-from-camera poses {frame: (centre, R, inliers)} for frames the fit left out, solved by
    PnP (RANSAC) from each frame's observations of 3D points that two or more kept cameras also
    see, so the pose comes from the well-placed part of the model. Frames with fewer than RELOC
    min_inliers inliers are not returned."""
    import cv2

    model_txt = Path(model_txt)
    cams = {}
    for line in (model_txt / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            q = line.split()
            cams[int(q[0])] = (q[1], np.array(q[4:], float))
    lines = [
        line
        for line in (model_txt / "images.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    imgs, id_of = {}, {}
    for header, obs in zip(lines[0::2], lines[1::2]):
        q = header.split()
        name = Path(q[9]).name
        id_of[name] = int(q[0])
        imgs[name] = (
            int(q[8]),
            np.array(obs.split(), float).reshape(-1, 3)
            if obs.strip()
            else np.zeros((0, 3)),
        )
    kept_ids = {id_of[n] for n in kept if n in id_of}
    pts = {}
    for line in (model_txt / "points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            q = line.split()
            if sum(int(i) in kept_ids for i in q[8::2]) >= 2:
                pts[int(q[0])] = np.array(q[1:4], float)
    out = {}
    for n in frames:
        if n not in imgs:
            continue
        cid, obs = imgs[n]
        obs = obs[[int(v) in pts for v in obs[:, 2]]] if len(obs) else obs
        if len(obs) < RELOC["min_inliers"]:
            continue
        model, prm = cams[cid]
        if model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE"):
            f, cx, cy = prm[0], prm[1], prm[2]
        else:
            f, cx, cy = prm[0], prm[2], prm[3]
        k1 = prm[3] if model in ("SIMPLE_RADIAL", "RADIAL") else 0.0
        k2 = prm[4] if model == "RADIAL" else 0.0
        xd = (obs[:, :2] - [cx, cy]) / f
        xu = xd.copy()
        for _ in range(10):  # undo the radial distortion
            r2 = (xu**2).sum(axis=1, keepdims=True)
            xu = xd / (1 + k1 * r2 + k2 * r2**2)
        X = np.array([pts[int(v)] for v in obs[:, 2]])
        ok, rv, tv, inl = cv2.solvePnPRansac(
            X,
            xu,
            np.eye(3),
            None,
            reprojectionError=RELOC["px"] / f,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inl is None or len(inl) < RELOC["min_inliers"]:
            continue
        i = inl[:, 0]
        ok, rv, tv = cv2.solvePnP(
            X[i], xu[i], np.eye(3), None, rv, tv, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        Rcw = cv2.Rodrigues(rv)[0]
        out[n] = (-Rcw.T @ tv[:, 0], Rcw.T, int(len(i)))
    return out


def observed_share(model_txt, names):
    """Per 3D point (points3D.txt order): the share of its observations made by the cameras in names."""
    raw = [
        line
        for line in (Path(model_txt) / "images.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    ids, i = set(), 0
    while i < len(raw):
        if raw[i].strip():
            q = raw[i].split()
            if Path(q[9]).name in names:
                ids.add(int(q[0]))
            i += 2
        else:
            i += 1
    out = []
    for line in (Path(model_txt) / "points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            img = [int(v) for v in line.split()[8::2]]
            out.append(sum(k in ids for k in img) / max(len(img), 1))
    return np.array(out)


def align_camera(
    p, leg, cam, buildings, out_dir, max_error, shared=None, pool=(), enu=None
):
    """shared: a rig's placement from its reference camera ({origin, basis, enu, fits, fits0}; this
    camera's model is in the same world frame), used instead of levelling and fitting this camera
    on its own. pool: the rig's other cameras, whose building points join this camera's in the fit.
    enu: the plan frame (default: this camera's first frame)."""
    folder = cam["folder"]
    meta = p.frames(folder)
    gc = json.loads(p.calibration_json(folder).read_text())
    model = p.reconstruction / cam["model"]
    names, C, R = load_cameras(model)
    origin, basis, g_spread, g_out = level_with_gravity(names, C, R, gc)
    if shared:
        origin, basis = shared["origin"], shared["basis"]
    xyz, rgb, track = load_points(model)
    L = (xyz - origin) @ basis.T
    Lc = (C - origin) @ basis.T
    # A folded reconstruction (cameras behind earlier ones, the capture-order check below) used to
    # fail the whole leg. Now the longest stretch between folds is kept when it holds FOLD_KEEP_MIN
    # cameras: it is fitted and judged on its own with the points its cameras mostly see, and the
    # other cameras are left out (no aligned pose: downstream they keep Mapillary's). A left turn at
    # a leg's end (the 8 frames it shares with the next leg) folded in every ALIKED run of one ROI;
    # the 21 frames of straight street before it fit the walls (0.68 m) but the fold cost all 30
    # GPS (Mapillary's positions) as the starting placement and the leg's metric scale
    enu = (
        shared["enu"]
        if shared
        else enu or LocalFrame(meta[names[0]]["lon"], meta[names[0]]["lat"])
    )
    if (
        gps_source(leg["id"]) == "measured"
    ):  # the device's GPS; Mapillary's computed position where there is none
        gps = np.array(
            [
                enu.xy(*(meta[n].get("gps") or (meta[n]["lon"], meta[n]["lat"])))
                for n in names
            ]
        )
    else:
        gps = np.array([enu.xy(meta[n]["lon"], meta[n]["lat"]) for n in names])

    fold = None
    keep = fold_keep(Lc, gps)
    keep_pt = None
    names_all, gps_all = list(names), gps.copy()
    if keep is not None and (keep[0] > 0 or keep[1] < len(names)):
        a, b = keep
        fold = {
            "kept_cameras": b - a,
            "kept": [names[a], names[b - 1]],
            "left_out": names[:a] + names[b:],
            "folds_at": [names[i] for i in backward_steps(Lc, gps)],
            "jumps_at": [names[i] for i in jump_steps(Lc, gps)],
        }
        keep_pt = observed_share(model, set(names[a:b])) >= 0.5
        names, C, R, Lc, gps = names[a:b], C[a:b], R[a:b], Lc[a:b], gps[a:b]

    s_gps = similarity_2d(Lc[:, :2], gps)[0]

    # ground plane from the road points near the path
    masks = p.masks / folder
    road, _ = labels(model, masks, MASK_ROAD)
    if keep_pt is not None:
        road = road & keep_pt
    near = (
        np.min(np.linalg.norm(L[:, None, :2] - Lc[None, ::3, :2], axis=2), axis=1)
        * s_gps
        < 25
    )
    cam_z = float(np.median(Lc[:, 2]))
    if (road & near).sum() >= 30:
        ground_z = float(np.median(L[road & near, 2]))
        ground_src = f"{int((road & near).sum())} road points"
    else:
        ground_z = cam_z - OPT["default_camera_height"] / s_gps
        ground_src = f"default camera height {OPT['default_camera_height']} m"
    cam_height = (cam_z - ground_z) * s_gps

    building, _ = labels(model, masks, MASK_BUILDING)
    h_m = (L[:, 2] - ground_z) * s_gps
    usable = (
        building
        & (track >= OPT["min_track"])
        & (h_m > OPT["min_height"])
        & (h_m < OPT["max_height"])
    )
    if keep_pt is not None:
        usable &= keep_pt

    # pieces: the leg cut where weak frames were dropped; each gets its own fit
    pieces = split_pieces(names, meta, cam.get("dropped_frames", {}))
    piece_of = {names[i]: k for k, pc in enumerate(pieces) for i in pc}
    cam_piece = np.array([piece_of[n] for n in names])
    pt_piece = (
        point_pieces(model, names, piece_of)
        if len(pieces) > 1
        else np.zeros(len(xyz), int)
    )

    lo = gps.min(axis=0) - OPT["margin"] - OPT["search_m"]
    hi = gps.max(axis=0) + OPT["margin"] + OPT["search_m"]
    footprints = []
    for tags, ring in buildings:
        xy = np.array([enu.xy(lon, lat) for lon, lat in ring])
        if np.all(xy.max(axis=0) >= lo) and np.all(xy.min(axis=0) <= hi):
            base, top, _ = building_heights(tags, 3.5, 3)
            footprints.append((xy, base, top))
    if not footprints:
        return {"aligned": False, "reason": "no OSM buildings near the track"}
    A = np.vstack([xy[:-1] for xy, _, _ in footprints])
    B = np.vstack([xy[1:] for xy, _, _ in footprints])
    ok = np.linalg.norm(B - A, axis=1) >= 0.5
    A, B = A[ok], B[ok]
    wmap = WallMap(A, B, footprints, lo, hi)
    owalls = (
        OrientedWalls(A, B, lo, hi) if WALLS["mode"] in ("oriented", "hybrid") else None
    )

    # building points thinned to one per point_grid cell, per piece
    sel = np.flatnonzero(usable)
    _, first = np.unique(
        np.c_[
            np.floor(L[sel, :2] / (OPT["point_grid"] / s_gps)).astype(np.int64),
            pt_piece[sel],
        ],
        axis=0,
        return_index=True,
    )
    sel = sel[np.sort(first)]
    if len(sel) < 30:
        return {"aligned": False, "reason": f"only {len(sel)} usable building points"}
    bpts, b_piece = L[sel, :2], pt_piece[sel]
    pooled = (
        np.vstack([rig_building_points(p, c, origin, basis, s_gps) for c in pool])
        if pool
        else np.zeros((0, 2))
    )
    bdirs = pdirs = None
    if WALLS["mode"] in ("oriented", "hybrid"):
        # facade directions from the unthinned building points (thinning leaves too few to trace lines)
        idx = np.flatnonzero(usable)
        D_all = facade_dirs(L[idx, :2], ORIENTED["radius_m"] / s_gps)
        bdirs = D_all[np.searchsorted(idx, sel)]
        pdirs = (
            facade_dirs(pooled, ORIENTED["radius_m"] / s_gps)
            if len(pooled)
            else np.zeros((0, 2))
        )
    right_share = float(np.mean(side_of_path(np.vstack([bpts, pooled]), [Lc[:, :2]])))
    both_sides = min(right_share, 1 - right_share) >= 0.2

    fits, fits0, piece_stats = [], [], []
    # how each piece's heading was set; a placement taken from elsewhere (rig reference, linked legs) keeps its record
    hinfos = dict(shared.get("hinfos", {})) if shared else {}
    p.results.mkdir(parents=True, exist_ok=True)
    for k, pc in enumerate(pieces):
        if shared:  # a rig camera: the reference camera's placement
            fits.append(np.asarray(shared["fits"][min(k, len(shared["fits"]) - 1)]))
            fits0.append(np.asarray(shared["fits0"][min(k, len(shared["fits0"]) - 1)]))
            continue
        mine = b_piece == k
        if (
            mine.sum() < 30
        ):  # too few points of its own: fit with the whole leg's points
            mine = np.ones(len(bpts), bool)
        hinfos[k] = {}
        prm, prm0 = fit_to_footprints(
            np.vstack([bpts[mine], pooled]),
            Lc[pc, :2],
            gps[pc],
            A,
            B,
            wmap,
            both_sides,
            owalls,
            np.vstack([bdirs[mine], pdirs]) if bdirs is not None else None,
            hinfos[k],
        )
        fits.append(prm)
        fits0.append(prm0)

    def place(params_list, XY, idx):
        out = np.empty_like(XY)
        for k, prm in enumerate(params_list):
            m = idx == k
            out[m] = apply_2d(prm, XY[m])
        return out

    def ptf(params_list, mask=None):
        P = place(params_list, bpts, b_piece)
        dd = np.minimum(
            segment_distances(P if mask is None else P[mask], A, B).min(axis=1), 50.0
        )
        return {
            "points": int(len(dd)),
            "median_m": float(np.median(dd)),
            "within_1m": float(np.mean(dd < 1.0)),
            "within_2m": float(np.mean(dd < 2.0)),
        }

    before, after = ptf(fits0), ptf(fits)
    cams_xy = place(fits, Lc[:, :2], cam_piece)
    inside = wmap.inside(cams_xy)
    # Cameras must follow capture order along the leg: a camera placed behind an earlier one (a step
    # backwards of more than a third of a normal step) means the reconstruction folded, e.g. a
    # stretch slid back over the previous one, however well its points fit the walls
    backward = [names[i] for i in backward_steps(Lc, gps)]
    reliable_old = (
        after["median_m"] <= max_error and inside.mean() <= 0.2 and not backward
    )
    # reliability (module docstring, step 6): this camera's own checks; a rig shares its reference's verdict
    D_b = (
        bdirs if bdirs is not None else facade_dirs(bpts, ORIENTED["radius_m"] / s_gps)
    )
    P_chk, D_chk = place(fits, bpts, b_piece), rotate_dirs(D_b, 0.0)
    for k in range(len(fits)):
        m = b_piece == k
        D_chk[m] = rotate_dirs(D_b[m], fits[k][0])
    if len(pooled):
        D_p = (
            pdirs
            if pdirs is not None
            else facade_dirs(pooled, ORIENTED["radius_m"] / s_gps)
        )
        P_chk = np.vstack([P_chk, apply_2d(fits[0], pooled)])
        D_chk = np.vstack([D_chk, rotate_dirs(D_p, fits[0][0])])
    checks = placement_checks(
        P_chk,
        D_chk,
        cams_xy,
        A,
        B,
        wmap,
        owalls if owalls is not None else OrientedWalls(A, B, lo, hi),
    )
    fails = verdict(checks, float(inside.mean()), backward, max_error)
    if (
        shared
        and shared.get("reliability") is not None
        and cam.get("rig") not in (None, folder)
    ):
        own = fails
        fails = list(shared["reliability"]["failed"])
        checks = {**checks, "own_failed": own, "verdict_from_rig_reference": cam["rig"]}
    reliable = not fails
    t_s = np.array([meta[n]["captured_at_ms"] for n in names]) / 1000
    for k, (pc, prm, prm0) in enumerate(zip(pieces, fits, fits0)):
        st = {
            "cameras": len(pc),
            "first": names[pc[0]],
            "last": names[pc[-1]],
            "building_points": int((b_piece == k).sum()),
            "points_to_footprint": ptf(fits, b_piece == k),
            "scale_vs_mapillary": math.exp(prm[1] - prm0[1]),
            "heading_change_deg": math.degrees(prm[0] - prm0[0]),
        }
        if hinfos.get(k):
            st.update(hinfos[k])
        if WALLS["mode"] in ("oriented", "hybrid"):
            st["oriented_line_points"] = int(
                (~np.isnan(bdirs[(b_piece == k)][:, 0])).sum()
            )
        if k:
            # the cut: aligned distance across it against the speed on either side
            i0, i1 = pieces[k - 1][-1], pc[0]
            v = lambda a, b: np.linalg.norm(cams_xy[b] - cams_xy[a]) / (t_s[b] - t_s[a])  # noqa: E731
            v_before = (
                v(max(pieces[k - 1][0], i0 - 3), i0)
                if i0 > pieces[k - 1][0]
                else float("nan")
            )
            v_after = v(i1, min(pc[-1], i1 + 3)) if i1 < pc[-1] else float("nan")
            # GPS only as a yardstick here; the model's own speed next to a weak gap can be off too
            st["cut"] = {
                "seconds": float(t_s[i1] - t_s[i0]),
                "aligned_distance_m": float(np.linalg.norm(cams_xy[i1] - cams_xy[i0])),
                "gps_distance_m": float(np.linalg.norm(gps[i1] - gps[i0])),
                "model_speed_distance_m": float(
                    np.nanmean([v_before, v_after]) * (t_s[i1] - t_s[i0])
                ),
            }
        piece_stats.append(st)

    s = math.exp(fits[0][1])
    cam_model, cam_w, cam_h, cam_params = read_camera(model)
    # what texturing needs: the COLMAP camera (its distortion is not undone) at the frame's size
    intrinsics = {
        "model": cam_model,
        "width": cam_w,
        "height": cam_h,
        "fx": cam_params[0],
        "params": cam_params,
    }
    cams_out = []
    for i, nme in enumerate(names):
        prm = fits[cam_piece[i]]
        a = prm[0]
        Rz = np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )
        R_enu = Rz @ basis @ R[i]
        hdg, pit, rol = heading_pitch_roll(R_enu)
        lon, lat = enu.lonlat(*cams_xy[i])
        cams_out.append(
            {
                "file": nme,
                "id": meta[nme].get("id"),
                "leg": leg["id"],
                "lon": lon,
                "lat": lat,
                "height_m": float((Lc[i, 2] - ground_z) * math.exp(prm[1])),
                "heading_deg": hdg,
                "pitch_deg": pit,
                "roll_deg": rol,
                # columns: the camera's x (right), y (down), z (forward) in east/north/up
                "rotation_cam_to_enu": np.round(R_enu, 8).tolist(),
                "intrinsics": intrinsics,
                "piece": int(cam_piece[i]),
                "pano_yaw_deg": meta[nme].get("pano_yaw_deg"),
                "gps": meta[nme].get("gps"),
                "mapillary_lonlat": [meta[nme]["lon"], meta[nme]["lat"]],
                "offset_to_mapillary_m": float(np.linalg.norm(cams_xy[i] - gps[i])),
                "leg_fit_m": after["median_m"],
                "reliable": bool(reliable),
            }
        )
    if fold and not shared:
        # the left-out frames re-placed from the kept part of the model (PnP), where that agrees
        # with the GPS step from the nearest kept camera; otherwise no pose from this leg (a frame
        # it shares with a neighbouring leg keeps that leg's, the rest Mapillary's downstream)
        at = {n: i for i, n in enumerate(names_all)}
        reloc = relocalize(model, fold["left_out"], names)
        fold["relocalized"], fold["not_relocalized"] = [], []
        for nme in fold["left_out"]:
            j = min(range(len(names)), key=lambda k: abs(at[names[k]] - at[nme]))
            if nme not in reloc:
                fold["not_relocalized"].append(nme)
                continue
            Cn, Rn, n_in = reloc[nme]
            Ln = (Cn - origin) @ basis.T
            prm = fits[cam_piece[j]]
            xy = apply_2d(prm, Ln[None, :2])[0]
            off = float(np.linalg.norm(xy - (cams_xy[j] + gps_all[at[nme]] - gps[j])))
            if off > RELOC["max_off_m"]:
                fold["not_relocalized"].append(nme)
                continue
            a = prm[0]
            Rz = np.array(
                [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
            )
            R_enu = Rz @ basis @ Rn
            hdg, pit, rol = heading_pitch_roll(R_enu)
            lon, lat = enu.lonlat(*xy)
            cams_out.append(
                {
                    "file": nme,
                    "id": meta[nme].get("id"),
                    "leg": leg["id"],
                    "lon": lon,
                    "lat": lat,
                    "height_m": float((Ln[2] - ground_z) * math.exp(prm[1])),
                    "heading_deg": hdg,
                    "pitch_deg": pit,
                    "roll_deg": rol,
                    "rotation_cam_to_enu": np.round(R_enu, 8).tolist(),
                    "intrinsics": intrinsics,
                    "piece": int(cam_piece[j]),
                    "pano_yaw_deg": meta[nme].get("pano_yaw_deg"),
                    "gps": meta[nme].get("gps"),
                    "mapillary_lonlat": [meta[nme]["lon"], meta[nme]["lat"]],
                    "offset_to_mapillary_m": float(np.linalg.norm(xy - gps_all[at[nme]])),
                    "leg_fit_m": after["median_m"],
                    "reliable": bool(reliable),
                    "relocalized": {"inliers": n_in, "off_gps_step_m": round(off, 2)},
                }
            )
            fold["relocalized"].append(nme)

    n_w = np.array([len(pc) for pc in pieces], float)
    stats = {
        "leg": leg["id"],
        "folder": folder,
        "registered": len(names),
        "frames": cam["frames"],
        "gravity": {"spread_median_deg": g_spread, "outlier_cameras": g_out},
        "ground": ground_src,
        "camera_height_m": cam_height,
        "building_points": int(len(bpts)),
        "facades_both_sides": both_sides,
        "pieces": piece_stats,
        "points_to_footprint_median_m": {
            "mapillary_only": before["median_m"],
            "aligned": after["median_m"],
        },
        "points_to_footprint_within_1m": after["within_1m"],
        "points_to_footprint": {"before": before, "after": after},
        "scale_m_per_unit": s,
        "scale_vs_mapillary": float(
            np.average([st["scale_vs_mapillary"] for st in piece_stats], weights=n_w)
        ),
        "heading_change_deg": float(
            np.average([st["heading_change_deg"] for st in piece_stats], weights=n_w)
        ),
        "cameras_inside_buildings": int(inside.sum()),
        "cameras_out_of_order": backward,
        "fold_trim": fold,
        "gps_source": gps_source(leg["id"]),
        "gps_check": GPS_LEG.get(leg["id"], {}).get("reasons"),
        "rig_fit": (
            f"placed with the rig reference {cam['rig']}"
            if shared and cam.get("rig") not in (None, folder)
            else f"fitted with the building points of {len(pool)} other rig cameras ({len(pooled)} points)"
            if pool
            else None
        ),
        "aligned": True,
        "reliable": bool(reliable),
        "reliable_old": bool(reliable_old),
        "reliability": {
            **checks,
            "failed": fails,
            "notes": [
                f"heading unconstrained ({h['heading_from_facades']['unconstrained']})"
                for h in hinfos.values()
                if h.get("heading_from_facades", {}).get("unconstrained")
            ]
            + (
                [
                    f"reconstruction {fold_words(fold)}: "
                    f"{len(fold['left_out'])} cameras left out, {fold['kept_cameras']} kept "
                    f"({fold['kept'][0][6:10]}-{fold['kept'][1][6:10]})"
                    + (
                        f", {len(fold['relocalized'])} re-placed by PnP"
                        if fold.get("relocalized")
                        else ""
                    )
                ]
                if fold
                else []
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    (out_dir / "transform.json").write_text(
        json.dumps(
            {
                "description": "levelled L = basis @ (X_colmap - origin) (x right, y forward, z up); per piece, ENU metres "
                "from enu_origin: xy = s R(angle) L[:2] + t, height above ground = s (L[2] - ground_z). "
                "A 3D point belongs to the piece whose cameras observe it most (align.point_pieces)",
                "model": str(model),
                "origin": origin.tolist(),
                "basis": basis.tolist(),
                "ground_z": ground_z,
                "angle_rad": float(fits[0][0]),
                "scale_m_per_unit": s,
                "translation_m": fits[0][2:].tolist(),
                "pieces": [
                    {
                        "images": [names[i] for i in pc],
                        "angle_rad": float(prm[0]),
                        "scale_m_per_unit": math.exp(prm[1]),
                        "translation_m": prm[2:].tolist(),
                        "params_gps_only": prm0.tolist(),
                    }
                    for pc, prm, prm0 in zip(pieces, fits, fits0)
                ],
                "enu_origin_lon_lat": [enu.lon0, enu.lat0],
            },
            indent=1,
        )
    )

    # plan view: GPS-only placement vs fitted
    shown = (track >= OPT["min_track"]) & (h_m > 0.3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, plist, title in (
        (
            axes[0],
            fits0,
            f"GPS placement: building points to footprints {before['median_m']:.2f} m",
        ),
        (
            axes[1],
            fits,
            f"Fitted to OSM{f' in {len(pieces)} pieces' if len(pieces) > 1 else ''}: "
            f"{after['median_m']:.2f} m ({100 * after['within_1m']:.0f}% within 1 m)",
        ),
    ):
        for xy, _, _ in footprints:
            ax.fill(*xy.T, color="#e3e3e3", ec="#888", lw=0.6, zorder=0)
        ax.scatter(
            *place(plist, L[shown, :2], pt_piece[shown]).T,
            s=0.4,
            c=rgb[shown],
            zorder=1,
        )
        ax.scatter(
            *place(plist, bpts, b_piece).T,
            s=1.2,
            c="#d62728",
            zorder=2,
            label="building points",
        )
        cxy = place(plist, Lc[:, :2], cam_piece)
        for k in range(len(pieces)):
            ax.plot(
                *cxy[cam_piece == k].T,
                ".-",
                ms=3,
                lw=1,
                zorder=3,
                label=f"cameras, piece {k + 1}" if len(pieces) > 1 else "cameras",
            )
        ax.plot(*gps.T, "x", color="#555", ms=3, zorder=3, label="Mapillary positions")
        allp = np.vstack([cxy, gps])
        ax.set_xlim(allp[:, 0].min() - 40, allp[:, 0].max() + 40)
        ax.set_ylim(allp[:, 1].min() - 40, allp[:, 1].max() + 40)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"{p.name} {leg['id']}/{folder}: {len(names)} cameras, GeoCalib gravity levelling "
        f"(spread {g_spread:.1f} deg), camera height {cam_height:.2f} m, "
        f"{'reliable' if reliable else 'UNRELIABLE'}",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "plan.png", dpi=110)
    plt.close(fig)
    fitdata = None
    if (
        not shared and len(pieces) == 1
    ):  # what link_legs needs to refit this camera with its neighbours
        allb = np.vstack([bpts, pooled])
        ori = (
            oriented_setup(allb, Lc[:, :2], s_gps, np.vstack([bdirs, pdirs]))
            if owalls is not None
            else None
        )
        fitdata = {
            "bpts": allb,
            "Lc": Lc[:, :2],
            "gps": gps,
            "A": A,
            "B": B,
            "names": names,
            "owalls": owalls,
            "oriented": ori,
            "heading_held": bool(
                hinfos.get(0, {}).get("heading_from_facades", {}).get("used")
            ),
            "params0": fits0[0],
            "scale_sigma": OPT["scale_sigma_both"]
            if both_sides
            else OPT["scale_sigma_one"],
        }
    return {
        **stats,
        "cameras_out": cams_out,
        "_shared": {
            "origin": origin,
            "basis": basis,
            "enu": enu,
            "fits": fits,
            "fits0": fits0,
            "hinfos": hinfos,
            "reliability": {"failed": fails},
        },
        "_fitdata": fitdata,
    }


LINK_SIGMA = 0.3  # m: how closely two linked legs must place the cameras they share
LINK_SHAPE_M = 1.0  # two legs' shared cameras further apart than this after the best similarity: not linked
LINK_WORSE_M, LINK_WORSE_X = (
    0.5,
    1.5,
)  # a link making a leg's wall fit worse by more than both of these is undone


def wall_terms(params, bpts, A, B, owalls=None, ori=None):
    """The wall residual set-up of fit_to_footprints' refinement at params: (points, wall start,
    wall normal, robust weights) of the building points within match_dist of a wall (with
    --walls oriented: the line points, matched by direction within cap_m, weighted)."""
    wdir = (B - A) / np.linalg.norm(B - A, axis=1)[:, None]
    wnrm = np.stack([-wdir[:, 1], wdir[:, 0]], axis=1)
    if owalls is not None and ori is not None:
        line, pdir, pw = ori
        bpts = bpts[line]
        k, dk = owalls.nearest(apply_2d(params, bpts), rotate_dirs(pdir, params[0]))
        m, extra = dk < ORIENTED["cap_m"], np.sqrt(pw[dk < ORIENTED["cap_m"]])
    else:
        d = segment_distances(apply_2d(params, bpts), A, B)
        k = np.argmin(d, axis=1)
        m, extra = d[np.arange(len(bpts)), k] < OPT["match_dist"], 1.0
    S, Ad, Nd = bpts[m], A[k[m]], wnrm[k[m]]
    sw = extra / np.sqrt(
        1.0
        + (np.abs(np.sum((apply_2d(params, S) - Ad) * Nd, axis=1)) / OPT["wall_scale"])
        ** 2
    )
    return S, Ad, Nd, sw


def link_legs(legs, results):
    """Refine the fits of linked legs together (see the module docstring): every camera fitted on
    its own in two consecutive legs that share frames (reconstruct.py's leg overlap) gets one
    term holding the shared cameras together, LINK_SIGMA, weighted to count as much as the
    smaller leg's wall points. Returns ({leg id: {folder: params}} to place again, report)."""
    ents, report = {}, []
    for leg in legs:
        for folder, (entry, _, sh, fd) in results[leg["id"]].items():
            if fd is not None and entry.get("aligned"):
                ents[(leg["id"], folder)] = {
                    "fd": fd,
                    "x0": np.asarray(sh["fits"][0], float),
                }
    links = []
    for a, b in zip(legs, legs[1:]):
        if not a.get("link_next") or a["line"] != b["line"]:
            continue
        for folder in sorted(
            {k[1] for k in ents if k[0] == a["id"]}
            & {k[1] for k in ents if k[0] == b["id"]}
        ):
            fa, fb = ents[(a["id"], folder)]["fd"], ents[(b["id"], folder)]["fd"]
            common = sorted(set(fa["names"]) & set(fb["names"]))
            if len(common) >= 3:
                ia, ib = (
                    [fa["names"].index(n) for n in common],
                    [fb["names"].index(n) for n in common],
                )
                # the shared stretch must have the same shape in both legs (else one of them broke there)
                s_, ang, t = similarity_2d(fb["Lc"][ib], fa["Lc"][ia])
                Rm = np.array(
                    [[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]
                )
                m_per_unit = math.exp(ents[(a["id"], folder)]["x0"][1])
                shape = (
                    float(
                        np.median(
                            np.linalg.norm(
                                s_ * fb["Lc"][ib] @ Rm.T + t - fa["Lc"][ia], axis=1
                            )
                        )
                    )
                    * m_per_unit
                )
                if shape > LINK_SHAPE_M:
                    print(
                        f"link {a['id']}-{b['id']} {folder}: not linked, the shared cameras differ in shape by {shape:.1f} m"
                    )
                    report.append(
                        {
                            "legs": [a["id"], b["id"]],
                            "folder": folder,
                            "linked": False,
                            "reason": f"shared cameras differ in shape by {shape:.1f} m",
                        }
                    )
                    continue
                links.append(((a["id"], folder), (b["id"], folder), ia, ib, common))
    if not links:
        return {}, report
    while links:
        x0, x, keys, sl = solve_links(ents, links)
        worse = [
            k
            for k in keys
            if walls_median(ents, x, sl, k)
            > max(
                walls_median(ents, x0, sl, k) + LINK_WORSE_M,
                LINK_WORSE_X * walls_median(ents, x0, sl, k),
            )
        ]
        if not worse:
            break
        drop = [lk for lk in links if lk[0] in worse or lk[1] in worse]
        for lk in drop:
            print(
                f"link {lk[0][0]}-{lk[1][0]} {lk[0][1]}: undone, it made the wall fit of "
                + ", ".join(
                    f"{k[0]} {walls_median(ents, x0, sl, k):.2f} -> {walls_median(ents, x, sl, k):.2f} m"
                    for k in worse
                    if k in lk[:2]
                )
            )
            report.append(
                {
                    "legs": [lk[0][0], lk[1][0]],
                    "folder": lk[0][1],
                    "linked": False,
                    "reason": "made a leg's wall fit worse",
                }
            )
        links = [lk for lk in links if lk not in drop]
    if not links:
        return {}, report
    return finish_links(ents, links, x0, x, keys, sl, report)


def walls_median(ents, x, sl, k):
    fd = ents[k]["fd"]
    return float(
        np.median(
            np.minimum(
                segment_distances(apply_2d(x[sl[k]], fd["bpts"]), fd["A"], fd["B"]).min(
                    axis=1
                ),
                50,
            )
        )
    )


def solve_links(ents, links):
    keys = sorted({k for lk in links for k in lk[:2]})
    sl = {k: slice(4 * i, 4 * i + 4) for i, k in enumerate(keys)}
    x0 = np.concatenate([ents[k]["x0"] for k in keys])
    lower, upper = [], []
    for k in keys:
        a, ls, tx, ty = ents[k]["x0"]
        ds = FIXED
        da = (
            math.radians(FACADE_HDG["hold_deg"])
            if ents[k]["fd"].get("heading_held")
            else math.radians(5)
        )
        lower += [a - da, ls - ds, tx - 15, ty - 15]
        upper += [a + da, ls + ds, tx + 15, ty + 15]
    lower, upper = np.array(lower), np.array(upper)

    def link_gap(x, lk):
        ka, kb, ia, ib, _ = lk
        Pa = apply_2d(x[sl[ka]], ents[ka]["fd"]["Lc"][ia])
        Pb = apply_2d(x[sl[kb]], ents[kb]["fd"]["Lc"][ib])
        return Pa, Pb

    def path_len(P):
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    x = x0.copy()
    for _ in range(8):
        terms = {
            k: wall_terms(
                x[sl[k]],
                ents[k]["fd"]["bpts"],
                ents[k]["fd"]["A"],
                ents[k]["fd"]["B"],
                ents[k]["fd"].get("owalls"),
                ents[k]["fd"].get("oriented"),
            )
            for k in keys
        }
        wl = [
            math.sqrt(
                max(min(len(terms[lk[0]][0]), len(terms[lk[1]][0])), 1) / len(lk[4])
            )
            for lk in links
        ]

        def residuals(xx):
            r = []
            for k in keys:
                S, Ad, Nd, sw = terms[k]
                fd, prm, n = ents[k]["fd"], xx[sl[k]], max(len(S), 1)
                r.append(
                    sw
                    * np.sum((apply_2d(prm, S) - Ad) * Nd, axis=1)
                    / OPT["wall_scale"]
                )
                gs = (
                    ORIENTED["gps_sigma"]
                    if fd.get("oriented") is not None
                    else OPT["gps_sigma"]
                )
                r.append(
                    math.sqrt(n)
                    * (apply_2d(prm, fd["Lc"]).mean(axis=0) - fd["gps"].mean(axis=0))
                    / gs
                )
                r.append(
                    [math.sqrt(n) * (prm[1] - fd["params0"][1]) / fd["scale_sigma"]]
                )
                if HEADING["prior_deg"]:
                    r.append(
                        [
                            math.sqrt(n)
                            * math.degrees(prm[0] - fd["params0"][0])
                            / HEADING["prior_deg"]
                        ]
                    )
            for lk, w in zip(links, wl):
                Pa, Pb = link_gap(xx, lk)
                r.append(w * ((Pa - Pb) / LINK_SIGMA).ravel())
            return np.concatenate([np.ravel(v) for v in r])

        x = least_squares(
            residuals,
            np.clip(x, lower, upper),
            bounds=(lower, upper),
            x_scale=np.tile([0.01, 0.01, 1, 1], len(keys)),
        ).x
    return x0, x, keys, sl


def finish_links(ents, links, x0, x, keys, sl, report):
    def link_gap(xx, lk):
        ka, kb, ia, ib, _ = lk
        return apply_2d(xx[sl[ka]], ents[ka]["fd"]["Lc"][ia]), apply_2d(
            xx[sl[kb]], ents[kb]["fd"]["Lc"][ib]
        )

    def path_len(P):
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    for lk in links:
        g0, g1 = [
            np.linalg.norm(np.subtract(*link_gap(v, lk)), axis=1) for v in (x0, x)
        ]
        (a0, b0), (a1, b1) = link_gap(x0, lk), link_gap(x, lk)
        report.append(
            {
                "legs": [lk[0][0], lk[1][0]],
                "folder": lk[0][1],
                "linked": True,
                "shared_frames": len(lk[4]),
                "shared_cameras_apart_m": {
                    "before": float(np.median(g0)),
                    "after": float(np.median(g1)),
                },
                "scale_ratio": {
                    "before": path_len(a0) / max(path_len(b0), 1e-9),
                    "after": path_len(a1) / max(path_len(b1), 1e-9),
                },
            }
        )
        r = report[-1]
        print(
            f"link {lk[0][0]}-{lk[1][0]} {lk[0][1]}: {len(lk[4])} shared frames {r['shared_cameras_apart_m']['before']:.2f} -> "
            f"{r['shared_cameras_apart_m']['after']:.2f} m apart, scale ratio {r['scale_ratio']['before']:.3f} -> "
            f"{r['scale_ratio']['after']:.3f}"
        )
    for k in keys:
        print(
            f"  {k[0]}/{k[1]}: wall fit {walls_median(ents, x0, sl, k):.2f} -> {walls_median(ents, x, sl, k):.2f} m"
        )
    changed = {}
    for k in keys:
        changed.setdefault(k[0], {})[k[1]] = x[sl[k]]
    return changed, report


def align_roi(roi_dir, max_error=2.0, tag=None, link=True):
    t_start, secs = time.time(), {}
    p = RoiPaths(roi_dir, tag)
    if not p.legs_json.exists():
        raise SystemExit(f"{p.legs_json} not found: run reconstruct.py first")
    doc = json.loads(p.legs_json.read_text())
    p.alignment.mkdir(parents=True, exist_ok=True)
    # outputs of legs an earlier run planned but this one does not (e.g. rerun as one leg)
    current = {
        f"{leg['id']}_{cam['folder']}" for leg in doc["legs"] for cam in leg["cameras"]
    }
    for d in p.alignment.iterdir():
        if d.is_dir() and (d / "transform.json").exists() and d.name not in current:
            shutil.rmtree(d)
    for f in p.results.glob("fit_*.png") if p.results.exists() else []:
        if f.stem[len("fit_") :] not in current:
            f.unlink()
    roi = json.loads(p.roi_json.read_text())
    frames = {s["folder"]: p.frames(s["folder"]) for s in p.sequences()}
    lons = [f["lon"] for fr in frames.values() for f in fr.values()]
    lats = [f["lat"] for fr in frames.values() for f in fr.values()]
    ref = LocalFrame(*roi["lines"][0][0])
    pad = OPT["margin"] + OPT["search_m"] + 40
    # OSM outlines are downloaded once per ROI and reused: any earlier run's saved copy (tagged runs,
    # old results) is taken before asking Overpass, which is slow and often times out
    cache = p.alignment / "osm_buildings.json"
    saved = sorted(p.data.glob("alignment*/osm_buildings.json")) + sorted(
        p.data.glob("old/*/osm_buildings.json")
    )
    if not cache.exists() and saved:
        cache.write_bytes(saved[0].read_bytes())
    buildings = fetch_buildings(
        (
            min(lons) - pad / ref.kx,
            min(lats) - pad / ref.ky,
            max(lons) + pad / ref.kx,
            max(lats) + pad / ref.ky,
        ),
        cache=cache,
    )
    print(f"OSM: {len(buildings)} building outlines")
    secs["osm_buildings"] = time.time() - t_start
    GPS_LEG.clear()
    GPS_MEASURED_FOLDERS.clear()
    if GPS["source"] == "auto":
        for leg in doc["legs"]:
            src, why = gps_check(p, leg)
            GPS_LEG[leg["id"]] = {"source": src, "reasons": why}
            if src == "measured":
                GPS_MEASURED_FOLDERS.update(c["folder"] for c in leg["cameras"])
                print(
                    f"{leg['id']}: measured GPS (Mapillary's computed positions look broken: "
                    f"{'; '.join(why)})"
                )

    def align_leg(leg, override=None):
        """Every camera of a leg: {folder: (entry, cameras_out, _shared, _fitdata)}. override: {folder:
        params} placing those cameras (link_legs' refined fits) instead of fitting them."""
        override = override or {}
        # a rig's reference camera first (its fit places the rig's other cameras); output in leg order
        refs = {c["rig"] for c in leg["cameras"] if c.get("rig") and c.get("model")}
        shared, out = {}, {}
        for cam in sorted(leg["cameras"], key=lambda c: c["folder"] not in refs):
            entry = {
                "leg": leg["id"],
                "folder": cam["folder"],
                "frames": cam["frames"],
                "cameras": [
                    {"folder": cam["folder"], "registered": cam.get("registered", 0)}
                ],
                "aligned": False,
            }
            if not cam.get("model"):
                entry["reason"] = "no reconstruction"
                out[cam["folder"]] = (entry, [], None, None)
                print(f"{leg['id']}/{cam['folder']}: no reconstruction")
                continue
            rig = cam.get("rig")
            out_dir = p.alignment / f"{leg['id']}_{cam['folder']}"
            prev = results.get(leg["id"], {}).get(cam["folder"])
            if cam["folder"] in override and prev and prev[2]:
                sh = {**prev[2], "fits": [override[cam["folder"]]]}
                res = align_camera(
                    p, leg, cam, buildings, out_dir, max_error, shared=sh
                )
                res["linked"] = True
            elif rig and rig == cam["folder"]:
                pool = [
                    c
                    for c in leg["cameras"]
                    if c.get("rig") == rig and c is not cam and c.get("model")
                ]
                res = align_camera(
                    p, leg, cam, buildings, out_dir, max_error, pool=pool, enu=ref
                )
            else:
                res = align_camera(
                    p,
                    leg,
                    cam,
                    buildings,
                    out_dir,
                    max_error,
                    shared=shared.get(rig) if rig else None,
                    enu=ref,
                )
            if rig and rig == cam["folder"] and res.get("aligned"):
                shared[rig] = res["_shared"]
            sh, fd = res.pop("_shared", None), res.pop("_fitdata", None)
            if prev and cam["folder"] in override:
                fd = prev[3]  # keep the first pass's fit data for the record
            cams_out = res.pop("cameras_out", [])
            entry.update(res)
            out[cam["folder"]] = (entry, cams_out, sh, fd)
            if res.get("aligned"):
                ptf = res["points_to_footprint_median_m"]
                print(
                    f"{leg['id']}/{cam['folder']}: {res['registered']}/{cam['frames']} cameras, "
                    f"{res['building_points']} building points: {ptf['mapillary_only']:.2f} -> {ptf['aligned']:.2f} m "
                    f"({100 * res['points_to_footprint_within_1m']:.0f}% within 1 m), scale x{res['scale_vs_mapillary']:.2f}, "
                    f"heading {res['heading_change_deg']:+.1f} deg, camera height {res['camera_height_m']:.2f} m, "
                    f"gravity spread {res['gravity']['spread_median_deg']:.1f} deg"
                    + (
                        f", {len(res['cameras_out_of_order'])} cameras behind earlier ones "
                        f"({', '.join(n[6:10] for n in res['cameras_out_of_order'])})"
                        if res["cameras_out_of_order"]
                        else ""
                    )
                    + (" (placed with the linked legs)" if res.get("linked") else "")
                    + f" -> {'reliable' if res['reliable'] else 'UNRELIABLE (' + '; '.join(res['reliability']['failed']) + ')'}"
                    + (
                        ""
                        if res["reliable"] == res["reliable_old"]
                        else f" [old verdict: {'reliable' if res['reliable_old'] else 'unreliable'}]"
                    )
                )
            else:
                print(f"{leg['id']}/{cam['folder']}: not aligned ({res.get('reason')})")
        return out

    results = {}
    t0 = time.time()
    for leg in doc["legs"]:
        t1 = time.time()
        results[leg["id"]] = align_leg(leg)
        secs[f"fit {leg['id']}"] = time.time() - t1
    secs["fit_legs"] = time.time() - t0
    link_report = []
    if link and len(doc["legs"]) > 1:
        t0 = time.time()
        changed, link_report = link_legs(doc["legs"], results)
        secs["link_legs"] = time.time() - t0
        t0 = time.time()
        for leg in doc["legs"]:
            if changed.get(leg["id"]):
                print(f"{leg['id']}: placing again with the linked fit")
                results[leg["id"]] = align_leg(leg, changed[leg["id"]])
        secs["place_linked_legs"] = time.time() - t0

    legs_out, per_folder, n_frames = [], {}, 0
    for leg in doc["legs"]:
        for c in leg["cameras"]:
            entry, cams_out, _, _ = results[leg["id"]][c["folder"]]
            legs_out.append(entry)
            n_frames += c["frames"]
            # frames two legs share belong to the later leg (it starts with them)
            have = per_folder.setdefault(c["folder"], {})
            for co in cams_out:
                have[co["file"]] = co
    per_folder = {f: [d[k] for k in sorted(d)] for f, d in per_folder.items()}
    for folder, cams in per_folder.items():
        (p.alignment / f"{folder}.json").write_text(
            json.dumps({"folder": folder, "cameras": cams}, indent=1)
        )
    aligned = [e for e in legs_out if e.get("aligned")]
    summary = {
        "pipeline": "Alignment_Pipeline",
        "reconstruction": {
            k: doc.get(k)
            for k in (
                "features",
                "mapper",
                "refine_intrinsics",
                "gravity_priors",
                "mask_moving",
            )
        },
        "frames": n_frames,
        "leg_links": link_report,
        "frames_aligned": sum(e["registered"] for e in aligned),
        "frames_reliable": sum(e["registered"] for e in aligned if e["reliable"]),
        "procedure": {
            "fit": "shift",
            "walls": WALLS["mode"],
            "heading": HEADING["mode"],
            "heading_prior_deg": HEADING["prior_deg"],
            "gps": GPS["source"],
            "gps_legs": GPS_LEG if GPS["source"] == "auto" else None,
        },
        "frames_reliable_old": sum(
            e["registered"] for e in aligned if e.get("reliable_old")
        ),
        "legs": legs_out,
    }
    t0 = time.time()
    plot_overview(p, aligned, per_folder, buildings)
    from .plot_fit import plot_leg  # the sparse cloud on the footprints, plan and 3D

    for e in aligned:
        d = p.alignment / f"{e['leg']}_{e['folder']}"
        if (d / "transform.json").exists():
            plot_leg(p, d, buildings)
    secs["figures"] = time.time() - t0
    secs["total"] = time.time() - t_start
    summary["seconds"] = {k: round(v, 1) for k, v in secs.items()}
    p.summary_json.write_text(json.dumps(summary, indent=1))
    print(
        "align time: "
        + ", ".join(
            f"{k} {v:.0f} s" for k, v in secs.items() if not k.startswith("fit leg")
        )
    )
    print(
        f"{summary['frames_aligned']}/{n_frames} frames aligned, {summary['frames_reliable']} reliable; "
        f"wrote {p.summary_json}"
    )
    return summary


def plot_overview(p, aligned, per_folder, buildings):
    cams = [c for cs in per_folder.values() for c in cs]
    if not cams:
        return
    enu = LocalFrame(cams[0]["lon"], cams[0]["lat"])
    xy = np.array([enu.xy(c["lon"], c["lat"]) for c in cams])
    lo, hi = xy.min(axis=0) - 60, xy.max(axis=0) + 60
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, ring in buildings:
        r = np.array([enu.xy(lon, lat) for lon, lat in ring])
        if np.all(r.max(axis=0) >= lo) and np.all(r.min(axis=0) <= hi):
            ax.fill(*r.T, color="#e3e3e3", ec="#888", lw=0.5, zorder=0)
    for folder, cs in per_folder.items():
        a = np.array([enu.xy(c["lon"], c["lat"]) for c in cs])
        g = np.array(
            [
                enu.xy(
                    *(
                        c["gps"]
                        if (
                            GPS["source"] == "measured"
                            or folder in GPS_MEASURED_FOLDERS
                        )
                        and c.get("gps")
                        else c["mapillary_lonlat"]
                    )
                )
                for c in cs
            ]
        )
        rel = np.array([c["reliable"] for c in cs])
        ax.plot(*g.T, "x", color="#888", ms=3)
        ax.scatter(*a[rel].T, s=8, color="#1f4e9c", zorder=2)
        ax.scatter(*a[~rel].T, s=8, color="#d62728", zorder=2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(
        f"{p.name}: aligned cameras (blue reliable, red unreliable), "
        f"{'measured GPS' if GPS['source'] == 'measured' else 'GPS used (computed or measured per leg)' if GPS['source'] == 'auto' else 'Mapillary positions'} (x)",
        fontsize=10,
        loc="left",
    )
    p.results.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.results / "overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Fit each leg on its own (by default legs sharing frames are refined together)",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=2.0,
        help="Reliable if the median point-to-footprint distance is at most this (m)",
    )
    parser.add_argument("--tag", help="Separate run (see reconstruct.py --tag)")
    parser.add_argument(
        "--walls",
        choices=["hybrid", "oriented", "nearest"],
        default="hybrid",
        help="hybrid (default): oriented plus the points without a facade direction by nearest wall "
        "(HYBRID); oriented: matched by facade direction, far points kept, cross-street facades "
        "weighted (see the docstring); nearest: building points to the nearest footprint edge",
    )
    parser.add_argument(
        "--gps",
        choices=["auto", "computed", "measured"],
        default="auto",
        help="auto (default): Mapillary's computed positions, per leg the measured GPS where they look "
        "broken (gps_check); computed; measured",
    )
    parser.add_argument(
        "--heading-prior",
        type=float,
        help="Hold each leg's heading loosely to its GPS placement's: prior std in degrees (e.g. 3; off by default)",
    )
    parser.add_argument(
        "--heading",
        choices=["facades", "free"],
        default="facades",
        help="facades (default): from the facade directions against the OSM walls where they give a "
        "clear answer, then held within +-1 deg (see the docstring); free: searched within +-15 deg",
    )
    args = parser.parse_args()
    HEADING["mode"] = args.heading
    HEADING["prior_deg"] = args.heading_prior
    GPS["source"] = args.gps
    WALLS["mode"] = args.walls
    align_roi(args.roi_dir, args.max_error, args.tag, link=not args.no_link)


if __name__ == "__main__":
    main()
