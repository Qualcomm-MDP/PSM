"""Building heights from airborne LiDAR (USGS 3DEP, or local LAS/LAZ tiles).

With --lidar, each OSM building's height comes from the LiDAR points on its roof instead
of the OSM tags (height, else building:levels x 3.5 m, else 10.5 m). Buildings the LiDAR
can't measure keep their OSM height.

Where the points come from
  USGS 3DEP     the public Entwine Point Tiles (EPT) on AWS (s3://usgs-lidar-public), found
                with Hobu's boundary index; the newest collection that covers the area.
                US only. Downloaded tiles are cached (LIDAR_CACHE_DIR, default
                ~/.cache/lidar_eval: the same tiles as the geometry evaluation's, so an
                area is downloaded once for both).
  local tiles   --lidar-laz a.laz b.laz: any LAS/LAZ with a CRS in its header (or
                --lidar-laz-crs). Heights in feet are converted to metres.

How a height is measured, and why it doesn't come out too tall
  The mesh extrudes each footprint as one flat-topped block, so one number has to stand
  for the whole roof. The highest LiDAR point is a poor choice: trees over the roof,
  antennas, chimneys, rooftop units, a neighbour's taller wall and noise all sit above the
  roof. So:
  - only points that can be roof: noise (7, 18), water, bridges, ... are dropped; building
    class 6 when the collection classifies buildings, else unclassified points; and only
    those inside the footprint shrunk by `erode` (1 m), more than `min_above` (2 m) above
    the ground (a neighbour's wall or a tree at the edge isn't counted)
  - the roof is found as planes (sequential RANSAC); points on no plane (trees, antennas,
    equipment, noise) are ignored. roof="main" (default) takes the LARGEST plane's median,
    the main roof (for a pitched roof, mid-slope: the height a flat-topped block of the
    same volume would have). roof="top" takes the highest plane, as the geometry
    evaluation does. Without a planar roof, the median (main) or 95th percentile (top) of
    the roof points.
  - never above the 95th percentile of the roof points
  - height is measured from the HIGHEST LiDAR ground along the outline, because that is
    where extrude.py stands the roof on (roof = highest ground under the footprint +
    height): measured from the lowest ground, buildings on a slope would come out taller
    by the drop. Both ground and roof are LiDAR, so their vertical datums agree.
  - capped by what OSM says: at most a mapped `height` + max(50%, 6 m); with only
    `building:levels`, at most levels x 5 m + 6 m; and at most `max_height` if given.
  - towers (building=tower, man_made=tower or a tower:type tag) take their top instead:
    the 99th percentile of their roof points, without the building:levels cap. Their top
    is small, steep or open, so the main roof would be a lower level.
  - and not far below it: when the LiDAR comes out more than max(30%, 6 m) below a mapped
    `height`, the mapped height is kept. A tower's top is small, steep or open, so few
    points land on it and its largest surface is a lower level (Lurie Tower, 51 m tall,
    measured ~33 m). Levels-only and untagged buildings have no such floor.
  A building keeps its OSM height when the LiDAR has (almost) no roof points on it
  (coverage < 50%: built after the flight, or demolished), too little ground around it,
  too small a footprint, or it is a raised part (min_height, e.g. a bridge or overhang).
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
import shapely
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

from common.MeshUtils import (
    _get_utm_transformer,
    building_heights,
    get_corners,
    parse_metres,
)
from common.http_retry import get_with_retries
from common.reporter import Reporter, get_reporter, notify

USGS_RESOURCES = (
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/"
    "boundaries/resources.geojson"
)
RESOURCES_MAX_AGE_S = 7 * 24 * 3600
GROUND, BUILDING = 2, 6
VEGETATION = (3, 4, 5)
KEEP_CLASSES = (1, 2, 3, 4, 5, 6)  # drop noise (7, 18), water (9), bridge (17), ...


def default_cache_dir() -> Path:
    return Path(
        os.environ.get("LIDAR_CACHE_DIR") or Path.home() / ".cache" / "lidar_eval"
    )


class LidarUnavailable(RuntimeError):
    """No LiDAR here (outside the coverage, laspy missing): the OSM heights are used."""


@dataclass
class LidarParams:
    roof: str = "main"  # "main": the largest roof plane; "top": the highest one
    max_height: float | None = None  # absolute cap, m
    erode: float = 1.0  # footprint shrunk by this before taking roof points, m
    ring_in: float = 0.5  # ground points from this far outside the footprint ...
    ring_out: float = 3.0  # ... to this far, m
    min_above: float = 2.0  # roof points at least this far above the ground, m
    min_coverage: float = 0.5  # share of 1 m cells of the roof with a point
    min_ground_pts: int = 10
    min_roof_area: float = 10.0  # m^2, after erosion
    roof_pct_cap: float = 95.0  # the roof is never above this percentile of its points
    tower_pct: float = 99.0  # a tower's height: this percentile of its points
    ransac_thresh: float = 0.3
    ransac_min_share: float = 0.10
    ransac_max_outliers: float = 0.5
    # caps from the OSM tags
    tag_slack_ratio: float = 0.5
    tag_slack_m: float = 6.0
    # a mapped height is kept when the LiDAR is more than max(30%, 6 m) below it: a tower's
    # small, steep or open top gets few points, and its largest surface is a lower level
    tag_below_ratio: float = 0.3
    tag_below_m: float = 6.0
    level_max_m: float = 5.0
    level_slack_m: float = 6.0


@dataclass
class LidarPoints:
    """Points in the pipeline's UTM frame: x, y, z (m) and ASPRS class."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    c: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Reading the points
# ----------------------------------------------------------------------------
def _laspy():
    try:
        import laspy

        return laspy
    except ModuleNotFoundError:
        raise LidarUnavailable(
            "the laspy package is missing (poetry install --with eval)"
        ) from None


def _z_factor(crs: CRS | None) -> float:
    """Metres per unit of z: the vertical CRS's unit, else the horizontal one's
    (State Plane LAZ in feet usually has z in feet too), else 1."""
    if crs is None:
        return 1.0
    subs = crs.sub_crs_list if crs.is_compound else [crs]
    for sub in reversed(subs):  # the vertical part comes last in a compound CRS
        if sub.axis_info:
            axis = sub.axis_info[-1] if sub.is_vertical else sub.axis_info[0]
            factor = axis.unit_conversion_factor
            if factor and math.isfinite(factor):
                return float(factor)
    return 1.0


def _point_arrays(las, xmin, ymin, xmax, ymax):
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    c = np.asarray(las.classification).astype(np.uint8)
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    try:
        keep &= ~np.asarray(las.withheld, dtype=bool)
    except Exception:  # noqa: BLE001 - flag absent in some formats
        pass
    keep &= np.isin(c, KEEP_CLASSES)
    return x[keep], y[keep], z[keep], c[keep]


def _usgs_index(session: requests.Session, cache_dir: Path) -> dict:
    """Hobu's index of the USGS EPT collections, cached for a week."""
    path = cache_dir / "usgs_resources.geojson"
    if path.exists() and time.time() - path.stat().st_mtime < RESOURCES_MAX_AGE_S:
        return json.loads(path.read_text(encoding="utf-8"))
    r = get_with_retries(session, USGS_RESOURCES, timeout=60)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return r.json()


def _collection_year(name: str) -> int:
    years = [
        int(t)
        for t in name.replace("-", "_").split("_")
        if t.isdigit() and len(t) == 4 and 1990 <= int(t) <= 2100
    ]
    return max(years) if years else 0


def find_usgs_ept(lon: float, lat: float, index: dict) -> list[dict]:
    """The USGS EPT collections that contain (lon, lat), newest first."""
    pt = shapely.Point(lon, lat)
    hits = [
        f["properties"]
        for f in index.get("features", [])
        if shapely.geometry.shape(f["geometry"]).contains(pt)
    ]
    return sorted(hits, key=lambda p: _collection_year(p.get("name", "")), reverse=True)


def read_ept(
    url: str,
    aoi_utm: tuple[float, float, float, float],
    utm_crs: CRS,
    cache_dir: Path,
    reporter: Reporter,
    workers: int = 8,
) -> LidarPoints:
    """Every EPT node that overlaps the AOI, at full resolution."""
    laspy = _laspy()
    session = requests.Session()
    base = url.rsplit("/", 1)[0]
    name = base.rsplit("/", 1)[-1]
    info = get_with_retries(session, url, timeout=60).json()
    srs = info.get("srs", {})
    if srs.get("authority") and srs.get("horizontal"):
        ept_crs = CRS.from_user_input(f"{srs['authority']}:{srs['horizontal']}")
    else:
        ept_crs = CRS.from_wkt(srs["wkt"])
    vertical = srs.get("vertical")
    zf = _z_factor(CRS.from_user_input(f"EPSG:{vertical}")) if vertical else 1.0

    to_ept = Transformer.from_crs(utm_crs, ept_crs, always_xy=True)
    ax0, ay0, ax1, ay1 = aoi_utm
    ex, ey = to_ept.transform([ax0, ax1, ax1, ax0], [ay0, ay0, ay1, ay1])
    qx0, qy0, qx1, qy1 = min(ex), min(ey), max(ex), max(ey)
    bx0, by0, _, bx1, _, _ = info["bounds"]
    span = bx1 - bx0

    def overlaps(key: str) -> bool:
        d, X, Y, _ = (int(v) for v in key.split("-"))
        step = span / 2**d
        x0, y0 = bx0 + X * step, by0 + Y * step
        return not (x0 > qx1 or x0 + step < qx0 or y0 > qy1 or y0 + step < qy0)

    nodes, todo = set(), ["0-0-0-0"]
    while todo:  # the hierarchy is split over several files
        hkey = todo.pop()
        h = get_with_retries(session, f"{base}/ept-hierarchy/{hkey}.json", timeout=60)
        for key, count in h.json().items():
            if not overlaps(key):
                continue
            if count == -1:
                if key != hkey:
                    todo.append(key)
            elif count > 0:
                nodes.add(key)
    if not nodes:
        raise LidarUnavailable(f"the {name} collection has no points here")
    reporter.info(f"  LiDAR {name}: {len(nodes)} tiles over the area")

    cache = cache_dir / name
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(key: str):
        f = cache / f"{key}.laz"
        if not f.exists():
            r = get_with_retries(session, f"{base}/ept-data/{key}.laz", timeout=180)
            tmp = f.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(f)
        return _point_arrays(laspy.read(io.BytesIO(f.read_bytes())), qx0, qy0, qx1, qy1)

    with cf.ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(fetch, sorted(nodes)))
    x, y, z, c = (np.concatenate([p[k] for p in parts]) for k in range(4))
    ux, uy = Transformer.from_crs(ept_crs, utm_crs, always_xy=True).transform(x, y)
    return LidarPoints(
        np.asarray(ux),
        np.asarray(uy),
        z * zf,
        c,
        {"source": name, "year": _collection_year(name), "url": url},
    )


def read_laz_files(
    paths: list[str],
    aoi_utm: tuple[float, float, float, float],
    utm_crs: CRS,
    reporter: Reporter,
    laz_crs: str | None = None,
) -> LidarPoints:
    laspy = _laspy()
    parts = []
    for p in paths:
        with laspy.open(p) as fh:
            crs = CRS.from_user_input(laz_crs) if laz_crs else fh.header.parse_crs()
            if crs is None:
                raise LidarUnavailable(
                    f"{Path(p).name} has no CRS in its header (give --lidar-laz-crs)"
                )
            hcrs = crs.sub_crs_list[0] if crs.is_compound else crs
            zf = _z_factor(crs)
            to_f = Transformer.from_crs(utm_crs, hcrs, always_xy=True)
            back = Transformer.from_crs(hcrs, utm_crs, always_xy=True)
            ax0, ay0, ax1, ay1 = aoi_utm
            fx, fy = to_f.transform([ax0, ax1, ax1, ax0], [ay0, ay0, ay1, ay1])
            n = 0
            for chunk in fh.chunk_iterator(5_000_000):
                x, y, z, c = _point_arrays(chunk, min(fx), min(fy), max(fx), max(fy))
                ux, uy = back.transform(x, y)
                parts.append((np.asarray(ux), np.asarray(uy), z * zf, c))
                n += len(x)
            reporter.info(
                f"  LiDAR {Path(p).name}: {n:,} points in the area"
                + (f" (z in feet, x{zf:.4f} to metres)" if abs(zf - 1) > 1e-6 else "")
            )
    if not parts or not sum(len(q[0]) for q in parts):
        raise LidarUnavailable("the LAS/LAZ files have no points in this area")
    x, y, z, c = (np.concatenate([q[k] for q in parts]) for k in range(4))
    return LidarPoints(x, y, z, c, {"source": ", ".join(Path(p).name for p in paths)})


def load_points(
    aoi_utm: tuple[float, float, float, float],
    utm_crs: CRS,
    reporter: Reporter,
    ept_url: str | None = None,
    laz: list[str] | None = None,
    laz_crs: str | None = None,
    cache_dir: Path | None = None,
) -> LidarPoints:
    cache_dir = Path(cache_dir or default_cache_dir())
    if laz:
        return read_laz_files(laz, aoi_utm, utm_crs, reporter, laz_crs)
    if not ept_url:
        to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(
            (aoi_utm[0] + aoi_utm[2]) / 2, (aoi_utm[1] + aoi_utm[3]) / 2
        )
        hits = find_usgs_ept(lon, lat, _usgs_index(requests.Session(), cache_dir))
        if not hits:
            raise LidarUnavailable("no USGS 3DEP collection covers this area")
        if len(hits) > 1:
            reporter.info(
                "  USGS collections here: " + ", ".join(h["name"] for h in hits)
            )
        ept_url = hits[0]["url"]
    return read_ept(ept_url, aoi_utm, utm_crs, cache_dir, reporter)


# ----------------------------------------------------------------------------
# One building's height
# ----------------------------------------------------------------------------
def robust_plane(px, py, pz, iters=3):
    """z = a + b*(x-x0) + c*(y-y0), refit after dropping |res| > 2.5 NMAD."""
    x0, y0 = px.mean(), py.mean()
    keep = np.ones(len(pz), bool)
    coef = np.array([np.median(pz), 0.0, 0.0])
    for _ in range(iters):
        if keep.sum() < 10:
            break
        A = np.column_stack([np.ones(keep.sum()), px[keep] - x0, py[keep] - y0])
        coef, *_ = np.linalg.lstsq(A, pz[keep], rcond=None)
        res = pz - (coef[0] + coef[1] * (px - x0) + coef[2] * (py - y0))
        nmad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 0.02
        keep = np.abs(res) <= 2.5 * nmad
    return lambda qx, qy: coef[0] + coef[1] * (qx - x0) + coef[2] * (qy - y0)


def ransac_roof(
    px, py, pz, thresh=0.3, min_share=0.10, min_pts=20, max_planes=4, iters=300,
    max_tilt_deg=60.0, seed=0,
):  # fmt: skip
    """Roof planes by sequential RANSAC. Returns (top, main, n_planes, outlier_share) or
    None: top = the highest plane (99th percentile of its fitted elevations), main = the
    largest plane's median fitted elevation. Points on no plane don't count."""
    n_all = len(pz)
    if n_all < max(min_pts, 3):
        return None
    rng = np.random.default_rng(seed)
    cos_max = math.cos(math.radians(max_tilt_deg))
    x0, y0 = float(np.mean(px)), float(np.mean(py))
    P = np.column_stack([px - x0, py - y0, pz])
    remaining = np.arange(n_all)
    fitted, sizes = [], []
    while len(fitted) < max_planes and len(remaining) >= max(min_pts, 3):
        Q = P[remaining]
        best = None
        for _ in range(iters):
            a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9 or abs(nrm[2]) / ln < cos_max:
                continue
            inl = np.abs((Q - a) @ (nrm / ln)) < thresh
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < max(min_pts, min_share * n_all):
            break
        A = np.column_stack([np.ones(best.sum()), Q[best, 0], Q[best, 1]])
        coef, *_ = np.linalg.lstsq(A, Q[best, 2], rcond=None)
        res = Q[:, 2] - (coef[0] + coef[1] * Q[:, 0] + coef[2] * Q[:, 1])
        inl = np.abs(res) < thresh
        if inl.sum() < max(min_pts, min_share * n_all):
            break
        fitted.append(coef[0] + coef[1] * Q[inl, 0] + coef[2] * Q[inl, 1])
        sizes.append(int(inl.sum()))
        remaining = remaining[~inl]
    if not fitted:
        return None
    top = float(max(np.percentile(f, 99) for f in fitted))
    main = float(np.median(fitted[int(np.argmax(sizes))]))
    return top, main, len(fitted), 1.0 - sum(sizes) / n_all


def _cells_covered(px, py, cell=1.0) -> int:
    if len(px) == 0:
        return 0
    return len(
        np.unique(
            np.floor(px / cell).astype(np.int64) * 1_000_003
            + np.floor(py / cell).astype(np.int64)
        )
    )


def prepare_points(pts: LidarPoints) -> dict:
    """Ground and roof candidates, indexed for lookups by position."""
    x, y, z, c = pts.x, pts.y, pts.z, pts.c
    if len(x) == 0:
        raise LidarUnavailable("no LiDAR points in this area")
    building_class = (c == BUILDING).sum() > 0.005 * len(c)
    gsel = c == GROUND
    rsel = (c == BUILDING) if building_class else (c == 1)
    ctx = {
        "gx": x[gsel], "gy": y[gsel], "gz": z[gsel],
        "rx": x[rsel], "ry": y[rsel], "rz": z[rsel],
        "box": box(x.min(), y.min(), x.max(), y.max()),
        "building_class": bool(building_class),
    }  # fmt: skip
    ctx["gtree"] = cKDTree(np.column_stack([ctx["gx"], ctx["gy"]]))
    ctx["rtree"] = cKDTree(np.column_stack([ctx["rx"], ctx["ry"]]))
    return ctx


def osm_height_tag(tags: dict) -> float | None:
    """The building's mapped height (height, else building:height), m."""
    return parse_metres(tags.get("height") or tags.get("building:height")) or None


def lidar_far_below_tag(h: float, tags: dict, p: LidarParams) -> bool:
    """Whether a LiDAR height is so far below the mapped height that the tag is kept."""
    tag = osm_height_tag(tags)
    if tag is None:
        return False
    return h < tag - max(p.tag_below_ratio * tag, p.tag_below_m)


def osm_height_cap(tags: dict, p: LidarParams) -> tuple[float | None, str]:
    """The tallest a LiDAR height may be, given what OSM says about the building."""
    tag = parse_metres(tags.get("height") or tags.get("building:height"))
    if tag:
        return tag + max(p.tag_slack_ratio * tag, p.tag_slack_m), "height tag"
    levels = (
        parse_metres(tags.get("building:levels"))
        if tags.get("building:levels")
        else None
    )
    if levels:
        return levels * p.level_max_m + p.level_slack_m, "levels tag"
    return None, ""


def is_tower(tags: dict) -> bool:
    """OSM says it's a tower: building=tower, man_made=tower, or any tower:type."""
    return (
        tags.get("building") == "tower"
        or tags.get("man_made") == "tower"
        or bool(tags.get("tower:type"))
    )


def measure_building(poly, ctx: dict, p: LidarParams, tower: bool = False) -> dict:
    """LiDAR height of one footprint (UTM polygon): {"status": "ok", "height": ...} or
    {"status": <why not>}. The height is roof - highest ground along the outline.
    tower: the top (tower_pct percentile of the points), not the main roof: a tower's
    top is small, steep or open, so its largest surface is a lower level."""
    if not ctx["box"].contains(poly.buffer(p.ring_out)):
        return {"status": "outside_lidar"}
    inner = poly.buffer(-p.erode)
    if inner.is_empty or inner.area < p.min_roof_area:
        return {"status": "too_small"}
    gx, gy, gz, rx, ry, rz = (ctx[k] for k in ("gx", "gy", "gz", "rx", "ry", "rz"))
    minx, miny, maxx, maxy = poly.bounds
    ctr = ((minx + maxx) / 2, (miny + maxy) / 2)
    rad = math.hypot(maxx - minx, maxy - miny) / 2 + p.ring_out + 0.5

    # sorted: RANSAC samples by position, so the same points must come in the same order
    gi = np.sort(np.asarray(ctx["gtree"].query_ball_point(ctr, rad), dtype=np.int64))
    ring = poly.buffer(p.ring_out).difference(poly.buffer(p.ring_in))
    if len(gi):
        gi = gi[shapely.contains_xy(ring, gx[gi], gy[gi])]
    if len(gi) < p.min_ground_pts:
        return {"status": "no_ground"}
    plane = robust_plane(gx[gi], gy[gi], gz[gi])
    outline = np.asarray(poly.exterior.segmentize(1.0).coords)[:, :2]
    g_outline = plane(outline[:, 0], outline[:, 1])
    g_low, g_high = float(g_outline.min()), float(g_outline.max())

    ri = np.sort(np.asarray(ctx["rtree"].query_ball_point(ctr, rad), dtype=np.int64))
    if len(ri):
        ri = ri[shapely.contains_xy(inner, rx[ri], ry[ri])]
        ri = ri[rz[ri] > g_low + p.min_above]
    ri = ri[np.lexsort((rz[ri], ry[ri], rx[ri]))]
    coverage = min(1.0, _cells_covered(rx[ri], ry[ri]) / max(inner.area, 1.0))
    m: dict[str, Any] = {"coverage": round(coverage, 3), "n_roof_pts": int(len(ri))}
    if coverage < p.min_coverage or len(ri) < 5:
        return {**m, "status": "not_in_lidar"}

    z = rz[ri]
    if tower:
        roof = float(np.percentile(z, p.tower_pct))
        m.update(
            status="ok",
            roof_method=f"tower: {p.tower_pct:g}th percentile",
            height=round(roof - g_high, 2),
            ground_drop=round(g_high - g_low, 2),
        )
        return m
    rs = ransac_roof(
        rx[ri], ry[ri], z, thresh=p.ransac_thresh, min_share=p.ransac_min_share
    )
    if rs and rs[3] <= p.ransac_max_outliers:
        roof = rs[1] if p.roof == "main" else rs[0]
        m["roof_method"] = f"plane ({rs[2]} found)"
    else:  # no plane, or most points on none (towers, domes, open structures)
        roof = float(np.percentile(z, 50 if p.roof == "main" else 95))
        m["roof_method"] = "median" if p.roof == "main" else "95th percentile"
    roof = min(roof, float(np.percentile(z, p.roof_pct_cap)))
    m.update(
        status="ok",
        height=round(roof - g_high, 2),
        ground_drop=round(g_high - g_low, 2),
    )
    return m


# ----------------------------------------------------------------------------
# All the buildings, and the pipeline stage
# ----------------------------------------------------------------------------
def element_key(element: dict) -> str:
    return f"{element.get('type', 'way')}/{element.get('id')}"


def _closed(element: dict) -> bool:
    """A closed outline: 4+ points, the last one back at the first."""
    g = element.get("geometry") or []
    return (
        len(g) >= 4
        and abs(g[0]["lat"] - g[-1]["lat"]) < 1e-9
        and abs(g[0]["lon"] - g[-1]["lon"]) < 1e-9
    )


def _outlines(elements: list[dict]):
    """(key, outline element, tags) for every outline extrude.py builds: each building
    way, and each member way of a relation that is a closed outline (key
    "relation/<id>/way/<member>"; the relation's tags decide the OSM caps, as its member
    ways carry none of their own). Inner members (courtyards) are skipped: they have no
    roof of their own, and keep extrude.py's height."""
    for el in elements:
        if el.get("type") == "relation":
            for m in el.get("members", []):
                if (
                    m.get("type", "way") == "way"
                    and m.get("role", "outer") != "inner"
                    and _closed(m)
                ):
                    key = f"relation/{el.get('id')}/way/{m.get('ref')}"
                    yield key, m, el.get("tags", {})
        elif "geometry" in el:
            yield element_key(el), el, el.get("tags", {})


def _footprint(element: dict, bbox) -> Polygon | None:
    corners = get_corners(element, bbox)
    if len(corners) < 3:
        return None
    poly = Polygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    return poly if poly.geom_type == "Polygon" and poly.area > 1.0 else None


def lidar_building_heights(
    elements: list[dict],
    bbox,
    params: LidarParams | None = None,
    reporter: Reporter | None = None,
    points: LidarPoints | None = None,
    **source,
) -> tuple[dict[str, float], list[dict]]:
    """({outline key: height in m}, one row per outline) for the OSM buildings: their
    ways, and the outer outlines of their relations (see _outlines).

    `points` (already in the pipeline's UTM frame) or `source` (ept_url, laz, laz_crs,
    cache_dir for load_points)."""
    p = params or LidarParams()
    reporter = reporter or get_reporter()
    feats = []
    for key, outline, tags in _outlines(elements):
        poly = _footprint(outline, bbox)
        if poly is not None:
            feats.append((key, tags, poly))
    if not feats:
        return {}, []

    lon0 = (bbox.min_lon + bbox.max_lon) / 2
    lat0 = (bbox.min_lat + bbox.max_lat) / 2
    utm_crs = _get_utm_transformer(lon0, lat0).target_crs
    if points is None:
        pad = p.ring_out + 2.0
        x0, y0, x1, y1 = shapely.unary_union([g for *_, g in feats]).bounds
        points = load_points(
            (x0 - pad, y0 - pad, x1 + pad, y1 + pad), utm_crs, reporter, **source
        )
    ctx = prepare_points(points)
    reporter.info(
        f"  {len(ctx['gx']):,} ground points, {len(ctx['rx']):,} roof candidates ("
        + (
            "building class"
            if ctx["building_class"]
            else "no building class: unclassified"
        )
        + ")"
    )

    heights, rows = {}, []
    for key, tags, poly in feats:
        base, osm_top, source_tag = building_heights(tags)
        row = {"key": key, "osm_height": osm_top, "osm_source": source_tag}
        if base > 0:
            rows.append({**row, "status": "raised_part"})
            continue
        tower = is_tower(tags)
        m = measure_building(poly, ctx, p, tower=tower)
        row.update(m)
        if m["status"] == "ok" and lidar_far_below_tag(m["height"], tags, p):
            m["status"] = "lidar_far_below_height_tag"
            row.update(m, lidar_height=m["height"])
        if m["status"] == "ok":
            h, capped = m["height"], []
            cap, why = osm_height_cap(tags, p)
            if tower and why == "levels tag":
                cap = None  # a tower's few levels say little about its height
            if cap is not None and h > cap:
                h, capped = cap, [why]
            if p.max_height is not None and h > p.max_height:
                h, capped = p.max_height, ["max height"]
            row["lidar_height"], row["used_height"] = m["height"], round(h, 2)
            row["capped_by"] = ", ".join(capped)
            heights[row["key"]] = float(h)
        rows.append(row)

    _report(rows, points.meta, reporter)
    return heights, rows


def _report(rows: list[dict], meta: dict, reporter: Reporter) -> None:
    used = [r for r in rows if "used_height" in r]
    kept: dict[str, int] = {}
    for r in rows:
        if "used_height" not in r:
            kept[r["status"]] = kept.get(r["status"], 0) + 1
    src = meta.get("source", "LiDAR")
    reporter.info(
        f"  LiDAR heights for {len(used)} of {len(rows)} buildings ({src})"
        + (
            "; OSM heights kept for "
            + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(kept.items()))
            if kept
            else ""
        )
    )
    if used:
        diff = np.array([r["used_height"] - r["osm_height"] for r in used])
        n_capped = sum(bool(r["capped_by"]) for r in used)
        reporter.info(
            f"  vs OSM: median {np.median(diff):+.1f} m, "
            f"{(diff > 0).sum()} taller, {(diff < 0).sum()} lower"
            + (
                f"; {n_capped} capped by the OSM tags or the max height"
                if n_capped
                else ""
            )
        )


def lidar_stage(value, state):
    """Measure building heights from LiDAR when --lidar is on; pass the input through.
    The heights go to build_mesh as state["lidar_heights"]; without LiDAR (no coverage,
    a download failure) the buildings keep their OSM heights and the user is told."""
    if not state.get_metadata("use_lidar", False):
        return value
    reporter = get_reporter(state)
    opts = dict(state.get_metadata("lidar_options", None) or {})
    params = LidarParams(
        **{k: opts.pop(k) for k in list(opts) if k in LidarParams.__dataclass_fields__}
    )
    try:
        heights, rows = lidar_building_heights(
            value["osm"]["elements"],
            state.require_metadata("bbox"),
            params,
            reporter,
            **opts,
        )
        state.set_metadata("lidar_heights", heights)
        state.set_metadata("lidar_rows", rows)
        if rows and not heights:
            notify(
                state,
                "No LiDAR heights: the LiDAR measured none of the buildings; they keep "
                "their OpenStreetMap heights",
            )
    except Exception as exc:  # noqa: BLE001 - OSM heights are the old behaviour
        notify(
            state,
            f"No LiDAR heights for this area ({_short_error(exc)}); the buildings keep "
            "their OpenStreetMap heights",
        )
    return value


def _short_error(exc: Exception) -> str:
    text = str(exc)
    return text.split(" for url: ")[0] if " for url: " in text else text[:200]
