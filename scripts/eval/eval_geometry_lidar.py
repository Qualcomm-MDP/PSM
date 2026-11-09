"""
eval_geometry_lidar.py
Geometry evaluation of the pipeline's 3D buildings (+ terrain) against airborne LiDAR.

It evaluates the model the pipeline wrote: run the pipeline (web app or src/main.py),
then point this script at the output. Nothing in the pipeline changes.
  which      OSM (fetched again for the model's bounding box, as the fetch stage does)
             splits the mesh into buildings and gives each its id, name and height tags
  geometry   from the .glb: roof = highest horizontal face inside the footprint, base =
             lowest one, model terrain = the .glb's ground mesh
  where      the .glb is centred on the model, so the script works out where it sits in
             UTM from the job's bounding box (job.json) and the terrain grid (with
             terrain) or the OSM outlines (flat model)
OSM buildings with nothing in the mesh are reported as "not_in_mesh" (e.g. ones mapped as
relations, which extrude.py does not build yet). A model built with terrain gets every
metric; a flat one only the height. LiDAR (USGS 3DEP, fetched automatically) is only the
reference.

How results are judged (draft requirement QR-06)
  QR-06a  terrain accuracy  PASS/FAIL. Terrain data is complete and measured. Share of buildings whose
                            terrain error |M2| <= --terrain-tol (1 m) must reach
                            --terrain-target (95%) at the lower end of its 95% CI.
  QR-06b  building height   BASELINE, not pass/fail. Heights come from OSM, which often has
                            no measured height (known data limitation, Risk 7). The share
                            within --height-tol (3 m) is reported by height confidence, and
                            --height-baseline compares a new run with an earlier one
                            (IMPROVED / NO CLEAR CHANGE / WORSE).
  height confidence (per building, from OSM tags; needs no LiDAR, so the pipeline can ship it)
          high = explicit `height` tag, medium = `building:levels` only, low = no tag (default)

Metrics (per building, in meters, signed = model - LiDAR, + means model too high/tall)
  M1  height error      model height - LiDAR height. Model height = mesh roof - highest
                        model ground under the footprint (the height extrude.py used:
                        the OSM tag, else levels x 3.5 m, else 10.5 m)
                        LiDAR height = roof top - lowest ground at the footprint edge
                        (same definition as the OSM `height` tag: lowest ground contact
                        to top of roof). No terrain involved.
  M2  terrain error     model terrain - LiDAR ground, at the LiDAR ground points around
                        the building (median per building).
      base error        model ground_low under the footprint - LiDAR ground_low.
  M3  roof elev. error  model roof elevation - LiDAR roof top elevation (absolute):
                        what the viewer sees. Decomposes exactly as
                        M3 = M1 + base error + platform lift,
                        where platform lift = model roof - height - model ground_low
                        (extrude.py puts roofs at the HIGHEST ground under the footprint).
  M3_lowground          M3 if the roof were placed at ground_low + height instead
                        (= M3 - lift): lets you test the placement convention.

LiDAR reference per building
  ground   robust plane fitted to ground-class (2) points in a ring 0.5-3 m outside
           the footprint, evaluated along the footprint outline -> ground_low / ground_high
  roof     points inside the footprint eroded by 1 m: building class (6) when the
           collection classifies buildings, otherwise non-ground, non-vegetation points
           more than 2 m above ground. roof top (default --roof-method ransac) = highest
           roof plane found by sequential RANSAC; points on no plane (trees, antennas,
           rooftop equipment, noise) are ignored. --roof-method percentile: 95th percentile.
           The 95th percentile and the median are always reported for comparison.
  coverage share of 1 m cells of the eroded footprint that contain a roof point;
           < 0.5 -> "not_in_lidar" (built after the flight, demolished, or occluded)

Run (from the repo root, in the project environment)
  poetry run pip install "laspy[lazrs]"            # once: reads the LiDAR tiles
  poetry run python scripts/eval/eval_geometry_lidar.py --latest     # newest web app job
  poetry run python scripts/eval/eval_geometry_lidar.py output/jobs/<id>
  poetry run python scripts/eval/eval_geometry_lidar.py model.glb --bbox W S E N
                                      # a .glb from src/main.py: give its --bbox
  poetry run python scripts/eval/eval_geometry_lidar.py --from-dir output/jobs/<id>/geometry_eval
                                      # evaluate again without reading the .glb or OSM
  ... --laz tile1.laz tile2.laz       local LiDAR instead of USGS
  ... --ept https://.../ept.json      a specific USGS collection
  ... --height-baseline 0.34          compare the height share with an earlier run

Outputs (in geometry_eval/ next to the .glb, e.g. output/jobs/<id>/geometry_eval/)
  summary.md (start here), summary.json, per_building.csv,
  per_building_errors.geojson (open in QGIS), 01_scorecard.png,
  02_height_error_by_source.png, 03_height_scatter.png, 04_placement_vs_slope.png,
  05_height_error_map.png, and what it read from the mesh and OSM:
  buildings_footprints_utm.geojson, terrain_grid.npz, osm_buildings.json
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import requests
    import shapely
    from pyproj import CRS, Transformer
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import cKDTree
    from shapely.geometry import box, mapping, shape
except ModuleNotFoundError as _exc:
    sys.exit(
        f"\nMissing package: {_exc.name}\nInstall into {sys.executable} with:\n"
        f"  {sys.executable} -m pip install numpy scipy shapely pyproj requests "
        f'"laspy[lazrs]" matplotlib\n'
    )

USGS_RESOURCES = (
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/"
    "boundaries/resources.geojson"
)
GROUND, BUILDING = 2, 6
VEGETATION = (3, 4, 5)
KEEP_CLASSES = (1, 2, 3, 4, 5, 6)  # drop noise (7, 18), water (9), bridge (17), ...
EMBED_M = 0.1  # mesh_builder/extrude.py EMBED_M (read from there)
METRICS = [
    "M1_height_err",
    "M2_terrain_err",
    "M2_base_err",
    "M3_roof_err",
    "M3_roof_err_lowground",
]
CONFIDENCE = {"height": "high", "levels": "medium", "default": "low"}
GROUPS = [
    ("high", "height", "high confidence (height tag)"),
    ("medium", "levels", "medium confidence (levels tag)"),
    ("low", "default", "low confidence (no tag, default)"),
]


# ----------------------------------------------------------------------------
# Inputs: the pipeline's buildings and terrain
# ----------------------------------------------------------------------------
def load_footprints(path):
    d = json.loads(Path(path).read_text())
    name = (d.get("crs") or {}).get("properties", {}).get("name", "")
    epsg = name.split(":")[-1] if name else None
    if not epsg or not epsg.isdigit():
        sys.exit(f"[input] {path} has no EPSG code in its 'crs' member")
    feats = [(shape(f["geometry"]), f["properties"]) for f in d["features"]]
    print(f"[input] {len(feats)} footprints, EPSG:{epsg}")
    return feats, f"EPSG:{epsg}"


def load_terrain(path):
    if not Path(path).exists():
        print(f"[input] no {Path(path).name}: terrain metrics (M2) skipped")
        return None
    g = np.load(path, allow_pickle=True)
    x, y, z = g["x"], g["y"], g["z"]
    interp = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=np.nan)
    print(
        f"[input] terrain grid {len(x)}x{len(y)} ({str(g['source'])}, "
        f"{float(g['spacing']):g} m), {np.nanmin(z):.1f}-{np.nanmax(z):.1f} m"
    )
    return lambda xs, ys: interp(np.column_stack([ys, xs]))


def outline_parts(element, to_xy, way_ids=()):
    """
    Footprint polygons (UTM) of an OSM building from an Overpass "out geom" element:
      way       its outline
      relation  a multipolygon (e.g. Duderstadt Center): the outer members' rings minus
                the inner ones (courtyards). A ring may be split over several member
                ways, so the members are joined first. Outer members that are also
                fetched as building ways themselves (`way_ids`) are skipped, so a
                building is not counted twice.
    Returns a list of Polygons (one per separate part); empty when there is no usable
    outline.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    if element.get("type") == "way":
        geom = element.get("geometry") or []
        if len(geom) < 3:
            return []
        shape_ = Polygon(to_xy(geom))
    elif element.get("type") == "relation":
        lines = {"outer": [], "inner": []}
        for m in element.get("members", []):
            role = "inner" if m.get("role") == "inner" else "outer"
            if m.get("type") != "way" or len(m.get("geometry") or []) < 2:
                continue
            if role == "outer" and m.get("ref") in way_ids:
                continue
            lines[role].append(LineString(to_xy(m["geometry"])))
        if not lines["outer"]:
            return []
        outer = unary_union(list(polygonize(unary_union(lines["outer"]))))
        inner = (
            unary_union(list(polygonize(unary_union(lines["inner"]))))
            if lines["inner"]
            else None
        )
        shape_ = outer.difference(inner) if inner is not None else outer
    else:
        return []
    if not shape_.is_valid:
        shape_ = shape_.buffer(0)
    parts = list(getattr(shape_, "geoms", [shape_]))
    return [g for g in parts if g.geom_type == "Polygon" and g.area > 1.0]


def _glb_parts(path):
    """{"buildings": Trimesh, "terrain": Trimesh} from the .glb, node transforms applied.
    The ground is the mesh named "terrain" (export_mesh); every other mesh is buildings,
    joined into one (a textured model can load as several pieces, e.g. geometry_0, ...)."""
    import trimesh

    scene = trimesh.load(path, force="scene")
    pieces = {"buildings": [], "terrain": []}
    names = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname]
        if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
            continue  # points, lines or empty meshes: not part of the model's geometry
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        m.apply_transform(T)
        kind = "terrain" if "terrain" in f"{node} {gname}".lower() else "buildings"
        pieces[kind].append(m)
        names.append(f"{gname} ({kind}, {len(m.faces):,} faces)")
    print(f"[input] {Path(path).name}: " + ", ".join(names))
    if not pieces["buildings"]:
        sys.exit(f"[input] {path}: no building mesh")
    return {k: trimesh.util.concatenate(v) for k, v in pieces.items() if v}


def _terrain_from_mesh(xyz, z_datum):
    """The terrain stage's grid (Terrain) back from the .glb's ground mesh vertices."""
    from terrain import Terrain

    xyz = np.unique(np.round(xyz, 3), axis=0)  # one vertex per grid point
    xs = np.unique(np.round(xyz[:, 0], 2))
    ys = np.unique(np.round(xyz[:, 1], 2))
    if len(xs) * len(ys) != len(xyz):
        sys.exit("[input] the .glb's terrain is not a regular grid")
    order = np.lexsort((xyz[:, 0], xyz[:, 1]))  # row by row, as Terrain.to_mesh
    z = xyz[order, 2].reshape(len(ys), len(xs)) - z_datum
    return Terrain(xs=xs, ys=ys, z=z, z_datum=z_datum)


def find_model(path=None, latest=False, jobs_dir="output/jobs"):
    """The model .glb: `path` is a .glb or a folder holding model.glb (a web app job,
    output/jobs/<id>/); with `latest`, the newest output/jobs/*/model.glb."""
    if latest:
        found = sorted(
            Path(jobs_dir).glob("*/model.glb"), key=lambda q: q.stat().st_mtime
        )
        if not found:
            sys.exit(f"[input] no {jobs_dir}/*/model.glb: run the pipeline first")
        return found[-1]
    q = Path(path)
    q = q / "model.glb" if q.is_dir() else q
    if not q.exists():
        sys.exit(f"[input] {q} not found")
    return q


def build_from_glb(glb, out, bbox=None, osm_cache=True):
    """
    The buildings and terrain of a pipeline .glb, in UTM metres and absolute elevation,
    one footprint per OSM building. The geometry comes from the mesh; OSM (fetched again
    for the model's bounding box, as the fetch stage does) gives each building its
    outline, id, name and height tags:
      roof  = highest horizontal face of the mesh inside the footprint
      base  = lowest horizontal face inside the footprint
      height (above the highest ground, as extrude.py) = roof - highest ground under
             the footprint (model terrain); without terrain: roof (ground = 0)
    An OSM building with nothing in the mesh is "not_in_mesh" (e.g. one mapped as a
    relation, which extrude.py does not build yet).

    The bounding box comes from the job.json next to the .glb (web app jobs) or `bbox`.
    export_mesh moved the model from UTM by the box centre and then by the centre of its
    bounds, and with terrain measured heights from the terrain's lowest point; the .glb
    keeps none of these, so they are worked out again: with terrain, from the terrain
    stage's grid for the same box (same elevation tiles, same grid); without, from the
    OSM outlines the mesh was extruded from.

    Writes buildings_footprints_utm.geojson, terrain_grid.npz and osm_buildings.json to
    `out` (read back by --from-dir) and returns (feats, utm_crs, terrain_fn, embed_m).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from shapely.ops import unary_union

    from common.constants import BoundingBox
    from common.MeshUtils import (
        _get_utm_transformer,
        building_heights,
        get_base_and_height,
        get_corners,
    )
    from common.providers import OSM_MAP_FEATURES, OSMClient
    from mesh_builder.extrude import EMBED_M as embed_m
    from mesh_builder.extrude import ground_under

    glb = Path(glb)
    job = glb.parent / "job.json"
    if bbox is None and job.exists():
        b = json.loads(job.read_text())["bbox"]
        bbox = (b["west"], b["south"], b["east"], b["north"])
    if bbox is None:
        sys.exit(
            f"[input] no job.json next to {glb.name}: pass the --bbox the model was "
            "built with (WEST SOUTH EAST NORTH)"
        )
    box_ = BoundingBox.from_list(list(bbox))
    lon0, lat0 = (box_.min_lon + box_.max_lon) / 2, (box_.min_lat + box_.max_lat) / 2
    to_utm_tf = _get_utm_transformer(lon0, lat0)
    epsg = to_utm_tf.target_crs.to_epsg()
    ox, oy = to_utm_tf.transform(lon0, lat0)
    parts = _glb_parts(glb)

    cache = out / "osm_buildings.json"
    if osm_cache and cache.exists():
        data = json.loads(cache.read_text())
        print(f"[input] OSM buildings from {cache} (delete it to fetch again)")
    else:
        print(f"[input] OSM buildings for {box_.to_str()} (outlines and tags)")
        data = OSMClient().fetch(box_, [OSM_MAP_FEATURES.BUILDING])
        cache.write_text(json.dumps(data))
    elements = data.get("elements", [])
    way_ids = {e.get("id") for e in elements if e.get("type") == "way"}

    def to_xy(points):  # the pipeline's projection (MeshUtils.get_corners)
        return get_corners({"geometry": points}, box_)

    outlines = [
        (el, poly) for el in elements for poly in outline_parts(el, to_xy, way_ids)
    ]

    # where export_mesh put the model: glb (X, Y, Z) = (x - ox - cx, z - cz, -(y - oy - cy))
    if "terrain" in parts:
        from terrain import build_terrain

        grid = build_terrain(box_)  # the terrain stage's grid for this box
        V = parts["terrain"].vertices
        cx = grid.xs[0] - ox - V[:, 0].min()
        cy = grid.ys[0] - oy + V[:, 2].max()
        cz = -V[:, 1].min()  # the terrain's lowest point is its datum
        z_datum = grid.z_datum
    else:
        built = [p for el, p in outlines if el.get("type") == "way"]
        if not built:
            sys.exit("[input] no OSM buildings for this box")
        x0, y0, x1, y1 = unary_union(built).bounds
        V = parts["buildings"].vertices
        cx = (x0 + x1) / 2 - ox - (V[:, 0].min() + V[:, 0].max()) / 2
        cy = (y0 + y1) / 2 - oy + (V[:, 2].min() + V[:, 2].max()) / 2
        cz = -V[:, 1].min()  # buildings stand on flat ground at 0
        z_datum = 0.0

    def to_utm(v):  # back from the .glb to UTM (x, y) and elevation
        v = np.asarray(v, float)
        return np.c_[v[:, 0] + cx + ox, -v[:, 2] + cy + oy, v[:, 1] + cz + z_datum]

    bld = parts["buildings"]
    tri = to_utm(bld.vertices)[bld.faces]  # (n, 3, 3)
    nz = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(nz, axis=1)
    nz = np.divide(nz[:, 2], area2, out=np.zeros(len(tri)), where=area2 > 0)
    ctr = tri.mean(axis=1)
    # horizontal faces: roofs and floors. (Their up/down direction is not used: export
    # flips z, which turns the faces' winding inside out.)
    flat = np.abs(nz) > 0.9
    terrain = None
    if "terrain" in parts:
        terrain = _terrain_from_mesh(to_utm(parts["terrain"].vertices), z_datum)
        diff = np.abs(terrain.z - grid.ground(*np.meshgrid(terrain.xs, terrain.ys)))
        if not np.isfinite(diff).all() or diff.max() > 0.5:
            print(
                "[input] WARNING: the .glb's terrain differs from the elevation tiles by up "
                f"to {np.nanmax(diff):.2f} m (tiles updated since the model was built?)"
            )
    print(
        f"[input] {glb}: {len(tri):,} building faces, "
        + (
            f"terrain grid {len(terrain.xs)}x{len(terrain.ys)}"
            if terrain
            else "flat ground"
        )
    )

    feats, used = [], np.zeros(len(tri), bool)
    for el, poly in outlines:
        tags = el.get("tags", {})
        base, top_tag = get_base_and_height(el)
        props = {
            "osm_id": f"{el.get('type')}/{el.get('id')}",
            "name": tags.get("name", ""),
            "min_height": round(base, 3),
            "height_source": building_heights(tags)[2],
            "osm_rule_height": round(top_tag, 3),
        }
        inner = poly.buffer(-0.05) if poly.area > 1.0 else poly
        x0, y0, x1, y1 = poly.bounds
        near = (
            (ctr[:, 0] >= x0)
            & (ctr[:, 0] <= x1)
            & (ctr[:, 1] >= y0)
            & (ctr[:, 1] <= y1)
        )
        idx = np.flatnonzero(near & flat)
        inside = idx[shapely.contains_xy(inner, ctr[idx, 0], ctr[idx, 1])]
        props["in_mesh"] = bool(len(inside))
        if not len(inside):
            feats.append((poly, props))
            continue
        used[inside] = True
        z = tri[inside][:, :, 2]
        roof, floor = float(z.max()), float(z.min())
        if terrain is not None:
            _, g_high = ground_under(np.asarray(poly.exterior.coords)[:-1], terrain)
            props["height"] = round(roof - (g_high + z_datum), 3)
            props["roof_elev_m"] = round(roof, 3)
            props["base_elev_m"] = round(floor, 3)
        else:
            props["height"] = round(roof, 3)
        feats.append((poly, props))

    n_in = sum(p["in_mesh"] for _, p in feats)
    missing = sorted({p["name"] or p["osm_id"] for _, p in feats if not p["in_mesh"]})
    stray = flat & ~used
    print(f"[input] {len(feats)} OSM footprints, {n_in} found in the mesh, EPSG:{epsg}")
    if missing:
        print(
            f"[input] in OSM but not in the mesh: {len(missing)} "
            f"(e.g. {', '.join(missing[:5])})"
        )
    if stray.any():
        share = stray.sum() / max(flat.sum(), 1)
        print(
            f"[input] {'WARNING: ' if share > 0.1 else ''}{share * 100:.0f}% of the mesh's "
            "roof and floor faces lie in no OSM footprint "
            + (
                "(wrong bounding box?)"
                if share > 0.1
                else "(OSM changed since the model was built?)"
            )
        )
    if not n_in:
        sys.exit(
            "[input] no OSM building matches the mesh: wrong .glb or bounding box?"
        )

    (out / "buildings_footprints_utm.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
                },
                "features": [
                    {"type": "Feature", "geometry": mapping(g), "properties": p}
                    for g, p in feats
                ],
            }
        )
    )
    terrain_fn = None
    if terrain is not None:
        np.savez_compressed(
            out / "terrain_grid.npz",
            x=terrain.xs,
            y=terrain.ys,
            z=terrain.z + terrain.z_datum,
            source=f"{glb.name} terrain",
            spacing=terrain.spacing,
        )
        terrain_fn = lambda xs, ys: terrain.ground(xs, ys) + terrain.z_datum  # noqa: E731
    return feats, f"EPSG:{epsg}", terrain_fn, float(embed_m)


# ----------------------------------------------------------------------------
# LiDAR: USGS 3DEP Entwine Point Tiles (EPT) on AWS, or local LAS/LAZ
# ----------------------------------------------------------------------------
def _laspy():
    try:
        import laspy

        return laspy
    except ModuleNotFoundError:
        sys.exit('Missing package laspy. Install: pip install "laspy[lazrs]"')


def _session():
    s = requests.Session()
    a = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504)
        )
    )
    s.mount("https://", a)
    s.mount("http://", a)
    return s


def find_usgs_ept(lon, lat, session):
    """All USGS EPT collections that contain (lon, lat), newest year first."""
    r = session.get(USGS_RESOURCES, timeout=60)
    r.raise_for_status()
    pt = shapely.Point(lon, lat)
    hits = [
        f["properties"]
        for f in r.json()["features"]
        if shape(f["geometry"]).contains(pt)
    ]

    def year(p):
        ys = [
            int(t)
            for t in p["name"].replace("-", "_").split("_")
            if t.isdigit() and len(t) == 4
        ]
        return max(ys) if ys else 0

    return sorted(hits, key=year, reverse=True)


def _point_arrays(las, xmin, ymin, xmax, ymax):
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    c = np.asarray(las.classification).astype(np.uint8)
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    try:
        keep &= ~np.asarray(las.withheld, dtype=bool)
    except Exception:  # noqa: BLE001 - flag absent in some formats
        pass
    return x[keep], y[keep], z[keep], c[keep]


def read_ept(url, aoi_utm, utm_crs, cache_dir, workers=8):
    """Read every EPT node that overlaps the AOI (all depths = full resolution)."""
    laspy = _laspy()
    s = _session()
    base = url.rsplit("/", 1)[0]
    info = s.get(url, timeout=60)
    info.raise_for_status()
    info = info.json()
    srs = info.get("srs", {})
    if srs.get("authority") and srs.get("horizontal"):
        ept_crs = CRS.from_user_input(f"{srs['authority']}:{srs['horizontal']}")
    else:
        ept_crs = CRS.from_wkt(srs["wkt"])
    print(
        f"[lidar] EPT {base.rsplit('/', 1)[-1]}: {info.get('points', 0):,} points, "
        f"horizontal {ept_crs.to_string()}, vertical {srs.get('vertical', '?')}"
    )

    to_ept = Transformer.from_crs(utm_crs, ept_crs, always_xy=True)
    ax0, ay0, ax1, ay1 = aoi_utm
    ex, ey = to_ept.transform([ax0, ax1, ax1, ax0], [ay0, ay0, ay1, ay1])
    qx0, qy0, qx1, qy1 = min(ex), min(ey), max(ex), max(ey)
    bx0, by0, _, bx1, _, _ = info["bounds"]
    span = bx1 - bx0

    def overlaps(key):
        d, X, Y, _ = (int(v) for v in key.split("-"))
        step = span / 2**d
        x0, y0 = bx0 + X * step, by0 + Y * step
        return not (x0 > qx1 or x0 + step < qx0 or y0 > qy1 or y0 + step < qy0)

    nodes, todo = [], ["0-0-0-0"]
    while todo:  # walk hierarchy files
        hkey = todo.pop()
        h = s.get(f"{base}/ept-hierarchy/{hkey}.json", timeout=60)
        h.raise_for_status()
        for key, count in h.json().items():
            if not overlaps(key):
                continue
            if count == -1:
                if key != hkey:
                    todo.append(key)
            elif count > 0:
                nodes.append(key)
    nodes = sorted(set(nodes))
    print(
        f"[lidar] {len(nodes)} EPT nodes overlap the AOI; downloading (cached in {cache_dir})"
    )

    cache = Path(cache_dir) / base.rsplit("/", 1)[-1]
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(key):
        f = cache / f"{key}.laz"
        if not f.exists():
            r = s.get(f"{base}/ept-data/{key}.laz", timeout=180)
            r.raise_for_status()
            f.write_bytes(r.content)
        las = laspy.read(io.BytesIO(f.read_bytes()))
        return _point_arrays(las, qx0, qy0, qx1, qy1)

    parts = []
    with cf.ThreadPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(fetch, nodes), 1):
            parts.append(part)
            if i % 25 == 0 or i == len(nodes):
                print(f"[lidar]   {i}/{len(nodes)} nodes")
    x, y, z, c = (np.concatenate([p[k] for p in parts]) for k in range(4))
    ux, uy = Transformer.from_crs(ept_crs, utm_crs, always_xy=True).transform(x, y)
    return (
        np.asarray(ux),
        np.asarray(uy),
        z,
        c,
        {"source": "ept", "url": url, "vertical": srs.get("vertical")},
    )


def read_laz_files(paths, aoi_utm, utm_crs, laz_crs=None):
    laspy = _laspy()
    parts = []
    for p in paths:
        with laspy.open(p) as fh:
            crs = CRS.from_user_input(laz_crs) if laz_crs else fh.header.parse_crs()
            if crs is None:
                sys.exit(
                    f"[lidar] {p} has no CRS in its header; pass --laz-crs EPSG:xxxx"
                )
            hcrs = crs.sub_crs_list[0] if crs.is_compound else crs
            to_f = Transformer.from_crs(utm_crs, hcrs, always_xy=True)
            ax0, ay0, ax1, ay1 = aoi_utm
            fx, fy = to_f.transform([ax0, ax1, ax1, ax0], [ay0, ay0, ay1, ay1])
            back = Transformer.from_crs(hcrs, utm_crs, always_xy=True)
            n = 0
            for chunk in fh.chunk_iterator(5_000_000):
                x, y, z, c = _point_arrays(chunk, min(fx), min(fy), max(fx), max(fy))
                ux, uy = back.transform(x, y)
                parts.append((np.asarray(ux), np.asarray(uy), z, c))
                n += len(x)
            unit = hcrs.axis_info[0].unit_name if hcrs.axis_info else "?"
            print(
                f"[lidar] {Path(p).name}: {n:,} points in AOI ({hcrs.to_string()}, {unit})"
            )
    if not parts:
        sys.exit("[lidar] no points read")
    x, y, z, c = (np.concatenate([p[k] for p in parts]) for k in range(4))
    return x, y, z, c, {"source": "laz", "files": [str(p) for p in paths]}


def load_lidar(args, aoi_utm, utm_crs, out):
    """Points (UTM x, y, z, class) for the AOI, cached as one .npz per source + AOI."""
    tag = json.dumps(
        [args.laz, args.ept, [round(v, 1) for v in aoi_utm]], sort_keys=True
    )
    cache_npz = out / f"lidar_points_{hashlib.md5(tag.encode()).hexdigest()[:10]}.npz"
    if cache_npz.exists() and not args.refresh_lidar:
        d = np.load(cache_npz, allow_pickle=True)
        print(f"[lidar] using cached points {cache_npz.name} ({len(d['x']):,} points)")
        return d["x"], d["y"], d["z"], d["c"], json.loads(str(d["meta"]))

    if args.laz:
        x, y, z, c, meta = read_laz_files(args.laz, aoi_utm, utm_crs, args.laz_crs)
    else:
        url = args.ept
        if not url:
            to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
            lon, lat = to_ll.transform(
                (aoi_utm[0] + aoi_utm[2]) / 2, (aoi_utm[1] + aoi_utm[3]) / 2
            )
            hits = find_usgs_ept(lon, lat, _session())
            if not hits:
                sys.exit("[lidar] no USGS EPT collection covers this AOI; use --laz")
            print(
                "[lidar] USGS collections here: " + ", ".join(h["name"] for h in hits)
            )
            url = hits[0]["url"]
        x, y, z, c, meta = read_ept(url, aoi_utm, utm_crs, args.cache_dir, args.workers)

    counts = {int(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))}
    meta["class_counts"] = counts
    keep = np.isin(c, KEEP_CLASSES)
    x, y, z, c = x[keep], y[keep], z[keep], c[keep]
    np.savez_compressed(cache_npz, x=x, y=y, z=z, c=c, meta=json.dumps(meta))
    return x, y, z, c, meta


# ----------------------------------------------------------------------------
# Per-building reference values
# ----------------------------------------------------------------------------
def robust_plane(px, py, pz, iters=3):
    """z = a + b*(x-x0) + c*(y-y0), refit after dropping |res| > 2.5 NMAD."""
    x0, y0 = px.mean(), py.mean()
    keep = np.ones(len(pz), bool)
    coef, rms = np.array([np.median(pz), 0.0, 0.0]), float("nan")
    for _ in range(iters):
        if keep.sum() < 10:
            break
        A = np.column_stack([np.ones(keep.sum()), px[keep] - x0, py[keep] - y0])
        coef, *_ = np.linalg.lstsq(A, pz[keep], rcond=None)
        res = pz - (coef[0] + coef[1] * (px - x0) + coef[2] * (py - y0))
        nmad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 0.02
        keep = np.abs(res) <= 2.5 * nmad
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    grade = float(math.hypot(coef[1], coef[2]) * 100)  # ground slope in percent
    return (
        (lambda qx, qy: coef[0] + coef[1] * (qx - x0) + coef[2] * (qy - y0)),
        rms,
        grade,
    )


def outline_points(poly, step=1.0):
    pts = [np.asarray(poly.exterior.segmentize(step).coords)[:, :2]]
    return np.vstack(pts)


def ransac_roof(
    px,
    py,
    pz,
    thresh=0.3,
    min_share=0.10,
    min_pts=20,
    max_planes=4,
    iters=300,
    max_tilt_deg=60.0,
    seed=0,
):
    """
    Roof surfaces by sequential RANSAC: find the plane with the most points within
    `thresh` m, refit it by least squares, remove its points, repeat (up to max_planes).
    A plane must hold >= min_share of the roof points (and >= min_pts) and be tilted
    < max_tilt_deg; points on no plane (trees, antennas, rooftop equipment, noise) are
    ignored. Returns (top, main, n_planes, outlier_share) or None when no plane is found:
      top  = highest roof surface: 99th percentile of the planes' fitted elevations at their
             own points (fitted values lie on the plane, so noise and outliers don't count)
      main = the largest plane's median fitted elevation (the main roof surface)
    """
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
        inl = np.abs(res) < thresh  # final members, by vertical residual
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


def cells_covered(px, py, cell=1.0):
    if len(px) == 0:
        return 0
    return len(
        np.unique(
            np.floor(px / cell).astype(np.int64) * 1_000_003
            + np.floor(py / cell).astype(np.int64)
        )
    )


def prepare_lidar(pts, min_above=2.0, verbose=True):
    """Split LiDAR points into ground and roof candidates and index them (KD-trees)."""
    x, y, z, c = pts
    has_bldg_class = (c == BUILDING).sum() > 0.005 * len(c)
    has_veg_class = np.isin(c, VEGETATION).sum() > 0.005 * len(c)
    gsel = c == GROUND
    rsel = (c == BUILDING) if has_bldg_class else np.isin(c, (1, BUILDING))
    ctx = {
        "gx": x[gsel],
        "gy": y[gsel],
        "gz": z[gsel],
        "rx": x[rsel],
        "ry": y[rsel],
        "rz": z[rsel],
        "box": box(x.min(), y.min(), x.max(), y.max()),
        "building_class": bool(has_bldg_class),
        "vegetation_class": bool(has_veg_class),
    }
    ctx["gtree"] = cKDTree(np.column_stack([ctx["gx"], ctx["gy"]]))
    ctx["rtree"] = cKDTree(np.column_stack([ctx["rx"], ctx["ry"]]))
    if verbose:
        print(
            f"[lidar] {gsel.sum():,} ground points, {rsel.sum():,} roof candidates "
            f"({'building class 6' if has_bldg_class else 'no building class: non-ground > ' + str(min_above) + ' m'}"
            f"{'' if has_bldg_class or has_veg_class else ', no vegetation class (trees may count as roof)'})"
        )
    return ctx


def measure_building(poly, ctx, a, min_height=0.0):
    """
    LiDAR reference values for one footprint (UTM polygon). Returns a dict with
    `status` ("ok" or why not) and, as far as it got: ground low/high (plane fit to the
    ground ring, read along the outline), grade, roof top (a.roof_pct), roof median,
    coverage, LiDAR height (roof top - lowest ground). `_ring` holds the ring's ground
    points (x, y, z) for terrain comparisons; drop it before writing files.
    `a` needs: ring_in, ring_out, erode, min_roof_area, min_ground_pts, min_above,
    min_coverage, roof_pct.
    """
    m = {"status": "ok"}
    if not ctx["box"].contains(poly.buffer(a.ring_out)):
        return {"status": "outside_lidar"}
    if (min_height or 0) > 0:
        return {
            "status": "raised_part"
        }  # bridge / overhang: not a ground-standing prism
    inner = poly.buffer(-a.erode)
    if inner.is_empty or inner.area < a.min_roof_area:
        return {"status": "too_small"}
    gx, gy, gz, rx, ry, rz = (ctx[k] for k in ("gx", "gy", "gz", "rx", "ry", "rz"))
    minx, miny, maxx, maxy = poly.bounds
    ctr = ((minx + maxx) / 2, (miny + maxy) / 2)
    rad = math.hypot(maxx - minx, maxy - miny) / 2 + a.ring_out + 0.5

    # sorted: KD-tree query order depends on the tree, and RANSAC samples by position, so
    # unsorted indices could give different roofs for the same building in different runs
    gi = np.sort(np.asarray(ctx["gtree"].query_ball_point(ctr, rad), dtype=np.int64))
    ring = poly.buffer(a.ring_out).difference(poly.buffer(a.ring_in))
    if len(gi):
        gi = gi[shapely.contains_xy(ring, gx[gi], gy[gi])]
    m["n_ground_pts"] = int(len(gi))
    if len(gi) < a.min_ground_pts:
        m["status"] = "no_ground"
        return m
    plane, ground_rms, grade = robust_plane(gx[gi], gy[gi], gz[gi])
    ol = outline_points(poly)
    g_outline = plane(ol[:, 0], ol[:, 1])
    g_low, g_high = float(g_outline.min()), float(g_outline.max())
    m["_ring"] = (gx[gi], gy[gi], gz[gi])

    ri = np.sort(np.asarray(ctx["rtree"].query_ball_point(ctr, rad), dtype=np.int64))
    if len(ri):
        ri = ri[shapely.contains_xy(inner, rx[ri], ry[ri])]
        ri = ri[rz[ri] > g_low + a.min_above]
    # order by coordinates: RANSAC samples by position, so the same points read in a
    # different order (another AOI, another run's cache) must give the same roof
    ri = ri[np.lexsort((rz[ri], ry[ri], rx[ri]))]
    coverage = min(1.0, cells_covered(rx[ri], ry[ri]) / max(inner.area, 1.0))
    m.update(
        n_roof_pts=int(len(ri)),
        roof_coverage=round(coverage, 3),
        lidar_ground_low=round(g_low, 2),
        lidar_ground_high=round(g_high, 2),
        lidar_ground_rms=round(ground_rms, 3),
        ground_drop_m=round(g_high - g_low, 2),
        site_grade_pct=round(grade, 2),
    )
    if coverage < a.min_coverage or len(ri) < 5:
        m["status"] = "not_in_lidar"
        return m
    roof_pct = float(np.percentile(rz[ri], a.roof_pct))
    roof_med = float(np.percentile(rz[ri], 50))
    m.update(lidar_roof_pct=round(roof_pct, 2), lidar_roof_median=round(roof_med, 2))
    rs = ransac_roof(
        rx[ri],
        ry[ri],
        rz[ri],
        thresh=getattr(a, "ransac_thresh", 0.3),
        min_share=getattr(a, "ransac_min_share", 0.10),
    )
    if rs:
        m.update(
            ransac_roof_top=round(rs[0], 2),
            ransac_roof_main=round(rs[1], 2),
            ransac_planes=rs[2],
            ransac_outlier_share=round(rs[3], 3),
        )
    method = getattr(a, "roof_method", "percentile")
    max_out = getattr(a, "ransac_max_outliers", 0.5)
    if method == "ransac" and rs and rs[3] <= max_out:
        roof_top, m["roof_method_used"] = rs[0], "ransac"
    else:
        # no plane, or most points on no plane (towers, spires, open structures): a roof
        # plane would describe only a small part of the building, so use the percentile
        roof_top = roof_pct
        m["roof_method_used"] = (
            "percentile"
            if method != "ransac"
            else "percentile (no roof plane)"
            if not rs
            else f"percentile (roof not planar: {rs[3] * 100:.0f}% of points on no plane)"
        )
    m.update(
        lidar_roof_top=round(roof_top, 2),
        lidar_height=round(roof_top - g_low, 2),
        lidar_height_median=round(roof_med - g_low, 2),
    )
    if rs:
        m["lidar_height_main"] = round(rs[1] - g_low, 2)
    return m


def evaluate_buildings(feats, terrain, pts, a):
    ctx = prepare_lidar(pts, a.min_above)
    rows = []
    for poly, p in feats:
        row = {
            k: p.get(k)
            for k in (
                "osm_id",
                "name",
                "height",
                "min_height",
                "height_source",
                "osm_rule_height",
                "base_elev_m",
                "roof_elev_m",
            )
        }
        row["in_mesh"] = p.get("in_mesh", True)
        row["footprint_m2"] = round(poly.area, 1)
        row["height_confidence"] = CONFIDENCE.get(p.get("height_source"), "low")
        if row["in_mesh"] is False:  # in OSM, but the model has no such building
            row["status"] = "not_in_mesh"
            rows.append(row)
            continue
        m = measure_building(poly, ctx, a, p.get("min_height") or 0)
        ring = m.pop("_ring", None)
        row.update(m)
        rows.append(row)
        if row["status"] != "ok":
            continue
        roof_top, g_low, h_lidar = (
            row["lidar_roof_top"],
            row["lidar_ground_low"],
            row["lidar_height"],
        )

        h = float(p["height"])
        row["M1_height_err"] = round(h - h_lidar, 3)
        row["height_within_tol"] = bool(abs(h - h_lidar) <= a.height_tol)
        row["M1_height_err_vs_median"] = round(h - row["lidar_height_median"], 3)

        if p.get("roof_elev_m") is not None:
            roof_model = float(p["roof_elev_m"])
            g_low_model = float(p["base_elev_m"]) + a.embed
            lift = roof_model - h - g_low_model
            row["platform_lift"] = round(lift, 3)
            row["M2_base_err"] = round(g_low_model - g_low, 3)
            row["M3_roof_err"] = round(roof_model - roof_top, 3)
            row["M3_roof_err_lowground"] = round(roof_model - lift - roof_top, 3)
        if terrain is not None and ring is not None:
            tz = terrain(ring[0], ring[1])
            ok = np.isfinite(tz)
            if ok.any():
                row["M2_terrain_err"] = round(float(np.median(tz[ok] - ring[2][ok])), 3)
    return rows, {
        "building_class": ctx["building_class"],
        "vegetation_class": ctx["vegetation_class"],
    }


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def describe(e, n_boot=2000, seed=0):
    e = np.asarray([v for v in e if v is not None and np.isfinite(v)], float)
    if len(e) == 0:
        return {"n": 0}

    def stats(v):
        med = np.median(v)
        return {
            "bias_mean": v.mean(),
            "median": med,
            "rmse": np.sqrt(np.mean(v**2)),
            "mae": np.mean(np.abs(v)),
            "nmad": 1.4826 * np.median(np.abs(v - med)),
            "within_1m": np.mean(np.abs(v) <= 1.0),
            "within_3m": np.mean(np.abs(v) <= 3.0),
        }

    out = {"n": int(len(e)), **{k: float(v) for k, v in stats(e).items()}}
    if len(e) >= 5:
        rng = np.random.default_rng(seed)
        boots = [stats(e[rng.integers(0, len(e), len(e))]) for _ in range(n_boot)]
        for k in ("median", "rmse", "within_1m", "within_3m"):
            lo, hi = np.percentile([b[k] for b in boots], [2.5, 97.5])
            out[f"{k}_ci95"] = [float(lo), float(hi)]
    return out


def share_ci(vals, tol, n_boot=2000, seed=0):
    """Share of |v| <= tol with a bootstrap 95% CI -> (share, lo, hi, hits, n) or None."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    hit = np.abs(v) <= tol
    if len(v) >= 5:
        rng = np.random.default_rng(seed)
        boots = [hit[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = hit.mean()
    return float(hit.mean()), float(lo), float(hi), int(hit.sum()), int(len(v))


def summarize(rows, grade_cut):
    ok = [r for r in rows if r["status"] == "ok"]
    groups = {"all": ok}
    for conf, _, label in GROUPS:
        groups[label] = [r for r in ok if r["height_confidence"] == conf]
    groups[f"flat site (<{grade_cut:g}% grade)"] = [
        r for r in ok if r["site_grade_pct"] < grade_cut
    ]
    groups[f"sloped site (>={grade_cut:g}% grade)"] = [
        r for r in ok if r["site_grade_pct"] >= grade_cut
    ]
    res = {}
    for g, rs in groups.items():
        res[g] = {
            m: describe([r.get(m) for r in rs])
            for m in METRICS + ["M1_height_err_vs_median"]
        }
        res[g]["n_buildings"] = len(rs)
    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    return res, status


def _fmt(s, k, pct=False):
    if s.get("n", 0) == 0 or k not in s:
        return "–"
    v = s[k]
    txt = (
        f"{v * 100:.0f}%"
        if pct
        else f"{v:+.2f}"
        if k in ("bias_mean", "median")
        else f"{v:.2f}"
    )
    ci = s.get(f"{k}_ci95")
    if ci:
        txt += (
            f" [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]"
            if pct
            else f" [{ci[0]:.2f}, {ci[1]:.2f}]"
        )
    return txt


def _example(rows, group):
    """The building whose height error is the median of the group: a worked calculation."""
    ok = [
        r
        for r in rows
        if r["status"] == "ok" and (group == "all" or r["height_confidence"] == group)
    ]
    if not ok:
        return []
    ok.sort(key=lambda r: r["M1_height_err"])
    r = ok[len(ok) // 2]
    L = [
        f"Building `{r['osm_id']}`{(' (' + r['name'] + ')') if r.get('name') else ''}, "
        f"height from OSM `{r['height_source']}` tag:",
        "",
        f"1. LiDAR ground along the footprint edge (plane fit to {r['n_ground_pts']} ground points "
        f"0.5–3 m outside): lowest {r['lidar_ground_low']:.2f} m, highest {r['lidar_ground_high']:.2f} m.",
        (
            f"2. LiDAR roof top = highest roof plane found by RANSAC among {r['n_roof_pts']} roof "
            f"points inside the footprint (shrunk 1 m; {r.get('ransac_planes')} plane(s), "
            f"{(r.get('ransac_outlier_share') or 0) * 100:.0f}% of points on no plane) = "
            f"{r['lidar_roof_top']:.2f} m."
            if r.get("roof_method_used") == "ransac"
            else f"2. LiDAR roof top = 95th percentile of {r['n_roof_pts']} roof points inside the "
            f"footprint (shrunk 1 m) = {r['lidar_roof_top']:.2f} m"
            + (
                f" — {r['roof_method_used'].split('(', 1)[1].rstrip(')')}, so RANSAC was not used."
                if "(" in str(r.get("roof_method_used"))
                else "."
            )
        ),
        f"3. LiDAR height = {r['lidar_roof_top']:.2f} − {r['lidar_ground_low']:.2f} = "
        f"{r['lidar_height']:.2f} m.",
        f"4. M1 = model height − LiDAR height = {float(r['height']):.2f} − {r['lidar_height']:.2f} = "
        f"**{r['M1_height_err']:+.2f} m**.",
    ]
    if r.get("M3_roof_err") is not None:
        L.append(
            f"5. M3 = model roof elevation − LiDAR roof top = {float(r['roof_elev_m']):.2f} − "
            f"{r['lidar_roof_top']:.2f} = **{r['M3_roof_err']:+.2f} m** "
            f"(= M1 {r['M1_height_err']:+.2f} + base error {r['M2_base_err']:+.2f} "
            f"+ platform lift {r['platform_lift']:+.2f})."
        )
    return L


def write_markdown(path, summary, status, meta, args, req, rows):
    names = {
        "M1_height_err": "M1 height (model − LiDAR)",
        "M1_height_err_vs_median": "M1 vs LiDAR median roof",
        "M2_terrain_err": "M2 terrain at building",
        "M2_base_err": "M2 base placement",
        "M3_roof_err": "M3 roof elevation (as built)",
        "M3_roof_err_lowground": "M3 if roof = ground_low + height",
    }
    A = summary["all"]
    t, h = req["terrain"], req["height"]
    miss = [r for r in rows if r.get("in_mesh") is False]
    miss_names = sorted({r["name"] or r["osm_id"] for r in miss})
    n_miss_rel = sum(str(r["osm_id"]).startswith("relation/") for r in miss)
    fp, ev = h["footprints_by_confidence"], h["evaluated_by_confidence"]
    n_ev = sum(ev.values())

    L = [
        "# LiDAR geometry evaluation",
        "",
        "Errors are model − LiDAR in metres: + means the model is too tall or too high. "
        "Brackets are 95% bootstrap confidence ranges.",
        "",
        "## 1. Terrain (pass/fail)",
        "",
        f"- **QR-06a Terrain accuracy: {t['result']}.** {t['text'][0].upper() + t['text'][1:]}.",
    ]
    s_ = A["M2_base_err"]
    if s_.get("n"):
        L.append(
            f"- Base placement (information): {s_['within_1m'] * 100:.0f}% within ±1 m, "
            f"median {s_['median']:+.2f} m."
        )
    L += [
        "",
        "## 2. Building height (baseline, not pass/fail)",
        "",
        "Heights come from OSM, which often has no measured height. This is a known data "
        "limitation (Risk 7), so it is reported as a baseline to improve on rather than as a "
        "pass/fail requirement.",
        "",
        f"- **Where heights come from** ({n_ev} evaluated buildings): "
        f"{ev['high']} with a measured `height` tag, {ev['medium']} with only "
        f"`building:levels`, {ev['low']} with no height (pipeline default).",
        f"- **Baseline:** {h.get('text', 'no heights evaluated')}.",
    ]
    for conf, _, label in GROUPS:
        g = h["groups"].get(conf)
        if g:
            med = summary[label]["M1_height_err"]["median"]
            L.append(
                f"  - {label}: {g['share'] * 100:.0f}% within ±{h['tol_m']:g} m, "
                f"median {med:+.2f} m, n = {g['n']}"
            )
    if miss:
        L.append(
            f"- **In OSM but not in the mesh:** {len(miss)} footprint(s), e.g. "
            f"{', '.join(miss_names[:5])}"
            + (
                f". {n_miss_rel} of them are OSM relations (multipolygons), which "
                "`extrude.py` does not build yet"
                if n_miss_rel
                else ""
            )
            + ". They are listed in `per_building.csv` as `not_in_mesh`."
        )
    L += [
        f"- **Height confidence flag:** every building in `per_building.csv` and "
        f"`per_building_errors.geojson` carries `height_confidence` (high / medium / low), taken "
        f"from its OSM tags. All {sum(fp.values())} footprints: {fp['high']} high, "
        f"{fp['medium']} medium, {fp['low']} low.",
        "",
    ]
    s_ = A["M3_roof_err"]
    if s_.get("n"):
        L += [
            "## 3. Roof elevation, what the viewer sees (information)",
            "",
            f"- {s_['within_3m'] * 100:.0f}% within ±3 m, median {s_['median']:+.2f} m. "
            "Roof error = height error + base error + platform lift, so section 2 explains most of it.",
            "",
        ]
    L += [
        "## Test conditions (for Section 4)",
        "",
        f"- LiDAR: {meta.get('url') or meta.get('files')}",
        f"- LiDAR classifies buildings: {meta['building_class']}; classifies vegetation: "
        f"{meta['vegetation_class']}",
        "- Buildings: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())),
        "- Reference definitions: roof = "
        + (
            f"highest roof plane by RANSAC (±{args.ransac_thresh:g} m, planes ≥ "
            f"{args.ransac_min_share * 100:.0f}% of the points; P{args.roof_pct:g} instead when no "
            f"plane is found or > {args.ransac_max_outliers * 100:.0f}% of points are on no plane)"
            if args.roof_method == "ransac"
            else f"P{args.roof_pct:g}"
        )
        + f" of points inside the footprint shrunk {args.erode:g} m; ground = robust plane through ground points {args.ring_in:g}–"
        f"{args.ring_out:g} m outside, evaluated on the footprint edge; building counted only if "
        f"≥ {args.min_coverage * 100:.0f}% of its roof area has LiDAR returns",
        f"- Sloped site: ground plane grade ≥ {args.grade_cut:g}%",
        f"- Decision rule: PASS when the whole 95% range clears the target, FAIL when it is "
        f"entirely below, otherwise INCONCLUSIVE; fewer than {args.min_n} buildings is INCONCLUSIVE",
        f"- Command: `python {' '.join([Path(sys.argv[0]).name] + sys.argv[1:])}`",
        "",
        "## Example calculation",
        "",
    ]
    L += _example(rows, "high" if ev["high"] else "all")
    L += [
        "",
        "## Figures",
        "",
        "- `01_scorecard.png`: terrain against its target, and the height baseline by confidence",
        "- `02_height_error_by_source.png`: each building's height error, by height confidence",
        "- `03_height_scatter.png`: model height vs real height",
        "- `04_placement_vs_slope.png`: roof placement error by site grade",
        "- `05_height_error_map.png`: where the heights differ most",
        "",
        "## Detailed statistics",
        "",
        "NMAD = spread that ignores outliers.",
        "",
    ]
    for g, block in summary.items():
        L += [
            f"### {g}  (n = {block['n_buildings']})",
            "",
            "| metric | n | bias | median | RMSE | NMAD | ≤1 m | ≤3 m |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m, label in names.items():
            s_ = block[m]
            if s_.get("n", 0) == 0:
                continue
            L.append(
                f"| {label} | {s_['n']} | {_fmt(s_, 'bias_mean')} | {_fmt(s_, 'median')} | "
                f"{_fmt(s_, 'rmse')} | {_fmt(s_, 'nmad')} | {_fmt(s_, 'within_1m', True)} | "
                f"{_fmt(s_, 'within_3m', True)} |"
            )
        L.append("")
    path.write_text("\n".join(L))


# ----------------------------------------------------------------------------
# Plots (matplotlib; reference palette, light surface)
# ----------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SRC_COLOR = {"height": "#2a78d6", "levels": "#eb6834", "default": "#1baf7a"}
SRC_LABEL = {
    "height": "high: height tag",
    "levels": "medium: levels tag",
    "default": "low: no tag (default)",
}


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
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def _head(fig, title, how_to_read):
    """Question-style title plus a one-line 'how to read' note, left-aligned."""
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
    fig.text(0.02, 0.925, how_to_read, ha="left", va="top", fontsize=9.5, color=INK2)


def make_plots(rows, feats, out, a):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from matplotlib.patches import Patch
    except ModuleNotFoundError:
        print("[plots] matplotlib not installed; skipping plots")
        return
    _style(plt)
    for old in (
        "height_scatter.png",
        "height_error_hist.png",
        "roof_error_vs_slope.png",
        "height_error_map.png",
    ):
        (out / old).unlink(missing_ok=True)  # names used by earlier versions
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return
    srcs = [
        s
        for s in ("height", "levels", "default")
        if any(r["height_source"] == s for r in ok)
    ]
    has_m2 = any(r.get("M2_terrain_err") is not None for r in ok)
    has_m3 = any(r.get("M3_roof_err") is not None for r in ok)

    # 01 scorecard: terrain against its target (pass/fail), heights as a baseline
    items = []  # (label, share tuple, target or None, accent)
    if has_m2:
        items.append(
            (
                f"Terrain next to building, ±{a.terrain_tol:g} m",
                share_ci([r.get("M2_terrain_err") for r in ok], a.terrain_tol),
                a.terrain_target,
                True,
            )
        )
    if has_m3:
        items.append(
            (
                "Base placement, ±1 m",
                share_ci([r.get("M2_base_err") for r in ok], 1.0),
                None,
                False,
            )
        )
    for conf, src, _ in GROUPS:
        if src in srcs:
            items.append(
                (
                    f"Height, {SRC_LABEL[src]}, ±{a.height_tol:g} m",
                    share_ci(
                        [r["M1_height_err"] for r in ok if r["height_source"] == src],
                        a.height_tol,
                    ),
                    None,
                    False,
                )
            )
    items.append(
        (
            f"Height, all buildings, ±{a.height_tol:g} m",
            share_ci([r["M1_height_err"] for r in ok], a.height_tol),
            None,
            False,
        )
    )
    items = [it for it in items if it[1]]
    fig, ax = plt.subplots(figsize=(7.6, 0.58 * len(items) + 2.1))
    fig.subplots_adjust(left=0.40, right=0.97, top=0.80, bottom=0.13)
    y = np.arange(len(items))[::-1]
    for yi, (lab, (p, lo, hi, k, n), tgt, acc) in zip(y, items):
        ax.barh(
            yi, p * 100, height=0.56, color="#2a78d6" if acc else "#c3c2b7", zorder=2
        )
        ax.plot([lo * 100, hi * 100], [yi, yi], color=INK, lw=1.2, zorder=3)
        txt = f"{p * 100:.0f}%  ({k} of {n})"
        if p >= 0.5 or hi > 0.8:  # label inside the bar, clear of the CI line
            right = (
                min(p, lo, tgt if tgt is not None else 1.0) * 100 - 2
            )  # keep clear of the target mark
            ax.text(
                right,
                yi,
                txt,
                va="center",
                ha="right",
                fontsize=9.5,
                color="#ffffff" if acc else INK,
                fontweight="bold",
                zorder=4,
            )
        else:
            ax.text(
                hi * 100 + 2, yi, txt, va="center", fontsize=9.5, color=INK, zorder=4
            )
        if tgt is not None:
            ax.plot(
                [tgt * 100] * 2,
                [yi - 0.42, yi + 0.42],
                color=INK,
                lw=1.4,
                ls=(0, (3, 2)),
                zorder=5,
            )
            ax.text(
                tgt * 100,
                yi + 0.46,
                f"target {tgt * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK2,
            )
    n_req = sum(1 for it in items if it[2] is not None)
    if n_req and len(items) > n_req:
        ax.axhline(len(items) - n_req - 0.5, color=GRID, lw=1, zorder=1)
    ax.set_yticks(y, [lab for lab, *_ in items])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Buildings within tolerance of the LiDAR (%)")
    ax.grid(axis="y", visible=False)
    _head(
        fig,
        "Terrain is judged against a target; heights are a measured baseline",
        "Blue = pass/fail requirement. Grey = information and baseline. Black line = 95% range.",
    )
    fig.savefig(out / "01_scorecard.png", dpi=160)
    plt.close(fig)

    # 02 height error per height source: every building is a dot
    errs = np.array([r["M1_height_err"] for r in ok])
    lim = float(max(6.0, min(20.0, np.percentile(np.abs(errs), 98))))
    fig, ax = plt.subplots(figsize=(7.6, 1.0 * len(srcs) + 2.2))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.20)
    ax.axvspan(-3, 3, color="#f0efec", zorder=0)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    rng = np.random.default_rng(1)
    for i, s in enumerate(srcs[::-1]):
        e = np.array([r["M1_height_err"] for r in ok if r["height_source"] == s])
        ec = np.clip(e, -lim, lim)
        ax.scatter(
            ec,
            i + rng.uniform(-0.18, 0.18, len(e)),
            s=18,
            color=SRC_COLOR[s],
            edgecolor=SURFACE,
            linewidth=0.6,
            zorder=3,
        )
        med = float(np.median(e))
        mx_ = float(np.clip(med, -lim, lim))
        ax.scatter([mx_], [i + 0.27], marker="v", s=60, color=INK, zorder=4)
        ax.text(
            mx_,
            i + 0.34,
            f"median {med:+.1f} m",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(
        range(len(srcs)),
        [
            f"{SRC_LABEL[s]}\n(n={sum(r['height_source'] == s for r in ok)})"
            for s in srcs[::-1]
        ],
    )
    ax.set_ylim(-0.6, len(srcs) - 0.1)
    ax.set_xlim(-lim * 1.02, lim * 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Height error, model − LiDAR (m)")
    ax.text(
        0.0,
        -0.16,
        "← model too short",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=INK2,
    )
    ax.text(
        1.0,
        -0.16,
        "model too tall →",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK2,
    )
    ax.text(
        0,
        len(srcs) - 0.15,
        "±3 m (one storey)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK2,
    )
    _head(
        fig,
        "How far off are the heights, by height confidence?",
        f"One dot per building, grouped by where its height came from. ▼ = median. "
        f"Beyond ±{lim:.0f} m drawn at the edge.",
    )
    fig.savefig(out / "02_height_error_by_source.png", dpi=160)
    plt.close(fig)

    # 03 OSM height vs LiDAR height, with the regions named
    hi = (
        max(max(r["lidar_height"] for r in ok), max(float(r["height"]) for r in ok))
        * 1.05
    )
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.10)
    ax.fill_between([0, hi], [-3, hi - 3], [3, hi + 3], color="#f0efec", lw=0, zorder=0)
    ax.plot([0, hi], [0, hi], color=INK2, lw=1, zorder=1)
    for s in srcs:
        rs = [r for r in ok if r["height_source"] == s]
        ax.scatter(
            [r["lidar_height"] for r in rs],
            [float(r["height"]) for r in rs],
            s=24,
            color=SRC_COLOR[s],
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
            label=f"{SRC_LABEL[s]} (n={len(rs)})",
        )
    ax.text(hi * 0.04, hi * 0.95, "model too tall", fontsize=10, color=INK2, va="top")
    ax.text(
        hi * 0.96, hi * 0.05, "model too short", fontsize=10, color=INK2, ha="right"
    )
    ax.set(
        xlim=(0, hi),
        ylim=(0, hi),
        xlabel="Real height from LiDAR (m)",
        ylabel="Height in our model (m)",
    )
    ax.set_aspect("equal")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.88),
        title="Height confidence",
        title_fontsize=9,
        alignment="left",
    )
    _head(
        fig,
        "Does the model height match the real height?",
        "One dot per building. On the diagonal = correct. Grey band = within ±3 m.",
    )
    fig.savefig(out / "03_height_scatter.png", dpi=160)
    plt.close(fig)

    # 04 placement on slopes: median roof error not explained by the height tag
    rs = [r for r in ok if r.get("M3_roof_err") is not None]
    if rs:
        g1, g2 = a.grade_cut / 2, a.grade_cut
        bins = [
            (f"flat\n< {g1:g}% grade", 0, g1),
            (f"gentle\n{g1:g}–{g2:g}%", g1, g2),
            (f"sloped\n≥ {g2:g}% grade", g2, 1e9),
        ]
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        fig.subplots_adjust(left=0.20, right=0.97, top=0.74, bottom=0.25)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        labels, k = [], 0
        for lab, lo_, hi_ in bins:
            b = [r for r in rs if lo_ <= r["site_grade_pct"] < hi_]
            if not b:
                continue
            built = float(np.median([r["M3_roof_err"] - r["M1_height_err"] for r in b]))
            low = float(
                np.median([r["M3_roof_err_lowground"] - r["M1_height_err"] for r in b])
            )
            ax.plot([low, built], [k, k], color="#c3c2b7", lw=2, zorder=2)
            ax.scatter(
                [built], [k], s=70, color="#2a78d6", edgecolor=SURFACE, lw=1.5, zorder=3
            )
            ax.scatter(
                [low], [k], s=70, color="#eb6834", edgecolor=SURFACE, lw=1.5, zorder=3
            )
            for v, c, off in ((built, "#2a78d6", 0.22), (low, "#eb6834", -0.22)):
                ax.text(
                    v,
                    k + off,
                    f"{v:+.2f} m",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=INK,
                )
            labels.append(f"{lab}\n(n={len(b)})")
            k += 1
        ax.set_yticks(range(k), labels)
        ax.set_ylim(-0.6, k - 0.4)
        ax.invert_yaxis()
        span = max(1.0, max(abs(v) for v in ax.get_xlim()))
        ax.set_xlim(-span, span)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Placement error: roof elevation error minus height error (m)")
        ax.legend(
            handles=[
                Patch(color="#2a78d6", label="as built: roof on highest ground"),
                Patch(color="#eb6834", label="alternative: roof on lowest ground"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.45, -0.30),
            ncol=2,
        )
        _head(
            fig,
            "Does putting roofs on the highest ground raise them too much on slopes?",
            "Median per group. Closer to 0 = better placement. Height-tag errors are removed.",
        )
        fig.savefig(out / "04_placement_vs_slope.png", dpi=160)
        plt.close(fig)

    # 05 map of the height error
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    b = np.array([g.bounds for g, _ in feats])
    w, h = b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min()
    fw = 7.6
    fig, ax = plt.subplots(
        figsize=(fw, min(9.0, max(4.0, fw * 0.82 * h / max(w, 1)) + 1.4))
    )
    fig.subplots_adjust(left=0.02, right=0.84, top=0.86, bottom=0.08)
    for (poly, p), r in zip(feats, rows):
        for g in getattr(poly, "geoms", [poly]):
            xs, ys = g.exterior.xy
            if r["status"] == "ok":
                ax.fill(
                    xs,
                    ys,
                    color=cmap(norm(np.clip(r["M1_height_err"], -5, 5))),
                    ec=INK2,
                    lw=0.3,
                )
            else:
                ax.fill(xs, ys, facecolor="none", ec="#a3a29d", lw=0.5, hatch="////")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.grid(False)
    # 100 m scale bar
    x0, y0 = b[:, 0].min(), b[:, 1].min() - 0.04 * h
    ax.plot([x0, x0 + 100], [y0, y0], color=INK, lw=2)
    ax.text(
        x0 + 50,
        y0 - 0.015 * h,
        "100 m",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK2,
    )
    cax = fig.add_axes([0.87, 0.25, 0.025, 0.5])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-5, -3, 0, 3, 5]
    )
    cb.ax.set_yticklabels(["≤ −5", "−3", "0", "+3", "≥ +5"], fontsize=8.5)
    cb.outline.set_visible(False)
    cax.text(
        0.5,
        1.04,
        "too tall",
        transform=cax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK2,
    )
    cax.text(
        0.5,
        -0.04,
        "too short",
        transform=cax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK2,
    )
    ax.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor="#a3a29d",
                hatch="////",
                label="not evaluated (no LiDAR match)",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.01),
        fontsize=8.5,
    )
    _head(
        fig,
        "Which buildings' heights differ most from the LiDAR?",
        "Red = model too tall, blue = too short, pale = about right (model − LiDAR, m).",
    )
    fig.savefig(out / "05_height_error_map.png", dpi=160)
    plt.close(fig)


def assess(rows, a):
    """
    QR-06a terrain (PASS / FAIL / INCONCLUSIVE / NOT TESTED), judged on the 95% CI:
      PASS if the CI's lower end >= terrain_target, FAIL if its upper end < terrain_target,
      otherwise INCONCLUSIVE; fewer than min_n buildings is always INCONCLUSIVE.
    QR-06b height (BASELINE): share within height_tol by height confidence. With
      --height-baseline: IMPROVED if the CI's lower end > baseline, WORSE if its upper end
      < baseline, otherwise NO CLEAR CHANGE.
    """
    ok = [r for r in rows if r["status"] == "ok"]
    res = {}

    sh = share_ci([r.get("M2_terrain_err") for r in ok], a.terrain_tol)
    if sh is None:
        res["terrain"] = {
            "result": "NOT TESTED",
            "text": "no terrain grid, so terrain was not tested",
        }
    else:
        p, lo, hi, k, n = sh
        r = (
            "INCONCLUSIVE"
            if n < a.min_n
            else "PASS"
            if lo >= a.terrain_target
            else "FAIL"
            if hi < a.terrain_target
            else "INCONCLUSIVE"
        )
        note = f" (only {n} buildings, need {a.min_n})" if n < a.min_n else ""
        res["terrain"] = {
            "result": r,
            "share": p,
            "ci95": [lo, hi],
            "hits": k,
            "n": n,
            "tol_m": a.terrain_tol,
            "target": a.terrain_target,
            "text": f"{p * 100:.0f}% of {n} buildings within ±{a.terrain_tol:g} m "
            f"(95% CI {lo * 100:.0f}–{hi * 100:.0f}%), target "
            f"≥ {a.terrain_target * 100:.0f}%{note}",
        }

    h = {"tol_m": a.height_tol, "groups": {}}
    sh = share_ci([r.get("M1_height_err") for r in ok], a.height_tol)
    if sh:
        p, lo, hi, k, n = sh
        h.update(share=p, ci95=[lo, hi], hits=k, n=n)
        h["text"] = (
            f"{p * 100:.0f}% of {n} buildings within ±{a.height_tol:g} m "
            f"(95% CI {lo * 100:.0f}–{hi * 100:.0f}%)"
        )
        if a.height_baseline is not None:
            b = a.height_baseline
            c = (
                "INCONCLUSIVE"
                if n < a.min_n
                else "IMPROVED"
                if lo > b
                else "WORSE"
                if hi < b
                else "NO CLEAR CHANGE"
            )
            h["change"] = {"result": c, "baseline": b}
            h["text"] += f"; against the {b * 100:.0f}% baseline: {c}"
    for conf, _, label in GROUPS:
        sh = share_ci(
            [r.get("M1_height_err") for r in ok if r["height_confidence"] == conf],
            a.height_tol,
        )
        if sh:
            h["groups"][conf] = {
                "label": label,
                "share": sh[0],
                "ci95": [sh[1], sh[2]],
                "hits": sh[3],
                "n": sh[4],
            }
    # how much of the map has a measured height at all (all footprints, evaluated or not)
    h["footprints_by_confidence"] = {
        c: sum(r["height_confidence"] == c for r in rows)
        for c in ("high", "medium", "low")
    }
    h["evaluated_by_confidence"] = {
        c: sum(r["height_confidence"] == c for r in ok)
        for c in ("high", "medium", "low")
    }
    res["height"] = h
    return res


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Evaluate the pipeline's 3D building geometry against LiDAR"
    )
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "model",
        nargs="?",
        help="the pipeline's output: a job folder (output/jobs/<id>/) or a .glb",
    )
    inp.add_argument(
        "--latest",
        action="store_true",
        help="the newest job in output/jobs/ (run from the repo root)",
    )
    inp.add_argument(
        "--from-dir",
        metavar="DIR",
        help="evaluate again from what an earlier run saved in DIR (no .glb or OSM)",
    )
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="the box the model was built with; only for a .glb without job.json "
        "(made with src/main.py)",
    )
    ap.add_argument(
        "--out",
        help="output folder (default geometry_eval/ next to the .glb, or DIR with --from-dir)",
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--laz", nargs="+", help="local LAS/LAZ tiles instead of USGS EPT")
    src.add_argument(
        "--ept", help="EPT ept.json URL (default: newest USGS collection here)"
    )
    ap.add_argument("--laz-crs", help="CRS of --laz files if their header has none")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "lidar_eval"))
    ap.add_argument(
        "--refresh-lidar", action="store_true", help="ignore cached AOI points"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--lidar-z-offset",
        type=float,
        default=0.0,
        help="added to LiDAR z (datum correction), m",
    )
    # reference definitions
    ap.add_argument(
        "--roof-method",
        choices=["ransac", "percentile"],
        default="ransac",
        help="roof top: ransac = highest roof plane (ignores trees, equipment, noise); "
        "percentile = --roof-pct of the roof points",
    )
    ap.add_argument(
        "--roof-pct",
        type=float,
        default=95,
        help="roof top percentile (percentile method)",
    )
    ap.add_argument(
        "--ransac-thresh", type=float, default=0.3, help="RANSAC plane tolerance, m"
    )
    ap.add_argument(
        "--ransac-min-share",
        type=float,
        default=0.10,
        help="smallest roof plane kept, as a share of the roof points (0-1)",
    )
    ap.add_argument(
        "--ransac-max-outliers",
        type=float,
        default=0.5,
        help="if more roof points than this share lie on no plane (towers, spires), "
        "use the percentile for that building instead (0-1)",
    )
    ap.add_argument(
        "--erode", type=float, default=1.0, help="footprint inward buffer, m"
    )
    ap.add_argument(
        "--ring-in", type=float, default=0.5, help="ground ring inner offset, m"
    )
    ap.add_argument(
        "--ring-out", type=float, default=3.0, help="ground ring outer offset, m"
    )
    ap.add_argument(
        "--min-above",
        type=float,
        default=2.0,
        help="roof points must be this far above ground",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="roof cell coverage to count as present",
    )
    ap.add_argument("--min-ground-pts", type=int, default=10)
    ap.add_argument(
        "--min-roof-area", type=float, default=10.0, help="m^2 after erosion"
    )
    ap.add_argument(
        "--embed",
        type=float,
        default=None,
        help="foundation depth below the lowest ground, m (default: extrude.py EMBED_M)",
    )
    ap.add_argument(
        "--grade-cut",
        type=float,
        default=5.0,
        help="ground grade (%%) from which a site counts as sloped",
    )
    # QR-06a terrain (pass/fail) and QR-06b height (baseline)
    ap.add_argument(
        "--terrain-tol", type=float, default=1.0, help="terrain tolerance, m"
    )
    ap.add_argument(
        "--terrain-target",
        type=float,
        default=0.95,
        help="share of buildings within --terrain-tol needed to PASS (0-1)",
    )
    ap.add_argument(
        "--height-tol",
        type=float,
        default=3.0,
        help="height tolerance, m (about one storey)",
    )
    ap.add_argument(
        "--height-baseline",
        type=float,
        help="share within --height-tol from an earlier run (0-1), e.g. 0.40: "
        "reports IMPROVED / NO CLEAR CHANGE / WORSE",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="fewer evaluated buildings than this -> INCONCLUSIVE",
    )
    return ap


def main():
    a = build_parser().parse_args()

    if a.model or a.latest:
        glb = find_model(a.model, a.latest)
        out = Path(a.out) if a.out else glb.parent / "geometry_eval"
        out.mkdir(parents=True, exist_ok=True)
        feats, utm_crs, terrain, embed = build_from_glb(glb, out, a.bbox)
    else:
        out = Path(a.out or a.from_dir)
        out.mkdir(parents=True, exist_ok=True)
        src = Path(a.from_dir)
        feats, utm_crs = load_footprints(src / "buildings_footprints_utm.geojson")
        terrain = load_terrain(src / "terrain_grid.npz")
        embed = EMBED_M
    if a.embed is None:
        a.embed = embed

    b = np.array([g.bounds for g, _ in feats])
    pad = a.ring_out + 10
    aoi = (
        b[:, 0].min() - pad,
        b[:, 1].min() - pad,
        b[:, 2].max() + pad,
        b[:, 3].max() + pad,
    )
    x, y, z, c, meta = load_lidar(a, aoi, utm_crs, out)
    z = z + a.lidar_z_offset
    print(f"[lidar] {len(x):,} points kept; classes {meta['class_counts']}")

    rows, cls = evaluate_buildings(feats, terrain, (x, y, z, c), a)
    meta.update(cls)

    # vertical datum / unit sanity check
    t_err = [r["M2_terrain_err"] for r in rows if r.get("M2_terrain_err") is not None]
    if t_err:
        med = float(np.median(t_err))
        gl = np.median([r["lidar_ground_low"] for r in rows if "lidar_ground_low" in r])
        print(f"[check] median model terrain − LiDAR ground = {med:+.2f} m")
        if abs(gl * 0.3048 - (gl + med)) < 5 or abs(med) > 5:
            print(
                "[check] WARNING: large vertical offset. Check that both are metres on the same "
                "vertical datum (NAVD88). Feet would show as LiDAR ~3.28x the terrain. "
                "Use --lidar-z-offset to correct a known datum shift."
            )

    summary, status = summarize(rows, a.grade_cut)
    req = assess(rows, a)
    print(f"[QR-06a terrain] {req['terrain']['result']}: {req['terrain']['text']}")
    print(
        f"[QR-06b height]  BASELINE: {req['height'].get('text', 'no heights evaluated')}"
    )

    # outputs
    cols = [
        "osm_id",
        "name",
        "status",
        "in_mesh",
        "height_source",
        "osm_rule_height",
        "height_confidence",
        "height",
        "min_height",
        "footprint_m2",
        "n_ground_pts",
        "n_roof_pts",
        "roof_coverage",
        "lidar_ground_low",
        "lidar_ground_high",
        "lidar_ground_rms",
        "ground_drop_m",
        "site_grade_pct",
        "roof_method_used",
        "lidar_roof_top",
        "lidar_roof_pct",
        "lidar_roof_median",
        "ransac_roof_top",
        "ransac_roof_main",
        "ransac_planes",
        "ransac_outlier_share",
        "lidar_height",
        "lidar_height_median",
        "lidar_height_main",
        "base_elev_m",
        "roof_elev_m",
        "platform_lift",
        "M1_height_err",
        "height_within_tol",
        "M1_height_err_vs_median",
        "M2_terrain_err",
        "M2_base_err",
        "M3_roof_err",
        "M3_roof_err_lowground",
    ]
    with open(out / "per_building.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    epsg = utm_crs.split(":")[1]
    (out / "per_building_errors.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
                },
                "features": [
                    {"type": "Feature", "geometry": mapping(g), "properties": r}
                    for (g, _), r in zip(feats, rows)
                ],
            }
        )
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "lidar": meta,
                "status_counts": status,
                "requirements": req,
                "groups": summary,
                "settings": {k: v for k, v in vars(a).items() if k not in ("laz",)},
            },
            indent=2,
            default=str,
        )
    )
    write_markdown(out / "summary.md", summary, status, meta, a, req, rows)
    make_plots(rows, feats, out, a)

    s = summary["all"]
    print(
        "\n[summary] status: "
        + ", ".join(f"{k} {v}" for k, v in sorted(status.items()))
    )
    for m in METRICS:
        d = s[m]
        if d.get("n"):
            print(
                f"  {m:24s} n={d['n']:4d}  median {d['median']:+6.2f}  RMSE {d['rmse']:5.2f}  "
                f"NMAD {d['nmad']:5.2f}  ≤3 m {d['within_3m'] * 100:4.0f}%"
            )
    print(f"[done] wrote {out.resolve()}")


if __name__ == "__main__":
    main()