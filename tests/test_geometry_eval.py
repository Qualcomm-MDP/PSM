"""scripts/eval/eval_geometry_lidar.py on a synthetic scene: no network, known answers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common.MeshUtils import _get_utm_transformer
from mesh_builder.extrude import EMBED_M
from terrain import Terrain
from common.constants import BoundingBox

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "eval" / "eval_geometry_lidar.py"
)
spec = importlib.util.spec_from_file_location("eval_geometry_lidar", SCRIPT)
assert spec is not None and spec.loader is not None
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

BBOX = BoundingBox(42.2830, 42.2850, -83.7485, -83.7455)
SLOPE = 0.05  # model ground rises 5 cm per metre east
Z_DATUM = 250.0
LIDAR_GROUND_OFFSET = 0.2  # the real ground is 20 cm above the model terrain
# (tags, real height above the lowest ground) per building
BUILDINGS = [
    ({"height": "12"}, 12.5),  # height tag, 0.5 m too short
    ({"building:levels": "4"}, 18.0),  # 14 m from levels, 4 m too short
    ({}, 6.0),  # untagged: 10.5 m default, 4.5 m too tall
]
TO_UTM = _get_utm_transformer(
    (BBOX.min_lon + BBOX.max_lon) / 2, (BBOX.min_lat + BBOX.max_lat) / 2
)


def _element(i, tags):
    lon0, lat0, dlon, dlat = -83.7480 + 0.0007 * i, 42.2836, 0.0003, 0.0002
    ring = [
        (lon0, lat0),
        (lon0 + dlon, lat0),
        (lon0 + dlon, lat0 + dlat),
        (lon0, lat0 + dlat),
        (lon0, lat0),
    ]
    return {
        "type": "way",
        "id": 100 + i,
        "tags": {"building": "yes", **tags},
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in ring],
    }


ELEMENTS = [_element(i, tags) for i, (tags, _) in enumerate(BUILDINGS)]


def _terrain():
    xs0, ys0 = TO_UTM.transform(BBOX.min_lon - 0.001, BBOX.min_lat - 0.001)
    xs1, ys1 = TO_UTM.transform(BBOX.max_lon + 0.001, BBOX.max_lat + 0.001)
    xs, ys = np.arange(xs0, xs1, 2.0), np.arange(ys0, ys1, 2.0)
    XX, _ = np.meshgrid(xs, ys)
    return Terrain(xs=xs, ys=ys, z=SLOPE * (XX - xs[0]), z_datum=Z_DATUM)


def _lidar(terrain, feats):
    """Ground points on the real ground; flat roofs at the real heights, plus a tree."""
    from shapely import contains_xy
    from shapely.ops import unary_union

    b = np.array([g.bounds for g, _ in feats])
    gx, gy = np.meshgrid(
        np.arange(b[:, 0].min() - 20, b[:, 2].max() + 20, 0.5),
        np.arange(b[:, 1].min() - 20, b[:, 3].max() + 20, 0.5),
    )
    gx, gy = gx.ravel(), gy.ravel()
    out = ~contains_xy(unary_union([g for g, _ in feats]), gx, gy)
    gx, gy = gx[out], gy[out]

    def real_ground(x, y):
        return terrain.ground(x, y) + Z_DATUM + LIDAR_GROUND_OFFSET

    parts = [(gx, gy, real_ground(gx, gy), np.full(len(gx), 2, np.uint8))]
    for (poly, _), (_, h) in zip(feats, BUILDINGS):
        ox, oy = np.asarray(poly.exterior.coords).T
        low = real_ground(ox, oy).min()  # the plane's lowest point is a corner
        rx, ry = np.meshgrid(
            np.arange(poly.bounds[0], poly.bounds[2], 0.7),
            np.arange(poly.bounds[1], poly.bounds[3], 0.7),
        )
        rx, ry = rx.ravel(), ry.ravel()
        inside = contains_xy(poly, rx, ry)
        rx, ry = rx[inside], ry[inside]
        parts.append((rx, ry, np.full(len(rx), low + h), np.ones(len(rx), np.uint8)))
    # a tree 6 m above the first roof: RANSAC must ignore it
    cx, cy = feats[0][0].centroid.coords[0]
    tx, ty = cx + np.random.default_rng(0).normal(0, 1.0, (2, 60))
    tz = parts[1][2][0] + 6.0 + np.random.default_rng(1).normal(0, 0.8, 60)
    parts.append((tx, ty, tz, np.ones(60, np.uint8)))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


@pytest.fixture
def scene(monkeypatch, tmp_path):
    terrain = _terrain()
    monkeypatch.setattr(
        "common.providers.OSMClient.fetch",
        lambda self, bbox, f=None: {"elements": ELEMENTS},
    )
    monkeypatch.setattr("terrain.build_terrain", lambda bbox, **kw: terrain)
    bbox = (BBOX.min_lon, BBOX.min_lat, BBOX.max_lon, BBOX.max_lat)
    feats, crs, terrain_fn, embed = ev.build_from_pipeline(bbox, True, tmp_path)
    a = ev.build_parser().parse_args(["--from-dir", str(tmp_path)])
    a.embed = embed
    pts = _lidar(terrain, feats)
    return feats, crs, terrain_fn, a, pts, tmp_path


def test_buildings_are_placed_as_extrude_py_places_them(scene):
    feats, crs, _, a, _, out = scene
    assert crs == "EPSG:32617" and a.embed == EMBED_M
    assert [p["height_source"] for _, p in feats] == ["height", "levels", "default"]
    assert [p["height"] for _, p in feats] == [12.0, 14.0, 10.5]
    terrain = _terrain()
    for poly, p in feats:
        ox, _ = np.asarray(poly.exterior.coords).T
        g_low = SLOPE * (ox.min() - terrain.xs[0]) + Z_DATUM
        g_high = SLOPE * (ox.max() - terrain.xs[0]) + Z_DATUM
        assert p["base_elev_m"] == pytest.approx(g_low - EMBED_M, abs=0.02)
        assert p["roof_elev_m"] == pytest.approx(g_high + p["height"], abs=0.02)
    assert (out / "buildings_footprints_utm.geojson").exists()
    assert (out / "terrain_grid.npz").exists()


def test_metrics_match_the_known_errors(scene):
    feats, _, terrain_fn, a, pts, _ = scene
    rows, _ = ev.evaluate_buildings(feats, terrain_fn, pts, a)
    assert [r["status"] for r in rows] == ["ok"] * 3
    for r, (_, real_h) in zip(rows, BUILDINGS):
        assert r["roof_method_used"] == "ransac"  # the tree is ignored
        assert r["lidar_height"] == pytest.approx(real_h, abs=0.05)
        assert r["M1_height_err"] == pytest.approx(r["height"] - real_h, abs=0.05)
        assert r["M2_terrain_err"] == pytest.approx(-LIDAR_GROUND_OFFSET, abs=0.02)
        # roof error = height error + base error + platform lift
        assert r["M3_roof_err"] == pytest.approx(
            r["M1_height_err"] + r["M2_base_err"] + r["platform_lift"], abs=0.01
        )


def test_results_do_not_depend_on_the_point_order():
    """RANSAC samples points by position, so the same points read in another order (another
    area, another cache) must still give the same roof: on a noisy, cluttered roof an
    unsorted sample gives a different answer each time."""
    from shapely.geometry import box

    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(-10, 40, 0.5), np.arange(-10, 30, 0.5))
    gx, gy = gx.ravel(), gy.ravel()
    outside = ~((gx > 0) & (gx < 30) & (gy > 0) & (gy < 20))
    gx, gy = gx[outside], gy[outside]
    rx, ry = rng.uniform(1, 29, 3000), rng.uniform(1, 19, 3000)
    rz = 110 + 0.12 * ry + rng.normal(0, 0.15, 3000)  # sloped, noisy roof
    clutter = rng.random(3000) < 0.45
    rz[clutter] += rng.uniform(0.5, 4, clutter.sum())
    pts = (
        np.r_[gx, rx],
        np.r_[gy, ry],
        np.r_[np.full(len(gx), 100.0), rz],
        np.r_[np.full(len(gx), 2), np.ones(3000)].astype(np.uint8),
    )
    a = ev.build_parser().parse_args(["--from-dir", "unused"])
    heights = set()
    for seed in range(6):
        order = np.random.default_rng(seed).permutation(len(pts[0]))
        ctx = ev.prepare_lidar(tuple(p[order] for p in pts), verbose=False)
        heights.add(ev.measure_building(box(0, 0, 30, 20), ctx, a)["lidar_height"])
    assert len(heights) == 1


def test_terrain_is_judged_and_heights_are_a_baseline(scene):
    feats, _, terrain_fn, a, pts, _ = scene
    rows, _ = ev.evaluate_buildings(feats, terrain_fn, pts, a)
    req = ev.assess(rows, a)
    # 3 buildings < --min-n 30: too few to pass
    assert req["terrain"]["result"] == "INCONCLUSIVE"
    assert req["terrain"]["share"] == 1.0
    assert req["height"]["hits"] == 1  # only the height-tagged building is within 3 m
    assert set(req["height"]["groups"]) == {"high", "medium", "low"}
