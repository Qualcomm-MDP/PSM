"""scripts/eval/eval_geometry_lidar.py on a synthetic scene: the pipeline builds and exports
a .glb, the script evaluates that file. No network, known answers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from common import ProgressMonitor
from common.constants import BoundingBox
from common.MeshUtils import _get_utm_transformer
from common.pipeline_chain import PipelineState
from mesh_builder.extrude import EMBED_M, extrude_buildings
from pipeline import export_mesh
from terrain import Terrain

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
    # a multipolygon relation with a courtyard, like the Duderstadt Center: extrude.py
    # does not build relations, so it is not in the mesh
    ({"building:levels": "3", "type": "multipolygon"}, 15.4),
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


def _relation(i, tags):
    """The same square as _element(i), as a relation: its outer ring split over two member
    ways (as OSM often does) and an inner ring (a courtyard)."""
    lon0, lat0, dlon, dlat = -83.7480 + 0.0007 * i, 42.2836, 0.0003, 0.0002

    def pts(ring):
        return [{"lat": lat0 + fy * dlat, "lon": lon0 + fx * dlon} for fx, fy in ring]

    return {
        "type": "relation",
        "id": 900 + i,
        "tags": {"building": "university", **tags},
        "members": [
            {
                "type": "way",
                "ref": 1,
                "role": "outer",
                "geometry": pts([(0, 0), (1, 0), (1, 1)]),
            },
            {
                "type": "way",
                "ref": 2,
                "role": "outer",
                "geometry": pts([(1, 1), (0, 1), (0, 0)]),
            },
            {
                "type": "way",
                "ref": 3,
                "role": "inner",
                "geometry": pts(
                    [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.3, 0.3)]
                ),
            },
        ],
    }


ELEMENTS = [
    (_relation if tags.get("type") == "multipolygon" else _element)(i, tags)
    for i, (tags, _) in enumerate(BUILDINGS)
]


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


def _export(tmp_path, terrain):
    """The pipeline's mesh and .glb for ELEMENTS (extrude_buildings + export_mesh)."""
    mesh = extrude_buildings({"elements": ELEMENTS}, BBOX, terrain=terrain)
    path = tmp_path / "model.glb"
    state = PipelineState(
        initial_input=None,
        current_value=None,
        metadata={
            "bbox": BBOX,
            "terrain": terrain,
            "output_path": str(path),
            "progress_monitor": ProgressMonitor(verbose=False),
        },
    )
    export_mesh(mesh, state)
    return path


BOX = (BBOX.min_lon, BBOX.min_lat, BBOX.max_lon, BBOX.max_lat)


@pytest.fixture
def osm(monkeypatch):
    """OSM and the elevation tiles as the pipeline saw them."""
    monkeypatch.setattr(
        "common.providers.OSMClient.fetch",
        lambda self, bbox, f=None: {"elements": ELEMENTS},
    )
    monkeypatch.setattr("terrain.build_terrain", lambda bbox, **kw: _terrain())


@pytest.fixture
def scene(osm, tmp_path):
    terrain = _terrain()
    glb = _export(tmp_path, terrain)
    out = tmp_path / "eval"
    out.mkdir()
    feats, crs, terrain_fn, embed = ev.build_from_glb(glb, out, BOX)
    a = ev.build_parser().parse_args(["--from-dir", str(out)])
    a.embed = embed
    pts = _lidar(terrain, feats)
    return feats, crs, terrain_fn, a, pts, out


def test_buildings_are_read_back_from_the_glb(scene):
    feats, crs, _, a, _, out = scene
    assert crs == "EPSG:32617" and a.embed == EMBED_M
    assert [p["in_mesh"] for _, p in feats] == [True, True, True, False]
    assert [p["height_source"] for _, p in feats] == [
        "height",
        "levels",
        "default",
        "levels",
    ]
    terrain = _terrain()
    for poly, p in feats[:3]:
        # the height extrude.py used, the walls down to the lowest ground - EMBED_M and
        # the roof on the highest ground, all measured in the .glb
        assert p["height"] == pytest.approx(p["osm_rule_height"], abs=0.01)
        ox, _ = np.asarray(poly.exterior.coords).T
        g_low = SLOPE * (ox.min() - terrain.xs[0]) + Z_DATUM
        g_high = SLOPE * (ox.max() - terrain.xs[0]) + Z_DATUM
        assert p["base_elev_m"] == pytest.approx(g_low - EMBED_M, abs=0.02)
        assert p["roof_elev_m"] == pytest.approx(g_high + p["height"], abs=0.02)
    for name in (
        "buildings_footprints_utm.geojson",
        "terrain_grid.npz",
        "osm_buildings.json",
    ):
        assert (out / name).exists()


def test_metrics_match_the_known_errors(scene):
    feats, _, terrain_fn, a, pts, _ = scene
    rows, _ = ev.evaluate_buildings(feats, terrain_fn, pts, a)
    assert [r["status"] for r in rows] == ["ok", "ok", "ok", "not_in_mesh"]
    for r, (_, real_h) in zip(rows[:3], BUILDINGS):
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


def test_buildings_missing_from_the_mesh_are_reported(scene):
    """A relation (e.g. Duderstadt Center) has its outline in its members: the script
    reads it, with the courtyard cut out, and reports that the mesh has no such
    building."""
    feats, _, terrain_fn, a, pts, _ = scene
    poly, props = feats[3]
    assert props["osm_id"] == "relation/903" and props["in_mesh"] is False
    assert len(poly.interiors) == 1  # the courtyard
    rows, _ = ev.evaluate_buildings(feats, terrain_fn, pts, a)
    assert rows[3]["status"] == "not_in_mesh"


def test_a_flat_model_gets_heights_only(osm, tmp_path):
    glb = _export(tmp_path, terrain=None)
    feats, _, terrain_fn, _ = ev.build_from_glb(glb, tmp_path, BOX)
    assert terrain_fn is None
    assert [p.get("height") for _, p in feats[:3]] == pytest.approx([12.0, 14.0, 10.5])
    assert all("roof_elev_m" not in p for _, p in feats)


def test_a_web_app_job_gives_its_own_bounding_box(osm, tmp_path):
    job = tmp_path / "jobs" / "abc123"
    job.mkdir(parents=True)
    _export(job, _terrain())
    west, south, east, north = BOX
    (job / "job.json").write_text(
        json.dumps(
            {"bbox": {"west": west, "south": south, "east": east, "north": north}}
        )
    )
    glb = ev.find_model(latest=True, jobs_dir=tmp_path / "jobs")
    assert glb == job / "model.glb" == ev.find_model(job)
    feats, _, _, _ = ev.build_from_glb(glb, tmp_path)
    assert [p["in_mesh"] for _, p in feats] == [True, True, True, False]


def test_buildings_split_over_several_meshes_are_joined(osm, tmp_path):
    """A textured model can load as several meshes (geometry_0, geometry_1, ...)."""
    import trimesh

    glb = _export(tmp_path, _terrain())
    scene = trimesh.load(glb, force="scene")
    buildings, ground = scene.geometry["buildings"], scene.geometry["terrain"]
    split = trimesh.Scene()
    for k, faces in enumerate(np.array_split(np.arange(len(buildings.faces)), 3)):
        split.add_geometry(
            buildings.submesh([faces], append=True), geom_name=f"geometry_{k}"
        )
    split.add_geometry(ground, node_name="terrain", geom_name="terrain")
    split.export(glb)
    feats, _, _, _ = ev.build_from_glb(glb, tmp_path, BOX)
    assert [p["in_mesh"] for _, p in feats] == [True, True, True, False]
    assert [p["height"] for _, p in feats[:3]] == pytest.approx(
        [12.0, 14.0, 10.5], abs=0.01
    )


def test_a_glb_without_job_json_needs_its_bounding_box(osm, tmp_path):
    glb = _export(tmp_path, _terrain())
    with pytest.raises(SystemExit, match="--bbox"):
        ev.build_from_glb(glb, tmp_path)


def test_relation_members_already_fetched_as_ways_are_not_counted_twice():
    relation = _relation(0, {})

    def to_xy(points):
        return [TO_UTM.transform(p["lon"], p["lat"]) for p in points]

    # member way 1 is also a building way of its own: only member 2 is left, which is
    # not a closed ring, so the relation adds no footprint
    assert ev.outline_parts(relation, to_xy, way_ids={1}) == []
    assert len(ev.outline_parts(relation, to_xy)) == 1