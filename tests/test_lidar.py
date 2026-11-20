"""Building heights from LiDAR (src/lidar), on synthetic point clouds."""

import io

import numpy as np
import pytest
from pyproj import CRS

from common.pipeline_chain import PipelineState
from lidar import LidarParams, LidarPoints, lidar_building_heights, lidar_stage
from lidar import lidar as lidar_module
from lidar.lidar import _z_factor, osm_height_cap
from mesh_builder.extrude import extrude_buildings
from tests.test_terrain import BBOX, _building, _utm_corners

GROUND, BUILDING, UNCLASSIFIED, HIGH_VEG, NOISE = 2, 6, 1, 5, 18
GROUND_Z = 250.0


def _cloud(
    roof_h=12.0,
    slope=0.0,
    tree=False,
    antenna=False,
    noise=False,
    gable=0.0,
    roof_class=BUILDING,
    seed=0,
):
    """Points for the test building (_building) and the ground around it, in UTM.
    Ground rises `slope` m per m east; the roof is flat at roof_h above the HIGHEST ground
    under the footprint, or a gable whose ridge is `gable` m above that (eaves at roof_h)."""
    rng = np.random.default_rng(seed)
    c = _utm_corners(_building())
    x0, y0 = c[:, 0].min(), c[:, 1].min()
    x1, y1 = c[:, 0].max(), c[:, 1].max()

    def ground(x):
        return GROUND_Z + slope * (x - x0)

    g_high = ground(x1)
    gx, gy = np.meshgrid(
        np.arange(x0 - 20, x1 + 20, 0.7), np.arange(y0 - 20, y1 + 20, 0.7)
    )
    gx, gy = gx.ravel(), gy.ravel()
    off = (gx < x0 - 0.2) | (gx > x1 + 0.2) | (gy < y0 - 0.2) | (gy > y1 + 0.2)
    gx, gy = gx[off], gy[off]
    xs, ys, zs, cs = (
        [gx],
        [gy],
        [ground(gx) + rng.normal(0, 0.03, len(gx))],
        [np.full(len(gx), GROUND)],
    )

    rx, ry = np.meshgrid(
        np.arange(x0 + 0.2, x1 - 0.2, 0.5), np.arange(y0 + 0.2, y1 - 0.2, 0.5)
    )
    rx, ry = rx.ravel(), ry.ravel()
    ym = (y0 + y1) / 2
    rz = g_high + roof_h + gable * (1 - np.abs(ry - ym) / ((y1 - y0) / 2))
    xs.append(rx), ys.append(ry), zs.append(rz + rng.normal(0, 0.05, len(rx)))
    cs.append(np.full(len(rx), roof_class))

    if tree:  # a big tree over a third of the roof, classified as vegetation or not
        t = rx < x0 + (x1 - x0) / 3
        xs.append(rx[t]), ys.append(ry[t])
        zs.append(g_high + roof_h + 8 + rng.normal(0, 1.5, t.sum()))
        cs.append(
            np.full(t.sum(), UNCLASSIFIED if roof_class == UNCLASSIFIED else HIGH_VEG)
        )
    if antenna:  # a mast: a few points far above the roof
        mx, my = (x0 + x1) / 2, ym
        xs.append(np.full(30, mx)), ys.append(np.full(30, my))
        zs.append(g_high + roof_h + np.linspace(0, 25, 30))
        cs.append(np.full(30, roof_class))
    if noise:  # birds and multipath: high-noise class, plus a few unflagged spikes
        xs.append(rx[:50]), ys.append(ry[:50]), zs.append(rz[:50] + 80)
        cs.append(np.full(50, NOISE))
    x, y, z, cl = (np.concatenate(a) for a in (xs, ys, zs, cs))
    keep = np.isin(cl, lidar_module.KEEP_CLASSES)  # as the readers do
    return LidarPoints(
        x[keep], y[keep], z[keep], cl[keep].astype(np.uint8), {"source": "test"}
    )


def _heights(points, tags=None, params=None):
    el = _building()
    el["tags"] = {"building": "yes", **(tags or {})}
    heights, rows = lidar_building_heights([el], BBOX, params, points=points)
    return heights.get("way/1"), rows[0]


def test_flat_roof_height():
    h, row = _heights(_cloud(roof_h=12.0))
    assert row["status"] == "ok"
    assert h == pytest.approx(12.0, abs=0.2)


@pytest.mark.parametrize(
    "extra",
    [{"antenna": True}, {"noise": True}, {"tree": True}],
)
def test_masts_noise_and_trees_do_not_make_it_taller(extra):
    h, _ = _heights(_cloud(roof_h=12.0, **extra))
    assert h == pytest.approx(12.0, abs=0.3)


def test_unclassified_trees_over_the_roof_do_not_make_it_taller():
    # no building or vegetation class: trees are roof candidates, but lie on no roof plane
    h, row = _heights(_cloud(roof_h=12.0, tree=True, roof_class=UNCLASSIFIED))
    assert row["status"] == "ok"
    assert h == pytest.approx(12.0, abs=0.3)


def test_slope_does_not_add_the_ground_drop():
    # 0.1 m/m over a ~16 m wide footprint: measured from the lowest ground the building
    # would come out ~1.6 m taller, and extrude.py adds the drop again under the roof
    h, row = _heights(_cloud(roof_h=12.0, slope=0.1))
    assert row["ground_drop"] > 1.0
    assert h == pytest.approx(12.0, abs=0.3)


def test_pitched_roof_main_is_mid_slope_top_is_the_ridge():
    cloud = _cloud(roof_h=6.0, gable=4.0)
    main, _ = _heights(cloud)
    top, _ = _heights(cloud, params=LidarParams(roof="top"))
    assert 6.5 < main < 9.5  # between the eaves (6) and the ridge (10)
    assert top == pytest.approx(10.0, abs=0.4)


def test_capped_by_the_osm_tags_and_max_height():
    cloud = _cloud(roof_h=40.0)
    h, row = _heights(cloud, {"height": "12"})
    assert h == pytest.approx(18.0) and row["capped_by"] == "height tag"  # 12 + 6
    h, row = _heights(cloud, {"building:levels": "3"})
    assert h == pytest.approx(21.0) and row["capped_by"] == "levels tag"  # 3 x 5 + 6
    h, row = _heights(cloud, params=LidarParams(max_height=25.0))
    assert h == pytest.approx(25.0) and row["capped_by"] == "max height"
    h, row = _heights(cloud)  # no tags, no cap: the measurement
    assert h == pytest.approx(40.0, abs=0.3) and row["capped_by"] == ""


def test_a_mapped_height_is_kept_when_the_lidar_is_far_below_it():
    # Lurie Tower: tagged 51 m, the LiDAR's largest surface ~33 m up
    cloud = _cloud(roof_h=33.0)
    h, row = _heights(cloud, {"height": "51"})
    assert h is None  # extrude.py uses the OSM height
    assert row["status"] == "lidar_far_below_height_tag"
    assert row["lidar_height"] == pytest.approx(33.0, abs=0.3)
    # within max(30%, 6 m) of the tag: the LiDAR height stands
    h, _ = _heights(cloud, {"height": "38"})
    assert h == pytest.approx(33.0, abs=0.3)
    # no floor from levels alone (3.5 m a level is only a guess) or without tags
    h, _ = _heights(_cloud(roof_h=12.0), {"building:levels": "8"})
    assert h == pytest.approx(12.0, abs=0.3)


def _tower_cloud():
    """A deck ~33 m up over most of the footprint, the top at 51 m over a small part
    (as a belfry with a pointed roof leaves it)."""
    cloud = _cloud(roof_h=33.0)
    roof = np.flatnonzero(cloud.c == BUILDING)
    top = roof[::8]  # ~12% of the roof points, spread over the footprint
    z = cloud.z.copy()
    z[top] += 18.0
    return LidarPoints(cloud.x, cloud.y, z, cloud.c, cloud.meta)


def test_towers_take_their_top():
    cloud = _tower_cloud()
    h, row = _heights(cloud, {"building": "tower"})
    assert h == pytest.approx(51.0, abs=0.5) and row["roof_method"].startswith("tower")
    h, _ = _heights(cloud, {"building": "yes", "man_made": "tower"})
    assert h == pytest.approx(51.0, abs=0.5)
    h, _ = _heights(cloud, {"building": "yes", "tower:type": "bell_tower"})
    assert h == pytest.approx(51.0, abs=0.5)
    # a few levels don't cap a tower; they still cap other buildings
    h, _ = _heights(cloud, {"building": "tower", "building:levels": "3"})
    assert h == pytest.approx(51.0, abs=0.5)
    h, _ = _heights(cloud, {"building": "yes", "building:levels": "3"})
    assert h == pytest.approx(21.0)  # 3 x 5 + 6
    # untagged: the main roof, as for any building
    h, _ = _heights(cloud, {"building": "yes"})
    assert h == pytest.approx(33.0, abs=0.5)


def test_osm_height_cap():
    p = LidarParams()
    assert osm_height_cap({"height": "40"}, p) == (60.0, "height tag")  # +50%
    assert osm_height_cap({"building:levels": "2"}, p) == (16.0, "levels tag")
    assert osm_height_cap({}, p) == (None, "")


def test_building_missing_from_the_lidar_keeps_its_osm_height():
    cloud = _cloud()
    roof = cloud.c == BUILDING
    empty = LidarPoints(cloud.x[~roof], cloud.y[~roof], cloud.z[~roof], cloud.c[~roof])
    h, row = _heights(empty)
    assert h is None and row["status"] == "not_in_lidar"


def test_raised_part_keeps_its_osm_height():
    h, row = _heights(_cloud(), {"min_height": "5", "height": "12"})
    assert h is None and row["status"] == "raised_part"


def test_extrusion_uses_the_lidar_height():
    el = _building()
    mesh = extrude_buildings({"elements": [el]}, BBOX, heights={"way/1": 17.5})
    assert (-mesh.vertices[:, 2]).max() == pytest.approx(17.5)
    # other buildings, and tiny heights (at least a storey): as before
    mesh = extrude_buildings({"elements": [el]}, BBOX, heights={"way/2": 30.0})
    assert (-mesh.vertices[:, 2]).max() == pytest.approx(10.0)
    mesh = extrude_buildings({"elements": [el]}, BBOX, heights={"way/1": 0.5})
    assert (-mesh.vertices[:, 2]).max() == pytest.approx(2.5)


def test_relation_outlines_are_measured_and_built_at_their_lidar_height():
    # a multipolygon relation (like the Duderstadt Center): its outer member way carries
    # no tags; extrude.py builds it at the relation's height, so LiDAR measures it too
    way = _building()
    member = {"type": "way", "ref": 77, "role": "outer", "geometry": way["geometry"]}
    rel = {
        "type": "relation",
        "id": 5,
        "tags": {"building": "yes", "type": "multipolygon"},
        "members": [member],
    }
    heights, rows = lidar_building_heights([rel], BBOX, points=_cloud(roof_h=16.0))
    assert rows[0]["key"] == "relation/5/way/77"
    assert heights["relation/5/way/77"] == pytest.approx(16.0, abs=0.3)
    mesh = extrude_buildings({"elements": [rel]}, BBOX, heights=heights)
    assert (-mesh.vertices[:, 2]).max() == pytest.approx(16.0, abs=0.3)


def test_z_in_feet_is_converted():
    assert _z_factor(CRS.from_epsg(32617)) == pytest.approx(1.0)
    assert _z_factor(CRS.from_epsg(2253)) == pytest.approx(
        0.3048006, rel=1e-5
    )  # Michigan South, feet
    navd88_ft = CRS.from_user_input("EPSG:2253+6360")
    assert _z_factor(navd88_ft) == pytest.approx(0.3048006, rel=1e-5)
    navd88_m = CRS.from_user_input("EPSG:2253+5703")  # feet across, metres up
    assert _z_factor(navd88_m) == pytest.approx(1.0)


def _state(**metadata):
    return PipelineState(initial_input=None, current_value=None, metadata=metadata)


def test_stage_does_nothing_when_off(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should not read LiDAR")

    monkeypatch.setattr(lidar_module, "load_points", fail)
    state = _state(bbox=BBOX)
    value = {"osm": {"elements": [_building()]}}
    assert lidar_stage(value, state) is value
    assert state.get_metadata("lidar_heights", None) is None


def test_stage_passes_the_heights_and_options_on(monkeypatch):
    seen = {}

    def load(aoi, crs, reporter, **source):
        seen.update(source)
        return _cloud(roof_h=14.0)

    monkeypatch.setattr(lidar_module, "load_points", load)
    state = _state(
        bbox=BBOX,
        use_lidar=True,
        lidar_options={"roof": "top", "max_height": None, "laz": None, "ept_url": None},
    )
    lidar_stage({"osm": {"elements": [_building()]}}, state)
    assert state.get_metadata("lidar_heights", {})["way/1"] == pytest.approx(
        14.0, abs=0.3
    )
    assert seen == {"laz": None, "ept_url": None}


def test_stage_falls_back_to_osm_heights_on_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise lidar_module.LidarUnavailable("no USGS 3DEP collection covers this area")

    monkeypatch.setattr(lidar_module, "load_points", fail)
    state = _state(bbox=BBOX, use_lidar=True)
    with pytest.warns(UserWarning, match="OpenStreetMap heights"):
        lidar_stage({"osm": {"elements": [_building()]}}, state)
    assert state.get_metadata("lidar_heights", None) is None
    (notice,) = state.get_metadata("notices", [])
    assert "no USGS 3DEP collection" in notice


def test_newest_usgs_collection_first():
    square = {
        "type": "Polygon",
        "coordinates": [[[-84, 42], [-83, 42], [-83, 43], [-84, 43], [-84, 42]]],
    }
    index = {
        "features": [
            {
                "geometry": square,
                "properties": {"name": "MI_Washtenaw_2017", "url": "a"},
            },
            {
                "geometry": square,
                "properties": {"name": "MI_Washtenaw_2020_D20", "url": "b"},
            },
        ]
    }
    hits = lidar_module.find_usgs_ept(-83.7, 42.28, index)
    assert [h["url"] for h in hits] == ["b", "a"]
    assert lidar_module.find_usgs_ept(-80.0, 42.28, index) == []


def test_laz_tiles_in_feet_give_heights_in_metres(tmp_path):
    laspy = pytest.importorskip("laspy")
    from pyproj import Transformer

    cloud = _cloud(roof_h=12.0)
    utm = CRS.from_epsg(32617)
    ft_crs = CRS.from_epsg(2253)  # NAD83 / Michigan South (ft); z in feet too
    fx, fy = Transformer.from_crs(utm, ft_crs, always_xy=True).transform(
        cloud.x, cloud.y
    )
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(ft_crs)
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [float(np.min(fx)), float(np.min(fy)), 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = fx, fy, cloud.z / 0.3048
    las.classification = cloud.c
    path = tmp_path / "tile.laz"
    las.write(path)

    el = _building()
    heights, rows = lidar_building_heights([el], BBOX, laz=[str(path)])
    assert rows[0]["status"] == "ok"
    assert heights["way/1"] == pytest.approx(12.0, abs=0.3)


def test_usgs_ept_collection_is_read_in_the_pipeline_frame(tmp_path, monkeypatch):
    """A one-tile EPT collection in Web Mercator, as USGS publishes them."""
    laspy = pytest.importorskip("laspy")
    from pyproj import Transformer

    cloud = _cloud(roof_h=12.0)
    merc = CRS.from_epsg(3857)
    mx, my = Transformer.from_crs(CRS.from_epsg(32617), merc, always_xy=True).transform(
        cloud.x, cloud.y
    )
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [float(np.min(mx)), float(np.min(my)), 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z, las.classification = mx, my, cloud.z, cloud.c
    buf = io.BytesIO()
    las.write(buf, do_compress=True)
    span = 4096.0
    x0, y0 = float(np.min(mx)) - 100, float(np.min(my)) - 100
    files = {
        "ept.json": {
            "bounds": [x0, y0, 0, x0 + span, y0 + span, span],
            "srs": {"authority": "EPSG", "horizontal": "3857", "vertical": "5703"},
        },
        # the root's child 1-0-0-0 holds the points; 1-1-1-0 is far away
        "ept-hierarchy/0-0-0-0.json": {"0-0-0-0": 0, "1-0-0-0": len(mx), "1-1-1-0": 5},
        "ept-data/1-0-0-0.laz": buf.getvalue(),
    }
    asked = []

    class _Resp:
        def __init__(self, body):
            self.content = body if isinstance(body, bytes) else b""
            self._json = body

        def json(self):
            return self._json

    def get(session, url, **kwargs):
        key = url.split("/MI_Test_2021/", 1)[1]
        asked.append(key)
        return _Resp(files[key])

    monkeypatch.setattr(lidar_module, "get_with_retries", get)
    heights, rows = lidar_building_heights(
        [_building()],
        BBOX,
        ept_url="https://bucket/MI_Test_2021/ept.json",
        cache_dir=tmp_path,
    )
    assert heights["way/1"] == pytest.approx(12.0, abs=0.3)
    assert "ept-data/1-1-1-0.laz" not in asked  # tiles off the area are not downloaded
    assert (tmp_path / "MI_Test_2021" / "1-0-0-0.laz").exists()  # cached
