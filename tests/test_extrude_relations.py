"""Buildings mapped as relations (multipolygons), and outlines that can't be extruded."""

import numpy as np
import pytest

from common.per_patch_confidence import (
    OPAQUE,
    SEE_THROUGH,
    clip_confidence_score,
    confidence_score,
)
from mesh_builder.extrude import extrude_buildings
from tests.test_terrain import BBOX


def _ring(lon0, lat0, dlon=0.0002, dlat=0.0001):
    ring = [
        (lon0, lat0),
        (lon0 + dlon, lat0),
        (lon0 + dlon, lat0 + dlat),
        (lon0, lat0 + dlat),
        (lon0, lat0),
    ]
    return [{"lat": lat, "lon": lon} for lon, lat in ring]


OUTER = _ring(-83.7472, 42.2838)
COURTYARD = _ring(-83.74715, 42.28382, 0.00005, 0.00003)


def _relation(members, tags=None, rid=7):
    return {
        "type": "relation",
        "id": rid,
        "tags": {"building": "yes", "type": "multipolygon", **(tags or {})},
        "members": members,
    }


def _member(geometry, role="outer", ref=100):
    return {"type": "way", "ref": ref, "role": role, "geometry": geometry}


def _z_up(mesh):
    return -mesh.vertices[:, 2]  # meshes are z-down


def test_a_relation_is_built_at_its_own_height():
    flagged = []
    mesh = extrude_buildings(
        {"elements": [_relation([_member(OUTER)], {"height": "20"})]},
        BBOX,
        flagged=flagged,
    )
    # the member way has no tags: the relation's height, not the 10.5 m default
    assert _z_up(mesh).max() == pytest.approx(20.0)
    assert flagged == []


def test_a_courtyard_is_not_built_as_a_block():
    alone = extrude_buildings({"elements": [_relation([_member(OUTER)])]}, BBOX)
    with_inner = extrude_buildings(
        {
            "elements": [
                _relation([_member(OUTER), _member(COURTYARD, "inner", ref=101)])
            ]
        },
        BBOX,
    )
    assert len(with_inner.faces) == len(alone.faces)


def test_an_outer_member_fetched_as_its_own_building_is_built_once():
    way = {"type": "way", "id": 100, "tags": {"building": "yes"}, "geometry": OUTER}
    flagged = []
    mesh = extrude_buildings(
        {"elements": [way, _relation([_member(OUTER, ref=100)])]},
        BBOX,
        flagged=flagged,
    )
    single = extrude_buildings({"elements": [way]}, BBOX)
    assert len(mesh.faces) == len(single.faces)
    # the relation itself added nothing, so it is reported
    assert flagged == [("relation 7", "no outer outline in data")]


def test_a_way_and_a_relation_with_the_same_number_are_both_built():
    way = {"type": "way", "id": 7, "tags": {"building": "yes"}, "geometry": OUTER}
    other = _ring(-83.7465, 42.2838)
    mesh = extrude_buildings(
        {"elements": [way, _relation([_member(other, ref=200)], rid=7)]}, BBOX
    )
    single = extrude_buildings({"elements": [way]}, BBOX)
    assert len(mesh.faces) == 2 * len(single.faces)


def test_outlines_that_cannot_be_built_are_flagged_not_crashed_on():
    way = {"type": "way", "id": 1, "tags": {"building": "yes"}, "geometry": OUTER}
    two_points = {
        "type": "way",
        "id": 2,
        "tags": {"building": "yes"},
        "geometry": OUTER[:2],
    }
    no_outline = {"type": "way", "id": 3, "tags": {"building": "yes"}}
    # a ring split over two member ways: neither half is a closed outline
    split = _relation([_member(OUTER[:3], ref=300), _member(OUTER[2:], ref=301)], rid=9)
    flagged = []
    mesh = extrude_buildings(
        {"elements": [way, two_points, no_outline, split]}, BBOX, flagged=flagged
    )
    assert len(mesh.faces) > 0  # the good building is still built
    assert dict(flagged) == {
        "way 2": "outline has too few points",
        "way 3": "no outline in data",
        "relation 9": "outline is split over several ways (not built yet)",
    }


def test_confidence_is_compared_with_the_threshold_in_the_same_units():
    # 0.5 and 0.3 against the 0.4 threshold (blend sharpens both sides by the same power)
    raw = np.array([0.5, 0.3, 0.0])
    assert list(clip_confidence_score(raw)) == [OPAQUE, SEE_THROUGH, SEE_THROUGH]
    assert list(clip_confidence_score(raw**8, sharpen=8)) == [
        OPAQUE,
        SEE_THROUGH,
        SEE_THROUGH,
    ]
    # a square-on, clear, close patch is opaque; an oblique, far one is not
    assert confidence_score(np.array([3.0]), 1.0, 1.0) == OPAQUE
    assert confidence_score(np.array([40.0]), 0.5, 1.0) == SEE_THROUGH
