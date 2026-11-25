"""Buildings mapped as relations (multipolygons), outlines that can't be extruded, and the
texture confidence."""

import numpy as np

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


def test_a_relation_is_built_from_its_member_outlines():
    """A building mapped as a relation (the Duderstadt Center) is extruded from its members."""
    mesh = extrude_buildings({"elements": [_relation([_member(OUTER)])]}, BBOX)
    assert len(mesh.faces) > 0


def test_outlines_that_cannot_be_built_are_skipped_not_crashed_on():
    way = {"type": "way", "id": 1, "tags": {"building": "yes"}, "geometry": OUTER}
    two_points = {
        "type": "way",
        "id": 2,
        "tags": {"building": "yes"},
        "geometry": OUTER[:2],
    }
    no_outline = {"type": "way", "id": 3, "tags": {"building": "yes"}}
    mesh = extrude_buildings({"elements": [way, two_points, no_outline]}, BBOX)
    single = extrude_buildings({"elements": [way]}, BBOX)
    assert len(mesh.faces) == len(single.faces)  # only the good building


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
