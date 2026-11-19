"""Mapillary image search: tiles Mapillary keeps failing on are asked for in quarters."""

import pytest
import requests

from common.constants import BoundingBox
from common.providers import mapillary
from common.providers.mapillary import MapillaryClient

TILE = BoundingBox(
    min_lat=37.7843, max_lat=37.7848, min_lon=-122.4082, max_lon=-122.4077
)


class FakeResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or []

    def json(self):
        return {"data": self._data}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error for url: x", response=self
            )


@pytest.fixture
def requests_seen(monkeypatch):
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", "MLY|1|test")
    monkeypatch.setattr(mapillary.time, "sleep", lambda s: None)
    seen = []
    return seen


def _serve(monkeypatch, seen, answer):
    def get(url, params=None, **kwargs):
        seen.append(params["bbox"])
        return answer(params["bbox"])

    monkeypatch.setattr(mapillary.requests, "get", get)


def test_a_tile_mapillary_fails_on_is_asked_for_in_quarters(monkeypatch, requests_seen):
    whole = TILE.to_str()

    def answer(bbox):
        if bbox == whole:
            return FakeResponse(500)  # too dense to answer at once
        return FakeResponse(200, [{"id": bbox}])  # one image per quarter

    _serve(monkeypatch, requests_seen, answer)
    images = MapillaryClient()._fetch_bbox(TILE, ["id"], 2000)
    assert len(images) == 4
    assert (
        requests_seen.count(whole) == mapillary.REQUEST_RETRIES
    )  # retried, then split
    assert len(requests_seen) == mapillary.REQUEST_RETRIES + 4


def test_a_client_error_fails_at_once(monkeypatch, requests_seen):
    _serve(monkeypatch, requests_seen, lambda bbox: FakeResponse(403))
    with pytest.raises(RuntimeError, match="403"):
        MapillaryClient()._fetch_bbox(TILE, ["id"], 2000)
    assert len(requests_seen) == 1  # no retries, no splitting


def test_an_outage_still_gives_up(monkeypatch, requests_seen):
    _serve(monkeypatch, requests_seen, lambda bbox: FakeResponse(503))
    with pytest.raises(RuntimeError, match="503"):
        MapillaryClient()._fetch_bbox(TILE, ["id"], 2000)
    # the tile, its 4 quarters, then the first quarter's 4 quarters: the first of those
    # (at the deepest level) gives up and the error ends the search
    tiles = len(requests_seen) // mapillary.REQUEST_RETRIES
    assert len(requests_seen) % mapillary.REQUEST_RETRIES == 0
    assert tiles == 1 + 1 + 1  # tile, first quarter, first quarter's first quarter
