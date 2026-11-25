import pytest
import requests

from common.constants import BoundingBox
from common.providers import osm
from common.providers.osm import OSMClient

BBOX = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_post(monkeypatch):
    """Replace requests.post with a queue of canned responses."""
    calls = []
    responses = []

    def _post(url, **kwargs):
        calls.append(url)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(osm.requests, "post", _post)
    monkeypatch.setattr(osm.time, "sleep", lambda _: None)
    return calls, responses


def make_client():
    return OSMClient(rounds=2, backoff_s=0, urls=("https://a", "https://b"))


@pytest.fixture(autouse=True)
def overpass_up(monkeypatch):
    """Every test starts with Overpass not marked as down."""
    monkeypatch.setattr(OSMClient, "_overpass_down_until", 0.0)


def _api(monkeypatch):
    asked = []

    def elements(self, bbox, keep):
        asked.append(bbox)
        return {"elements": ["from the api"]}

    monkeypatch.setattr(osm.OsmApiClient, "elements", elements)
    return asked


def test_fetch_uses_the_osm_api(fake_post, monkeypatch):
    # the buildings come from the main OpenStreetMap API; Overpass is not asked (d7265a6)
    calls, responses = fake_post
    responses += [FakeResponse(200, {"elements": ["from overpass"]})]
    asked = _api(monkeypatch)

    assert make_client().fetch(BBOX) == {"elements": ["from the api"]}
    assert calls == [] and len(asked) == 1


def test_fetch_falls_back_to_the_osm_api_and_skips_overpass_while_down(
    fake_post, monkeypatch
):
    calls, responses = fake_post
    responses += [FakeResponse(504)] * 4  # 2 servers x 2 rounds
    asked = _api(monkeypatch)

    assert make_client().fetch(BBOX) == {"elements": ["from the api"]}
    assert asked == [(BBOX.min_lon, BBOX.min_lat, BBOX.max_lon, BBOX.max_lat)]
    tried = len(calls)
    # the next job goes straight to the API
    assert make_client().fetch(BBOX) == {"elements": ["from the api"]}
    assert len(calls) == tried and len(asked) == 2


def test_raises_when_overpass_and_the_osm_api_both_fail(fake_post, monkeypatch):
    _, responses = fake_post
    responses += [FakeResponse(504)] * 4

    def down(self, bbox, keep):
        raise requests.HTTPError("HTTP 503")

    monkeypatch.setattr(osm.OsmApiClient, "elements", down)
    with pytest.raises(requests.HTTPError):
        make_client().fetch(BBOX)


# Overpass: each server in turn, for a few rounds


def test_overpass_falls_back_to_next_endpoint_on_overload(fake_post):
    calls, responses = fake_post
    responses += [FakeResponse(504), FakeResponse(200, {"elements": [1]})]

    assert make_client()._post("query") == {"elements": [1]}
    assert calls == ["https://a", "https://b"]


def test_overpass_retries_another_round_after_all_endpoints_fail(fake_post):
    calls, responses = fake_post
    responses += [
        requests.Timeout(),
        FakeResponse(429),
        FakeResponse(200, {"elements": []}),
    ]

    assert make_client()._post("query") == {"elements": []}
    assert calls == ["https://a", "https://b", "https://a"]


def test_overpass_does_not_retry_client_errors(fake_post):
    calls, responses = fake_post
    responses += [FakeResponse(400)]

    with pytest.raises(requests.HTTPError):
        make_client()._post("query")
    assert calls == ["https://a"]
