import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import geocode
from api.app import create_app
from common import PipelineChain
from common.reporter import get_reporter

BBOX = {"west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422}


def fetch(value, state):
    reporter = get_reporter(state)
    reporter.info("Fetched 2 OSM elements")
    for _ in reporter.progress(range(2), desc="Downloading images"):
        pass
    return {
        "osm": {"elements": [{"id": 1, "geometry": []}, {"id": 2}]},
        "mapillary": {"img1": {}, "img2": {}},
    }


def export(value, state):
    Path(state.metadata["output_path"]).write_bytes(b"glTF-binary")
    images = Path(state.metadata["image_dir"])
    images.mkdir(parents=True, exist_ok=True)
    (images / "img1_original.jpg").write_bytes(b"jpeg")
    (images / "images.json").write_text(
        json.dumps(
            [
                {
                    "id": "img1",
                    "heading": 90,
                    "original": "img1_original.jpg",
                    "cleaned": None,
                }
            ]
        )
    )
    return value


def fake_pipeline():
    pipeline = PipelineChain("fake")
    pipeline.add_stage("fetch", fetch, label="Fetch data", detail="OSM + Mapillary")
    pipeline.add_stage("export", export, label="Export", detail="Write .glb")
    return pipeline


def failing_pipeline():
    def no_images(value, state):
        raise RuntimeError("No images found.")

    pipeline = PipelineChain("failing")
    pipeline.add_stage("fetch", no_images, label="Fetch data")
    pipeline.add_stage("export", export, label="Export")
    return pipeline


def make_client(tmp_path, factory=fake_pipeline):
    return TestClient(
        create_app(data_dir=tmp_path, pipeline_factory=factory, serve_web=False)
    )


def wait_for(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_pipeline_endpoint_lists_stages(tmp_path):
    client = make_client(tmp_path)

    stages = client.get("/api/pipeline").json()["stages"]

    assert [s["name"] for s in stages] == ["fetch", "export"]
    assert stages[0]["label"] == "Fetch data"


def test_job_runs_and_serves_model_and_images(tmp_path):
    client = make_client(tmp_path)

    created = client.post("/api/jobs", json={"bbox": BBOX, "name": "Test block"})
    assert created.status_code == 201
    job = wait_for(client, created.json()["id"])

    assert job["status"] == "done"
    assert job["stages"] == {"fetch": "done", "export": "done"}
    assert job["result"]["buildings"] == 1
    assert job["result"]["photos"] == 2
    assert job["result"]["thumbnail"] == "img1_original.jpg"

    model = client.get(f"/api/jobs/{job['id']}/model.glb?download=true")
    assert model.status_code == 200
    assert model.content == b"glTF-binary"
    assert "psm-test-block" in model.headers["content-disposition"]

    images = client.get(f"/api/jobs/{job['id']}/images").json()["images"]
    assert images[0]["original"] == f"/api/jobs/{job['id']}/images/img1_original.jpg"
    assert client.get(images[0]["original"]).content == b"jpeg"


def test_failed_job_reports_error_and_stage(tmp_path):
    client = make_client(tmp_path, failing_pipeline)

    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    job = wait_for(client, job_id)

    assert job["status"] == "failed"
    assert job["error"] == "No images found."
    assert job["stages"] == {"fetch": "error", "export": "pending"}
    assert client.get(f"/api/jobs/{job_id}/model.glb").status_code == 404


def test_finished_jobs_are_reloaded_after_restart(tmp_path):
    client = make_client(tmp_path)
    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    wait_for(client, job_id)

    restarted = make_client(tmp_path)

    jobs = restarted.get("/api/jobs").json()["jobs"]
    assert [j["id"] for j in jobs] == [job_id]
    assert jobs[0]["status"] == "done"


@pytest.mark.parametrize(
    "bbox, message",
    [
        ({**BBOX, "west": BBOX["east"] + 0.01}, "West must be less than east"),
        ({**BBOX, "east": BBOX["west"] + 0.1}, "Keep each side under"),
    ],
)
def test_rejects_bad_bounding_boxes(tmp_path, bbox, message):
    client = make_client(tmp_path)

    resp = client.post("/api/jobs", json={"bbox": bbox})

    assert resp.status_code == 422
    assert message in json.dumps(resp.json())


def test_image_names_cannot_escape_job_folder(tmp_path):
    client = make_client(tmp_path)
    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    wait_for(client, job_id)

    assert client.get(f"/api/jobs/{job_id}/images/..%2Fjob.json").status_code == 404


def test_geocode_parses_and_caches_results(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "name": "Pierpont Commons",
                    "display_name": "Pierpont Commons, Bonisteel Blvd, Ann Arbor",
                    "lat": "42.2913",
                    "lon": "-83.7173",
                    "boundingbox": ["42.2908", "42.2918", "-83.7180", "-83.7166"],
                }
            ]

    def fake_get(url, **kwargs):
        calls.append(kwargs["params"]["q"])
        return FakeResponse()

    monkeypatch.setattr(geocode.requests, "get", fake_get)
    monkeypatch.setattr(geocode, "_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(geocode, "_cache", type(geocode._cache)())
    client = make_client(tmp_path)

    first = client.get("/api/geocode", params={"q": "Pierpont Commons"}).json()
    client.get("/api/geocode", params={"q": "pierpont commons "})

    place = first["results"][0]
    assert place["name"] == "Pierpont Commons"
    assert place["bbox"] == {
        "west": -83.7180,
        "south": 42.2908,
        "east": -83.7166,
        "north": 42.2918,
    }
    assert calls == ["Pierpont Commons"]


def test_real_pipeline_stages_drive_the_timeline():
    from pipeline import build_pipeline

    assert [s["name"] for s in build_pipeline().describe()] == [
        "fetch",
        "align_poses",
        "terrain",
        "build_mesh",
        "naip",
        "texturing",
        "export",
        "evaluate_geometry",
        "evaluate_texture",
    ]


def test_suggest_uses_photon_and_handles_points_and_extents(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "features": [
                    {
                        "geometry": {"coordinates": [-83.7176, 42.2911]},
                        "properties": {
                            "name": "Pierpont Commons",
                            "city": "Ann Arbor",
                            "state": "Michigan",
                            "country": "United States",
                            "extent": [-83.7180, 42.2916, -83.7170, 42.2907],
                        },
                    },
                    {
                        "geometry": {"coordinates": [-83.7483, 42.2808]},
                        "properties": {
                            "housenumber": "300",
                            "street": "South Main Street",
                            "city": "Ann Arbor",
                        },
                    },
                ]
            }

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return FakeResponse()

    monkeypatch.setattr(geocode.requests, "get", fake_get)
    monkeypatch.setattr(geocode, "_suggest_cache", type(geocode._suggest_cache)())
    client = make_client(tmp_path)

    resp = client.get(
        "/api/suggest", params={"q": "pierp", "lat": 42.2911, "lon": -83.7176}
    )
    client.get("/api/suggest", params={"q": "pierp", "lat": 42.2912, "lon": -83.7177})

    first, second = resp.json()["results"]
    assert first["address"] == "Pierpont Commons, Ann Arbor, Michigan, United States"
    assert first["bbox"] == {
        "west": -83.7180,
        "south": 42.2907,
        "east": -83.7170,
        "north": 42.2916,
    }
    assert second["name"] == "300 South Main Street"
    assert second["bbox"]["west"] == second["bbox"]["east"] == -83.7483
    assert calls[0][0] == geocode.PHOTON_URL
    assert calls[0][1]["lat"] == 42.29
    assert len(calls) == 1  # nearby bias point reuses the cached answer


def test_suggest_ignores_short_queries(tmp_path):
    client = make_client(tmp_path)

    assert client.get("/api/suggest", params={"q": "ab"}).json() == {"results": []}


def test_websocket_sends_snapshot_then_live_job_updates(tmp_path):
    client = make_client(tmp_path)

    with client.websocket_connect("/api/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot == {"type": "snapshot", "jobs": []}

        job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]

        updates = []
        while True:
            msg = ws.receive_json()
            if msg["type"] != "job":
                continue  # log and progress messages are covered separately
            assert msg["job"]["id"] == job_id
            updates.append(msg["job"])
            if msg["job"]["status"] in ("done", "failed"):
                break

    statuses = [u["status"] for u in updates]
    assert statuses[0] == "queued"
    assert statuses[-1] == "done"
    assert "running" in statuses
    # Each stage is reported as it starts and finishes.
    fetch_states = [u["stages"]["fetch"] for u in updates]
    assert "running" in fetch_states and fetch_states[-1] == "done"


def test_websocket_snapshot_includes_existing_jobs(tmp_path):
    client = make_client(tmp_path)
    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    wait_for(client, job_id)

    with client.websocket_connect("/api/ws") as ws:
        snapshot = ws.receive_json()

    assert [j["id"] for j in snapshot["jobs"]] == [job_id]
    assert snapshot["jobs"][0]["status"] == "done"


def test_job_log_is_saved_and_served(tmp_path):
    client = make_client(tmp_path)
    job_id = client.post("/api/jobs", json={"bbox": BBOX, "name": "Log test"}).json()[
        "id"
    ]
    wait_for(client, job_id)

    text = client.get(f"/api/jobs/{job_id}/log").text
    lines = [line.split("  ", 1)[1] for line in text.splitlines()]

    assert lines[0].startswith(f"Job {job_id} · Log test")
    assert "── Fetch data ──" in lines
    assert "Fetched 2 OSM elements" in lines
    assert lines[-1].startswith("Done in")
    download = client.get(f"/api/jobs/{job_id}/log?download=true")
    assert f"psm-{job_id}-pipeline.log" in download.headers["content-disposition"]


def test_failed_job_log_contains_traceback(tmp_path):
    client = make_client(tmp_path, failing_pipeline)
    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    wait_for(client, job_id)

    text = client.get(f"/api/jobs/{job_id}/log").text

    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: No images found." in text
    assert text.splitlines()[-1].endswith("Failed after 0s: No images found.")


def test_websocket_streams_log_lines_before_final_status(tmp_path):
    client = make_client(tmp_path)

    with client.websocket_connect("/api/ws") as ws:
        ws.receive_json()  # snapshot
        job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] == "job" and msg["job"]["status"] == "done":
                break

    logs = [m for m in messages if m["type"] == "log"]
    assert all(m["job_id"] == job_id for m in logs)
    assert [m["n"] for m in logs] == list(range(1, len(logs) + 1))
    assert any(m["line"].endswith("Fetched 2 OSM elements") for m in logs)
    assert logs[-1]["line"].split("  ", 1)[1].startswith("Done in")


def test_mapillary_token_is_redacted_from_logs_and_errors(tmp_path, monkeypatch):
    secret = "MLY|1234567890|abcdefsecret"
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", secret)

    def leaky(value, state):
        raise RuntimeError(
            "403 Client Error for url: https://graph.mapillary.com/images"
            f"?access_token={secret}&bbox=1,2,3,4 (token {secret})"
        )

    pipeline = PipelineChain("leaky")
    pipeline.add_stage("fetch", leaky, label="Fetch data")
    client = make_client(tmp_path, lambda: pipeline)

    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    job = wait_for(client, job_id)
    log = client.get(f"/api/jobs/{job_id}/log").text

    assert job["status"] == "failed"
    for text in (job["error"], log):
        assert secret not in text
        assert "access_token=[REDACTED]" in text


def test_worker_keeps_running_after_a_job_crashes_outside_the_pipeline(
    tmp_path, monkeypatch
):
    import api.jobs

    real_reporter = api.jobs.JobReporter
    calls = []

    def flaky_reporter(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return real_reporter(*args, **kwargs)

    monkeypatch.setattr(api.jobs, "JobReporter", flaky_reporter)
    client = make_client(tmp_path)

    first = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    second = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]

    crashed = wait_for(client, first)
    assert crashed["status"] == "failed"
    assert crashed["error"] == "disk full"
    assert wait_for(client, second)["status"] == "done"


def test_lagging_live_client_is_cut_off_instead_of_missing_updates():
    from api.live import _MAX_QUEUED, OVERFLOW, JobHub

    hub = JobHub()
    lagging = hub.subscribe()

    for i in range(_MAX_QUEUED + 1):
        hub._fan_out({"type": "job", "n": i})

    # The backlog is replaced by one message telling the handler to close, and
    # the client gets nothing more until it reconnects (with a fresh snapshot).
    assert lagging.qsize() == 1
    assert lagging.get_nowait() is OVERFLOW
    hub._fan_out({"type": "job", "n": "later"})
    assert lagging.empty()


def optional_stages_pipeline(seen):
    def record(value, state):
        seen["use_terrain"] = state.get_metadata("use_terrain", None)
        seen["use_naip"] = state.get_metadata("use_naip", None)
        return {"osm": {"elements": []}, "mapillary": {}}

    def stage(value, state):
        return value

    def factory():
        pipeline = PipelineChain("optional")
        pipeline.add_stage("fetch", record, label="Fetch data")
        pipeline.add_stage("terrain", stage, label="Terrain")
        pipeline.add_stage("naip", stage, label="Aerial photo")
        return pipeline

    return factory


@pytest.mark.parametrize(
    "request_options, options, use, stages",
    [
        (
            {},
            {"terrain": False, "naip": False},
            (False, False),
            {"fetch": "done", "terrain": "skipped", "naip": "skipped"},
        ),
        (
            {"terrain": True},
            {"terrain": True, "naip": False},
            (True, False),
            {"fetch": "done", "terrain": "done", "naip": "skipped"},
        ),
        (
            # Aerial photos are draped on the terrain, so they turn it on.
            {"naip": True},
            {"terrain": True, "naip": True},
            (True, True),
            {"fetch": "done", "terrain": "done", "naip": "done"},
        ),
    ],
)
def test_terrain_and_naip_options(tmp_path, request_options, options, use, stages):
    seen = {}
    client = make_client(tmp_path, optional_stages_pipeline(seen))

    created = client.post("/api/jobs", json={"bbox": BBOX, **request_options}).json()
    assert {k: created["options"][k] for k in options} == options
    job = wait_for(client, created["id"])

    assert job["status"] == "done"
    assert (seen["use_terrain"], seen["use_naip"]) == use
    assert job["stages"] == stages
    log = client.get(f"/api/jobs/{job['id']}/log").text
    # A skipped stage doesn't get a section in the log either.
    assert ("── Terrain ──" in log) == options["terrain"]


def test_profile_option_writes_json_and_job_summary(tmp_path):
    client = make_client(tmp_path)
    created = client.post("/api/jobs", json={"bbox": BBOX, "profile": True}).json()
    assert created["options"]["profile"] is True
    job = wait_for(client, created["id"])

    assert job["status"] == "done"
    stages = job["result"]["profile"]["stages"]
    assert [s["name"] for s in stages] == ["fetch", "export"]
    assert all(s["status"] == "ok" for s in stages)

    report = client.get(f"/api/jobs/{job['id']}/profile.json")
    assert report.status_code == 200
    assert [s["name"] for s in report.json()["stages"]] == ["fetch", "export"]
    download = client.get(f"/api/jobs/{job['id']}/profile.json?download=true")
    assert "profile.json" in download.headers.get("content-disposition", "")


def test_profile_off_has_no_file(tmp_path):
    client = make_client(tmp_path)
    job = wait_for(client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"])
    assert "profile" not in job["result"]
    assert client.get(f"/api/jobs/{job['id']}/profile.json").status_code == 404


def test_profile_marks_optional_stages_skipped(tmp_path):
    client = make_client(tmp_path, optional_stages_pipeline({}))
    job = wait_for(
        client,
        client.post("/api/jobs", json={"bbox": BBOX, "profile": True}).json()["id"],
    )
    assert {s["name"]: s["status"] for s in job["result"]["profile"]["stages"]} == {
        "fetch": "ok",
        "terrain": "skipped",
        "naip": "skipped",
    }


def test_profile_is_pushed_live_after_each_stage(tmp_path):
    client = make_client(tmp_path)

    with client.websocket_connect("/api/ws") as ws:
        ws.receive_json()  # snapshot
        job_id = client.post("/api/jobs", json={"bbox": BBOX, "profile": True}).json()[
            "id"
        ]
        profiles = []
        while True:
            msg = ws.receive_json()
            if msg["type"] != "job" or msg["job"]["id"] != job_id:
                continue
            if msg["job"]["profile"]:
                profiles.append(msg["job"]["profile"])
            if msg["job"]["status"] == "done":
                break

    # While a stage runs it is listed as running; the last update is the finished profile.
    running = [
        [s["name"] for s in p["stages"] if s["status"] == "running"] for p in profiles
    ]
    assert ["fetch"] in running and ["export"] in running
    final = profiles[-1]
    assert final["finished"] is True
    assert [(s["name"], s["status"]) for s in final["stages"]] == [
        ("fetch", "ok"),
        ("export", "ok"),
    ]
    assert {"cpu_time_s", "peak_memory_mb", "memory_delta_mb"} <= set(
        final["stages"][0]
    )
    assert client.get(f"/api/jobs/{job_id}").json()["profile"] == final


def test_failed_job_keeps_its_profile(tmp_path):
    client = make_client(tmp_path, failing_pipeline)
    job_id = client.post("/api/jobs", json={"bbox": BBOX, "profile": True}).json()["id"]
    job = wait_for(client, job_id)

    assert job["status"] == "failed"
    stages = {s["name"]: s["status"] for s in job["profile"]["stages"]}
    assert stages == {"fetch": "error"}
    assert job["profile"]["finished"] is True
    report = client.get(f"/api/jobs/{job_id}/profile.json")
    assert report.status_code == 200
    assert report.json()["stages"][0]["error"] == "No images found."


def test_jobs_saved_before_live_profiles_still_load(tmp_path):
    client = make_client(tmp_path)
    job_id = wait_for(
        client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    )["id"]
    saved = tmp_path / job_id / "job.json"
    data = json.loads(saved.read_text())
    data.pop("profile")
    saved.write_text(json.dumps(data))

    job = make_client(tmp_path).get(f"/api/jobs/{job_id}").json()
    assert job["profile"] is None


# ---- the geometry evaluation (evaluate_geometry) ----


def evaluating_pipeline(result, seen=None):
    """fetch + export + an evaluate_geometry stage that writes like the real one."""

    def evaluate(value, state):
        if seen is not None:
            seen.update(state.metadata)
        if not state.get_metadata("evaluate_geometry", False):
            return value
        get_reporter(state).info("[lidar] 1,234 points kept")
        out = Path(state.metadata["evaluation_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_bytes(b"# LiDAR geometry evaluation\n")
        (out / "01_scorecard.png").write_bytes(b"png")
        (out / "lidar_points_abc.npz").write_bytes(b"cached points")
        state.set_metadata("geometry_evaluation", result)
        return value

    def factory():
        pipeline = fake_pipeline()
        pipeline.add_stage(
            "evaluate_geometry", evaluate, label="Evaluate geometry", detail="LiDAR"
        )
        return pipeline

    return factory


DONE = {
    "status": "done",
    "terrain": {"result": "PASS", "share": 0.97},
    "files": ["summary.md", "01_scorecard.png"],
}


def test_evaluation_is_off_by_default_and_skipped(tmp_path):
    seen = {}
    client = make_client(tmp_path, evaluating_pipeline(DONE, seen))
    created = client.post("/api/jobs", json={"bbox": BBOX}).json()
    assert created["options"]["evaluate_geometry"] is False
    job = wait_for(client, created["id"])

    assert job["status"] == "done"
    assert job["stages"]["evaluate_geometry"] == "skipped"
    assert seen["evaluate_geometry"] is False
    assert "evaluation" not in job["result"]
    assert client.get(f"/api/jobs/{job['id']}/evaluation").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/evaluation/summary.md").status_code == 404


def test_evaluation_results_and_files_are_served(tmp_path):
    seen = {}
    client = make_client(tmp_path, evaluating_pipeline(DONE, seen))
    created = client.post(
        "/api/jobs", json={"bbox": BBOX, "evaluate_geometry": True, "profile": True}
    ).json()
    job = wait_for(client, created["id"])
    jid = job["id"]

    assert job["status"] == "done" and job["stages"]["evaluate_geometry"] == "done"
    assert seen["evaluation_dir"] == str(tmp_path / jid / "geometry_eval")
    assert job["result"]["evaluation"] == DONE
    assert client.get(f"/api/jobs/{jid}/evaluation").json() == DONE
    # its progress went to the job log like any stage's
    log = client.get(f"/api/jobs/{jid}/log").text
    assert "── Evaluate geometry ──" in log and "[lidar] 1,234 points kept" in log
    # the profile counts it as a stage that ran
    stages = {s["name"]: s["status"] for s in job["result"]["profile"]["stages"]}
    assert stages["evaluate_geometry"] == "ok"

    report = client.get(f"/api/jobs/{jid}/evaluation/summary.md")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/markdown")
    assert report.text == "# LiDAR geometry evaluation\n"
    png = client.get(f"/api/jobs/{jid}/evaluation/01_scorecard.png?download=true")
    assert png.headers["content-type"] == "image/png"
    assert f"psm-{jid}-01_scorecard.png" in png.headers["content-disposition"]


def test_evaluation_serves_only_its_own_files(tmp_path):
    client = make_client(tmp_path, evaluating_pipeline(DONE))
    jid = wait_for(
        client,
        client.post("/api/jobs", json={"bbox": BBOX, "evaluate_geometry": True}).json()[
            "id"
        ],
    )["id"]
    base = f"/api/jobs/{jid}/evaluation"
    # the cached LiDAR exists but is not one of the outputs; a listed one not written
    assert client.get(f"{base}/lidar_points_abc.npz").status_code == 404
    assert client.get(f"{base}/per_building.csv").status_code == 404
    for name in ("..%2Fjob.json", "%2E%2E%2Fjob.json", "..%2F..%2Fsettings.json"):
        assert client.get(f"{base}/{name}").status_code == 404
    assert client.get("/api/jobs/nope/evaluation").status_code == 404


def test_an_unavailable_evaluation_keeps_the_job_done(tmp_path):
    reason = "[lidar] no USGS EPT collection covers this AOI; use --laz"
    client = make_client(
        tmp_path,
        evaluating_pipeline({"status": "unavailable", "reason": reason}),
    )
    job = wait_for(
        client,
        client.post("/api/jobs", json={"bbox": BBOX, "evaluate_geometry": True}).json()[
            "id"
        ],
    )
    assert job["status"] == "done"
    assert job["result"]["evaluation"] == {"status": "unavailable", "reason": reason}
    assert client.get(f"/api/jobs/{job['id']}/model.glb").status_code == 200


# ---- the texture evaluation (evaluate_texture) ----


def texture_pipeline(result, seen=None):
    """fetch + export + both evaluation stages; the texture one writes like the real one."""

    def evaluate_geometry(value, state):
        return value

    def evaluate_texture(value, state):
        if seen is not None:
            seen.update(state.metadata)
        if not state.get_metadata("evaluate_texture", False):
            return value
        get_reporter(state).info("[texture] O (obstruction) 0.250")
        out = Path(state.metadata["texture_evaluation_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_bytes(b"# UV texture quality\n")
        (out / "02_coverage.png").write_bytes(b"png")
        (out / "scratch.npz").write_bytes(b"not an output")
        state.set_metadata("texture_evaluation", result)
        return value

    def factory():
        pipeline = fake_pipeline()
        pipeline.add_stage("evaluate_geometry", evaluate_geometry, label="Geometry")
        pipeline.add_stage("evaluate_texture", evaluate_texture, label="Texture")
        return pipeline

    return factory


TEXTURE_DONE = {
    "status": "done",
    "confidence": 0.71,
    "partial": False,
    "terms": {"O": 0.25, "D": 0.0, "F": 0.5},
    "files": ["summary.md", "02_coverage.png"],
}


def test_texture_evaluation_is_off_by_default_and_skipped(tmp_path):
    seen = {}
    client = make_client(tmp_path, texture_pipeline(TEXTURE_DONE, seen))
    created = client.post("/api/jobs", json={"bbox": BBOX}).json()
    assert created["options"]["evaluate_texture"] is False
    job = wait_for(client, created["id"])

    assert job["status"] == "done"
    assert job["stages"]["evaluate_texture"] == "skipped"
    assert seen["evaluate_texture"] is False
    assert "texture_evaluation" not in job["result"]
    base = f"/api/jobs/{job['id']}/texture-evaluation"
    assert client.get(base).status_code == 404
    assert client.get(f"{base}/summary.md").status_code == 404


def test_texture_evaluation_results_and_files_are_served(tmp_path):
    seen = {}
    client = make_client(tmp_path, texture_pipeline(TEXTURE_DONE, seen))
    created = client.post(
        "/api/jobs", json={"bbox": BBOX, "evaluate_texture": True, "profile": True}
    ).json()
    job = wait_for(client, created["id"])
    jid = job["id"]

    assert job["status"] == "done" and job["stages"]["evaluate_texture"] == "done"
    # the geometry evaluation stays off and its result shape untouched
    assert job["stages"]["evaluate_geometry"] == "skipped"
    assert "evaluation" not in job["result"]
    assert seen["texture_evaluation_dir"] == str(tmp_path / jid / "texture_eval")
    assert job["result"]["texture_evaluation"] == TEXTURE_DONE
    base = f"/api/jobs/{jid}/texture-evaluation"
    assert client.get(base).json() == TEXTURE_DONE
    log = client.get(f"/api/jobs/{jid}/log").text
    assert "[texture] O (obstruction) 0.250" in log
    stages = {s["name"]: s["status"] for s in job["result"]["profile"]["stages"]}
    assert (
        stages["evaluate_texture"] == "ok" and stages["evaluate_geometry"] == "skipped"
    )

    report = client.get(f"{base}/summary.md")
    assert report.headers["content-type"].startswith("text/markdown")
    assert report.content == b"# UV texture quality\n"
    png = client.get(f"{base}/02_coverage.png?download=true")
    assert png.headers["content-type"] == "image/png"
    assert f"psm-{jid}-texture-02_coverage.png" in png.headers["content-disposition"]


def test_texture_evaluation_serves_only_its_own_files(tmp_path):
    client = make_client(tmp_path, texture_pipeline(TEXTURE_DONE))
    jid = wait_for(
        client,
        client.post("/api/jobs", json={"bbox": BBOX, "evaluate_texture": True}).json()[
            "id"
        ],
    )["id"]
    base = f"/api/jobs/{jid}/texture-evaluation"
    assert client.get(f"{base}/scratch.npz").status_code == 404  # not an output
    assert (
        client.get(f"{base}/03_stretch.png").status_code == 404
    )  # listed, not written
    for name in ("..%2Fjob.json", "%2E%2E%2Fjob.json", "..%2Fmodel.glb"):
        assert client.get(f"{base}/{name}").status_code == 404
    # the geometry endpoint doesn't serve the texture evaluation's files, nor vice versa
    assert client.get(f"/api/jobs/{jid}/evaluation/02_coverage.png").status_code == 404


def test_an_unavailable_texture_evaluation_keeps_the_job_done(tmp_path):
    reason = "model.glb has no textured building mesh to evaluate"
    client = make_client(
        tmp_path, texture_pipeline({"status": "unavailable", "reason": reason})
    )
    job = wait_for(
        client,
        client.post("/api/jobs", json={"bbox": BBOX, "evaluate_texture": True}).json()[
            "id"
        ],
    )
    assert job["status"] == "done"
    assert job["result"]["texture_evaluation"]["status"] == "unavailable"
    assert client.get(f"/api/jobs/{job['id']}/model.glb").status_code == 200


def _aligned_pipeline(roi):
    """A fake pipeline whose alignment used the ROI folder roi (as align_poses records it)."""

    def align(value, state):
        state.set_metadata("pose_alignment", {"roi_folder": str(roi), "added": 2})
        return value

    def factory():
        pipeline = PipelineChain("aligned")
        pipeline.add_stage("fetch", fetch, label="Fetch data")
        pipeline.add_stage("align_poses", align, label="Align poses")
        pipeline.add_stage("export", export, label="Export")
        return pipeline

    return factory


def test_deleting_a_job_removes_its_files_and_alignment_folder(tmp_path, monkeypatch):
    from pose_alignment import geo_common

    rois = tmp_path / "alignment"
    roi = rois / "mapillary_roi_1"
    (roi / "data" / "mapillary").mkdir(parents=True)
    (roi / "data" / "mapillary" / "frame_0001.jpg").write_bytes(b"x" * 5000)
    monkeypatch.setattr(geo_common, "ROIS_DIR", rois)
    client = make_client(tmp_path / "jobs", _aligned_pipeline(roi))

    job = wait_for(client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"])
    assert job["result"]["alignment_folder"] == str(roi)
    storage = client.get(f"/api/jobs/{job['id']}/storage").json()
    assert (
        storage["alignment_folder"] == str(roi) and storage["alignment_bytes"] == 5000
    )
    assert storage["alignment_kept"] is None and storage["job_bytes"] > 0

    deleted = client.delete(f"/api/jobs/{job['id']}")
    assert deleted.status_code == 200 and deleted.json()["not_removed"] == []
    assert not roi.exists() and not (tmp_path / "jobs" / job["id"]).exists()
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    # gone from the list of runs, now and after the app restarts
    assert job["id"] not in [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
    restarted = make_client(tmp_path / "jobs", _aligned_pipeline(roi))
    assert job["id"] not in [j["id"] for j in restarted.get("/api/jobs").json()["jobs"]]


def test_an_alignment_folder_another_job_uses_is_kept(tmp_path, monkeypatch):
    from pose_alignment import geo_common

    rois = tmp_path / "alignment"
    roi = rois / "mapillary_roi_2"
    roi.mkdir(parents=True)
    monkeypatch.setattr(geo_common, "ROIS_DIR", rois)
    client = make_client(tmp_path / "jobs", _aligned_pipeline(roi))

    first = wait_for(client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"])
    second = wait_for(
        client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    )
    storage = client.get(f"/api/jobs/{first['id']}/storage").json()
    assert second["id"] in storage["alignment_kept"]
    client.delete(f"/api/jobs/{first['id']}")
    assert roi.exists()  # still the second job's
    client.delete(f"/api/jobs/{second['id']}")
    assert not roi.exists()


def test_a_folder_outside_the_alignment_outputs_is_never_deleted(tmp_path, monkeypatch):
    from pose_alignment import geo_common

    monkeypatch.setattr(geo_common, "ROIS_DIR", tmp_path / "alignment")
    elsewhere = tmp_path / "my_photos" / "mapillary_roi_9"
    elsewhere.mkdir(parents=True)
    client = make_client(tmp_path / "jobs", _aligned_pipeline(elsewhere))

    job = wait_for(client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"])
    assert (
        "not an alignment folder"
        in client.get(f"/api/jobs/{job['id']}/storage").json()["alignment_kept"]
    )
    client.delete(f"/api/jobs/{job['id']}")
    assert elsewhere.exists()


def test_delete_photos_removes_only_the_downloaded_photos(tmp_path, monkeypatch):
    """With delete_photos, a finished job's Mapillary photos go; their metadata, the alignment's
    other outputs and the job's model, overlays and previews stay."""
    from pose_alignment import geo_common

    rois = tmp_path / "alignment"
    roi = rois / "mapillary_roi_3"
    seq = roi / "data" / "mapillary" / "line1_cam0_abc"
    (roi / "data" / "mapillary" / "originals").mkdir(parents=True)
    seq.mkdir(parents=True)
    (seq / "frame_0001.jpg").write_bytes(b"x" * 3000)
    (seq / "cameras.json").write_text("{}")
    (roi / "data" / "mapillary" / "originals" / "123.jpg").write_bytes(b"x" * 7000)
    (roi / "data" / "masks").mkdir(parents=True)
    (roi / "data" / "masks" / "frame_0001.png").write_bytes(b"m")
    monkeypatch.setattr(geo_common, "ROIS_DIR", rois)
    client = make_client(tmp_path / "jobs", _aligned_pipeline(roi))

    created = client.post("/api/jobs", json={"bbox": BBOX, "delete_photos": True}).json()
    assert created["options"]["delete_photos"] is True
    job = wait_for(client, created["id"])
    assert job["status"] == "done"
    assert job["result"]["photos_deleted"] == {"files": 2, "bytes": 10000}
    assert not (seq / "frame_0001.jpg").exists()
    assert not (roi / "data" / "mapillary" / "originals" / "123.jpg").exists()
    assert (seq / "cameras.json").exists() and (roi / "data" / "masks" / "frame_0001.png").exists()
    job_dir = tmp_path / "jobs" / job["id"]
    assert (job_dir / "model.glb").exists() and (job_dir / "images").exists()
    assert "Deleted 2 Mapillary photos" in client.get(f"/api/jobs/{job['id']}/log").text


def test_photos_are_kept_by_default_and_for_a_shared_folder(tmp_path, monkeypatch):
    from pose_alignment import geo_common

    rois = tmp_path / "alignment"
    roi = rois / "mapillary_roi_4"
    (roi / "data" / "mapillary").mkdir(parents=True)
    photo = roi / "data" / "mapillary" / "frame_0001.jpg"
    photo.write_bytes(b"x" * 100)
    monkeypatch.setattr(geo_common, "ROIS_DIR", rois)
    client = make_client(tmp_path / "jobs", _aligned_pipeline(roi))

    first = wait_for(client, client.post("/api/jobs", json={"bbox": BBOX}).json()["id"])
    assert first["options"]["delete_photos"] is False and photo.exists()
    # the second job asks to delete them, but the first job uses the same folder
    second = wait_for(
        client,
        client.post("/api/jobs", json={"bbox": BBOX, "delete_photos": True}).json()["id"],
    )
    assert second["status"] == "done" and "photos_deleted" not in second["result"]
    assert photo.exists()


def test_delete_photos_default_can_be_saved(tmp_path):
    client = make_client(tmp_path)
    saved = client.put("/api/settings", json={"defaults": {"delete_photos": True}}).json()
    assert saved["defaults"]["delete_photos"] is True
