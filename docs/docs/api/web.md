---
title: Web API
sidebar_position: 6
---

# Web API

The web app's backend (`src/api/app.py`, FastAPI) serves the frontend from `web/` and the API below. Interactive docs for the running server are at `/api/docs`.

Start it with `docker compose up api`, or from source:

```bash
poetry run uvicorn api.app:create_app --factory --app-dir src
```

## Jobs

| Method | Path                               | Description                                                                                      |
| ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| GET    | `/api/jobs`                        | All jobs, newest first: `{"jobs": [Job, ...]}`                                                   |
| POST   | `/api/jobs`                        | Start a job (see below). Returns the new `Job` with status `201`.                                |
| GET    | `/api/jobs/{id}`                   | One job.                                                                                         |
| GET    | `/api/jobs/{id}/model.glb`         | The model, once the job is done. `?download=true` sends it as a file named after the job.       |
| GET    | `/api/jobs/{id}/log`               | The pipeline log as plain text, one timestamped line per entry. `?download=true` sends a file.  |
| GET    | `/api/jobs/{id}/profile.json`      | The full profiler report of a job run with `profile` (updated after each stage while it runs). `?download=true` sends it as a file. `404` without a profile. |
| GET    | `/api/jobs/{id}/evaluation`        | The geometry evaluation's results of a job run with `evaluate_geometry` (`result.evaluation`). `404` without one. |
| GET    | `/api/jobs/{id}/evaluation/{name}` | One file the evaluation wrote: `summary.md`, `summary.json`, `per_building.csv`, `per_building_errors.geojson`, the charts `01_scorecard.png` to `05_height_error_map.png`, `buildings_footprints_utm.geojson` or `osm_buildings.json`. `?download=true` sends it as a file. `404` for any other name. |
| GET    | `/api/jobs/{id}/images`            | The saved photo previews: `{"images": [{"id", "heading", "original", "cleaned"}]}` (URLs).      |
| GET    | `/api/jobs/{id}/images/{name}`     | One preview image (JPEG).                                                                        |

### Starting a job

`POST /api/jobs` with:

```json
{
  "bbox": { "west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422 },
  "name": "North Campus",
  "remove_obstructions": true,
  "seg_model": "large",
  "device": "auto",
  "terrain": false,
  "naip": false,
  "profile": false,
  "evaluate_geometry": false
}
```

| Field                 | Default         | Values                                                                   |
| --------------------- | --------------- | ------------------------------------------------------------------------ |
| `bbox`                | *(required)*    | degrees; west < east, south < north, each side at most 2500 m            |
| `name`                | `"Custom area"` | up to 120 characters                                                     |
| `remove_obstructions` | `true`          | run Mask2Former + LaMa                                                   |
| `seg_model`           | `"large"`       | `"large"` (Swin-Large) or `"tiny"` (Swin-Tiny)                           |
| `device`              | `"auto"`        | `"auto"`, `"cpu"` or `"cuda"` (`422` if the server has no CUDA GPU)      |
| `terrain`             | `false`         | the `--terrain` stage                                                    |
| `naip`                | `false`         | the `--naip` stage; turns `terrain` on                                   |
| `profile`             | `false`         | measure each stage (the CLI's `--profile`): `job.profile` and `/profile.json` |
| `evaluate_geometry`   | `false`         | compare the model with USGS LiDAR after export (the CLI's `--evaluate-geometry`): `result.evaluation` and `/evaluation` |

Invalid requests get `422` with a message in `detail`.

### The Job object

```json
{
  "id": "7c536a830e38",
  "name": "North Campus",
  "bbox": { "west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422 },
  "options": { "remove_obstructions": true, "seg_model": "large", "device": "auto", "terrain": false, "naip": false },
  "stages": { "fetch": "done", "terrain": "skipped", "build_mesh": "done", "naip": "skipped", "texturing": "running", "export": "pending" },
  "status": "running",
  "error": null,
  "created_at": "2026-11-03T17:18:35+00:00",
  "started_at": "2026-11-03T17:18:35+00:00",
  "finished_at": null,
  "result": {},
  "profile": null
}
```

- `status`: `queued`, `running`, `done` or `failed`.
- `stages`: each stage is `pending`, `running`, `done`, `error` or `skipped` (an optional stage that's off for this job).
- `result`, once done: `buildings`, `photos`, `faces`, `model_bytes`, and `thumbnail` (a preview image name). A profiled job also has a copy of its final `profile` there.
- `result.evaluation`, for a job run with `evaluate_geometry`: `{"status": "unavailable" | "failed", "reason": ...}` when the evaluation couldn't run here or failed (the job is still `done`), otherwise:

  | Field       | Meaning                                                                                                                             |
  | ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
  | `status`    | `"done"`                                                                                                                             |
  | `terrain`   | QR-06a: `result` (`PASS`, `FAIL`, `INCONCLUSIVE` or `NOT TESTED`), `text`, `tol_m`, `target`, and `share`, `ci95`, `hits`, `n` when tested |
  | `height`    | QR-06b: `tol_m`, `text`, `all` and `groups.high` / `medium` / `low` (each `share`, `ci95`, `hits`, `n`)                               |
  | `medians`   | per metric (`M1_height_err`, `M2_terrain_err`, `M2_base_err`, `M3_roof_err`, `M3_roof_err_lowground`): `n`, `median`, `rmse`, `nmad` |
  | `buildings` | `total` OSM buildings, `evaluated`, and `by_status` (`ok`, `not_in_mesh`, `not_in_lidar`, `too_small`, `no_ground`, ...)            |
  | `lidar`     | the LiDAR used: `source`, `url`, `vertical`                                                                                          |
  | `files`     | the files written, for `/evaluation/{name}`                                                                                          |
- `profile`: `null` unless the job was started with `profile`. Otherwise it's updated (and pushed over the WebSocket as a `job` message) when each stage starts and ends, and kept when the job fails:

  | Field               | Meaning                                                                    |
  | ------------------- | -------------------------------------------------------------------------- |
  | `finished`          | `false` while the job runs                                                 |
  | `started_at`, `finished_at` | ISO times; `finished_at` is `null` until the run ends              |
  | `total_wall_time_s` | the run's wall time, so far or in total                                    |
  | `cpu_count`         | the server's CPU cores                                                     |
  | `gpu`               | the NVIDIA GPU's name, or `null`                                           |
  | `stages`            | one entry per stage that has started, in run order                         |

  Each stage has `name`, `status` (`running`, `ok`, `skipped` or `error`), `wall_time_s` (for a running stage, its time so far), `cpu_time_s`, `peak_memory_mb`, `memory_delta_mb`, `gpu_delta_allocated_mb` (`"N/A"` without an NVIDIA GPU) and `error`. `/profile.json` has the full report (see [Evaluation](../pipeline/evaluation.md#pipeline-profiler)).

Jobs run one at a time. A job that was running when the server stopped is marked `failed` when it starts again.

## Live updates

`/api/ws` is a WebSocket. The client doesn't send anything; the server sends JSON messages:

| `type`     | Fields                    | When                                                                                         |
| ---------- | ------------------------- | -------------------------------------------------------------------------------------------- |
| `snapshot` | `jobs`                    | Right after connecting: every job.                                                           |
| `job`      | `job`                     | A job was created, or its status or a stage changed (including its `profile`).               |
| `log`      | `job_id`, `n`, `line`     | A new pipeline log line. `n` counts from 1, so lines already loaded from `/log` can be skipped. |
| `progress` | `job_id`, `text`          | A progress bar moved (e.g. photo downloads). An empty `text` means it finished.             |

A client that falls too far behind (1000 queued messages) is disconnected; reconnecting gets a fresh `snapshot`.

## Settings

| Method | Path                              | Description                                                                                                   |
| ------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| GET    | `/api/settings`                   | Token status and job defaults (never the token itself).                                                       |
| PUT    | `/api/settings`                   | Change them: `mapillary_token`, `clear_mapillary_token`, `defaults`. Returns the settings plus `token_saved_to`. |
| GET    | `/api/settings/mapillary/token`   | The current token, for the Settings dialog: `{"token": ...}` (not cached).                                   |
| POST   | `/api/settings/mapillary/test`    | Check a token with Mapillary: `{"token": ...}`, or `{}` for the current one. Returns `{"ok", "message"}`.    |

`GET /api/settings` returns:

```json
{
  "mapillary": {
    "set": true,
    "source": "env",
    "hint": "…ab12",
    "env_available": true,
    "env_file": ".env.development",
    "env_file_exists": true
  },
  "defaults": { "remove_obstructions": true, "seg_model": "large", "device": "auto" }
}
```

`source` is `"saved"` (saved in the web app), `"env"` (from the env file or environment) or `null`. `token_saved_to` is `"env_file"` when saving created the env file, otherwise `"settings"`.

## Other endpoints

| Method | Path                                  | Description                                                                                         |
| ------ | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| GET    | `/api/pipeline`                       | The pipeline's stages: `{"stages": [{"name", "label", "detail"}]}`.                                 |
| GET    | `/api/system`                         | `{"cuda", "seg_models", "max_side_m"}`.                                                             |
| GET    | `/api/geocode?q=...`                  | Full address search (Nominatim). At least 3 characters.                                             |
| GET    | `/api/suggest?q=...&lat=...&lon=...`  | Suggestions while typing (Photon), ranked towards `lat`/`lon` when given.                           |

Both searches return `{"results": [...]}`, each with a name, address, `lat`, `lon` and a `bbox`.
