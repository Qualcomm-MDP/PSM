"""Evaluations of the pipeline's output.

geometry_lidar compares the exported model's buildings and terrain with USGS 3DEP
airborne LiDAR (command line: scripts/eval/eval_geometry_lidar.py). The pipeline runs
it after export as its optional stage evaluate_geometry, when "evaluate_geometry" is set
(the web app's "Evaluate geometry against LiDAR" option, the CLI's
--evaluate-geometry).

texture_uv scores the buildings' UV texture (the "UV Mapping Texture Quality Metric":
obstruction, distortion and FID; command line: scripts/eval/eval_texture_uv.py). The
pipeline runs it as its optional last stage, evaluate_texture, when "evaluate_texture"
is set (the web app's "Evaluate texture quality" option, the CLI's --evaluate-texture).
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from common.redact import redact
from common.reporter import CONSOLE, Reporter, get_reporter

EVALUATION_DIR = "geometry_eval"  # next to the .glb, as the command line writes it
TEXTURE_EVALUATION_DIR = "texture_eval"  # texture_uv.EVALUATION_DIR
# texture_uv.MASK_SUFFIX: texturing's photo texels, saved next to the .glb by the export
TEXTURE_MASK_SUFFIX = ".texture_mask.png"
# texture_uv.OUTPUT_FILES: the texture evaluation's files the web app may serve
TEXTURE_OUTPUT_FILES = (
    "summary.md",
    "01_scorecard.png",
    "02_coverage.png",
    "03_stretch.png",
    "per_group.csv",
    "summary.json",
)
# What run_evaluation writes, in reading order: the files the web app may serve
# (lidar_points_*.npz, the cached LiDAR, and terrain_grid.npz are not).
OUTPUT_FILES = (
    "summary.md",
    "01_scorecard.png",
    "02_height_error_by_source.png",
    "03_height_scatter.png",
    "04_placement_vs_slope.png",
    "05_height_error_map.png",
    "per_building.csv",
    "per_building_errors.geojson",
    "summary.json",
    "buildings_footprints_utm.geojson",
    "osm_buildings.json",
)


def evaluate_model(
    glb: str | Path,
    bbox: tuple[float, float, float, float],
    *,
    out: str | Path | None = None,
    osm: dict[str, Any] | None = None,
    grid: Any = None,
    reporter: Reporter | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Evaluate a model .glb built for `bbox` (west, south, east, north); never raises.

    Returns the compact summary (geometry_lidar.web_summary) with status "done", or
    {"status": "unavailable" | "failed", "reason": ...} when it can't run here (no
    USGS LiDAR for the area, laspy not installed, no internet) or crashed. Progress
    goes to `reporter` (the job log), or is printed without one. `overrides` are
    command-line settings (geometry_lidar.build_parser), e.g. terrain_tol=0.5."""
    out = Path(out) if out is not None else Path(glb).parent / EVALUATION_DIR
    try:
        import requests

        from evaluation import geometry_lidar as gl
    except ImportError as exc:  # its numerical packages are the project's own
        return _not_run("unavailable", f"Missing package: {exc.name}", reporter)

    token = gl._REPORTER.set(reporter)
    try:
        a = gl.settings(bbox=list(bbox), **overrides)
        return gl.run_evaluation(a, out, glb=Path(glb), osm=osm, grid=grid)
    except gl.Unavailable as exc:
        return _not_run("unavailable", _reason(exc), reporter)
    except requests.RequestException as exc:
        return _not_run(
            "unavailable",
            f"Couldn't download the LiDAR or elevation data (no internet, or the "
            f"server is down): {exc}",
            reporter,
        )
    except SystemExit as exc:  # the command line's stops: input it can't evaluate
        return _not_run("failed", _reason(exc), reporter)
    except Exception as exc:  # noqa: BLE001 - the model is written; keep the job
        if reporter is not None:
            reporter.error(redact(traceback.format_exc()))
        return _not_run("failed", f"{type(exc).__name__}: {exc}", reporter)
    finally:
        gl._REPORTER.reset(token)


def _reason(exc: SystemExit) -> str:
    return str(exc.code if exc.code is not None else "stopped").strip()


def _not_run(
    status: str, reason: str, reporter: Reporter | None, name: str = "Geometry"
) -> dict[str, Any]:
    reason = redact(reason)
    if len(reason) > 400:
        reason = reason[:397] + "..."
    what = "not available" if status == "unavailable" else "failed"
    message = f"{name} evaluation {what}: {reason}. The model is kept."
    if reporter is not None:
        reporter.warning(message)
    else:
        print(message)
    return {"status": status, "reason": reason}


def evaluate_geometry_stage(value, state):
    """Pipeline stage after export: evaluate the written .glb against LiDAR when
    "evaluate_geometry" is on. Its result (evaluate_model) goes to the
    "geometry_evaluation" metadata; the job never fails because of it."""
    if not state.get_metadata("evaluate_geometry", False):
        return value
    reporter = get_reporter(state)
    path = state.get_metadata("output_path", None)
    if not path or not Path(path).exists():
        result = _not_run(
            "unavailable",
            "no .glb was written",
            None if reporter is CONSOLE else reporter,
        )
    else:
        b = state.require_metadata("bbox")
        result = evaluate_model(
            path,
            (b.min_lon, b.min_lat, b.max_lon, b.max_lat),
            out=state.get_metadata("evaluation_dir", None),
            osm=(state.get_metadata("provider_data", None) or {}).get("osm"),
            grid=state.get_metadata("terrain", None),
            # the console: the command line's own printed output, as when run alone
            reporter=None if reporter is CONSOLE else reporter,
        )
    state.set_metadata("geometry_evaluation", result)
    return value


def evaluate_texture_model(
    glb: str | Path,
    *,
    out: str | Path | None = None,
    reporter: Reporter | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Score a model .glb's UV texture (texture_uv); never raises.

    Returns the compact summary (texture_uv.web_summary) with status "done", or
    {"status": "unavailable" | "failed", "reason": ...} when the model has nothing to
    evaluate or the evaluation crashed. A missing Inception-v3 (no torch / torchvision,
    or its weights can't be downloaded) is not a failure: F is left out and the result
    is "partial". `overrides` are command-line settings (texture_uv.build_parser), e.g.
    tau=50."""
    out = Path(out) if out is not None else Path(glb).parent / TEXTURE_EVALUATION_DIR
    try:
        from evaluation import texture_uv as tu
    except ImportError as exc:
        return _not_run(
            "unavailable", f"Missing package: {exc.name}", reporter, "Texture"
        )

    token = tu._REPORTER.set(reporter)
    try:
        return tu.run(tu.settings(**overrides), Path(glb), out)
    except tu.Unavailable as exc:
        return _not_run("unavailable", _reason(exc), reporter, "Texture")
    except Exception as exc:  # noqa: BLE001 - the model is written; keep the job
        if reporter is not None:
            reporter.error(redact(traceback.format_exc()))
        return _not_run("failed", f"{type(exc).__name__}: {exc}", reporter, "Texture")
    finally:
        tu._REPORTER.reset(token)


def evaluate_texture_stage(value, state):
    """Pipeline stage after export: score the written .glb's UV texture when
    "evaluate_texture" is on. Its result (evaluate_texture_model) goes to the
    "texture_evaluation" metadata; the job never fails because of it."""
    if not state.get_metadata("evaluate_texture", False):
        return value
    reporter = get_reporter(state)
    quiet = None if reporter is CONSOLE else reporter
    path = state.get_metadata("output_path", None)
    if not path or not Path(path).exists():
        result = _not_run("unavailable", "no .glb was written", quiet, "Texture")
    else:
        overrides = {}
        device = state.get_metadata("seg_device", None)
        if device and device != "gpu":
            overrides["device"] = device
        result = evaluate_texture_model(
            path,
            out=state.get_metadata("texture_evaluation_dir", None),
            reporter=quiet,
            **overrides,
        )
    state.set_metadata("texture_evaluation", result)
    return value
