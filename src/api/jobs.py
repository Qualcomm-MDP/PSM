"""Background job runner for the web API.

Jobs run one at a time on a single worker thread: the segmentation and
inpainting models are large, so running several pipelines at once would just
compete for the same CPU/GPU memory. Each job keeps its files in
<data_dir>/<job_id>/ (job.json, pipeline.log, model.glb, images/), so finished
jobs survive a restart.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.joblog import JobReporter
from common import PipelineChain, ProgressMonitor
from common.constants import BoundingBox
from common.redact import redact

# Stages that only do something when their job option is on.
OPTIONAL_STAGES = {"terrain": "terrain", "naip": "naip"}

SEG_MODELS = {
    # Mapillary Vistas: trained on Mapillary's own crowdsourced photos (no Tiny exists)
    "large": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "tiny": "facebook/mask2former-swin-tiny-cityscapes-semantic",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    name: str
    bbox: dict[str, float]  # west, south, east, north
    options: dict[str, Any]
    stages: dict[str, str]  # stage name -> pending | running | done | error
    status: str = "queued"  # queued | running | done | failed
    error: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    def __init__(
        self,
        data_dir: Path,
        pipeline_factory: Callable[[], PipelineChain],
        publish: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """publish receives live messages for browsers:
        {"type": "job", "job": {...}} after every job change, and
        {"type": "log" | "progress", "job_id": ..., ...} for pipeline output.
        It's called from the worker thread, so it must be quick and thread-safe."""
        self.data_dir = Path(data_dir)
        self._publish = publish
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline_factory = pipeline_factory
        self.stages = pipeline_factory().describe()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._load_existing()
        self._worker = threading.Thread(target=self._run_forever, daemon=True)
        self._worker.start()

    # ---- public API ----------------------------------------------------

    def submit(self, name: str, bbox: dict[str, float], options: dict[str, Any]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            name=name,
            bbox=bbox,
            options=options,
            stages={
                s["name"]: "skipped" if _turned_off(s["name"], options) else "pending"
                for s in self.stages
            },
        )
        with self._lock:
            self._jobs[job.id] = job
            self._save(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def alignment_folder(self, job: Job) -> Path | None:
        """The pose alignment's folder this job made (outputs/alignment/mapillary_roi_N): from its
        result, or else its log ("alignment folder: ..." when the stage made it, or the alignment
        pipeline's "ROI folder ..." line in jobs from before)."""
        folder = (job.result or {}).get("alignment_folder")
        if not folder:
            log = self.job_dir(job.id) / "pipeline.log"
            if log.exists():
                m = re.search(
                    r"(?:ROI folder|alignment folder:) (\S*mapillary_roi_\d+)",
                    log.read_text(encoding="utf-8", errors="replace"),
                )
                folder = m.group(1) if m else None
        return Path(folder) if folder else None

    def storage(self, job_id: str) -> dict[str, Any]:
        """What deleting the job removes from this computer, and how big it is."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        out: dict[str, Any] = {
            "job_folder": str(self.job_dir(job.id)),
            "job_bytes": _folder_bytes(self.job_dir(job.id)),
            "alignment_folder": None,
            "alignment_bytes": 0,
            "alignment_kept": None,
        }
        folder = self.alignment_folder(job)
        if folder is not None and folder.exists():
            out["alignment_folder"] = str(folder)
            out["alignment_bytes"] = _folder_bytes(folder)
            out["alignment_kept"] = self._alignment_kept(job, folder)
        return out

    def delete(self, job_id: str) -> dict[str, Any]:
        """Remove a finished job and its files: its folder here (model, previews, log) and its pose
        alignment folder (downloaded photos, masks, reconstructions), unless another job uses that
        folder or it is not an alignment ROI folder. A queued or running job cannot be deleted
        (ValueError). Returns what was removed (storage())."""
        info = self.storage(job_id)
        with self._lock:
            job = self._jobs[job_id]
            if job.status in ("queued", "running"):
                raise ValueError("A queued or running job cannot be deleted.")
            del self._jobs[job_id]
        # job.json first: whatever else fails, the job no longer comes back when the app restarts
        (self.job_dir(job_id) / "job.json").unlink(missing_ok=True)
        left = []
        folders = [self.job_dir(job_id)]
        if info["alignment_folder"] and not info["alignment_kept"]:
            folders.append(Path(info["alignment_folder"]))
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
            if folder.exists():
                left.append(str(folder))
        if self._publish:
            self._publish({"type": "job_deleted", "id": job_id})
        info["not_removed"] = left
        return info

    def _alignment_kept(self, job: Job, folder: Path) -> str | None:
        """Why the alignment folder must stay (None: it can go with the job)."""
        from pose_alignment.geo_common import ROIS_DIR

        try:
            inside = folder.resolve().parent == Path(ROIS_DIR).resolve()
        except OSError:
            inside = False
        if not inside or not folder.name.startswith("mapillary_roi_"):
            return f"not an alignment folder under {ROIS_DIR}"
        with self._lock:
            others = [j for j in self._jobs.values() if j.id != job.id]
        users = [j.id for j in others if self.alignment_folder(j) == folder]
        return f"also used by job {', '.join(users)}" if users else None

    # ---- internals -----------------------------------------------------

    def _save(self, job: Job) -> None:
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        data = job.to_dict()
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(d / "job.json")
        if self._publish:
            self._publish({"type": "job", "job": data})

    def _update(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for k, v in changes.items():
                setattr(job, k, v)
            self._save(job)

    def _load_existing(self) -> None:
        for path in self.data_dir.glob("*/job.json"):
            try:
                job = Job(**json.loads(path.read_text()))
            except (OSError, TypeError, ValueError):
                continue
            if job.status in ("queued", "running"):
                # The process stopped mid-run; that work is lost.
                job.status = "failed"
                job.error = "The server restarted before this job finished."
                job.stages = {
                    k: ("error" if v == "running" else v) for k, v in job.stages.items()
                }
                self._save(job)
            self._jobs[job.id] = job

    def _run_forever(self) -> None:
        # This is the only worker, so nothing a single job does (e.g. a disk
        # error writing its log or job.json) may stop it: later jobs would
        # stay queued forever.
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                if job is not None:
                    self._run(job)
            except Exception as exc:
                print(f"[{job_id}] Job crashed outside the pipeline:", flush=True)
                print(redact(traceback.format_exc()), flush=True)
                self._mark_crashed(job_id, exc)
            finally:
                self._queue.task_done()

    def _mark_crashed(self, job_id: str, exc: Exception) -> None:
        """Best effort: record the failure; the job may not even be writable."""
        job = self.get(job_id)
        if job is None or job.status in ("done", "failed"):
            return
        try:
            self._update(
                job,
                status="failed",
                error=_describe_error(exc),
                finished_at=_now(),
                stages={
                    k: ("error" if v == "running" else v) for k, v in job.stages.items()
                },
            )
        except Exception:
            # Can't save it; at least tell connected browsers.
            job.status = "failed"
            if self._publish:
                self._publish({"type": "job", "job": job.to_dict()})

    def _run(self, job: Job) -> None:
        job_dir = self.job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        model_path = job_dir / "model.glb"

        def emit(message: dict[str, Any]) -> None:
            if self._publish:
                self._publish({**message, "job_id": job.id})

        log = JobReporter(job_dir / "pipeline.log", emit, prefix=f"[{job.id}] ")
        started = time.monotonic()
        self._update(job, status="running", started_at=_now())
        # Finish and close the log before the final status goes out, so a page
        # that reloads the log when the job ends gets every line.
        try:
            outcome = self._run_pipeline(job, job_dir, model_path, log)
            elapsed = time.monotonic() - started
            if outcome["status"] == "done":
                log.info(f"Done in {elapsed:.0f}s.")
            else:
                log.info(f"Failed after {elapsed:.0f}s: {outcome['error']}")
        finally:
            log.close()
        self._update(job, finished_at=_now(), **outcome)

    def _run_pipeline(
        self, job: Job, job_dir: Path, model_path: Path, log: JobReporter
    ) -> dict[str, Any]:
        """Run the pipeline and return the job's final status fields."""
        b = job.bbox
        log.info(
            f"Job {job.id} · {job.name} · bbox W {b['west']:.5f} S {b['south']:.5f} "
            f"E {b['east']:.5f} N {b['north']:.5f}"
        )
        log.info("Options: " + ", ".join(f"{k}={v}" for k, v in job.options.items()))
        labels = {s["name"]: s["label"] for s in self.stages}

        def on_stage(name: str, event: str) -> None:
            if job.stages.get(name) == "skipped":
                return  # the stage runs but does nothing for this job
            if event == "start":
                log.info(f"── {labels.get(name, name)} ──")
            state = {"start": "running", "done": "done", "error": "error"}[event]
            with self._lock:
                if name in job.stages:
                    job.stages[name] = state
                self._save(job)

        bbox = BoundingBox(
            min_lat=b["south"], max_lat=b["north"], min_lon=b["west"], max_lon=b["east"]
        )
        opts = job.options
        try:
            state = self._pipeline_factory().run(
                bbox,
                metadata={
                    "bbox": bbox,
                    "progress_monitor": ProgressMonitor(verbose=False),
                    # jobs from before the option existed used Mapillary's raw GPS
                    "pose_source": opts.get("poses", "raw"),
                    "aligned_only": opts.get("aligned_only", False),
                    "alignment_options": {},
                    "remove_obstructions": opts.get("remove_obstructions", True),
                    "fill_obstructions": opts.get("fill_obstructions", True),
                    "sky_only": opts.get("sky_only", False),
                    "seg_model": SEG_MODELS.get(opts.get("seg_model", "large")),
                    "seg_device": opts.get("device", "auto"),
                    "use_terrain": opts.get("terrain", False),
                    "use_naip": opts.get("naip", False),
                    "output_path": str(model_path),
                    "image_dir": str(job_dir / "images"),
                    "reporter": log,
                },
                on_stage=on_stage,
            )
        except Exception as exc:
            log.error(traceback.format_exc())
            return {"status": "failed", "error": _describe_error(exc)}

        return {"status": "done", "result": _summarize(state, job_dir)}


def _turned_off(stage: str, options: dict[str, Any]) -> bool:
    """Whether an optional stage is off for a job (shown as skipped)."""
    option = OPTIONAL_STAGES.get(stage)
    return option is not None and not options.get(option, False)


def _describe_error(exc: Exception) -> str:
    msg = redact(str(exc)).strip() or type(exc).__name__
    return msg if len(msg) <= 300 else msg[:297] + "..."


def _summarize(state: Any, job_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # what the user should know about this result (common.reporter.notify)
    notices = state.get_metadata("notices", None) or []
    if notices:
        result["warnings"] = list(notices)
    fetched = state.stage_outputs.get("fetch") or {}
    elements = (fetched.get("osm") or {}).get("elements", [])
    result["buildings"] = sum(1 for e in elements if "geometry" in e)
    result["photos"] = len(fetched.get("mapillary") or {})
    aligned = state.get_metadata("pose_alignment", None)
    if aligned and aligned.get("roi_folder"):
        result["alignment_folder"] = aligned["roi_folder"]
    if aligned:
        result["aligned_photos"] = aligned.get("overridden", 0) + aligned.get(
            "added", 0
        )
    mesh = state.stage_outputs.get("texturing")
    faces = getattr(mesh, "faces", None)
    if faces is not None:
        result["faces"] = int(len(faces))
    model_path = job_dir / "model.glb"
    if model_path.exists():
        result["model_bytes"] = model_path.stat().st_size
    index = job_dir / "images" / "images.json"
    if index.exists():
        try:
            images = json.loads(index.read_text())
        except ValueError:
            images = []
        if images:
            result["thumbnail"] = images[0].get("cleaned") or images[0]["original"]
    return result


def _folder_bytes(folder: Path) -> int:
    """Total size of the files under a folder (0 if it is missing)."""
    if not folder.exists():
        return 0
    return sum(
        f.stat().st_size
        for f in folder.rglob("*")
        if f.is_file() and not f.is_symlink()
    )
