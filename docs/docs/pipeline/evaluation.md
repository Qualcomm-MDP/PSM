# Evaluation

PSM is evaluated on speed and resource use, measured by the built-in profiler, and on output quality, assessed with user surveys.

## Pipeline profiler

`PipelineProfiler` (`src/common/profiler.py`) measures each stage. Turn it on from the command line or the web app.

**Command line:** `--profile PATH` writes the report to `PATH` when the run ends, and logs a summary table:

```bash
poetry run python src/main.py --bbox W S E N --output map.glb --profile results.json
```

**Web app:** **Profile this run** writes the report to `profile.json` in the job's folder (`output/jobs/<job id>/`). It's updated after every stage while the job runs, and kept when a job fails. The job's details show it as charts and a table, live; see [Profiling a job](../web-app.md#profiling-a-job). It's also served at `/api/jobs/<job id>/profile.json` (see [Web API](../api/web.md)).

The summary table goes through the pipeline's reporter, like the rest of its output: to the console on the command line, and into the job's log in the web app.

Memory is measured with Python's `tracemalloc`: it sees Python and NumPy allocations, not the whole process (for example not PyTorch's own buffers). Tracing every allocation makes a profiled run slower.

### Metrics per stage

| Metric                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `wall_time_s`            | Real elapsed time (includes waiting on downloads) |
| `cpu_time_s`             | CPU time used                                     |
| `peak_memory_mb`         | Peak Python memory traced during the stage        |
| `memory_delta_mb`        | Memory at the end minus at the start              |
| `gpu_delta_allocated_mb` | Change in allocated GPU memory (NVIDIA only)      |
| `status`                 | `ok`, `skipped` (optional stage was off), or `error` |
| `error`                  | The error message, for a stage that failed        |

### Output

The JSON report has `pipeline`, `started_at`, `finished_at` (empty until the run ends), `total_wall_time_s` (up to the end of the run), `system` (Python, OS, CPU and GPU), and `stages`, one entry per stage with the metrics above. Today's pipeline reports `fetch`, `align_poses`, `terrain`, `build_mesh`, `naip`, `texturing`, and `export`; optional stages that were off (`terrain` without `--terrain`, `naip` without `--naip`) are `skipped`. The example below is from **Design Review 1**, when there were four stages.

```json
{
  "pipeline": "photorealistic-street-maps",
  "started_at": "2026-03-25T18:00:00Z",
  "total_wall_time_s": 405.3,
  "system": {
    "python": "3.12.0",
    "os": "Linux-6.1.0-x86_64",
    "cpu_name": "x86_64",
    "cpu_count": 8,
    "gpu": {
      "available": true,
      "name": "NVIDIA RTX 4090",
      "total_vram_mb": 24564.0,
      "count": 1
    }
  },
  "stages": [
    { "name": "fetch", "wall_time_s": 7.607, "cpu_time_s": 0.073, "peak_memory_mb": 2.5, "status": "ok" },
    { "name": "build_mesh", "wall_time_s": 0.196, "cpu_time_s": 0.196, "peak_memory_mb": 0.6, "status": "ok" },
    { "name": "texturing", "wall_time_s": 377.8, "cpu_time_s": 286.2, "peak_memory_mb": 12440.0, "status": "ok" },
    { "name": "export", "wall_time_s": 17.483, "cpu_time_s": 3.316, "peak_memory_mb": 283.2, "status": "ok" }
  ]
}
```

```
Stage                 Wall(s)   CPU(s)   Peak MB  GPU Alloc MB   Status
------------------------------------------------------------------------
fetch                    7.61     0.07       2.5           0.0       ok
build_mesh               0.20     0.20       0.6           0.0       ok
texturing              377.80   286.20   12440.0        1060.0       ok
export                  17.48     3.32     283.2           0.0       ok
------------------------------------------------------------------------
TOTAL                  405.30
```

## Geometry evaluation against LiDAR

`src/evaluation/geometry_lidar.py` measures how close the exported model is to the real buildings and ground, using USGS 3DEP airborne LiDAR (US only) as the reference. It reads the geometry straight from the `.glb`: each building's roof and base from the mesh, and the ground from the model's terrain. OSM only splits the mesh into buildings and gives each its id, name and height tags.

| Metric                 | What it measures (model − LiDAR, metres)                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **M1** height          | model height − LiDAR height (roof top above the lowest ground at the footprint's edge)                                 |
| **M2** terrain         | model ground − LiDAR ground around the building (median per building)                                                  |
| Base placement         | the model's lowest ground under the footprint − the LiDAR's                                                            |
| **M3** roof elevation  | the model's roof − the LiDAR roof top, in absolute elevation: what the viewer sees                                     |

The LiDAR roof top is the highest roof plane found by RANSAC, so trees, antennas and rooftop equipment are ignored. Results are judged against draft requirement QR-06:

- **QR-06a terrain (pass/fail):** at least 95% of buildings within ±1 m of the LiDAR ground, at the lower end of the 95% confidence interval. Fewer than 30 evaluated buildings is inconclusive.
- **QR-06b height (baseline):** the share within ±3 m, split by height confidence: `height` tag (high), `building:levels` (medium) or the default (low). `--height-baseline` compares a run with an earlier one.

### Running it

It needs [laspy](https://laspy.readthedocs.io/) to read the LiDAR tiles. It's in the optional `eval` group, and the Docker image includes it:

```bash
poetry install --with eval
```

- **Web app:** turn on **Evaluate geometry against LiDAR** (see [Web App](../web-app.md#evaluating-geometry)). It runs as the pipeline's last stage, `evaluate_geometry`, and writes into the job's `geometry_eval/` folder.
- **Command line, with the pipeline:** `poetry run python src/main.py --bbox ... --terrain --evaluate-geometry` writes `geometry_eval/` next to the `.glb`.
- **Command line, on a finished model:**

```bash
poetry run python scripts/eval/eval_geometry_lidar.py --latest              # newest web app job
poetry run python scripts/eval/eval_geometry_lidar.py output/jobs/<job id>
poetry run python scripts/eval/eval_geometry_lidar.py model.glb --bbox W S E N   # a .glb from src/main.py
```

`--help` lists the tolerances, the roof method and local LiDAR (`--laz`). Downloaded LiDAR tiles are cached in `$LIDAR_CACHE_DIR` (in Docker, `/cache/lidar` in the model-cache volume), else `~/.cache/lidar_eval`.

When it can't run (no USGS LiDAR covers the area, laspy is missing, or there's no internet), the pipeline stage records why and the job still finishes with its model.

### Output

`summary.md` (start here), `summary.json`, `per_building.csv`, `per_building_errors.geojson` (open in QGIS), five charts (`01_scorecard.png` to `05_height_error_map.png`), and what it read from the mesh and OSM (`buildings_footprints_utm.geojson`, `terrain_grid.npz`, `osm_buildings.json`).

## Texture quality (UV mapping)

`src/evaluation/texture_uv.py` scores the buildings' texture with one number, following the "UV Mapping Texture Quality Metric":

```text
Confidence = 1 − (α·O + β·D + γ·F),   α + β + γ = 1,   α, β, γ ≥ 0
```

| Term | What it measures | Default weight |
| ---- | ---------------- | -------------- |
| **O** obstruction | how much of the surface has no photo texture: O = 1 − \|M_res ∩ M_gt\| / \|M_gt\| | α = 0.35 |
| **D** distortion | the share of the surface (by area) whose texture is stretched: a triangle counts when its stretch s = σ1 / σ2 > s_max (1.5), the singular values of its surface → texture Jacobian | β = 0.25 |
| **F** realism | F = 1 − exp(−FID / τ), the Fréchet Inception Distance between Inception-v3 features (2048-d pool) of K random patches (2048) of the texture and of a reference | γ = 0.40 |

Each term is between 0 and 1, so the confidence is too; 1 is perfect.

### How it reads this pipeline's model

The paper compares a texture with a ground-truth texture of the same UV layout. This pipeline has no such texture, so by default:

- **The texture** (I_res) is the buildings' atlas in the `.glb`. The ground (terrain and NAIP aerial photo) is not evaluated.
- **M_gt** is the texels the mesh's UV triangles cover.
- **M_res** is the texels in M_gt that hold real photo texture. Texturing records which texels came from a photo, not left grey because no photo saw them and not painted over by obstruction removal, in `model.texture_mask.png` next to the `.glb`. For models made before that file existed, every texel that isn't texturing's grey counts.
- **O is weighted by surface area.** Each triangle counts with its area times the share of its texels in M_res. The atlas gives every triangle its own texel density (its photo's resolution), so counting texels as the paper does would weigh close-up walls more. Triangles no photo saw have no chart at all (texturing points them at one grey texel), so they count as fully missing. The paper's texel count is reported too (`o_texels`).
- **D is measured in texels**, since the atlas isn't square. Triangles with no chart are left out of D (O counts them), as are triangles with no 3D area. A chart triangle of zero area (its corners in a line) is infinitely stretched, so it counts as distorted.
- **The FID reference** is the job's own street photos (the originals saved in `images/`), sampled the same way: patches of 64 texels, resized to Inception's 299 px. The texture's patches must lie at least 90% inside M_res.
- **τ is 100 by default**, a starting value. Calibrate it as the paper says: run the evaluation on results you consider good and use their median FID as τ (`--tau`).

With `--gt-texture` (and optionally `--gt-mask` and `--res-mask`), the paper's definition is used as written: O counts texels against the ground truth's mask, and the FID compares with the ground-truth texture.

If Inception-v3 can't be loaded (torch or torchvision missing, or its weights can't be downloaded), F is left out: the confidence is computed from O and D with their weights renormalised to sum to 1, and is marked **partial**. The weights download once (about 100 MB) into `$TORCH_HOME`, the model-cache volume in Docker. On a CPU the FID takes a few minutes; a GPU is used when there is one.

### Running it

- **Web app:** turn on **Evaluate texture quality** (see [Web App](../web-app.md#evaluating-texture)). It runs as the pipeline's last stage, `evaluate_texture`, and writes into the job's `texture_eval/` folder.
- **Command line, with the pipeline:** `poetry run python src/main.py --bbox ... --output out/model.glb --evaluate-texture` writes `texture_eval/` next to the `.glb`, with reference photos in `images/`.
- **Command line, on a finished model:**

```bash
poetry run python scripts/eval/eval_texture_uv.py --latest                  # newest web app job
poetry run python scripts/eval/eval_texture_uv.py output/jobs/<job id>
poetry run python scripts/eval/eval_texture_uv.py model.glb --reference photos/
poetry run python scripts/eval/eval_texture_uv.py model.glb --gt-texture gt.png   # the paper's setting
```

`--alpha`, `--beta`, `--gamma`, `--s-max`, `--tau`, `--k`, `--patch`, `--seed` and `--device` change the settings; `--no-fid` skips the FID.

### Output

`summary.md` (start here), `summary.json` (every setting and count), `per_group.csv` (O and D per group of connected triangles, about one building each), and three charts: `01_scorecard.png` (the terms), `02_coverage.png` (the atlas with the texels that should hold photo texture and don't in red) and `03_stretch.png` (stretch across the surface).

## Comparing runs

The JSON reports make it easy to compare runs with and without obstruction removal (`--no-seg`), on different hardware, or on different area sizes. Three benchmark runs from **Design Review 1** (older pipeline):

| Profile         | Hardware    | Segmentation | Total Time |
| --------------- | ----------- | ------------ | ---------- |
| `noseg_desktop` | Desktop GPU | Off          | 4m 52s     |
| `seg_desktop`   | Desktop GPU | On           | 6m 43s     |
| `noseg_laptop`  | Laptop CPU  | Off          | 15m 53s    |

Texturing dominated the runtime in all three, and segmentation added about 2 minutes on the desktop GPU.

## Success criteria

| Criterion          | Target                                    | Measurement                   |
| ------------------ | ----------------------------------------- | ----------------------------- |
| Mesh generation    | Uncorrupted mesh file                     | Automated validation          |
| Texture generation | Textures produced from street-view images | Automated validation          |
| Mesh quality       | Average rating ≥ 8/10                     | User survey (20 U-M students) |
| Texture quality    | Average rating ≥ 8/10                     | User survey (20 U-M students) |
| Total runtime      | < 25 min per 247 acres                    | Profiler                      |
| Usability          | Average rating ≥ 7/10                     | User survey                   |

:::info[User survey]

Quality is assessed by asking 20 randomly chosen University of Michigan students to rate PSM buildings against photographs of the real ones. This evaluation is in progress.

:::

## Source code

- `src/common/profiler.py`: `PipelineProfiler`, `StageMetrics`
- `src/evaluation/__init__.py`: the `evaluate_geometry` and `evaluate_texture` stages, `evaluate_model` and `evaluate_texture_model`
- `src/evaluation/geometry_lidar.py`: the geometry evaluation (`scripts/eval/eval_geometry_lidar.py` runs it)
- `src/evaluation/texture_uv.py`: the texture evaluation (`scripts/eval/eval_texture_uv.py` runs it)
