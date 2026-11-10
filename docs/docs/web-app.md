# Web App

The web app runs the same pipeline as the command line, with a map to pick the area, live progress, a 3D preview and a history of jobs. Start it with `docker compose up --build api` and open <http://localhost:8000> (see [Getting Started](getting-started.mdx)).

## Picking an area

- **Search** for an address or place. Suggestions appear as you type; press **Search** for a full search. Choosing a result frames a sensible area around it.
- **Draw area**: press it, then drag a box on the map. **Esc** cancels.
- **Coordinates**: type west, south, east and north directly.
- **Open larger map** shows the same map in a bigger dialog.

The size of the area is shown under the coordinates. Each side can be at most 2.5 km.

![The Area panel: address search, the OSM map with the selected area drawn on it, Generate model, and the area's coordinates and size](/img/screenshots/pick-area.webp)

## Options

| Option                  | What it does                                                                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Remove obstructions** | Erase cars, people, poles and similar from the street photos before projecting them (Mask2Former + LaMa). Slower, cleaner facades.                                                     |
| **Segmentation model**  | Mask2Former **Swin-Large** (best quality) or **Swin-Tiny** (much faster on a CPU).                                                                                                       |
| **Device**              | **Auto** uses an NVIDIA GPU when the server has one, otherwise the CPU. **GPU** is only available on the GPU image with a GPU.                                                          |
| **Terrain**             | Put buildings on real ground heights and add the ground to the model.                                                                                                                   |
| **Aerial photos**       | Texture the ground and roofs with NAIP aerial photos. Continental US only; turns on Terrain (and locks it on while it's on). A note appears if the area looks outside NAIP's coverage. |
| **Profile this run**    | Measure each stage's time and memory, shown live in the job's details. See [Profiling a job](#profiling-a-job).                                                                         |
| **Evaluate geometry against LiDAR** | After export, compare the model's buildings and ground with USGS airborne LiDAR. US only, needs internet, adds a few minutes. A note appears if the area looks outside the US. See [Evaluating geometry](#evaluating-geometry). |
| **Evaluate texture quality** | After export, score the buildings' texture: how much of it has no photo, how stretched it is and how real it looks. Downloads an Inception model (about 100 MB) the first time; a GPU is recommended, a CPU takes a few minutes. See [Evaluating texture](#evaluating-texture). |

The starting values come from the defaults in [Settings](#settings).

![The Options panel: camera poses, Aligned photos only, Remove obstructions, segmentation model, device, Terrain and Aerial photos](/img/screenshots/options.webp)

## Running a job

Press **Generate model**. While a job runs:

- The **timeline** shows each pipeline stage as pending, running, done, or failed. Stages turned off for this job (Terrain, Aerial photo, Evaluate geometry, Evaluating texture) show as **Skipped**.
- The **pipeline log** streams the job's output live, including progress bars such as the photo downloads. **Download log** saves it as a text file.
- The 3D preview is greyed out, and **Generate model** is disabled until the job finishes.

Jobs run one at a time. Finished jobs survive a page reload or a server restart; a job that was still running when the server stopped is marked as failed.

![A job running: the 3D preview greyed out with its current stage, the timeline with the stages done so far and the skipped ones, and the live pipeline log](/img/screenshots/job-running.webp)

When the job finishes, the model loads into the **3D preview** (drag to orbit, scroll to zoom). **.glb** downloads it, and **Larger view** opens a bigger viewer. The **Street imagery** section shows the first 12 photos of the job, before and after obstruction removal.

![The finished model in the 3D preview, with the completed timeline and the pipeline log below](/img/screenshots/preview.webp)

![The Street imagery section: the job's street photos, with the Original and Cleaned toggle](/img/screenshots/street-imagery.webp)

## Recent jobs

Every job is listed with its area, status, building count and duration. Click a row to open its details:

![The Recent jobs table: each job's area, bounding box, status, building count and duration](/img/screenshots/recent-jobs.webp)


- the model, with stats and a download button
- the job's settings (area, options, device) and its error, if it failed
- the full pipeline log
- the street photos


![A job's details: the model, its stats and download, the job's settings, the pipeline log and its photos](/img/screenshots/job-details.webp)

Jobs are saved under `output/jobs/<job id>/` (in Docker, the `output` folder in your project) with `job.json`, `pipeline.log`, `model.glb` and the photo previews.

## Profiling a job

Turn on **Profile this run** in the Options panel to measure each pipeline stage's time and memory. The job's details then get a **Profile** section:

- **Headline figures:** total time, the slowest stage and its share of the run, the highest memory peak and where it happened, and CPU use (CPU time divided by wall time; over 100% means several cores were busy).
- **Time per stage** and **Peak memory per stage:** one bar per stage, with its value at the end. Hover or focus a bar for its wall time, CPU time and peak memory. Stages that are off for the job are labelled **skipped**; a stage that failed has a red bar marked **⚠ error**, and its error is in the tooltip.
- **All metrics:** a table with every stage's wall time, CPU time, CPU %, peak memory, memory change and GPU memory.
- **Download JSON** saves the full report (`profile.json`, the same format as the command line's `--profile`, see [Evaluation](pipeline/evaluation.md#pipeline-profiler)).

The profile **updates live** while the job runs: after every stage, and the running stage's time ticks every second. You can open it at any point:

- **View profile**, next to the progress bar, opens the current job's profile.
- In **Recent jobs**, profiled jobs carry a **Profiled** marker; open the job to see its profile, finished or not.

A job that fails keeps its profile up to the failed stage. Peak memory is what Python's `tracemalloc` sees: Python and NumPy allocations, not the whole process (for example not PyTorch's own buffers or GPU memory). Profiling makes a run slower, because `tracemalloc` records every allocation.

## Evaluating geometry

Turn on **Evaluate geometry against LiDAR** to check the finished model against real measurements. After the model is exported, the **Evaluate geometry** stage downloads the USGS 3DEP airborne LiDAR for the area and compares each building and the ground with it (the same evaluation as the command line's, see [Evaluation](pipeline/evaluation.md#geometry-evaluation-against-lidar)). Its progress streams into the pipeline log like any other stage. Turn on **Terrain** as well to get every metric: a flat model only gets the building heights.

The job's details then get an **Evaluation** section, below the Profile, with a **Geometry · LiDAR** part:

- **Terrain verdict:** **✓ Pass**, **✗ Fail**, **? Inconclusive** or **– Not tested**, with the share of buildings whose ground is within ±1 m of the LiDAR ground and its 95% confidence interval. It passes when the interval's lower end reaches 95%; fewer than 30 evaluated buildings is always inconclusive.
- **Headline figures:** how many OSM buildings were evaluated, the share of heights within ±3 m (about one storey), and the median height and terrain errors. Errors are model − LiDAR: positive means the model is too tall or too high.
- **Heights by confidence:** the share within ±3 m for buildings whose OSM height came from a `height` tag (high confidence), from `building:levels` (medium), or from the default when there is no tag (low). This is a baseline to compare runs with, not pass/fail: OSM often has no measured height.
- **Errors:** the median, RMSE and NMAD of each metric: height (M1), terrain at the building (M2), base placement, and roof elevation (M3).
- **Not evaluated:** buildings left out and why, for example in OSM but not in the model, or too small to measure.
- **Charts:** the evaluation's scorecard, height errors by source, model against LiDAR height, placement against slope, and an error map. Click one to open it full size.
- **Full report:** the evaluation's `summary.md`, and downloads of the report, the per-building CSV, a GeoJSON map of the errors (open it in QGIS) and the full summary.

The results appear when the job finishes. **View evaluation**, next to the progress bar, opens the current job's evaluation, and **Recent jobs** marks evaluated jobs **Evaluated**.

If the evaluation can't run here (no USGS LiDAR for the area, no internet, or the LiDAR reader missing), the section says why and the job still finishes with its model. The same goes for an evaluation that fails. The results are saved in the job's `geometry_eval/` folder.

## Evaluating texture

Turn on **Evaluate texture quality** to score the buildings' texture with one number, the **confidence** (0 to 1, 1 is perfect; see [Evaluation](pipeline/evaluation.md#texture-quality-uv-mapping)). After the model is exported, the **Evaluating texture** stage reads the texture back from the `.glb` and measures three things, each from 0 (best) to 1 (worst):

- **O, without photo texture:** the share of the buildings' surface no street photo textured: never seen by a photo, or painted over where obstructions (cars, people, poles) were removed.
- **D, stretched texture:** the share of the surface whose texture is stretched more one way than the other (more than 1.5×).
- **F, realism penalty:** how unlike the job's street photos the texture looks, from the Fréchet Inception Distance (FID) between the two: F = 1 − exp(−FID / τ). τ is a starting value until it is calibrated.

The confidence is 1 − (0.35·O + 0.25·D + 0.40·F). Its progress streams into the pipeline log like any other stage. It works with any options; it doesn't need Terrain.

The **Evaluation** section then gets a **Texture · UV quality** part (next to Geometry when both ran):

- **Full or partial score:** if the Inception model can't be loaded (for example, no internet for its first download), F is left out and the confidence is computed from O and D, reweighted. The section says **Partial score** and why.
- **Headline figures:** the confidence and its formula, O, D, and F with its FID and τ, each with a one-line explanation.
- **Building groups with the most missing texture:** groups of connected triangles (about one building each), with their area, position from the model's centre, O and D.
- **Charts:** the scorecard, the texture atlas with the texels that should hold photo texture and don't in red, and how stretched the texture is across the surface. Click one to open it full size.
- **Full report:** the evaluation's `summary.md`, and downloads of the report, the per-group CSV and the full summary.

**View evaluation** and the **Evaluated** marker in Recent jobs cover either evaluation. If the texture evaluation can't run (for example, the model has no texture), the part says why and the job still finishes with its model. The results are saved in the job's `texture_eval/` folder, and texturing saves which texels came from a photo in `model.texture_mask.png`.

## Settings

The gear icon in the header opens Settings.

![The Settings dialog: the Mapillary token's status and masked value with Change and Validate Token, and the defaults for new jobs](/img/screenshots/settings.webp)

### Mapillary access token

- The status shows whether a token is set and where it comes from: **Saved here**, or **From .env** (the env file for your `APP_ENV`, e.g. `.env.development`), with its last four characters.
- The token shows as dots; **Show** reveals it.
- Press **Change**, paste the new token, then **Save token**. PSM asks you to confirm before it replaces the current one. New jobs use it straight away.
- **Validate Token** checks a token with Mapillary.
- **Remove saved token** (or **Use the .env token instead**) removes the one saved here, after confirming.

Where a saved token goes:

- If the env file doesn't exist yet, saving creates it (e.g. `.env` in your project folder) with `MAPILLARY_ACCESS_TOKEN=...`.
- Otherwise the token goes into `output/jobs/settings.json` and takes priority over the env file's.

Both files are readable by your user only. See [Configuration](configuration.md).

### Defaults for new jobs

Obstruction removal, segmentation model and device. They fill the Options panel each time PSM opens.

Without a token, a notice next to **Generate model** links to Settings, and pressing Generate opens Settings instead of starting a job that would fail.

## Who can reach it

The server listens on `127.0.0.1` (this computer only) by default. It has no login, and Settings can show the Mapillary token, so only open it to your network if you trust everyone on it:

```bash
PSM_HOST=0.0.0.0 docker compose up api
```

## Light and dark mode

The sun/moon button switches themes; the choice is remembered in the browser. By default the app follows the system setting.

![The web app in dark mode](/img/screenshots/app-overview-dark.webp)
