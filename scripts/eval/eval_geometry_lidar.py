"""Geometry evaluation of a pipeline model against USGS LiDAR (see
src/evaluation/geometry_lidar.py for what it measures and its options).

  poetry install --with eval        # once: laspy, which reads the LiDAR tiles
  poetry run python scripts/eval/eval_geometry_lidar.py --latest
  poetry run python scripts/eval/eval_geometry_lidar.py output/jobs/<id>

The web app runs the same evaluation as the job's last stage ("Evaluate geometry
against LiDAR" in the options).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.geometry_lidar import main  # noqa: E402

if __name__ == "__main__":
    main()
