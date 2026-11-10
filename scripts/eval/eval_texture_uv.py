"""UV texture quality of a pipeline model (see src/evaluation/texture_uv.py for what it
measures, how it reads this pipeline's model and its options).

  poetry run python scripts/eval/eval_texture_uv.py --latest
  poetry run python scripts/eval/eval_texture_uv.py output/jobs/<id>
  poetry run python scripts/eval/eval_texture_uv.py model.glb --gt-texture gt.png

The web app runs the same evaluation as the job's last stage ("Evaluate texture
quality" in the options).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.texture_uv import main  # noqa: E402

if __name__ == "__main__":
    main()
