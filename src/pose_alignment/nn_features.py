"""The old command for learned_features.py, which now always matches by nearest neighbour.

Usage: python -m pose_alignment.nn_features --images <dir> --out <dir> [--overlap 10] [--masks <dir>]
"""

from . import learned_features

if __name__ == "__main__":
    learned_features.main(default_matcher="nn")
