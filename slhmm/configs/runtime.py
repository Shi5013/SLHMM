"""Filesystem locations owned by the standalone :mod:`slhmm` package."""

from pathlib import Path
import os


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("SLHMM_PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
DATA_ROOT = PROJECT_ROOT / "data"
INPUT_DATA_ROOT = DATA_ROOT / "original_10831" / "observations"
ARTIFACT_ROOT = PROJECT_ROOT / "intermediate"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
