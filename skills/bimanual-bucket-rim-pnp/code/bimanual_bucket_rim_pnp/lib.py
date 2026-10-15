"""Library pieces this controller reuses from the skill library (bimanual-box-pnp, generic modules only):
core/vlm.py (Gemini questions, points, masks: a wrapper over the gemini skill's shared helper), core/frames.py (saved-frame depth,
projection, pixel -> base point, ray / plane), core/runner.py (steps, checkpoints, resume, STOP file),
core/footprint.py, core/geom.py, core/kin.py (pinocchio FK / IK on robot-model/)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = next(p for p in HERE.parents if (p / 'api' / 'runtime.py').exists())
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from bimanual_box_pnp.core import vlm, frames  # noqa: E402,F401  (this package's __init__ puts bimanual-box-pnp on sys.path)
