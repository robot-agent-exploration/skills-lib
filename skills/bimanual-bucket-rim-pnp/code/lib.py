"""Library pieces this controller reuses from the skill library (bimanual-box-pnp, generic modules only):
core/vlm.py (Gemini questions, points, masks with the [y, x] polygon fix), core/frames.py (saved-frame depth,
projection, pixel -> base point, ray / plane), core/runner.py (steps, checkpoints, resume, STOP file),
core/footprint.py, core/geom.py, core/kin.py (pinocchio FK / IK on robot-model/)."""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = next(p for p in HERE.parents if (p / 'api' / 'runtime.py').exists())
_SIBLING = HERE.parents[1] / 'bimanual-box-pnp' / 'code'          # when this file is skill-library/skills/<slug>/code/
BOXPNP_CODE = _SIBLING if (_SIBLING / 'boxpnp').exists() else \
    WORKSPACE / 'skill-library' / 'skills' / 'bimanual-box-pnp' / 'code'
for p in (BOXPNP_CODE, WORKSPACE, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

vlm = importlib.import_module('boxpnp.core.vlm')
frames = importlib.import_module('boxpnp.core.frames')
