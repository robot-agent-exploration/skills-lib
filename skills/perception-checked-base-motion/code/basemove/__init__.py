"""basemove: perception-checked base motion for RB-Y1 task skills (plans drawn on the camera image, every
segment checked against fresh depth of its swept footprint, reverse only along the path just driven)."""
import sys
from pathlib import Path

_RT = Path(__file__).resolve().parents[3] / 'rby1-task-runtime' / 'code'
if str(_RT) not in sys.path:
    sys.path.insert(0, str(_RT))
from rby1rt import use      # noqa: E402

use('box-table-perception')
