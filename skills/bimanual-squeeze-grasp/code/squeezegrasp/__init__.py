"""squeezegrasp: bimanual closed-ball squeeze of a box (from the floor with a squat, from a table top), release."""
import sys
from pathlib import Path

_RT = Path(__file__).resolve().parents[3] / 'rby1-task-runtime' / 'code'
if str(_RT) not in sys.path:
    sys.path.insert(0, str(_RT))
from rby1rt import use      # noqa: E402

use('box-table-perception', 'perception-checked-base-motion')
