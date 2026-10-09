"""boxtransfer: move a squeezed box up / down (torso paths, squeeze-regulated Cartesian moves), set it down on the floor, place it on a table by force touchdown."""
import sys
from pathlib import Path

_RT = Path(__file__).resolve().parents[3] / 'rby1-task-runtime' / 'code'
if str(_RT) not in sys.path:
    sys.path.insert(0, str(_RT))
from rby1rt import use      # noqa: E402

use('bimanual-squeeze-grasp', 'box-table-perception')
