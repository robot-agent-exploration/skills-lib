"""boxperc: box / table-top perception for the box skills (Gemini-seeded, depth for metres)."""
import sys
from pathlib import Path

_RT = Path(__file__).resolve().parents[3] / 'rby1-task-runtime' / 'code'
if str(_RT) not in sys.path:
    sys.path.insert(0, str(_RT))
