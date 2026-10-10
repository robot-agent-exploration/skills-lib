"""Put the shared helpers (geom, kin, chair_ctl from the office-chair-tuck skill) on sys.path: code/push_chair in the
workspace, skills/office-chair-tuck/code in the skill library."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _cand in (HERE.parent / 'push_chair', HERE.parents[1] / 'office-chair-tuck' / 'code'):
    if (_cand / 'chair_ctl.py').exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
else:
    raise ImportError('pick_place needs geom.py / kin.py / chair_ctl.py (office-chair-tuck skill code)')
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
