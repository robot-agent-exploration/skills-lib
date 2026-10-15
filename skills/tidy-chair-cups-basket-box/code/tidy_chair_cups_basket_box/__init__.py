"""Modules of the tidy-chair-cups-basket-box skill. The entry is code/run.py; the hyper-parameters are in params.py."""
import sys
from pathlib import Path

for _slug in ('office-chair-tuck', 'oc-hand-reach', 'bimanual-box-pnp'):  # skills whose packages these modules import
    _code = str(Path(__file__).resolve().parents[3] / _slug / 'code')  # skills/<slug>/code
    if _code not in sys.path:
        sys.path.append(_code)
