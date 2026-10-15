"""The demonstrated bandu tower (measured 2026-10-10) and the table spots used to take it apart.

All pieces sit on the base centreline (the base's long axis runs along base +y). Offsets are
along that axis from the base centre; heights are relative to the base top. The base itself is
never moved by the controller.
"""
import math

import sys as _sys                                                   # skill folder: siblings import each other by name
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

from pieces import PIECES, hand_conflicts

from bandu_params import BASE, MAX_OPENING_SPREAD, SPOTS  # hyper-parameters: bandu_params.py

# piece -> (offset along the base axis, support piece or None for the base top,
#           bottom height above the base top when seated, how it goes on)
SLOTS = {
    'ring':   dict(offset=+0.011, support=None, bottom=0.0, method='place'),
    'barrel': dict(offset=+0.054, support=None, bottom=0.0, method='place'),
    'goblet': dict(offset=-0.036, support=None, bottom=0.0, method='place'),
    'urn':    dict(offset=-0.036, support='goblet', bottom=0.0475, method='insert'),
    'dowel':  dict(offset=+0.011, support='ring', bottom=0.0, method='insert'),
    'knob':   dict(offset=-0.036, support='urn', bottom=0.0749, method='insert'),
    # added by the user 13:42 ("put it on the barrel"): stands on the barrel top, leaning towards +y
    'slant':  dict(offset=+0.054, support='barrel', bottom=0.040, method='place'),
}
# supports whose opening the wrist camera measures before something is inserted into them:
# rim height above the support's bottom, opening and rim radii (m)
OPENINGS = {'urn': dict(rim=0.0455, r_in=0.010, r_out=0.013)}

# low pieces first (the hand must not sweep over tall stacks); a piece only after its support
ASSEMBLY_ORDER = ['ring', 'barrel', 'slant', 'goblet', 'urn', 'dowel', 'knob']
DISASSEMBLY_ORDER = ASSEMBLY_ORDER[::-1]


# piece footprint radius (m) and height above its bottom (m) for the hand-footprint check
SHAPE = {
    'ring': (0.019, 0.016), 'barrel': (0.014, 0.040), 'goblet': (0.023, 0.061),
    'urn': (0.016, 0.041), 'knob': (0.013, 0.057), 'dowel': (0.008, 0.113), 'slant': (0.022, 0.070),
}
# yaws of the grasps used to pick a piece standing on the table (double grasp = centring)
TABLE_PICK_YAWS = {'slant': (math.pi / 2,)}
DEFAULT_PICK_YAWS = (0.0, math.pi / 2)


def slot_xy(name, base=BASE):
    s = SLOTS[name]
    return (base['center'][0] + s['offset'] * base['axis'][0], base['center'][1] + s['offset'] * base['axis'][1])


def slot_bottom_z(name, base=BASE):
    return base['top_z'] + SLOTS[name]['bottom']


def objects(state):
    """Footprint objects for hand_conflicts from a state {name: {'xy', 'bottom_z'[, 'radius',
    'height']}} (radius/height default to the piece's SHAPE)."""
    out = []
    for name, s in state.items():
        r, h = SHAPE.get(name, (None, None))
        r, h = s.get('radius', r), s.get('height', h)
        out.append(dict(name=name, xy=tuple(s['xy']), radius=r, top_z=s['bottom_z'] + h))
    return out


def grasp_conflicts(state, name, xy, bottom_z, yaws, margin=0.015):
    """Pieces the hand (body and open fingers) would hit when its TCP is at this piece's grasp
    height at xy."""
    z = bottom_z + PIECES[name]['grasp']
    hits = set()
    for yaw in yaws:
        hits.update(hand_conflicts((xy[0], xy[1], z), yaw, objects(state), margin=margin, ignore=(name,),
                                   opening=PIECES[name]['open'], ignore_fingers=supports_below(name, xy)))
    return sorted(hits)


def supports_below(name, xy, tol=0.012):
    """The chain of supports under a piece standing at its tower slot (empty on the table)."""
    if math.dist(xy, slot_xy(name)) > tol:
        return ()
    out, sup = [], SLOTS[name]['support']
    while sup:
        out.append(sup)
        sup = SLOTS[sup]['support']
    return tuple(out)


def tower_state(base=BASE):
    return {n: dict(xy=slot_xy(n, base), bottom_z=slot_bottom_z(n, base)) for n in SLOTS}


def check_plan(table_z=0.093, log=print):
    """Simulate disassembly onto SPOTS and reassembly; report every hand-footprint conflict."""
    both, single = DEFAULT_PICK_YAWS, (math.pi / 2,)
    state, problems = tower_state(), []
    for n in DISASSEMBLY_ORDER:
        s = state.pop(n)
        for what, xy, bz, yaws in (('pick from tower', s['xy'], s['bottom_z'], single),
                                   ('place on table', SPOTS[n], table_z, single)):
            c = grasp_conflicts(state, n, xy, bz, yaws)
            if c:
                problems.append(f'{n}: {what} hits {c}')
        state[n] = dict(xy=SPOTS[n], bottom_z=table_z)
    for n in ASSEMBLY_ORDER:
        s = state.pop(n)
        c = grasp_conflicts(state, n, s['xy'], s['bottom_z'], TABLE_PICK_YAWS.get(n, both))
        if c:
            problems.append(f'{n}: pick from table hits {c}')
        xy, bz = slot_xy(n), slot_bottom_z(n)
        c = grasp_conflicts(state, n, xy, bz, single)
        if c:
            problems.append(f'{n}: put on tower hits {c}')
        state[n] = dict(xy=xy, bottom_z=bz)
    for p in problems:
        log(p)
    return problems


if __name__ == '__main__':
    print('plan ok' if not check_plan() else 'plan has conflicts')
