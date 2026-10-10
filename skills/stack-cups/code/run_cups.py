"""Stack two cups (RB-Y1): put one upright cup into another with a fingers-down rim pinch, perception-checked. Run
from the workspace root, one step at a time; read the returned frames / checkpoint before the next step.

  python code/pick_place/run_cups.py scan "silver cup on the left" "silver cup in the front middle"
  python code/pick_place/run_cups.py pick  "silver cup in the front middle" left
  python code/pick_place/run_cups.py align "silver cup on the left" left   # -> alignment checkpoint (look!)
  python code/pick_place/run_cups.py insert left                           # lower, release at the wedge / rim contact
  python code/pick_place/run_cups.py stack "silver cup on the left" "silver cup in the front middle" [--agent]
        # pick, align, insert. The automatic checks decide (alignment: depth offset < 2.4 cm AND Gemini yes;
        # held / nested: depth AND Gemini); when they are unsure or disagree the step stops with ok None and a
        # checkpoint - the agent looks at the frames and continues with insert, or with
  python code/pick_place/run_cups.py align "silver cup on the left" left --dx 0.01 --dy -0.02   # agent's correction
        # --agent: always stop at the alignment checkpoint (the agent decides every insertion)

State: output/cup_state.json (held cup, pinch geometry, target fix)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _deps  # noqa: F401,E402
import numpy as np  # noqa: E402

import chair_ctl as C  # noqa: E402
import geom  # noqa: E402
import pp_cups as K  # noqa: E402
import pp_grasp as G  # noqa: E402
import pp_motion as M  # noqa: E402
import pp_perception as PP  # noqa: E402

STATE = geom.ROOT / 'output' / 'cup_state.json'
DESK_LOOK = (0.50, 0.10, 0.80)
CUPS = ['silver cup on the left', 'silver cup in the front middle']
ALIGN_Q = ("The robot's gripper holds a silver cup by its rim above another cup on the desk. Is the bottom of the "
           'hanging cup directly above the opening of the cup below, so that lowering it straight down puts it inside?')


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {'held': {}}


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1, default=float))


def scan(ctx, names=None, log=print):
    """Locate the cups (hands moved out of the view first) and report which hand reaches each fingers-down."""
    names = names or CUPS
    st = load_state()
    M.clear_view(ctx, held=[s for s, v in st['held'].items() if v], log=log)
    loc = K.locate(ctx, names, look=DESK_LOOK, log=log)
    s = C.status(ctx)
    j = {'left_arm': s['left_arm'], 'right_arm': s['right_arm']}
    hands = {}
    for n, c in loc['measured']['cups'].items():
        if c.get('ok'):
            hands[n] = {side: (lambda p: p['rim_angle'] if p['ok'] else None)(
                G.plan_rim(dict(c, rim_in=K.CUP_RIM_IN), side, s['torso'], j, pitches=(0,))) for side in ('left', 'right')}
    st['scan'] = {'frame': loc['frames'][0], 'cups': loc['measured']['cups']}
    save_state(st)
    return {'ok': loc['ok'], 'why': loc['why'], 'frames': loc['frames'],
            'measured': {'cups': {n: {k: c.get(k) for k in ('centre', 'r', 'rim_z', 'height')}
                                  for n, c in loc['measured']['cups'].items()}, 'rim_point_by_hand': hands}}


def pick(ctx, name, side, log=print):
    st = load_state()
    if st['held'].get(side):
        return {'ok': False, 'why': f'{side} already holds {st["held"][side]["name"]}'}
    r = K.pick_cup(ctx, name, side, look=DESK_LOOK, log=log)
    if r['ok'] or r['ok'] is None:                        # None: holding, held check unsure -> agent checkpoint
        st['held'][side] = {'name': name, 'pinch': r['measured'].get('pinch')}
        save_state(st)
    return r


def align(ctx, target, side, relocate=False, nudge=None, log=print):
    st = load_state()
    h = st['held'].get(side)
    if not h or not h.get('pinch'):
        return {'ok': False, 'why': f'{side} holds no cup with a known pinch'}
    if h.get('target_name') == target and h.get('target') and not relocate:
        T, loc = dict(h['target'], ok=True), {'frames': []}     # the held cup now hides it: keep the last fix
    else:
        loc = K.locate(ctx, [target], look=DESK_LOOK, log=log)      # fresh fix (cups get knocked)
        T = loc['measured']['cups'][target]
        if not T.get('ok'):
            return {'ok': False, 'why': f'target: {T.get("why")}', 'frames': loc['frames']}
    r = K.align_over(ctx, side, T, h['pinch'], nudge=nudge, log=log)
    h['target'] = {k: T[k] for k in ('centre', 'r', 'rim_z')}
    h['target_name'] = target
    st['held'][side] = h
    save_state(st)
    r['frames'] = loc['frames'] + r['frames']
    return r


def insert(ctx, side, log=print):
    st = load_state()
    h = st['held'].get(side)
    if not h or 'target' not in h:
        return {'ok': False, 'why': 'align first'}
    r = K.insert(ctx, side, h['target'], h['pinch'], log=log)
    if r['ok']:
        st['held'][side] = None
        save_state(st)
    return r


def auto_ok(ctx, r, log=print):
    """--auto gate at the alignment checkpoint: depth offset small AND Gemini yes (each alone was right on 2 of 3
    recorded frames, wrong on different ones)."""
    off = r['measured'].get('offset')
    if off is None or np.linalg.norm(off) > 2 * K.ALIGN_TOL:
        return False
    c = PP.check(ctx, r['frames'][-1], ALIGN_Q, log=log)
    return bool(c.get('ok') and c.get('yes'))


def stack(ctx, base, cup, side='left', agent=False, log=print):
    """Pick `cup`, align it over `base`, insert. Automatic checks decide; the agent steps in (ok None + checkpoint)
    when they are unsure or disagree, or at every alignment with agent=True."""
    r = pick(ctx, cup, side, log=log)
    if not r['ok']:
        return r
    r = align(ctx, base, side, log=log)
    if not r['ok']:
        return r
    if agent:
        return dict(r, ok=None, why='checkpoint: look at the frames, then run insert (or align --dx --dy)')
    if not auto_ok(ctx, r, log=log):
        return dict(r, ok=None, why='alignment checks unsure / disagree: agent looks, then insert or align --dx --dy')
    return insert(ctx, side, log=log)


def main(ctx, step, args=(), log=print):
    a = [x for x in args if not x.startswith('--')]
    if step == 'scan':
        return scan(ctx, a or None, log=log)
    if step == 'pick':
        return pick(ctx, a[0], a[1] if len(a) > 1 else 'left', log=log)
    if step == 'align':
        nudge = None
        if '--dx' in args or '--dy' in args:
            g = lambda k: float(args[args.index(k) + 1]) if k in args else 0.0
            nudge = (g('--dx'), g('--dy'))
            a = [x for i, x in enumerate(args) if not x.startswith('--') and (i == 0 or args[i - 1] not in ('--dx', '--dy'))]
        return align(ctx, a[0], a[1] if len(a) > 1 else 'left', relocate='--relocate' in args, nudge=nudge, log=log)
    if step == 'insert':
        return insert(ctx, a[0] if a else 'left', log=log)
    if step == 'stack':
        return stack(ctx, a[0], a[1], side=a[2] if len(a) > 2 else 'left', agent='--agent' in args, log=log)
    return {'ok': False, 'why': f'unknown step {step}'}


if __name__ == '__main__':
    sys.path.insert(0, str(geom.ROOT))
    from api.runtime import Context
    with Context() as _ctx:
        res = main(_ctx, sys.argv[1], sys.argv[2:])
    print(json.dumps(res, indent=1, default=str)[:8000])
