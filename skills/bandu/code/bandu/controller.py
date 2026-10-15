"""Autonomous bandu controller: build the demonstrated tower (tower.py) from pieces standing on
the table, and take it apart again onto table spots for practice.

Run from the workspace root with <skill>/code on PYTHONPATH (paths below are relative to the working directory):
  python3 -m bandu.controller assemble [--fresh] [--only N]
  python3 -m bandu.controller disassemble [--only N]
  python3 -m bandu.controller survey             # observe and report, no motion beyond parking
--fresh ignores the piece record (output/bandu/state.json) and labels pieces from images only.
--only N stops after N pieces. Logs and report: output/runs/<timestamp>/.
From Python use the skill entry run.py: main(ctx, mode=...) -> {ok, measured, why}.
"""
import json
import math
import sys
import time
from pathlib import Path

from . import align                                                         # noqa: E402
from . import holes                                                         # noqa: E402
from . import perceive                                                      # noqa: E402
from . import pieces                                                        # noqa: E402
from . import tower                                                         # noqa: E402
from .motion import Motion, table_z_at                                # noqa: E402
from .pieces import PIECES                                            # noqa: E402
from .tower import ASSEMBLY_ORDER, DISASSEMBLY_ORDER, SLOTS, SPOTS    # noqa: E402

ROOT = Path('.')            # the working directory: the workspace root (harness convention)
STATE_PATH = ROOT / 'output' / 'bandu' / 'state.json'
RUNS = ROOT / 'output' / 'runs'

from .params import (  # hyper-parameters: params.py
    GRASP_YAW, PRIOR_MATCH_R, PICK_ATTEMPTS, TABLE_PLACE_TOL, SPOT_CLEAR_R, ALT_SPOTS, MIN_GRIPPER_SPAN,
    BALL_PICK_YAWS,
)


# ----------------------------------------------------------------------------- record

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {'pieces': {}}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    STATE_PATH.write_text(json.dumps(state, indent=1))


def world_from(state, unknown=()):
    """Footprint objects for the hand-conflict check: every piece where we believe it is, plus
    unidentified clusters (as generic cylinders)."""
    w = {n: dict(xy=s['xy'], bottom_z=s['bottom_z']) for n, s in state['pieces'].items()
         if s.get('where') in ('table', 'tower')}
    for i, o in enumerate(unknown):
        w[f'unknown{i}'] = dict(xy=o['xy'], bottom_z=o['table_z'], radius=max(o['extent']) / 2,
                                height=o['height'])
    return w


# ----------------------------------------------------------------------------- steps

def survey(ctx, m, state, log, fresh=False):
    """Park, observe, find the pieces on the table and which slots of the tower are occupied.
    Labels: height class from exterior depth, confirmed by Gemini (one call per survey); the
    record's table positions are reused when a matching cluster is within PRIOR_MATCH_R."""
    perceive.park(ctx, m)
    fid = perceive.snapshot(ctx)
    cloud = perceive.wood_cloud(fid)
    objs = perceive.table_objects(fid)
    rec = {} if fresh else state['pieces']
    tower_xy = {n: (rec[n]['xy'] if rec.get(n, {}).get('where') == 'tower' else tower.slot_xy(n)) for n in SLOTS}
    presence = perceive.slot_presence(cloud, tower_xy)
    labels = ['?'] * len(objs)
    if objs:
        try:
            labels = perceive.gemini_labels(ctx, fid, objs, save=str(RUNS / 'last_labels.png'))
        except Exception as e:                     # labels then come from heights and the record
            log(f'gemini labels failed: {e}')
    table, unknown = {}, []
    for o, lab in zip(objs, labels):
        o['label'] = lab
        prior = [n for n, s in rec.items() if s.get('where') == 'table'
                 and math.dist(s['xy'], o['xy']) < PRIOR_MATCH_R and n in o['possible']]
        if o['lying']:
            name = None
        elif lab in o['possible']:
            name = lab
        elif lab == '?' and len(prior) == 1:
            name = prior[0]
        elif lab == '?' and len(o['classes']) == 1:
            name = o['classes'][0]
        else:
            name = None
        if name is None or name in table:
            unknown.append(o)
            continue
        o['name'] = name
        if name in prior:                          # the robot put it there: its own record is sharper
            o['xy_record'] = rec[name]['xy']
            o['bottom_z'] = rec[name]['bottom_z']
            o['yaw'] = rec[name].get('yaw')
        elif 'lean' in PIECES[name]:               # lean direction and grasp-height axis from depth slices
            o['xy'], o['yaw'] = perceive.lean_pose(cloud, o['xy'], o['table_z'], PIECES[name]['grasp'])
        table[name] = o
    on_tower = {n: presence[n][1] and n not in table for n in SLOTS}
    log(f'survey {fid}: table ' + ', '.join(f"{n}@({o['xy'][0]:.3f},{o['xy'][1]:.3f}) h{1000 * o['height']:.0f}"
                                            for n, o in table.items()))
    log('  tower points: ' + ', '.join(f'{n} {presence[n][0]}{"+" if on_tower[n] else "-"}' for n in SLOTS))
    for o in unknown:
        log(f"  unidentified cluster at ({o['xy'][0]:.3f},{o['xy'][1]:.3f}) h{1000 * o['height']:.0f} "
            f"classes {o['classes']} gemini '{o['label']}' lying {o['lying']}")
    return {'ok': True, 'frame': fid, 'table': table, 'on_tower': on_tower, 'unknown': unknown,
            'points': {n: presence[n][0] for n in SLOTS}}


def sync_state(state, scene):
    """Rewrite the record from a survey: table pieces at their observed (or recorded) place,
    tower pieces at their slot (keeping a recorded tower xy)."""
    new = {}
    for n, o in scene['table'].items():
        new[n] = dict(where='table', xy=o.get('xy_record', o['xy']),
                      bottom_z=o.get('bottom_z', o['table_z']), known_bottom='bottom_z' in o, yaw=o.get('yaw'))
    for n, on in scene['on_tower'].items():
        if on:
            old = state['pieces'].get(n, {})
            xy = old['xy'] if old.get('where') == 'tower' else tower.slot_xy(n)
            new[n] = dict(where='tower', xy=xy, bottom_z=tower.slot_bottom_z(n), yaw=old.get('yaw'))
    for n, old in state['pieces'].items():           # not seen this time: keep as a hint only
        if n not in new:
            new[n] = dict(old, where='unseen')
    state['pieces'] = new
    return state


def pick_from_table(ctx, m, name, state, scene, log):
    """Double grasp (yaw 0 then 90) so the piece ends centred under the TCP. Retries once
    from a fresh observation."""
    for attempt in range(PICK_ATTEMPTS):
        s = state['pieces'][name]
        xy, bz = s['xy'], s['bottom_z']
        center = True
        if 'lean' in PIECES[name]:                 # one grasp with TCP x along the lean keeps it leaning
            if s.get('yaw') is None:
                return {'ok': False, 'why': f'lean direction of {name} unknown'}
            yaw, yaws = s['yaw'], (s['yaw'],)
        elif 'ball' in PIECES[name]:               # locate the ball from above, one grasp right there
            world = world_from(state, scene['unknown'])
            free = [w for w in BALL_PICK_YAWS if not tower.grasp_conflicts(world, name, xy, bz, (w,))]
            yaw = free[0] if free else GRASP_YAW   # round: the hand turns back to GRASP_YAW on the way
            yaws, center = (yaw,), False
        else:
            yaw, yaws = GRASP_YAW, tower.TABLE_PICK_YAWS.get(name, tower.DEFAULT_PICK_YAWS)
        hits = tower.grasp_conflicts(world_from(state, scene['unknown']), name, xy, bz, yaws)
        if hits:
            return {'ok': False, 'why': f'hand would hit {hits} when picking {name} at {xy}', 'blockers': hits}
        if not center:
            xy = locate_ball(ctx, m, name, xy, bz, log) or xy
        try:
            r = pieces.pick(m, name, xy[0], xy[1], yaw=yaw, center=center,
                            table_z=bz if s.get('known_bottom') else None, log=log)
            s.update(where='held', bottom_z=r['table_z'])
            return {'ok': True, 'table_z': r['table_z'], 'xy': xy}
        except RuntimeError as e:
            log(f'{name}: pick attempt {attempt + 1} failed: {e}')
            if attempt + 1 == PICK_ATTEMPTS:
                return {'ok': False, 'why': f'pick failed: {e}'}
            scene.update(survey(ctx, m, state, log, fresh=True))
            if name not in scene['table']:
                return {'ok': False, 'why': f'{name} not found on the table after a failed pick'}
            o = scene['table'][name]
            state['pieces'][name] = dict(where='table', xy=o['xy'], bottom_z=o['table_z'], known_bottom=False,
                                         yaw=o.get('yaw'))


def clear_blocker(ctx, m, name, state, scene, log):
    """Move a table piece that blocks another pick to a free table spot."""
    spot = choose_spot(name, state, scene)
    if spot is None:
        return {'ok': False, 'why': f'no free table spot to move {name} to'}
    log(f'{name}: in the way, moving it to ({spot[0]:.3f},{spot[1]:.3f})')
    r = pick_from_table(ctx, m, name, state, scene, log)
    if not r['ok']:
        return r
    r = put_on_table(ctx, m, name, spot, state, log)
    if r['ok'] and name in scene['table']:
        scene['table'][name]['xy'] = state['pieces'][name]['xy']
    return r


def straightener(ctx, name, log):
    """For pieces that can hang tilted in the grip (ball/body grasps): rotate them vertical before
    they touch down (a tilted piece lands on one edge and can turn over when released)."""
    if name not in align.PROFILES:
        return None
    return lambda mm: align.straighten(ctx, mm, name, log=log)


def check_upright(ctx, m, name, state, log):
    """Tilt (deg) of a just-inserted piece on the tower from both exterior cameras (None: not
    checked). A seated urn reads 0-3 deg (an upright goblet reads ~2), one caught on the rim 22-24."""
    if name not in align.UPRIGHT_ROWS:
        return None
    s = state['pieces'][name]
    perceive.park(ctx, m)
    fid = perceive.snapshot(ctx)
    tilt, rms = align.upright(fid, s['xy'], s['bottom_z'] + s.get('seat_err', 0.0), align.UPRIGHT_ROWS[name])
    log(f"{name}: upright check {'n/a' if tilt is None else f'{tilt:.1f} deg (rms {rms:.2f} px)'}  frame {fid}")
    return tilt


def put_back(ctx, m, name, xy, bottom_z, log, yaw=GRASP_YAW):
    """Recovery: set a held piece down where it came from (xy = TCP xy it was picked at) and
    release it."""
    lean = PIECES[name].get('lean', 0.0)
    bx, by = xy[0] - lean * math.cos(yaw), xy[1] - lean * math.sin(yaw)
    try:
        r = pieces.place(m, name, bx, by, bottom_z, yaw=yaw, tol=TABLE_PLACE_TOL, log=log,
                         straighten=straightener(ctx, name, log))
        return {'ok': True, 'bottom_z': r['contact_z'] - PIECES[name]['grasp'], 'xy': list(r['tcp_xy'])}
    except RuntimeError as e:
        return {'ok': False, 'why': f'could not put {name} back: {e}'}


def put_on_tower(ctx, m, name, state, log):
    """Place (base-level pieces) or insert by tactile search (stacked pieces). On failure the
    piece goes back to the table where it came from."""
    slot = SLOTS[name]
    bz = tower.slot_bottom_z(name)
    world = world_from(state)
    if slot['method'] == 'place':
        xy = tower.slot_xy(name)
    else:
        sup = state['pieces'].get(slot['support'], {})
        if sup.get('where') != 'tower':
            return {'ok': False, 'why': f"{name} needs {slot['support']} on the tower first"}
        xy = sup.get('opening_xy') or sup['xy']
    hits = tower.grasp_conflicts(world, name, xy, bz, (GRASP_YAW,))
    if hits:
        return {'ok': False, 'why': f'hand would hit {hits} when putting {name} on the tower'}
    try:
        if slot['method'] == 'place':
            r = pieces.place(m, name, xy[0], xy[1], bz, yaw=GRASP_YAW, log=log)
            xy = r['tcp_xy']
        else:
            op = tower.OPENINGS.get(slot['support'])
            if PIECES[name].get('visual') and op:     # adaptive tilt + offset correction with the exterior cameras
                z_rim = sup['bottom_z'] + sup.get('seat_err', 0.0) + op['rim']
                r = align.insert_visual(ctx, m, name, xy, z_rim, bz, yaw=GRASP_YAW, log=log)
            else:
                rim = sup['bottom_z'] + op['rim'] - bz if op else None
                r = pieces.insert(m, name, xy[0], xy[1], bz, yaw=GRASP_YAW,
                                  fine=bool(sup.get('opening_xy')), rim=rim, log=log)
            xy = r['xy']
    except pieces.NotSeated as e:                  # released in the opening, sits high: needs a look
        state['pieces'][name] = dict(where='unseated', xy=list(xy), bottom_z=bz, yaw=GRASP_YAW)
        return {'ok': False, 'why': str(e)}
    except pieces.PieceLost as e:
        state['pieces'][name]['where'] = 'unseen'
        m.gripper_open(0.06)
        return {'ok': False, 'why': str(e), 'lost': True}
    except RuntimeError as e:
        log(f'{name}: {e}; putting it back on the table')
        s = state['pieces'][name]
        back = put_back(ctx, m, name, s['xy'], s['bottom_z'], log, yaw=s.get('yaw') or GRASP_YAW)
        if back['ok']:
            s.update(where='table', bottom_z=back['bottom_z'], known_bottom=True, xy=back['xy'])
        return {'ok': False, 'why': str(e), 'put_back': back}
    state['pieces'][name] = dict(where='tower', xy=list(xy), bottom_z=bz, yaw=GRASP_YAW,
                                 seat_err=r.get('seat_err', r.get('contact_z', bz + PIECES[name]['grasp']) - bz - PIECES[name]['grasp']))
    return {'ok': True, 'xy': list(xy), 'seat_err': state['pieces'][name]['seat_err']}


def measure_opening(ctx, m, name, state, log):
    """Before picking a piece that goes into a support with a visible opening (tower.OPENINGS),
    measure that opening with the wrist camera (gripper empty) so the insertion starts on it."""
    sup_name = SLOTS[name]['support']
    if SLOTS[name]['method'] != 'insert' or sup_name not in tower.OPENINGS:
        return None
    sup = state['pieces'].get(sup_name, {})
    sup.pop('opening_xy', None)
    if sup.get('where') != 'tower':
        return None
    spec = tower.OPENINGS[sup_name]
    try:
        r = holes.locate_opening(ctx, m, sup['xy'], sup['bottom_z'] + spec['rim'], r_in=spec['r_in'],
                                 r_out=spec['r_out'], log=log)
    except RuntimeError as e:
        log(f'{sup_name} opening: {e}; using its recorded xy')
        return None
    off = [1000 * (r['xy'][i] - sup['xy'][i]) for i in (0, 1)]
    log(f"{sup_name} opening at ({r['xy'][0]:.4f},{r['xy'][1]:.4f}): {off[0]:+.1f},{off[1]:+.1f} mm from its axis, "
        f"views {1000 * r['spread']:.1f} mm apart")
    if r['spread'] <= tower.MAX_OPENING_SPREAD:
        sup['opening_xy'] = r['xy']
    return r


def locate_ball(ctx, m, name, xy, bottom_z, log):
    """Centre of a ball-topped piece standing on the table from two opposite wrist views (the ball
    is a dark disc on the white table, same matched filter as an opening), or None."""
    r_ball = PIECES[name]['ball']
    z = (bottom_z if bottom_z is not None else table_z_at(xy)) + PIECES[name]['grasp']
    try:
        r = holes.locate_opening(ctx, m, xy, z, r_in=r_ball, r_out=r_ball + 0.006, log=log)
    except RuntimeError as e:
        log(f'{name}: ball not located ({e}); using the exterior estimate')
        return None
    if r['spread'] > tower.MAX_OPENING_SPREAD:
        log(f"{name}: ball views {1000 * r['spread']:.1f} mm apart; using the exterior estimate")
        return None
    log(f"{name}: ball at ({r['xy'][0]:.4f},{r['xy'][1]:.4f}), {1000 * math.dist(r['xy'], xy):.1f} mm from the exterior estimate")
    return r['xy']


def relocate_by_opening(ctx, m, name, state, log):
    """Re-measure a tower piece that has a visible opening (wrist camera) and move it and the
    supports under it in the record (a knock shifted the goblet+urn 10 mm once; a grasp at the
    stale xy closed on air)."""
    s, spec = state['pieces'][name], tower.OPENINGS[name]
    try:
        r = holes.locate_opening(ctx, m, s['xy'], s['bottom_z'] + spec['rim'], r_in=spec['r_in'],
                                 r_out=spec['r_out'], log=log)
    except RuntimeError as e:
        log(f'{name} opening: {e}; keeping its recorded xy')
        return
    if r['spread'] > tower.MAX_OPENING_SPREAD:
        log(f"{name} opening views {1000 * r['spread']:.1f} mm apart; keeping its recorded xy")
        return
    log(f"{name}: opening at ({r['xy'][0]:.4f},{r['xy'][1]:.4f}), {1000 * math.dist(r['xy'], s['xy']):.1f} mm from the record")
    sup = name
    while sup:
        state['pieces'][sup]['xy'] = list(r['xy'])
        sup = SLOTS[sup]['support']


def take_from_tower(ctx, m, name, state, log):
    s = state['pieces'][name]
    if name in tower.OPENINGS:
        relocate_by_opening(ctx, m, name, state, log)
    others = {k: v for k, v in world_from(state).items() if k != name}
    hits = tower.grasp_conflicts(others, name, s['xy'], s['bottom_z'], (GRASP_YAW,))
    if hits:
        return {'ok': False, 'why': f'hand would hit {hits} when taking {name} off the tower'}
    yaw = s.get('yaw') or GRASP_YAW
    try:
        pieces.pick(m, name, s['xy'][0], s['xy'][1], yaw=yaw, center=False, table_z=s['bottom_z'], log=log)
    except RuntimeError as e:
        return {'ok': False, 'why': f'pick from tower failed: {e}'}
    s.update(where='held', yaw=yaw)
    return {'ok': True}


def choose_spot(name, state, scene):
    """The planned spot for this piece if free and conflict-free for its later double-grasp
    pick, else the first alternative that is."""
    taken = [o['xy'] for o in scene['table'].values()] + [o['xy'] for o in scene['unknown']]
    taken += [s['xy'] for n, s in state['pieces'].items() if s.get('where') == 'table']
    world = world_from(state, scene['unknown'])
    for xy in [SPOTS[name]] + ALT_SPOTS:
        if any(math.dist(xy, t) < SPOT_CLEAR_R for t in taken):
            continue
        if tower.grasp_conflicts(world, name, xy, table_z_at(xy), (0.0, math.pi / 2)):
            continue
        return xy
    return None


def put_on_table(ctx, m, name, xy, state, log):
    try:
        r = pieces.place(m, name, xy[0], xy[1], table_z_at(xy), yaw=GRASP_YAW, tol=TABLE_PLACE_TOL, log=log,
                         straighten=straightener(ctx, name, log))
    except RuntimeError as e:
        return {'ok': False, 'why': f'placing on the table failed: {e}'}
    bz = r['contact_z'] - PIECES[name]['grasp']
    state['pieces'][name] = dict(where='table', xy=list(r['tcp_xy']), bottom_z=bz, known_bottom=True, yaw=r['yaw'])
    return {'ok': True, 'bottom_z': bz}


def verify_tower(ctx, m, state, log, expect):
    """Park, observe and compare tower occupancy with `expect` (set of pieces on the tower)."""
    perceive.park(ctx, m)
    fid = perceive.snapshot(ctx)
    cloud = perceive.wood_cloud(fid)
    xy = {n: (state['pieces'][n]['xy'] if state['pieces'].get(n, {}).get('where') == 'tower' else tower.slot_xy(n))
          for n in SLOTS}
    pres = perceive.slot_presence(cloud, xy)
    # thresholds rise with the stack level, so a piece under an expected one also reads present
    wrong = [n for n in SLOTS if pres[n][1] != (n in expect)]
    log(f'verify {fid}: ' + ', '.join(f'{n} {pres[n][0]}{"+" if pres[n][1] else "-"}' for n in SLOTS))
    return {'ok': not wrong, 'frame': fid, 'wrong': wrong, 'points': {n: pres[n][0] for n in SLOTS}}


# ----------------------------------------------------------------------------- main

def assemble(ctx, m, state, log, fresh=False, only=None):
    scene = survey(ctx, m, state, log, fresh=fresh)
    sync_state(state, scene)
    save_state(state)
    done = []
    for name in ASSEMBLY_ORDER:
        if scene['on_tower'][name]:
            log(f'{name}: already on the tower')
            continue
        if name not in scene['table']:
            return {'ok': False, 'step': f'find {name}', 'why': f'{name} is neither on the table nor on the tower',
                    'done': done}
        if only is not None and len(done) >= only:
            return {'ok': True, 'partial': True, 'done': done}
        t0 = time.time()
        measure_opening(ctx, m, name, state, log)
        r = pick_from_table(ctx, m, name, state, scene, log)
        movable = [b for b in r.get('blockers', ()) if state['pieces'].get(b, {}).get('where') == 'table']
        if movable and len(movable) == len(r['blockers']):
            for b in movable:                         # a piece not needed yet stands in the way: move it
                rb = clear_blocker(ctx, m, b, state, scene, log)
                save_state(state)
                if not rb['ok']:
                    return {'ok': False, 'step': f'move {b} out of the way of {name}', **rb, 'done': done}
            r = pick_from_table(ctx, m, name, state, scene, log)
        save_state(state)
        if not r['ok']:
            return {'ok': False, 'step': f'pick {name}', **r, 'done': done}
        r = put_on_tower(ctx, m, name, state, log)
        save_state(state)
        if not r['ok'] and r.get('lost'):          # dropped while lifting: find it again and retry once
            log(f'{name}: {r["why"]}; re-surveying')
            scene.update(survey(ctx, m, state, log))
            if name in scene['table']:
                o = scene['table'][name]
                state['pieces'][name] = dict(where='table', xy=o['xy'], bottom_z=o['table_z'],
                                             known_bottom=False, yaw=o.get('yaw'))
                r = pick_from_table(ctx, m, name, state, scene, log)
                if r['ok']:
                    r = put_on_tower(ctx, m, name, state, log)
                save_state(state)
        if not r['ok']:
            return {'ok': False, 'step': f'put {name}', **r, 'done': done}
        tilt = check_upright(ctx, m, name, state, log)
        if tilt is not None and tilt > align.MAX_SEATED_TILT:     # caught on the rim: take it out, redo once
            log(f'{name}: tilted {tilt:.1f} deg on the tower, re-seating')
            spot = choose_spot(name, state, scene)
            rr = take_from_tower(ctx, m, name, state, log)
            if rr['ok'] and spot is not None:
                rr = put_on_table(ctx, m, name, spot, state, log)
            if rr['ok']:
                scene['table'][name] = dict(xy=state['pieces'][name]['xy'], table_z=state['pieces'][name]['bottom_z'])
                rr = pick_from_table(ctx, m, name, state, scene, log)
            if rr['ok']:
                rr = put_on_tower(ctx, m, name, state, log)
            save_state(state)
            tilt = check_upright(ctx, m, name, state, log) if rr['ok'] else None
            if not rr['ok'] or (tilt is not None and tilt > align.MAX_SEATED_TILT):
                return {'ok': False, 'step': f're-seat {name}', 'why': rr.get('why', f'still tilted {tilt:.1f} deg'), 'done': done}
        expect = {n for n, s in state['pieces'].items() if s.get('where') == 'tower'}
        v = verify_tower(ctx, m, state, log, expect)
        if not v['ok']:
            return {'ok': False, 'step': f'verify {name}', 'why': f"tower check disagrees for {v['wrong']}",
                    'frame': v['frame'], 'done': done}
        done.append(name)
        log(f'{name}: on the tower ({time.time() - t0:.0f} s)')
    return {'ok': True, 'done': done}


def disassemble(ctx, m, state, log, only=None):
    scene = survey(ctx, m, state, log)
    sync_state(state, scene)
    save_state(state)
    done = []
    for name in DISASSEMBLY_ORDER:
        if not scene['on_tower'][name]:
            continue
        if only is not None and len(done) >= only:
            return {'ok': True, 'partial': True, 'done': done}
        t0 = time.time()
        spot = choose_spot(name, state, scene)
        if spot is None:
            return {'ok': False, 'step': f'spot for {name}', 'why': 'no free conflict-free table spot', 'done': done}
        r = take_from_tower(ctx, m, name, state, log)
        save_state(state)
        if not r['ok']:
            return {'ok': False, 'step': f'take {name}', **r, 'done': done}
        r = put_on_table(ctx, m, name, spot, state, log)
        save_state(state)
        if not r['ok']:
            return {'ok': False, 'step': f'put {name} on table', **r, 'done': done}
        done.append(name)
        log(f'{name}: on the table at ({spot[0]:.3f},{spot[1]:.3f}) ({time.time() - t0:.0f} s)')
    expect = {n for n, s in state['pieces'].items() if s.get('where') == 'tower'}
    v = verify_tower(ctx, m, state, log, expect)
    return {'ok': v['ok'], 'done': done, 'verify': v}


def main(ctx, mode='assemble', fresh=False, only=None, log=print):
    m = Motion(ctx, log=log)
    state = load_state()
    held = [n for n, p in state['pieces'].items() if p.get('where') == 'held']
    if held:     # the gripper's is_grasped flag reads False for our width-0 grasps, so trust the record
        return {'ok': False, 'step': 'start', 'why': f'record says {held} is held; put it down first'}
    g = m.status()['gripper']
    if g.get('max_width', 0.08) < MIN_GRIPPER_SPAN:
        return {'ok': False, 'step': 'start', 'why': f"gripper max_width {1000 * g['max_width']:.1f} mm: it was homed "
                f"with something between the fingers (widths read low, cannot close fully); ask the operator to re-home it"}
    if g['width'] < 0.03:
        m.gripper_open(0.06)
    t0 = time.time()
    if mode == 'assemble':
        r = assemble(ctx, m, state, log, fresh=fresh, only=only)
    elif mode == 'disassemble':
        r = disassemble(ctx, m, state, log, only=only)
    elif mode == 'survey':
        r = survey(ctx, m, state, log, fresh=fresh)
        r = {'ok': True, 'table': {n: o['xy'] for n, o in r['table'].items()}, 'on_tower': r['on_tower'],
             'unknown': [o['xy'] for o in r['unknown']], 'frame': r['frame']}
    else:
        raise ValueError(mode)
    try:
        perceive.park(ctx, m)
    except Exception as e:                          # report anyway
        log(f'park failed: {e}')
    r['seconds'] = round(time.time() - t0)
    return r


if __name__ == '__main__':
    from api.runtime import Context
    args = sys.argv[1:]
    mode = args[0] if args else 'assemble'
    only = int(args[args.index('--only') + 1]) if '--only' in args else None
    run_dir = RUNS / time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / 'log.txt', 'w')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'mode {mode} args {args}')
    with Context() as session:
        report = main(session, mode=mode, fresh='--fresh' in args, only=only, log=log)
    (run_dir / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    log(json.dumps(report, default=str))
