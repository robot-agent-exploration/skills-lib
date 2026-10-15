"""Entry of peg-insertion: chain the session's step functions in the order the 2026-10-07 full run used them.

THIS WRAPPER NEVER RAN ON THE ROBOT. It was written when the skill was imported into the library. In the
session the agent ran the same functions one at a time (step.py, l3_tilt_insert.py, fitpeg.py from the shell)
and did the grasps, carries, releases and surface touches between them with direct tool calls. main() only
dispatches to those functions, in the order the caller lists, and stops at the first step that fails.

    main(ctx, steps=[{'step': name, ...params}, ...]) -> {ok, measured: {steps: [...]}, why}

Steps (all positions base frame, metres; yaws in degrees unless the name says rad):
  register   {frame_id, init: [cx, cy, yaw_rad], z_top, z_slope_x=0, x_ref=0, span?, steps?, refine?}
             register.fit on a saved wrist frame (no motion); writes output/inspect/register_<frame>.jpg (CAD
             outlines at the fitted pose) and returns a checkpoint for it. Add the wrist-camera bias yourself.
  fit_peg    {frame_id, shape, init_xy, z_top, span=0.006}  fitpeg.fit_both (no motion); writes the overlay
             to output/inspect/ and returns a perception checkpoint for it.
  move       {pos, quat?, tol_mm?, timeout_s?, ...}  motion.move (franka_move_pose, stall-tolerant)
  gripper    {action, width?, force?, speed?}       franka_gripper, passed through (the agent's hand calls)
  insert     {hole_xy, yaw_deg, z_surface, depth, search='hop'}  insert.insert (phase-1 L1/L2 probe-down)
  hop        {xy, z_surf, yaw_deg}                   step.hop: hover 1.5 mm above, guarded drop, rings
  centre     {centre, z, yaw_deg}                    step.centre: fixture.centre_in_hole (+-a, +-b walls)
  centre_tri {centre, z, yaw_deg, hole_yaw_deg}      step.centre_tri: fixture.centre_triangle
  l3         l3_tilt_insert params (psi_deg, c, rel_deg, shape, vertex, Lb, z_ref, z_board, hole, r_hv,
             margin, tilt_deg, depth_in, upright_steps, upright_press, yaw_wiggle_deg, Fb, Mb, target_depth)
  pull       l3_tilt_insert.wiggle_pull params (psi_deg, c, Lb, Fb, yaw_wiggle_deg?)

The order of the full run for one peg (the session's hexagon, 2026-10-07 02:28-02:49):
  fit_peg -> gripper open -> move above -> move down -> gripper grasp 55 N -> move up/over L1 ->
  hop (L1; the session did this drop by hand) -> gripper release, grasp again -> centre (L1) ->
  move up, over L2 (target = centred TCP + R(board yaw)(CAD_L2 - CAD_L1)) -> hop (L2) -> gripper release,
  grasp again centred -> centre (L2) -> [agent: z_ref touch beside L3, free-space wrench Fb/Mb] ->
  move above L3 -> l3 -> gripper open -> move up.
Logs: every line printed by the steps also goes to output/runs/peg_insertion_<UTC time>.log (working dir =
workspace root, as the session's `tee output/runs/...`).
"""
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from peg_insertion import board  # noqa: E402
from peg_insertion import fitpeg  # noqa: E402
from peg_insertion import insert as insert_mod  # noqa: E402
from peg_insertion import l3_tilt_insert  # noqa: E402
from peg_insertion import motion  # noqa: E402
from peg_insertion import register  # noqa: E402
from peg_insertion import step as step_mod  # noqa: E402
from peg_insertion.geom import load_frame, project, rgb  # noqa: E402
from peg_insertion.plan import quat  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _register(ctx, p):
    f = load_frame(p['frame_id'], p.get('frames_dir'))
    z0, k, x0 = p['z_top'], p.get('z_slope_x', 0.0), p.get('x_ref', 0.0)
    z = (lambda xy: z0 + k * (xy[:, 0] - x0)) if k else z0
    kw = {key: tuple(p[key]) for key in ('span', 'steps') if key in p}
    if 'refine' in p:
        kw['refine'] = p['refine']
    score, pose = register.fit(f, 'wrist', tuple(p['init']), z, **kw)
    # overlay for the checkpoint: CAD hole outlines (red) and board outline (green) projected at the fitted pose
    out = Path('output/inspect')
    out.mkdir(parents=True, exist_ok=True)
    overlay = str(out / f'register_{p["frame_id"]}.jpg')
    img = rgb(f, 'wrist')
    polys = [(board.hole_outline_base(pose, name, 0.0), (0, 0, 255)) for name in board.MODEL['holes']]
    polys.append((board.board_outline_base(pose, 0.0), (0, 255, 0)))
    for P, colour in polys:
        P[:, 2] = z(P[:, :2]) if callable(z) else z
        uv = project(f, 'wrist', P).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [uv], True, colour, 1)
    cv2.imwrite(overlay, img)
    return {'ok': score > 0, 'score': score, 'pose': [float(v) for v in pose], 'overlay': overlay,
            'checkpoint': {'subagent': 'haiku', 'frames': [overlay],
                           'question': 'Does every red outline lie exactly on the rim of a hole in the board?',
                           'answer': 'yes | no, plus which outlines are off and by how many pixels', 'pass_if': 'yes',
                           'on_fail': 'multi-start fit (x +-1.5 cm, y +-3 cm, yaw +-4 deg) and look again'}}


def _fit_peg(ctx, p):
    out = Path('output/inspect')
    out.mkdir(parents=True, exist_ok=True)
    overlay = str(out / f'fit_{p["shape"]}_{p["frame_id"]}.jpg')
    best, _ = fitpeg.fit_both(p['frame_id'], p['shape'], tuple(p['init_xy']), p['z_top'], span=p.get('span', 0.006),
                              out=overlay)
    return {'ok': True, 'centre': best['centre'], 'yaw_deg': float(np.degrees(best['yaw'])), 'score': best['score'],
            'margin': best['margin'], 'polarity': best['polarity'],
            'checkpoint': {'subagent': 'haiku', 'frames': [overlay],
                           'question': "Does the red outline lie exactly on the peg's top face (same size and turn)?",
                           'answer': 'yes | no, plus the offset in pixels', 'pass_if': 'yes',
                           'on_fail': 'measure the top-face corners in a zoomed crop and back-project them '
                                      '(the session did this for the triangle); never grasp on a bad fit'}}


def _move(ctx, p):
    p = dict(p)
    r = motion.move(ctx, p.pop('pos'), p.pop('quat', None), p.pop('tol_mm', None), p.pop('timeout_s', None), **p)
    return {'ok': True, **r}


def _gripper(ctx, p):
    r = ctx.value('franka_gripper', **p)
    return {'ok': bool(r.get('is_grasped')) if p.get('action') == 'grasp' else True, **r}


def _insert(ctx, p):
    yaw = np.radians(p['yaw_deg'])
    return insert_mod.insert(ctx, tuple(p['hole_xy']), quat(yaw), p['z_surface'], p['depth'], yaw=yaw,
                             search=p.get('search', 'hop'))


STEPS = {
    'register': _register,
    'fit_peg': _fit_peg,
    'move': _move,
    'gripper': _gripper,
    'insert': _insert,
    'hop': step_mod.hop,
    'centre': step_mod.centre,
    'centre_tri': step_mod.centre_tri,
    'l3': lambda ctx, p: l3_tilt_insert.run(ctx, p),
    'pull': lambda ctx, p: l3_tilt_insert.wiggle_pull(ctx, p),
}


def main(ctx, steps, log_dir='output/runs'):
    """Run the listed steps in order; stop at the first one that fails. Returns {ok, measured, why}."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f'peg_insertion_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.log'
    done = []
    with open(log_path, 'a') as fh, contextlib.redirect_stdout(_Tee(sys.stdout, fh)):
        for k, spec in enumerate(steps):
            spec = dict(spec)
            name = spec.pop('step')
            if name not in STEPS:
                return {'ok': False, 'measured': {'steps': done, 'log': str(log_path)},
                        'why': f'step {k}: unknown step {name!r} (known: {sorted(STEPS)})'}
            print(f'== step {k}: {name} {spec}', flush=True)
            try:
                r = STEPS[name](ctx, spec)
            except motion.MotionError as e:
                done.append({'step': name, 'ok': False, 'error': str(e)})
                return {'ok': False, 'measured': {'steps': done, 'log': str(log_path)},
                        'why': f'step {k} ({name}): robot refused or failed a move: {e}'}
            r = dict(r) if isinstance(r, dict) else {'ok': bool(r), 'result': r}
            print(f'   -> {r}', flush=True)
            done.append({'step': name, **r})
            if r.get('ok') is False:
                why = r.get('reason') or r.get('stage') or 'step returned ok False'
                return {'ok': False, 'measured': {'steps': done, 'log': str(log_path)},
                        'why': f'step {k} ({name}) failed: {why}'}
    return {'ok': True, 'measured': {'steps': done, 'log': str(log_path)}, 'why': f'{len(done)} steps ok'}
