"""find-the-cube entry: the shell game on the Franka FR3 (watch the hide-and-shuffle from the
wrist camera, lift the cup that holds the object away, grasp and lift the object).

Library wrapper around the session's `controller.main` (unchanged pipeline): it adds the run
directory, report.json / log.txt (as the session's CLI wrote them) and the library's
{ok, measured, why} return. Run from the workspace root (api/, runtime/, output/):

    python <skill>/code/run.py [--setup] [--resume JSON] [--object-name NAME]

or, from Python, main(ctx, resume=None, setup=False, object_name='white cube').
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller as C  # noqa: E402


def _why(rep):
    if rep.get('ok'):
        return ''
    failed = rep.get('failed') or 'exception'
    step = rep.get('steps', {}).get(failed) or {}
    return f"{failed}: {step.get('why') or rep.get('error') or 'failed'}"


def _measured(rep, out):
    steps = {k: dict(v) for k, v in rep.get('steps', {}).items()}
    if 'wait_shuffle' in steps:                       # the reply carries the whole robot status
        reply = steps['wait_shuffle'].pop('reply', None) or {}
        steps['wait_shuffle']['operator_reply'] = reply.get('reply') if isinstance(reply, dict) else reply
    find = steps.get('find_cube', {})
    ident = steps.get('identify_cup', {})
    cup = steps.get('rescue_cup') or steps.get('remove_cup') or {}
    grasp = steps.get('grasp_cube', {})
    run_dir = str(out)
    return {
        'failed_step': rep.get('failed'),
        'error': rep.get('error'),
        'duration_s': rep.get('duration_s'),
        'run_dir': run_dir,
        'object_xy': (find.get('cube') or {}).get('xy'),
        'object_model': find.get('model'),
        'cover_id': ident.get('cover_id'),
        'cup_px': ident.get('cup_px'),
        'events': ident.get('events'),
        'cup_xyz': cup.get('cup'),
        'cup_dropped_at': cup.get('drop'),
        'grasp_width_m': grasp.get('width'),
        'grasp_yaw_deg': grasp.get('yaw'),
        'grasp_attempts': grasp.get('attempts'),
        'steps': steps,
        'checkpoints': [
            {'step': 'identify_cup', 'subagent': 'agent (you) or a strong model',
             'frames': [run_dir + '/tracks.jpg'],
             'question': 'Follow the red C track: is C the cup that covered the object (or that the object was '
                         'dropped into), and does it end on a real cup, apart from the others?',
             'answer': 'yes / no + which cup (pixel) holds the object',
             'pass_if': 'yes',
             'on_fail': 're-run identification with replay.py on run_dir/wrist (or from the stream), then '
                        'run again with resume={"final_px": ..., "cover_id": k}'},
            {'step': 'grasp_cube (after ok)', 'subagent': 'haiku',
             'frames': ['franka_observe wrist + left after the run'],
             'question': 'Is the object held between the gripper fingers, above the table?',
             'answer': 'yes / no',
             'pass_if': 'yes and 0.006 < grasp_width_m < 0.078',
             'on_fail': 'look where the object is; it may still be under or inside the lifted cup'},
        ],
    }


def main(ctx, resume=None, setup=False, object_name='white cube', out_dir=None):
    """Whole task. resume: {'final_px': [[u, v], ...], 'cover_id': k, 'model': {...}} from an
    earlier identification (replay.py) starts at locate_cups on the scene as it is.
    Returns {'ok', 'measured', 'why'}; report.json and log.txt go to output/runs/<timestamp>/."""
    out = Path(out_dir) if out_dir else C.ROOT / 'output' / 'runs' / time.strftime('%Y%m%dT%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    del C.LOG[:]                                      # one log per run when called repeatedly
    t_start = time.time()
    rep = C.main(ctx, out_dir=out, resume=resume, setup=setup, object_name=object_name)
    rep['duration_s'] = round(time.time() - t_start, 1)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    (out / 'log.txt').write_text('\n'.join(C.LOG))
    measured = _measured(rep, out)
    if resume:                                        # the identification came from the caller
        measured['cover_id'] = resume.get('cover_id')
    return {'ok': bool(rep.get('ok')), 'measured': measured, 'why': _why(rep)}


if __name__ == '__main__':
    import argparse
    from api.runtime import Context
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resume', help='JSON {"final_px": [[u, v], ...], "cover_id": k} to start at locate_cups')
    ap.add_argument('--setup', action='store_true', help='first ask the operator to lay out the scene')
    ap.add_argument('--object-name', default='white cube', help='object named in the operator request')
    args = ap.parse_args()
    with Context() as session:
        r = main(session, resume=json.loads(args.resume) if args.resume else None, setup=args.setup,
                 object_name=args.object_name)
    print(json.dumps(r, indent=1, default=str))
