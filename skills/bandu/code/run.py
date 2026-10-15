"""Entry of the bandu skill: main(ctx, mode='assemble', ...) -> {ok, measured, why}.

A thin wrapper around controller.main (the session's controller, unchanged), which returns
{ok, step, why, done, ...}; everything except ok and why is passed on in `measured`.
Like the session's command line, each call logs to output/runs/<timestamp>/log.txt and writes
report.json there; the piece record is output/bandu/state.json (all relative to the working
directory, i.e. run from the workspace root).

    main(ctx, mode='assemble')            # pieces standing on the table -> the tower
    main(ctx, mode='assemble', fresh=True)  # ignore the piece record, label pieces from images
    main(ctx, mode='disassemble')         # tower -> table spots (practice reset)
    main(ctx, mode='survey')              # park, observe, report what is where (no other motion)

Scene parameters (only the defaults were run on the robot, 2026-10-10):
    base_center (x, y) m and base_top_z m of the 2x2 block base: default (0.5245, 0.032), 0.110;
    the skill does not localize the base, so pass values you measured.
    spots {piece: (x, y)}: where disassembly sets pieces down (default tower.SPOTS).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandu import controller  # noqa: E402
from bandu import tower  # noqa: E402

RUNS = Path('output') / 'runs'


def main(ctx, mode='assemble', fresh=False, only=None, base_center=None, base_top_z=None, spots=None, log=print):
    run_dir = RUNS / time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / 'log.txt', 'w')

    def _log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        if log is not None:
            log(line)
        logf.write(line + '\n')
        logf.flush()

    saved_base, saved_spots = dict(tower.BASE), dict(tower.SPOTS)
    try:
        if base_center is not None:
            tower.BASE['center'] = (float(base_center[0]), float(base_center[1]))
        if base_top_z is not None:
            tower.BASE['top_z'] = float(base_top_z)
        if spots:
            tower.SPOTS.update({n: (float(v[0]), float(v[1])) for n, v in spots.items()})
        _log(f'mode {mode} fresh {fresh} only {only} base {tower.BASE["center"]} top {tower.BASE["top_z"]}')
        r = controller.main(ctx, mode=mode, fresh=fresh, only=only, log=_log)
    finally:
        tower.BASE.clear()
        tower.BASE.update(saved_base)
        tower.SPOTS.clear()
        tower.SPOTS.update(saved_spots)
    (run_dir / 'report.json').write_text(json.dumps(r, indent=1, default=str))
    _log(json.dumps(r, default=str))
    logf.close()
    ok = bool(r.get('ok'))
    measured = {k: v for k, v in r.items() if k not in ('ok', 'why')}
    measured['run_dir'] = str(run_dir)
    return {'ok': ok, 'measured': measured, 'why': r.get('why', '' if ok else f"stopped at {r.get('step')}")}
