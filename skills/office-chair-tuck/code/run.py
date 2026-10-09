"""Entry of the office-chair-tuck skill: one step of the step-wise chair controller (see run_tuck.py).

    main(ctx, step='verify', args=())  ->  {ok, measured, why, frames, checkpoint}

Steps: find | approach [plan] | desk | measure | square | grasp | transport [pull] [plan] | pull <m> | turn_to <yaw> | drive <m> | tuck | release | verify.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('office_chair_run_tuck', HERE / 'run_tuck.py')
_rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rt)


def main(ctx, step='verify', args=(), log=print):
    r = _rt.main(ctx, step, *[str(a) for a in args], log=log)
    r.setdefault('measured', {k: v for k, v in r.items() if k not in ('ok', 'why', 'frames', 'checkpoint')})
    r.setdefault('why', '')
    return r


if __name__ == '__main__':
    import json
    sys.path.insert(0, str(Path.cwd()))
    from api.runtime import Context
    with Context() as ctx:
        print(json.dumps(main(ctx, sys.argv[1] if len(sys.argv) > 1 else 'verify', sys.argv[2:]), indent=1,
                         default=str))
