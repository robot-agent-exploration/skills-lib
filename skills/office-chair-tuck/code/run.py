"""Entry of the office-chair-tuck skill: one step of the step-wise chair controller (see run_tuck.py).

    main(ctx, step='verify', args=())  ->  {ok, measured, why, frames, checkpoint}

Steps: find | approach [plan] | desk | measure | square | grasp | transport [pull] [plan] | pull <m> | turn_to <yaw> | drive <m> | tuck | release | verify.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from office_chair_tuck import run_tuck as _rt  # noqa: E402


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
