"""Skill entry (bimanual-bucket-rim-pnp): pick a ~5-gallon bucket off the floor with BOTH grippers pinching its rim
(fingers OPEN around the wall, one in, one out), carry it and place it on a black-top cabinet / table next to what
stands there. Step-wise with perception checkpoints (agent / haiku / Gemini) through the bimanual-box-pnp runner.

    main(ctx, start=None, resume=None, checkpoints=True, params=None, stop_after=None)
      -> {ok, measured, why, checkpoint: {step, next, brief}, next, outdir}
ok is True only when the last step (verify) finished; at a checkpoint answer the brief and resume with
main(ctx, resume=r['outdir'], start=r['next'], params={...}). Steps and briefs: controller.py, code in steps.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bimanual_bucket_rim_pnp import controller  # noqa: E402


def main(ctx, start=None, resume=None, checkpoints=True, params=None, stop_after=None):
    return controller.main(ctx, start=start, resume=resume, checkpoints=checkpoints, params=params,
                           stop_after=stop_after)


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--resume')
    ap.add_argument('--stop-after')
    ap.add_argument('--params')
    ap.add_argument('--no-checkpoints', action='store_true')
    a = ap.parse_args()
    from api.runtime import Context
    with Context() as session:
        r = main(session, start=a.start, resume=a.resume, checkpoints=not a.no_checkpoints,
                 params=json.loads(a.params) if a.params else None, stop_after=a.stop_after)
    print(json.dumps(r, indent=1, default=str)[:5000])
