"""Entry of the air-fry-pan skill: main(ctx, steps=('open', 'banana', 'close'), resume=None, run_dir=None)
-> {ok, measured, why}.

A thin wrapper (added on import into the library) around controller.main, which returns the session's
report dict {run_dir, steps: {prepare, perceive, open, banana, ball, close}, ok, error?, final_state?,
seconds} and writes it to <run_dir>/report.json next to calls.jsonl (every tool call and reply, frame
ids, plan decisions). run_dir defaults to output/runs/<timestamp>/ under the working directory.

    main(ctx)                                        # open, banana, close
    main(ctx, steps='open,banana,ball,close')        # also an orange ball (tested once)
    main(ctx, steps='none')                          # perceive only: moves to the top-view pose and home
    main(ctx, steps='close', resume='<run dir name>')  # finish an episode stopped after the open/banana steps
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller  # noqa: E402


def main(ctx, steps=('open', 'banana', 'close'), resume=None, run_dir=None):
    if isinstance(steps, str):
        steps = tuple(s.strip() for s in steps.split(','))
    report = controller.main(ctx, steps=tuple(steps), run_dir=run_dir, resume=resume)
    return {'ok': bool(report.get('ok')), 'measured': report, 'why': report.get('error', '')}
