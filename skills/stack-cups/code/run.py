"""Entry of the stack-cups skill: main(ctx, step='stack', args=(), log=print) -> {ok, measured, why}.

Steps (run_cups): scan, pick, align, insert, stack. ok None = the automatic checks were unsure or disagreed: the
returned checkpoint (frames + question) is for the agent, who continues with insert or align --dx --dy.
    main(ctx, 'scan', ['silver cup on the left', 'silver cup in the front middle'])
    main(ctx, 'stack', ['silver cup on the left', 'silver cup in the front middle'])     # base, cup to put in
    main(ctx, 'insert', ['left'])                                                      # after the agent looked"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack_cups import run_cups  # noqa: E402


def main(ctx, step='stack', args=(), log=print):
    out = run_cups.main(ctx, step, list(args), log=log)
    out.setdefault('ok', False)
    out.setdefault('why', '')
    out.setdefault('measured', {k: v for k, v in out.items() if k not in ('ok', 'why', 'measured')})
    return out
