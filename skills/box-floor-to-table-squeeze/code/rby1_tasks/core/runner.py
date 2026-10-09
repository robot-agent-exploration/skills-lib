"""Step runner: a task is a list of Step(name, primitive, **kwargs); each primitive is fn(rt, **kwargs) -> dict
and raises StepFailed when its precondition or completion check fails.

Checkpoints (agent in the loop): steps built with checkpoint=True stop the run after they finish; the runner
saves the world and returns {'checkpoint': name, 'next': next step}. The agent looks at
the step's result / pictures and resumes with start=<next> (or re-runs a planning step). A step may return
{'goto': <step name>} to loop (e.g. re-plan an approach until the stance is reached)."""
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import robot
from .world import World


class StepFailed(Exception):
    pass


class Step:
    def __init__(self, step_name, fn, /, checkpoint=False, **kwargs):
        self.name, self.fn, self.checkpoint, self.kwargs = step_name, fn, checkpoint, kwargs


class Runtime:
    """What a primitive gets: the tool context, the world, logging, observations (with the odometry at capture)
    and base motions (logged, so a reverse can be limited to the straight path just driven)."""

    def __init__(self, ctx, outdir, world=None, log_print=True):
        self.ctx = ctx
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.world = world or World()
        self.frames = []
        self.frame_odom = self.world.notes.setdefault('frame_odom', {})
        self.motions = self.world.notes.setdefault('motions', [])
        self._logf = open(self.out / 'log.txt', 'a')
        self._print = log_print

    def log(self, *a):
        line = f'[{time.strftime("%H:%M:%S")}] ' + ' '.join(str(x) for x in a)
        if self._print:
            print(line, flush=True)
        self._logf.write(line + '\n')
        self._logf.flush()

    def v(self, tool, **kw):
        return self.ctx.value(tool, **kw)

    def status(self):
        return robot.status(self.ctx)

    def odom(self):
        return self.status()['odometry']

    def pose(self):
        o = self.odom()
        return (o['x'], o['y'], o['yaw'])

    def held_box_region(self):
        """Base-frame box around a held object (between the balls), removed from obstacle checks."""
        if not self.world.held:
            return None
        s = self.status()
        r, l = np.array(s['right_ee']['position']), np.array(s['left_ee']['position'])
        lo, hi = np.minimum(r, l), np.maximum(r, l)
        return {'min': [lo[0] - 0.10, lo[1] - 0.02, lo[2] - 0.45], 'max': [hi[0] + 0.30, hi[1] + 0.02, hi[2] + 0.05]}

    def observe(self, point=None):
        if point is not None:
            robot.look(self.ctx, point)
        fid, odom = robot.observe_full(self.ctx)
        self.frames.append(fid)
        self.frame_odom[fid] = odom
        return fid

    def base(self, forward=0.0, turn=0.0, slow=False):
        out = robot.base_move(self.ctx, self.log, forward=forward, turn=turn, slow=slow)
        for r in out:
            self.motions.append({'forward': r['forward'], 'turn': r['turn']})
        return out

    def reversible(self):
        """Straight distance driven forward since the last turn (minus reverses since): the only reverse allowed."""
        d = 0.0
        for m in reversed(self.motions):
            if abs(m['turn']) > 0.05:
                break
            d += m['forward']
        return max(0.0, d)

    def save(self):
        self.world.save(self.out / 'world.json')


def run_steps(rt, steps, start=None, stop_after=None, checkpoints=False, max_jumps=8, params=None):
    names = [s.name for s in steps]
    if start is not None and start not in names:
        return {'ok': False, 'why': f'unknown step {start}', 'steps': []}
    report = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': [], 'outdir': str(rt.out)}
    ok, why, checkpoint, nxt = True, '', None, None
    i, jumps = (names.index(start) if start else 0), 0
    first = i
    while i < len(steps):
        step = steps[i]
        t0 = time.time()
        rt.log(f'== {step.name} {step.kwargs if step.kwargs else ""}')
        res = {}
        kwargs = dict(step.kwargs)
        if params and step.name == names[first]:
            kwargs.update(params)                  # the agent's choice for the step it resumes
        try:
            res = step.fn(rt, **kwargs) or {}
            report['steps'].append({'step': step.name, 'ok': True, 'seconds': round(time.time() - t0, 1), 'result': res})
            rt.log(f'== {step.name} ok ({time.time() - t0:.0f} s): {json.dumps(res, default=str)[:400]}')
        except StepFailed as e:
            ok, why = False, f'{step.name}: {e}'
        except RuntimeError as e:                 # a refused or stopped tool call
            ok, why = False, f'{step.name}: tool error {str(e)[:800]}'
        except Exception as e:                    # a bug: stop and keep the trace
            ok, why = False, f'{step.name}: {type(e).__name__} {e}\n{traceback.format_exc()}'
        rt.save()
        if not ok:
            report['steps'].append({'step': step.name, 'ok': False, 'seconds': round(time.time() - t0, 1), 'why': why})
            rt.log(f'== {step.name} FAILED: {why}')
            try:
                rt.v('rby1_hold')
            except Exception:
                pass
            break
        if res.get('goto') and jumps < max_jumps:
            jumps += 1
            i = names.index(res['goto'])
        else:
            i += 1
        if stop_after == step.name:
            break
        if checkpoints and step.checkpoint:
            checkpoint, nxt = step.name, (names[i] if i < len(names) else None)
            rt.log(f'== CHECKPOINT after {step.name}; next {nxt}')
            break
    report.update({'ok': ok, 'why': why, 'checkpoint': checkpoint, 'next': nxt, 'held': rt.world.held,
                   'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'frames': rt.frames})
    (rt.out / f'report_{time.strftime("%H%M%S")}.json').write_text(json.dumps(report, indent=1, default=str))
    return report
