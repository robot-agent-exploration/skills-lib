"""Step runner: a task is a list of Step(name, primitive, **kwargs); each primitive is fn(rt, **kwargs) -> dict
and raises StepFailed when its precondition or completion check fails.

Checkpoints (agent in the loop): a step built with `checkpoint=` stops the run after it finishes (when the run
has checkpoints on). The runner saves the world and returns {'checkpoint': {step, next, brief}}: the brief is the
perception question for a subagent / Gemini (frames, question, expected answer, pass_if, on_fail), built from the
step's result. The agent answers it and resumes with start=<next>. A step may return {'goto': <step>} to loop.

Monitoring (closed loop by a subagent): while a run is going the runner writes <run>/status.json (current step,
time) and every motion tool call first checks for a STOP file (<run>/STOP or <workspace>/output/STOP). A monitor
(e.g. a haiku subagent watching runtime/stream) writes the file with its reason; the run stops before its next
motion, holds and reports the reason. A motion already dispatched finishes (the host serializes tool calls)."""
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import robot
from .world import World

MOTION_TOOLS = {'rby1_base_move', 'rby1_cartesian', 'rby1_gripper', 'rby1_jog', 'rby1_joint_group',
                'rby1_joint_impedance', 'rby1_move_body', 'rby1_move_eef', 'rby1_move_joints',
                'rby1_optimal_control', 'rby1_look_at'}


class StepFailed(Exception):
    pass


class StopRequested(Exception):
    """A monitor asked the run to stop (STOP file). Not a RuntimeError: primitives that treat tool errors as
    handled stops must not swallow it."""


class Step:
    def __init__(self, step_name, fn, /, checkpoint=None, **kwargs):
        """checkpoint: None (no stop), True (stop, generic brief), a dict brief (frames filled from the result)
        or fn(result) -> brief."""
        self.name, self.fn, self.checkpoint, self.kwargs = step_name, fn, checkpoint, kwargs


class GuardedCtx:
    """The tool context with a STOP-file check before every motion tool call."""

    def __init__(self, ctx, stop_files):
        self._ctx, self.stop_files = ctx, [Path(p) for p in stop_files]

    def stop_reason(self):
        for f in self.stop_files:
            if f.exists():
                return f'{f}: {f.read_text()[:300].strip()}'
        return None

    def request(self, method, params):
        if method == 'tools/call' and params.get('name') in MOTION_TOOLS:
            why = self.stop_reason()
            if why:
                raise StopRequested(why)
        return self._ctx.request(method, params)

    def call(self, tool, **arguments):
        result = self.request('tools/call', {'name': tool, 'arguments': arguments})
        if result.get('isError'):
            text = next((b.get('text', '') for b in result.get('content', []) if b.get('type') == 'text'), '')
            raise RuntimeError('MCP tool failed: ' + text)
        return result

    def value(self, tool, **arguments):
        result = self.call(tool, **arguments)
        return json.loads(next(b['text'] for b in result['content'] if b.get('type') == 'text'))

    def __getattr__(self, k):
        return getattr(self._ctx, k)


def workspace_root():
    return next((p for p in Path(__file__).resolve().parents if (p / 'api' / 'runtime.py').exists()),
                Path.cwd())


class Runtime:
    """What a primitive gets: the (guarded) tool context, the world, logging, observations (with the odometry at
    capture) and base motions (logged, so a reverse can be limited to the straight path just driven)."""

    def __init__(self, ctx, outdir, world=None, log_print=True):
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ctx = ctx if isinstance(ctx, GuardedCtx) else GuardedCtx(
            ctx, [self.out / 'STOP', workspace_root() / 'output' / 'STOP'])
        self.world = world or World()
        self.frames = []
        self.frame_odom = self.world.notes.setdefault('frame_odom', {})
        self.motions = self.world.notes.setdefault('motions', [])
        self._logf = open(self.out / 'log.txt', 'a')
        self._print = log_print
        self.step = None

    def log(self, *a):
        line = f'[{time.strftime("%H:%M:%S")}] ' + ' '.join(str(x) for x in a)
        if self._print:
            print(line, flush=True)
        self._logf.write(line + '\n')
        self._logf.flush()

    def heartbeat(self, **kw):
        """<run>/status.json for a monitor: current step and time (and anything the step adds)."""
        d = {'step': self.step, 'time': time.strftime('%H:%M:%S'), 't': time.time(), **kw}
        (self.out / 'status.json').write_text(json.dumps(d, default=str))

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


def frame_paths(res):
    """jpg paths of the frames a step result names (keys frame / frames / *_frame)."""
    ids = []
    for k, v in (res or {}).items():
        if k == 'frames' and isinstance(v, (list, tuple)):
            ids += list(v)
        elif (k == 'frame' or k.endswith('_frame')) and isinstance(v, str):
            ids.append(v)
    root = workspace_root() / 'runtime' / 'frames'
    return [str(root / f / 'head_camera.jpg') if not str(f).endswith('.jpg') else str(f) for f in ids]


def make_brief(step, res):
    cp = step.checkpoint
    if callable(cp):
        brief = cp(res)
    elif isinstance(cp, dict):
        brief = dict(cp)
    else:
        brief = {'subagent': 'haiku', 'question': f'Review the result of {step.name}: does the scene match it?'}
    brief.setdefault('frames', frame_paths(res))
    pics = [v for k, v in (res or {}).items() if k in ('picture', 'sides_picture') and v]
    if pics:
        brief['frames'] = list(brief['frames']) + pics
    return brief


def run_steps(rt, steps, start=None, stop_after=None, checkpoints=False, max_jumps=8, params=None):
    names = [s.name for s in steps]
    if start is not None and start not in names:
        return {'ok': False, 'why': f'unknown step {start}', 'steps': [], 'checkpoint': None, 'next': start}
    report = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': [], 'outdir': str(rt.out)}
    ok, why, checkpoint, nxt = True, '', None, None
    i, jumps = (names.index(start) if start else 0), 0
    first = i
    while i < len(steps):
        step = steps[i]
        t0 = time.time()
        rt.step = step.name
        rt.heartbeat(state='running')
        rt.log(f'== {step.name} {step.kwargs if step.kwargs else ""}')
        res = {}
        kwargs = dict(step.kwargs)
        if params and step.name == names[first]:
            kwargs.update(params)                  # the agent's choice for the step it resumes
        try:
            res = step.fn(rt, **kwargs) or {}
            report['steps'].append({'step': step.name, 'ok': True, 'seconds': round(time.time() - t0, 1), 'result': res})
            rt.log(f'== {step.name} ok ({time.time() - t0:.0f} s): {json.dumps(res, default=str)[:400]}')
        except StopRequested as e:
            ok, why = False, f'{step.name}: STOP requested by the monitor: {e}'
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
                rt.ctx._ctx.value('rby1_hold')
            except Exception:
                pass
            nxt = step.name
            break
        if res.get('goto') and jumps < max_jumps:
            jumps += 1
            i = names.index(res['goto'])
        else:
            i += 1
        if stop_after == step.name:
            nxt = names[i] if i < len(names) else None
            break
        if checkpoints and step.checkpoint:
            nxt = names[i] if i < len(names) else None
            checkpoint = {'step': step.name, 'next': nxt, 'brief': make_brief(step, res)}
            rt.log(f'== CHECKPOINT after {step.name}; next {nxt}; brief {json.dumps(checkpoint["brief"])[:600]}')
            break
    rt.heartbeat(state='stopped' if not ok else ('checkpoint' if checkpoint else 'done'), why=why)
    report.update({'ok': ok, 'why': why, 'checkpoint': checkpoint, 'next': nxt, 'held': rt.world.held,
                   'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'frames': rt.frames})
    try:
        from . import vlm
        report['gemini'] = vlm.usage()
    except Exception:
        pass
    (rt.out / f'report_{time.strftime("%H%M%S")}.json').write_text(json.dumps(report, indent=1, default=str))
    return report


def run_task(ctx, steps, task='task', start=None, stop_after=None, resume=None, checkpoints=True, outdir=None,
             params=None):
    """Run a composition (list of Step) in a new or resumed run directory under <workspace>/output/runs/.
    -> {ok, why, checkpoint, next, outdir, measured}; ok is True only when the last step finished."""
    if resume:
        outdir = Path(resume)
        world = World.load(outdir / 'world.json') if (outdir / 'world.json').exists() else World()
    else:
        outdir = Path(outdir or workspace_root() / 'output' / 'runs' / f'{time.strftime("%Y%m%d_%H%M%S")}_{task}')
        world = World()
    rt = Runtime(ctx, outdir, world)
    rep = run_steps(rt, steps, start=start, stop_after=stop_after, checkpoints=checkpoints, params=params)
    done = rep['ok'] and rep['checkpoint'] is None and stop_after is None and rep['next'] is None
    why = rep['why'] or (f"checkpoint after {rep['checkpoint']['step']}: answer the brief, then resume with "
                         f"start={rep['next']!r}" if rep['checkpoint'] else ('finished' if done else 'stopped'))
    return {'ok': bool(done), 'why': why, 'checkpoint': rep['checkpoint'], 'next': rep['next'], 'outdir': str(outdir),
            'measured': {s['step']: s.get('result') for s in rep['steps'] if s['ok']}, 'gemini': rep.get('gemini')}
