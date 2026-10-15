"""Thin, logged wrappers around the robot MCP tools."""
import json
import time
from pathlib import Path

import numpy as np

from kin import SAFE_MIN, SAFE_MAX

from air_fry_pan_params import MAX_JOINT_STEP, JOINT_TARGET_MARGIN  # hyper-parameters: air_fry_pan_params.py


class MotionError(RuntimeError):
    pass


class Robot:
    def __init__(self, ctx, log_dir):
        self.ctx = ctx
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_dir / 'calls.jsonl', 'a')

    def log(self, kind, **data):
        rec = {'t': round(time.time(), 3), 'kind': kind, **data}
        self._log.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
        self._log.flush()

    def call(self, tool, **args):
        try:
            out = self.ctx.value(tool, **args)
        except RuntimeError as e:
            self.log('error', tool=tool, args=args, error=str(e)[:2000])
            raise MotionError(f'{tool} failed: {str(e)[:600]}') from e
        self.log('call', tool=tool, args=args, reply=out)
        return out

    # --- state -------------------------------------------------------------------------
    def status(self):
        return self.call('franka_status')

    def joints(self):
        return np.array(self.status()['joints'])

    def require_armed(self):
        st = self.status()
        if not st['motion']['armed']:
            raise MotionError('motion is not armed by the operator')
        return st

    # --- motion ------------------------------------------------------------------------
    MAX_CART_STEP = 0.35    # m per move_pose call (service limit 0.40)

    def move_pose(self, pos, quat=None, relative=False, tol_mm=None, timeout_s=20.0):
        if not relative:
            cur = np.array(self.status()['tcp_position'])
            dist = float(np.linalg.norm(np.asarray(pos, float) - cur))
            if dist > self.MAX_CART_STEP:      # split long straight moves into shorter segments
                n = int(np.ceil(dist / self.MAX_CART_STEP))
                for k in range(1, n):
                    self._move_pose(cur + k / n * (np.asarray(pos, float) - cur), quat, False, None, timeout_s)
        return self._move_pose(pos, quat, relative, tol_mm, timeout_s)

    def _move_pose(self, pos, quat=None, relative=False, tol_mm=None, timeout_s=20.0):
        args = {'position': [float(x) for x in pos], 'timeout_s': timeout_s}
        if quat is not None:
            args['quaternion'] = [float(x) for x in quat]
        if relative:
            args['relative'] = True
        if tol_mm is not None:
            args['tolerance_mm'] = tol_mm
        return self.call('franka_move_pose', **args)

    def try_move_pose(self, *a, **k):
        """move_pose that returns (ok, reply_or_error) instead of raising on a timeout."""
        try:
            return True, self.move_pose(*a, **k)
        except MotionError as e:
            return False, str(e)

    def move_joints(self, q_target):
        q_target = np.asarray(q_target, float)
        if np.any(q_target < SAFE_MIN + JOINT_TARGET_MARGIN) or np.any(q_target > SAFE_MAX - JOINT_TARGET_MARGIN):
            raise MotionError(f'joint target too close to limits: {np.round(q_target, 3).tolist()}')
        q0 = self.joints()
        n = max(1, int(np.ceil(np.max(np.abs(q_target - q0)) / MAX_JOINT_STEP)))
        for k in range(1, n + 1):
            q = q0 + k / n * (q_target - q0)
            self.call('franka_move_joints', positions=[float(x) for x in q])
        return self.joints()

    def home(self):
        from kin import HOME
        if np.max(np.abs(self.joints() - HOME)) > MAX_JOINT_STEP:
            self.move_joints(HOME)          # split into allowed chunks first
        return self.call('franka_move_joints', preset='home')

    # --- gripper -----------------------------------------------------------------------
    def wait_gripper(self, timeout_s=6.0):
        t0 = time.time()
        while True:
            g = self.status()['gripper']
            if not g['is_moving'] or time.time() - t0 > timeout_s:
                return g
            time.sleep(0.3)

    def grasp(self, width, force, speed=0.04):
        self.call('franka_gripper', action='grasp', width=width, force=force, speed=speed)
        time.sleep(0.5)
        return self.wait_gripper()

    def open(self, speed=0.05):
        self.call('franka_gripper', action='open', speed=speed)
        time.sleep(0.3)
        return self.wait_gripper()

    def close_fingers(self, speed=0.05):
        self.call('franka_gripper', action='move', width=0.0, speed=speed)
        time.sleep(0.3)
        return self.wait_gripper()

    # --- perception --------------------------------------------------------------------
    def observe(self, cameras=('left', 'right', 'wrist'), project=None):
        args = {'cameras': list(cameras)}
        if project is not None:
            args['project'] = [[float(v) for v in p] for p in project]
        out = self.ctx.value('franka_observe', **args)   # images stay out of the log
        self.log('observe', frame_id=out['frame_id'], tcp=out.get('tcp_position'))
        return out['frame_id']
