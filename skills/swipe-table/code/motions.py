"""Motion primitives for towel grasping and wiping (gripper pointing down)."""
import json
import time

Q_DOWN = [1.0, 0.0, 0.0, 0.0]
from swipe_table_params import Z_FLOOR, PRESS_DEPTH, PRESS_TOL_MM, STROKE_TIMEOUT_S, MAX_REFUSED  # hyper-parameters: swipe_table_params.py


def _error_state(exc):
    """robot_state dict from a RuntimeError raised by ctx.call, or None."""
    text = str(exc)
    start = text.find('{')
    if start < 0:
        return None, None
    try:
        err = json.loads(text[start:])['error']
    except (ValueError, KeyError, TypeError):
        return None, None
    return err, err.get('robot_state')


def move(ctx, pos, tol=None, timeout=None, log=None):
    """Plain move_pose (gripper down). Returns the reply dict; raises on failure."""
    args = {'position': [float(v) for v in pos], 'quaternion': Q_DOWN}
    if tol is not None:
        args['tolerance_mm'] = tol
    if timeout is not None:
        args['timeout_s'] = timeout
    reply = ctx.value('franka_move_pose', **args)
    if log:
        log(f"move {args['position']} -> {reply.get('measured_position')} err {reply.get('position_error_mm')} mm")
    return reply


def press_move(ctx, x, y, z_cmd, log=None):
    """Move to (x, y, z_cmd) where z_cmd is below the contact height, accepting a stall on
    the cloth. Returns (measured_position, reached). A timeout is normal contact; any other
    failure (or control_uncertain) raises."""
    z_cmd = max(Z_FLOOR, z_cmd)
    try:
        r = ctx.value('franka_move_pose', position=[float(x), float(y), float(z_cmd)], quaternion=Q_DOWN,
                      tolerance_mm=PRESS_TOL_MM, timeout_s=STROKE_TIMEOUT_S)
        meas = r['measured_position']
        if log:
            log(f"press ({x:.3f},{y:.3f},{z_cmd:.3f}) -> {[round(v, 4) for v in meas]} err {r['position_error_mm']} mm")
        return meas, True
    except RuntimeError as exc:
        err, state = _error_state(exc)
        if err is None or err.get('control_uncertain'):
            raise
        if err.get('type') == 'TimeoutError' and state:
            meas = state['tcp_position']
            if log:
                log(f"press ({x:.3f},{y:.3f},{z_cmd:.3f}) timeout -> {[round(v, 4) for v in meas]}")
            return meas, False
        if err.get('type') == 'TargetRejected' or err.get('no_action_executed'):
            if log:
                log(f"press ({x:.3f},{y:.3f},{z_cmd:.3f}) REFUSED ({err.get('type')}): {str(err.get('reason'))[:160]}")
            return None, 'rejected'
        raise


def scrub(ctx, waypoints, z_contact, log=None):
    """Drag the held towel pad through waypoints [(x, y), ...] with a light press.
    z_contact: measured TCP height where the pad touches the table. Adapts z_contact from
    each stroke's measured height. Returns a list of measured positions."""
    out = []
    refused = 0
    for x, y in waypoints:
        z_cmd = z_contact - PRESS_DEPTH
        meas, reached = press_move(ctx, x, y, z_cmd, log=log)
        if reached == 'rejected':          # a wall: skip this waypoint, continue from where the arm holds
            refused += 1
            if refused > MAX_REFUSED:
                raise RuntimeError(f'scrub: {refused} waypoints refused by the controller')
            continue
        out.append(meas)
        lateral = ((meas[0] - x) ** 2 + (meas[1] - y) ** 2) ** 0.5
        if meas[2] > max(Z_FLOOR, z_cmd) + 0.004:
            z_contact = meas[2]                 # pad stalled here: this is the contact height
        if lateral > 0.03:
            raise RuntimeError(f'scrub stroke stalled laterally: target ({x:.3f},{y:.3f}) measured {meas}')
    return out, z_contact


def boustrophedon(x0, x1, y0, y1, spacing, along='y'):
    """Zig-zag waypoints covering the rectangle with parallel strokes."""
    pts = []
    if along == 'y':
        n = max(1, int(round((x1 - x0) / spacing)))
        xs = [x0 + i * (x1 - x0) / n for i in range(n + 1)]
        for i, x in enumerate(xs):
            pts += [(x, y0), (x, y1)] if i % 2 == 0 else [(x, y1), (x, y0)]
    else:
        n = max(1, int(round((y1 - y0) / spacing)))
        ys = [y0 + i * (y1 - y0) / n for i in range(n + 1)]
        for i, y in enumerate(ys):
            pts += [(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]
    return pts

