"""Motion helpers for the bandu task on the FR3 (all through the MCP tools via ctx).

Measured on this robot (2026-10-10):
- franka_move_pose overshoots DOWN at the end of a descent: 8 cm -> ~18 mm, 2 cm -> ~4.5 mm,
  1 cm -> ~3 mm. A guarded move at 10 mm/s overshoots only ~1.5 mm and stops on contact.
  So: move_pose never ends closer than APPROACH_DZ above a target near a surface; the last
  APPROACH_DZ is a guarded move.
- Closed fingertip ends are ~12.4 mm below the TCP (touch the table at TCP z 0.1054).
- Workspace floor: TCP z >= 0.10.
"""
import math
import time

TABLE_Z = 0.093          # table top (depth and contact agree within ~2 mm)
FINGERTIP_BELOW_TCP = 0.0124
TCP_Z_MIN = 0.1005       # workspace box floor is 0.10
APPROACH_DZ = 0.04       # move_pose stops this far above a low target; guarded move does the rest
TRANSIT_Z = 0.27         # safe travel height (tallest stack top ~0.24)
GUARD_SPEED = 0.01       # m/s for the final descent
GUARD_FORCE = 4.0        # N above baseline = contact
MAX_STEP = 0.35          # m per move_pose call (tool limit 0.4)
MAX_DESCENT = 0.12       # m per move_pose descent before a guarded approach (overshoot <= ~2.5 cm)


def table_z_at(xy):
    """Table height from fingertip touches (2026-10-10, 9 points, residuals <= 1.6 mm): flat
    0.093 up to x = 0.50, then rising ~4.6 mm per 10 cm towards the far edge."""
    return TABLE_Z + 0.0462 * max(0.0, xy[0] - 0.50)


def yaw_quat(yaw):
    """Gripper pointing down, rotated by yaw (rad) about base z. yaw=0: fingers close along base y;
    yaw=pi/2: fingers close along base x."""
    # q = Rz(yaw) * Rx(pi)  ->  [cos(yaw/2), sin(yaw/2), 0, 0]
    return [math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0]


def wrap_grasp_yaw(yaw):
    """Fingers are symmetric: map any yaw to (-pi/2, pi/2]."""
    while yaw > math.pi / 2 + 1e-3:           # tolerance: a stored 1.5708 must stay +90, not flip to -90
        yaw -= math.pi
    while yaw <= -math.pi / 2 + 1e-3:
        yaw += math.pi
    return yaw


class Motion:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log

    def status(self):
        return self.ctx.value('franka_status')

    def tcp(self):
        s = self.status()
        return s['tcp_position'], s['tcp_quaternion']

    def move(self, pos, yaw=None, quat=None, **kw):
        if quat is None and yaw is not None:
            quat = yaw_quat(yaw)
        args = dict(position=[float(v) for v in pos])
        if quat is not None:
            args['quaternion'] = [float(v) for v in quat]
        args.update(kw)
        return self.ctx.value('franka_move_pose', **args)

    def transit(self, x, y, yaw, z=TRANSIT_Z):
        """Rise to z at the current xy, then travel level to (x, y, z) with the new yaw."""
        p, _ = self.tcp()
        if p[2] < z - 0.005:
            self.move([p[0], p[1], z])
            p, _ = self.tcp()
        dist = math.hypot(x - p[0], y - p[1])
        steps = max(1, math.ceil(dist / MAX_STEP))
        for i in range(1, steps):     # move_pose refuses more than 0.4 m per call
            f = i / steps
            self.move([p[0] + f * (x - p[0]), p[1] + f * (y - p[1]), z], yaw=yaw)
        return self.move([x, y, z], yaw=yaw)

    def descend(self, z_target, speed=GUARD_SPEED, force=GUARD_FORCE, retract=0.0005):
        """From the current pose: move_pose down to z_target + APPROACH_DZ (if higher), then a
        guarded move to z_target. Returns the guarded-move reply (contact=True means something
        was touched before z_target)."""
        p, q = self.tcp()
        z_target = max(z_target, TCP_Z_MIN)
        z_pre = z_target + APPROACH_DZ
        while p[2] > z_pre + 0.003:            # long move_pose descents overshoot ~20 % of the distance
            self.move([p[0], p[1], max(z_pre, p[2] - MAX_DESCENT)], quat=q)
            p, q = self.tcp()
        dist = p[2] - z_target
        if dist <= 0.0005:
            return {'contact': False, 'final_position': p, 'travel_m': 0.0}
        return self.ctx.value('franka_guarded_move', direction=[0, 0, -1],
                              max_distance_m=round(min(dist, 0.05), 4),
                              speed_mps=speed, force_threshold_n=force, retract_m=retract)

    def lift(self, dz=None, z=None):
        p, q = self.tcp()
        target = z if z is not None else p[2] + dz
        return self.move([p[0], p[1], target], quat=q)

    def gripper_open(self, width=0.08, speed=0.05):
        if width >= 0.079:
            return self.ctx.value('franka_gripper', action='open')
        return self.ctx.value('franka_gripper', action='move', width=width, speed=speed)

    def grasp(self, width=0.0, force=20.0, speed=0.03):
        return self.ctx.value('franka_gripper', action='grasp', width=width, force=force, speed=speed)

    def place_down(self, max_drop, speed=GUARD_SPEED, force=3.0):
        """Guarded move straight down until the held object touches (max_drop m at most, never
        below the workspace floor)."""
        p, _ = self.tcp()
        dist = min(max_drop, 0.05, p[2] - TCP_Z_MIN)
        if dist <= 0.0005:
            return {'contact': False, 'final_position': p, 'travel_m': 0.0, 'reason': 'at floor'}
        return self.ctx.value('franka_guarded_move', direction=[0, 0, -1],
                              max_distance_m=round(dist, 4),
                              speed_mps=speed, force_threshold_n=force, retract_m=0.0005)


def descend_to_grasp(m, x, y, yaw, z_grasp, width_open=0.08, retries=3, shift=0.006, log=print):
    """Open, move above (x, y), guarded-descend to z_grasp. If a finger touches something on the
    way down (contact=True), lift, shift `shift` m in the direction the hand was pushed and retry.
    Returns (ok, (x, y), last_reply)."""
    r = None
    for attempt in range(retries + 1):
        m.transit(x, y, yaw, z=max(z_grasp + APPROACH_DZ + 0.02, 0.16))
        m.gripper_open(width_open)
        start, _ = m.tcp()
        r = m.descend(z_grasp)
        if not r.get('contact'):
            return True, (x, y), r
        end = r.get('final_position') or m.tcp()[0]
        dx, dy = end[0] - start[0], end[1] - start[1]
        n = math.hypot(dx, dy)
        log(f'descent contact at z={end[2]:.4f}, lateral push ({dx*1000:.1f},{dy*1000:.1f}) mm')
        m.lift(z=max(z_grasp + APPROACH_DZ + 0.02, 0.16))
        if n < 0.0008:
            return False, (x, y), r          # pushed straight up: landed on top of something
        x, y = x + shift * dx / n, y + shift * dy / n
    return False, (x, y), r


def touch_down_open(m, x, y, yaw, z_table_guess=TABLE_Z, back_off=0.003, width_open=0.08, max_above=0.008):
    """Grasp approach for low objects: open, stop APPROACH_DZ above where the fingertips would meet
    the table, then guarded-descend until the fingertips touch the table and back off `back_off`.
    The table is not flat (up to ~6 mm higher on the far side), so contact is found, not assumed.
    Returns (ok, info). ok=False when contact came more than `max_above` above the expected table
    contact (fingers landed on the object) or no contact at all within range."""
    tcp_at_table = z_table_guess + FINGERTIP_BELOW_TCP
    m.transit(x, y, yaw, z=max(tcp_at_table + APPROACH_DZ + 0.03, 0.16))
    m.gripper_open(width_open)
    m.move([x, y, tcp_at_table + APPROACH_DZ], yaw=yaw)
    p, _ = m.tcp()
    dist = min(p[2] - TCP_Z_MIN, APPROACH_DZ + 0.012)
    r = m.ctx.value('franka_guarded_move', direction=[0, 0, -1], max_distance_m=round(dist, 4),
                    speed_mps=GUARD_SPEED, force_threshold_n=3.0, retract_m=back_off)
    info = {'contact': r.get('contact'), 'contact_z': (r.get('contact_position') or [None] * 3)[2],
            'final': r.get('final_position')}
    if not r.get('contact'):
        info['why'] = 'no table contact within range'
        return False, info
    if info['contact_z'] > tcp_at_table + max_above:
        info['why'] = f"contact {info['contact_z']:.4f} is {1000 * (info['contact_z'] - tcp_at_table):.1f} mm above expected table contact"
        m.lift(dz=0.03)
        return False, info
    info['table_z'] = info['contact_z'] - FINGERTIP_BELOW_TCP
    return True, info


def regrasp_move(m, gx, gy, gyaw, px, py, pyaw, carry_z=0.17, z_table_guess=TABLE_Z, min_width=0.04, force=25):
    """Pick a low object (table-touch approach) at (gx, gy, gyaw), carry it at carry_z (well above
    neighbours: level moves sag up to ~2.5 cm) to (px, py, pyaw), lower and guarded-place it.
    Returns a dict with ok and measurements."""
    ok, info = touch_down_open(m, gx, gy, gyaw, z_table_guess=z_table_guess)
    if not ok:
        return {'ok': False, 'stage': 'approach', **info}
    hang = info['final'][2] - info['table_z']           # object bottom below TCP
    g = m.grasp(width=0.0, force=force)
    if g['width'] < min_width:
        m.gripper_open(0.08); m.lift(z=carry_z)
        return {'ok': False, 'stage': 'grasp', 'width': g['width']}
    m.lift(z=carry_z)
    m.transit(px, py, pyaw, z=carry_z)
    m.move([px, py, z_table_guess + hang + 0.03], yaw=pyaw)
    r = m.place_down(0.05)
    if not r.get('contact'):
        return {'ok': False, 'stage': 'place', 'reason': 'no contact (still holding)'}
    m.ctx.value('franka_gripper', action='move', width=min(0.08, g['width'] + 0.015), speed=0.02)
    m.lift(z=carry_z)
    return {'ok': True, 'grasp_width': g['width'], 'table_z': info['table_z'], 'hang': hang,
            'placed_tcp': r.get('contact_position')}


def push(m, x, y, direction, distance, z_table=TABLE_Z, tip_clear=0.006, yaw=0.0, force=1.5, speed=0.005):
    """Nudge with the closed fingers: start at (x, y) (pusher clear of the object), lower until the
    fingertips are tip_clear above the table, slide `distance` along `direction` (unit xy) with a
    guarded move (stops early if resistance exceeds `force`), back off and lift."""
    m.ctx.value('franka_gripper', action='move', width=0.0)
    m.transit(x, y, yaw, z=0.16)
    z = z_table + FINGERTIP_BELOW_TCP + tip_clear
    r0 = m.descend(z)
    if r0.get('contact'):
        m.lift(z=0.16)
        return {'ok': False, 'why': 'unexpected contact while lowering the pusher', 'at': r0.get('contact_position')}
    d = [float(direction[0]), float(direction[1]), 0.0]
    r = m.ctx.value('franka_guarded_move', direction=d, max_distance_m=round(distance, 4),
                    speed_mps=speed, force_threshold_n=force, retract_m=0.001)
    m.lift(z=0.16)
    return {'ok': True, 'contact': r.get('contact'), 'travel_m': r.get('travel_m'),
            'final': r.get('final_position'), 'peak': r.get('wrench_peak_change')}
