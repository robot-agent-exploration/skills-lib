"""Pick and place the bandu pieces (measured 2026-10-10 on the real set).

Heights are relative to the surface the piece stands on; the table is not flat (z 0.093 near
the robot, up to ~0.101 at x ~0.66), so every pick first finds the table by touching it with
the open fingertips beside the piece.
"""
import math

import sys as _sys                                                   # skill folder: siblings import each other by name
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

from motion import (FINGERTIP_BELOW_TCP, GUARD_SPEED, TCP_Z_MIN, table_z_at, wrap_grasp_yaw)

BASE_TOP_ABOVE_TABLE = 0.0165      # block thickness

# grasp: TCP height above the piece's bottom; width: measured grasp width; open: approach width
PIECES = {
    'ring':   dict(grasp=0.0180, width=0.0376, open=0.075, height=0.016),
    'barrel': dict(grasp=0.0190, width=0.0271, open=0.075, height=0.040),
    'goblet': dict(grasp=0.0405, width=0.0370, open=0.080, height=0.050),
    'urn':    dict(grasp=0.0280, width=0.0314, open=0.075, height=0.041, grip=40.0,
                   seat_tol=0.0045),   # seated reads +2.6..+3.4 mm; a rim catch read +4.0 (24 deg): the upright check decides
    # ball=radius: located from above with the wrist camera and grasped once there (an off-centre
    # grasp pushes the ball to the centre and the knob hangs tilted ~1.3 deg per mm; >3 deg jams in the urn)
    'knob':   dict(grasp=0.0446, width=0.0243, open=0.075, height=0.0566, grip=40.0, tap=True, ball=0.01215,
                   visual=True),       # inserted by align.insert_visual (tilt + offset corrected by camera)
    'dowel':  dict(grasp=0.0977, width=0.0156, open=0.075, height=0.1127, grip=40.0),
    # oblique cylinder (added by the user 13:42): stands on a flat end, body leans ~14 deg. Grasped
    # with TCP x along the lean (fingers close across it) it keeps the lean; its bottom centre is
    # then `lean` m behind the TCP along -TCP x. Single grasp only (a cross grasp would tip it).
    'slant':  dict(grasp=0.0445, width=0.0249, open=0.060, height=0.069, lean=0.011),
}
# where a stacked piece's bottom ends up above the base top when seated
SEAT_ABOVE_BASE = {'urn': 0.0475, 'knob': 0.0749, 'dowel': 0.0}
from bandu_params import (  # hyper-parameters: bandu_params.py
    WIDTH_TOL, GRIP_N, PALM_ABOVE_FINGERTIPS, PIECES_TRANSIT_Z as TRANSIT_Z, CARRY_BOTTOM_Z, PROBE_SPEED,
    PROBE_FORCE, SEATED_TOL_DEFAULT, MAX_PROBES,
)


class PieceLost(RuntimeError):
    pass


class NotSeated(RuntimeError):
    """Released inside the opening but not down on its seat (no longer held)."""


def check_held(m, name):
    """Raise PieceLost if the gripper no longer reports a grasp (call after lifting a piece)."""
    g = m.status()['gripper']
    if not g.get('is_grasped'):
        raise PieceLost(f"{name}: lost after lifting (gripper width {1000 * g['width']:.1f} mm, not grasped)")


def carry_height(name):
    return max(0.32, CARRY_BOTTOM_Z + PIECES[name]['grasp'])


def find_table(m, x, y, yaw, open_width, z_guess=0.093, max_above=0.010, piece_height=0.10):
    """Open, lower the fingertips around the piece until they touch the table; back off 3 mm.
    Returns table z or raises RuntimeError (contact far above the expected table = landed on
    the piece or a neighbour)."""
    tcp_at_table = z_guess + FINGERTIP_BELOW_TCP
    m.transit(x, y, yaw, z=TRANSIT_Z)
    m.gripper_open(open_width)
    pre = tcp_at_table + max(0.12, piece_height + 0.03)
    m.move([x, y, pre], yaw=yaw)
    # long guarded descent in two legs (<= 5 cm each); stop on first contact
    for _ in range(4):
        p, _ = m.tcp()
        dist = min(0.05, p[2] - TCP_Z_MIN, p[2] - (tcp_at_table - 0.012))
        if dist <= 0.001:
            break
        r = m.ctx.value('franka_guarded_move', direction=[0, 0, -1], max_distance_m=round(dist, 4),
                        speed_mps=0.015 if p[2] > tcp_at_table + 0.06 else GUARD_SPEED,
                        force_threshold_n=3.0, retract_m=0.003)
        if r.get('contact'):
            cz = r['contact_position'][2]
            if cz > tcp_at_table + max_above:
                m.lift(dz=0.05)
                raise RuntimeError(f'touched something at TCP z {cz:.4f}, {1000 * (cz - tcp_at_table):.0f} mm above the expected table')
            return cz - FINGERTIP_BELOW_TCP
    raise RuntimeError('no table contact')


def piece_yaw(name, yaw):
    """Symmetric pieces: any yaw is equivalent to one in (-90, 90] deg. Leaning pieces keep the
    yaw (TCP x points along the lean); J7 allows about -105..195 deg."""
    if 'lean' not in PIECES[name]:
        return wrap_grasp_yaw(yaw)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return yaw + 2 * math.pi if yaw < math.radians(-105) else yaw


def pick(m, name, x, y, yaw=math.pi / 2, center=True, table_z=None, log=print):
    """Pick a standing piece from the table at TCP xy (x, y). With center=True it is grasped twice
    (yaw+90deg, then yaw) so it ends up centred under the TCP in both axes (not for leaning
    pieces). Leaves it held at its grasp height."""
    spec = PIECES[name]
    yaw = piece_yaw(name, yaw)
    center = center and 'lean' not in spec
    yaws = [wrap_grasp_yaw(yaw + math.pi / 2), yaw] if center else [yaw]
    if table_z is None and spec['height'] > PALM_ABOVE_FINGERTIPS - 0.004:
        table_z = table_z_at((x, y))      # taller pieces would hit the palm before the fingertips reach the table
    if table_z is None:
        table = find_table(m, x, y, yaws[0], spec['open'], z_guess=table_z_at((x, y)), piece_height=spec['height'])
        z = table + spec['grasp']
    else:                                   # known table: go straight to grasp height (guarded)
        table = table_z
        z = table + spec['grasp']
        m.transit(x, y, yaws[0], z=TRANSIT_Z)
        m.gripper_open(spec['open'])
        r = m.descend(z)
        if r.get('contact'):
            m.lift(dz=0.05)
            raise RuntimeError(f"{name}: unexpected contact at TCP z {r['contact_position'][2]:.4f} on the way to grasp height")
    for i, w in enumerate(yaws):
        if i > 0:
            m.gripper_open(spec['open'])
            m.move([x, y, z + 0.006], yaw=w)       # rotate clear of the table
            m.ctx.value('franka_guarded_move', direction=[0, 0, -1], max_distance_m=0.006,
                        speed_mps=GUARD_SPEED, force_threshold_n=3.0, retract_m=0.0)
        else:
            m.move([x, y, z])
        g = m.grasp(width=0.0, force=spec.get('grip', GRIP_N))
        log(f'{name}: grasp yaw {math.degrees(w):.0f} width {1000 * g["width"]:.1f} mm grasped {g.get("is_grasped")}')
        if abs(g['width'] - spec['width']) > WIDTH_TOL or not g.get('is_grasped', True):
            m.gripper_open(spec['open'])
            m.lift(dz=0.06)
            raise RuntimeError(f'{name}: grasp width {g["width"]:.4f}, expected {spec["width"]:.4f}')
    return {'table_z': table, 'grasp_z': z}


def place(m, name, x, y, bottom_z, yaw=math.pi / 2, carry_z=None, tol=0.006, log=print, straighten=None):
    """Carry the held piece at carry_z so its bottom centre comes over (x, y), lower until its
    bottom touches near bottom_z (guarded), release and lift. Raises if the contact is far from
    bottom_z. Returns the TCP xy used (differs from (x, y) for leaning pieces)."""
    spec = PIECES[name]
    yaw = piece_yaw(name, yaw)
    if carry_z is None:
        carry_z = carry_height(name)
    lean = spec.get('lean', 0.0)
    x, y = x + lean * math.cos(yaw), y + lean * math.sin(yaw)
    m.lift(z=carry_z)
    check_held(m, name)
    m.transit(x, y, yaw, z=carry_z)
    if straighten is not None:             # make a piece that hangs tilted vertical before it touches down
        straighten(m)
    tcp_seat = bottom_z + spec['grasp']
    r = m.descend(tcp_seat + 0.012)          # guarded: a contact here means something is in the way
    if r.get('contact'):
        m.lift(z=carry_z)
        raise RuntimeError(f"{name}: touched something at TCP z {r['contact_position'][2]:.4f} before the seat (still holding)")
    p, _ = m.tcp()
    r = m.place_down(min(0.05, p[2] - tcp_seat + 0.012))
    if not r.get('contact'):
        raise RuntimeError(f'{name}: no contact while placing (still holding)')
    cz = r['contact_position'][2]
    err = cz - tcp_seat
    log(f'{name}: placed, contact {1000 * err:+.1f} mm vs expected seat')
    if abs(err) > tol:
        m.lift(dz=0.03)
        raise RuntimeError(f'{name}: seated {1000 * err:+.1f} mm from expected (still holding)')
    m.ctx.value('franka_gripper', action='move', width=min(0.08, spec['width'] + 0.02), speed=0.02)
    m.lift(z=max(carry_z, cz + 0.05))
    return {'contact_z': cz, 'seat_err': err, 'tcp_xy': (x, y), 'yaw': yaw}


# Hand body in the TCP frame (pointing down): housing x in [-0.035, 0.035], y in [-0.105, 0.105]
# with its underside ~0.035 above the TCP; the wrist camera bracket adds x in [-0.095, -0.035]
# from ~0.05 above the TCP. (TCP x = (cos yaw, sin yaw), TCP y = (sin yaw, -cos yaw) in base.)
HAND_BOXES = [((-0.035, 0.035), (-0.105, 0.105), 0.035), ((-0.095, -0.035), (-0.06, 0.0), 0.05)]


FINGER_HALF_X, FINGER_THICK = 0.011, 0.012     # finger pad half-width (TCP x) and finger thickness (TCP y)


def hand_conflicts(tcp_xyz, yaw, objects, margin=0.012, ignore=(), opening=0.08, ignore_fingers=()):
    """Objects (dicts with name, xy, radius, top_z) that the hand would hit at this pose: the
    housing and camera bracket (HAND_BOXES) and the two fingers, open to `opening`, reaching
    down to the fingertips (an open finger landed on the dowel's top once)."""
    c, s = math.cos(yaw), math.sin(yaw)
    half = opening / 2
    boxes = HAND_BOXES + [((-FINGER_HALF_X, FINGER_HALF_X), (half, half + FINGER_THICK), -FINGERTIP_BELOW_TCP),
                          ((-FINGER_HALF_X, FINGER_HALF_X), (-half - FINGER_THICK, -half), -FINGERTIP_BELOW_TCP)]
    out = []
    for o in objects:
        if o['name'] in ignore:
            continue
        dx, dy = o['xy'][0] - tcp_xyz[0], o['xy'][1] - tcp_xyz[1]
        u, v = dx * c + dy * s, dx * s - dy * c          # in TCP x / TCP y
        r = o.get('radius', 0.0) + margin
        for k, ((x0, x1), (y0, y1), dz) in enumerate(boxes):
            if k >= len(HAND_BOXES) and o['name'] in ignore_fingers:
                continue                  # the support right below: the open fingers pass outside it
            inside = (x0 - r <= u <= x1 + r) and (y0 - r <= v <= y1 + r)
            if inside and o['top_z'] > tcp_xyz[2] + dz - margin:
                out.append(o['name'])
                break
    return out


CIRCLE = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]


class _Inside(Exception):
    """A probe ended inside the opening (raised to leave the search)."""


def insert(m, name, x, y, seat_bottom_z, hover=0.012, yaw=math.pi / 2, carry_z=None, fine=False, rim=None,
           log=print):
    """Peg-in-hole for the held piece by touch. Hover its bottom `hover` above the expected rim and
    probe straight down (guarded, 1.5 N). Off the hole the piece stops on the rim; a partial overlap
    stops lower on the rounded edge, so the search walks downhill. fine=True (target measured by
    the wrist camera, ~1-2 mm): neighbours at 1/0.5 mm first; otherwise 8 probes on a 2.5 mm
    circle, neighbours at 1.5/0.75/0.4 mm, then a 5 mm circle. A probe that ends well inside the
    opening (>= 8 mm below the rim; rim = rim height above the seat if known) is pressed on at
    4 N, or for 'tap' pieces (the knob: a ball grip lets it jam tilted) released there and tapped
    down on its top with the closed fingers. Lateral moves happen above the highest contact seen.
    Releases and lifts when the bottom reaches the seat (within 6 mm); raises after MAX_PROBES
    (still holding, lifted) or NotSeated if a released piece stays high."""
    spec = PIECES[name]
    SEATED_TOL = spec.get('seat_tol', SEATED_TOL_DEFAULT)
    yaw = wrap_grasp_yaw(yaw)
    if carry_z is None:
        carry_z = carry_height(name)
    seat_tcp = seat_bottom_z + spec['grasp']
    m.lift(z=carry_z)
    check_held(m, name)
    m.transit(x, y, yaw, z=carry_z)
    hover_z = seat_tcp + 0.015 + hover
    r = m.descend(hover_z)
    if r.get('contact'):
        m.lift(z=carry_z)
        raise RuntimeError(f"{name}: touched something at TCP z {r['contact_position'][2]:.4f} above the hover height (still holding)")
    probes = []                               # (x, y, contact height above seat)
    safe = [hover_z]

    def press(px, py):
        """The bottom is inside the opening but stuck: push on at 4 N towards the seat."""
        p, _ = m.tcp()
        r = m.ctx.value('franka_guarded_move', direction=[0, 0, -1],
                        max_distance_m=round(max(0.0006, min(0.03, p[2] - seat_tcp + 0.004)), 4),
                        speed_mps=PROBE_SPEED, force_threshold_n=4.0, retract_m=0.0)
        h = (r.get('contact_position') or m.tcp()[0])[2] - seat_tcp
        log(f'{name}: press -> {1000 * h:+.1f} mm vs seat')
        probes.append((px, py, h))

    def release_and_tap(px, py):
        m.ctx.value('franka_gripper', action='move', width=min(0.08, spec['width'] + 0.008), speed=0.01)
        p, _ = m.tcp()
        m.move([p[0], p[1], p[2] + 0.015], yaw=yaw, tolerance_mm=0.5)
        m.ctx.value('franka_gripper', action='move', width=0.0, speed=0.05)
        top_tcp = seat_bottom_z + spec['height'] + FINGERTIP_BELOW_TCP   # closed fingertips on a seated top
        p, _ = m.tcp()
        r = m.ctx.value('franka_guarded_move', direction=[0, 0, -1],
                        max_distance_m=round(max(0.001, min(0.04, p[2] - top_tcp + 0.006)), 4),
                        speed_mps=PROBE_SPEED, force_threshold_n=2.0, retract_m=0.002)
        h = (r.get('contact_position') or m.tcp()[0])[2] - top_tcp
        log(f'{name}: released in the opening, tapped its top: {1000 * h:+.1f} mm vs seated')
        m.lift(z=carry_z)
        if abs(h) > 0.004:
            raise NotSeated(f'{name}: released in the opening but its top is {1000 * h:+.1f} mm from seated')
        return {'xy': (px, py), 'seat_err': h, 'probes': len(probes), 'tapped': True}

    def probe(px, py):
        p, _ = m.tcp()
        z0 = max(safe)
        if p[2] < z0 - 0.0005:
            m.move([p[0], p[1], z0], yaw=yaw, tolerance_mm=0.5)
        m.move([px, py, z0], yaw=yaw, tolerance_mm=0.5)
        r = m.ctx.value('franka_guarded_move', direction=[0, 0, -1],
                        max_distance_m=round(min(0.05, z0 - seat_tcp + 0.004), 4),
                        speed_mps=PROBE_SPEED, force_threshold_n=PROBE_FORCE, retract_m=0.0)
        cz = (r.get('contact_position') or m.tcp()[0])[2]
        h = cz - seat_tcp
        probes.append((px, py, h))
        safe.append(cz + 0.003)
        log(f'{name}: probe ({1000 * (px - x):+.1f},{1000 * (py - y):+.1f}) mm -> {1000 * h:+.1f} mm vs seat')
        rim_h = rim if rim is not None else max(t[2] for t in probes)
        if SEATED_TOL <= h < rim_h - 0.008:   # well inside the opening: do not lift away
            if spec.get('tap'):
                raise _Inside(px, py)
            press(px, py)
            safe[-1] = max(safe[-1], seat_tcp + rim_h + 0.003)
        return probes[-1][2]

    def done():
        return bool(probes) and probes[-1][2] < SEATED_TOL

    def lowest():
        return min(probes, key=lambda t: t[2])

    def descend_steps(steps):
        for step in steps:
            while not done() and len(probes) < MAX_PROBES:
                bx, by, bh = lowest()
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    if probe(bx + dx, by + dy) < SEATED_TOL or len(probes) >= MAX_PROBES:
                        break
                if done() or lowest()[2] > bh - 0.0008:
                    break                     # no neighbour clearly lower: refine the step

    def search():
        probe(x, y)
        if fine and not done():
            descend_steps((0.001, 0.0005))
        for radius in (0.0025, 0.005):
            if not done():
                for cx, cy in CIRCLE:
                    if probe(x + radius * cx, y + radius * cy) < SEATED_TOL or len(probes) >= MAX_PROBES:
                        break
            descend_steps((0.0015, 0.00075, 0.0004))
            rim_h = max(t[2] for t in probes)
            if done() or len(probes) >= MAX_PROBES or lowest()[2] < rim_h - 0.003:
                break                         # found the hole's edge; a wider circle will not help

    try:
        search()
    except _Inside as e:
        return release_and_tap(*e.args)
    if done():
        px, py, h = probes[-1]
        m.ctx.value('franka_gripper', action='move', width=min(0.08, spec['width'] + 0.02), speed=0.02)
        m.lift(z=carry_z)
        return {'xy': (px, py), 'seat_err': h, 'probes': len(probes)}
    m.lift(z=carry_z)
    raise RuntimeError(f'{name}: not inserted after {len(probes)} probes (lowest {1000 * lowest()[2]:+.1f} mm; still holding)')
