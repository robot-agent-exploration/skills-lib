"""Tilted vertex-first insertion into a tight (L3) hole, then wiggle-push with force feedback.

Usage (peg already held, gripper vertical at yaw psi, lifted above the board; from the workspace root,
for api.runtime, with <skill>/code on PYTHONPATH):
    python -m peg_insertion.l3_tilt_insert '<json params>'
(In the session: code/peg_insertion/l3_tilt_insert.py.)
Params: psi_deg, c (in-hand offset a,b m), rel_deg, Lb, z_board, hole (x,y), r_hv, margin,
        tilt_deg, depth_in, z_ref (TCP z of vertical surface contact at the hole), Fb, Mb.
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from .fixture import gripper_matrix
from .motion import move, status
from .tilt import (place_vertex, quat_xyzw, tcp_for_vertex, tool_rot, vertex_gripper_point,
                  vertex_world)


def run(ctx, p, log=print):
    psi = np.radians(p['psi_deg']); c = np.array(p['c']); rel = np.radians(p['rel_deg'])
    Lb, z_board = p['Lb'], p['z_board']; h = np.array(p['hole']); r_hv, m = p['r_hv'], p['margin']
    R0 = tool_rot(psi); g = vertex_gripper_point(p.get('shape', 'triangle'), p.get('vertex', 0), c, rel, Lb)
    tcp_c = np.r_[h - gripper_matrix(psi) @ c, z_board + Lb]
    w = vertex_world(tcp_c, R0, g); u = (w[:2] - h) / np.linalg.norm(w[:2] - h)
    U = np.r_[u, 0]; P = np.array([-u[1], u[0]])
    tilt = np.radians(p['tilt_deg']); Rt = tool_rot(psi, u, tilt)

    def Q(e, s):
        return np.r_[h + (r_hv - m + s) * u, z_board + e]

    def rep(tag, v):
        log(f'  {tag}: vertex along {((v[:2]-h)@u)*1000:6.2f} perp {((v[:2]-h)@P)*1000:+5.2f} '
            f'h {(v[2]-z_board)*1000:+5.2f} mm')

    move(ctx, tcp_for_vertex(Q(0.012, 0), Rt, g), quat_xyzw(Rt), tol_mm=1.0, timeout_s=8)
    v, _, _ = place_vertex(ctx, move, Q(0.002, 0), Rt, g); rep('hover', v)
    d_in = p['depth_in']
    for e in (0.0008, 0.0003, -0.0002, -0.5 * d_in, -0.8 * d_in, -d_in):
        v, err, _ = place_vertex(ctx, move, Q(e, 0), Rt, g); rep(f'descend {e*1000:+.1f}', v)
        if e < 0 and v[2] - z_board > -0.0001 and err[2] < -0.0003:
            return {'ok': False, 'stage': 'descend', 'reason': 'vertex rests on surface (not over hole)'}
    blocked = False
    for k in range(1, 11):
        s = 0.0003 * k
        v, err, _ = place_vertex(ctx, move, Q(-d_in, s), Rt, g, tries=2); rep(f'slide {s*1000:.1f}', v)
        if (err[:2] @ u) > 0.0003:
            blocked = True
            break
        if v[2] - z_board > -0.0003:
            return {'ok': False, 'stage': 'slide', 'reason': 'vertex climbed out of the hole'}
    if not blocked:
        return {'ok': False, 'stage': 'slide', 'reason': 'corner not found'}
    # seat the vertex in the corner: push into the corner while stepping sideways along the wall
    along = lambda vv: (vv[:2] - h) @ u
    perp = lambda vv: (vv[:2] - h) @ P
    for k in range(p.get('seat_iters', 6)):
        best_v, best_a = v, along(v)
        for dp in (0.00015, -0.00015):
            tgt = np.r_[h + (along(v) + 0.0003) * u + (perp(v) + dp) * P, z_board - d_in]
            vv, _, _ = place_vertex(ctx, move, tgt, Rt, g, tries=2)
            if along(vv) > best_a + 0.00005:
                best_v, best_a = vv, along(vv)
        rep(f'seat {k}', best_v)
        if best_a <= along(v) + 0.00005:
            break
        v = best_v
    Qc = v.copy()
    n_up = p.get('upright_steps', 16)
    steps = np.linspace(np.degrees(tilt), 0, n_up + 1)[1:]
    for i, thd in enumerate(steps):
        Ri = tool_rot(psi, u, np.radians(thd))
        Qi = Qc + 0.00015 * U + np.array([0, 0, -p.get('upright_press', 0.0006)])
        v, _, _ = place_vertex(ctx, move, Qi, Ri, g, tries=2); rep(f'upright {thd:.2f}deg', v)
        if v[2] - z_board > 0.0003:
            log('  vertex lifted out of the hole during upright rotation')
            tcp = np.array(status(ctx)['tcp_position'])
            move(ctx, tcp + [0, 0, 0.004], quat_xyzw(Ri), tol_mm=1.0, timeout_s=3)
            return {'ok': False, 'stage': 'upright', 'reason': 'vertex lifted out (hole estimate off)'}
    return wiggle_push(ctx, p, log)


def wiggle_push(ctx, p, log=print, cycles=8, amp_deg=0.7, f_max=10.0):
    psi = np.radians(p['psi_deg']); c = np.array(p['c']); Lb = p['Lb']; z_ref = p['z_ref']
    Fb, Mb = np.array(p['Fb']), np.array(p['Mb'])
    R = tool_rot(psi)
    g_piv = np.array([c[0], c[1], Lb - 0.003])
    tcp = np.array(status(ctx)['tcp_position'])
    log(f'  wiggle start depth {(z_ref - tcp[2])*1000:.2f} mm')
    if z_ref - tcp[2] < 0.0005:
        return {'ok': False, 'stage': 'wiggle', 'reason': 'peg not in the hole at wiggle start'}
    yaw_amp = p.get('yaw_wiggle_deg', 0.0)
    pattern = [(0, 0, 0)] + [(a * amp_deg, b * amp_deg, w * yaw_amp) for a, b, w in
                             ((1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, -1, -1), (1, 1, -1), (-1, -1, 1),
                              (1, -1, 1), (-1, 1, -1))]
    zc, best = tcp[2], z_ref - tcp[2]
    for cyc in range(cycles):
        for ax, ay, az in pattern:
            s = status(ctx); tcp = np.array(s['tcp_position']); W = np.array(s['tcp_wrench'])
            dF = W[:3] - Fb
            piv = vertex_world(tcp, R, g_piv)
            Rw = Rotation.from_rotvec([np.radians(ax), np.radians(ay), np.radians(az)]) * R
            dxy = np.clip(-dF[:2] / 3000.0, -0.0001, 0.0001)
            tgt = tcp_for_vertex(piv + np.r_[dxy, 0.0], Rw, g_piv)
            if dF[2] > -f_max:
                zc = min(zc, tcp[2]) - 0.0003
            tgt[2] = min(tgt[2], zc)
            r = move(ctx, tgt, quat_xyzw(Rw), tol_mm=0.5, timeout_s=1.0)
            d = z_ref - r['measured_position'][2]
            best = max(best, d)
        log(f'  cycle {cyc}: depth {d*1000:6.2f} mm (best {best*1000:.2f})  dF ({dF[0]:+.1f},{dF[1]:+.1f},{dF[2]:+.1f})')
        if best > p.get('target_depth', 0.019):
            return {'ok': True, 'depth_m': best}
    # finish vertical
    tcp = np.array(status(ctx)['tcp_position'])
    move(ctx, tcp, quat_xyzw(R), tol_mm=0.5, timeout_s=1.0)
    return {'ok': best > 0.012, 'depth_m': best, 'reason': None if best > 0.012 else 'jammed'}


def wiggle_pull(ctx, p, log=print, cycles=10, amp_deg=0.7, f_max=12.0, rise=0.022):
    """Extract a held peg from a tight hole: rock about a pivot near the tip while a steady spring
    (integral off, ~3 N/mm) pulls up, the ladder climbing only while the pull is below f_max."""
    psi = np.radians(p['psi_deg']); c = np.array(p['c']); Lb = p['Lb']
    Fb = np.array(p['Fb'])
    R = tool_rot(psi)
    g_piv = np.array([c[0], c[1], Lb - 0.003])
    z0 = np.array(status(ctx)['tcp_position'])[2]
    yaw_amp = p.get('yaw_wiggle_deg', 0.0)
    pattern = [(0, 0, 0)] + [(a * amp_deg, b * amp_deg, w * yaw_amp) for a, b, w in
                             ((1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, -1, -1), (1, 1, -1), (-1, -1, 1),
                              (1, -1, 1), (-1, 1, -1))]
    zc = z0
    for cyc in range(cycles):
        for ax, ay, az in pattern:
            s = status(ctx); tcp = np.array(s['tcp_position']); W = np.array(s['tcp_wrench'])
            dF = W[:3] - Fb
            piv = vertex_world(tcp, R, g_piv)
            Rw = Rotation.from_rotvec([np.radians(ax), np.radians(ay), np.radians(az)]) * R
            dxy = np.clip(-dF[:2] / 3000.0, -0.0001, 0.0001)
            tgt = tcp_for_vertex(piv + np.r_[dxy, 0.0], Rw, g_piv)
            if dF[2] < f_max:
                zc = max(zc, tcp[2]) + 0.0003
            tgt[2] = max(tgt[2], zc)
            r = move(ctx, tgt, quat_xyzw(Rw), tol_mm=0.5, timeout_s=1.0, integral=False)
            up = r['measured_position'][2] - z0
            if up > rise:
                log(f'  out: rose {up*1000:.1f} mm')
                move(ctx, r['measured_position'], quat_xyzw(R), tol_mm=0.5, timeout_s=2.0)
                return {'ok': True, 'rise_m': up}
        log(f'  cycle {cyc}: rose {up*1000:6.2f} mm  dF ({dF[0]:+.1f},{dF[1]:+.1f},{dF[2]:+.1f})')
    tcp = np.array(status(ctx)['tcp_position'])
    move(ctx, tcp, quat_xyzw(R), tol_mm=0.5, timeout_s=1.0, integral=False)
    return {'ok': False, 'rise_m': up, 'reason': 'still stuck'}


if __name__ == '__main__':
    from api.runtime import Context
    params = json.loads(sys.argv[1])
    with Context() as ctx:
        assert status(ctx)['gripper']['is_grasped'], 'no peg held'
        if params.get('mode') == 'pull':
            print(json.dumps(wiggle_pull(ctx, params)))
        else:
            print(json.dumps(run(ctx, params)))
