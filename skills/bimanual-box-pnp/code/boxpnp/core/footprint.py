"""Robot footprint for base-motion checks: layers (polygon in the base frame, lowest z of that robot part),
poses along a move (turns pivot about the wheel axle), and the robot's own points in a depth frame.

Why: run A2 (2026-10-08) turned and reversed the base into a box and a cabinet with unchecked moves."""
import json
from pathlib import Path

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union

from . import geom, kin

RES = 0.04
FLOOR_Z = 0.06
OBST_Z = (0.15, 1.7)
MIN_PTS = 3                 # points per cell (stride-2 sampling) to call it occupied / free

BASE_POLY = unary_union([sbox(-0.335, -0.259, 0.304, 0.259),            # base body (mesh bounds)
                         sbox(0.128, 0.24, 0.328, 0.29), sbox(0.128, -0.29, 0.328, -0.24)])   # wheels
ARMS_READY = (sbox(0.20, -0.37, 0.64, 0.37), 1.0)        # arms at ready: hands to x 0.62 above z ~1.05
CARRY = (sbox(0.20, -0.36, 0.62, 0.36), 0.75)            # carry posture: box bottom ~0.8, grippers, forearms


def footprint_layers(posture='ready', margin=0.06):
    """[(polygon in the base frame, lowest z of that robot part)] buffered by `margin`."""
    layers = [(BASE_POLY.buffer(margin), 0.0)]
    if posture == 'ready':
        layers.append((ARMS_READY[0].buffer(margin), ARMS_READY[1]))
    elif posture == 'carry':
        layers.append((CARRY[0].buffer(margin), CARRY[1]))
    return layers


def place(poly, pose):
    """A base-frame polygon placed at odometry pose (x, y, yaw)."""
    return affinity.translate(affinity.rotate(poly, pose[2], origin=(0, 0), use_radians=True), pose[0], pose[1])


def sample_move(pose, move, step_m=0.02, step_rad=0.03):
    """Poses along one base move from `pose` (odom): {'forward': d} straight, {'turn': a} about the wheel axle."""
    x, y, th = pose
    out = []
    if 'forward' in move:
        d = move['forward']
        n = max(1, int(np.ceil(abs(d) / step_m)))
        for k in range(n + 1):
            s = d * k / n
            out.append((x + s * np.cos(th), y + s * np.sin(th), th))
    else:
        a = move['turn']
        n = max(1, int(np.ceil(abs(a) / step_rad)))
        ax, ay = x + kin_axle() * np.cos(th), y + kin_axle() * np.sin(th)
        for k in range(n + 1):
            t = th + a * k / n
            out.append((ax - kin_axle() * np.cos(t), ay - kin_axle() * np.sin(t), t))
    return out


def kin_axle():
    return 0.228


def _to_odom(odom, xy):
    c, s = np.cos(odom['yaw']), np.sin(odom['yaw'])
    return np.stack([odom['x'] + c * xy[:, 0] - s * xy[:, 1], odom['y'] + s * xy[:, 0] + c * xy[:, 1]], 1)


SELF_FRAMES = (['link_torso_%d' % i for i in range(6)] + ['link_head_0', 'link_head_1', 'link_head_2'] +
               [f'link_{s}_arm_{i}' for s in ('right', 'left') for i in range(7)] +
               ['ee_right', 'ee_left', 'ee_finger_r1', 'ee_finger_l1'])
CHAINS = [['link_torso_0', 'link_torso_1', 'link_torso_2', 'link_torso_3', 'link_torso_5', 'link_head_0', 'link_head_2']] + \
         [['link_torso_5'] + [f'link_{s}_arm_{i}' for i in range(7)] + [f'ee_{s}', f'ee_finger_{s[0]}1'] for s in ('right', 'left')]


def self_mask(pts, joints, held_box=None, radius=0.14, base_margin=0.08, arm_envelope=(0.10, 0.70, 0.45, 0.85)):
    """Points belonging to the robot: inside the base footprint (+margin, below 0.5 m), near the link chain
    (capsules from FK of the frame's joints), or inside a held box region {'min': xyz, 'max': xyz}."""
    q = kin.q_from(torso=[joints[f'torso_{i}'] for i in range(6)],
                   right=[joints[f'right_arm_{i}'] for i in range(7)],
                   left=[joints[f'left_arm_{i}'] for i in range(7)],
                   head=[joints['head_0'], joints['head_1']])
    fr = kin.frames(q, SELF_FRAMES)
    P = {k: v.translation.copy() for k, v in fr.items()}
    for s in ('right', 'left'):                                    # closed round fingers reach 0.195 + 0.03 m
        R = fr[f'ee_{s}'].rotation
        P[f'ee_finger_{s[0]}1'] = P[f'ee_{s}'] - 0.23 * R[:, 2]
    m = (pts[:, 0] > -0.335 - base_margin) & (pts[:, 0] < 0.304 + base_margin) & \
        (np.abs(pts[:, 1]) < 0.29 + base_margin) & (pts[:, 2] < 0.5)
    if arm_envelope is not None:      # the arms' own space now (anything real there would already touch them)
        x0, x1, ymax, zmin = arm_envelope
        m |= (pts[:, 0] > x0) & (pts[:, 0] < x1) & (np.abs(pts[:, 1]) < ymax) & (pts[:, 2] > zmin)
    for chain in CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            A, Bp = P[a], P[b]
            d = Bp - A
            L2 = float(d @ d)
            t = np.clip(((pts - A) @ d) / L2, 0, 1) if L2 > 1e-9 else np.zeros(len(pts))
            m |= np.linalg.norm(pts - (A + t[:, None] * d), axis=1) < radius
    if held_box is not None:
        lo, hi = np.array(held_box['min']), np.array(held_box['max'])
        m |= np.all((pts >= lo) & (pts <= hi), axis=1)
    return m
