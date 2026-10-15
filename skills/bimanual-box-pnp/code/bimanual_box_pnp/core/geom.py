"""Planar frames: base <-> odometry. The wheel odometry pose {x, y, yaw} is the base origin in the odom frame."""
import numpy as np


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def base_to_odom(pose, p):
    c, s = np.cos(pose['yaw']), np.sin(pose['yaw'])
    p = np.asarray(p, float)
    return np.array([pose['x'] + c * p[0] - s * p[1], pose['y'] + s * p[0] + c * p[1], *p[2:]])


def odom_to_base(pose, p):
    c, s = np.cos(pose['yaw']), np.sin(pose['yaw'])
    p = np.asarray(p, float)
    dx, dy = p[0] - pose['x'], p[1] - pose['y']
    return np.array([c * dx + s * dy, -s * dx + c * dy, *p[2:]])


def dir_base_to_odom(pose, v):
    c, s = np.cos(pose['yaw']), np.sin(pose['yaw'])
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def dir_odom_to_base(pose, v):
    c, s = np.cos(pose['yaw']), np.sin(pose['yaw'])
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
