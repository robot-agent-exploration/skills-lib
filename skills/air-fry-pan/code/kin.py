"""FR3 forward/inverse kinematics (modified DH) for planning joint-space moves.

Only used to pick well-conditioned joint targets offline; the robot service still
checks every path.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# (a, d, alpha) modified DH, Craig convention, then flange and hand TCP.
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
from air_fry_pan_params import HOME  # hyper-parameters: air_fry_pan_params.py


def _tf(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def fk_all(q):
    """List of 4x4 transforms: joint frames 1..7, flange, TCP."""
    T = np.eye(4)
    out = []
    for (a, d, al), th in zip(DH, q):
        T = T @ _tf(a, d, al, th)
        out.append(T.copy())
    T = T @ _tf(0, 0.107, 0, 0)
    out.append(T.copy())
    hand = np.eye(4)
    hand[:3, :3] = R.from_euler('z', -np.pi / 4).as_matrix()
    hand[2, 3] = 0.1034
    out.append(T @ hand)
    return out


def fk(q):
    """TCP position and quaternion [x, y, z, w]."""
    T = fk_all(q)[-1]
    return T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()


def ik(pos, quat, seed, iters=300, tol=1e-4):
    """Damped least-squares IK with joint-limit clamping; returns (q, ok, err)."""
    q = np.array(seed, dtype=float)
    Rt = R.from_quat(quat).as_matrix()
    pos = np.asarray(pos, dtype=float)
    for _ in range(iters):
        Ts = fk_all(q)
        T = Ts[-1]
        e_p = pos - T[:3, 3]
        e_r = R.from_matrix(Rt @ T[:3, :3].T).as_rotvec()
        e = np.r_[e_p, e_r]
        if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 1e-3:
            return q, True, e
        J = np.zeros((6, 7))
        p_e = T[:3, 3]
        for i in range(7):
            z = Ts[i][:3, 2]
            p = Ts[i][:3, 3]
            J[:3, i] = np.cross(z, p_e - p)
            J[3:, i] = z
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        # pull gently toward the seed posture in the null space
        N = np.eye(7) - np.linalg.pinv(J) @ J
        dq += N @ (0.05 * (np.asarray(seed) - q))
        q = np.clip(q + dq, Q_MIN + 0.02, Q_MAX - 0.02)
    return q, False, e


def quat_tool(z_axis, y_axis):
    """Quaternion of a TCP frame with given z (approach) and y (finger travel) axes."""
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    y = np.asarray(y_axis, float); y -= z * (y @ z); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return R.from_matrix(np.c_[x, y, z]).as_quat()


def margin(q):
    return float(np.min(np.minimum(np.asarray(q) - Q_MIN, Q_MAX - np.asarray(q))))


def ik_best(pos, quat, seeds=(), n_random=40, rng_seed=0, prefer=None, w_pref=0.05):
    """Multi-start IK; returns (q, margin) maximizing joint margin (minus distance to `prefer`)."""
    rng = np.random.default_rng(rng_seed)
    cands = list(seeds) + [rng.uniform(Q_MIN + 0.1, Q_MAX - 0.1) for _ in range(n_random)]
    best, best_score = None, -np.inf
    for s in cands:
        q, ok, _ = ik(pos, quat, s, iters=250)
        if not ok:
            continue
        score = margin(q) - (w_pref * np.linalg.norm(q - prefer) if prefer is not None else 0.0)
        if score > best_score:
            best, best_score = q, score
    return best, (margin(best) if best is not None else None)

# The robot service validates move_joints targets against limits that look like the Panda's
# for j7 (+-2.8973) minus 0.2 rad; plan against the intersection of Panda and FR3 limits.
PANDA_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
SAFE_MIN = np.maximum(PANDA_MIN, Q_MIN)
SAFE_MAX = np.minimum(PANDA_MAX, Q_MAX)


def safe_margin(q):
    """Joint margin (rad) against the conservative Panda/FR3 limit intersection."""
    q = np.asarray(q)
    return float(np.min(np.minimum(q - SAFE_MIN, SAFE_MAX - q)))
