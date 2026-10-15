"""Wrist-camera top-view perception for the wooden bandu pieces on the white table.

Metric xy comes from the wrist ray through a pixel intersected with a known horizontal plane
(the piece's top height), not from wrist depth (unreliable close up). Exterior depth agrees
with this within a few mm.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

FRAMES = Path('runtime') / 'frames'     # saved frames, relative to the working directory (workspace root)

# Wood: hue 5-30, saturation > 50 (table: saturation < 25).
WOOD_H = (5, 30)
WOOD_S_MIN = 50
WOOD_V_MIN = 40

# Wrist camera in the TCP frame (fixed; from meta.json): t = (-0.0725, 0.0305, -0.0804).
CAM_IN_TCP = np.array([-0.0725, 0.0305, -0.0804])


def camera_offset(yaw):
    """Base-frame xy offset camera - TCP for a gripper pointing down with this yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    # TCP frame axes in base: x=(c,s,0), y=(s,-c,0), z=(0,0,-1)
    ox = CAM_IN_TCP[0] * c + CAM_IN_TCP[1] * s
    oy = CAM_IN_TCP[0] * s - CAM_IN_TCP[1] * c
    return np.array([ox, oy])


def view_pose_for(target_xy, yaw=0.0, cam_z=0.40, back=0.06):
    """TCP position that puts the wrist camera `back` m behind target (opposite to the camera's
    forward tilt) at height cam_z, so the target is seen at ~10 deg from vertical."""
    c, s = math.cos(yaw), math.sin(yaw)
    forward = np.array([c, s])          # camera tilts towards TCP +x
    cam_xy = np.asarray(target_xy, float) - back * forward
    tcp_xy = cam_xy - camera_offset(yaw)
    # pointing down, the camera sits -CAM_IN_TCP[2] = 0.0804 m above the TCP
    return [float(tcp_xy[0]), float(tcp_xy[1]), cam_z + CAM_IN_TCP[2]]


class Frame:
    def __init__(self, frame_id, camera='wrist'):
        self.frame_id = frame_id
        self.camera = camera
        folder = FRAMES / frame_id
        self.meta = json.loads((folder / 'meta.json').read_text())
        cam = self.meta['cameras'][camera]
        self.K = np.array(cam['K'])
        self.T = np.array(cam['T_base_cam'])
        self.rgb = cv2.imread(str(folder / cam['rgb']))

    def ray(self, uv):
        d = np.linalg.solve(self.K, [uv[0], uv[1], 1.0])
        d = self.T[:3, :3] @ (d / np.linalg.norm(d))
        return self.T[:3, 3], d

    def to_plane(self, uv, z):
        o, d = self.ray(uv)
        return o + d * (z - o[2]) / d[2]

    def project(self, p):
        P = np.linalg.inv(self.T) @ np.r_[np.asarray(p, float), 1.0]
        return (self.K @ (P[:3] / P[2]))[:2]

    def wood_mask(self, open_px=5):
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 0] >= WOOD_H[0]) & (hsv[..., 0] <= WOOD_H[1]) &
             (hsv[..., 1] > WOOD_S_MIN) & (hsv[..., 2] > WOOD_V_MIN)).astype(np.uint8) * 255
        if open_px:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
        return m

    def blobs(self, min_area=600):
        """Connected wood regions: list of dicts with pixel stats and contour."""
        m = self.wood_mask()
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        out = []
        for i in range(1, n):
            if stats[i, 4] < min_area:
                continue
            region = (lab == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            out.append({'id': i, 'area_px': int(stats[i, 4]), 'bbox': [int(v) for v in stats[i, :4]],
                        'centroid_px': [float(cent[i][0]), float(cent[i][1])], 'contour': c,
                        'mask': region})
        return out

    def rect_on_plane(self, contour, z):
        """Min-area rectangle of a contour after mapping its points to the plane z (metric)."""
        pts = np.array([self.to_plane((float(p[0]), float(p[1])), z)[:2] for p in contour[:, 0]], np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        # report long side as length, angle of the long side (rad, base frame)
        if w >= h:
            length, width, theta = w, h, math.radians(ang)
        else:
            length, width, theta = h, w, math.radians(ang + 90)
        theta = (theta + math.pi / 2) % math.pi - math.pi / 2
        return {'center': [float(cx), float(cy)], 'length': float(length), 'width': float(width),
                'long_axis_angle': float(theta), 'points': pts}

    def circle_on_plane(self, contour, z):
        """Centre and diameter of a roughly circular blob's top face mapped to plane z."""
        pts = np.array([self.to_plane((float(p[0]), float(p[1])), z)[:2] for p in contour[:, 0]], np.float32)
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        return {'center': [float(cx), float(cy)], 'diameter': float(2 * r)}
