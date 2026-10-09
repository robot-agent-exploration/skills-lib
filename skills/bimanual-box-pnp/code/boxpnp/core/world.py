"""World model shared by the primitives: objects anchored in the odometry frame, what the hands hold.

Detections arrive in the base frame of the moment they were taken; the world stores them in the odometry
frame so later steps can predict where an object is after base moves (re-observe before acting on it)."""
import json
import time

import numpy as np

from . import geom


class World:
    def __init__(self, data=None):
        data = data or {}
        self.objects = data.get('objects', {})
        self.held = data.get('held')               # name of the held object or None
        self.grip = data.get('grip', {})           # grasp data of the held object (F0 baselines, contacts, ...)
        self.notes = data.get('notes', {})

    def to_json(self):
        return {'objects': self.objects, 'held': self.held, 'grip': self.grip, 'notes': self.notes}

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def put(self, name, kind, det, odom):
        """Store a base-frame detection (centre, axes, corners ...) of `kind` seen at odometry `odom`."""
        o = {'kind': kind, 'odom_at_detection': dict(odom), 'time': time.time(), 'frame': det.get('frame')}
        o['center_odom'] = geom.base_to_odom(odom, det['center'][:2])[:2].tolist()
        for k in ('long_axis', 'short_axis'):
            if k in det:
                o[k + '_odom'] = geom.dir_base_to_odom(odom, det[k]).tolist()
        if 'corners' in det:
            o['corners_odom'] = [geom.base_to_odom(odom, c)[:2].tolist() for c in det['corners']]
        for k in ('size', 'top_z', 'support_z', 'z_spread'):
            if k in det:
                o[k] = det[k]
        self.objects[name] = o
        return o

    def in_base(self, name, odom):
        """The object as a base-frame dict for the current odometry (same keys as a detection)."""
        o = self.objects[name]
        d = {'center': geom.odom_to_base(odom, o['center_odom'])[:2].tolist()}
        for k in ('long_axis', 'short_axis'):
            if k + '_odom' in o:
                d[k] = geom.dir_odom_to_base(odom, o[k + '_odom']).tolist()
        if 'corners_odom' in o:
            d['corners'] = [geom.odom_to_base(odom, c)[:2].tolist() for c in o['corners_odom']]
        for k in ('size', 'top_z', 'support_z', 'z_spread', 'frame'):
            if k in o:
                d[k] = o[k]
        if 'short_axis' in d:
            d['yaw_short'] = float(np.arctan2(d['short_axis'][1], d['short_axis'][0]))
        return d
