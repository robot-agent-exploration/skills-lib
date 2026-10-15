"""Parse binary STL files and print bounding boxes / cross-section info."""
import struct
import sys

import numpy as np


def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if 84 + n * 50 == len(data):
        rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([
            ('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        return rec['v'].astype(float)
    # ascii fallback
    verts = []
    for line in data.decode(errors='ignore').splitlines():
        s = line.strip().split()
        if s and s[0] == 'vertex':
            verts.append([float(x) for x in s[1:4]])
    return np.array(verts).reshape(-1, 3, 3)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        tri = read_stl(path)
        v = tri.reshape(-1, 3)
        print(path.split('/')[-1], 'triangles', len(tri))
        print('  min', np.round(v.min(0), 3), 'max', np.round(v.max(0), 3), 'size', np.round(v.max(0) - v.min(0), 3))
