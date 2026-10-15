"""Track green cups through the recorded stream (30 Hz) with k-means seeded by the
previous centres, so each cluster stays attached to its cup while cups slide around."""
import cv2 as cv
import numpy as np

SCALE = 0.5                      # process at half resolution
HSV_LO = (55, 90, 50)            # green cup colour (OpenCV H 0-180)
HSV_HI = (100, 255, 255)
MIN_BLOB_PX = 150                # at half resolution


def green_mask(image, roi=None):
    small = cv.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if roi is not None:  # full-resolution x0, y0, x1, y1
        keep = np.zeros_like(mask)
        x0, y0, x1, y1 = [int(v * SCALE) for v in roi]
        keep[y0:y1, x0:x1] = 1
        mask = mask * keep
    return mask


def blobs(mask):
    """Centroids (full-res px) and areas of green connected components."""
    n, _, stats, cents = cv.connectedComponentsWithStats(mask)
    out = [(cents[i] / SCALE, stats[i, cv.CC_STAT_AREA]) for i in range(1, n)
           if stats[i, cv.CC_STAT_AREA] >= MIN_BLOB_PX]
    return sorted(out, key=lambda b: -b[1])


def kmeans_step(points, centres, iters=5):
    """Lloyd iterations seeded at the previous centres; returns new centres and counts."""
    c = centres.copy()
    counts = np.zeros(len(c))
    for _ in range(iters):
        d = ((points[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(len(c)):
            sel = points[lab == k]
            counts[k] = len(sel)
            if len(sel) > 0:
                c[k] = sel.mean(0)
    return c, counts


def track(camera, t0_ms, t1_ms, init_centres, roi=None, step=1, log=None):
    """Follow len(init_centres) cups from t0 to t1. Returns [(t_ms, centres)] (full-res px).
    Not used by the controller (shuffle.identify does the tracking); needs the harness's api.stream."""
    from api import stream            # imported here (library import: was at module level)
    frames = stream.frames(camera, since_ms=t0_ms, until_ms=t1_ms)[::step]
    centres = np.array(init_centres, dtype=float) * SCALE
    history = []
    for t, path in frames:
        image = cv.imread(str(path))
        if image is None:
            continue
        mask = green_mask(image, roi)
        ys, xs = np.nonzero(mask)
        if len(xs) > 50:
            pts = np.stack([xs, ys], 1).astype(float)
            new, counts = kmeans_step(pts, centres)
            # a cluster that lost (almost) all its pixels stays where it was (occluded)
            for k in range(len(centres)):
                if counts[k] >= 40:
                    centres[k] = new[k]
        history.append((t, centres / SCALE))
        if log is not None:
            log.append((t, [tuple(np.round(c / SCALE, 1)) for c in centres]))
    return history


def draw(image, centres, labels=None, colour=(0, 0, 255)):
    out = image.copy()
    for i, c in enumerate(centres):
        p = (int(c[0]), int(c[1]))
        cv.circle(out, p, 10, colour, 2)
        cv.putText(out, str(labels[i] if labels else i), (p[0] + 12, p[1]), cv.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return out
