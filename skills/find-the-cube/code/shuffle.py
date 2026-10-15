"""Which cup hides the cube? Replay of the shuffle seen by the wrist camera at the watch pose.

Cups: green blobs tracked at 30 Hz by k-means seeded with the previous centres.
Cube: its last known pixel P.
- 30 Hz: a template of the cube is matched around P -> visible / occluded.
- 1 Hz: the depth frame is searched for a cube-shaped bump; repeated detections away from P
  mean the operator moved the cube (P moves, the template is re-captured there).
- The cube is covered by cup k when it is occluded at P while k's track sits on P for
  COVER_FRAMES frames. The last covering wins; seeing the cube again cancels it.
"""
import time

import cv2 as cv
import numpy as np

import tracker

from find_the_cube_params import (  # hyper-parameters: find_the_cube_params.py
    TPL_HALF, SEARCH, VISIBLE_SCORE, FOLLOW_SCORE, UNCOVER_SCORE, TPL_DIRT_MAX, COVER_PX, COVER_FRAMES,
    MOVE_MIN_PX, MOVE_CONFIRM_PX, CUBE_SIZE_MM, CUBE_AREA, CUBE_MIN_H, HOLE_AREA, HOLE_WHITE,
    HOLE_COLOUR_DIST, HOLE_DARK_V, HOLE_SKIN_MAX, INSIDE_PX, INSIDE_FRAMES,
)


def gray_half(image):
    return cv.resize(cv.cvtColor(image, cv.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5,
                     interpolation=cv.INTER_AREA)


def crop_template(gray, p):
    u, v, h = int(round(p[0] * 0.5)), int(round(p[1] * 0.5)), TPL_HALF
    if u - h < 0 or v - h < 0 or u + h > gray.shape[1] or v + h > gray.shape[0]:
        return None
    return gray[v - h:v + h, u - h:u + h].copy()


def object_like(small_bgr, model):
    """Pixels with the object's colour (Lab distance, L weighted 0.3); none without a model."""
    if model is None or model.get('lab') is None:
        return np.zeros(small_bgr.shape[:2], bool)
    d = cv.cvtColor(small_bgr, cv.COLOR_BGR2LAB).astype(float) - np.array(model['lab'])
    return np.sqrt((0.3 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) < HOLE_COLOUR_DIST


def template_dirt(image, p, model=None):
    """Fraction of green (cup) or skin (hand) pixels in the template patch around p; pixels of
    the object's own colour never count as skin."""
    u, v, h = int(round(p[0] * 0.5)), int(round(p[1] * 0.5)), TPL_HALF
    patch = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)[max(0, v - h):v + h, max(0, u - h):u + h]
    if patch.size == 0:
        return 1.0
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    green = cv.inRange(hsv, tracker.HSV_LO, tracker.HSV_HI) > 0
    skin = (hsv[..., 0] < 25) & (hsv[..., 1] > 40) & (hsv[..., 1] < 180) & (hsv[..., 2] > 60) & ~object_like(patch, model)
    return float((green | skin).mean())


def score_at(gray, tpl, p, s=SEARCH):
    """Best template correlation within s (half-res px) of p, and where (full-res px)."""
    u, v, h = int(round(p[0] * 0.5)), int(round(p[1] * 0.5)), TPL_HALF
    y0, y1 = max(0, v - h - s), min(gray.shape[0], v + h + s)
    x0, x1 = max(0, u - h - s), min(gray.shape[1], u + h + s)
    win = gray[y0:y1, x0:x1]
    if win.shape[0] < 2 * h or win.shape[1] < 2 * h:
        return 0.0, np.array(p, float)
    r = cv.matchTemplate(win, tpl, cv.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv.minMaxLoc(r)
    return float(mx), np.array([(x0 + loc[0] + h) / 0.5, (y0 + loc[1] + h) / 0.5])


def cube_in_cup(image, roi, model=None):
    """Object-coloured, object-sized holes inside green cup blobs (the object lying in an
    upright cup). Colour: Lab distance to the object's colour (white rule without a model).
    Returns [(centroid full-res px, area half-res px)]."""
    small = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    green = cv.inRange(hsv, tracker.HSV_LO, tracker.HSV_HI)
    green = cv.morphologyEx(green, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    x0, y0, x1, y1 = [int(v * 0.5) for v in roi]
    keep = np.zeros_like(green)
    keep[y0:y1, x0:x1] = 255
    green &= keep
    contours, hierarchy = cv.findContours(green, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    if model is not None and model.get('lab') is not None:
        lab = cv.cvtColor(small, cv.COLOR_BGR2LAB).astype(float)
        d = lab - np.array(model['lab'])
        like = np.sqrt((0.3 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) < HOLE_COLOUR_DIST
        lo, hi = max(60, 0.2 * model['area']), max(HOLE_AREA[1], 2.5 * model['area'])
    else:
        like = (hsv[..., 1] < 70) & (hsv[..., 2] > 140)
        lo, hi = HOLE_AREA
    skin = (hsv[..., 0] < 25) & (hsv[..., 1] > 40) & (hsv[..., 1] < 180) & (hsv[..., 2] > 60) & ~like   # a red/orange
    out = []                                                                   # object is not skin
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] < 0:                   # outer boundary of a blob, not a hole
            continue
        area = cv.contourArea(c)
        if not lo <= area <= hi:
            continue
        m = np.zeros(green.shape, np.uint8)
        cv.drawContours(m, [c], -1, 1, -1)
        m = m.astype(bool) & (green == 0) & (hsv[..., 2] >= HOLE_DARK_V)   # drop the shadowed wall
        n = m.sum()
        if n < max(40, 0.5 * lo) or like[m].mean() < HOLE_WHITE or skin[m].mean() > HOLE_SKIN_MAX:
            continue
        ys, xs = np.nonzero(m)
        out.append((np.array([xs.mean(), ys.mean()]) / 0.5, int(n)))
    return out


def plausible_cube(det, model=None):
    """A depth detection that looks like the object seen before the shuffle."""
    if det is None:
        return False
    w, h = det['size_mm']
    height = det.get('top_z', det['z']) - det['table_z']
    if model is None:
        return (CUBE_SIZE_MM[0] <= min(w, h) and max(w, h) <= CUBE_SIZE_MM[1]
                and CUBE_AREA[0] <= det['area'] <= CUBE_AREA[1] and det['z'] - det['table_z'] >= CUBE_MIN_H)
    mw, mh = model['size_mm']
    return (min(w, h) >= 0.6 * min(mw, mh) and max(w, h) <= min(90.0, 1.5 * max(mw, mh))
            and 0.4 * model['area'] <= det['area'] <= 2.5 * model['area']
            and height >= max(0.008, 0.5 * model['height']))


def init_cups(image, roi, k):
    mask = tracker.green_mask(image, roi)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if len(pts) < 200 * k:
        return None
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, c = cv.kmeans(pts, k, None, crit, 8, cv.KMEANS_PP_CENTERS)
    return np.array(sorted(c.tolist()), float) / tracker.SCALE   # sorted: stable labels


def identify(frames, depth_frames, cube_px, detect, roi, n_cups=4, log=print, model=None):
    """frames: [(t_ms, path)] wrist RGB; depth_frames: [(t_ms, path)] uint16 mm depth;
    detect(image, depth_m) -> cube dict (controller.find_cube) or None.
    Returns {'ok', 'cover_id', 'final_px', 'cube_px', 'events', 'hist', 'why'}."""
    first = cv.imread(str(frames[0][1]))
    centres = init_cups(first, roi, n_cups)
    if centres is None:
        return {'ok': False, 'why': 'cups not visible at shuffle start', 'events': [], 'hist': []}
    c = centres * tracker.SCALE
    P = np.array(cube_px, float)
    tpl = crop_template(gray_half(first), P)
    if tpl is None:
        return {'ok': False, 'why': 'cube too close to the image border', 'events': [], 'hist': []}
    covered_by, run, run_id, cands = None, 0, None, []
    in_run, in_id = 0, None
    events, hist = [], []
    di = 0
    while di < len(depth_frames) and depth_frames[di][0] < frames[0][0]:
        di += 1
    for t, path in frames:
        im = cv.imread(str(path))
        if im is None:
            continue
        gray = gray_half(im)
        # cups
        mask = tracker.green_mask(im, roi)
        ys, xs = np.nonzero(mask)
        if len(xs) > 50:
            new, cnt = tracker.kmeans_step(np.stack([xs, ys], 1).astype(float), c)
            for k in range(n_cups):
                if cnt[k] >= 40:
                    c[k] = new[k]
        full = c / tracker.SCALE
        # cube moved? (1 Hz depth)
        while di < len(depth_frames) and depth_frames[di][0] <= t:
            td, dpath = depth_frames[di]
            di += 1
            depth = cv.imread(str(dpath), cv.IMREAD_UNCHANGED)
            det = detect(im, depth.astype(float) / 1000.0) if depth is not None else None
            if plausible_cube(det, model) and np.hypot(*(np.array(det['px']) - P)) > MOVE_MIN_PX:
                px = np.array(det['px'])
                cands = [q for q in cands if np.hypot(*(q - px)) < MOVE_CONFIRM_PX] + [px]
                need = 3 if covered_by is not None else 2     # stricter while hidden under a cup
                if len(cands) >= need:
                    ntpl = crop_template(gray, px)
                    if ntpl is not None and template_dirt(im, px, model) <= TPL_DIRT_MAX:
                        P, tpl = px, ntpl
                        if covered_by is not None:
                            events.append(('uncovered', t, covered_by))
                        covered_by, run, run_id = None, 0, None
                        events.append(('moved', t, P.round(1).tolist()))
                        log('cube moved to', P.round(1).tolist(), 'at', t)
                    cands = []
            else:
                cands = []
        # cube visible at P? (30 Hz template). A pushed or hand-moved cube no longer matches
        # its template (scores ~0.75 even where it is), so moves are left to the depth check.
        score, loc = score_at(gray, tpl, P)
        # cube dropped into an upright cup (only once it has left P)
        if score < VISIBLE_SCORE:
            holes = cube_in_cup(im, roi, model)
            seen_k = None
            for hc, _ in holes:
                d = np.hypot(*(full - hc).T)
                k = int(d.argmin())
                if d[k] < INSIDE_PX:
                    seen_k = k
                    break
            if seen_k is not None:
                in_run = in_run + 1 if in_id == seen_k else 1
                in_id = seen_k
                if in_run >= INSIDE_FRAMES and covered_by != seen_k:
                    covered_by, run, run_id = seen_k, 0, None
                    events.append(('dropped_in', t, seen_k, np.round(holes[0][0], 1).tolist()))
                    log('cube seen inside cup', seen_k, 'at', t)
            else:
                in_run, in_id = 0, None
        if score >= VISIBLE_SCORE and (covered_by is None or score >= UNCOVER_SCORE):
            if covered_by is not None:
                events.append(('uncovered', t, covered_by))
                log('cube visible again at', t, '- cancelling cover by', covered_by)
                covered_by = None
            run, run_id = 0, None
            if score >= FOLLOW_SCORE:
                P = loc
        elif covered_by is None:   # once covered, the cube travels with that cup until seen again
            d = np.hypot(*(full - P).T)
            k = int(d.argmin())
            if d[k] < COVER_PX:
                run = run + 1 if run_id == k else 1
                run_id = k
                if run >= COVER_FRAMES:
                    covered_by = k
                    events.append(('covered', t, k, P.round(1).tolist()))
                    log('cup', k, 'covered the cube at', t, 'at', P.round(1).tolist())
            else:
                run, run_id = 0, None
        hist.append((t, full.copy(), P.copy(), score))
    final = hist[-1][1]
    sep = min(np.hypot(*(final[i] - final[j])) for i in range(n_cups) for j in range(i + 1, n_cups))
    out = {'cover_id': covered_by, 'final_px': final.round(1).tolist(), 'cube_px': P.round(1).tolist(),
           'events': events, 'hist': hist, 'min_sep_px': float(sep), 'final_score': hist[-1][3]}
    if covered_by is None:
        return dict(out, ok=False, why='the cube was not seen being covered by a cup')
    if sep < 45:
        return dict(out, ok=False, why=f'two cup tracks ended on one cup (separation {sep:.0f}px)')
    return dict(out, ok=True, cup_px=final[covered_by].tolist())


def save_sheet(frames, result, path, roi, count=16):
    """Contact sheet: cup tracks (red C = the cup holding the cube), cube box (orange when
    the cube is in view at P, grey when occluded), template score."""
    paths = dict(frames)
    hist = result['hist']
    if not hist:
        return
    idx = np.linspace(0, len(hist) - 1, count).astype(int)
    tiles = []
    for i in idx:
        t, cups, P, score = hist[i]
        im = cv.imread(str(paths[t]))
        for k, p in enumerate(cups):
            col = (0, 0, 255) if k == result['cover_id'] else (255, 0, 0)
            cv.circle(im, (int(p[0]), int(p[1])), 14, col, 4)
            cv.putText(im, 'C' if k == result['cover_id'] else str(k), (int(p[0]) + 16, int(p[1]) - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
        col = (0, 200, 255) if score >= VISIBLE_SCORE else (128, 128, 128)
        cv.rectangle(im, (int(P[0]) - 30, int(P[1]) - 30), (int(P[0]) + 30, int(P[1]) + 30), col, 4)
        im = cv.resize(im[0:roi[3]], (480, 195))
        cv.putText(im, time.strftime('%H:%M:%S', time.localtime(t / 1000)) + f'.{t % 1000:03d} s={score:.2f}',
                   (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
