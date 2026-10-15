"""Offline test of the two-cup stacking perception on recorded frames (no robot motion; Gemini calls only).

  python skill-library/skills/stack-cups/code/offline_cups.py

References: what the frames show and what happened next on the robot (2026-10-09, RUNS.md C1 / C2).
Report: output/pick_place/offline/report_cups.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stack_cups import pp_cups as K  # noqa: E402  (first: the package puts office-chair-tuck on sys.path)
from office_chair_tuck import geom  # noqa: E402
from stack_cups import pp_perception as PP  # noqa: E402
from stack_cups import pp_vis  # noqa: E402

OUT = geom.ROOT / 'output' / 'pick_place' / 'offline'
LOCATE = [('2026100920221219ee6d', ['silver cup on the left', 'silver cup in the front middle'])]
ALIGN = [  # frame, target (centre, rim_z), aligned? (from what happened when lowering)
    ('202610092025339e0094', {'centre': [0.399, 0.276], 'rim_z': 0.848}, False),   # C2 first look: 2.5 cm off
    ('2026100920261890a2b0', {'centre': [0.399, 0.276], 'rim_z': 0.848}, True),    # C2 after 1 correction: nested
    ('202610092004047deae8', {'centre': [0.424, 0.241], 'rim_z': 0.845}, True),    # C1: nested when lowered
]
HELD = [('20261009202408c70a64', 0.743, 'held'),        # C2 cup lifted after the pinch
        ('202610092036186271a2', 0.743, 'held'),        # a cup lifted 6 cm (old check misjudged it)
        ('20261009203653d0e3e7', 0.743, 'not held')]    # gripper empty above a standing cup
NESTED = [('2026100920270122eb78', 'yes'), ('2026100920221219ee6d', 'no')]


def rim_band_centre(f, mask):
    P, _ = pp_vis.mask_points(f, cv2.dilate(mask, np.ones((15, 15), np.uint8)), erode=0)
    zr = np.percentile(P[:, 2], 99)
    ring = P[P[:, 2] > zr - 0.008]
    return np.array([np.percentile(ring[:, 0], [1, 99]).mean(), np.percentile(ring[:, 1], [1, 99]).mean()])


def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    rep, passed = {}, []
    rows = []
    for fid, names in LOCATE:
        seg = PP.segment_named(ctx, fid, names, log=lambda *a: None)
        for n in names:
            c = PP.cup(ctx, fid, n, log=lambda *a: None)
            band = rim_band_centre(geom.load(fid), seg['masks'][n]) if n in seg['masks'] else None
            d = None if band is None or not c.get('ok') else float(np.linalg.norm(np.array(c['centre']) - band))
            ok = d is not None and d < 0.015 and 0.04 < c['r'] < 0.055
            print(f"locate {fid} {n:32s} {'PASS' if ok else 'FAIL'} lips {c.get('centre')} r {c.get('r')} "
                  f"band {None if band is None else band.round(3)} diff {None if d is None else round(d, 3)}")
            rows.append({'frame': fid, 'cup': n, 'lips': c.get('centre'), 'r': c.get('r'), 'rim_z': c.get('rim_z'),
                         'band': None if band is None else band.round(3).tolist(), 'pass': ok})
            passed.append(ok)
    rep['locate'] = rows
    rows = []
    for fid, T, exp in ALIGN:
        b, low = K.held_bottom(ctx, fid, T['rim_z'], log=lambda *a: None)
        off = None if b is None else float(np.linalg.norm(b - np.array(T['centre'])))
        got = off is not None and off < 2 * K.ALIGN_TOL
        g = PP.check(ctx, fid, "The robot's gripper holds a silver cup by its rim above another cup on the desk. Is the "
                               'bottom of the hanging cup directly above the opening of the cup below, so that lowering '
                               'it straight down puts it inside?', log=None)
        print(f"align {fid} {'PASS' if got == exp else 'FAIL'} expected {exp} depth offset {None if off is None else round(off, 3)} "
              f"| gemini {g.get('answer')} ({'agrees' if (g.get('answer') == 'yes') == exp else 'disagrees'})")
        rows.append({'frame': fid, 'expected': exp, 'offset': off, 'depth_pass': got == exp, 'gemini': g.get('answer'),
                     'gemini_pass': (g.get('answer') == 'yes') == exp, 'both_gate': bool(got and g.get('answer') == 'yes')})
        passed.append(got == exp)
    rep['align'] = rows
    rows = []
    for fid, sz, exp in HELD:
        v, low = K.held_check(ctx, fid, sz, log=lambda *a: None)
        ok = v == exp or (exp == 'not held' and v == 'unsure')      # unsure = ask, never a wrong release
        print(f"held {fid} {'PASS' if ok else 'FAIL'} expected {exp} got {v} lowest {low}")
        rows.append({'frame': fid, 'expected': exp, 'got': v, 'lowest': low, 'pass': ok})
        passed.append(ok)
    rep['held'] = rows
    rows = []
    for fid, exp in NESTED:
        c = PP.check(ctx, fid, 'Is one silver cup nested inside another silver cup (one inside the other, upright, both '
                               'rims together at the top)?', log=None)
        ok = c.get('answer') == exp
        print(f"nested {fid} {'PASS' if ok else 'FAIL'} expected {exp} got {c.get('answer')} ({c.get('reason')})")
        rows.append({'frame': fid, 'expected': exp, 'got': c.get('answer'), 'pass': ok})
        passed.append(ok)
    rep['nested'] = rows
    rep['summary'] = {'passed': int(sum(passed)), 'total': len(passed), 'gemini': PP.vlm.usage()}
    (OUT / 'report_cups.json').write_text(json.dumps(rep, indent=1, default=str))
    print('SUMMARY', rep['summary'])
    return rep


if __name__ == '__main__':
    sys.path.insert(0, str(geom.ROOT))
    from api.runtime import Context
    with Context() as _ctx:
        main(_ctx)
