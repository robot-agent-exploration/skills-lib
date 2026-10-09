"""Does an explicit [x, y] instruction change Gemini's polygon order? Decide each answer's order by which reading
puts the polygon vertices inside its own box_2d ([ymin, xmin, ymax, xmax]); edge from mask + depth for both."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(p for p in HERE.parents if (p / 'api' / 'runtime.py').exists())))
import numpy as np, cv2
from boxpnp.core import vlm, frames as pf
from api.runtime import Context
F = '20261009000618718682'
WHAT = ('the flat black table top surface of the white drawer cabinet (only the black top surface: not the white '
        'cabinet front below it, not the cardboard box or the robot arms above it)')
PROMPTS = {
 'A_explicit_xy': ('Give the segmentation mask for {w}. Output a JSON list of segmentation masks where each entry contains '
                   'the 2D bounding box in the key "box_2d" as [ymin, xmin, ymax, xmax] normalized to 0-1000, the '
                   'segmentation mask in key "mask" as a polygon: a list of points, EACH POINT WRITTEN AS [x, y] - x '
                   '(the horizontal position, the column) FIRST and y (the vertical position, the row) SECOND - both '
                   'normalized to 0-1000, and the text label in the key "label".'),
 'B_unspecified': vlm.SEG_PROMPT,
}
blk, (W, H), _ = vlm.image_block(F)
pts, pix, _, _ = pf.load(F, stride=2, max_depth=3.5)

def edge_from(poly_px):
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.array(poly_px, np.int32)], 1)
    P = pts[m[pix[:, 1], pix[:, 0]] > 0]
    P = P[(P[:, 2] > 0.55) & (P[:, 2] < 0.9)]
    if len(P) < 100:
        return None, len(P)
    z = np.median(P[:, 2]); P = P[np.abs(P[:, 2] - z) < 0.03]
    P = P[np.abs(P[:, 1]) < 0.65]
    return (round(float(np.percentile(P[:, 0], 2)), 3) if len(P) > 50 else None), len(P)

with Context() as ctx:
    for name, tmpl in PROMPTS.items():
        for trial in (1, 2):
            text = tmpl.format(w=WHAT) if '{w}' in tmpl else tmpl.format(what=WHAT)
            t0 = time.time()
            r = json.loads(ctx.call('gemini_generate', messages=[{'role': 'user', 'content': [{'type': 'text', 'text': text}, blk]}],
                                    response_format={'type': 'json_object'}, reasoning={'effort': 'minimal'},
                                    max_tokens=16384)['content'][0]['text'])
            dt = time.time() - t0
            ans = json.loads(r['message']['content'])
            ans = ans if isinstance(ans, list) else next((v for v in ans.values() if isinstance(v, list)), [ans])
            m = max(ans, key=lambda e: len(e.get('mask') or []))
            y0, x0, y1, x1 = m['box_2d']
            poly = np.array(m['mask'], float)
            inside = lambda xs, ys: float(np.mean((xs >= x0 - 15) & (xs <= x1 + 15) & (ys >= y0 - 15) & (ys <= y1 + 15)))
            in_xy = inside(poly[:, 0], poly[:, 1])          # read as [x, y]
            in_yx = inside(poly[:, 1], poly[:, 0])          # read as [y, x]
            e_xy = edge_from(np.c_[poly[:, 0] / 1000 * 1280, poly[:, 1] / 1000 * 720])
            e_yx = edge_from(np.c_[poly[:, 1] / 1000 * 1280, poly[:, 0] / 1000 * 720])
            print(f'{name} #{trial}: {dt:.1f}s, {len(poly)} vertices, first {poly[0].astype(int).tolist()}, box_2d {m["box_2d"]} | '
                  f'inside-box as [x,y] {in_xy:.2f} vs as [y,x] {in_yx:.2f} -> order {"[x,y]" if in_xy > in_yx else "[y,x]"} | '
                  f'edge as [x,y] {e_xy[0]} ({e_xy[1]} pts), as [y,x] {e_yx[0]} ({e_yx[1]} pts)')
