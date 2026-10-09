"""Gemini (`gemini_generate`) perception on head-camera frames: structured questions, pointing, yes / no checks.

Geometry stays in code (depth, FK); Gemini answers what a hand-written rule answers badly: which object is the
target, whether it is fully visible, where its top face / a table's corners are, whether a grasp or a placement
looks right. Every answer is evidence for the caller, not a decision.

Measured on this robot (2026-10-08, session practice_pnp_box_61befa):
- 640x360 JPEG (q85, ~35 kB) + json_schema + reasoning effort low: 2-6 s, ~$0.004, points within ~6 px of the
  depth geometry (a haiku subagent took 65 s and put one corner 49 px off on the same frame);
- a full 1280x720 request timed out at the provider after ~65 s and kept a $2.81 reservation of the $10
  session budget: keep images at WIDTH, max_tokens <= 2048;
- the host serializes every MCP call, Gemini included: ask between motion calls, never expect it to overlap one.
Points come as [y, x] normalized to 0-1000 (Gemini's native convention) and are converted to full-size pixels."""
import base64
import json
import time
from pathlib import Path

import cv2

from .frames import frame_dir

WIDTH = 640                 # px sent to Gemini (tested 640; larger requests risk expensive timeouts)
MAX_TOKENS = 2048           # covers low-effort reasoning + a JSON answer (tested)
MIN_BUDGET_USD = 0.5        # refuse to call below this remaining budget
PT = {"type": "array", "items": {"type": "integer"},
      "description": "[y, x] normalized to 0-1000 over the image height and width"}

_last = {'remaining': None}
calls = []                  # (seconds, cost, name) of every call in this process, for reports


def image_block(src, width=WIDTH, crop=None, quality=85):
    """(MCP image block, (w, h) of the image the points refer to, (x0, y0) offset). src: frame id / dir or a
    jpg path; crop: (x0, y0, x1, y1) in full-size pixels."""
    p = Path(str(src))
    path = p if p.suffix.lower() in ('.jpg', '.jpeg', '.png') else frame_dir(src) / 'head_camera.jpg'
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    x0 = y0 = 0
    if crop is not None:
        x0, y0, x1, y1 = [int(v) for v in crop]
        im = im[y0:y1, x0:x1]
    h, w = im.shape[:2]
    if w > width:
        im = cv2.resize(im, (width, round(width * h / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    block = {"type": "image", "mimeType": "image/jpeg", "data": base64.b64encode(buf.tobytes()).decode('ascii')}
    return block, (w, h), (x0, y0)


def to_px(pt, size, offset=(0, 0)):
    """[y, x] in 0-1000 -> [u, v] pixels of the full-size frame."""
    return [round(pt[1] / 1000 * size[0] + offset[0]), round(pt[0] / 1000 * size[1] + offset[1])]


def ask(ctx, question, images, schema, name='answer', effort='low', max_tokens=MAX_TOKENS, log=None):
    """One structured question about up to 8 image blocks. -> {ok, answer, seconds, cost_usd, why}.
    Never raises for a provider problem: a timeout / refusal / truncated answer is {ok: False, why}."""
    if _last['remaining'] is not None and _last['remaining'] < MIN_BUDGET_USD:
        return {'ok': False, 'why': f"Gemini budget low ({_last['remaining']:.2f} USD left)"}
    content = [{"type": "text", "text": question}] + list(images)
    t0 = time.time()
    try:
        reply = ctx.call("gemini_generate", messages=[{"role": "user", "content": content}],
                         response_format={"type": "json_schema",
                                          "json_schema": {"name": name, "strict": True, "schema": schema}},
                         reasoning={"effort": effort}, max_tokens=max_tokens)
    except RuntimeError as e:
        return {'ok': False, 'why': f'gemini error: {str(e)[:300]}', 'seconds': round(time.time() - t0, 1)}
    dt = round(time.time() - t0, 1)
    r = json.loads(reply["content"][0]["text"])
    _last['remaining'] = r.get('budget', {}).get('session_cost_remaining_usd')
    calls.append((dt, r.get('cost_usd'), name))
    msg = r.get('message') or {}
    if r.get('finish_reason') != 'stop' or msg.get('refusal') or not msg.get('content'):
        return {'ok': False, 'why': f"gemini finish {r.get('finish_reason')} refusal {msg.get('refusal')}",
                'seconds': dt}
    try:
        ans = json.loads(msg['content'])
    except ValueError:
        return {'ok': False, 'why': f"gemini returned non-JSON {msg['content'][:200]}", 'seconds': dt}
    if log:
        log(f'gemini {name} {dt:.1f}s ${r.get("cost_usd")}: {json.dumps(ans)[:300]}')
    return {'ok': True, 'answer': ans, 'seconds': dt, 'cost_usd': r.get('cost_usd'),
            'remaining_usd': _last['remaining']}


def check(ctx, frames, question, log=None, crop=None):
    """Yes / no question about one or more frames -> {ok, yes, reason, seconds} (ok False = no usable answer)."""
    schema = {"type": "object", "additionalProperties": False, "required": ["answer", "reason"],
              "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
                             "reason": {"type": "string", "description": "one short sentence"}}}
    blocks = [image_block(f, crop=crop)[0] for f in ([frames] if isinstance(frames, (str, Path)) else frames)]
    r = ask(ctx, question + ' Answer yes, no or unsure, with a one-sentence reason.', blocks, schema,
            name='check', log=log)
    if not r['ok']:
        return r
    a = r['answer']
    return {'ok': a['answer'] != 'unsure', 'yes': a['answer'] == 'yes', 'reason': a['reason'],
            'seconds': r['seconds'], 'answer': a['answer']}


def point(ctx, frame, prompt, points, flags=None, lists=None, log=None, crop=None):
    """Ask for named points (and boolean flags / string lists) on one frame.
    points: {name: 'description'} for single points, or {name: ('description', n)} for a list of n points.
    -> {ok, px: {name: [u, v] or [[u, v], ...]}, flags: {...}, lists: {...}, seconds, raw}"""
    props, req = {}, []
    for k, d in points.items():
        if isinstance(d, tuple):
            props[k] = {"type": "array", "items": PT, "description": d[0]}
        else:
            props[k] = dict(PT, description=f'{d}; [y, x] normalized to 0-1000')
        req.append(k)
    for k, d in (flags or {}).items():
        props[k] = {"type": "boolean", "description": d}
        req.append(k)
    for k, d in (lists or {}).items():
        props[k] = {"type": "array", "items": {"type": "string"}, "description": d}
        req.append(k)
    schema = {"type": "object", "additionalProperties": False, "required": req, "properties": props}
    block, size, off = image_block(frame, crop=crop)
    text = prompt + ' Points are [y, x] normalized to 0-1000.'
    r = ask(ctx, text, [block], schema, name='point', log=log)
    if not r['ok']:
        return r
    a = r['answer']
    px = {}
    try:
        for k, d in points.items():
            if isinstance(d, tuple):
                px[k] = [to_px(p, size, off) for p in a[k] if len(p) == 2]
            else:
                px[k] = to_px(a[k], size, off) if len(a[k]) == 2 else None
    except (TypeError, KeyError, IndexError) as e:
        return {'ok': False, 'why': f'malformed points {e}: {a}', 'seconds': r['seconds']}
    return {'ok': True, 'px': px, 'flags': {k: a[k] for k in (flags or {})}, 'lists': {k: a[k] for k in (lists or {})},
            'seconds': r['seconds'], 'raw': a}


def usage():
    """Calls made by this process: count, total seconds, total cost."""
    return {'calls': len(calls), 'seconds': round(sum(c[0] for c in calls), 1),
            'cost_usd': round(sum(c[1] or 0 for c in calls), 4)}
