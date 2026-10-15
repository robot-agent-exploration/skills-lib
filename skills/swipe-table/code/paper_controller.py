"""Autonomous paper-towel wiping: pull sheets out of the tissue box, pile them on the soy-sauce
stain, grasp the pile as a wad, scrub, throw the wad into the waste bin, verify; repeat with fresh
sheets while residue remains.

  1. preconditions -> home; observe: tissue box, a sheet sticking out, waste bin
  2. stain: none -> this frame is the clean reference, ask the operator to pour (re-ask <= 3);
     already there -> --reference frame (or the stored towel-scene reference after a check)
     --dry-region x0,x1,y0,y1: no pour, wipe that area (dry test)
  3. passes (MAX_PASSES): pull SHEETS_PER_PASS sheets, dropping them spread over the region ->
     locate the pile, wad-grasp it -> scrub the region (fitted to safe reach) -> dispose in bin ->
     sweep leftover sheets into the bin -> home -> verify (exterior + wrist residue vs reference);
     next pass on the residue bbox
  4. home, gripper open, report

Second entry of the swipe-table skill (the towel controller is controller.py; this module reuses
its steps / motions / perception). Library entry: main(ctx, ...) -> {ok, measured, why}
(wraps the session's controller, run_logged). Run from the workspace root (run_in_background):
  python <skill>/code/paper_controller.py [--reference <frame>] [--dry-region x0,x1,y0,y1] [--sheets N]
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # sibling modules: motions, perception, steps, tissues

import numpy as np

import motions as M
import perception as P
import steps as S
import tissues as T

from swipe_table_params import (  # hyper-parameters: swipe_table_params.py
    SHEETS_PER_PASS, PAPER_CONTROLLER_MAX_PASSES as MAX_PASSES, STAIN_MARGIN, RESIDUE_MARGIN,
    VERIFY_ROI_MARGIN, CLEAN_AREA_M2, CLEAN_WRIST_M2, STAIN_PRESENT_M2, STAIN_PRESENT_NOREF_M2, STAIN_ROI,
    POUR_ASKS, BOX_MOVED_M, LEFTOVER_RISE_M, LEFTOVER_MIN_M2, MAX_LEFTOVER_PICKS, STORED_REFERENCE,
)

POUR_MESSAGE = (
    "Ready for the soy sauce. Please pour it onto the table somewhere 40-65 cm in front of the robot "
    "base and within about 25 cm of its centre line, at least 15 cm away from the tissue box and the bin. "
    "Step back and reply when done. The arm stays still until you reply.")


def expand(b, m):
    return (b[0] - m, b[1] + m, b[2] - m, b[3] + m)


def scene_mask(box, bin_info, margin=0.05):
    """Map cells of the tissue box and the bin (+margin): never stain, never residue."""
    m = np.zeros(P.grid()[0].shape, bool)
    if box and box.get('found'):
        m |= P.region_mask(box['x'][0] - margin, box['x'][1] + margin, box['y'][0] - margin - 0.05, box['y'][1] + margin)
    if bin_info and bin_info.get('found'):           # +8 cm on the far side: bag flap / shadow
        m |= P.region_mask(bin_info['rim_x'][0] - margin, bin_info['rim_x'][1] + margin + 0.08,
                           bin_info['rim_y'][0] - margin, bin_info['rim_y'][1] + margin)
    return m


def drop_points(region, n):
    """Spread n drop points over the region: centre first, then the four quarter points."""
    x0, x1, y0, y1 = region
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    qx, qy = (x1 - x0) / 4, (y1 - y0) / 4
    pts = [(cx, cy), (cx - qx, cy - qy), (cx + qx, cy + qy), (cx - qx, cy + qy), (cx + qx, cy - qy)]
    return [pts[i % len(pts)] for i in range(n)]


def pull_one(ctx, log, box0, bin_box, tries=3):
    """Pull one sheet out of the box (re-localized every time). Ends holding it at PULL_Z."""
    for attempt in range(1, tries + 1):
        frame = S.observe(ctx, log, f'sheet (try {attempt})')
        box = T.tissue_box(frame)
        sheet = T.protruding_sheet(frame, box, exclude_box=bin_box) if box.get('found') else {'found': False}
        if not sheet.get('found'):
            S.go_home(ctx, log)                      # the arm may hide the box: look from home
            frame = S.observe(ctx, log, 'sheet from home')
            box = T.tissue_box(frame)
            sheet = T.protruding_sheet(frame, box, exclude_box=bin_box) if box.get('found') else {'found': False}
        if not box.get('found'):
            raise S.StepError('tissue box not found')
        # compare outline centres: the median of purple points shifts ~3 cm when sheets cover the top
        bc = lambda b: ((b['x'][0] + b['x'][1]) / 2, (b['y'][0] + b['y'][1]) / 2)
        moved = np.hypot(bc(box)[0] - bc(box0)[0], bc(box)[1] - bc(box0)[1])
        if moved > BOX_MOVED_M:
            raise S.StepError(f'tissue box moved {moved * 100:.1f} cm (was it lifted with a sheet?)')
        if not sheet.get('found'):
            raise S.StepError('no sheet sticking out of the tissue box')
        tgt = T.grasp_point(box, sheet)
        log(f'sheet top {sheet["top_z"]:.3f} -> pinch at {np.round(tgt, 3).tolist()}')
        S.check_reachable(ctx, [tgt], log)
        g = T.pull_sheet(ctx, tgt, log)
        g2 = T.gripper_after(ctx, 1.0)
        if g2['is_grasped']:                      # judged after the lift: a held sheet is kept
            return True
        log(f'pinch missed (grasped at pinch {g["is_grasped"]}, after lift {g2["is_grasped"]}); retry')
        ctx.value('franka_gripper', action='open')
    raise S.StepError(f'could not pull a sheet in {tries} tries')


def sweep_leftovers(ctx, log, ref, keep_out, bin0):
    """Pick every sheet left on the table (the wad grasp leaves some; scrubbing drags them) and
    bin it. Looks from home each time. Returns (picked, given_up) lists of positions."""
    X, Y = P.grid()
    # 10 cm: paper hanging over the bin edge 6.6 cm out was once picked as a 'leftover' (wet run 2)
    near_bin = P.region_mask(bin0['rim_x'][0] - 0.10, bin0['rim_x'][1] + 0.14, bin0['rim_y'][0] - 0.10, bin0['rim_y'][1] + 0.10)
    exclude = keep_out | near_bin                    # paper hanging over the bin's edge is 'in the bin'
    picked, given_up, fails = [], [], {}
    for it in range(MAX_LEFTOVER_PICKS + 4):
        S.go_home(ctx, log)
        frame = S.observe(ctx, log, f'leftovers ({it})')
        blobs = [b for b in T.leftover_paper(frame, ref, exclude)
                 if b['rise'] >= LEFTOVER_RISE_M and b['area_m2'] >= LEFTOVER_MIN_M2]
        key = lambda b: (round(b['high'][0] / 0.03), round(b['high'][1] / 0.03))
        blobs = [b for b in blobs if fails.get(key(b), 0) < 2]
        log(f'leftover paper: {[(round(b["area_m2"] * 1e4), np.round(b["high"], 3).tolist()) for b in blobs]}')
        if not blobs or len(picked) >= MAX_LEFTOVER_PICKS:
            break
        b = blobs[0]
        x, y = b['high']
        ok, _ = S.pose_ok(ctx, (x, y, 0.105))
        if not ok:
            log(f'leftover at {np.round(b["high"], 3).tolist()} not safely reachable: left for the operator')
            fails[key(b)] = 99
            given_up.append(b['high'])
            continue
        if T.pick_paper(ctx, x, y, log):
            p = ctx.value('franka_status')['tcp_position']
            if p[0] > 0.60:                          # carry height is out of reach far out
                M.move(ctx, [0.58, p[1], 0.38], log=log)
            T.dispose(ctx, bin0, log, recentre=S.recentre)
            picked.append(b['high'])
        else:
            fails[key(b)] = fails.get(key(b), 0) + 1
            if fails[key(b)] >= 2:
                given_up.append(b['high'])
    return picked, given_up


def run(ctx, log, reference=None, dry_region=None, sheets=SHEETS_PER_PASS, max_passes=MAX_PASSES):
    report = {'ok': False, 'passes': [], 'frames': {}}
    t0 = time.time()
    if ctx.value('status').get('control_uncertain'):
        raise S.StepError('control_uncertain; refusing to move')
    rs = ctx.value('franka_status')
    if not rs['motion']['armed']:
        raise S.StepError('motion is not armed')
    if rs['gripper'].get('is_grasped'):
        raise S.StepError('gripper already holds something')
    ctx.value('franka_gripper', action='open')
    S.go_home(ctx, log)

    frame = S.observe(ctx, log, 'start')
    report['frames']['start'] = frame['id']
    box0 = T.tissue_box(frame)
    bin0 = T.bin_opening(frame)
    if not box0.get('found'):
        raise S.StepError('tissue box not found')
    if not bin0.get('found'):
        raise S.StepError('waste bin opening not found')
    log(f"box top {box0['top_z']:.3f} x {np.round(box0['x'], 3).tolist()} y {np.round(box0['y'], 3).tolist()}; "
        f"bin opening {np.round(bin0['centre'], 3).tolist()} rim z {bin0['rim_top_z']:.3f}")
    keep_out = scene_mask(box0, bin0)
    bin_box = (bin0['rim_x'][0] - 0.03, bin0['rim_x'][1] + 0.10, bin0['rim_y'][0] - 0.03, bin0['rim_y'][1] + 0.03)

    if dry_region is not None:
        ref = frame
        region = tuple(dry_region)
        centre = ((region[0] + region[1]) / 2, (region[2] + region[3]) / 2)
        log(f'DRY TEST on region {np.round(region, 3).tolist()}')
    else:
        stored = None
        ref_id = reference or STORED_REFERENCE
        if (P.FRAMES/ref_id).is_dir():
            stored = P.load_frame(ref_id)
            ok, stats = P.reference_matches(frame, stored, exclude=keep_out)
            log(f'stored reference {ref_id}: {"usable" if ok else "NOT usable"} {stats}')
            stored = stored if ok else None
        stain = P.locate_stain(frame, stored, exclude=keep_out, roi=STAIN_ROI)
        present = STAIN_PRESENT_M2 if stored is not None else STAIN_PRESENT_NOREF_M2
        log(f'stain at start: {stain["found"]} {stain.get("area_m2", 0) * 1e4:.1f} cm2 '
            f'({"vs reference" if stored is not None else "no reference"})')
        if stain['found'] and stain['area_m2'] > present:
            ref = stored
        else:
            ref = frame
            message = POUR_MESSAGE
            for ask in range(1, POUR_ASKS + 1):
                reply = ctx.value('request_operator', kind='reset', message=message, timeout_s=900)
                log(f'operator (pour, ask {ask}): {reply.get("status")} {reply.get("reply")!r}')
                if reply.get('status') != 'answered':
                    raise S.StepError(f'operator did not confirm the pour: {reply}')
                time.sleep(2.0)
                frame = S.observe(ctx, log, f'after pour (ask {ask})')
                report['frames']['after_pour'] = frame['id']
                stain = P.locate_stain(frame, ref, exclude=keep_out, roi=STAIN_ROI)
                if stain['found']:
                    log(f"stain: centre {np.round(stain['centre'], 3).tolist()} bbox {np.round(stain['bbox'], 3).tolist()} "
                        f"area {stain['area_m2'] * 1e4:.1f} cm2")
                    break
                message = 'I cannot see a soy-sauce stain on the table yet. ' + POUR_MESSAGE
        if not stain['found']:
            raise S.StepError('no stain found')
        report['stain'] = {'centre': stain['centre'], 'bbox': stain['bbox'], 'area_m2': stain['area_m2']}
        region = expand(stain['bbox'], STAIN_MARGIN)
        centre = stain['centre']
    if ref is not None:
        report['frames']['reference'] = ref['id']

    wiped = region
    for n in range(1, max_passes + 1):
        fitted = S.fit_region(ctx, region, log)
        log(f'=== pass {n}: region {np.round(region, 3).tolist()} -> fitted {np.round(fitted, 3).tolist()}')
        # 1. pile fresh sheets on the stain
        for i, xy in enumerate(drop_points(fitted, sheets), 1):
            pull_one(ctx, log, box0, bin_box)
            T.drop_at(ctx, xy, log)
            log(f'sheet {i}/{sheets} dropped at {np.round(xy, 3).tolist()}')
        # 2. grasp the pile as a wad
        frame = S.observe(ctx, log, 'pile')
        pile = T.paper_pile(frame, fitted)
        if not pile.get('found'):
            raise S.StepError(f'paper pile not found on the region ({pile})')
        px, py = pile['upper']
        top = pile['top_z']
        log(f'pile upper {np.round(pile["upper"], 3).tolist()} top z {top:.3f}')
        g = S._press_grasp(ctx, px, py, max(top + 0.06, 0.20), M.Z_FLOOR, 20.0, 40.0, log)
        if not g['is_grasped']:
            raise S.StepError('could not grasp the paper pile')
        # 3. scrub (the wad stays down on the stain)
        w = S.wipe_region(ctx, fitted, log, mode='drag')
        # 4. throw it away, then pick up any sheets left on the table
        T.dispose(ctx, bin0, log, recentre=S.recentre)
        picked, given_up = sweep_leftovers(ctx, log, ref, keep_out, bin0)
        log(f'leftovers picked {len(picked)}, given up {len(given_up)}')
        S.go_home(ctx, log)
        # 5. verify
        frame = S.observe(ctx, log, f'verify pass {n}')
        roi = expand(wiped, VERIFY_ROI_MARGIN)
        roi = (max(roi[0], 0.33), min(roi[1], 0.80), max(roi[2], -0.42), min(roi[3], 0.42))
        res = S.measure_residue(frame, ref, roi, log, towel_masks=[keep_out], wiped=wiped)
        wr = res.get('wrist') or {}
        clean = res['area_m2'] < CLEAN_AREA_M2 and ((not wr) or wr.get('error') or wr.get('area_m2', 0) < CLEAN_WRIST_M2)
        report['passes'].append({'pass': n, 'region': w['region'], 'frame': frame['id'],
                                 'leftovers_picked': len(picked), 'leftovers_given_up': given_up,
                                 'residue_m2': res['area_m2'], 'wrist_residue_m2': wr.get('area_m2'),
                                 'seconds': round(time.time() - t0)})
        if clean:
            report['ok'] = True
            log(f'clean after pass {n}')
            break
        log(f'NOT clean after pass {n}: exterior {res["area_m2"] * 1e4:.1f} cm2 wrist {wr.get("area_m2", 0) * 1e4:.2f} cm2')
        if res.get('bbox_all'):
            region = expand(res['bbox_all'], RESIDUE_MARGIN)
            wiped = (min(wiped[0], region[0]), max(wiped[1], region[1]), min(wiped[2], region[2]), max(wiped[3], region[3]))
    report['seconds'] = round(time.time() - t0)
    if not report['ok']:
        report['reason'] = f'residue remains after {len(report["passes"])} passes'
    return report


def run_logged(ctx, **kw):
    """The session's controller main (renamed on import): run() with a log file; returns its report."""
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = Path('output')/'runs'/f'paper_{stamp}'     # relative to the working directory (workspace root)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out/'log.txt', 'w')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    try:
        report = run(ctx, log, **kw)
    except Exception as exc:
        log(f'STOPPED: {type(exc).__name__}: {exc}')
        log(traceback.format_exc())
        report = {'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}
        try:
            ctx.value('franka_hold')
            st = ctx.value('franka_status')
            report['robot_at_stop'] = {'tcp': st['tcp_position'], 'gripper': st['gripper']}
        except Exception as exc2:
            log(f'hold/status after stop failed: {exc2}')
    report['log'] = str(out/'log.txt')
    (out/'report.json').write_text(json.dumps(report, indent=1, default=str))
    log(f'report: {json.dumps(report, default=str)}')
    return report


def main(ctx, reference=None, dry_region=None, sheets=SHEETS_PER_PASS, max_passes=MAX_PASSES):
    """Library entry (wrapper added on import; the procedure itself is run_logged, unchanged).

    reference: clean-table frame id (arm home) to use when a stain is already present.
    dry_region: (x0, x1, y0, y1) dry test on this area, no pour and no stain detection.
    sheets: sheets pulled and piled per pass; max_passes: pile/scrub/dispose/verify passes.
    Returns {'ok', 'measured': the controller's report without ok/reason, 'why'}."""
    report = run_logged(ctx, reference=reference, dry_region=dry_region, sheets=sheets, max_passes=max_passes)
    ok = bool(report.get('ok'))
    why = f'clean after pass {len(report.get("passes", []))}' if ok else report.get('reason', 'not clean')
    return {'ok': ok, 'measured': {k: v for k, v in report.items() if k not in ('ok', 'reason')}, 'why': why}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference')
    parser.add_argument('--dry-region', help='x0,x1,y0,y1: dry test on this area, no pour')
    parser.add_argument('--sheets', type=int, default=SHEETS_PER_PASS)
    parser.add_argument('--max-passes', type=int, default=MAX_PASSES)
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())             # the workspace root holds api/runtime.py
    from api.runtime import Context
    dry = [float(v) for v in args.dry_region.split(',')] if args.dry_region else None
    with Context() as session:
        result = main(session, reference=args.reference, dry_region=dry, sheets=args.sheets, max_passes=args.max_passes)
    shown = dict(result, measured={k: v for k, v in result['measured'].items() if k != 'passes'})
    print(json.dumps(shown, indent=1, default=str))
