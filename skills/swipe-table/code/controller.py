"""Autonomous soy-sauce wiping controller (task swipe_table).

Pipeline (each step re-observes; stops with a report on anything it was not built for):
  1. preconditions: motion armed, no uncertain control, gripper empty -> home
  2. observe: towel on the table; is there already a stain?
     - no stain: this frame is the clean reference; ask the operator to pour soy sauce
     - stain present: the --reference / stored clean frame if it matches the live view
       (perception.reference_matches), else no reference (local table colour)
  3. localize the stain (cells darker or yellower than the clean table in both exterior cameras)
  4. pass loop (MAX_PASSES):
       grasp towel (wad; flat towel -> pinch, heap, re-grasp) -> wipe the region
       ('carry' on pass 1, 'drag' after soap so the soapy underside stays down)
       -> park the towel where it was picked up -> home -> measure residue
       residue small -> done; else re-wipe the residue bbox + margin (pass 2 as is;
       before pass 3 the operator adds water + soap, then the towel is dragged soapy side down)
  5. home, gripper open, report

Library entry: main(ctx, ...) -> {ok, measured, why} (wraps the session's controller, run_logged).
Run from the workspace root (logs go to output/runs/<stamp>/, frames are read from runtime/frames/):
  python <skill>/code/controller.py [--no-pour-request] [--max-passes N] [--reference <frame>]
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import perception as P
import steps as S

from swipe_table_params import (  # hyper-parameters: swipe_table_params.py
    MAX_PASSES, STAIN_MARGIN, RESIDUE_MARGIN, VERIFY_ROI_MARGIN, CLEAN_AREA_M2, CLEAN_AREA_NOREF_M2,
    CLEAN_WRIST_M2, SOAP_BEFORE_PASS, STAIN_PRESENT_M2, POUR_ASKS, STORED_REFERENCE, TOWEL_SPOT_MIN_DIST,
)

POUR_MESSAGE = (
    "Ready for the soy sauce. Please pour about a tablespoon of soy sauce onto the table "
    "(a puddle roughly 5-10 cm across), somewhere 40-65 cm in front of the robot base and "
    "within about 25 cm of its centre line, at least 15 cm away from the towel, and do not "
    "move the towel. Step back and reply when done. The arm stays still until you reply.")
SOAP_MESSAGE = (
    "A thin soy-sauce film is still on the table after wiping. Please add more water and a "
    "little soap to the UNDERSIDE of the towel and put it back at about the same spot as a "
    "loose heap with the soapy side down. Please do not touch the stain area. Reply when done; "
    "the arm stays still at home until you reply.")


def expand(b, m):
    return (b[0] - m, b[1] + m, b[2] - m, b[3] + m)


def union(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def run(ctx, log, pour_request=True, max_passes=MAX_PASSES, resume=None, reference=None):
    """resume: {'reference': frame_id, 'region': (x0, x1, y0, y1), 'mode': 'carry'|'drag'} skips the
    scene check / stain localization and starts the pass loop on that region."""
    report = {'ok': False, 'passes': [], 'frames': {}}
    t0 = time.time()

    # 1. preconditions
    st = ctx.value('status')
    if st.get('control_uncertain'):
        raise S.StepError('status reports control_uncertain; refusing to move')
    rs = ctx.value('franka_status')
    if not rs['motion']['armed']:
        raise S.StepError('motion is not armed by the operator')
    if rs['gripper'].get('is_grasped'):
        raise S.StepError('gripper already holds something; expected an empty gripper')
    ctx.value('franka_gripper', action='open')
    S.go_home(ctx, log)

    if resume:
        ref = P.load_frame(resume['reference'])
        report['frames']['reference'] = ref['id']
        region = tuple(resume['region'])
        log(f'resume: reference {ref["id"]} region {np.round(region, 3).tolist()} mode {resume.get("mode", "carry")}')
        return passes(ctx, log, report, ref, region, resume.get('mode', 'carry'),
                      ((region[0] + region[1]) / 2, (region[2] + region[3]) / 2), max_passes, t0)

    # 2. scene check
    frame = S.observe(ctx, log, 'start')
    report['frames']['start'] = frame['id']
    S.find_towel(frame, log)
    stored = None
    ref_id = reference or STORED_REFERENCE      # e.g. the clean frame of a run that failed before wiping
    if (P.FRAMES/ref_id).is_dir():
        stored = P.load_frame(ref_id)
        ok, stats = P.reference_matches(frame, stored, exclude=P.towel_mask(frame) | P.towel_mask(stored))
        log(f'stored clean reference {ref_id}: {"usable" if ok else "NOT usable"} {stats}')
        if not ok:
            stored = None
    masks = [P.towel_mask(frame)] + ([P.towel_mask(stored)] if stored is not None else [])
    stain = S.find_stain(frame, stored, log, towel_masks=masks)
    if stain['found'] and stain['area_m2'] > STAIN_PRESENT_M2:
        ref = stored
        log('stain already present at start: ' + ('using the stored clean reference' if ref is not None
                                                   else 'working without a clean reference'))
    else:
        ref = frame
        report['frames']['reference'] = ref['id']
        if not pour_request:
            raise S.StepError('no stain on the table and --no-pour-request given')
        message = POUR_MESSAGE
        for ask in range(1, POUR_ASKS + 1):
            reply = ctx.value('request_operator', kind='reset', message=message, timeout_s=900)
            log(f'operator (pour, ask {ask}): {reply.get("status")} {reply.get("reply")!r}')
            if reply.get('status') != 'answered':
                raise S.StepError(f'operator did not confirm the pour: {reply}')
            time.sleep(2.0)
            frame = S.observe(ctx, log, f'after pour (ask {ask})')
            report['frames']['after_pour'] = frame['id']
            masks = [P.towel_mask(frame), P.towel_mask(ref)]
            if S.find_stain(frame, ref, log, towel_masks=masks)['found']:
                break
            message = ('I cannot see a soy-sauce stain on the table yet. ' + POUR_MESSAGE +
                       f' (Your last reply was: {str(reply.get("reply"))[:200]!r})')
    if ref is not None:
        report['frames']['reference'] = ref['id']

    # 3. localize the stain
    towel_masks = [P.towel_mask(frame)] + ([P.towel_mask(ref)] if ref is not None else [])
    stain = S.find_stain(frame, ref, log, towel_masks=towel_masks)
    if not stain['found']:
        raise S.StepError('no stain found on the table')
    report['stain'] = {'centre': stain['centre'], 'bbox': stain['bbox'], 'area_m2': stain['area_m2']}
    region = expand(stain['bbox'], STAIN_MARGIN)
    return passes(ctx, log, report, ref, region, 'carry', stain['centre'], max_passes, t0)


def passes(ctx, log, report, ref, region, mode, centre, max_passes, t0):
    """4. wipe / park / verify passes until clean, asking for water + soap between passes."""
    wiped = region
    for n in range(1, max_passes + 1):
        log(f'=== pass {n} mode {mode} region {np.round(region, 3).tolist()}')
        g = S.grasp_towel(ctx, log, lift_z=S.SAFE_Z if mode == 'carry' else 0.0)
        w = S.wipe_region(ctx, region, log, mode=mode,
                          centre=centre if n == 1 else None)
        park = S.choose_park(g['pickup'], wiped)
        log(f'park towel at {np.round(park, 3).tolist()} (picked up at {np.round(g["pickup"], 3).tolist()})')
        S.park_towel(ctx, park, log)
        S.go_home(ctx, log)
        frame = S.observe(ctx, log, f'verify pass {n}')
        roi = expand(wiped, VERIFY_ROI_MARGIN)
        roi = (max(roi[0], 0.33), min(roi[1], 0.80), max(roi[2], -0.42), min(roi[3], 0.42))
        masks = [P.towel_mask(frame)] + ([P.towel_mask(ref)] if ref is not None else [])
        res = S.measure_residue(frame, ref, roi, log, towel_masks=masks, wiped=wiped)
        limit = CLEAN_AREA_M2 if ref is not None else CLEAN_AREA_NOREF_M2
        wr = res.get('wrist') or {}
        wrist_clean = (not wr) or wr.get('error') or wr.get('area_m2', 0.0) < CLEAN_WRIST_M2
        entry = {'pass': n, 'mode': mode, 'region': w['region'], 'grasp_attempts': g['attempts'],
                 'frame': frame['id'], 'residue_m2': res['area_m2'], 'residue_bbox': res.get('bbox'),
                 'wrist_residue_m2': wr.get('area_m2'), 'wrist_bbox': wr.get('bbox'),
                 'seconds': round(time.time() - t0)}
        report['passes'].append(entry)
        if res.get('error'):
            raise S.StepError(f'cannot verify: {res["error"]}')
        if res['area_m2'] < limit and wrist_clean:
            report['ok'] = True
            log(f'clean after pass {n}: exterior {res["area_m2"] * 1e4:.1f} cm2 < {limit * 1e4:.0f}, '
                f'wrist {wr.get("area_m2", 0) * 1e4 if wr else float("nan"):.2f} cm2 < {CLEAN_WRIST_M2 * 1e4:.0f}')
            break
        log(f'NOT clean after pass {n}: exterior {res["area_m2"] * 1e4:.1f} cm2, wrist '
            f'{wr.get("area_m2", 0) * 1e4 if wr else float("nan"):.2f} cm2')
        if n == max_passes:
            break
        if res.get('bbox_all'):
            region = expand(res['bbox_all'], RESIDUE_MARGIN)    # wipe where residue is (trial 5)
        wiped = union(wiped, region)
        if n + 1 >= SOAP_BEFORE_PASS:
            reply = ctx.value('request_operator', kind='help', message=SOAP_MESSAGE, timeout_s=900)
            log(f'operator (soap): {reply.get("status")} {reply.get("reply")!r}')
            if reply.get('status') != 'answered':
                raise S.StepError(f'operator did not confirm soap/water: {reply}')
            mode = 'drag'          # soap is on the towel's underside: keep it down
        else:
            mode = 'carry'         # plain re-wipe with the same towel (trial 3b: enough for thin streaks)

    report['seconds'] = round(time.time() - t0)
    if not report['ok']:
        report['reason'] = f'residue remains after {len(report["passes"])} passes'
    return report


def run_logged(ctx, pour_request=True, max_passes=MAX_PASSES, resume=None, reference=None):
    """The session's controller main (renamed on import): runs run() with a log file and returns
    its report dict {ok, passes, frames, stain, seconds, reason?, robot_at_stop?, log}."""
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = Path('output')/'runs'/stamp          # relative to the working directory (workspace root)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out/'log.txt', 'w')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    try:
        report = run(ctx, log, pour_request=pour_request, max_passes=max_passes, resume=resume, reference=reference)
    except Exception as exc:  # stop safely and report
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


def main(ctx, pour_request=True, max_passes=MAX_PASSES, resume=None, reference=None):
    """Library entry (wrapper added on import; the wiping itself is run_logged, unchanged).

    pour_request: ask the operator to pour when the table is clean at the start.
    max_passes: wipe/verify passes (soap is requested before pass SOAP_BEFORE_PASS).
    resume: {'reference': clean frame id, 'region': (x0, x1, y0, y1), 'mode': 'carry'|'drag'}
            skips the scene check and stain localization (agent-supplied wipe region).
    reference: clean-table frame id (arm home) to use when a stain is already present.
    Returns {'ok', 'measured': the controller's report without ok/reason, 'why'}."""
    report = run_logged(ctx, pour_request=pour_request, max_passes=max_passes, resume=resume, reference=reference)
    ok = bool(report.get('ok'))
    why = f'clean after pass {len(report.get("passes", []))}' if ok else report.get('reason', 'not clean')
    return {'ok': ok, 'measured': {k: v for k, v in report.items() if k not in ('ok', 'reason')}, 'why': why}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-pour-request', action='store_true')
    parser.add_argument('--max-passes', type=int, default=MAX_PASSES)
    parser.add_argument('--reference', help='clean-table frame id (arm home) to use if a stain is already present')
    parser.add_argument('--resume-reference', help='clean-table frame id; with --resume-region skips to the passes')
    parser.add_argument('--resume-region', help='x0,x1,y0,y1 gripper-path region to wipe')
    parser.add_argument('--resume-mode', default='carry', choices=['carry', 'drag'])
    args = parser.parse_args()
    resume = None
    if args.resume_reference and args.resume_region:
        resume = {'reference': args.resume_reference, 'mode': args.resume_mode,
                  'region': [float(v) for v in args.resume_region.split(',')]}
    sys.path.insert(0, os.getcwd())             # the workspace root holds api/runtime.py
    from api.runtime import Context
    with Context() as session:
        result = main(session, pour_request=not args.no_pour_request, max_passes=args.max_passes, resume=resume,
                      reference=args.reference)
    shown = dict(result, measured={k: v for k, v in result['measured'].items() if k != 'passes'})
    print(json.dumps(shown, indent=1, default=str))
