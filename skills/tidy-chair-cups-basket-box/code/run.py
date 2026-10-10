"""Entry of tidy-chair-cups-basket-box: the task-level steps that are not a library skill of their own. The subtasks
themselves run the library skills step by step (see SKILL.md, Procedure); this entry adds:

    main(ctx, 'sweep_check', moves=[['turn', -1.5]], frames=[...], obstacles=[['cabinet', [[x, y], ...]]], extra=None)
    main(ctx, 'chair_route', moves=[['pull', 0.7], ['turn', -1.5708], ['drive', 1.42], ['turn', 1.5708]])
    main(ctx, 'drop', side='right', target=[0.62, -0.05, 0.62], hold=0.20, pitch=-0.7, release=True)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(ctx, step, log=print, **args):
    """-> {ok, measured, why}"""
    if step == 'sweep_check':
        import numpy as np
        from shapely.geometry import Polygon, box
        import scene_map as M
        import sweep_check as SC
        od = ctx.value('rby1_status')['odometry']
        pose = tuple(args.get('pose') or (od['x'], od['y'], od['yaw']))
        moves = [tuple(m) for m in args['moves']]
        pts = None
        if args.get('frames'):      # frames taken at the CURRENT pose unless frame_odom gives their poses
            fo = args.get('frame_odom') or {}
            pts = np.vstack([M.frame_obstacles(f, fo.get(f, od), zlim=(0.05, 1.2)) for f in args['frames']])[:, :2]
        obst = [(n, Polygon(p)) for n, p in args.get('obstacles', [])]
        extra = [box(*args['extra'])] if args.get('extra') else None
        r = SC.check_moves(pose, moves, obstacles=obst, points=pts, margin=args.get('margin', 0.06), extra=extra)
        end = SC.poses_along(pose, moves)[-1]
        return {'ok': r['ok'], 'why': '' if r['ok'] else f'swept footprint: {r}',
                'measured': {**r, 'end_pose': [round(v, 4) for v in end]}}
    if step == 'chair_route':
        import chair_route
        r = chair_route.run(ctx, args['moves'], log=log)
        return {'ok': r['ok'], 'why': r.get('why', ''), 'measured': {'odometry': r.get('odometry')}}
    if step == 'drop':
        import drop
        return drop.drop(ctx, args['side'], args['target'], hold=args.get('hold', 0.20),
                         pitch=args.get('pitch', -0.7), release=args.get('release', True), log=log)
    return {'ok': False, 'why': f'unknown step {step}', 'measured': {}}


if __name__ == '__main__':
    import json
    sys.path.insert(0, str(HERE.parents[3]))
    from api.runtime import Context
    with Context() as ctx:
        print(json.dumps(main(ctx, sys.argv[1], **(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})),
                         default=str))
