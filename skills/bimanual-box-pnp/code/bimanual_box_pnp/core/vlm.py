"""Gemini perception on saved head-camera frames, through the harness gemini skill's shared module
(`.agents/skills/gemini/scripts/vlm.py`: checks, points, masks, measured settings and the [y, x] polygon fix).

This file only resolves this package's frame ids (`runtime/frames/<id>`) to the head-camera image and keeps the
call names the perception code uses. Change Gemini behaviour in the shared module, not here."""
import importlib.util
from pathlib import Path

from .frames import frame_dir

CAMERA_IMAGE = 'head_camera.jpg'


def _load_shared():
    for root in Path(__file__).resolve().parents:
        path = root / '.agents' / 'skills' / 'gemini' / 'scripts' / 'vlm.py'
        if path.is_file():
            spec = importlib.util.spec_from_file_location('gemini_vlm', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError('the gemini skill helper .agents/skills/gemini/scripts/vlm.py is not in this workspace; '
                      'it needs a session with Gemini enabled')


_shared = _load_shared()
ask, to_px, usage, calls = _shared.ask, _shared.to_px, _shared.usage, _shared.calls
SEG_PROMPT, WIDTH, MAX_TOKENS, PT = _shared.SEG_PROMPT, _shared.WIDTH, _shared.MAX_TOKENS, _shared.PT


def image(src):
    """A frame id / frame directory -> its head-camera image path; image files and arrays pass through."""
    if not isinstance(src, (str, Path)):
        return src
    path = Path(str(src))
    if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        return path
    return frame_dir(src) / CAMERA_IMAGE


def image_block(src, **kwargs):
    return _shared.image_block(image(src), **kwargs)


def check(ctx, frames, question, log=None, crop=None):
    frames = frames if isinstance(frames, (list, tuple)) else [frames]
    return _shared.check(ctx, [image(f) for f in frames], question, log=log, crop=crop)


def point(ctx, frame, prompt, points, flags=None, lists=None, log=None, crop=None):
    return _shared.point(ctx, image(frame), prompt, points, flags=flags, lists=lists, log=log, crop=crop)


def segment(ctx, frame, what, effort='minimal', log=None):
    return _shared.segment(ctx, image(frame), what, effort=effort, log=log)
