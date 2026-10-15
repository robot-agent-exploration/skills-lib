#!/usr/bin/env python3
"""Check a skill library (one embodiment branch) against the format in CONTRIBUTING.md.

    python3 check_skills.py [LIBRARY]                  every skill in the library
    python3 check_skills.py [LIBRARY] --base origin/rby1
                                                       only what HEAD changed since its merge base
                                                       with that ref, plus the contribution rules

LIBRARY is a checkout of an embodiment branch (default: the current directory). The
check reads files as data: code is compiled, never run. Exit status 1 when anything
breaks the format. Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

STATUSES = ('draft', 'verified', 'deprecated')
SECTIONS = ('When to use', 'Preconditions', 'Procedure', 'Code', 'Checks', 'Failure modes', 'Evidence')
RUN_KEYS = ('session', 'date', 'commit', 'outcome', 'unattended', 'conditions', 'checks', 'images')
VERIFIED_MIN_RUNS = 2       # unattended successful runs of the committed code
MAX_IMAGE_BYTES = 400_000
MAX_FILE_BYTES = 2_000_000
MAX_SKILL_BYTES = 25_000_000
SLUG = re.compile(r'^[a-z0-9][a-z0-9-]{1,60}$')
INDEX_ROW = re.compile(r'^\|\s*\[([^\]]+)\]\(skills/([^/)]+)/SKILL\.md\)\s*\|\s*([^|]*?)\s*\|', re.M)


def frontmatter(text):
    match = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    if not match:
        return None
    fields = {}
    for line in match.group(1).splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            value = re.sub(r'\s+#.*$', '', value)  # a trailing YAML comment
            fields[key.strip()] = value.strip().strip('"\'')
    return fields


def list_field(value):
    """A frontmatter list written inline: `[a, b]` or `a, b`."""
    return [item.strip().strip('"\'') for item in value.strip().strip('[]').split(',') if item.strip()]


def validate_skill(folder):
    """Problems of one skill folder (a list of strings; empty when it follows the format)."""
    folder = Path(folder)
    slug = folder.name
    problems = []

    def bad(message):
        problems.append(f'skills/{slug}: {message}')
    links = [p for p in [folder, *folder.rglob('*')] if p.is_symlink()]
    if links:  # refuse before reading anything through a link
        bad(f'symlinks are not allowed: {[str(p.relative_to(folder.parent)) for p in links][:5]}')
        return problems
    if not SLUG.match(slug):
        bad('folder name must be lower-case letters, digits and hyphens')
    total = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if total > MAX_SKILL_BYTES:
        bad(f'{total / 1e6:.1f} MB is over the {MAX_SKILL_BYTES / 1e6:g} MB per skill')
    for p in folder.rglob('*'):
        if p.is_file() and p.stat().st_size > MAX_FILE_BYTES:
            bad(f'{p.relative_to(folder)} is over {MAX_FILE_BYTES / 1e6:g} MB')
    skill = folder/'SKILL.md'
    if not skill.is_file():
        bad('SKILL.md missing')
        return problems
    text = skill.read_text(errors='replace')
    meta = frontmatter(text)
    if meta is None:
        bad('SKILL.md must begin with --- frontmatter ---')
        return problems
    for key in ('name', 'description', 'status', 'robot', 'entry'):
        if not meta.get(key):
            bad(f'frontmatter needs {key}')
    if meta.get('name') and meta['name'] != slug:
        bad(f"frontmatter name {meta['name']!r} must equal the folder name")
    status = meta.get('status')
    if status and status not in STATUSES:
        bad(f'status must be one of {", ".join(STATUSES)}')
    headings = set(re.findall(r'^##\s+(.+?)\s*$', text, re.M))
    for section in SECTIONS:
        if section not in headings:
            bad(f'SKILL.md needs a "## {section}" section')
    if not re.search(r'^###\s+Perception checkpoints\s*$', text, re.M):
        bad('SKILL.md needs a "### Perception checkpoints" table under "## Checks" (step, subagent, frames, '
            'question, expected answer, pass criterion, on fail; or "None")')
    entry = meta.get('entry')
    if entry:
        code = (folder/entry).resolve()
        if not code.is_relative_to(folder.resolve()) or not code.is_file() or code.suffix != '.py':
            bad(f'entry {entry} must be a .py file inside the skill folder')
        else:
            source = code.read_text(errors='replace')
            try:
                compile(source, entry, 'exec')  # syntax only: nothing runs
            except (SyntaxError, ValueError) as exc:
                bad(f'entry {entry} does not compile: {str(exc)[:200]}')
            if not re.search(r'^def main\(\s*ctx\b', source, re.M):
                bad(f'entry {entry} must define main(ctx, ...)')
    for p in folder.rglob('*.py'):
        if entry and p.resolve() == (folder/entry).resolve():
            continue
        try:
            compile(p.read_text(errors='replace'), str(p.relative_to(folder)), 'exec')
        except (SyntaxError, ValueError) as exc:
            bad(f'{p.relative_to(folder)} does not compile: {str(exc)[:200]}')
    images = folder/'evidence'/'images'
    image_files = [p for p in images.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')] if images.is_dir() else []
    if not image_files:
        bad('evidence/images/ needs at least one image (a visual reference of the start state or result)')
    for p in image_files:
        if p.stat().st_size > MAX_IMAGE_BYTES:
            bad(f'{p.relative_to(folder)} is {p.stat().st_size // 1000} kB (at most {MAX_IMAGE_BYTES // 1000} kB)')
    for target in re.findall(r'!\[[^\]]*\]\(([^)\s]+)\)', text):
        if '://' not in target and not (folder/target).is_file():
            bad(f'SKILL.md shows {target}, which is not a file in the skill')
    runs_file = folder/'evidence'/'runs.jsonl'
    runs = []
    if runs_file.is_file():
        for n, line in enumerate(runs_file.read_text(errors='replace').splitlines(), start=1):
            if not line.strip():
                continue
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                bad(f'evidence/runs.jsonl line {n} is not JSON')
                continue
            missing = [k for k in RUN_KEYS if k not in run]
            if missing:
                bad(f'evidence/runs.jsonl line {n} lacks {missing}')
                continue
            if run['outcome'] not in ('success', 'failure'):
                bad(f'evidence/runs.jsonl line {n}: outcome must be success or failure')
            if not isinstance(run['unattended'], bool):
                bad(f'evidence/runs.jsonl line {n}: unattended must be true or false')
            for image in run['images'] if isinstance(run['images'], list) else []:
                path = (folder/image).resolve()
                if not path.is_relative_to(images.resolve()) or not path.is_file():
                    bad(f'evidence/runs.jsonl line {n}: image {image} is not a file under evidence/images/')
            runs.append(run)
    elif status == 'verified':
        bad('verified needs evidence/runs.jsonl')
    if status == 'verified':
        good = [r for r in runs if r.get('outcome') == 'success' and r.get('unattended') is True and r.get('images')]
        if len(good) < VERIFIED_MIN_RUNS:
            bad(f'verified needs at least {VERIFIED_MIN_RUNS} unattended successful runs with images in '
                f'evidence/runs.jsonl (has {len(good)}); keep it draft until then')
    return problems


def validate_tree(tree, slugs, changed=None, deleted=()):
    """Problems of a library tree for the given skills; `changed` (paths) applies the contribution rules."""
    tree = Path(tree)
    problems = []
    if changed is not None:
        outside = [p for p in changed if not (p == 'INDEX.md' or p.startswith('skills/'))]
        if outside:
            problems.append(f'a contribution may only change INDEX.md and skills/; it changed {sorted(outside)[:10]}')
    for slug in sorted({p.split('/')[1] for p in deleted if p.startswith('skills/') and p.count('/') >= 2}):
        if not (tree/'skills'/slug/'SKILL.md').is_file():
            problems.append(f'skills/{slug} was deleted; set its status to deprecated instead')
    index_path = tree/'INDEX.md'
    if index_path.is_symlink() or not index_path.is_file():
        return problems + ['INDEX.md must be a file at the library root']
    index = index_path.read_text(errors='replace')
    rows = {slug: status.strip('` ') for _, slug, status in INDEX_ROW.findall(index)}
    for slug in slugs:
        folder = tree/'skills'/slug
        if not folder.is_dir():
            continue
        problems += validate_skill(folder)
        meta = frontmatter((folder/'SKILL.md').read_text(errors='replace')) if (folder/'SKILL.md').is_file() else None
        if slug not in rows:
            problems.append(f'INDEX.md needs a row linking skills/{slug}/SKILL.md')
        elif meta and meta.get('status') and rows[slug] != meta['status']:
            problems.append(f'INDEX.md lists skills/{slug} as {rows[slug]!r}, its SKILL.md says {meta["status"]!r}')
        for used in list_field((meta or {}).get('uses', '')):
            if not (tree/'skills'/used/'SKILL.md').is_file():
                problems.append(f'skills/{slug}: uses {used!r}, which is not a skill in the library')
    for slug in rows:
        if not (tree/'skills'/slug/'SKILL.md').is_file():
            problems.append(f'INDEX.md links skills/{slug}/SKILL.md, which does not exist')
    return problems


def git_paths(tree, base, diff_filter):
    out = subprocess.run(['git', '-C', str(tree), 'diff', '--name-only', f'--diff-filter={diff_filter}',
                          f'{base}...HEAD'], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('library', nargs='?', default='.', help='checkout of an embodiment branch')
    parser.add_argument('--base', help='check only what HEAD changed since its merge base with this ref')
    args = parser.parse_args(argv)
    tree = Path(args.library)
    if args.base:
        changed = git_paths(tree, args.base, 'd')
        deleted = git_paths(tree, args.base, 'D')
        slugs = sorted({p.split('/')[1] for p in changed if p.startswith('skills/') and p.count('/') >= 2})
        problems = validate_tree(tree, slugs, changed, deleted)
    else:
        skills = tree/'skills'
        slugs = sorted(p.name for p in skills.iterdir() if p.is_dir()) if skills.is_dir() else []
        problems = validate_tree(tree, slugs)
    for problem in problems:
        print(problem)
    print(f'{len(slugs)} skill(s) checked: ' + (f'{len(problems)} problem(s)' if problems else 'OK'))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
