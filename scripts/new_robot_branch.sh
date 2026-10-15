#!/usr/bin/env bash
# Create the empty library branch for a new robot from templates/robot-branch/ (maintainers).
#
#   scripts/new_robot_branch.sh <branch> "<robot name>" [README.md]
#
# <branch> is the robot's backend key (lower-case letters, digits, hyphens). The optional
# README replaces the template's, e.g. one written from the robot's new-robot issue.
# Builds one orphan commit and points the local branch <branch> at it, without touching
# your working tree; review it with `git show <branch>`, then `git push origin <branch>`.
set -euo pipefail

usage='usage: scripts/new_robot_branch.sh <branch> "<robot name>" [README.md]'
branch=${1:?$usage}
robot=${2:?$usage}
readme=${3:-}

if ! [[ $branch =~ ^[a-z0-9][a-z0-9-]{1,30}$ ]] || [[ $branch == main || $branch == contrib ]]; then
    echo "branch must be the robot's backend key: lower-case letters, digits and hyphens, not main or contrib" >&2
    exit 1
fi
root=$(git rev-parse --show-toplevel)
template=$root/templates/robot-branch
[[ -d $template ]] || { echo "no $template: run this from a checkout of main" >&2; exit 1; }
if git rev-parse -q --verify "refs/heads/$branch" >/dev/null; then
    echo "a local branch $branch already exists" >&2
    exit 1
fi
if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
    echo "origin already has a branch $branch" >&2
    exit 1
fi
if [[ -n $readme && ! -f $readme ]]; then
    echo "no README file $readme" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/tree"
cp -R "$template/." "$tmp/tree/"
if [[ -n $readme ]]; then
    cp "$readme" "$tmp/tree/README.md"
fi
python3 - "$tmp/tree/README.md" "$robot" <<'EOF'
import sys
path, robot = sys.argv[1], sys.argv[2]
text = open(path, encoding='utf-8').read()
open(path, 'w', encoding='utf-8').write(text.replace('{{ROBOT}}', robot))
EOF

export GIT_INDEX_FILE=$tmp/index
git --work-tree="$tmp/tree" add -A
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -m "Start the $robot library: README, empty index, check-skills workflow")
git branch "$branch" "$commit"

python3 "$root/scripts/check_skills.py" "$tmp/tree"
echo "Created $branch at $(git rev-parse --short "$commit"):"
git ls-tree -r --name-only "$branch" | sed 's/^/  /'
echo "Review with: git show $branch"
echo "Publish with: git push origin $branch"
echo "Then add $branch to the table in README.md on main."
