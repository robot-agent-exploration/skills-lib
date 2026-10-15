# Adding a new robot

Each robot embodiment has its own library branch (`franka`, `rby1`, ...). A pull request
can only target a branch that already exists, and contributors open pull requests from
their fork, so a new robot gets its branch in three steps: an issue, a branch a
maintainer creates, and a pull request with the first skills.

## When a robot needs its own branch

A branch holds skills whose code runs unchanged on every robot of that branch: the same
tools, kinematics and end effector. Open a new branch when that is not true: another arm or
base, another gripper or hand, or another harness interface. The same robot in another lab
or with its cameras placed differently stays on its branch: say so in the skills'
`## When to use` and in the `conditions` of their runs.

## 1. Open an issue

Open an issue with the **New robot** template. It asks for:

- the branch name: the robot's backend key in the harness, lower-case letters, digits and
  hyphens (`franka`, `rby1`, `ur5e-robotiq`);
- the robot: arms and their DoF, grippers or hands, mobile base, cameras and where they are
  mounted, the base frame the code uses;
- how the code reaches the robot: the harness and its context object (for RAX,
  `api.runtime.Context`), the tools the skills call, and a link to their documentation;
- the setup the harness reports, which skills read instead of copying it: camera
  parameters, named postures, limits and workspace
  ([CONTRIBUTING.md](CONTRIBUTING.md#3-make-skills-modular-so-new-tasks-can-compose-them)
  section 3);
- other dependencies of the skills, and the first skills you plan to contribute.

## 2. A maintainer creates the branch

From a checkout of `main`:

```bash
scripts/new_robot_branch.sh <branch> "<robot name>" [README.md]
git show <branch>                 # review
git push origin <branch>
```

The script builds the branch as one orphan commit from
[`templates/robot-branch/`](templates/robot-branch/): a `README.md` describing the robot
(the template's, with the name filled in, or the README given, written from the issue),
an `INDEX.md` with the table header, an empty `skills/`, and the
[`check-skills` workflow](templates/robot-branch/.github/workflows/check-skills.yml) that
runs the checker on every push and pull request. It refuses a name that is taken locally or
on `origin`, and it leaves your working tree alone.

Then add the branch to the table in [README.md](README.md) on `main` and answer the issue
with the branch's link.

## 3. Contribute the first skills

Open a pull request into the new branch from your fork, as
[CONTRIBUTING.md](CONTRIBUTING.md#5-commit-and-merge) section 5 describes:

```bash
git fetch upstream
git checkout -b contrib/<branch>/<topic> upstream/<branch>
# add skills/<slug>/ and their INDEX.md rows
python3 ../check_skills.py . --base upstream/<branch>   # the checker, from CONTRIBUTING.md section 4
git push -u origin contrib/<branch>/<topic>
```

Skills learned before the library existed are imported as section 6 describes. The first
pull request from a new contributor waits for a maintainer to approve its workflow run;
the check only reads files, it never runs the skills.

## Running the harness on the new robot

To have agents read and grow the library on the robot, the harness needs a backend for it
and the skill-library tools: see the RAX [real-robot adaptation
guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/REAL_ROBOT_ADAPTATION.md#a-skill-library-that-persists-across-sessions),
with [`real/rainbow`](https://github.com/robot-agent-exploration/rax/tree/real/rainbow) as
the reference. Section 5 of CONTRIBUTING.md shows how to start the harness's library from
the branch and publish what its sessions learn.
