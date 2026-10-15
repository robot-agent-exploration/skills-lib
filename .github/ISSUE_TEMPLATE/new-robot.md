---
name: New robot
about: Ask for a library branch for a robot that has none yet
title: "New robot: <robot name>"
labels: new robot
---

<!-- See NEW_ROBOT.md. A maintainer creates the branch from this issue; then you open a pull
request with your skills into it. -->

**Branch name** (the robot's backend key, lower-case, e.g. `franka`, `rby1`):

**The robot**: model; arms and their DoF; grippers or hands; mobile base, if any; cameras
(where they are mounted, RGB or RGB-D); the base frame the code uses.

**How the code reaches the robot**: the harness and its context object (for RAX,
`api.runtime.Context`), the tools the skills call, and a link to their documentation.

**The setup the harness reports** (skills read it instead of copying it): camera
parameters, named postures, limits and workspace.

**Other dependencies** of the skills (Python packages, models, services):

**The first skills you plan to contribute**, and the sessions they come from:

**Why an existing branch does not fit** (for a robot close to one that has a branch, such as
the same arm with another gripper):
