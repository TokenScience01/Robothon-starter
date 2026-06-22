# Aegis Campus Patrol - Judge Brief

Registration UUID: 190f2760-b68b-44ee-b805-a6a492a2fa6c

## Why This Entry Is Stronger Now

Aegis Campus Patrol + DexTriage is a reproducible MuJoCo hybrid task built
around the packaged Aegis quadruped plus a procedural five-finger manipulation
station. The submission exports a full judge evidence pack, not only a demo
video: mission report, trajectory, storyboard, SRT narration, sensor manifest,
fixed-seed stress replay, policy card, rubric scorecard, challenge evidence,
and submission manifest.

The task is a long-horizon campus safety run: leave dispatch, patrol a waypoint
route, detect a blocked walkway, execute a closed-loop detour, inspect a
suspicious package, return to dispatch, then trigger a five-finger DexTriage
hand that grasps a vial, rotates the cap, places the vial into a sterile pod,
presses an audit button, and exports machine-readable pass/fail evidence.

## Inspect First

1. `demo.mp4` - generated 75 second demo with HUD, minimap, state labels, anomaly status, clearance telemetry, and DexTriage overlays.
2. `storyboard.png` - eight keyframes covering patrol, inspection, five-finger grasp, cap rotation, pod placement, and completion.
3. `mission_report.json` - pass/fail checks, mission score, stress summary, and rubric alignment.
4. `stress_eval.json` - 32 fixed-seed replay perturbations for obstacle shifts and range bias.
5. `sensor_manifest.json` - exported state, pose, range, clearance, package distance, and anomaly channels.
6. `patrol_policy_card.json` - closed-loop FSM inputs, outputs, thresholds, and limitations.
7. `rubric_scorecard.json` - explicit mapping to the public Robothon judging rubric.
8. `validate_submission.py` - one command package validator.

## Quantitative Evidence

- Waypoints completed: 7/7
- Mission success: true
- Avoidance count: at least 1
- Safety clearance target: 0.36 m
- Suspicious-package anomaly detected: true
- Max anomaly score: 0.8729
- Five-finger contact count: 5/5
- Cap rotation: 218 degrees
- Audit-button press: 28 mm
- Final vial-to-pod distance: 0.0 m
- Return-to-dispatch threshold: 0.65 m
- Stress replay rollouts: 32 fixed seeds
- Generated artifacts: video, trajectory, report, storyboard, narration, sensor manifest, stress replay, policy card, challenge evidence, rubric scorecard, and manifest

## Rubric Mapping

- Reproducibility: one command regenerates all main artifacts; validator checks UUID, video, trajectory, report, scorecard, stress replay, policy card, sensor manifest, and manifest.
- MuJoCo depth: Aegis URDF import, freejoint base, 12 leg joint targets, five-finger hand, 15 hinge joints, 15 actuators, fingertip touch sensors, vial/cap free bodies, audit-button slide joint, generated scene geoms, lights, camera, waypoint markers, hard obstacles, package zone, and range site.
- Task design: campus security route with blocked walkway, obstacle detour, suspicious-package inspection, return-to-base completion, and medication DexTriage manipulation.
- Control: closed-loop finite-state planner uses range, obstacle identity, waypoint state, package distance, safety projection, return threshold, then drives a deterministic five-finger manipulation phase.
- Dexterity: five-finger grasp, cap rotation over 200 degrees, vial-to-pod placement, and audit-button press.
- Engineering quality: compact submission folder, deterministic run, machine-readable artifacts, fixed-seed replay checks, and explicit limitations.
- Presentation: video HUD/minimap, storyboard, SRT narration, and judge-first metrics reduce ambiguity for AI scoring.
- Innovation: turns a quadruped campus-security scenario into a reproducible evidence-export benchmark.

## Honest Scope

The gait and five-finger hand policy are deterministic rather than learned
torque-control policies. The stress evaluation is a fixed-seed replay over the
generated trajectory evidence, not a full randomized physics rerender. This is
intentional: the submission optimizes reproducibility and judge verifiability
for a hybrid patrol-and-manipulation task.
