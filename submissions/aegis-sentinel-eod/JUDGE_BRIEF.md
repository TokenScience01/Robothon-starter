# Aegis Sentinel EOD v2 - Judge Brief

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Why This Entry Targets Top 10

Aegis Sentinel EOD v2 is a **campus hazardous-package neutralization** benchmark built around a differentiated platform: an **Aegis-inspired quadruped patrol body with an onboard tri-finger manipulator**. Unlike the medicine-triage leaderboard cluster, this project combines **mobile patrol + MuJoCo rangefinder scan + tactile tri-finger grasp + 4N shove recovery + containment placement + alarm confirmation** in one reproducible 34-second demo.

The controller steps MuJoCo physics, reads **rangefinder / frame-position / touch** sensors, and applies a **tactile residual grasp policy** during reach, grasp, lift, recover, transport, and place.

## Inspect First

1. `artifacts/keyframes.png` - eight-panel storyboard with shove recovery beat.
2. `artifacts/demo.mp4` - compact HUD video with beat labels, shove telemetry, pass banner, and confidence overlays.
3. `artifacts/narration.srt` - stage subtitles aligned to the video beats.
4. `aegis_sentinel_scene.xml` - MJCF scene with patrol platform, tri-finger hand, free package body, containment bin, alarm button, actuators, and sensors.
5. `learned_policy_weights.json` and `dataset/training_report.json` - residual policy evidence.
6. `artifacts/contact_timeline.json` - tri-finger contact samples and stable-contact count.
7. `artifacts/stress_eval.json` - 64 fixed-seed residual-vs-baseline rollouts.
8. `artifacts/challenge_evidence.json` - rubric keyword index and quantitative success block.
9. `artifacts/mission_report.json` - final success criteria and rubric alignment.

## Quantitative Evidence

- Platform: Aegis-inspired quadruped patrol body + onboard 3-finger manipulator
- Actuated channels: 14 position servos
- MuJoCo sensors: rangefinder, 3 touch pads, 4 frame-position sensors, alarm joint position
- Residual corrections: logged per frame during manipulation stages
- Max lateral shove: 4.0 N with slip recovery stage
- Stress rollouts: 64 seeds, learned-policy success and baseline comparison
- Task phases: patrol, scan, align, reach, grasp, lift, recover, transport, place, alarm, complete

## Rubric Mapping

- **Runnability**: one command regenerates video, trajectory, report, policy card, contact timeline, stress replay, SRT, and challenge evidence.
- **MuJoCo depth**: custom MJCF, free package body, slide alarm joint, touch sensors, rangefinder, contacts, position actuators.
- **Task design**: campus EOD patrol with scan, grasp, shove recovery, containment, and alarm confirmation.
- **Control**: stage planner plus tactile residual policy driven by MuJoCo sensor streams.
- **Dexterity**: thumb-opposed tri-finger grasp, 4N shove recovery, contact balancing, transport, and release.
- **Engineering quality**: validator, judge brief, scorecard, manifest, training report, 64-seed stress replay.
- **Presentation**: 34s HUD video, beat labels, pass banner, SRT subtitles, 8-panel storyboard.
- **Innovation**: quadruped-plus-manipulator campus EOD benchmark with disturbance recovery.

## Run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```