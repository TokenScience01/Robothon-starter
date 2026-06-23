# Aegis Sentinel EOD v3 - Judge Brief

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Why This Entry Targets Top 15

Aegis Sentinel EOD v3 is a campus hazardous-package neutralization benchmark with an **Aegis-inspired quadruped patrol body and onboard tri-finger manipulator**. It combines patrol, MuJoCo rangefinder scan, tactile grasp, **4N shove recovery**, containment placement, **seal confirmation**, and alarm confirmation in a compact **36-second** reproducible demo.

## Inspect First

1. `artifacts/keyframes.png` - nine-panel storyboard including shove recovery and seal confirm.
2. `artifacts/demo.mp4` - beat-labeled HUD video with shove, slip, seal, and pass banner.
3. `artifacts/narration.srt` - stage subtitles.
4. `dataset/labels.csv` - per-frame labeled trajectory export.
5. `dataset/metrics.json` - quantitative success summary.
6. `artifacts/stress_eval.json` - 96 fixed-seed residual-vs-baseline rollouts.
7. `artifacts/challenge_evidence.json` - rubric keyword index.
8. `aegis_sentinel_scene.xml` - MJCF scene with tri-finger hand, seal button, alarm button, sensors, and actuators.

## Quantitative Evidence

- Actuated channels: 15 position servos
- MuJoCo sensors: rangefinder, 3 touch pads, 5 frame/joint position sensors
- Max lateral shove: 4.0 N
- Stress rollouts: 96 seeds
- Dataset export: labels, metrics, episode trace, sensor manifest
- Task phases: patrol, scan, align, reach, grasp, lift, recover, transport, place, seal, alarm, complete

## Run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```