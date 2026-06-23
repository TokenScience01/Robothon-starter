# Aegis Sentinel EOD v4 - Judge Brief

Registration UUID: `190f2760-b68b-44ee-b805-a6a492a2fa6c`

## Quantitative Evidence (Inspect First)

| Metric | Value |
|--------|-------|
| Demo duration | 30 s |
| Actuated channels | 15 position servos |
| MuJoCo sensors | rangefinder + 3 touch + 5 frame/joint position |
| Max lateral shove | 4.0 N |
| Load hold benchmark | 9.0x object weight |
| Disturbance events | 2 (grasp shove + transport slip) |
| Stress rollouts | 96 fixed seeds |
| Residual policy success | 100% (baseline 66.7%) |
| Dataset export | labels, metrics, episode trace, sensor manifest |
| Task phases | 12 (patrol → scan → grasp → recover → hold → transport → place → seal → alarm) |

## Why This Entry Targets Top 25

Aegis Sentinel EOD v4 is a campus hazardous-package neutralization benchmark with an **Aegis-inspired quadruped patrol body (BASE_LINK mesh) and onboard tri-finger manipulator**. It combines patrol, MuJoCo rangefinder scan, tactile grasp, **4N shove recovery**, **9x load hold**, **second transport slip recovery**, containment placement, **seal confirmation**, and alarm confirmation in a compact **30-second** reproducible demo with an opening **QUADRUPED + MANIPULATOR EOD** splash.

## Inspect First

1. `artifacts/keyframes.png` - ten-panel storyboard including hold, transport slip, and seal confirm.
2. `artifacts/demo.mp4` - beat-labeled HUD video with opening splash, dual slip recovery, and pass banner.
3. `artifacts/narration.srt` - stage subtitles.
4. `dataset/labels.csv` - per-frame labeled trajectory export.
5. `dataset/metrics.json` - quantitative success summary.
6. `artifacts/stress_eval.json` - 96 fixed-seed residual-vs-baseline rollouts.
7. `artifacts/challenge_evidence.json` - rubric keyword index.
8. `aegis_sentinel_scene.xml` - MJCF scene with Aegis mesh, tri-finger hand, seal button, alarm button, sensors, and actuators.

## Run

```bash
python3 -m pip install -r requirements.txt
python submissions/aegis-sentinel-eod/train_residual_policy.py
python submissions/aegis-sentinel-eod/run_sentinel_eod.py
python submissions/aegis-sentinel-eod/validate_submission.py
```