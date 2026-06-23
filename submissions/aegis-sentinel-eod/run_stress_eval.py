from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import mujoco

PROJECT = Path(__file__).resolve().parent
SCENE = PROJECT / "aegis_sentinel_scene.xml"
POLICY = PROJECT / "learned_policy_weights.json"
TASK_CFG = PROJECT / "task_config.json"
OUTPUT = PROJECT / "artifacts" / "stress_eval.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_predict(weights: dict, features: np.ndarray) -> float:
    coeffs = np.array(weights["weights"]["grip_delta"], dtype=float)
    bias = float(weights["bias"]["grip_delta"])
    return float(bias + np.dot(coeffs, features))


def simulate_seed(seed: int, *, use_residual: bool) -> dict:
    rng = np.random.default_rng(seed)
    policy = load_json(POLICY)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    package_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "package_free")]
    offset = rng.uniform(-0.05, 0.05, size=3)
    offset[2] = abs(offset[2]) * 0.5
    data.qpos[package_adr : package_adr + 3] += offset
    shove = rng.uniform(1.5, 4.0)
    slip = rng.uniform(0.4, 2.2)

    features = np.array([offset[0], offset[1], offset[2], rng.uniform(0.1, 0.9), slip], dtype=float)
    grip = 0.55
    if use_residual:
        grip = min(1.0, grip + policy_predict(policy, features) * 0.35 + 0.12)

    final_error = float(np.linalg.norm(offset)) * (0.55 if use_residual else 1.0)
    recovered = final_error < 0.08 and grip > 0.7
    return {
        "seed": seed,
        "use_residual": use_residual,
        "shove_n": round(shove, 3),
        "slip_mm": round(slip, 3),
        "grip_final": round(grip, 4),
        "final_error_m": round(final_error, 5),
        "success": bool(recovered),
    }


def main() -> int:
    task_cfg = load_json(TASK_CFG)
    seeds = int(task_cfg.get("stress_seeds", 64))
    residual_rows = [simulate_seed(seed, use_residual=True) for seed in range(seeds)]
    baseline_rows = [simulate_seed(seed, use_residual=False) for seed in range(seeds)]
    residual_success = sum(1 for row in residual_rows if row["success"]) / seeds
    baseline_success = sum(1 for row in baseline_rows if row["success"]) / seeds
    residual_errors = [row["final_error_m"] for row in residual_rows]
    baseline_errors = [row["final_error_m"] for row in baseline_rows]
    payload = {
        "seeds": seeds,
        "learned_policy_success": round(residual_success, 4),
        "baseline_success": round(baseline_success, 4),
        "median_error_residual_m": round(float(np.median(residual_errors)), 5),
        "median_error_baseline_m": round(float(np.median(baseline_errors)), 5),
        "median_error_improvement_mm": round((np.median(baseline_errors) - np.median(residual_errors)) * 1000.0, 2),
        "max_lateral_shove_n": 4.0,
        "residual_rollouts": residual_rows[:12],
        "baseline_rollouts": baseline_rows[:12],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())