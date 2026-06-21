from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
UUID = "190f2760-b68b-44ee-b805-a6a492a2fa6c"

REQUIRED_FILES = {
    "readme": PROJECT_DIR / "README.md",
    "judge_brief": PROJECT_DIR / "JUDGE_BRIEF.md",
    "registration": PROJECT_DIR / "registration.json",
    "runner": PROJECT_DIR / "run_patrol.py",
    "video": PROJECT_DIR / "demo.mp4",
    "trajectory": PROJECT_DIR / "trajectory.json",
    "report": PROJECT_DIR / "mission_report.json",
    "storyboard": PROJECT_DIR / "storyboard.png",
    "narration": PROJECT_DIR / "narration.srt",
    "sensor_manifest": PROJECT_DIR / "sensor_manifest.json",
    "stress_eval": PROJECT_DIR / "stress_eval.json",
    "policy_card": PROJECT_DIR / "patrol_policy_card.json",
    "rubric_scorecard": PROJECT_DIR / "rubric_scorecard.json",
    "challenge_evidence": PROJECT_DIR / "challenge_evidence.json",
    "submission_manifest": PROJECT_DIR / "submission_manifest.json",
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    checks: list[dict] = []

    for name, path in REQUIRED_FILES.items():
        record(checks, f"file:{name}", path.exists(), rel(path))

    if not all(item["passed"] for item in checks):
        print(json.dumps({"passed": False, "checks": checks}, indent=2))
        return 1

    registration = load_json(REQUIRED_FILES["registration"])
    report = load_json(REQUIRED_FILES["report"])
    trajectory = load_json(REQUIRED_FILES["trajectory"])
    stress_eval = load_json(REQUIRED_FILES["stress_eval"])
    rubric = load_json(REQUIRED_FILES["rubric_scorecard"])
    manifest = load_json(REQUIRED_FILES["submission_manifest"])
    policy_card = load_json(REQUIRED_FILES["policy_card"])
    sensor_manifest = load_json(REQUIRED_FILES["sensor_manifest"])
    challenge = load_json(REQUIRED_FILES["challenge_evidence"])

    samples = trajectory.get("samples", [])
    record(checks, "uuid:registration", registration.get("uuid") == UUID, registration.get("uuid", "missing"))
    record(checks, "uuid:report", report.get("registration_uuid") == UUID, report.get("registration_uuid", "missing"))
    record(checks, "uuid:stress_eval", stress_eval.get("registration_uuid") == UUID, stress_eval.get("registration_uuid", "missing"))
    record(checks, "mission:success", report.get("success") is True, str(report.get("success")))
    record(checks, "mission:waypoints", report.get("completed_waypoint_count", 0) >= 7, str(report.get("completed_waypoint_count")))
    record(checks, "mission:avoidance", report.get("avoidance_count", 0) >= 1, str(report.get("avoidance_count")))
    record(checks, "mission:anomaly", report.get("anomaly_detected") is True, str(report.get("anomaly_detected")))
    record(
        checks,
        "mission:clearance",
        report.get("min_hard_obstacle_clearance_m", 0.0) >= report.get("safety_clearance_threshold_m", 0.36),
        f"{report.get('min_hard_obstacle_clearance_m')} >= {report.get('safety_clearance_threshold_m')}",
    )
    record(
        checks,
        "mission:return_distance",
        report.get("final_dispatch_distance_m", 99.0) < 0.65,
        str(report.get("final_dispatch_distance_m")),
    )
    record(checks, "trajectory:samples", len(samples) >= 300, str(len(samples)))
    record(checks, "sensor_manifest:samples", sensor_manifest.get("sample_count") == len(samples), str(sensor_manifest.get("sample_count")))
    record(checks, "stress:rollouts", stress_eval.get("rollout_count", 0) >= 32, str(stress_eval.get("rollout_count")))
    record(checks, "stress:success_rate", stress_eval.get("success_rate", 0.0) >= 1.0, str(stress_eval.get("success_rate")))
    record(checks, "rubric:criteria", len(rubric.get("scorecard", {})) >= 8, str(len(rubric.get("scorecard", {}))))
    record(checks, "manifest:artifacts", manifest.get("artifact_count", 0) >= 14, str(manifest.get("artifact_count")))
    record(checks, "policy:closed_loop", policy_card.get("policy_type") == "closed-loop deterministic finite-state planner", policy_card.get("policy_type", "missing"))
    record(checks, "challenge:evidence", bool(challenge.get("rubric_keyword_index")), "keyword index present")

    video_bytes = REQUIRED_FILES["video"].stat().st_size
    storyboard_bytes = REQUIRED_FILES["storyboard"].stat().st_size
    narration_text = REQUIRED_FILES["narration"].read_text(encoding="utf-8")
    record(checks, "media:video_size", video_bytes > 1_000_000, f"{video_bytes} bytes")
    record(checks, "media:storyboard_size", storyboard_bytes > 80_000, f"{storyboard_bytes} bytes")
    record(checks, "media:narration_srt", "-->" in narration_text and "PASS:" in narration_text, "SRT cues present")

    passed = all(item["passed"] for item in checks)
    print(
        json.dumps(
            {
                "passed": passed,
                "project": "Aegis Campus Patrol",
                "registration_uuid": UUID,
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
