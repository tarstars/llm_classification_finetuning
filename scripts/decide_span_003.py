"""Record v3 stopping/adoption decisions from actual evidence, without new inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, json_write


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root, aggregate = ROOT / "runs/span_003", REPO / "results/span_003"
    cached = read(aggregate / "cached_results.json")
    native = read(aggregate / "native_profile_results.json")
    verified = read(aggregate / "classifier_verification.json")
    local = read(aggregate / "local_audit.json")
    assert local["passed"] and local["private_probability_sha256"] == cached["private_predictions_sha256"]
    assert digest(REPO / "scripts/audit_span_003.py") == local["audit_program_sha256"]
    frozen = read(aggregate / "frozen_cached_recipes.json")
    receipt, official = read(aggregate / "receipt.json"), read(aggregate / "official_result.json")
    assert receipt["ref"] == official["ref"] == 57979825
    status = official["status"].split(".")[-1]
    terminal = status in {"COMPLETE", "ERROR", "FAILED", "INVALID", "CANCELED", "CANCELLED"}
    is_verified = status == "COMPLETE"
    score = float(official["public_score"]) if is_verified else None
    assert score is None or np.isfinite(score) and score >= 0
    baseline = .97025
    best = min(baseline, score) if is_verified else baseline
    selected_route = frozen["selected"]["selected_entropy"]
    route = native["routes"][selected_route]
    passing = [name for name, value in native["routes"].items()
               if value["cached_quality_gate_passed"] and value["runtime_guard_passed"]
               and native["numerical_identity_passed"]]
    assert not passing and not native["cached_numerical_recipe_reusable"]
    assert not (root / "jobs/cascade_followup/push_intent.json").exists()
    primary_blend = cached["contrasts"]["selected_blend_vs_vector"]
    conditional = cached["contrasts"]["conditional_vs_vector_tie"]
    tasks = {
        "H43": {"experiment_complete": True, "conditional_tie_preserving_adopted": False,
                "conditional_vs_calibrated_vector": conditional,
                "nested_selected_vs_vector": primary_blend,
                "best_cached_family": frozen["selected"]["selected_blend"],
                "deployed": False,
                "reason": "Conditional tie preservation loses to the calibrated full-vector control. Postcalibration helps the full mixture, but deployable runtime/cache compatibility is absent."},
        "H39": {"experiment_complete": True, "selected_cached_route": selected_route,
                "cached_primary_contrast": cached["contrasts"]["selected_entropy_vs_classifier"],
                "selected_route_runtime_guard_passed": route["runtime_guard_passed"],
                "native_cache_identity_passed": native["numerical_identity_passed"],
                "passing_deployment_routes": passing, "optional_new_group_inference_executed": False,
                "deployed": False,
                "reason": "20% route misses the current projected runtime guard; 10% cached gain is uncertain; neither passes numerical identity. Stop at predeclared gates."},
        "E02": {"experiment_complete": True, "offline_hardware_artifact_verified": native["hardware_passed"],
                "numerical_compatibility_passed": native["numerical_identity_passed"],
                "failed_identity_guard_preserved": True, "new_consistent_rescore_executed": False,
                "reason": "Native CUDA and actual two-device inference work, but cache identity fails. No threshold relaxation, cache pooling or further GPU expansion."},
        "E03": {"experiment_complete": terminal, "offline_identity_passed": verified["identity"]["passed"],
                "visible_csvs_verified": all(row["valid"] for row in verified["independently_validated_previews"].values()),
                "official_submission": receipt["ref"], "official_status": status,
                "officially_verified": is_verified, "official_log_loss": score,
                "improves_best_recorded_official": is_verified and score < baseline,
                "reason": "One frozen classifier submitted; monitor its exact receipt. No refit or leaderboard retuning."},
    }
    if is_verified:
        manifest_path = REPO / "results/span_002/research_candidate.json"
        manifest = read(manifest_path)
        script_version = int(parse_qs(urlparse(official["url"]).query)["scriptVersionId"][0])
        deployed = {"span": 3, "verified_at": datetime.now(timezone.utc).isoformat(),
                    "source_kernel": "arseniytrushin/llm-preference-span-003-classifier-predict",
                    "kernel_version": 1, "script_version_id": script_version,
                    "submission_ref": receipt["ref"], "official_log_loss": score,
                    "officially_verified": True, "best_recorded_official": score < baseline,
                    "frozen_manifest": "results/span_002/research_candidate.json",
                    "frozen_manifest_sha256": digest(manifest_path),
                    "adapter_sha256": manifest["adapter_sha256"], "temperature": manifest["temperature"],
                    "class_order": manifest["class_order"], "prediction_recipe": manifest["prediction_recipe"],
                    "old_official_candidate_preserved": True,
                    "note": "New verification record; do not modify the historical frozen v2 manifest or attempt002 selection."}
        destination = root / "deployed_candidate.json"
        if destination.exists():
            previous = read(destination)
            assert {k: v for k, v in previous.items() if k != "verified_at"} == {k: v for k, v in deployed.items() if k != "verified_at"}
            deployed = previous
        for directory in [root, aggregate]:
            json_write(directory / "deployed_candidate.json", deployed)
    result = {"span": 3, "generated_at": datetime.now(timezone.utc).isoformat(),
              "all_focus_measurements_complete": all(item["experiment_complete"] for item in tasks.values()),
              "tasks": tasks, "official_submissions_made": 1,
              "best_recorded_official_before_span": baseline, "best_recorded_official_now": best,
              "source_sha256": digest(Path(__file__).resolve()),
              "input_sha256": {name: digest(aggregate / name) for name in
                               ["cached_results.json", "native_profile_results.json", "classifier_verification.json",
                                "local_audit.json", "receipt.json", "official_result.json"]},
              "selection_limits": "Inspected development; grouped nested fits; conditional group-bootstrap intervals. Native numeric cause unisolated; no cross-cohort additive gain claim."}
    for directory in [root, aggregate]:
        json_write(directory / "decisions.json", result)
    print(json.dumps({"all_focus_measurements_complete": result["all_focus_measurements_complete"],
                      "official_status": status, "official_score": score, "best_recorded_official": best}, indent=2))


if __name__ == "__main__":
    main()
