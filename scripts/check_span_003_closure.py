"""Verify the completed v3/v4 boundary after terminal audit and notebook execution."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import nbformat

REPO = Path(__file__).resolve().parents[1]
AGG = REPO / "results/span_003"
HEAVY = Path.home() / "database/llm_classification_finetuning/runs/span_003"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    subprocess.run([str(REPO / ".venv/bin/python"), "scripts/backlog.py", "check"], cwd=REPO, check=True)
    registry = read(REPO / "backlog/hypotheses.json")
    versions = {v["number"]: v for v in registry["versions"]}
    items = {item["id"]: item for item in registry["items"]}
    assert registry["active_version"] == 4 and versions[3]["state"] == "closed", "V3 is not sealed"
    assert versions[4]["state"] == "active" and not versions[4]["focus_ids"]
    assert versions[5]["state"] == "draft" and not versions[5]["item_ids"] and not versions[5]["focus_ids"]
    assert versions[1]["state"] == versions[2]["state"] == "closed"
    focus = {"E03", "H39", "E02", "H43"}
    assert set(versions[3]["focus_ids"]) == focus and len(versions[3]["item_ids"]) == 37
    assert all(items[key]["status"] in {"done", "rejected", "deferred"} and items[key]["evidence"] for key in focus)
    assert not any(item["status"] == "running" for item in items.values())
    new = {key for key in versions[4]["item_ids"] if items[key]["introduced_in"] == 4}
    assert {"H44", "E04"} <= new
    assert all(items[key]["discovered_during"] == 3 and items[key]["derived_from"] for key in new)
    carry = {key for key in versions[3]["item_ids"] if items[key]["status"] not in {"done", "rejected"}}
    assert set(versions[4]["item_ids"]) == carry | new
    assert not carry & new
    assert all(items[key]["status"] in {"planned", "deferred"} for key in versions[4]["item_ids"])
    historical = {}
    for pattern in ["v1.*", "v2.*", "v3.start.*"]:
        for path in sorted((REPO / "backlog/versions").glob(pattern)):
            relative = str(path.relative_to(REPO))
            assert path.read_bytes() == subprocess.check_output(["git", "show", "a18feff:" + relative], cwd=REPO)
            historical[relative] = digest(path)
    assert len(historical) == 10
    terminal, local = read(AGG / "terminal_audit.json"), read(AGG / "local_audit.json")
    assert terminal["passed"] and local["passed"]
    assert terminal["local_audit_sha256"] == digest(AGG / "local_audit.json")
    assert local["audit_program_sha256"] == digest(REPO / "scripts/audit_span_003.py")
    assert local["private_probability_sha256"] == digest(HEAVY / "cached_evaluated_predictions.npz")
    assert terminal["all_recorded_jobs_terminal"] and not terminal["nonterminal_owned_notebooks"]
    assert len(terminal["jobs"]) == 4 and all(job["remote_source_matches"] and job["internet_disabled"] for job in terminal["jobs"])
    assert terminal["quota"]["reserved_seconds"] == 0
    assert all(terminal[key] for key in ["within_original_compute_cap", "within_original_deadline", "no_paid_scaling", "no_quota_refresh_used"])
    assert terminal["original_start"] == "2026-11-04T07:42:17+00:00"
    assert terminal["original_deadline"] == "2026-11-06T07:42:17+00:00"
    now = datetime.now(timezone.utc)
    assert now < datetime.fromisoformat(terminal["original_deadline"])
    assert (now - datetime.fromisoformat(terminal["checked_at"])).total_seconds() < 3600, "Refresh the remote terminal observation"
    decisions = read(AGG / "decisions.json")
    assert decisions["all_focus_measurements_complete"] and decisions["official_submissions_made"] == 1
    assert digest(REPO / "scripts/decide_span_003.py") == decisions["source_sha256"]
    for name, sha in decisions["input_sha256"].items():
        assert digest(AGG / name) == sha, name
    intent = read(AGG / "submission_intent.json")
    assert digest(REPO / "scripts/submit_span_003.py") == intent["controller_sha256"]
    assert digest(REPO / "results/span_002/research_candidate.json") == intent["manifest_sha256"]
    assert read(AGG / "receipt.json")["ref"] == terminal["official_submission"]["ref"] == 57979825
    accounting = read(AGG / "official_accounting.json")
    assert accounting["within_original_cap"]
    monitor = (HEAVY / "official/monitor.lock").open("a")
    fcntl.flock(monitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(HEAVY / "official/official_state.json")["stage"] == "official_terminal"
    fcntl.flock(monitor, fcntl.LOCK_UN)
    files = [AGG / name for name in ["cached_results.json", "frozen_cached_recipes.json", "classifier_verification.json",
             "native_profile_results.json", "decisions.json", "local_audit.json", "terminal_audit.json",
             "submission_intent.json", "receipt.json", "official_result.json", "official_accounting.json"]]
    files += [REPO / "docs" / f"SPAN_003_{name}.md" for name in ["RESULTS", "H39", "H43", "E02", "E03", "COMPLETION"]]
    if terminal["official_submission"]["status"] == "COMPLETE":
        candidate_path = AGG / "deployed_candidate.json"
        candidate = read(candidate_path)
        assert candidate == read(HEAVY / "deployed_candidate.json")
        assert candidate["officially_verified"] and candidate["submission_ref"] == 57979825
        assert candidate["official_log_loss"] == float(terminal["official_submission"]["public_score"])
        assert candidate["frozen_manifest_sha256"] == intent["manifest_sha256"]
        assert candidate["adapter_sha256"] == intent["adapter_sha256"]
        files.append(candidate_path)
    notebook_path = REPO / "notebooks/14_backlog_v3_span.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [output for cell in code for output in cell.outputs]
    assert len(code) == 5 and all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 3
    text = "\n".join(output.get("text", "") for output in outputs)
    assert "v3 state: closed" in text and "57979825" in text
    assert "SubmissionStatus." + terminal["official_submission"]["status"] in text
    link = Path.home() / "notebooks/llm_classification_finetuning"
    assert link.is_symlink() and link.resolve() == REPO / "notebooks"
    files += [notebook_path, REPO / "scripts/check_span_003_closure.py", REPO / "backlog/hypotheses.json",
              REPO / "backlog/versions/v3.final.json", REPO / "backlog/versions/v3.final.md",
              REPO / "backlog/versions/v4.start.json", REPO / "backlog/versions/v4.start.md"]
    result = {"checked_at": now.isoformat(), "passed": True, "active_version": 4,
              "v3_closed": True, "v3_focus_resolved": sorted(focus), "v4_focus_empty": True,
              "v4_new": sorted(new), "v4_carryovers": len(carry), "v4_tasks": len(versions[4]["item_ids"]),
              "v5_empty_draft": True, "no_running_tasks": True,
              "historical_snapshots_unchanged_from": "a18feff", "historical_snapshot_sha256": historical,
              "official_submission": terminal["official_submission"],
              "conservative_compute_seconds_bound": terminal["conservative_span_compute_seconds_bound"],
              "terminal_audit_at": terminal["checked_at"], "notebook_executed_code_cells": 5,
              "notebook_figures": 3, "notebook_errors": 0, "notebook_link_resolves": True,
              "artifact_sha256": {str(path.relative_to(REPO)): digest(path) for path in files},
              "publication": "Verify remote main equals local HEAD after the closing commit/push; do not execute v4."}
    (AGG / "closure_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
