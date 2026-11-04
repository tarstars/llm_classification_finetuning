"""Verify the sealed v2 handoff without launching work or changing the backlog."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import nbformat

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/span_002"
HEAVY = Path.home() / "database/llm_classification_finetuning/runs/span_002"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    subprocess.run([str(REPO / ".venv/bin/python"), "scripts/backlog.py", "check"],
                   cwd=REPO, check=True)
    registry = read(REPO / "backlog/hypotheses.json")
    versions = {v["number"]: v for v in registry["versions"]}
    items = {item["id"]: item for item in registry["items"]}
    assert registry["active_version"] == 3
    assert versions[1]["state"] == versions[2]["state"] == "closed"
    assert versions[3]["state"] == "active" and not versions[3]["focus_ids"]
    assert versions[4]["state"] == "draft" and not versions[4]["item_ids"]
    assert len(versions[2]["item_ids"]) == 39
    assert len(versions[3]["item_ids"]) == 37
    assert set(versions[2]["focus_ids"]) == {"H40", "E01", "H03", "H37", "H42"}
    assert all(items[key]["status"] == "done" for key in versions[2]["focus_ids"])
    assert not any(item["status"] == "running" for item in items.values())
    new = {key for key in versions[3]["item_ids"] if items[key]["introduced_in"] == 3}
    assert new == {"H43", "E02", "E03"}
    carried = set(versions[3]["item_ids"]) - new
    expected = {key for key in versions[2]["item_ids"]
                if items[key]["status"] not in {"done", "rejected"}}
    assert carried == expected and len(carried) == 34
    assert all(items[key]["status"] in {"planned", "deferred"}
               for key in versions[3]["item_ids"])

    preserved = {}
    for pattern in ["v1.*", "v2.start.*"]:
        for path in sorted((REPO / "backlog/versions").glob(pattern)):
            relative = str(path.relative_to(REPO))
            original = subprocess.check_output(["git", "show", f"8e2fadc:{relative}"], cwd=REPO)
            assert path.read_bytes() == original, relative
            preserved[relative] = digest(path)

    audit = read(RESULTS / "terminal_audit.json")
    assert audit["all_recorded_jobs_terminal"] and not audit["nonterminal_owned_notebooks"]
    assert len(audit["jobs"]) == 6 and all(job["remote_source_matches"] for job in audit["jobs"])
    assert audit["quota"]["reserved_seconds"] == 0
    assert audit["within_original_compute_cap"] and audit["within_original_deadline"]
    assert audit["no_quota_refresh_used"] and audit["no_paid_scaling"]
    assert audit["original_started_at"] == "2026-11-03T18:23:46+00:00"
    assert audit["original_deadline"] == "2026-11-05T18:23:46+00:00"
    assert datetime.now(timezone.utc) < datetime.fromisoformat(audit["original_deadline"])
    assert (datetime.now(timezone.utc) - datetime.fromisoformat(audit["checked_at"])).total_seconds() < 3600

    decisions = read(RESULTS / "decisions.json")
    assert decisions["all_focus_measurements_complete"]
    assert not decisions["new_competition_submission_made"]
    assert decisions["tasks"]["H03"]["accepted_as_research_reference"]
    assert not decisions["tasks"]["H03"]["officially_deployed"]
    candidate = read(RESULTS / "research_candidate.json")
    assert candidate == read(HEAVY / "research_candidate.json")
    assert not candidate["officially_verified"] and not candidate["confirmation_used_for_selection"]
    assert candidate["analysis_results_sha256"] == digest(RESULTS / "exposure_results.json")
    assert candidate["source_predictions_sha256"] == digest(
        HEAVY / "jobs/exposure/output/exposure/dev_predictions.npz")
    assert candidate["adapter_sha256"] == digest(
        HEAVY / "jobs/exposure/output/exposure/adapter/adapter_model.safetensors")
    assert candidate["class_order"] == ["winner_model_a", "winner_model_b", "winner_tie"]

    notebook_path = REPO / "notebooks/13_backlog_v2_span.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) == 6 and all(cell.execution_count is not None for cell in cells)
    outputs = [output for cell in cells for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    images = sum("image/png" in output.get("data", {}) for output in outputs)
    assert images == 5
    text = "\n".join(output.get("text", "") for output in outputs)
    assert "v2 state: closed" in text and "officially_verified" in text
    link = Path.home() / "notebooks/llm_classification_finetuning"
    assert link.is_symlink() and link.resolve() == REPO / "notebooks"

    files = [RESULTS / name for name in ["calibration_results.json", "instrumentation_results.json",
             "exposure_results.json", "expanded_results.json", "packing_diagnostics.json",
             "research_candidate.json", "decisions.json", "terminal_audit.json"]]
    files += [REPO / "docs" / f"SPAN_002_{name}.md"
              for name in ["RESULTS", "H03", "H37", "H40", "H42", "E01"]]
    files += [REPO / "scripts/check_span_002_closure.py", notebook_path]
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "active_version": 3, "v2_closed": True, "v3_focus_empty": True,
        "v3_discoveries": sorted(new), "v3_carryovers": len(carried),
        "v3_tasks": 37, "v4_empty_draft": True, "no_running_tasks": True,
        "terminal_resource_audit_at": audit["checked_at"],
        "gpu_session_hours": audit["span_gpu_session_hours"],
        "historical_snapshots_unchanged_from": "8e2fadc",
        "historical_snapshot_sha256": preserved,
        "candidate_manifest_and_private_export_match": True,
        "official_submission_made": False,
        "notebook": {"executed_code_cells": len(cells), "figures": images, "errors": 0},
        "notebook_link_resolves": True,
        "artifact_sha256": {str(path.relative_to(REPO)): digest(path) for path in files},
        "publication": "Verify remote main equals local HEAD after the closing commit and push.",
    }
    (RESULTS / "closure_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
