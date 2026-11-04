"""Submit one verified frozen classifier, then monitor its exact receipt safely."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, json_write


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fields(obj):
    names = ["ref", "status", "public_score", "private_score", "date", "message", "description", "url"]
    return {name: str(getattr(obj, name)) if name in {"status", "date"} else getattr(obj, name)
            for name in names if hasattr(obj, name)}


def main():
    root, aggregate = ROOT / "runs/span_003", REPO / "results/span_003"
    directory = root / "official"
    directory.mkdir(exist_ok=True)
    lock = (directory / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / "span_plan.json").read_text())
    config = json.loads((REPO / "configs/span_003.json").read_text())
    deadline = datetime.fromisoformat(plan["deadline"]).timestamp()
    token = Path.home() / ".kaggle/access_token"
    if token.exists():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    original_send = requests.Session.send
    def timed_send(session, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (10, 45)
        return original_send(session, request, **kwargs)
    requests.Session.send = timed_send
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    def save(name, value):
        json_write(directory / name, value)
        json_write(aggregate / name, value)

    def quota():
        response = read_retry(api.quota_view, 'quota')
        gpu = response.gpu_quota
        return {"used_seconds": gpu.time_used.total_seconds(),
                "reserved_seconds": gpu.time_reserved.total_seconds(),
                "allowed_seconds": gpu.total_time_allowed.total_seconds(),
                "refresh_time": str(response.quota_refresh_time),
                "paid_scaling": gpu.is_pay_to_scale_enabled}

    def state(stage, **values):
        value = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **values}
        save("official_state.json", value)
        print(json.dumps(value), flush=True)

    def read_retry(call, operation):
        for attempt in range(3):
            try:
                return call()
            except requests.exceptions.RequestException as error:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
                if status in {400, 401, 403, 404} or attempt == 2:
                    raise
                state('read_retry', operation=operation, attempt=attempt+1,
                      error_type=type(error).__name__)
                time.sleep(3*(attempt+1))

    if not (directory / "receipt.json").exists():
        assert not (directory / "submission_intent.json").exists(), "Reconcile an ambiguous submission; never retry blindly"
        verification = json.loads((aggregate / "classifier_verification.json").read_text())
        assert verification["identity"]["passed"]
        assert all(value["valid"] for value in verification["independently_validated_previews"].values())
        cap = config["job_seconds_caps"]["classifier_predict"]
        q = quota()
        spent = q["used_seconds"] - plan["quota_used_at_start"]
        assert not q["paid_scaling"] and q["reserved_seconds"] == 0
        assert q["refresh_time"] == plan["initial_quota_refresh_time"]
        assert plan["gpu_session_seconds_cap"] - spent >= cap
        assert q["allowed_seconds"] - q["used_seconds"] >= cap
        assert time.time() + cap + 900 < deadline
        state("preflight_remote_inventory")
        notebooks = list(read_retry(lambda: api.kernels_list(mine=True, page_size=100, sort_by="dateRun"), 'owned_inventory') or [])
        assert len(notebooks) < 100, "Require complete owned inventory"
        unknown = []
        for index, notebook in enumerate(notebooks):
            try:
                status = str(read_retry(lambda: api.kernels_status(notebook.ref), 'owned_status_'+str(index)).status).split(".")[-1]
                assert status in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "DEAD"}, "Other notebook active"
            except Exception as error:
                code = getattr(getattr(error, "response", None), "status_code", None)
                date = str(getattr(notebook, "last_run_time", ""))
                assert code == 404 and date and date[:10] < plan["started_at"][:10], "Unresolved notebook observation"
                unknown.append({"kernel_sha256": hashlib.sha256(notebook.ref.encode()).hexdigest(),
                                "status": code, "last_run_time": date})
            if index % 5 == 0 or index+1 == len(notebooks):
                state('preflight_inventory_progress', checked=index+1, total=len(notebooks))
        folder = root / "notebooks/classifier_predict"
        push = json.loads((root / "jobs/classifier_predict/push.json").read_text())
        reference = push["ref"].removeprefix("/code/")
        version = push["version_number"]
        assert str(read_retry(lambda: api.kernels_status(reference + "/" + str(version)), 'exact_classifier_status').status).endswith("COMPLETE")
        source = folder / "job.ipynb"
        pushed = json.loads((root / "jobs/classifier_predict/push_intent.json").read_text())
        assert digest(source) == pushed["notebook_sha256"]
        remote = root / "audited_sources/classifier_predict"
        remote.mkdir(parents=True, exist_ok=True)
        read_retry(lambda: api.kernels_pull(reference + "/" + str(version), path=str(remote), metadata=True, quiet=True), 'exact_remote_source')
        pulled = list(remote.glob("*.ipynb"))
        assert len(pulled) == 1
        cells = lambda p: [(cell.cell_type, cell.source) for cell in nbformat.read(p, as_version=4).cells]
        assert cells(pulled[0]) == cells(source), "Actual remote source differs"
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        assert metadata["enable_internet"] is False
        preview = root / "jobs/classifier_predict/output"
        exported = json.loads((preview / "frozen_manifest.json").read_text())
        original = json.loads((REPO / "results/span_002/research_candidate.json").read_text())
        assert all(exported[key] == value for key, value in original.items())
        assert set(exported) - set(original) == {"tokenizer_sha256"}
        assert digest(ROOT / "runs/span_002/jobs/exposure/output/exposure/adapter/adapter_model.safetensors") == original["adapter_sha256"]
        assert digest(preview / "submission.csv") == verification["independently_validated_previews"]["classifier_predict"]["csv_sha256"]
        summary = json.loads((preview / "prediction/prediction_summary.json").read_text())
        session = json.loads((preview / "session_summary.json").read_text())
        setup = max(0., session["seconds"] - summary["prediction_seconds"])
        bundle = json.loads((aggregate / "classifier_predict_bundle.json").read_text())
        projected = bundle["profile_rate"] * 25000 * 1.2 + 600 + setup
        assert projected <= cap
        recent = list(read_retry(lambda: api.competition_submissions("llm-classification-finetuning"), 'recent_submissions') or [])
        message = "Span003 frozen H03 step1438; T0.9441963863; no refit"
        assert not any(str(getattr(item, "description", "")) == message for item in recent), "Potential prior accepted submission needs reconciliation"
        q = quota()
        spent = q['used_seconds'] - plan['quota_used_at_start']
        assert q['reserved_seconds'] == 0 and not q['paid_scaling']
        assert q['refresh_time'] == plan['initial_quota_refresh_time']
        assert spent+cap <= plan['gpu_session_seconds_cap']
        assert q['used_seconds']+cap <= q['allowed_seconds']
        assert time.time()+cap+900 < deadline
        intent = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": reference,
                  "version": version, "notebook_sha256": digest(source), "manifest_sha256": digest(REPO / "results/span_002/research_candidate.json"),
                  "adapter_sha256": original["adapter_sha256"], "quota_before": q,
                  "observed_span_charged_before": spent, "hidden_execution_ceiling_seconds": cap,
                  "conservative_span_seconds_bound": spent + cap,
                  "span_cap_seconds": plan["gpu_session_seconds_cap"], "projection_rows_assumption": 25000,
                  "projected_seconds_including_measured_setup": projected,
                  "all_owned_observed_terminal": True, "owned_notebook_count": len(notebooks),
                  "unobserved_old_notebooks": unknown, "confirmation_used": False,
                  "message": message, "new_gpu_jobs_allowed_during_hidden_run": False}
        intent['controller_sha256'] = digest(Path(__file__).resolve())
        save("submission_intent.json", intent)
        state("submitting_frozen_candidate", kernel=reference, version=version)
        receipt = fields(api.competition_submit_code(file_name="submission.csv", message=message,
            competition="llm-classification-finetuning", kernel=reference, kernel_version=version))
        save("receipt.json", receipt)
        assert receipt.get("ref"), "Submission not accepted; reconcile the intent instead of retrying"
        state("accepted", receipt=receipt)
    receipt = json.loads((directory / "receipt.json").read_text())
    intent = json.loads((directory / "submission_intent.json").read_text())
    while time.time() < deadline:
        try:
            submissions = read_retry(lambda: api.competition_submissions("llm-classification-finetuning"), 'official_result') or []
            matches = [fields(item) for item in submissions if str(getattr(item, "ref", "")) == str(receipt["ref"])]
            if not matches:
                state("accepted_receipt_not_yet_visible", receipt=receipt["ref"])
            else:
                result = matches[0]
                save("official_result.json", result)
                state("waiting_for_official_result", submission=result)
                status = result["status"].split(".")[-1]
                if status in {"COMPLETE", "ERROR", "FAILED", "INVALID", "CANCELED", "CANCELLED"}:
                    final_quota = quota()
                    charged = final_quota["used_seconds"] - plan["quota_used_at_start"]
                    bound = max(charged, intent["conservative_span_seconds_bound"])
                    accounting = {"quota_after": final_quota, "measured_span_quota_charge_seconds": charged,
                                  "hidden_receipt_elapsed_seconds": time.time() - datetime.fromisoformat(intent["created_at"]).timestamp(),
                                  "conservative_span_compute_seconds_bound": bound,
                                  "within_original_cap": bound <= plan["gpu_session_seconds_cap"],
                                  "method": "Other goal GPU jobs stopped before submission; no new GPU jobs allowed. Measured account charges are separate from a conservative hidden ceiling, not assumed zero when quota is unchanged. Receipt elapsed includes queue/obfuscation, not measured GPU time."}
                    save("official_accounting.json", accounting)
                    if status == "COMPLETE":
                        assert result.get("public_score") not in [None, ""] and np.isfinite(float(result["public_score"]))
                    state("official_terminal", submission=result, accounting=accounting)
                    return
        except Exception as error:
            state("observation_retry", error_type=type(error).__name__, receipt=receipt["ref"])
        time.sleep(55)
    state("original_deadline_reached", receipt=receipt["ref"])
    raise TimeoutError("Inspect/stop the accepted submission; never duplicate it")


if __name__ == "__main__":
    main()
