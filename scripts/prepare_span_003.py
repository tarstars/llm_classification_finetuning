"""Inventory live resources and immutable v3 inputs without launching experiments."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition, split_from_ids


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root, aggregate = ROOT / "runs/span_003", REPO / "results/span_003"
    root.mkdir(parents=True, exist_ok=True)
    activation = json.loads((aggregate / "activation.json").read_text())
    start = datetime.fromtimestamp(activation["goal_service_created_at"], timezone.utc)
    assert start.isoformat() == activation["started_at"]
    assert (start + timedelta(hours=48)).isoformat() == activation["deadline"]
    token = Path.home() / ".kaggle/access_token"
    if token.exists():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    quota = api.quota_view()
    gpu = quota.gpu_quota
    q = {"used_seconds": gpu.time_used.total_seconds(),
         "reserved_seconds": gpu.time_reserved.total_seconds(),
         "allowed_seconds": gpu.total_time_allowed.total_seconds(),
         "refresh_time": str(quota.quota_refresh_time),
         "pay_to_scale_enabled": gpu.is_pay_to_scale_enabled}
    q["remaining_seconds"] = q["allowed_seconds"] - q["used_seconds"] - q["reserved_seconds"]
    owned = list(api.kernels_list(mine=True, page_size=100, sort_by="dateRun") or [])
    assert len(owned) < 100, "Paginate before claiming a complete job inventory"
    jobs, missing = [], []
    terminal = {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "DEAD"}
    for notebook in owned:
        ref = notebook.ref
        try:
            state = str(api.kernels_status(ref).status).split(".")[-1]
            if "llm-preference-" in ref or state not in terminal:
                jobs.append({"kernel": ref, "status": state})
        except Exception as error:
            code = getattr(getattr(error, "response", None), "status_code", None)
            last_run = str(getattr(notebook, "last_run_time", ""))
            missing.append({"kernel_sha256": hashlib.sha256(ref.encode()).hexdigest(),
                            "http_status": code, "last_run_time": last_run,
                            "error_type": type(error).__name__})
            assert code == 404 and last_run and last_run[:10] < start.date().isoformat(), \
                "A recent/unexplained job observation must be reconciled before launch"
    inventory = {"observed_at": datetime.now(timezone.utc).isoformat(), "quota": q,
                 "owned_notebook_count": len(owned), "project_jobs": jobs,
                 "nonterminal_jobs": [x for x in jobs if x["status"] not in terminal],
                 "unobserved_old_notebooks": missing}
    for directory in [root, aggregate]:
        json_write(directory / "resource_inventory.json", inventory)
        initial = directory / "resource_inventory_initial.json"
        if not initial.exists():
            json_write(initial, inventory)
    initial = json.loads((root / "resource_inventory_initial.json").read_text())
    assert not initial["nonterminal_jobs"] and initial["quota"]["reserved_seconds"] == 0
    assert not initial["quota"]["pay_to_scale_enabled"]
    plan = {"span": 3, "started_at": activation["started_at"], "deadline": activation["deadline"],
            "gpu_session_seconds_cap": min(43200, initial["quota"]["remaining_seconds"]),
            "quota_used_at_start": initial["quota"]["used_seconds"],
            "initial_quota_refresh_time": initial["quota"]["refresh_time"],
            "per_job_seconds_cap": 25200, "focus": activation["focus"],
            "seed": activation["seed"], "confirmation_used": False}
    for directory in [root, aggregate]:
        path = directory / "span_plan.json"
        if path.exists():
            assert json.loads(path.read_text()) == plan, "Do not reset the original resource baseline"
        else:
            json_write(path, plan)
    candidate_path = REPO / "results/span_002/research_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    assert candidate == json.loads((ROOT / "runs/span_002/research_candidate.json").read_text())
    paths = {
        "candidate": candidate_path,
        "classifier": ROOT / "runs/span_002/jobs/exposure/output/exposure/dev_predictions.npz",
        "judge_fit": ROOT / "runs/span_002/jobs/ensemble-consistent/output/cached_current/judge_predictions.npz",
        "judge_extra": ROOT / "runs/span_002/jobs/ensemble-extra/output/gemma26/judge_predictions.npz",
        "adapter": ROOT / "runs/span_002/jobs/exposure/output/exposure/adapter/adapter_model.safetensors",
        "split": ROOT / "runs/attempt_002/split_ids.json",
        "raw_train": ROOT / "raw/train.csv",
    }
    assert digest(paths["adapter"]) == candidate["adapter_sha256"]
    frame, _ = read_competition(ROOT / "raw")
    parts = split_from_ids(frame, json.loads(paths["split"].read_text()))
    frame = frame.set_index("id", drop=False)
    groups = prompt_groups(frame)
    dev_ids = set(map(int, frame.id.iloc[parts["dev"]]))
    arrays = {name: np.load(paths[name]) for name in ["classifier", "judge_fit", "judge_extra"]}
    inputs = {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()}
    for name, data in arrays.items():
        ids = data["ids"]
        assert len(ids) == len(set(ids)) and set(map(int, ids)) <= dev_ids
        np.testing.assert_array_equal(data["labels"], frame.loc[ids, CLASSES].values.argmax(1))
        for key in ["original", "aligned_swap"] if name == "classifier" else ["original"]:
            p = data[key]
            assert p.shape == (len(ids), 3) and np.isfinite(p).all() and (p >= 0).all()
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        inputs[name].update(rows=len(ids), groups=int(groups.loc[ids].nunique()))
    assert len(arrays["classifier"]["ids"]) == 5746 and set(map(int, arrays["classifier"]["ids"])) == dev_ids
    judge_ids = np.concatenate([arrays[name]["ids"] for name in ["judge_fit", "judge_extra"]])
    assert len(judge_ids) == len(set(judge_ids)) == 2048
    excluded_groups = set(groups.loc[judge_ids])
    remaining = np.array([x for x in arrays["classifier"]["ids"] if groups.loc[x] not in excluded_groups])
    rng = np.random.default_rng(plan["seed"])
    samples = {"classifier_smoke": rng.choice(arrays["classifier"]["ids"], 64, replace=False).tolist(),
               "native_profile": rng.choice(judge_ids, 64, replace=False).tolist(),
               "cascade_followup": rng.choice(remaining, 1024, replace=False).tolist()}
    sample_path = root / "sample_ids.json"
    if sample_path.exists():
        assert json.loads(sample_path.read_text()) == samples
    else:
        json_write(sample_path, samples)
    inputs["sample_ids_sha256"] = digest(sample_path)
    inputs["sample_counts"] = {name: len(ids) for name, ids in samples.items()}
    inputs["source_sha256"] = {str(path.relative_to(REPO)): digest(path) for path in
        [Path(__file__).resolve(), REPO / "GOAL.md", REPO / "src/arena/joint_data.py"]}
    for directory in [root, aggregate]:
        json_write(directory / "artifact_inventory.json", inputs)
    pages = api.competition_list_pages("llm-classification-finetuning")
    page_records = [{"type": type(page).__name__, "fields": sorted(page.__dict__)} for page in pages or []]
    json_write(root / "competition_pages_structure.json", page_records)
    print(json.dumps({"plan": plan, "quota": q, "nonterminal_jobs": inventory["nonterminal_jobs"],
                      "inputs": {key: value.get("rows") for key, value in inputs.items() if isinstance(value, dict)},
                      "competition_pages_structure": page_records}, indent=2))


if __name__ == "__main__":
    main()
