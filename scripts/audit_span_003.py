"""Independently verify v3 artifacts; final mode also requires fresh terminal service state."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import nbformat
import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_calibration import grouped_folds
from arena.span_cascade import RecipeSet, candidate_groups

AGG = REPO / "results/span_003"
HEAVY = ROOT / "runs/span_003"
ROLES = ["classifier_smoke", "native_profile", "classifier_predict", "native_profile_retry"]
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "DEAD"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid(p):
    p = np.asarray(p, float)
    assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1 + 1e-8))
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-8)
    clipped = np.clip(p, 1e-12, 1)
    return clipped / clipped.sum(1, keepdims=True)


def losses(y, p):
    return -np.log(valid(p)[np.arange(len(y)), y])


def calibrated(p, temperature):
    logits = np.log(valid(p)) / temperature
    logits -= logits.max(1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(1, keepdims=True)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def embedded_modules(path):
    modules = {}
    for cell in nbformat.read(path, as_version=4).cells:
        if cell.cell_type != 'code':
            continue
        for call in ast.walk(ast.parse(cell.source)):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != 'write_text':
                continue
            target = call.func.value
            if (isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant)
                    and isinstance(target.right.value, str) and target.right.value.endswith('.py')
                    and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                modules[Path(target.right.value).name] = hashlib.sha256(call.args[0].value.encode()).hexdigest()
    return modules


def local_audit():
    plan, initial = read(HEAVY / "span_plan.json"), read(HEAVY / "resource_inventory_initial.json")
    activation, config = read(AGG / "activation.json"), read(REPO / "configs/span_003.json")
    assert plan == read(AGG / "span_plan.json")
    assert plan["started_at"] == activation["started_at"] == "2026-11-04T07:42:17+00:00"
    assert plan["deadline"] == activation["deadline"] == "2026-11-06T07:42:17+00:00"
    assert datetime.fromtimestamp(activation["goal_service_created_at"], timezone.utc).isoformat() == plan["started_at"]
    close(plan["gpu_session_seconds_cap"], min(43200, initial["quota"]["remaining_seconds"]))
    close(sum(config["job_seconds_caps"].values()) + config["recovery_reserve_seconds"]
          + config["completed_prediction_preview_charge_seconds"], plan["gpu_session_seconds_cap"])
    inventory = read(AGG / "artifact_inventory.json")
    input_hashes = {}
    for name, value in inventory.items():
        if isinstance(value, dict) and {"path", "sha256"} <= set(value):
            assert digest(Path(value["path"])) == value["sha256"], name
            input_hashes[name] = value["sha256"]
    sample_path = HEAVY / "sample_ids.json"
    assert digest(sample_path) == inventory["sample_ids_sha256"]
    samples = read(sample_path)
    split = read(inventory["split"]["path"])
    frame, _ = read_competition(ROOT / "raw")
    frame = frame.set_index("id", drop=False)
    report = read(AGG / "cached_results.json")
    cfg = report["config"]
    assert json.loads(subprocess.check_output(['git', 'show', '3cd8407:configs/span_003.json'], cwd=REPO)) == cfg
    original_inventory = subprocess.check_output(['git', 'show', '3cd8407:results/span_003/artifact_inventory.json'], cwd=REPO)
    assert hashlib.sha256(original_inventory).hexdigest() == digest(AGG / 'artifact_inventory.json')
    for name, sha in report["source_sha256"].items():
        blob = subprocess.check_output(["git", "show", "2536daf:" + name], cwd=REPO)
        assert hashlib.sha256(blob).hexdigest() == sha, name
    admin = {"job_seconds_caps", "recovery_reserve_seconds", "completed_prediction_preview_charge_seconds"}
    assert {k: v for k, v in cfg.items() if k not in admin} == {k: v for k, v in config.items() if k not in admin}
    raw_classifier = np.load(inventory["classifier"]["path"])
    judge = [np.load(inventory[name]["path"]) for name in ["judge_fit", "judge_extra"]]
    ids = np.concatenate([data["ids"] for data in judge])
    y = frame.loc[ids, CLASSES].values.argmax(1)
    q = valid(np.concatenate([data["original"] for data in judge]))
    groups = prompt_groups(frame.loc[ids]).to_numpy().astype(str)
    lookup = {int(rid): index for index, rid in enumerate(raw_classifier["ids"])}
    positions = [lookup[int(rid)] for rid in ids]
    original, swapped = raw_classifier["original"][positions], raw_classifier["aligned_swap"][positions]
    p = valid((original + swapped) / 2)
    assert len(ids) == len(set(ids)) == 2048 and set(map(int, ids)) <= set(split["dev"])
    private = HEAVY / "cached_evaluated_predictions.npz"
    assert digest(private) == report["private_predictions_sha256"]
    saved = np.load(private)
    for name, expected in [("ids", ids), ("labels", y), ("groups", groups)]:
        np.testing.assert_array_equal(saved[name], expected)
    random = np.array([int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{rid}".encode()).digest()[:8], "big") / 2**64 for rid in ids])
    np.testing.assert_array_equal(saved["random_scores"], random)
    reconstructed = 0
    for index, (train, val) in enumerate(grouped_folds(y, groups, cfg["outer_group_folds"], cfg["seed"])):
        record = report["folds"][index]
        assert not set(groups[train]) & set(groups[val])
        assert (record["train_rows"], record["validation_rows"]) == (len(train), len(val))
        for selected, choices in candidate_groups(cfg).items():
            assert record["selected"][selected] == min(choices, key=record["inner_log_loss"].get)
        model = RecipeSet.from_dict(record, cfg)
        predicted, masks = model.predict(p[val], q[val], random[val])
        for selected, chosen in record["selected"].items():
            predicted[selected] = predicted[chosen]
            if chosen in masks:
                masks[selected] = masks[chosen]
        for name, probability in predicted.items():
            np.testing.assert_allclose(probability, saved[name][val], atol=1e-12, rtol=0)
            reconstructed += 1
        for name, mask in masks.items():
            np.testing.assert_array_equal(mask, saved[name + "_mask"][val])
    for name, metric in report["models"].items():
        probabilities = valid(saved[name])
        row_loss = losses(y, probabilities)
        close(row_loss.mean(), metric["log_loss"])
        close((probabilities.argmax(1) == y).mean(), metric["accuracy"])
        close(np.mean(np.sum((probabilities - np.eye(3)[y])**2, axis=1)), metric["brier"])
        matrix = np.zeros((3, 3), int)
        np.add.at(matrix, (y, probabilities.argmax(1)), 1)
        np.testing.assert_array_equal(matrix, metric["confusion_matrix"])
        close(np.diag(matrix) / matrix.sum(1), metric["class_recall"])
        close([row_loss[y == label].mean() for label in range(3)], metric["class_loss"])
        f1 = 2 * np.diag(matrix) / np.maximum(matrix.sum(0) + matrix.sum(1), 1)
        close(f1.mean(), metric["macro_f1"])
        tie_loss = np.where(y == 2, -np.log(probabilities[:, 2]), -np.log(probabilities[:, :2].sum(1)))
        channel = metric["loss_decomposition"]
        close(tie_loss.mean(), channel["binary_tie_log_loss"])
        close(row_loss.mean() - tie_loss.mean(), channel["conditional_winner_contribution"])
    excluded = set(split['dev']) | set(split['confirmation'])
    train_ids = frame.index[~frame.index.isin(excluded)]
    assert len(train_ids) == 45985
    training_counts = frame.loc[train_ids, CLASSES].sum().to_numpy(float)
    prior = training_counts / training_counts.sum()
    constant = {'training_prior': np.tile(prior, (len(y), 1)),
                'majority_clipped_1e-12': np.tile(np.eye(3)[prior.argmax()], (len(y), 1)),
                'uniform': np.full((len(y), 3), 1/3)}
    for name, probability in constant.items():
        close(losses(y, probability).mean(), report['constant_controls'][name]['log_loss'])
        close((probability.argmax(1) == y).mean(), report['constant_controls'][name]['accuracy'])
    packing_path = REPO / 'results/span_002/packing_diagnostics.json'
    original_packing = subprocess.check_output(['git', 'show', '2536daf:results/span_002/packing_diagnostics.json'], cwd=REPO)
    assert hashlib.sha256(original_packing).hexdigest() == digest(packing_path)
    assert report['retention'] == read(packing_path)['cohorts']['ensemble_combined']
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(cfg["seed"]).integers(0, len(counts), size=(2000, len(counts)))
    denominators = counts[draws].sum(1)
    for name, contrast in report["contrasts"].items():
        candidate, baseline = name.split("_vs_")
        delta = losses(y, saved[baseline]) - losses(y, saved[candidate])
        values = np.bincount(inverse, weights=delta)
        interval = np.quantile(values[draws].sum(1) / denominators, [.025, .975])
        close(delta.mean(), contrast["gain"])
        close(interval, contrast["interval_95"])
        close([delta[y == label].sum() / len(y) for label in range(3)], contrast["weighted_class_contributions"])
        assert contrast["rows"] == len(y) and contrast["groups"] == len(counts) and contrast["replicates"] == 2000
    for name, metric in report["routing"].items():
        mask = saved[name + "_mask"]
        assert mask.dtype == bool and int(mask.sum()) == metric["rows"]
        close(mask.mean(), metric["fraction"])
        np.testing.assert_array_equal(np.bincount(y[mask], minlength=3), metric["human_class_counts"])
        np.testing.assert_allclose(saved[name][~mask], saved["classifier"][~mask], atol=1e-12)
    close((original.argmax(1) != swapped.argmax(1)).mean(), report["classifier_raw_order"]["decision_disagreement"])
    frozen = read(HEAVY / "frozen_cached_recipes.json")
    assert frozen == read(AGG / "frozen_cached_recipes.json") and frozen["config"] == cfg
    assert digest(HEAVY / "frozen_cached_recipes.json") == report["frozen_recipe_sha256"]
    assert datetime.fromisoformat(frozen['frozen_at']) < datetime.fromisoformat(read(HEAVY / 'jobs/native_profile/push_intent.json')['utc'])
    assert not set(groups) & set(prompt_groups(frame.loc[samples["cascade_followup"]]))
    assert set(samples["cascade_followup"]) <= set(split["dev"]) and len(samples["cascade_followup"]) == 1024
    assert not (HEAVY / "jobs/cascade_followup/push_intent.json").exists()
    for name, number in [("classifier_smoke", 64), ("native_profile", 64)]:
        assert len(samples[name]) == len(set(samples[name])) == number
        assert set(samples[name]) <= set(split["dev"])

    manifest = read(REPO / "results/span_002/research_candidate.json")
    assert manifest == read(ROOT / "runs/span_002/research_candidate.json")
    weight = ROOT / "runs/span_002/jobs/exposure/output/exposure/adapter/adapter_model.safetensors"
    assert digest(weight) == manifest["adapter_sha256"]
    old = read(REPO / "results/span_002/terminal_audit.json")["preserved_input_hashes"]
    assert digest(ROOT / "runs/attempt_002/model_selection.json") == old["selection"]
    assert digest(ROOT / "runs/attempt_002/jobs/qwen4b1024/output/joint_run/adapter/adapter_model.safetensors") == old["adapter"]
    directory = HEAVY / "jobs/classifier_smoke/output"
    reference = np.load(directory / "reference.npz")
    observed = np.load(directory / "smoke/smoke_predictions.npz")
    summary = read(directory / "smoke/smoke_summary.json")
    np.testing.assert_array_equal(reference["ids"], samples["classifier_smoke"])
    np.testing.assert_array_equal(observed["ids"], reference["ids"])
    source_indices = [lookup[int(rid)] for rid in reference['ids']]
    close(reference['original'], raw_classifier['original'][source_indices])
    close(reference['aligned_swap'], raw_classifier['aligned_swap'][source_indices])
    avg = (observed["original"] + observed["aligned_swap"]) / 2
    np.testing.assert_allclose(calibrated(avg, manifest["temperature"]), observed["probabilities"], atol=1e-12)
    for name, a, b in [("original", observed["original"], reference["original"]),
                       ("aligned_swap", observed["aligned_swap"], reference["aligned_swap"]),
                       ("average", avg, reference["average"]),
                       ("calibrated", observed["probabilities"], reference["calibrated"])]:
        difference = np.abs(a - b)
        close([difference.mean(), difference.max()], [summary["identity"][name]["mean"], summary["identity"][name]["maximum"]])
    assert summary["passed"] and summary["identity"]["average"]["mean"] <= .002 and summary["identity"]["average"]["maximum"] <= .03
    verification = read(AGG / "classifier_verification.json")
    close(losses(y, calibrated(p, manifest['temperature'])).mean(), report['historical_full_dev_fitted_temperature_reference']['log_loss'])
    test = pd.read_csv(ROOT / "raw/test.csv", usecols=["id"])
    columns = pd.read_csv(ROOT / "raw/sample_submission.csv").columns.tolist()
    for role, relative in [("classifier_smoke", "visible/submission.csv"), ("classifier_predict", "submission.csv")]:
        path = HEAVY / "jobs" / role / "output" / relative
        csv = pd.read_csv(path)
        assert csv.columns.tolist() == columns and csv.id.tolist() == test.id.tolist() and csv.id.is_unique
        valid(csv[CLASSES].to_numpy())
        assert digest(path) == verification["independently_validated_previews"][role]["csv_sha256"]
    embedded = {}
    for role in ROLES:
        notebook = HEAVY / 'notebooks' / role / 'job.ipynb'
        bundle = read(AGG / (role + '_bundle.json'))
        assert digest(notebook) == bundle['notebook_sha256']
        actual = embedded_modules(notebook)
        if role.startswith('classifier'):
            for source, sha in bundle['source_sha256'].items():
                assert actual[Path(source).name] == sha and digest(REPO / source) == sha
        else:
            assert actual['judge.py'] == bundle['judge_source_sha256']
            assert actual['judge_runtime.py'] == bundle['native_source_sha256']
        embedded[role] = {'notebook_sha256': digest(notebook), 'modules': actual}

    native = read(AGG / "native_profile_results.json")
    output = HEAVY / "jobs/native_profile_retry/output"
    native_ref, native_p = np.load(output / "native_reference.npz"), np.load(output / "native/judge_predictions.npz")
    np.testing.assert_array_equal(native_ref["ids"], samples["native_profile"])
    np.testing.assert_array_equal(native_p["ids"], native_ref["ids"])
    np.testing.assert_array_equal(native_p['labels'], frame.loc[native_p['ids'], CLASSES].values.argmax(1))
    judge_lookup = {int(rid): index for index, rid in enumerate(ids)}
    close(native_ref["original"], q[[judge_lookup[int(rid)] for rid in native_ref["ids"]]])
    delta = np.abs(native_p["original"] - native_ref["original"])
    close([delta.mean(), delta.max()], [native["gate"]["mean_probability_difference"], native["gate"]["max_probability_difference"]])
    valid(native_p["original"])
    assert not native["numerical_identity_passed"] and (delta.mean() > .01 or delta.max() > .1)
    assert native['gate']['mean_limit'] == cfg['native_identity_limits']['mean'] == .01
    assert native['gate']['max_limit'] == cfg['native_identity_limits']['maximum'] == .1
    diagnostic = native['loss_diagnostic']
    assert diagnostic['source_sha256'] == digest(REPO / 'scripts/diagnose_span_003_native.py')
    assert diagnostic['rows'] == 64 and not diagnostic['new_parameters_fitted']
    native_groups = prompt_groups(frame.loc[native_p['ids']]).to_numpy()
    _, native_inverse = np.unique(native_groups, return_inverse=True)
    native_counts = np.bincount(native_inverse)
    native_draws = np.random.default_rng(cfg['seed']).integers(0, len(native_counts), size=(2000, len(native_counts)))
    calibrator = RecipeSet.from_dict(frozen['parameters'], cfg).fits[1]
    for suffix, previous, observed in [
        ('raw', native_ref['original'], native_p['original']),
        ('fixed_calibration', calibrator.predict_proba(native_ref['original']), calibrator.predict_proba(native_p['original'])),
    ]:
        old_loss, new_loss = losses(native_p['labels'], previous), losses(native_p['labels'], observed)
        close(old_loss.mean(), diagnostic['old_' + suffix]['log_loss'])
        close(new_loss.mean(), diagnostic['current_' + suffix]['log_loss'])
        effect = diagnostic[suffix + '_loss_effect']
        sums = np.bincount(native_inverse, weights=old_loss-new_loss)
        close((old_loss-new_loss).mean(), effect['gain'])
        close(np.quantile(sums[native_draws].sum(1)/native_counts[native_draws].sum(1), [.025, .975]), effect['interval_95'])
    close((native_ref['original'].argmax(1) != native_p['original'].argmax(1)).mean(), diagnostic['argmax_disagreement'])
    assert native["hardware_passed"] and native["hardware"]["devices"].count("T4") == 2
    assert "not found" not in native["hardware"]["ldd"]
    assert len(native["gate"]["sampled_gpu_used_peak_mib"]) == 2 and min(native["gate"]["sampled_gpu_used_peak_mib"]) > 1024
    portable = output / "offline_native_runtime/manifest.json"
    assert digest(portable) == native["hardware"]["portable_manifest_sha256"]
    files = read(portable)["files"]
    inspection = read(HEAVY / "native_parent_inspect/official_runtime_inspection.json")
    wheel = HEAVY / "native_parent_inspect/official_runtime.whl"
    assert digest(wheel) == inspection["wheel_sha256"]
    with zipfile.ZipFile(wheel) as archive:
        blob = archive.read(inspection["library_path"])
    assert blob[:6].hex() == "7f454c460201" and len(blob) == inspection["library_bytes"] == 728800
    assert hashlib.sha256(blob).hexdigest() == inspection["library_sha256"]
    assert files["cuda12/" + inspection["library_path"]]["sha256"] == inspection["library_sha256"]
    assert files["licenses/llama.cpp-LICENSE"]["sha256"] == digest(HEAVY / "license_files/llama.cpp-LICENSE")
    assert len([name for name in files if "license" in name.lower()]) >= 3
    charges = []
    for role in ROLES:
        directory = HEAVY / "jobs" / role
        before, after = read(directory / "push_intent.json")["quota_before"], read(directory / "quota.json")
        charges.append({"role": role, "seconds": after["used_seconds"] - before["used_seconds"]})
    close(sum(row["seconds"] for row in charges), 738.297)
    intent, receipt = read(HEAVY / "official/submission_intent.json"), read(HEAVY / "official/receipt.json")
    assert receipt == read(AGG / "receipt.json") and receipt["ref"] == 57979825
    assert digest(REPO / "scripts/submit_span_003.py") == intent["controller_sha256"]
    assert digest(REPO / "results/span_002/research_candidate.json") == intent["manifest_sha256"]
    close(intent["observed_span_charged_before"], sum(row["seconds"] for row in charges))
    assert intent["conservative_span_seconds_bound"] <= plan["gpu_session_seconds_cap"]
    for source in read(AGG / 'official_sources.json')['pages']:
        assert digest(HEAVY / ('page_' + source['name'].replace('/', '_') + '.txt')) == source['sha256']
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": True,
              "scope": "Local artifacts only; official terminal and fresh remote checks are separate.",
              "original_start": plan["started_at"], "original_deadline": plan["deadline"],
              "input_sha256_verified": input_hashes, "cached_source_revision": "2536daf",
              "cached_rows": len(y), "cached_groups": len(counts), "reconstructed_fold_predictions": reconstructed,
              "training_only_constant_baselines_recomputed": list(constant),
              "unchanged_verified_packing_cohort_rows": report['retention']['rows'],
              "embedded_execution_modules": embedded,
              "independently_recomputed_models": len(report["models"]), "independently_recomputed_contrasts": len(report["contrasts"]),
              "unused_new_group_rows": 1024, "confirmation_used": False,
              "classifier_identity_passed": True, "visible_csvs_independently_verified": 2,
              "native_hardware_passed": True, "native_identity_passed": False,
              "native_manifest_files": len(files), "native_runtime_matches_published_wheel": True,
              "native_fixed_calibration_loss_diagnostic_recomputed": True,
              "completed_job_charges": charges, "completed_gpu_session_seconds": sum(row["seconds"] for row in charges),
              "accepted_submission": receipt["ref"], "controller_source_matches_intent": True,
              "audit_program_sha256": digest(Path(__file__).resolve()),
              "audit_software": {name: importlib.metadata.version(name) for name in
                                 ['numpy', 'scipy', 'scikit-learn', 'nbformat', 'kaggle', 'requests']},
              "private_probability_sha256": digest(private)}
    json_write(AGG / "local_audit.json", result)
    json_write(HEAVY / "local_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    started = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": False,
               "stage": "verification_running", "audit_program_sha256": digest(Path(__file__).resolve())}
    for directory in [AGG, HEAVY]:
        json_write(directory / 'local_audit.json', started)
        if args.final:
            json_write(directory / 'terminal_audit.json', started)
    try:
        local = local_audit()
    except Exception as error:
        for directory in [AGG, HEAVY]:
            json_write(directory / 'local_audit.json', {**started, 'stage': 'verification_failed', 'error_type': type(error).__name__})
        raise
    if not args.final:
        print(json.dumps(local, indent=2))
        return
    token = Path.home() / ".kaggle/access_token"
    if token.exists():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    send = requests.Session.send
    def bounded_send(session, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, 45)
        return send(session, request, **kwargs)
    requests.Session.send = bounded_send
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    def query(call):
        for attempt in range(3):
            try:
                return call()
            except requests.exceptions.RequestException as error:
                code = getattr(getattr(error, "response", None), "status_code", None)
                if code in {400, 401, 403, 404} or attempt == 2:
                    raise
                time.sleep(3*(attempt+1))
    submissions = query(lambda: api.competition_submissions("llm-classification-finetuning")) or []
    target = [item for item in submissions if str(item.ref) == "57979825"]
    assert len(target) == 1
    status = str(target[0].status).split(".")[-1]
    assert status in {"COMPLETE", "ERROR", "FAILED", "INVALID", "CANCELED", "CANCELLED"}, "Official evaluation is not terminal"
    recorded = read(AGG / "official_result.json")
    assert recorded["status"].split(".")[-1] == status
    if status == "COMPLETE":
        close(float(recorded["public_score"]), float(target[0].public_score))
    def check_job(role):
        directory = HEAVY / "jobs" / role
        push, intent = read(directory / "push.json"), read(directory / "push_intent.json")
        exact = push["ref"].removeprefix("/code/") + "/" + str(push["version_number"])
        state = str(query(lambda: api.kernels_status(exact)).status).split(".")[-1]
        assert state in TERMINAL
        source = HEAVY / "notebooks" / role / "job.ipynb"
        assert digest(source) == intent["notebook_sha256"]
        destination = HEAVY / "audited_sources" / ("final_" + role)
        destination.mkdir(parents=True, exist_ok=True)
        query(lambda: api.kernels_pull(exact, path=str(destination), metadata=True, quiet=True))
        files = list(destination.glob("*.ipynb"))
        assert len(files) == 1
        cells = lambda p: [(cell.cell_type, cell.source) for cell in nbformat.read(p, as_version=4).cells]
        assert cells(files[0]) == cells(source)
        assert read(destination / "kernel-metadata.json")["enable_internet"] is False
        return {"role": role, "kernel_version": exact, "status": state,
                "notebook_sha256": digest(source), "remote_source_matches": True, "internet_disabled": True}
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = list(executor.map(check_job, ROLES))
    owned = list(query(lambda: api.kernels_list(mine=True, page_size=100, sort_by="dateRun")) or [])
    assert len(owned) < 100
    def check_owned(item):
        try:
            current = str(query(lambda: api.kernels_status(item.ref)).status).split(".")[-1]
            assert current in TERMINAL
            return None
        except requests.exceptions.HTTPError as error:
            code = getattr(getattr(error, "response", None), "status_code", None)
            date = str(getattr(item, "last_run_time", ""))
            assert code == 404 and date and date[:10] < "2026-11-04"
            return {"kernel_sha256": hashlib.sha256(item.ref.encode()).hexdigest(), "http_status": code, "last_run_time": date}
    with ThreadPoolExecutor(max_workers=3) as executor:
        unknown = [value for value in executor.map(check_owned, owned) if value is not None]
    response = query(api.quota_view)
    q = {"used_seconds": response.gpu_quota.time_used.total_seconds(),
         "reserved_seconds": response.gpu_quota.time_reserved.total_seconds(),
         "allowed_seconds": response.gpu_quota.total_time_allowed.total_seconds(),
         "refresh_time": str(response.quota_refresh_time), "paid_scaling": response.gpu_quota.is_pay_to_scale_enabled}
    plan, intent = read(HEAVY / "span_plan.json"), read(AGG / "submission_intent.json")
    charged = q["used_seconds"] - plan["quota_used_at_start"]
    conservative = max(charged, intent["conservative_span_seconds_bound"])
    assert q["reserved_seconds"] == 0 and not q["paid_scaling"]
    assert q["refresh_time"] == plan["initial_quota_refresh_time"]
    assert conservative <= plan["gpu_session_seconds_cap"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(plan["deadline"])
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": True,
              "jobs": jobs, "all_recorded_jobs_terminal": True,
              "official_submission": {"ref": 57979825, "status": status, "public_score": recorded.get("public_score")},
              "owned_notebook_count": len(owned), "nonterminal_owned_notebooks": [], "unobserved_old_notebooks": unknown,
              "quota": q, "measured_span_quota_charge_seconds": charged,
              "conservative_span_compute_seconds_bound": conservative,
              "gpu_session_seconds_cap": plan["gpu_session_seconds_cap"], "within_original_compute_cap": True,
              "original_start": plan["started_at"], "original_deadline": plan["deadline"], "within_original_deadline": True,
              "no_paid_scaling": True, "no_quota_refresh_used": True, "no_optional_followup_executed": True,
              "local_audit_sha256": digest(AGG / "local_audit.json"),
              "accounting_limit": "Hidden code-competition GPU time may not be exposed in quota. Actual account charge and conservative source-bounded hidden allowance are reported separately; receipt elapsed is not GPU time."}
    json_write(AGG / "terminal_audit.json", result)
    json_write(HEAVY / "terminal_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
