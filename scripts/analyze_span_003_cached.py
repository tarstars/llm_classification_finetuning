"""Evaluate the predeclared H39/H43 cached experiments without requesting inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_cascade import RecipeSet, nested_recipes, select_recipes
from arena.span_heads import normalize, metrics, paired_group_gain, loss_rows, order_diagnostics, temperature
from arena.preference_diagnostics import loss_decomposition


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = time.monotonic()
    root, aggregate = ROOT / "runs/span_003", REPO / "results/span_003"
    config = json.loads((REPO / "configs/span_003.json").read_text())
    inventory = json.loads((aggregate / "artifact_inventory.json").read_text())
    caches = {}
    for name in ["classifier", "judge_fit", "judge_extra"]:
        path = Path(inventory[name]["path"])
        assert digest(path) == inventory[name]["sha256"]
        caches[name] = np.load(path)
    ids = np.concatenate([caches[name]["ids"] for name in ["judge_fit", "judge_extra"]])
    y = np.concatenate([caches[name]["labels"] for name in ["judge_fit", "judge_extra"]])
    q = normalize(np.concatenate([caches[name]["original"] for name in ["judge_fit", "judge_extra"]]))
    assert len(ids) == len(set(ids)) == 2048
    reference = caches["classifier"]
    lookup = {int(rid): index for index, rid in enumerate(reference["ids"])}
    selected = np.array([lookup[int(rid)] for rid in ids])
    np.testing.assert_array_equal(y, reference["labels"][selected])
    original, aligned = reference["original"][selected], reference["aligned_swap"][selected]
    p = normalize((original + aligned) / 2)
    frame, _ = read_competition(ROOT / "raw")
    frame = frame.set_index("id", drop=False)
    np.testing.assert_array_equal(y, frame.loc[ids, CLASSES].values.argmax(1))
    groups = prompt_groups(frame.loc[ids]).to_numpy().astype(str)
    u = np.array([int.from_bytes(hashlib.sha256(f"{config['seed']}:{rid}".encode()).digest()[:8], "big") / 2**64
                  for rid in ids])
    outputs, masks, folds = nested_recipes(p, q, y, groups, u, config,
        progress=lambda fold, choices: print("Finished outer fold", fold, choices, flush=True))
    models = {name: {**metrics(y, probability), "loss_decomposition": loss_decomposition(y, probability)}
              for name, probability in outputs.items()}
    contrasts = {}
    for baseline, candidate in [("classifier", name) for name in outputs if name != "classifier"] + [
        ("selected_random", "selected_entropy"), ("vector", "selected_entropy"),
        ("vector", "selected_blend"), ("vector_tie", "conditional"), ("judge", "vector")]:
        contrast = paired_group_gain(y, outputs[baseline], outputs[candidate], groups, config)
        delta = loss_rows(y, outputs[baseline]) - loss_rows(y, outputs[candidate])
        contrast["weighted_class_contributions"] = [float(delta[y == i].sum() / len(y)) for i in range(3)]
        assert np.isclose(sum(contrast["weighted_class_contributions"]), contrast["gain"])
        contrasts[f"{candidate}_vs_{baseline}"] = contrast
    routed = {name: {"rows": int(mask.sum()), "fraction": float(mask.mean()),
                    "human_class_counts": np.bincount(y[mask], minlength=3).tolist()}
              for name, mask in masks.items()}
    selected_recipes, scores = select_recipes(p, q, y, groups, u, config)
    fit = RecipeSet(config).fit(p, q, y, groups, u)
    reconstructed = RecipeSet.from_dict(json.loads(json.dumps(fit.to_dict())), config)
    a, _ = fit.predict(p, q, u)
    b, _ = reconstructed.predict(p, q, u)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "config": config,
              "fit_rows": len(y), "fit_groups": len(set(groups)),
              "selected": selected_recipes, "inner_log_loss": scores,
              "parameters": fit.to_dict(), "input_sha256": {name: inventory[name]["sha256"] for name in caches}}
    destination = root / "frozen_cached_recipes.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert {k: v for k, v in previous.items() if k != "frozen_at"} == {k: v for k, v in frozen.items() if k != "frozen_at"}
        frozen = previous
    else:
        json_write(destination, frozen)
    json_write(aggregate / "frozen_cached_recipes.json", frozen)
    packing = json.loads((REPO / "results/span_002/packing_diagnostics.json").read_text())
    prior = np.array(packing["constant_baselines"]["training_prior"])
    constants = {"training_prior": metrics(y, np.tile(prior, (len(y), 1))),
                 "majority_clipped_1e-12": metrics(y, np.tile(np.eye(3)[prior.argmax()], (len(y), 1))),
                 "uniform": metrics(y, np.full((len(y), 3), 1/3))}
    manifest = json.loads((REPO / "results/span_002/research_candidate.json").read_text())
    report = {"span": 3, "stage": "cached", "rows": len(y), "groups": len(set(groups)),
              "config": config, "models": models, "contrasts": contrasts, "routing": routed, "folds": folds,
              "constant_controls": constants,
              "historical_full_dev_fitted_temperature_reference": metrics(y, temperature(p, manifest["temperature"])),
              "classifier_raw_order": order_diagnostics(original, aligned),
              "retention": packing["cohorts"]["ensemble_combined"],
              "selection_limits": "Previously inspected development; nested grouped selection; conditional bootstrap intervals; confirmation unused.",
              "native_order_limit": "Judge is single-order; no native both-order invariance measured.",
              "hard_majority_note": "Exact hard-majority loss is infinite on errors. Finite diagnostic uses1e-12 clipping; deterministic argmax ties choose A.",
              "source_sha256": {name: digest(REPO / name) for name in [
                  "scripts/analyze_span_003_cached.py", "src/arena/span_cascade.py", "src/arena/span_calibration.py",
                  "src/arena/span_heads.py", "src/arena/preference_diagnostics.py", "configs/span_003.json", "docs/SPAN_003_PROTOCOL.md"]},
              "input_sha256": {name: inventory[name]["sha256"] for name in caches},
              "cpu_elapsed_seconds": time.monotonic() - started,
              "frozen_recipe_sha256": digest(destination)}
    for name in ["selected_entropy_vs_classifier", "vector_vs_classifier", "selected_blend_vs_vector", "conditional_vs_vector_tie"]:
        c = contrasts[name]
        c["passes_practical_positive_gate"] = c["gain"] >= config["cheap_useful_gain"] and c["interval_95"][0] > 0
    private = root / "cached_evaluated_predictions.npz"
    np.savez_compressed(private, ids=ids, labels=y, groups=groups, random_scores=u, **outputs,
                        **{name + "_mask": value for name, value in masks.items()})
    report["private_predictions_sha256"] = digest(private)
    json_write(root / "cached_results.json", report)
    json_write(aggregate / "cached_results.json", report)
    print(json.dumps({"models": {name: m["log_loss"] for name, m in models.items()},
                      "primary_cascade": contrasts["selected_entropy_vs_classifier"],
                      "primary_blending": contrasts["selected_blend_vs_vector"]}, indent=2))


if __name__ == "__main__":
    main()
