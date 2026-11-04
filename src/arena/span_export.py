"""Offline frozen-classifier reload verification and deadline-bounded prediction."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, TrainingArguments

from .data import CLASSES, TEXT_COLUMNS, json_write, parse_turns
from .joint_data import JointTokenizer, PreferenceDataset, PreferenceCollator, read_competition
from .joint_infer import load_inference_model
from .joint_train import PreferenceTrainer, probabilities, temper


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "model", "adapter", "manifest", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "predict"], required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--profile-rate", type=float, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    def expired(*unused):
        raise TimeoutError("Frozen inference deadline reached; no partial submission accepted")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(args.deadline - time.time())))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    config = manifest["config"]
    assert manifest["class_order"] == CLASSES and not manifest["confirmation_used_for_selection"]
    assert digest(args.adapter / "adapter_model.safetensors") == manifest["adapter_sha256"]
    assert digest(Path(__file__).with_name("joint_data.py")) == config["preprocessing_sha256"]
    assert digest(args.adapter / "tokenizer.json") == manifest["tokenizer_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.mode == "smoke":
        reference = np.load(args.reference)
        frame, _ = read_competition(args.raw)
        selected = frame.set_index("id", drop=False).loc[reference["ids"]]
    else:
        selected = pd.read_csv(args.raw / "test.csv")
        for name in TEXT_COLUMNS:
            selected[name] = selected[name].map(parse_turns)
        assert args.profile_rate > 0
        projected = len(selected) * args.profile_rate * 1.2 + 600
        assert projected < args.deadline - time.time(), "Measured hidden-row projection exceeds the frozen ceiling"
    original, swapped, diagnostics = JointTokenizer(tokenizer, config["max_length"]).encode_frame(selected)
    model = load_inference_model(args.model, args.adapter, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    assert torch.cuda.device_count() == 2
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    arguments = TrainingArguments(output_dir=str(args.out / "evaluation_tmp"),
        per_device_eval_batch_size=config["eval_batch_size"], fp16=not bf16, bf16=bf16,
        report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=True, disable_tqdm=True)
    trainer = PreferenceTrainer(model=model, args=arguments, processing_class=tokenizer,
                                data_collator=PreferenceCollator(tokenizer))
    views = []
    prediction_started = time.monotonic()
    for force in [False, True]:
        logits = trainer.predict(PreferenceDataset(original, swapped, force_swap=force)).predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        views.append(probabilities(logits))
    prediction_seconds = time.monotonic() - prediction_started
    average = (views[0] + views[1][:, [1, 0, 2]]) / 2
    calibrated = temper(average, manifest["temperature"])
    if trainer.is_world_process_zero():
        assert np.isfinite(calibrated).all() and (calibrated >= 0).all()
        np.testing.assert_allclose(calibrated.sum(1), 1, atol=1e-6)
        summary = {"mode": args.mode, "rows": len(selected), "prediction_seconds": prediction_seconds,
                   "seconds_per_row": prediction_seconds / len(selected), "diagnostics": diagnostics,
                   "program_seconds": time.monotonic() - started,
                   "adapter_sha256": manifest["adapter_sha256"], "temperature": manifest["temperature"],
                   "tokenizer_sha256": manifest["tokenizer_sha256"], "class_order": CLASSES,
                   "base_config_sha256": digest(args.model / "config.json"),
                   "peak_gpu_bytes_rank0": torch.cuda.max_memory_allocated(),
                   "software": {name: importlib.metadata.version(name) for name in
                                ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]}}
        if args.mode == "smoke":
            checks = {}
            for name, observed, expected in [
                ("original", views[0], reference["original"]),
                ("aligned_swap", views[1][:, [1, 0, 2]], reference["aligned_swap"]),
                ("average", average, reference["average"]),
                ("calibrated", calibrated, reference["calibrated"])]:
                delta = np.abs(observed - expected)
                checks[name] = {"mean": float(delta.mean()), "maximum": float(delta.max())}
            summary["identity"] = checks
            summary["passed"] = checks["average"]["mean"] <= .002 and checks["average"]["maximum"] <= .03
            json_write(args.out / "smoke_summary.json", summary)
            np.savez_compressed(args.out / "smoke_predictions.npz", ids=selected.id.to_numpy(),
                                original=views[0], aligned_swap=views[1][:, [1, 0, 2]], probabilities=calibrated)
            assert summary["passed"], "Offline export failed the frozen identity guard"
        else:
            submission = pd.DataFrame(calibrated, columns=CLASSES)
            submission.insert(0, "id", selected.id.to_numpy())
            assert submission.id.is_unique and submission.id.tolist() == selected.id.tolist()
            assert submission.columns.tolist() == pd.read_csv(args.raw / "sample_submission.csv").columns.tolist()
            submission.to_csv(args.out / "submission.csv", index=False)
            summary["probabilities_valid"] = True
            json_write(args.out / "prediction_summary.json", summary)
        print(json.dumps(summary), flush=True)
    trainer.accelerator.wait_for_everyone()
    signal.alarm(0)


if __name__ == "__main__":
    main()
