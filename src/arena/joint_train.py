"""QLoRA joint preference classification, with resumable checkpoints and swap evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
    BitsAndBytesConfig, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from .data import CLASSES, json_write
from .joint_data import (JointTokenizer, PreferenceDataset, PreferenceCollator,
                         read_competition, split_from_ids)


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(labels, p):
    assert p.shape == (len(labels), 3) and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    return {"log_loss": float(log_loss(labels, p, labels=[0,1,2])),
        "accuracy": float(accuracy_score(labels, p.argmax(1))),
        "balanced_accuracy": float(balanced_accuracy_score(labels, p.argmax(1))),
        "macro_f1": float(f1_score(labels, p.argmax(1), labels=[0,1,2], average="macro", zero_division=0)),
        "brier": float(np.mean(np.sum((p-np.eye(3)[labels])**2,axis=1)))}


def temper(p, temperature):
    return probabilities(np.log(np.clip(p, 1e-12, 1)) / temperature)


class PreferenceTrainer(Trainer):
    """Mean example loss, with accumulation/DDP scaling owned by Trainer.

    Qwen's generic classification forward accepts **kwargs but does not pass
    num_items_in_batch to its loss. Trainer otherwise assumes that loss was
    normalized over the accumulated global batch and scales it incorrectly.
    """
    def __init__(self, *args, group_training_by_length=False, **kwargs):
        self.group_training_by_length = group_training_by_length
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.loss_is_scaled_for_ga = False

    def _get_train_sampler(self, train_dataset=None):
        if not self.group_training_by_length:
            return super()._get_train_sampler(train_dataset)
        from transformers.trainer_pt_utils import LengthGroupedSampler
        dataset = self.train_dataset if train_dataset is None else train_dataset
        lengths = [max(len(a),len(b)) for a,b in zip(dataset.original,dataset.swapped)]
        # Group only training batches. Prediction ordering must stay sequential
        # so original/swapped probabilities match the external row IDs and labels.
        return LengthGroupedSampler(self.args.train_batch_size*self.args.gradient_accumulation_steps,
                                    lengths=lengths)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


class Journal(TrainerCallback):
    def __init__(self, path, max_seconds):
        self.path, self.max_seconds = Path(path), max_seconds
        self.started = None
        self.stopped_for_budget = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        record = {"step": state.global_step, "elapsed_seconds": time.monotonic()-self.started, **(logs or {})}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str)+"\n")
        print("JOINT_PROGRESS", json.dumps(record, default=str), flush=True)
        loss = (logs or {}).get("loss")
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError("Non-finite training loss; stopping instead of consuming the run budget")

    def on_step_end(self, args, state, control, **kwargs):
        stop = bool(self.max_seconds and time.monotonic()-self.started > self.max_seconds)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor(int(stop), device=torch.device("cuda",torch.cuda.current_device()))
            torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
            stop = bool(flag.item())
        if stop:
            self.stopped_for_budget = True
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = True
        return control


def load_base_model(model_path, config, training):
    cuda = torch.cuda.is_available()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(rank)
    native_bf16 = cuda and torch.cuda.get_device_capability(rank)[0] >= 8
    dtype = torch.bfloat16 if native_bf16 else (torch.float16 if cuda else torch.float32)
    quantize = config.get("quantize", True)
    if quantize and not cuda:
        raise RuntimeError("Full QLoRA requires a GPU; CPU is reserved for tiny correctness tests")
    kwargs = dict(num_labels=3, id2label=dict(enumerate(CLASSES)), label2id={c:i for i,c in enumerate(CLASSES)},
                  dtype=dtype, attn_implementation="sdpa", local_files_only=True)
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["score"])
        kwargs["device_map"] = {"": rank}
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), **kwargs)
    model.config.use_cache = False
    if quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, dtype


def load_trainable_model(model_path, config):
    model, dtype = load_base_model(model_path, config, training=True)
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], bias="none",
        target_modules=config["target_modules"], modules_to_save=["score"])
    model = get_peft_model(model, lora)
    assert any("score" in n and p.requires_grad for n,p in model.named_parameters())
    assert any("lora_" in n and p.requires_grad for n,p in model.named_parameters())
    return model, dtype


def compute_metrics(prediction):
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    return metrics(prediction.label_ids, probabilities(logits))


def make_trainer(model, tokenizer, train_dataset, dev_dataset, config, out, pilot_steps=0):
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    eval_steps = min(config["eval_steps"], max(1, pilot_steps//2)) if pilot_steps else config["eval_steps"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    total_steps = pilot_steps or math.ceil(config["epochs"]*len(train_dataset)/
        (config["train_batch_size"]*config["gradient_accumulation_steps"]*world))
    arguments = TrainingArguments(output_dir=str(out/"checkpoints"),
        num_train_epochs=config["epochs"], max_steps=pilot_steps if pilot_steps else -1,
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        warmup_steps=max(1,int(.05*total_steps)), lr_scheduler_type="cosine", max_grad_norm=1.,
        gradient_checkpointing=cuda, gradient_checkpointing_kwargs={"use_reentrant":False},
        bf16=bf16, fp16=cuda and not bf16, eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="log_loss", greater_is_better=False,
        logging_steps=5, logging_first_step=True, logging_nan_inf_filter=False,
        report_to="none", seed=config["seed"], data_seed=config["seed"],
        ddp_find_unused_parameters=False, ddp_timeout=1800, dataloader_num_workers=0, dataloader_pin_memory=cuda,
        remove_unused_columns=False, optim="adamw_torch", disable_tqdm=True)
    callbacks = [Journal(out/"training_progress.jsonl", config["max_train_seconds"])]
    if not pilot_steps:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    return PreferenceTrainer(model=model, args=arguments, train_dataset=train_dataset, eval_dataset=dev_dataset,
        group_training_by_length=config.get('group_training_by_length',False),
        processing_class=tokenizer, data_collator=PreferenceCollator(tokenizer),
        compute_metrics=compute_metrics, callbacks=callbacks)


def predict_swapped(trainer, original, swapped, labels):
    def one(force):
        ds = PreferenceDataset(original, swapped, labels, force_swap=force)
        logits = trainer.predict(ds).predictions
        if isinstance(logits, tuple): logits = logits[0]
        return probabilities(logits)
    return (one(False)+one(True)[:,[1,0,2]])*.5


def run_training(raw, model_path, split_path, config, out, pilot_steps=0, resume=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/"summary.json").exists():
        raise RuntimeError("This training result already exists; use a separate job directory")
    if config.get('preprocessing_sha256'):
        assert hashlib.sha256(Path(__file__).with_name('joint_data.py').read_bytes()).hexdigest()==config['preprocessing_sha256']
    if config.get('split_ids_sha256'):
        assert hashlib.sha256(Path(split_path).read_bytes()).hexdigest()==config['split_ids_sha256']
    set_seed(config["seed"])
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    frame,_=read_competition(raw)
    parts=split_from_ids(frame,json.loads(Path(split_path).read_text()))
    rng=np.random.default_rng(config["seed"])
    train_idx=parts["train"]
    if pilot_steps:
        train_idx=np.sort(rng.choice(train_idx,min(2048,len(train_idx)),replace=False))
    selection_idx=np.sort(rng.choice(parts["dev"],min(128 if pilot_steps else config["selection_dev_rows"],len(parts["dev"])),replace=False))
    encoder=JointTokenizer(tokenizer,config["max_length"])
    train_frame=frame.iloc[train_idx]
    original,swapped,train_diagnostics=encoder.encode_frame(train_frame)
    labels=train_frame[CLASSES].values.argmax(1)
    training=PreferenceDataset(original,swapped,labels,augment=True)
    dev_frame=frame.iloc[selection_idx]
    dev_orig,dev_swap,dev_diagnostics=encoder.encode_frame(dev_frame)
    dev_labels=dev_frame[CLASSES].values.argmax(1)
    development=PreferenceDataset(dev_orig,dev_swap,dev_labels)
    model,dtype=load_trainable_model(model_path,config)
    model.config.pad_token_id=tokenizer.pad_token_id
    model.config.problem_type="single_label_classification"
    trainer=make_trainer(model,tokenizer,training,development,config,out,pilot_steps)
    started=time.monotonic()
    result=trainer.train(resume_from_checkpoint=resume)
    train_seconds=time.monotonic()-started
    trainer.save_model(str(out/"adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(out/"adapter")
        json_write(out/"training_config.json",config)
    # The confirmation partition is never evaluated during model selection.
    final_idx=selection_idx if pilot_steps else parts["dev"]
    val_frame=frame.iloc[final_idx]
    val_orig,val_swap,_=encoder.encode_frame(val_frame)
    val_y=val_frame[CLASSES].values.argmax(1)
    prediction_started=time.monotonic()
    p=predict_swapped(trainer,val_orig,val_swap,val_y)
    prediction_seconds=time.monotonic()-prediction_started
    if trainer.is_world_process_zero():
        temperature=minimize_scalar(lambda t: metrics(val_y,temper(p,t))["log_loss"],bounds=(.5,3.),method="bounded").x
        np.savez_compressed(out/"dev_predictions.npz",ids=val_frame.id.to_numpy(),labels=val_y,probabilities=p)
        world=int(os.environ.get("WORLD_SIZE","1"))
        summary={"model_id":config["model_id"],"model_path":str(model_path),"pilot":bool(pilot_steps),
            "config":config,"dtype":str(dtype),"world_size":world,"train_rows":len(train_idx),
            "selection_dev_rows":len(selection_idx),"final_dev_rows":len(final_idx),
            "confirmation_evaluated":False,"global_steps":trainer.state.global_step,
            "epochs_completed":float(trainer.state.epoch),
            "stopped_for_budget":any(getattr(c,'stopped_for_budget',False) for c in trainer.callback_handler.callbacks),
            "adapter_sha256":hashlib.sha256((out/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
            "best_selection_dev_log_loss":trainer.state.best_metric,
            "best_checkpoint":trainer.state.best_model_checkpoint,
            "training_seconds":train_seconds,"seconds_per_optimizer_step":train_seconds/max(trainer.state.global_step,1),
            "development_prediction_seconds":prediction_seconds,
            "train_result":result.metrics,"development":metrics(val_y,p),
            "temperature":float(temperature),"calibrated_development":metrics(val_y,temper(p,temperature)),
            "trainable_parameters":model.get_nb_trainable_parameters()[0],
            "total_parameters":model.get_nb_trainable_parameters()[1],
            "peak_gpu_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "diagnostics":{"train":train_diagnostics,"development":dev_diagnostics},
            "software":{"python":platform.python_version(),"torch":torch.__version__}}
        json_write(out/"summary.json",summary)
        print("JOINT_RESULT",json.dumps(summary),flush=True)
    trainer.accelerator.wait_for_everyone()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--split",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pilot-steps",type=int,default=0)
    parser.add_argument("--resume")
    args=parser.parse_args()
    run_training(args.raw,args.model,args.split,json.loads(args.config.read_text()),args.out,args.pilot_steps,args.resume)


if __name__=="__main__":
    main()
