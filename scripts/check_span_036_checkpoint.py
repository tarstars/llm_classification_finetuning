"""Strictly verify the full published state on CPU before any GPU admission."""
from pathlib import Path
import resource
import sys
import time
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span036_pairrm import native_definitions,construct


def main():
    start=time.monotonic();pub=REPO/'results/span_036';folder=ROOT/'runs/span_036/input_asset/upload'
    assert not (pub/'checkpoint_cpu_check.json').exists()
    prep=read_json(pub/'asset_preparation.json');assert sha256(folder/'model.safetensors')==prep['weight_sha256']
    torch.set_num_threads(2)
    cls,collator_cls,_=native_definitions(folder)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True)
    collator_cls(1224,tokenizer,412)
    model=construct(folder,tokenizer,cls)
    state=load_file(str(folder/'model.safetensors'),device='cpu')
    loaded=model.load_state_dict(state,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert tuple(model.head_layer[1].weight.shape)==(1024,2048) and tuple(model.head_layer[4].weight.shape)==(1,1024)
    result=dict(at=utc_now(),passed=True,all_state_keys_strict=True,state_tensors=len(state),
        parameters=sum(p.numel() for p in model.parameters()),vocabulary_size=len(tokenizer),
        native_head_shape=True,weights_sha256=prep['weight_sha256'],task_targets_loaded=False,
        elapsed_seconds=time.monotonic()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    assert result['elapsed_seconds']<1200 and result['peak_rss_gib']<6
    atomic_json(pub/'checkpoint_cpu_check.json',result);print(result)


if __name__=='__main__':main()
