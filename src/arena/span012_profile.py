"""Matched packing probe; frozen tascj two-GPU inference, Apache-2.0 source.

Adapted from span010_order and tascj/kaggle-lmsys-chatbot-arena,
commit83cd93d50b9283c18711e8c63e4e1c6399c7b9ce. No preference-target reads.
"""

import argparse

import hashlib

import json

import os

from pathlib import Path

import subprocess

import sys

import time

import numpy as np

import pandas as pd

PROFILE_NAMES=['warmup_8192_first','profile_8192_first','warmup_6144','profile_6144',
               'warmup_8192_last','profile_8192_last']
ORDER_NAMES=['warmup','control','uniform','long','test','routed_control','routed_uniform','routed_long','routed_test']


def gpu_snapshot():
    """Read-only XML query; https://docs.nvidia.com/deploy/nvidia-smi/ ."""
    import xml.etree.ElementTree as ET
    try:
        result=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=10)
        if result.returncode:return dict(available=False,returncode=result.returncode)
        tree=ET.fromstring(result.stdout);devices=[]
        for gpu in tree.findall('gpu'):
            fields={'name':'product_name','sm_clock':'clocks/sm_clock','memory_clock':'clocks/mem_clock',
                    'utilization':'utilization/gpu_util','memory_used':'fb_memory_usage/used',
                    'temperature':'temperature/gpu_temp','pstate':'performance_state','power':'power_readings/power_draw'}
            record={key:gpu.findtext(path) for key,path in fields.items()}
            if record['power'] is None:record['power']=gpu.findtext('gpu_power_readings/power_draw')
            devices.append(record)
        return dict(available=len(devices)==2,devices=devices)
    except (OSError,subprocess.TimeoutExpired,ET.ParseError) as error:
        return dict(available=False,error_type=type(error).__name__)

CLASSES = ['winner_model_a', 'winner_model_b', 'winner_tie']

TEXT = ['prompt', 'response_a', 'response_b']

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def event(stage, **values):
    print('ARENA_EVENT ' + json.dumps({'stage': stage, **values}, allow_nan=False), flush=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def profile_indices(frame, uniform_rows=256, long_rows=64):
    """Label-free fixed random cohort plus a disjoint long-text stress cohort."""
    rng = np.random.default_rng(42)
    uniform = rng.choice(len(frame), min(uniform_rows, len(frame)), replace=False)
    length = frame[TEXT].fillna('').apply(lambda s: s.str.len()).sum(axis=1).to_numpy()
    available = np.ones(len(frame), dtype=bool)
    available[uniform] = False
    long = np.argsort(-length, kind='stable')
    long = long[available[long]][:long_rows]
    return uniform, long

def validate_probabilities(probs, rows):
    if probs.shape != (rows, 3) or not np.isfinite(probs).all():
        raise ValueError('Invalid probability shape or nonfinite values')
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability simplex')

def ensemble(original, swapped_llama):
    validate_probabilities(original, len(original))
    validate_probabilities(swapped_llama, len(original))
    result = (original + swapped_llama[:, [1, 0, 2]]) / 2
    validate_probabilities(result, len(original))
    return result

def route_indices(ids, gemma, aligned_llama, lengths, work_fraction=.1, row_fraction=.2):
    """Deterministic text-dependent route; labels never enter this function."""
    ids=np.asarray(ids);lengths=np.asarray(lengths,dtype=float)
    validate_probabilities(gemma,len(ids));validate_probabilities(aligned_llama,len(ids))
    if lengths.shape!=(len(ids),2) or not np.isfinite(lengths).all() or (lengths<=0).any():
        raise ValueError('Invalid model token lengths')
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate IDs')
    if not (0<=work_fraction<=1 and 0<=row_fraction<=1):raise ValueError('Invalid route budget')
    p=np.clip(gemma.astype(float),1e-15,1);q=np.clip(aligned_llama.astype(float),1e-15,1)
    middle=(p+q)/2
    disagreement=(np.sum(p*np.log(p/middle),1)+np.sum(q*np.log(q/middle),1))/2
    disagreement=np.maximum(disagreement,0)
    work=np.sum(lengths*np.maximum(1,lengths/2048),axis=1)
    order=np.lexsort((ids,-disagreement/work))
    selected=[];used=0.;budget=float(work.sum()*work_fraction)
    for index in order:
        if len(selected)>=int(len(ids)*row_fraction):break
        if disagreement[index]>0 and used+work[index]<=budget:
            selected.append(int(index));used+=work[index]
    return np.sort(np.asarray(selected,dtype=int)),disagreement,work

def infer_role(args):
    import torch
    import transformers
    import triton
    import xformers
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    from human_pref.data.processors import ProcessorPAB
    from human_pref.data.dataset import LMSYSDataset
    from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator
    from human_pref.utils import to_device

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    assert torch.cuda.device_count() == 2
    role = args.role
    is_gemma = role in ['m0']
    swapped = role in ['m3']
    if is_gemma:
        from human_pref.models.modeling_gemma2 import Gemma2ForSequenceClassification as Model
    else:
        from human_pref.models.modeling_llama import LlamaForSequenceClassification as Model
    model_path = Path(args.assets) / ('lmsys-checkpoints-0-0805' if is_gemma else 'lmsys-checkpoints-3-0805')
    small = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json']
    source = {name: digest(model_path / name) for name in small}
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    source['weight_shards'] = {name: {'bytes': (model_path / name).stat().st_size,
                                     'sha256': digest(model_path / name)} for name in shards}
    save(Path(args.out) / (role + '_checkpoint.json'), source)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    processor = ProcessorPAB(tokenizer, max_length=4096, support_system_role=not is_gemma)
    layers = 42 if is_gemma else 32
    devices = {'model.embed_tokens': 'cuda:0', 'model.norm': 'cuda:1', 'score': 'cuda:1'}
    devices.update({f'model.layers.{i}': f'cuda:{int(i >= layers // 2)}' for i in range(layers)})
    event('model_loading', role=role)
    model, loading = Model.from_pretrained(model_path, torch_dtype=torch.float16,
        device_map=devices, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    model.eval()
    assert {p.device.index for p in model.parameters()} == {0, 1}
    config = model.config
    dim = config.head_dim if is_gemma else config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq0, freq1 = inv_freq.to('cuda:0'), inv_freq.to('cuda:1')
    cold = time.monotonic() - started
    summary = {'role': role, 'cold_seconds': cold, 'cohorts': {}, 'versions': {
        'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
        'triton': triton.__version__, 'xformers': xformers.__version__}, 'loading_exact': True}
    names = args.cohorts.split(',')
    assert all(name in PROFILE_NAMES if args.mode=='profile' else name in ORDER_NAMES for name in names)
    for name in names:
        packing = int(name.split('_')[1]) if args.mode=='profile' else args.packing_tokens
        assert packing in [6144,8192]
        hardware_before = gpu_snapshot()
        frame = pd.read_parquet(Path(args.out) / (name + '.parquet'))
        if not len(frame):
            np.savez_compressed(Path(args.out)/(role+'_'+name+'.npz'),ids=frame.id.to_numpy(),
                                probabilities=np.empty((0,3)),lengths=np.empty(0,dtype=int))
            summary['cohorts'][name]=dict(rows=0,seconds=0,tokens=0,max_tokens=0)
            save(Path(args.out)/(role+'_summary.json'), summary)
            continue
        if swapped:
            frame[['response_a', 'response_b']] = frame[['response_b', 'response_a']].values
        # These placeholders serve the unchanged author dataset interface only.
        # Never call dataset.evaluate, and never pass original votes to the model.
        for i, col in enumerate(CLASSES):
            frame[col] = int(i == 0)
        path = Path(args.out) / (role + '_input.parquet')
        frame.to_parquet(path, index=False)
        dataset = LMSYSDataset(path, processor=processor, include_swap=False, is_parquet=True)
        # Re-forking workers after CUDA/tokenizer execution crashed in the
        # long-cohort control. Keep tokenization in this process for reliability.
        loader = DataLoader(dataset, batch_size=80, num_workers=0,
            collate_fn=ShardedMaxTokensCollator(max_tokens=packing, base_collator=VarlenCollator()))
        outs, lengths, token_hash = [], [], hashlib.sha256()
        previous = None
        for gpu in range(2):
            torch.cuda.reset_peak_memory_stats(gpu)
            torch.cuda.synchronize(gpu)
        begin = time.monotonic()
        # Same two-stage pipeline and block-diagonal causal attention as author.
        for batch_number, batch in enumerate(loader):
            for micro in batch:
                lengths.extend(micro['seq_lens'])
                token_hash.update(micro['input_ids'].numpy().tobytes())
                input_ids = to_device(micro['input_ids'], 'cuda:0')
                seq_info = to_device(dict(cu_seqlens=micro['cu_seqlens'],
                    position_ids=micro['position_ids'], max_seq_len=micro['max_seq_len'],
                    attn_bias=BlockDiagonalCausalMask.from_seqlens(micro['seq_lens'])), 'cuda:0')
                with torch.no_grad(), torch.cuda.amp.autocast():
                    logits = None
                    if previous is not None:
                        hidden, prev_seq = previous
                        logits = model.forward_part2(hidden, prev_seq, freq1)
                    hidden = model.forward_part1(input_ids, seq_info, freq0)
                    seq1, hidden1 = to_device([seq_info, hidden], 'cuda:1')
                    previous = hidden1, seq1
                    # Match the author: launch both GPU stages before the blocking
                    # host copy, so the two devices can overlap their work.
                    if logits is not None:
                        outs.append(logits.cpu())
            event('inference_progress', role=role, cohort=name, batch=batch_number,
                  rows=len(lengths), seconds=time.monotonic()-begin)
        if previous is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                outs.append(model.forward_part2(previous[0], previous[1], freq1).cpu())
        probs = torch.cat(outs, dim=0).softmax(-1).numpy()
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        seconds = time.monotonic() - begin
        validate_probabilities(probs, len(frame))
        np.savez_compressed(Path(args.out) / (role + '_' + name + '.npz'),
            ids=frame.id.to_numpy(), probabilities=probs, lengths=np.array(lengths))
        summary['cohorts'][name] = {'rows': len(frame), 'seconds': seconds,
            'tokens': int(sum(lengths)), 'max_tokens': max(lengths),
            'token_ids_sha256': token_hash.hexdigest(), 'packing_tokens': packing,
            'hardware_before': hardware_before, 'hardware_after': gpu_snapshot(),
            'peak_allocated_bytes': [torch.cuda.max_memory_allocated(g) for g in range(2)],
            'peak_reserved_bytes': [torch.cuda.max_memory_reserved(g) for g in range(2)]}
        save(Path(args.out) / (role + '_summary.json'), summary)
        path.unlink()
        event('cohort_complete', role=role, cohort=name, rows=len(frame), seconds=seconds)


def plain_inputs(path,frame):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=pa.schema([('id',pa.int64()),*[(key,pa.string()) for key in TEXT]])
    pq.write_table(pa.Table.from_pylist(frame[['id',*TEXT]].to_dict('records'),schema=schema),path,
                   version='1.0',compression='snappy')


def decide_packing(summaries,differences):
    base_names=['profile_8192_first','profile_8192_last']
    if not all(np.isfinite(v) and v>=0 for values in differences.values() for v in values.values()):
        raise ValueError('Invalid numerical-control difference')
    for role in ['m0','m3']:
        if any(differences[role][n]>=1e-5 for n in base_names):
            raise ValueError('Canonical8192raw control failed; preserve failure')
    base=sum(np.mean([summaries[r]['cohorts'][n]['seconds'] for n in base_names]) for r in ['m0','m3'])
    candidate=sum(summaries[r]['cohorts']['profile_6144']['seconds'] for r in ['m0','m3'])
    if base<=0 or candidate<=0:raise ValueError('Invalid measured duration')
    ratios={r:max(summaries[r]['cohorts'][n]['seconds'] for n in base_names)/
              min(summaries[r]['cohorts'][n]['seconds'] for n in base_names) for r in ['m0','m3']}
    stable=all(v<=1.10 for v in ratios.values())
    eligible=all(differences[r]['profile_6144']<1e-5 for r in ['m0','m3'])
    faster=candidate<=.95*base
    return dict(selected_packing=6144 if stable and eligible and faster else 8192,
        base_controls_passed=True,bracket_stable=stable,bracket_ratios=ratios,
        candidate_numerically_eligible=eligible,candidate_at_least_5pct_faster=faster,
        summed_8192_mean_seconds=float(base),summed_6144_seconds=float(candidate),
        candidate_speed_ratio=float(candidate/base),selection_uses_labels=False)


def orchestrate_profile(args):
    out=Path(args.out);out.mkdir(exist_ok=True)
    frame=pd.read_csv(Path(args.raw)/'train.csv',usecols=['id',*TEXT])
    control,_=profile_indices(frame);selected=frame.iloc[control]
    assert len(selected)==256 and selected.id.is_unique
    save(out/'profile_selection.json',dict(seed=42,uses_targets=False,control_ids=selected.id.tolist(),
                                          train_csv_sha256=digest(Path(args.raw)/'train.csv')))
    for name in PROFILE_NAMES:
        plain_inputs(out/(name+'.parquet'),selected.iloc[:80] if name.startswith('warmup_') else selected)
    for role in ['m0','m3']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--mode','profile','--role',role,
            '--raw',args.raw,'--assets',args.assets,'--out',args.out,'--cohorts',','.join(PROFILE_NAMES)],check=True)
    for role in ['m0','m3']:
        arrays=[np.load(out/(role+'_'+name+'.npz')) for name in PROFILE_NAMES if name.startswith('profile_')]
        assert all(np.array_equal(a['ids'],selected.id) for a in arrays)
        assert all(np.array_equal(a['lengths'],arrays[0]['lengths']) for a in arrays)
    save(out/'profile_complete.json',dict(complete=True,uses_targets=False,rows=256,warmup_rows=80,
        pack_sequence=[8192,6144,8192],models=['m0','m3'],settings_selected_by_quality=False))
    for path in out.glob('*.parquet'):path.unlink()
    event('profile_complete',complete=True,rows=256)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=['profile'],default='profile')
    parser.add_argument('--packing-tokens',type=int,default=8192)
    parser.add_argument('--role',choices=['m0','m3']);parser.add_argument('--cohorts')
    args=parser.parse_args();(infer_role if args.role else orchestrate_profile)(args)
