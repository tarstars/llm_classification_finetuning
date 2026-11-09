"""Record supported source-gate conclusions; no model weights or task targets."""
from pathlib import Path
import time
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    started=time.monotonic();pub=REPO/'results/span_037';run=ROOT/'runs/span_037/source_audit'
    assert not (pub/'H80_results.json').exists()
    observed=read_json(pub/'candidate_observation.json');chosen=read_json(pub/'candidates.json')
    assert [x['candidate'] for x in observed['observations']]==[x['ref'] for x in chosen['candidates']]
    for candidate in observed['observations']:
        for f in candidate['files']:
            assert sha256(Path(f['path']))==f['sha256']
            assert not Path(f['path']).name.endswith(('.safetensors','.pth','.bin','.zip'))
    native=ROOT/'runs/span_009/research/lmsys-checkpoints-0-0805'
    config=read_json(native/'config.json');index=read_json(native/'model.safetensors.index.json')
    assert config['model_type']=='gemma2' and config['hidden_size']==3584 and config['num_hidden_layers']==42
    head=sorted(k for k in index['weight_map'] if not k.startswith('model.'))
    assert head==['score.1.bias','score.1.weight','score.4.bias','score.4.weight']
    decisions=[dict(candidate=chosen['candidates'][0]['ref'],public_access=True,compatible_pair_established=False,
        exact_base_established=False,head_and_class_mapping_established=False,
        reason='HF release provides six weight archives and .gitattributes, no standalone card/config/source. Two relevant same-handle repositories are forks of retained second/third-place code, with no demonstrated link to these archives. No license/initialization/native-head lineage proof.'),
        dict(candidate=chosen['candidates'][1]['ref'],public_access=True,compatible_pair_established=False,
        exact_base_established=False,head_and_class_mapping_established=False,
        reason='Release contains two .pth weight files, empty dataset description and unknown dataset license. Retrieved Part1/Part2 author notebooks describe embedding extraction and CatBoost, not a documented loader for the released final-team weights. Their class order does not establish the checkpoint head or merge lineage.'),
        dict(candidate=chosen['candidates'][2]['ref'],public_access=True,compatible_pair_established=False,
        exact_base_established=False,head_and_class_mapping_established=False,
        reason='Public Apache2 adapter release names ./gemma-2-9b-it-4bit with revision null and saves classifier/score, but README is an unfilled template. Exact base initialization, tensor head layout, native text and class order remain unestablished. Public author notebook listing has no matching LMSYS/Gemma source.')]
    files=[p for p in run.rglob('*') if p.is_file()];total=sum(p.stat().st_size for p in files);assert total<50*1024**2
    result=dict(at=utc_now(),task='H80',complete=True,source_stage_passed=False,
        quality_measured=False,compatible_pair_established=False,decisions=decisions,
        native_reference=dict(base=config['_name_or_path'],architecture=config['architectures'],
            vocabulary=config['vocab_size'],head_keys=head,config_sha256=sha256(native/'config.json'),index_sha256=sha256(native/'model.safetensors.index.json')),
        small_source_bytes=total,source_files_sha256={str(p.relative_to(run)):sha256(p) for p in files},
        weights_downloaded=False,task_targets_loaded=False,new_gpu_executions=0,
        new_official_submissions=0,official_submission_eligible=False,decision='stop_no_proved_compatible_pair',
        scope='Bounded three-candidate source gate, not an absence proof over all public checkpoints',
        method_sha256={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_037_PLAN.md','scripts/discover_span_037_sources.py','scripts/inspect_span_037_candidates.py','scripts/trace_span_037_authors.py',str(Path(__file__).relative_to(REPO))]},
        elapsed_analysis_seconds=time.monotonic()-started)
    atomic_json(pub/'H80_results.json',result)
    print(dict(decision=result['decision'],small_source_bytes=total,weights_downloaded=False,new_gpu_executions=0))


if __name__=='__main__':main()
