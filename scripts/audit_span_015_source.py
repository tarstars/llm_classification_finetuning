"""Resolve H55 from source evidence without inferring unobserved weight quality."""
from pathlib import Path
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_015'
    root = ROOT / 'runs/span_015/source_audit'
    observed = read_json(pub / 'source_observation.json')
    for name, digest in observed['small_files_sha256'].items():
        assert sha256(root / name) == digest
    author = root / 'author'
    train = (author / 'src__train_ce_model_nice_memory_grad_gemma2_double_16bit.py').read_text()
    predict = (author / 'src__predict_test.py').read_text()
    preprocess = ROOT / 'runs/span_008/teacher_audit/src__prepare_data.py'
    prep = preprocess.read_text()
    notebook = read_json(root / 'parent/merge-gemma-model-step1.ipynb')
    merge = '\n'.join(''.join(c.get('source', [])) for c in notebook['cells'])
    config = read_json(root / 'parent/config.json')
    index = read_json(root / 'parent/model.safetensors.index.json')
    assert 'score.weight' in index['weight_map']
    assert 'param.requires_grad = True' in train and 'if "lora" in k or "score" in k:' in train
    assert 'strict=False' in merge and 'CAUSAL_LM' in merge
    assert "return 2" in prep and "return 1" in prep and "return 0" in prep
    assert 'User question:' in predict and 'Answer A:' in predict and 'Answer B:' in predict
    assert observed['adapter_metadata']['http_status'] == 403
    assert not observed['full_weights_downloaded'] and observed['new_gpu_executions'] == 0
    gates = [
        dict(requirement='Immutable earlier parent availability', resolved=True,
             evidence='COMPLETE; exact source/config/index hashes match the v9 observation.'),
        dict(requirement='Classifier architecture and head presence', resolved=True,
             evidence='42-layer Gemma2ForSequenceClassification; three labels; score.weight in shard4.'),
        dict(requirement='Exact checkpoint training and overlap', resolved=False,
             evidence='Original adapter metadata403; no manifest links this earlier artifact to a fold/run. Family code uses competition supervision.'),
        dict(requirement='Exact class order and restored trained head', resolved=False,
             evidence='Family code maps A/B/tie to0/1/2 and trains/saves score; merge discards strict=False load diagnostics. Exact adapter keys/run unavailable.'),
        dict(requirement='Exact native input and pooling contract', resolved=False,
             evidence='Final family inference uses word caps, balanced truncation and appendedEOS; training slices1100tokens/padEOS, parent config pad0. No exact earlier inference reference.'),
        dict(requirement='License and complete lineage', resolved=False,
             evidence='Repository has no top-level license; training file carries Apache header. Base Gemma terms do not establish permission/lineage of the inaccessible adapter.'),
        dict(requirement='Complementarity and measured deployment fit', resolved=False,
             evidence='Earlier18.48GB bf16 checkpoint has no verified score or runtime. No outputs measured;905filename is not a quality estimate.'),
    ]
    result = dict(task='H55', at=utc_now(), audit_complete=True,
        source=observed['source'], author_repository='https://github.com/shyoulala/LMSYS_BlackPearl',
        author_commit='05ac9b917a99aac43bfe0dbcb30e6e83a38688be', gates=gates,
        source_audit_passed=False, gpu_admitted=False, quality_measured=False,
        independent_local_quality=False, quality_variants_tested=0,
        new_gpu_executions=0, official_submissions=0, full_weights_downloaded=False,
        small_files_bytes=observed['small_files_bytes'],
        family_training_saves_trained_score_head=True, exact_head_restore_verified=False,
        random_head_demonstrated=False, exact_label_map_verified=False,
        checkpoint_declared_weight_bytes=index['metadata']['total_size'],
        parent_pad_token_id=config['pad_token_id'], final_family_context=2000,
        decision='Stop before GPU: exact earlier checkpoint provenance, class/head restoration and native format remain unsupported. Retain H60 official0.82987.',
        artifact_sha256=observed['small_files_sha256'],
        earlier_preprocessing_sha256=sha256(preprocess),
        source_sha256={str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
            'scripts/gather_span_015_source.py': sha256(REPO / 'scripts/gather_span_015_source.py')})
    atomic_json(pub / 'H55_source_audit.json', result)
    print(json.dumps(dict(audit_complete=True, source_audit_passed=False,
        resolved_requirements=sum(g['resolved'] for g in gates), requirements=len(gates),
        gpu_admitted=False, new_gpu_executions=0)))


if __name__ == '__main__':
    main()
