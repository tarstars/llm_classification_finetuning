"""Admit engineering preparation only, never weights or GPU from source alone."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_016'
    root = ROOT / 'runs/span_016'
    plan = read_json(pub / 'span_plan.json')
    assert not (pub / 'source_audit.json').exists(), 'Preserve source decision'
    assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
    observed = read_json(pub / 'source_observation.json')
    native = read_json(pub / 'tokenizer_validation.json')
    cohort = read_json(pub / 'cohort_manifest.json')
    for name, digest in native['source_files_sha256'].items():
        assert sha256(root / 'source_audit' / name) == digest
    for name, digest in cohort['artifact_sha256'].items():
        assert sha256(root / 'cohort' / name) == digest
    assert native['passed'] and all(c['native_reference_identical'] for c in native['checks'])
    assert all(all(m['immutable_checks'].values()) and m['license'] == 'apache-2.0' and not m['gated'] for m in observed['models'])
    index = read_json(root / 'source_audit/8B/model.safetensors.index.json')
    assert 'score.weight' in index['weight_map']
    assert cohort['rows'] == cohort['selected_unique_groups'] == 1024
    assert cohort['targets_read_after_durable_selection'] and not cohort['selection_uses_targets']
    cutoff = datetime.fromisoformat(cohort['post_publication_cutoff'])
    for size in plan['models']:
        metadata = read_json(root / 'source_audit' / size / 'metadata.json')
        assert datetime.fromisoformat(metadata['lastModified'].replace('Z', '+00:00')) <= cutoff
    assert datetime.fromisoformat(cohort['timestamp_range'][0]) > cutoff
    assert read_json(REPO / 'results/span_011/source_metadata_audit.json')['checkpoint_dates_before_2025']
    total = sum(m['total_weight_bytes'] for m in observed['models'])
    assert total < 18 * 1024**3 and shutil.disk_usage(root).free >= 40 * 1024**3
    result = dict(at=utc_now(), source_stage_passed=True, gpu_admitted=False,
        full_weights_downloaded=False, new_gpu_executions=0, native_template_validated=True,
        offline_assets_ready=False, offline_path='Stage one private Kaggle dataset from pinned HF blobs after exact worker rehearsal; verify complete hashes before GPU.',
        staged_weight_bytes=total, source_files_bytes=native['source_files_bytes'],
        remaining_requirements=['Finish and rehearse exact worker before any full weights',
            'Pin and validate Transformers4.52.3 runtime wheels',
            'Acquire/hash immutable assets and pin private Kaggle version',
            'Fresh complete resources/jobs/receipts and GPU admission',
            'Actual source/numeric/native/runtime/information gates before official trial'],
        out_of_time_source=True, independent_competition_cv=False,
        source_model_cutoff=cohort['post_publication_cutoff'],
        quality_timestamp_range=cohort['timestamp_range'], quality_rows=1024,
        training_overlap_limit='Selected votes postdate published checkpoints and exclude prior inspected groups; texts can preexist votes and domain transport is not established.',
        native_validation_transformers=native['local_transformers_version'],
        quality_measured=False, evidence_sha256={n: sha256(pub / n) for n in [
            'source_observation.json', 'tokenizer_validation.json', 'cohort_manifest.json']},
        source_sha256={str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
            'src/arena/span016_reward.py': sha256(REPO / 'src/arena/span016_reward.py'),
            'docs/SPAN_016_PILOT_PROTOCOL.md': sha256(REPO / 'docs/SPAN_016_PILOT_PROTOCOL.md')})
    atomic_json(pub / 'source_audit.json', result)
    print(json.dumps({k: result[k] for k in ['source_stage_passed', 'gpu_admitted',
        'quality_rows', 'source_model_cutoff', 'quality_timestamp_range', 'staged_weight_bytes']}))


if __name__ == '__main__':
    main()
