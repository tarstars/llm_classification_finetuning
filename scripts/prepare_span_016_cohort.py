"""Freeze fresh post-publication groups without consulting their human targets."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now
from arena.span011_data import LABELS, write_inference_inputs


def main():
    pub = REPO / 'results/span_016'
    root = ROOT / 'runs/span_016/cohort'
    root.mkdir(parents=True, exist_ok=True)
    plan = read_json(pub / 'span_plan.json')
    with exclusive_lock(root / 'prepare.lock'):
        assert not (root / 'selection.json').exists(), 'Preserve fixed selection'
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        source = ROOT / 'runs/span_016/source_audit'
        times = [read_json(source / size / 'metadata.json')['lastModified'] for size in plan['models']]
        cutoff = max(datetime.fromisoformat(t.replace('Z', '+00:00')) for t in times)
        old = ROOT / 'runs/span_011/cohort'
        previous = pd.concat([pd.read_parquet(old / (name + '_mapping.parquet'))
                              for name in ['learning', 'confirmation']])
        excluded = set(previous.group_id)
        # Deliberately do not project winner or model identities during selection.
        database = old / 'rows.sqlite'
        connection = sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True)
        query = 'SELECT r.payload,p.group_id FROM representatives p JOIN rows r ON r.uid=p.uid'
        candidates = []
        eligible = 0
        for payload, group in connection.execute(query):
            row = json.loads(payload)
            if datetime.fromisoformat(row['timestamp']) <= cutoff:
                continue
            eligible += 1
            if group in excluded:
                continue
            row['group_id'] = group
            priority = hashlib.sha256(('42:span016:sample:' + group).encode()).hexdigest()
            candidates.append((priority, row))
        candidates.sort(key=lambda item: (item[0], item[1]['source_uuid']))
        selected = sorted([r for _, r in candidates[:1024]], key=lambda r: r['id'])
        assert len(selected) == len({r['group_id'] for r in selected}) == 1024
        assert not {r['group_id'] for r in selected} & excluded
        selection = dict(at=utc_now(), cutoff=cutoff.isoformat(), seed=42,
            priority='sha256(42:span016:sample:<group_id>)', uses_targets=False,
            eligible_post_publication_groups=eligible, available_after_exclusions=len(candidates),
            previously_inspected_groups_excluded=len(excluded),
            ids=[r['id'] for r in selected], groups=[r['group_id'] for r in selected],
            timestamps=[r['timestamp'] for r in selected])
        # Durable target-blind IDs precede the first target query.
        atomic_json(root / 'selection.json', selection)
        write_inference_inputs(root / 'source.parquet', selected)
        pd.DataFrame([{k: r[k] for k in ['id', 'source_uuid', 'group_id', 'timestamp']}
                      for r in selected]).to_parquet(root / 'mapping.parquet', index=False)
        labels = []
        for row in selected:
            winner = connection.execute('SELECT winner FROM rows WHERE uid=?', (row['source_uuid'],)).fetchone()[0]
            assert winner in LABELS
            labels.append(LABELS[winner])
        connection.close()
        np.savez_compressed(root / 'labels.npz', ids=np.asarray(selection['ids'], dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int64), groups=np.asarray(selection['groups']))
        manifest = dict(at=utc_now(), rows=1024, source_dataset='lmarena-ai/arena-human-preference-140k',
            source_revision='6322995ab34d7c2693e3f47dd13fa5caa0789a74', source_license='cc-by-4.0',
            source_database_sha256=sha256(database), selection_sha256=sha256(root / 'selection.json'),
            selection_uses_targets=False, targets_read_after_durable_selection=True,
            post_publication_cutoff=cutoff.isoformat(),
            timestamp_range=[min(selection['timestamps']), max(selection['timestamps'])],
            eligible_post_publication_groups=eligible, available_after_exclusions=len(candidates),
            previously_inspected_groups_excluded=len(excluded), selected_unique_groups=1024,
            independent_competition_cv=False, out_of_time_source=True,
            artifact_sha256={n: sha256(root / n) for n in ['selection.json', 'source.parquet', 'mapping.parquet', 'labels.npz']},
            previous_mapping_sha256={n: sha256(old / n) for n in ['learning_mapping.parquet', 'confirmation_mapping.parquet']},
            source_sha256=sha256(Path(__file__)), new_gpu_executions=0)
        atomic_json(pub / 'cohort_manifest.json', manifest)
        print(json.dumps({k: manifest[k] for k in ['rows', 'timestamp_range', 'eligible_post_publication_groups',
            'available_after_exclusions', 'previously_inspected_groups_excluded', 'out_of_time_source']}))


if __name__ == '__main__':
    main()
