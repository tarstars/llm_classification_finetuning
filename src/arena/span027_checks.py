"""H74 input/source/canonical controls; one fixed packing and no quality change."""
from pathlib import Path
import json
import numpy as np

PACKING_COHORTS={'warmup','control','uniform','long','test',
    'routed_control','routed_uniform','routed_long','routed_test'}


def conversation_budget(role,name):
    allowed=({'warmup','control','uniform','long','test'} if role in ['m0','m3'] else
        {'warmup','routed_control','routed_uniform','routed_long','routed_test'} if role=='m0_reverse' else set())
    if name not in allowed:raise ValueError('Unknown role/cohort')
    return 4096


def after_cohort(args,summary,name):
    spec=json.loads(Path(args.spec).read_text());role=args.role;out=Path(args.out)
    assert spec['packing_tokens']==summary['packing_tokens']==10240
    expected=spec['token_references'][role][name];actual=summary['cohorts'][name]
    a=np.load(out/(role+'_'+name+'.npz'))
    check=dict(ids_match=a['ids'].tolist()==expected['ids'],lengths_match=a['lengths'].tolist()==expected['lengths'],
        token_hash_match=actual['token_ids_sha256']==expected['token_ids_sha256'],
        checkpoint_match=json.loads((out/(role+'_checkpoint.json')).read_text())==spec['checkpoint_references'][role])
    memory=[actual['peak_allocated_bytes'][g]/summary['device_total_memory_bytes'][g] for g in range(2)]
    check.update(memory_fractions=memory,memory_passed=all(x<=.95 for x in memory))
    if name in ['control','routed_control']:
        gold=np.load(Path(spec['control_directory'])/(role+'.npz'))
        assert np.array_equal(a['ids'],gold['ids'])
        diff=float(np.abs(a['probabilities'].astype(float)-gold['probabilities']).max())
        check.update(canonical_rows=len(a['ids']),canonical_max_abs=diff,canonical_passed=diff<=1e-5)
    check['passed']=all(check[k] for k in ['ids_match','lengths_match','token_hash_match','checkpoint_match','memory_passed']) and check.get('canonical_passed',True)
    path=out/(role+'_checks.json');saved=json.loads(path.read_text()) if path.exists() else {}
    saved[name]=check;path.write_text(json.dumps(saved,indent=2)+'\n')
    if not check['passed']:
        (out/'engineering_stop.json').write_text(json.dumps(dict(role=role,cohort=name,checks=check,official_submission_eligible=False),indent=2)+'\n')
    print('ARENA_EVENT '+json.dumps(dict(stage='packing_cohort_checks',role=role,cohort=name,passed=check['passed'])),flush=True)
    return check['passed']
