"""Public golden checks and hidden input safety for the same two-role forward."""
from pathlib import Path
import json
import numpy as np

PACKING_COHORTS={'warmup','control','uniform','long','test'}


def conversation_budget(role,name):
    if role not in ['m0','m3'] or name not in PACKING_COHORTS:
        raise ValueError('Unknown two-expert role/cohort')
    return 4096


def after_cohort(args,summary,name):
    spec=json.loads(Path(args.spec).read_text());role=args.role;out=Path(args.out)
    assert role in ['m0','m3'] and spec['mode'] in ['qualification','hidden']
    assert spec['packing_tokens']==summary['packing_tokens']==8192 and not spec['uses_targets']
    a=np.load(out/(role+'_'+name+'.npz'));actual=summary['cohorts'][name]
    check=dict(checkpoint_match=json.loads((out/(role+'_checkpoint.json')).read_text())==spec['checkpoint_references'][role],
        producer_joined=actual['producer_joined'],
        ids_match=a['ids'].tolist()==spec['frame_ids'][name])
    fractions=[actual['peak_allocated_bytes'][g]/summary['device_total_memory_bytes'][g] for g in range(2)]
    check.update(memory_fractions=fractions,memory_passed=all(x<=.95 for x in fractions))
    required=['checkpoint_match','producer_joined','ids_match','memory_passed']
    if spec['mode']=='qualification':
        expected=spec['token_references'][role][name];batch=spec['batch_references'][role][name]
        check.update(lengths_match=a['lengths'].tolist()==expected['lengths'],
            token_hash_match=actual['token_ids_sha256']==expected['token_ids_sha256'],
            microbatches_match=actual['batch_micro_sequence_lengths']==batch['groups'],
            host_tensors_match=actual['host_tensor_sha256']==batch['host_tensor_sha256'])
        required+=['lengths_match','token_hash_match','microbatches_match','host_tensors_match']
        if name=='control':
            gold=np.load(Path(spec['control_directory'])/(role+'.npz'))
            assert np.array_equal(a['ids'],gold['ids'])
            difference=float(np.abs(a['probabilities'].astype(float)-gold['probabilities']).max())
            check.update(canonical_max_abs=difference,canonical_passed=difference<=1e-5)
            required.append('canonical_passed')
    else:
        # Hidden IDs originate in the current test.csv; no public predictions are read.
        assert name in ['warmup','test']
        assert 'token_references' not in spec and 'control_directory' not in spec
        check['valid_lengths']=bool(len(a['lengths'])==len(a['ids']) and (a['lengths']>0).all()
            and (a['lengths']<=summary['max_position_embeddings']).all())
        required.append('valid_lengths')
    check['passed']=all(check[k] for k in required)
    path=out/(role+'_checks.json');saved=json.loads(path.read_text()) if path.exists() else {}
    saved[name]=check;path.write_text(json.dumps(saved,indent=2)+'\n')
    if not check['passed']:
        (out/'engineering_stop.json').write_text(json.dumps(dict(role=role,cohort=name,mode=spec['mode'],
            checks=check,official_submission_eligible=False),indent=2)+'\n')
    print('ARENA_EVENT '+json.dumps(dict(stage='two_expert_cohort_checks',role=role,cohort=name,passed=check['passed'])),flush=True)
    return check['passed']
