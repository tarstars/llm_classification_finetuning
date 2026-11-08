"""Label-free token/control checks and the early H72 runtime stop."""
from pathlib import Path
import json
import numpy as np

if __package__:
    from .span026_runtime import projection
else:
    from context_runtime import projection

CONTEXT_COHORTS={'warmup','control','uniform','context_warmup','context_probe',
    'changed_old','changed_extended'}


def conversation_budget(role,name):
    if role not in ['m0','m3','m0_reverse'] or name not in CONTEXT_COHORTS:
        raise ValueError('Unknown role/cohort')
    return 6144 if role=='m0' and name in ['context_warmup','context_probe','changed_extended'] else 4096


def after_cohort(args,summary,name):
    spec=json.loads(Path(args.spec).read_text());role=args.role;out=Path(args.out)
    expected=spec['token_references'][role][name];actual=summary['cohorts'][name]
    array=np.load(out/(role+'_'+name+'.npz'))
    assert array['ids'].tolist()==expected['ids']
    assert array['lengths'].tolist()==expected['lengths']
    assert actual['token_ids_sha256']==expected['token_ids_sha256']
    assert json.loads((out/(role+'_checkpoint.json')).read_text())==spec['checkpoint_references'][role]
    if name=='control':
        reference=np.load(Path(spec['control_directory'])/(role+'.npz'))
        assert np.array_equal(array['ids'],reference['ids'])
        diff=float(np.max(np.abs(array['probabilities']-reference['probabilities'])))
        assert diff<=1e-5,'Preserve failed canonical numerical control'
        (out/(role+'_control_check.json')).write_text(json.dumps(dict(passed=True,rows=len(array['ids']),max_abs_probability_difference=diff)))
    if role=='m0' and name=='context_probe':
        import torch
        parent=spec['parent_runtime'];timings=summary['cohorts']
        startup=max(0,summary['cold_seconds']-parent['g_cold_seconds'])+max(0,timings['warmup']['seconds']-parent['g_warmup_seconds'])
        result=projection(parent['projection_seconds'],parent['g_uniform_seconds'],parent['worst_g_uniform_seconds'],
            timings['uniform']['seconds'],actual['seconds'],actual['rows'],spec['changed_fraction'],
            timings['context_warmup']['seconds'],spec['runtime_selector_seconds'],extra_parent_startup_seconds=startup)
        memory=[actual['peak_allocated_bytes'][i]/torch.cuda.get_device_properties(i).total_memory for i in range(2)]
        result.update(memory_allocated_fractions=memory,memory_guard_passed=all(x<=.95 for x in memory),
            original_tokens=4096,extended_tokens=6144,rows=actual['rows'],control_passed=True,
            token_references_passed=True,information_complete=False)
        result['engineering_screen_passed']=result['engineering_screen_passed'] and result['memory_guard_passed']
        (out/'engineering_probe.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ARENA_EVENT '+json.dumps(dict(stage='context_engineering_probe',passed=result['engineering_screen_passed'],
            seconds=result['projection_seconds'],role=role,cohort=name)),flush=True)
        return result['engineering_screen_passed']
    return True
