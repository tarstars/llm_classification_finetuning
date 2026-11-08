"""Explicit cohort hooks around the frozen H72 native forward implementation."""
from pathlib import Path
import argparse
import json

import numpy as np

if __package__:
    from . import span026_engine as native
else:
    import context_engine as native


def budget(role,name):
    if role not in ['m0','m3','m0_reverse']:
        raise ValueError('Unknown model role')
    base=name in ['warmup','control','uniform','long','test']
    reverse=name.startswith('routed_') and name[7:] in ['control','uniform','long','test']
    extra=name=='context_warmup' or (name.startswith('extended_') and name[9:] in ['control','uniform','long','test'])
    if (role=='m0' and (base or extra)) or (role=='m3' and base) or (role=='m0_reverse' and (name=='warmup' or reverse)):
        return 6144 if extra else 4096
    raise ValueError('Unknown model/cohort combination')


def verify_cohort(args,summary,name):
    spec=json.loads(Path(args.spec).read_text());out=Path(args.out);role=args.role
    assert json.loads((out/(role+'_checkpoint.json')).read_text())==spec['checkpoint_references'][role]
    a=np.load(out/(role+'_'+name+'.npz'))
    assert a['ids'].tolist()==spec['actual_ids'][role][name]
    native.validate_probabilities(a['probabilities'],len(a['ids']))
    if spec['qualify']:
        reference=spec['token_references'][role][name]
        assert a['ids'].tolist()==reference['ids'] and a['lengths'].tolist()==reference['lengths']
        assert summary['cohorts'][name]['token_ids_sha256']==reference['token_ids_sha256']
    return True


def main():
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out','spec','role','cohorts']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();names=args.cohorts.split(',')
    for name in names:budget(args.role,name)
    # These three globals are the frozen engine's only configurable hooks.
    # Its model loader, precision, sharding, collator and forward code are reused.
    native.CONTEXT_COHORTS=set(names)
    native.conversation_budget=budget
    native.after_cohort=verify_cohort
    native.infer_role(args)


if __name__=='__main__':main()
