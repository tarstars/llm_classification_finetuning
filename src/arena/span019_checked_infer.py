"""Preserve native arithmetic and additionally check repeated raw logits."""
import sys
from pathlib import Path

import numpy as np


def repeated_logit_guard(forward, report):
    reference=None
    def guarded(models, entries, cohort):
        nonlocal reference
        logits,profile=forward(models,entries,cohort)
        if cohort=='parallel_reference':
            reference=logits.copy()
        elif cohort=='parallel_repeat':
            if reference is None or reference.shape!=logits.shape:
                raise ValueError('Repeated logits lack their matched reference')
            difference=float(np.max(np.abs(reference-logits)))
            passed=bool(np.isfinite(difference) and difference<=1e-5)
            report(dict(passed=passed,raw_logit_max_abs=difference,tolerance=1e-5,rows=len(logits)))
            if not passed:raise ValueError('Repeated raw logits differ from their matched reference')
        return logits,profile
    return guarded


if __name__=='__main__':
    from . import span019_infer as core
    output=Path(sys.argv[sys.argv.index('--out')+1])
    core.forward=repeated_logit_guard(core.forward,lambda result:core.save(output/'mirror_repeat_logits.json',result))
    core.main()
