"""Decompose saved NLL changes into single-view quality and averaging benefit."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=REPO/'results/span_007/monitor_alignment.json')
    parser.add_argument('--output',type=Path,default=REPO/'results/span_007/order_averaging_decomposition.json');args=parser.parse_args()
    assert not args.output.exists(),'Preserve the first descriptive result'
    values={}
    for cohort,modes in read_json(args.input)['cohorts'].items():
        raw=modes['order_average_raw'];first=modes['original_raw'];second=modes['aligned_swap_raw']
        view_before=(first['reference_log_loss']+second['reference_log_loss'])/2
        view_after=(first['candidate_log_loss']+second['candidate_log_loss'])/2
        j_before=view_before-raw['reference_log_loss'];j_after=view_after-raw['candidate_log_loss']
        assert min(j_before,j_after)>=-1e-12
        view_gain=view_before-view_after;averaging_gain=j_after-j_before;total=raw['reference_log_loss']-raw['candidate_log_loss']
        assert abs(total-view_gain-averaging_gain)<1e-12
        values[cohort]={'mean_single_view_loss_before':view_before,'mean_single_view_loss_after':view_after,
            'reference_averaging_benefit':j_before,'candidate_averaging_benefit':j_after,
            'mean_single_view_gain':view_gain,'change_in_averaging_benefit':averaging_gain,'order_averaged_gain':total}
    result={'at':utc_now(),'descriptive_identity_only':True,'cohorts':values,
            'interpretation':'A Jensen-gap identity conditional on saved predictions; not causal evidence that more disagreement improves quality.',
            'input_sha256':sha256(args.input),'source_sha256':sha256(Path(__file__))}
    atomic_json(args.output,result);print(values)


if __name__=='__main__':main()
