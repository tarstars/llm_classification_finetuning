"""Display actual H71pilot/quality state; never launch inference from a notebook."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# H71: targeted identical-answer audit

275 exact raw-identical pairs, fixed published candidate probabilities
`[.06,.06,.88]`, all other rows retain H60. Original votes overlap checkpoint
training: any local result is exploratory, not competition CV or a hidden-score
estimate. An information pilot cannot be submitted. [Plan](../docs/SPAN_025_PLAN.md)
· [Progress](../docs/SPAN_025_PROGRESS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_025'
cohort=json.loads((PUB/'cohort_audit.json').read_text())
display(pd.DataFrame([{'model':k,'rows':v['rows'],'mean_tokens':v['mean_tokens'],'max_tokens':v['max_tokens'],
 'control_tokens_exact':v['control_identity'],'swapped_tokens_identical':v['all_swapped_tokens_identical']}
 for k,v in cohort['roles'].items()]))
job=PUB/'jobs/identical_pilot'
for name in ['status.json','controller_state.json','progress.json']:
 if (job/name).exists():print(name,json.loads((job/name).read_text()))
quality=PUB/'information_audit.json'
if quality.exists():
 r=json.loads(quality.read_text())
 print('Decision:',r['decision'],'; official eligible:',r['official_submission_eligible'])
 print('Full-data weighted lower / upper gain:',r['weighted_lower_gain'],r['weighted_upper_gain'])
 print('Paired group95 lower-gain interval:',r['weighted_lower_gain_paired_group95'])
 display(pd.DataFrame([{'candidate_identical_loss':r['candidate_identical_log_loss'],
 'offroute_identical_loss':r['offroute_identical_log_loss'],'onroute_identical_loss':r['onroute_identical_log_loss']}]))
 print('Class counts A/B/Tie:',r['class_counts'])
 print('Canonical controls:',r['controls'])
else:
 print('Quality is not yet measured. No independent CV or competition score is available from this pilot.')
official=PUB/'official_result.json'
if official.exists():print('Actual official result:',json.loads(official.read_text()))
else:print('No official submission result for this span.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in executed.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    path=REPO/'notebooks/37_identical_pair_audit.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
