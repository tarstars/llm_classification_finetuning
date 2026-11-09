"""Execute Notebook44 from the actual H16 coverage and blind detector audit."""
from pathlib import Path
import hashlib

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V32: refusal descriptors failed the blind detector gate

The proposed conditional preference correction was **not fitted**. Fixed cached
post-checkpoint cohorts had sufficient detected coverage, but a blind review of
32 original training-only cases found 17 clear refusals, two ambiguous cases and
five false positives among 24 positive detections. Even the optimistic 19/24
fails the required 20/24. Eight reviewed nondetections contained no opening
refusal; this small sample does not establish recall.

The detector confuses assistant refusal with instruction compliance, fictional
dialogue, physical explanations and capability caveats. This is a feature-validity
finding, not a measured preference-loss effect. No H16 preference targets were
loaded, no residual fit, no new GPU work or official submission. Therefore no
new model/baseline quality comparison is available; the early stop is preserved.

The original float32 row-sum assertion failure is retained separately. The
correction reused the original inference probability validator without changing
predictions. Historical source inspection and domain shift remain limitations.

[Protocol](../docs/SPAN_032_PLAN.md) · [Result](../docs/SPAN_032_RESULTS.md).
The rank objective remains 19 or better on the actual official leaderboard.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_032'
prep=json.loads((PUB/'preparation.json').read_text())
audit=json.loads((PUB/'detector_audit.json').read_text())
coverage=pd.DataFrame(prep['coverage']).T
coverage['one_sided_fraction']=coverage.one_sided/coverage.rows
display(coverage)
print('Successful preparation seconds/RSS GiB:',prep['cpu_elapsed_seconds'],prep['peak_rss_gib'])
print('Detector gate:',audit['detector_gate_passed'],'; decision:',audit['decision'])
print('Clear precision:',audit['clear_precision'],'; optimistic precision:',audit['optimistic_precision'])
print('Required true positives:',audit['required_true_positives'],'of',audit['audited_positive'])
print('Annotation uncertainty:',audit['annotation_uncertainty'])
display(pd.DataFrame({'false_positive_category':audit['false_positive_categories']}))
display(pd.DataFrame({'ambiguous_category':audit['ambiguous_categories']}))
fig,ax=plt.subplots(figsize=(6,3))
ax.bar(['Clear refusal','Ambiguous','False positive'],[audit['clear_refusals'],audit['ambiguous'],audit['false_positives']])
ax.set(ylabel='Cases',title='Blind training-only review of 24 positive detections')
plt.show()
for name in ['preparation_assertion_failure.json','resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))
print('No new log loss, accuracy, calibration or class-prior/hard-majority metrics: stop occurred before H16 labels/fitting. Do not substitute a different evaluation.')''')])
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nb = NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.get('outputs', [])) for c in nb.cells if c.cell_type == 'code')
    for name in ['preparation', 'detector_audit', 'resource_close']:
        nb.metadata['span032_' + name + '_sha256'] = hashlib.sha256((REPO / 'results/span_032' / (name + '.json')).read_bytes()).hexdigest()
    path = REPO / 'notebooks/44_refusal_detector_audit.ipynb'
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
