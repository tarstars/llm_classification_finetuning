"""Execute the current H56 research notebook; refresh only while v16 is active."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO / 'backlog/hypotheses.json')['active_version'] == 16, 'Preserve historical review'
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V16: frozen preference rewards and later human votes

**Source stage passed; GPU quality is unmeasured until an actual run completes.**
Retain official0.82987/rank23; the rank<=19objective remains active.

Replace one expert with a pretrained scalar reward head and fit only a symmetric
three-parameter A/B/tie mapping. Native conversations preserve turns; each answer
is evaluated independently. Main8B/control0.6B, fixed protocol and weights.
The fresh1024groups postdate both published reward checkpoints and exclude
previously inspected groups. This is chronological source evidence, not Kaggle CV.
Text can preexist its new vote, and newer users/models may not represent hidden
competition data. See [the protocol](../docs/SPAN_016_PILOT_PROTOCOL.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_016'
def read(name):return json.loads((PUB/name).read_text())
q=read('source_observation.json');audit=read('source_audit.json')
display(pd.DataFrame([{'size':m['size'],'repository':m['repository'],'revision':m['revision'],
 'weight_bytes':m['total_weight_bytes'],'license':m['license'],'gated':m['gated'],
 'scalar_outputs':len(m['config']['id2label'])} for m in q['models']]))
print('Source stage passed:',audit['source_stage_passed'])
print('Source-stage GPU admission:',audit['gpu_admitted'])'''),
        nbf.v4.new_code_cell('''cohort=read('matched_cohort_manifest.json') if (PUB/'matched_cohort_manifest.json').exists() else read('cohort_manifest.json')
display(pd.DataFrame([{k:cohort.get(k) for k in ['rows','available_groups','excluded_groups',
 'post_publication_cutoff','timestamp_range','selected_bin_counts']}]))
assert cohort['targets_read_after_durable_selection'] and not cohort.get('uses_targets_for_selection',cohort.get('selection_uses_targets',False))
if (PUB/'matched_token_reference.json').exists():
 initial=read('token_reference.json');matched=read('matched_token_reference.json')
 display(pd.DataFrame([{'population':'initial unused source','tokens_per_pair':initial['cohorts']['long']['mean_pair_tokens'],'work_ratio':initial['source_to_target_reward_work_ratio']},
 {'population':'actual matched source','tokens_per_pair':matched['mean_pair_tokens'],'work_ratio':matched['source_to_target_reward_work_ratio']},
 {'population':'competition runtime512','tokens_per_pair':initial['cohorts']['uniform']['mean_pair_tokens'],'work_ratio':1.}]))
native=read('pinned_runtime_cpu_validation.json')
display(pd.DataFrame([{k:native[k] for k in ['passed','transformers','tokenizers','huggingface_hub','safetensors']}]))
print('Native rendering matches the card; prefix truncation retains final EOS/newline.')
print('Swap symmetry, gradients and held-out-group isolation were tested on CPU.')'''),
        nbf.v4.new_code_cell('''rehearsal=read('reward_pilot_cell_rehearsals.json')
print('Actual serialized worker rehearsals passed:',rehearsal['passed'])
print('Actual pipeline CPU bookkeeping tests:',rehearsal['actual_pipeline_tests'])
for name in ['asset_preparation.json','asset_upload.json','matched_input_upload.json',
 'jobs/reward_pilot/status.json','reward_pilot_source_verification.json']:
 if (PUB/name).exists():
  print(name);display(pd.DataFrame([read(name)]))
 else:print(name,': pending; no corresponding result is claimed')
if (PUB/'H56_information.json').exists():
 info=read('H56_information.json')
 display(pd.DataFrame([{'model':name,'log_loss':m['log_loss'],'accuracy':m['accuracy'],
  **m['class_log_loss']} for name,m in info['models'].items()]))
 display(pd.DataFrame([{k:info[k] for k in ['loss_reduction','paired_group95',
  'information_gate_passed','stronger_quality_gate_passed','official_submission_eligible','decision']}]))
else:print('H56_information.json: pending; actual model quality has not been audited.')
print('A source/CPU check is not a GPU quality estimate or a leaderboard improvement.')''')])
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'))
    result = NotebookClient(nb, timeout=120, kernel_name='python3',
        resources={'metadata': {'path': str(REPO)}}).execute()
    assert not any(o.output_type == 'error' for c in result.cells for o in c.get('outputs', []))
    path = REPO / 'notebooks/28_modern_reward_expert.ipynb'
    nbf.write(result, path)
    print(path)


if __name__ == '__main__':
    main()
