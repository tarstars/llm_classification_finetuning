"""Execute an explicitly interim native-classifier audit and asset notebook."""
from pathlib import Path
import json

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((REPO/'backlog/hypotheses.json').read_text())
    assert registry['active_version'] == 19, 'Preserve the historical notebook after closure'
    pub=REPO/'results/span_019'
    if (pub/'engineering_qualification.json').exists():
        gate=json.loads((pub/'engineering_qualification.json').read_text())
        stage=('FINAL engineering result: runtime gate '+('passed' if gate['engineering_gate_passed'] else 'FAILED')+
            '. Quality has not been evaluated.')
    elif (pub/'runtime_live_observation.json').exists():
        stage='INTERIM live runtime result: the nine-hour lower bound fails; terminal controls/audit pending. Quality has not been evaluated.'
    else:
        stage='INTERIM source and offline asset results. Quality and GPU runtime are unmeasured.'
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V19: accessible paired-classifier mirror

**'''+stage+'''**
Retain official 0.82987 / rank 23; the rank ≤19 goal remains active. This notebook
records the pinned mirror, verified head, input semantics, native matrix-layout
correction and available asset receipts. It does not estimate a competition score.

Candidate: [wath5/kgl_lmsys_pref_classif](https://huggingface.co/wath5/kgl_lmsys_pref_classif),
revision `cd24909db06583ecf00e65b6919c6cc1531c9885`. Original author-run lineage
is unverified; a card attribution or finite head does not authenticate training.
The fixed recipe and gates are in [protocol](../docs/SPAN_019_ASSET_PROTOCOL.md),
with [source results](../docs/SPAN_019_SOURCE_RESULTS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_019'
def read(name): return json.loads((PUB/name).read_text())
s=read('source_observation.json');c=read('native_cpu_controls.json');failure=read('native_cpu_attempt_1.json')
print('Pinned source:',s['repository'],s['revision'])
print('Tensor keys:',s['tensor_keys'],'; finite classifier shape:',s['classifier_head_shape'])
print('Config pad:',s['pad_token_id'],'; effective native pad:',c['effective_native_pad_token_id'])
print('EOS:',c['eos_token_id'],'; padding:',c['padding_side'],'; max input tokens:',c['maximum_input_tokens'])
display(pd.DataFrame(c['synthetic_input_checks']))
display(pd.DataFrame(c['synthetic_native_GPTQ_arithmetic']))
print('Preserved first failure: ordinary-layout synthetic projection max difference',failure['linear_max_abs'])
print('Weights were bit-identical:',failure['max_abs_weight']==0)
print('Correct native-layout synthetic projections match exactly. Whole-model GPU inference is a separate stage below.')
fixtures=['Empty','Short','Newline','Specials','Long Q','Long A','Long B','Long A + pad']
rows=c['synthetic_input_checks'];x=list(range(len(rows)))
fig,ax=plt.subplots(figsize=(10,3.5))
ax.bar([i-.18 for i in x],[r['tokens'] for r in rows],.36,label='Actual tokens')
ax.bar([i+.18 for i in x],[r['padded_tokens'] for r in rows],.36,label='Native padded length')
ax.set_xticks(x,fixtures,rotation=15);ax.set_ylabel('Tokens, synthetic inputs');ax.legend()
ax.set_title('Padding and appended EOS are part of the declared native reference')
fig.tight_layout();plt.show()
assert c['passed'] and not c['real_model_logits_compared'] and not s['quality_measured']'''),
        nbf.v4.new_code_cell('''selection=read('evaluation_selection.json')
print('Fixed evaluation counts:',{k:len(selection[k]) for k in ['control_ids','uniform_ids','long_ids','later_ids']})
print('Targets used for selection:',selection['uses_targets_for_selection'])
print('Original votes overlap checkpoint training; later cohort previously inspected and differs in domain.')
if (PUB/'token_reference.json').exists():
 tokens=read('token_reference.json')
 display(pd.DataFrame([{'cohort':name,**r} for name,r in tokens['cohorts'].items()])[
  ['cohort','rows','mean_tokens','max_tokens','seconds']])
 print('Token/absolute-position/pool fingerprints fixed before GPU inference; no winner labels read.')
if (PUB/'worker_cpu_validation.json').exists():
 checks=read('worker_cpu_validation.json')
 print('Tiny saved-model tests:',checks['native_test_count'],'; pipeline tests:',checks['pipeline_test_count'])
 print('CPU runtime:',checks['runtime'])
 print('Selected production backend:',checks['production_backend'])
print('Real GPU fit/speed verified at the earlier CPU preparation stage:',checks['production_backend_memory_speed_verified'])
for filename in ['weight_acquisition.json','dense_expansion.json']:
 if (PUB/filename).exists():
  r=read(filename)
  print(filename,{k:r[k] for k in ['complete','total_weight_file_bytes','tensor_keys','total_tensor_bytes',
    'head_bytes_identical','runtime_native_layout_required','new_gpu_executions','quality_measured','gpu_admitted'] if k in r})
 else: print(filename,'not complete at this notebook snapshot')
print('No training or new quantization. Offline source/asset checks do not admit a GPU or official submission.')
print('Separate GPU engineering evidence appears below when available; no quality conclusion follows from CPU checks.')'''),
        nbf.v4.new_code_cell('''for name in ['asset_upload.json','upload_live_observation.json']:
 if (PUB/name).exists():
  r=read(name)
  print(name,{k:r[k] for k in ['at','dataset','ready','status','process_exists','last_observed_open_file',
    'last_observed_file_read_position_bytes','file_total_bytes','service_ready_unproven'] if k in r})
if (PUB/'quality_preparation.json').exists():
 q=read('quality_preparation.json')
 display(pd.DataFrame([{'cohort':name,'total_rows':r['rows'],'routed_rows':r['selected_rows'],
  'routed_work_fraction':r['work_fraction']} for name,r in q['routes'].items()]))
 print('Comparator routes frozen from verified old predictions; winner labels used:',q['uses_winner_labels'])
if (PUB/'classifier_followup_cell_rehearsals.json').exists():
 r=read('classifier_followup_cell_rehearsals.json')
 print('Followup notebook local checks passed:',r['passed'])
 print('These are CPU orchestration checks; real model GPU exercised:',r['real_model_gpu_exercised'])
print('The separate followup is conditional on the initial engineering/runtime gate; no new score is available.')'''),
        nbf.v4.new_code_cell('''if (PUB/'engineering_qualification.json').exists():
 e=read('engineering_qualification.json');r=e['runtime']
 print('Terminal engineering checks passed:',e['engineering_checks_passed'])
 print('Native numerical controls:',e['native_controls'])
 print('Retained canonical probability max difference:',e['retained_canonical_probability_max_abs'])
 print('Conservative hidden runtime hours:',r['projected_hidden_seconds']/3600)
 print('Runtime gate passed:',e['engineering_gate_passed'])
 print('Human quality votes read:',e['human_votes_read'])
 display(pd.DataFrame([{'expert':'Retained Gemma','seconds_per_row':r['retained_seconds_per_row']},
  {'expert':'Mirror, both replicas wall time','seconds_per_row':r['mirror_parallel_seconds_per_row']}]))
 fig,ax=plt.subplots(figsize=(7,3))
 ax.barh(['Nine-hour limit','Conservative candidate'],[9,r['projected_hidden_seconds']/3600],color=['gray','tab:orange'])
 ax.set_xlabel('Elapsed hours for 25,000 rows');fig.tight_layout();plt.show()
elif (PUB/'runtime_live_observation.json').exists():
 r=read('runtime_live_observation.json')
 print('LIVE worker mirror profile:',r['mirror_profile_rows'],'rows in',r['mirror_profile_seconds'],'seconds')
 print('Hidden runtime LOWER BOUND hours:',r['lower_bound_hours'],'; startup/warmup/other costs omitted')
 print('Terminal artifact/canonical/memory audit pending:',r['final_engineering_audit_pending'])
 print('The lower bound already exceeds nine hours; this backend cannot advance to the quality followup.')
else: print('No GPU timing evidence at this snapshot.')
print('Official best remains0.82987/rank23. No new competition score or honest model-quality estimate.')'''),
    ])
    nb.metadata['kernelspec'] = dict(display_name='Python 3',language='python',name='python3')
    result = NotebookClient(nb,timeout=120,kernel_name='python3',
        resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None for c in result.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in result.cells if c.cell_type=='code' for o in c.get('outputs',[]))
    path=REPO/'notebooks/31_classifier_mirror_audit.ipynb'
    nbf.write(result,path)
    print(path)


if __name__ == '__main__':
    main()
