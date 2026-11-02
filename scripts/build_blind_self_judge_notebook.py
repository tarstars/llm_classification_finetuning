"""Publish aggregate evidence from the sealed 300-case self-judging experiment."""
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
result=json.loads((REPO/'results/attempt_004_self_judge/results.json').read_text())
rows={r['model']:r for r in result['models']}
mine=rows['codex_blind_subjective']
comparison=next(r for r in result['paired'] if r['against']=='gemma26_2048_original')
names={'codex_blind_subjective':'Codex: blind judgments',
       'gemma26_2048_original':'Gemma 26B-A4B: 2,048 tokens',
       'gemma26_1024_original':'Gemma 26B-A4B: 1,024 tokens',
       'qwen3_4b_lora_frozen':'Qwen3-4B: frozen LoRA', 'training_prior':'Training class prior'}
lines=['# Attempt 004: blind judging by the current Codex assistant','',
    f"**{mine['correct']}/300 correct ({mine['accuracy']:.2%})**, with a Wilson 95% interval of "
    f"[{mine['accuracy_wilson_95'][0]:.2%}, {mine['accuracy_wilson_95'][1]:.2%}].",'',
    'The current assistant personally read and judged all 300 conversations. No other model, subagent, web lookup or target feedback was used during judging.', '',
    '| Judge | Correct | Accuracy | Direct probability log loss | Five-fold temperature/tie log loss |',
    '| --- | ---: | ---: | ---: | ---: |']
for r in result['models']:
    cv=r.get('cross_fitted_temperature_tie',{}).get('log_loss')
    lines.append(f"| {names[r['model']]} | {r['correct']}/300 | {r['accuracy']:.2%} | {r['log_loss']:.5f} | {cv:.5f} |" if cv is not None else
                 f"| {names[r['model']]} | {r['correct']}/300 | {r['accuracy']:.2%} | {r['log_loss']:.5f} | — |")
lines += ['', 'Accuracy and direct log loss use the same frozen probabilities. The last column is a separate exploratory calibration, fitted in five prompt-group folds; its argmax accuracy may differ. Codex probabilities come from predeclared subjective confidence bins, not token likelihoods. Qwen probabilities already include the original frozen temperature.', '',
    f"Against 2,048-token Gemma, accuracy improves by **{comparison['accuracy_gain']*100:.2f} percentage points**, "
    f"with paired bootstrap 95% interval [{comparison['accuracy_gain_95'][0]*100:.2f}, {comparison['accuracy_gain_95'][1]*100:.2f}] points "
    f"and exact McNemar p={comparison['mcnemar_exact_p']:.3f}. The sample does not establish a reliable accuracy improvement.", '',
    f"For the user's single-alpha probability rule, the fitted optimum is **{mine['fixed_alpha_fitted_log_loss']:.5f}** "
    f"at alpha={mine['fixed_alpha_fit_same_rows']:.5f}; five-fold held-out alpha gives **{mine['fixed_alpha_cross_fitted_log_loss']:.5f}**. "
    'This result is far from the approximately 71.2% accuracy needed for 0.8 loss within that rule.', '',
    '## Error pattern', '',
    'Human labels contain 96 ties, but Codex predicts only 58. It recovers 35/96 ties (36.46%). '
    'Of 131 disagreements, 61 are missed human ties, 23 are predicted ties where the human selected a winner, and 47 reverse an A/B preference. '
    'On the 204 human-decisive examples Codex gets 134 correct (65.69%); Gemma gets 140 (68.63%). '
    'The net accuracy gain therefore comes from better tie agreement rather than higher agreement on decisive votes.', '',
    'Very-high-confidence judgments agree on 52/64 cases (81.25%); high-confidence judgments agree on 51/95 (53.68%). '
    'Confident correctness judgments can disagree with recorded preference votes. This does not prove that every disagreement is label noise; '
    'my own preference judgments and confidence estimates can also be wrong.', '',
    '## Blinding and audit', '',
    'Seed 20261116 selects 300 distinct prompt groups from the existing 512-example development sample. '
    'Examples included in the poster or prior manual reviews are excluded, leaving 496 eligible rows. '
    'Original row IDs and model names are hidden; A/B order is randomized. Complete conversations are shown, without extra truncation. '
    'During preparation, only ID/prompt-group metadata and text columns are read. Winner columns are first read by the scoring program after checking the committed seal.', '',
    f"The prediction seal was committed in **`{result['seal_commit'][:7]}`** before targets were loaded. "
    f"Judgment SHA-256: `{result['frozen']['judgments_sha256']}`.", '',
    'Predictions, full inputs, randomized-order mapping, reasons and row-level scored results remain in '
    '`~/database/llm_classification_finetuning/runs/attempt_004_self_judge/`. '
    'The repository contains aggregate evidence and source; no raw conversation text.', '',
    '## Interpretation and limits', '',
    'This tests the current assistant as a judge and does not isolate model size. It also changes reasoning, quantization, input length and judging protocol. '
    'The exact underlying assistant checkpoint is not established here. Public-dataset exposure during pretraining cannot be excluded. '
    'The sample was previously used for model development, and excluding inspected cases changes its distribution. '
    'These are paired exploratory results, not an official Kaggle score or a clean uncontaminated benchmark.', '',
    'Stronger general capability alone did not produce the hoped-for jump. The main next hypotheses are preference/rubric alignment, tie recognition, '
    'probability calibration and differing individual human preferences. The existing fine-tuned classifier remains competitive in log loss.', '',
    '[Executed notebook](notebooks/10_blind_self_judge.ipynb) · [Aggregate results](results/attempt_004_self_judge/results.json)', '',
    'Reproduce the sealed evaluation with `.venv/bin/python scripts/score_blind_self_judge.py`, then '
    '`.venv/bin/python scripts/build_blind_self_judge_notebook.py`. The scoring script verifies input, mapping and judgment hashes against the Git-committed seal.', '']
(REPO/'ATTEMPT_004_RESULTS.md').write_text('\n'.join(lines))
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[md('''# Attempt 004 — 300 blind judgments by this Codex assistant

All judgments and confidence bins were sealed and committed **before** human labels
were loaded. No other model, subagent or web lookup was used for judging. This is
a comparison of judging setups, with full conversations for Codex and truncated
input for Gemma, rather than an isolated test of model parameter count.
'''),code('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
repo=Path.cwd()
if repo.name=='notebooks':repo=repo.parent
r=json.loads((repo/'results/attempt_004_self_judge/results.json').read_text())
names={'codex_blind_subjective':'Codex blind','gemma26_2048_original':'Gemma 2048',
       'gemma26_1024_original':'Gemma 1024','qwen3_4b_lora_frozen':'Qwen3-4B LoRA','training_prior':'Class prior'}
print('Sealed before targets in commit:',r['seal_commit'])
print('Judgment hash:',r['frozen']['judgments_sha256'])
print('Human label counts A/B/tie:',r['human_label_counts'])
table=pd.DataFrame([{'judge':names[x['model']],'correct':x['correct'],'accuracy':x['accuracy'],
 'direct_log_loss':x['log_loss'],
 'CV_temperature_tie_loss':x.get('cross_fitted_temperature_tie',{}).get('log_loss'),
 'fixed_alpha_best_same_rows':x['fixed_alpha_fitted_log_loss'],
 'fixed_alpha_CV_loss':x['fixed_alpha_cross_fitted_log_loss']} for x in r['models']])
display(table)
assert len(r['models'])==5 and all(x['rows']==300 for x in r['models'])
assert r['models'][0]['correct']==169
'''),md('''Direct Codex probabilities use subjective confidence bins of 0.4, 0.5, 0.6,
0.7 and 0.8 for the chosen class, sharing the remainder equally. They are **not**
token likelihoods. The CV calibration column fits temperature and a tie intercept
on four folds and scores the fifth; all 300 prompt groups are distinct. Its argmax
accuracy may change. Qwen's direct probabilities use its original frozen calibration.
The best same-row alpha is optimistic; CV alpha instead uses each training fold's accuracy.
'''),code('''fig,ax=plt.subplots(figsize=(8,4))
models=r['models'][:-1]
values=np.array([x['accuracy'] for x in models])
ci=np.array([x['accuracy_wilson_95'] for x in models])
ax.errorbar(values,np.arange(len(models)),xerr=[values-ci[:,0],ci[:,1]-values],fmt='o',capsize=4)
ax.set_yticks(np.arange(len(models)),[names[x['model']] for x in models])
ax.axvline(.7119859693,color='tab:red',linestyle='--',label='Accuracy for 0.8 under fixed alpha')
ax.set(xlabel='Agreement with human votes; Wilson 95% intervals',xlim=(.4,.76))
ax.invert_yaxis();ax.legend(loc='lower right');fig.tight_layout()
figures=Path.home()/'database/llm_classification_finetuning/runs/attempt_004_self_judge/figures'
figures.mkdir(exist_ok=True)
fig.savefig(figures/'accuracy.png',dpi=160)
plt.show()
display(pd.DataFrame(r['paired'])[['against','accuracy_gain','accuracy_gain_95',
                                 'self_only_correct','other_only_correct','mcnemar_exact_p']])
'''),md('''The 3-point advantage over 2,048-token Gemma is uncertain: paired 95% interval
**−2.67 to +9.00 percentage points**, exact McNemar p=0.362. These data do not
establish a reliable accuracy improvement and do not approach the fixed-alpha 71.2% target.
'''),code('''cm=np.array(r['models'][0]['confusion_matrix'])
display(pd.DataFrame(cm,index=['Human A','Human B','Human tie'],columns=['Codex A','Codex B','Codex tie']))
missed_ties=int(cm[2,:2].sum()); extra_ties=int(cm[:2,2].sum())
wrong_winner=int(cm[0,1]+cm[1,0])
assert missed_ties+extra_ties+wrong_winner==300-r['models'][0]['correct']
display(pd.DataFrame({'error':['Missed human tie','Predicted tie; human chose winner','Reversed A/B vote'],
                      'rows':[missed_ties,extra_ties,wrong_winner]}))
display(pd.DataFrame(r['confidence_reliability']))
fig,ax=plt.subplots(figsize=(6,4))
reliability=r['confidence_reliability']
ax.plot([x['nominal_probability'] for x in reliability],[x['accuracy'] for x in reliability],'o-')
for x in reliability:ax.annotate(str(x['rows'])+' rows',(x['nominal_probability'],x['accuracy']),xytext=(4,5),textcoords='offset points')
ax.plot([0,1],[0,1],'--',color='gray')
ax.set(xlabel='Predeclared subjective confidence',ylabel='Observed vote agreement',xlim=(.3,.9),ylim=(.25,1.08))
fig.tight_layout();fig.savefig(figures/'confidence_reliability.png',dpi=160);plt.show()
'''),md('''Codex predicts 58 ties versus 96 recorded ties and recovers only 35 of those 96.
Missed ties account for 61 of 131 errors. On human-decisive cases Codex agrees on
134/204 (65.69%), versus Gemma's 140/204 (68.63%). The net gain comes from tie
recognition, despite weaker agreement on decisive votes.

These disagreements can reflect individual preferences, rubric mismatch, human
annotation noise **or mistakes by the current assistant**. They do not establish
which cause dominates. Judgments and reasons remain unchanged after unblinding.
'''),code('''print(json.dumps(r['protocol'],indent=2))
for note in r['notes']:print('-',note)
'''),md('''Full inputs, anonymization mapping, subjective judgments, reasons and scored
row-level results are under `~/database/llm_classification_finetuning/runs/attempt_004_self_judge`.
No raw conversations are included in this notebook. See `ATTEMPT_004_RESULTS.md`
and the prepare/show/record/freeze code in `scripts/blind_self_judge.py`.

The public dataset may have appeared in pretraining, the exact current assistant
checkpoint is not established, and the sample previously served model development.
Different context lengths, quantization and reasoning also limit causal claims about model level.
''')],metadata={'kernelspec':{'display_name':'Arena LLM (uv)','language':'python','name':'arena-llm'},'language_info':{'name':'python'}})
NotebookClient(nb,timeout=180,kernel_name='arena-llm',resources={'metadata':{'path':str(REPO)}}).execute()
nbf.write(nb,REPO/'notebooks/10_blind_self_judge.ipynb')
print('Updated attempt 004 report and executed notebook')
