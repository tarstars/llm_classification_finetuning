"""Build executed EDA notebooks and an evidence-based hypothesis report."""
import hashlib,json,sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
R=REPO/'results/eda_002'
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
setup='''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
REPO=next(p for p in [Path.cwd(),Path.cwd().parent,Path.home()/'prj/llm_classification_finetuning'] if (p/'results/eda_002').exists())
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
R=REPO/'results/eda_002'
FIG=ROOT/'reports/eda_002';FIG.mkdir(parents=True,exist_ok=True)
read=lambda name:json.loads((R/name).read_text())
dataset=read('dataset_summary.json');schema=read('schema.json')
plt.rcParams.update({'figure.figsize':(10,4),'axes.spines.top':False,'axes.spines.right':False})
def finish(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.svg'));plt.show()
'''


def write(name,cells):
    n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
    NotebookClient(n,timeout=240,kernel_name='arena-llm',resources={'metadata':{'path':str(REPO)}}).execute()
    nbf.write(n,REPO/'notebooks'/name)
    print('Executed',name,flush=True)


write('05_dataset_eda.ipynb',[
md('''# Dataset EDA: human preference comparisons

This is a comparison task: predicting human A/B/tie choices requires correctness,
instruction adherence, style and ambiguity judgments. This notebook describes the
provided labeled table, checks preparation, and identifies testable hypotheses.

Basic schema/counts use the supplied table. Target-association and duplicate-conflict
analyses use training + development, excluding confirmation. All model error analyses
are development-only. The previous confirmation score is historical, not a new test.
Raw text and per-row intermediate files stay under `~/database`; this notebook publishes
aggregate evidence. Three visible test rows cannot establish hidden-test distribution.
'''),code(setup),code('''display(pd.Series({k:dataset[k] for k in ['rows','unique_prompt_groups','prompt_groups_crossing_splits']}))
display(pd.Series(schema['missing_cells'],name='Missing CSV cells'))
display(pd.DataFrame({'train_columns':pd.Series(schema['columns']),'visible_test_columns':pd.Series(schema['test_columns'])}))
print('Visible test rows:',schema['visible_test_rows'])
'''),md('''## Labels and split integrity

The labels are fairly balanced. A constant majority choice is an accuracy baseline;
training frequencies are the appropriate probabilistic baseline. Repeated prompts
are grouped before splitting. Model identities and IDs are not deployed features.
'''),code('''counts=pd.Series(dataset['global_class_counts'])
counts.plot.bar(rot=0,color=['#2463a1','#2b9979','#bd8b30']);plt.ylabel('Rows');plt.title('Human preference labels');finish('target_counts')
display(pd.DataFrame(dataset['class_proportions_by_split']).T.rename(columns={'0':'A','1':'B','2':'Tie'}))
display(pd.Series(dataset['split_counts'],name='Partition sizes'))
assert dataset['prompt_groups_crossing_splits']==0
'''),md('''## Conversation structure, empty content and repetition

Null turns are preserved as empty text. Turn counts agree across the question and
both responses. Identical questions with different answers are not labeling conflicts.
An exact repeated comparison with different votes is stronger disagreement evidence,
but its small observed subset cannot estimate a global noise floor.
'''),code('''display(pd.Series(dataset['structural_counts'],name='Structural counts'))
display(pd.Series(dataset['null_turn_counts'],name='Null turn elements'))
display(pd.DataFrame(dataset['comparison_duplicates_train_dev']).T)
sizes=pd.Series(dataset['prompt_group_size_distribution'],dtype=float)
sizes.index=sizes.index.astype(int);sizes.sort_index().plot.bar(logy=True)
plt.xlabel('Rows per repeated prompt');plt.ylabel('Number of groups (log scale)');finish('prompt_group_sizes')
'''),md('''## Token lengths and retained content

The exact saved training tokenizer is used. The frozen packer reserves question and
answer fields and retains field beginnings/endings. A larger overall context still
leaves the 256-token question cap unless allocation also changes. Prospective
retention is not measured model quality.
'''),code('''q=pd.DataFrame(dataset['numeric_quantiles']).T
display(q[[c for c in ['0.5','0.9','0.95','0.99','1.0'] if c in q]].round(3))
ret=pd.DataFrame(read('retention.json'))
display(ret[ret.split=='dev'].round(4))
part=ret[ret.split=='dev'].set_index('packing')
part[['any_cut','prompt_cut','answer_cut']].plot.bar(rot=25)
plt.ylabel('Fraction of development examples');plt.title('Token loss under alternative packing budgets');finish('retention_policies')
'''),md('''## Languages and task types

Language labels are estimates from a local detector. Short or code-heavy prompts are
left unknown; the unknown group must not be treated as English or as non-English.
English regex task tags overlap and have false positives (for example, an animal
question mentioning “python”). Topic clusters use existing Qwen embeddings with
centroids fitted on training rows only; keywords are descriptive, not ground truth.
'''),code('''languages=pd.Series(dataset['language_counts_estimated']).sort_values(ascending=False)
display(languages.head(18).to_frame('Rows'))
display(pd.DataFrame(read('dataset_target_associations.json')).query("feature.str.startswith('task_')",engine='python'))
display(pd.DataFrame(dataset['prompt_clusters']))
'''),md('''## Preference associations and model metadata

Longer answers and refusal-like wording correlate with human choices, but these are
observational relationships: response quality and task difficulty also vary. Model
identities describe source coverage only; they are absent from the hidden-test input.
'''),code('''style=pd.DataFrame(read('style_calibration.json'))
length=style[style.feature=='answer_length_ratio'].set_index('bin')
length[['human_longer_win_rate','model_mean_longer_win_probability']].plot.bar(rot=20)
plt.ylabel('Win frequency / predicted probability');plt.title('Length associations on development');finish('length_preference')
display(pd.Series(read('model_identity_counts.json')).head(20).to_frame('Appearances in train + dev'))
display(read('identity_diagnostic.json'))
'''),md('''## What the dataset does and does not establish

No split overlap or conversation-length mismatch was found. Class imbalance and
rare exact-comparison conflicts are weak explanations of the remaining score gap.
Long and multi-turn comparisons are harder, but causal benefits from packing changes
need model experiments. Language identification and lexical task tags are imperfect.
See notebook 06 for paired model diagnostics, uncertainty and proposed experiments.
''')])

write('06_model_eda_and_hypotheses.ipynb',[
md('''# Frozen-model EDA and improvement hypotheses

The official Qwen3-4B submission scored **0.97025**. All new error analysis below uses
the 5,746 development examples; the previous confirmation result is not reused to
select improvements. Intervals resample entire prompt groups. Slice comparisons are
exploratory and are not corrected for multiple comparisons or prior model selection.
'''),code(setup+'''\nmodel=read('model_summary.json')
slices=pd.DataFrame(read('model_slices.json'))
diagnostic=read('inference_diagnostic.json')
post=read('postprocessing_diagnostics.json')
'''),code('''display(pd.DataFrame({k:model[k] for k in ['frozen_calibrated','raw_swap_average','prior','majority','uniform']}).T)
print('Development log-loss group-bootstrap interval:',model['log_loss_95_group_bootstrap'])
print('Development accuracy group-bootstrap interval:',model['accuracy_95_group_bootstrap'])
'''),md('''## Training duration and checkpoint selection

Training stopped for budget at step 877 / 0.61 epochs. The selected checkpoint was
800. Late selection loss was fairly flat, so incomplete training alone does not prove
large remaining gains. Logged training minibatches use augmentation and dropout;
their loss is not a clean train-versus-validation generalization-gap measurement.
'''),code('''history=pd.read_json(REPO/'results/attempt_002/qwen4b1024/training_progress.jsonl',lines=True)
training=history.dropna(subset=['loss']);validation=history.dropna(subset=['eval_log_loss'])
plt.plot(training.step,training.loss,alpha=.6,label='Training minibatches')
plt.plot(validation.step,validation.eval_log_loss,'o-',label='Fixed selection subset')
plt.xlabel('Optimizer step');plt.ylabel('Cross entropy');plt.legend();finish('training_curve')
display(validation[['step','epoch','eval_log_loss','eval_accuracy']])
'''),md('''## Class discrimination and calibration

Tie recall and AUC are lower than for A/B winners. Predicted tie frequency is close
to its prevalence, so this is not simply a minority-class problem. Temperature
scaling already repairs much of the overall underconfidence; this improvement is
included in the official result and cannot be counted again as a future gain.
'''),code('''display(pd.DataFrame(model['per_class']).round(4))
sns.heatmap(np.array(model['confusion_counts']),annot=True,fmt='d',xticklabels=['A','B','Tie'],yticklabels=['A','B','Tie'])
plt.xlabel('Predicted');plt.ylabel('Human label');finish('confusion_matrix')
for key,label in [('reliability_raw','Before temperature'),('reliability_calibrated','After temperature')]:
    bins=pd.DataFrame(model[key]['bins']);plt.plot(bins.confidence,bins.accuracy,'o-',label=label)
plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('Confidence');plt.ylabel('Observed accuracy');plt.legend();finish('reliability')
print('ECE before/after:',model['reliability_raw']['ece'],model['reliability_calibrated']['ece'])
print('Calibration log-loss gain:',model['calibration_gain'],model['calibration_gain_95_group_bootstrap'])
'''),md('''## Where errors concentrate

These overlapping groups differ in difficulty. In particular, multi-turn, code-like
and long-prompt slices often also lose context. The loss of untruncated examples is
already far above 0.8. Removing a few catastrophic examples is not a demonstrated
route to that target.
'''),code('''chosen=slices[(slices.rows>=100)&slices.feature.isin(['multi_turn','task_code','cut_1024_cap256','prompt_cut_1024_cap256','answer_embedding_cosine','language_group'])]
display(chosen[['feature','value','rows','log_loss','accuracy','tie_rate','truncation_rate','delta_ci_low','delta_ci_high']].round(4))
display(slices[slices.feature.isin(['multi_turn_x_truncation','task_code_x_truncation'])].round(4))
display(pd.DataFrame(model['loss_tail']))
display(pd.Series(model['confident_wrong']))
'''),md('''## Controlled inference diagnostics: answer order and context

Before inference, 512 development rows were sampled uniformly with seed 20261107,
without selecting on errors. Original and swapped predictions were recomputed with
the frozen adapter. Only changed inputs needed additional 2,048-token inference.
The training tokenizer, adapter hash and existing temperature were retained.

**The 512-row subset is easier than the full development set. Its absolute scores
are not new Kaggle estimates.** The paired changes and their intervals are the
relevant measurements. Swap averaging is already deployed; its observed gain is
not an additional available improvement. The context test uses the existing frozen
temperature and has no independently refitted long-context calibration.
'''),code('''display(pd.DataFrame({k:diagnostic[k] for k in ['raw_original','raw_swap_average','calibrated_original','calibrated_swap_average','calibrated_2048']}).T.round(5))
display(pd.Series({k:diagnostic[k] for k in ['rows','context_changed_rows','order_argmax_disagreement_fraction','order_flip_fraction_95_group_bootstrap','second_position_mean_probability_advantage','second_position_advantage_95_group_bootstrap','context_2048_gain','context_2048_gain_95_group_bootstrap','context_2048_accuracy_change','baseline_reload_max_absolute_probability_difference','inference_seconds','peak_gpu_bytes']}))
gains=[diagnostic['raw_swap_averaging_gain'],diagnostic['context_2048_gain']]
ci=np.array([diagnostic['raw_swap_averaging_gain_95_group_bootstrap'],diagnostic['context_2048_gain_95_group_bootstrap']])
plt.errorbar(gains,[0,1],xerr=np.array([np.array(gains)-ci[:,0],ci[:,1]-np.array(gains)]),fmt='o',capsize=4)
plt.axvline(0,color='gray',ls='--');plt.yticks([0,1],['Swap averaging (already used)','2,048-token pilot'])
plt.xlabel('Paired log-loss reduction; positive is better');finish('paired_inference_gains')
'''),md('''## Cheap postprocessing hypotheses

Five folds cross-fit only the calibration/blend parameters on fixed development
predictions. These are not out-of-fold language models: checkpoint selection already
used part of development. A tie-specific intercept preserves A/B swap symmetry.
The structural model was trained on the same training partition as LoRA. Both
exploratory gains have intervals spanning zero, so neither is a confirmed improvement.
'''),code('''display(pd.DataFrame({k:post[k] for k in ['calibration_oof','tie_calibration_oof','structural_blend_oof','structural_base']}).T.round(5))
display(pd.DataFrame(post['folds']).round(4))
print('Tie adjustment gain:',post['tie_calibration_gain'],post['tie_calibration_gain_95_group_bootstrap'])
print('Structural blend gain:',post['blend_gain'],post['blend_gain_95_group_bootstrap'])
'''),md('''## Qualitative review and hypotheses

The local private gallery contains selected difficult cases with complete text.
The table below contains paraphrases only. These deliberately selected examples
illustrate failure mechanisms; they cannot estimate their population frequency.
Two factual examples have primary-source checks. Subjective votes are not relabeled.
'''),code('''display(pd.DataFrame(read('manual_case_notes.json'))[['id','theme','observation','interpretation']])
print('Private full-text gallery:',ROOT/'eda/attempt_002/private_error_gallery.html')
display(pd.DataFrame(read('hypotheses.json')))
'''),md('''## Recommended experiments

The evidence prioritizes learning a more stable and semantically useful comparison:
paired-order training/consistency or a shared utility-plus-tie architecture; more
attention to subtle correctness and explicit instruction constraints; and a controlled
continuation to test convergence. Longer/round-aware packing remains a candidate,
but the small context pilot does not establish a large gain. Class weighting,
unverified label cleaning and additional scalar calibration are weak first priorities.

Use the development set for exploratory choices. The old confirmation partition has
already been evaluated. Any subsequent evaluation must disclose this history and
must not call it a new untouched confirmation. No new model was trained or submitted
as part of this EDA.
''')])

manifest={'frozen_model_hash':json.loads((REPO/'results/attempt_002/model_selection.json').read_text())['adapter_sha256'],
    'feature_fingerprint':json.loads((ROOT/'eda/attempt_002/feature_fingerprint.json').read_text()),
    'analysis_program_sha256':hashlib.sha256((REPO/'src/arena/eda_analysis.py').read_bytes()).hexdigest(),
    'development_rows':5746,'diagnostic_rows':512,'confirmation_used_for_new_model_diagnostics':False,
    'private_gallery':str(ROOT/'eda/attempt_002/private_error_gallery.html'),
    'notebooks':['05_dataset_eda.ipynb','06_model_eda_and_hypotheses.ipynb']}
json_write(R/'analysis_manifest.json',manifest)
