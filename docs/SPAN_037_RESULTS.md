# V37: no compatible classifier pair established in the bounded audit

H80 stops at its source gate. Three new public candidates were inspected, but
none establishes the exact base, classifier head, class mapping and checkpoint
lineage needed for the frozen weight-average proposal. No model weights, task
labels, new quality fitting, GPU execution or official submission occurred.
This is an inconclusive compatibility result, not a measured quality failure
or a claim that no suitable public model exists.

The retained Gemma2 reference has a two-layer MLP head with four tensors:
score.1.weight/bias and score.4.weight/bias. It is a full FP16 Gemma2-9B state,
42 layers, hidden3584, vocabulary256000. The current Llama expert has a different
architecture and cannot be averaged with it. V23 had already established that
the retained author's release has no additional public 9B checkpoint; those
known missing releases and prior401/403/lineage stops were not reopened.

## Actual observations

One bounded catalog stage made two HF and two Kaggle queries. HF lists are
limited to100results; Kaggle lists to20. A truncated list is not a complete
inventory of models. Before candidate-file reads, exactly three named new
candidates were frozen. Primary metadata, small configs and author notebook
code were saved under the heavy project root; notebook outputs were discarded.

| Candidate | What is available | Remaining requirement |
| --- | --- | --- |
| [RoschildRui/LMSYS_Gemma2_0728_0905](https://huggingface.co/RoschildRui/LMSYS_Gemma2_0728_0905) | Exact revision a82777dad6a02a76b555c2b448c938519112aecd, six model archives, no standalone model card/config | Exact base, head/class order, license and archive-to-training-code lineage unproved. Two relevant same-handle GitHub repositories are forks of retained second/third-place solutions; neither establishes these archives. |
| [Human Bias weights](https://www.kaggle.com/datasets/kishanvavdara/lmsys-gemma-9b-weightsteam-human-bias) | Public release contains two .pth files; description empty and dataset license unknown | Retrieved author's Part1/Part2 notebooks cover frozen embeddings plus CatBoost. They do not document loading the final-team state, its head, exact base or merge lineage. Their label order cannot be assigned to unrelated weight files. |
| [Justin Gemma2 adapters](https://www.kaggle.com/datasets/justin1357/lmsys-gemma2-lora-weights/versions/19) | Public Apache2 dataset v19, several adapter checkpoints; configs save classifier/score, use rank64 and a local 4-bit base path | Base revision is null, model card is an unfilled template, native head layout/class order/serialization and exact initialization unproved. The bounded public author notebook listing contains no matching source. |

The adapter's presence does not prove that its head matches the incumbent MLP.
No tensor parameters were downloaded to guess shapes, no labels were used to
choose a class permutation, and no generic tag or title was promoted into
checkpoint lineage. Availability was established; compatibility was not.

## Decision and accounting

Do not create a weight average or substitute a backbone, head, base revision or
mirror under H80. Stop without the conditional numerical/GPU phase. There is no
candidate loss, gain, interval or class metric to report; repeating old baseline
numbers would not be a new experiment. The existing baseline results remain in
Notebook48 and their original reports.

All observed source bytes and SHA hashes are in the aggregate result. Fresh
terminal resource/history/board accounting is recorded separately, preserving
the original November14quota anchor and
2026-11-10T08:37:37.343197UTC deadline. Expected new GPU charge and official
count are zero and were checked against the live service. Saved source bytes
total592122(0.565MiB). Fresh13:41UTCNov9board is24/309,best.82987,
rank19/20cutoff.82915and leader.82665. Quota remains38277.944sused/
69722.056sfree,0reserved/paidFalse; all8officials and lastv36pilot are terminal.
The source-only close does not refresh the entire owned catalog; the last full
inventory is v36and a fresh full admission remains required before any future GPU.

Carry35unfinished tasks into v38, with no automatic source/model rescue queued.
Preserve146historical snapshots, exact goals, user notebooks07/12and notebook
link. H80done means the bounded audit finished; the persistent rank<=19goal
remains unachieved.

[Protocol](SPAN_037_PLAN.md) · [Frozen candidates](../results/span_037/candidates.json) ·
[Source result](../results/span_037/H80_results.json) ·
[Author trace](../results/span_037/author_trace.json) ·
[Executed Notebook49](../notebooks/49_classifier_weight_compatibility.ipynb).
