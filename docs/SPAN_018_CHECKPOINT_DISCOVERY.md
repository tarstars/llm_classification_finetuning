# A newly accessible native three-class checkpoint

H64's fixed teacher failed its information gate. Read-only planning then found
[wath5/kgl_lmsys_pref_classif](https://huggingface.co/wath5/kgl_lmsys_pref_classif),
pinned to `cd24909db06583ecf00e65b6919c6cc1531c9885`.
This is a different artifact from H55's earlier Kaggle merge. Its public card
credits the BlackPearl author, links that author's repository and includes a
three-class inference example with an explicit A/B/tie mapping. These are mirror
claims, not independently authenticated final-author lineage or performance.

Public metadata reports an ungated 42-layer Gemma2 classifier. Its config names
the later `v7_dpo_16bit_01234_8bit_all` parent; the weight index contains
`score.weight` in shard 3. Three GPTQ 8-bit shards declare 10,361,028,608 tensor
bytes, with public immutable LFS hashes. The card uses a 2000-token input with an
appended EOS and a custom last-non-padding classifier. Both configured pad and
EOS IDs are 1, so pooling/tokenization must be reproduced carefully. Presence
in the index alone does not verify the head's shape, values or training.

Aggregate evidence is in `results/span_018/next_candidate_discovery.json`.
Small configs and metadata, including quantization calibration text, stay under
ROOT/runs/span_018/next_candidate_discovery and are not committed. No checkpoint
weights, new model outputs or GPU work were obtained. No quality estimate is
available. The model card does not declare a separate checkpoint license;
applicable Gemma terms and mirror provenance need to be recorded in the audit.

Queue a distinct follow-up: inspect the pinned card, tensor headers and native
loading/quantization path; freeze one declared mirror recipe before any quality
lookup. Do not retry the private Kaggle dataset or silently substitute the old
parent. Compare one fixed replacement ensemble against H60 with matched base
controls. Original votes overlap competition training, and later votes differ
from the competition domain. Report both limits. Actual two-T4 memory, numerical
identity and conservative full-submission runtime remain advancement gates.
An accessible artifact is a reason to test, not evidence of score improvement.
