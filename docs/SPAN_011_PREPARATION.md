# H59: later-vote cohort prepared without label selection

On2026-11-07 the exact seven public Arena140k Parquet shards at revision
6322995ab34d7c2693e3f47dd13fa5caa0789a74 finished downloading within the original
30minute deadline. Their1613803705compressed bytes matched all pinned LFS hashes.
No raw conversations, labels or per-row outputs are published in this repository.

Streaming135634rows yielded49118eligible English, first-evaluation, single-turn
text pairs after rejecting64459language,13416later-evaluation,8593turn-count,
47nontext/empty and1known competition-pair overlap. Overlap hashing is invariant
to A/Bswap and preserves turn boundaries. Model inputs retain the actual order
and original text. Prompt/session connected components leave45703eligible groups.
The fixed representative and group-sampling hashes are separate, so repeated
votes do not increase a group's selection probability.

All2048predeclared length-bin quotas were filled after tokenizing7493group
representatives. The deterministic split has1024learning and1024confirmation
groups with no shared group, session, normalized prompt, UUID or numeric ID.
Selected votes run from2025-04-17 to2025-07-24, after both frozen checkpoint
versions were published on2024-08-05. This establishes chronology for these
specific supervised checkpoints, not independent competition CV or independence
from every possible pretraining corpus. Transfer across later model populations
remains an empirical risk. Match on length does not establish identical domains.

The actual cached author Gemma tokenizer and ProcessorPAB reproduced the accepted
256row CPU/GPU token fingerprint exactly. Mean selected prepared length947.388
tokens compares with926.573for the full original training population. Frozen
bin counts are488,495,438,413,111,66,22,15. Processing used108.358seconds and
1083514880peak RSS bytes, two threads. Download plus preparation stayed below
the one-hour review bound. All heavy artifacts are under
database/llm_classification_finetuning/runs/span_011.

Targets were interpreted only after selection/splitting, then saved separately.
Neither learning nor confirmation outcomes have been inspected at this stage.
GPU input Parquet contains exactly id,prompt,response_a,response_b, using plain
Arrow types compatible with the pinned older Kaggle image. It contains no votes,
model identities, source tags or predictions. Confirmation assets are not uploaded
until the learning temperature is frozen.

The authenticated Kaggle page refresh again confirmed that public external data
and pretrained models are permitted, internet must be disabled, and GPU notebooks
are limited to9hours. The rules forbid hand-labeling; this experiment uses
existing published human votes. Source attribution and terms accompany the
private text-only Kaggle input asset. See the official
[code requirements](https://www.kaggle.com/competitions/llm-classification-finetuning/code-requirements)
and [rules](https://www.kaggle.com/competitions/llm-classification-finetuning/rules).

Seven focused tests passed, covering label-blind conversion, rejection and
grouping, plain input schema, temperature argmax/swap behavior, and the actual
learning/confirmation/hidden orchestration with a CPU worker fixture. The fixture
caught an event-field collision; it was fixed before any GPU source was accepted.
Both public and hidden paths perform exactly the two original expert passes.
The calibration stage and actual runtime/identity checks remain unmeasured.

Evidence: results/span_011/source_download.json,
results/span_011/cohort_manifest.json, tests/test_span011_data.py and
tests/test_span011_calibration.py. The phase admission records quota, original
48hour deadline and a conservative label-free pilot forecast separately in
results/span_011/span_plan.json. No local quality or official gain is claimed.

While the exact accepted pilot remained QUEUED, the official controller was
prepared in scripts/submit_span_011_calibration.py. It requires an independently
confirmed loss reduction>=.001 with paired95%lower>0, a completed class/length/
domain review, matching frozen source/temperature/evidence/output hashes and the
original numerical/runtime gates. A CPU admission test passed for a valid witness
and rejected absent domain review, insufficient point gain, an interval crossing
zero, stale evidence and a changed output file. Fresh quota/jobs and a committed
controller are required before a mutation. An accepted receipt is monitored
without re-submission. No deployment decision or official intent exists yet;
these checks are preparation, not evidence of calibration quality.

The exact accepted pilot entered RUNNING at10:55:53UTC after roughly14minutes
queued. Runtime audit now explicitly adds the observed session time outside all
model/cold/cohort timers to the projected startup cost, before the separate600s
reserve. This covers package setup, process imports and orchestration instead of
implicitly absorbing them into the reserve. A synthetic timing check passed and
rejects contradictory sequential timers. The worker's original projection is
retained for reconciliation; inference, controls and32400s threshold are unchanged.
