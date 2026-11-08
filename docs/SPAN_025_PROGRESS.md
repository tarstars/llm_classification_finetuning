# V25: identical-pair preparation passed; no GPU admitted yet

All275exact raw-identical answer IDs are frozen without labels. The current
original512sample contains none, so their quality remains unmeasured. The fixed
identical-only rule is[.06,.06,.88]; no parameters were fitted.

Three CPU tests verify the two H60routing outcomes, conservative per-label gain
bounds and unchanged tie channel. Exact author-tokenizer checks now pass:

| Expert | Canonical256tokens | All275swapped inputs | Mean / maximum tokens |
| --- | --- | --- | ---: |
| Gemma original | Exact H60fingerprint | Identical |365.30 /3474|
| Llama swapped | Exact H60fingerprint | Identical |356.29 /3338|

Preparation initially stopped because the retained Llama tokenizer had not been
staged locally. Its9,084,673bytes were downloaded once from the original public
version, unwrapped locally and verified against actual H60's SHA256. No model
weight was downloaded. The existing frozen cohort manifest was preserved.

The current local Transformers5.22.1/Tokenizers0.23.3then failed the Llama control
fingerprint despite identical tokenizer file bytes. This failure is retained in
local_control_token_failure.json. Repeating the same label-free check with the
already available pinned Transformers4.42.3/Tokenizers0.19.1produced **exact**
Gemma and Llama control fingerprints, and original/swapped identity for every
identical pair. The local pinned CPU check is not actual GPU-backend validation.

Successful token preparation took7.60CPUelapsed seconds. All98historical snapshot
files and usernotebooks07/12remain protected. No labels for this new stratum,
model predictions, GPU launch or official submission have been queried.

Next: build an information-only worker using unchanged retained H60expert code;
rehearse the actual notebook cells and run fresh quota/jobs/receipt/dependency
admission before the one<=1800s pilot. No pilot is live. A positive original
population-weighted bound would still need actual full deployment qualification
under the plan, followed by a legitimate frozen official trial. Rank goal remains
active; last complete board23/312, official best0.82987.

[Plan](SPAN_025_PLAN.md) · [cohort/token audit](../results/span_025/cohort_audit.json) ·
[preserved local mismatch](../results/span_025/local_control_token_failure.json) ·
[tokenizer staging](../results/span_025/tokenizer_staging.json).

The information worker is now built using the exact retained H60engine hash.
Five CPU checks pass; the actual notebook final cell passes success, failure and
owned-process watchdog rehearsals, plus Jupyter startup. The frozen source also
contains the complete post-run control/quality auditor, including accepted image
and dataset checks. The resource plan records87,895.611free seconds, zero
reserved and paid scaling off, with the inherited deadline unchanged. Publish
the bundle, then run fresh full admission before the single1800s pilot. No
candidate labels or probabilities have been read; no GPU is admitted yet.

The exact pilot was accepted at15:23UTC as `arseniytrushin/llm-preference-span025-identical-audit/1`; RUNNING at15:24UTC. Full admission found no other active jobs, all7official receipts terminal, fourpublic dependencies pinnedv1, reserved0/paidFalse. Accepted notebook cells and offline settings independently verified; image/machine/private/GPU/dataset fields match. Controller session27732/PID392536observes this one version. No official submission; quality remains unmeasured until terminal artifact/control audit.

Both model passes finished; the worker reported information_complete after774.814s. At15:43:56UTC the output catalog already listed42files, including session_summary.json and all predictions. The service still reportsRUNNING at15:46UTC. This is not a failed execution or permission to launch a replacement. The controller remains live on the sameversion; terminal retrieval/control/quality auditing remains pending.
