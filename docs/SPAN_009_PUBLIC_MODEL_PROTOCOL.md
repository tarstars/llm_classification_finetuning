# H54 public model admission: tascj ensemble

Frozen recipe: author notebook tascj0/lmsys-0805; repository
https://github.com/tascj/kaggle-lmsys-chatbot-arena at
83cd93d50b9283c18711e8c63e4e1c6399c7b9ce (Apache-2.0). The four author Kaggle
datasets below are accessible and currently version1, published August4–5,2024:

| Dataset | Purpose |
| --- | --- |
| tascj0/lmsys-checkpoints-0-0805/1 | Full fp16 Gemma2-9B trained classifier |
| tascj0/lmsys-checkpoints-3-0805/1 | Full fp16 Llama3-8B trained classifier |
| tascj0/lmsys-modules-0805/1 | Custom author models, kernels, processor/collators |
| tascj0/lmsys-packages/1 | Author Triton2.2/xformers cp310 wheels |

Dataset metadata declares CC0-1.0. Underlying Gemma and Llama model terms still
apply. The author training README describes original competition train plus
external33k/deduplicated21k, pseudo-label and staged training. Local heldouts may
overlap supervised training. This is allowed supervised deployment, not a clean
teacher benchmark. BlackPearl's referenced final dataset returns403; do not bypass
access controls or substitute an unknown mirror.

Keep custom author model modules and ProcessorPAB unchanged. Gemma original
answer order; Llama swapped order and probabilities aligned [1,0,2]; arithmetic
mean with weights[.5,.5], matching the author rather than later forks' .57/.43.
4096 conversation tokens plus judge/chat suffix; batch80, packed8192token budget,
fp16, equal model layers on two T4s, author's two-stage pipeline. No newly fitted
calibration, no answer generation, no available identity features or votes.
Dummy label columns satisfy the author dataset interface; no evaluate() call.

Use the exact author image e3f9bf91d1b7b36bf77554565d27c5733585fbe5e2b7f0832e068824e7accc3c
and offline wheels. Fail early if its expected Python3.10 is not honored.
The wrapper adds strict state-dict loading, SHA256 of actually loaded checkpoint
shards, input token hashes, device/memory/throughput telemetry and CSV validation.

Profile seed42:256 uniform rows plus64 disjoint largest total-character rows,
selected without targets. IDs and input hash are frozen before GPU in
results/span_009/profile_selection.json. The uniform cohort estimates runtime;
the long cohort stresses memory/maximum lengths. These are not independent
quality measurements. Run identical preprocessing/inference on the three visible
test rows and hidden rerun. Public worker<=2280seconds inside2400service cap;
hidden worker<=32280 inside32400seconds. A hidden flag or changed test hash
skips public profiling only. Require all engineering gates and a full conservative
projection before one official code submission. Never automatically repush.

Version1 failed before any notebook cell: the old image's papermill requires an
explicit kernelspec. Exact statusERROR, charged10.620seconds, zero worker output;
the failure and source are preserved. A distinct runtime-fix attempt adds the
Python3 kernelspec, rehearses actual notebook-engine startup plus the unchanged
final worker cell, and is capped at2200service seconds (2080worker seconds).
Both attempts together must remain below the original2400second phase cap.
No model, preprocessing, class mapping, packing or ensemble weight changes.

The runtime-fix control's first160 uniform Gemma rows take148.92seconds. Source
inspection finds a wrapper scheduling mismatch: logits.cpu() was called before
launching GPU0's next stage, serializing work the author overlaps. Prepare a
distinct pipeline-fix attempt that places the host copy after both stage launches,
exactly matching the author. The control then failed when a new DataLoader worker
segfaulted at the long-cohort transition after CUDA/tokenizer execution. All256
uniform Gemma rows completed:253.491seconds,240851tokens, peak10.18/8.48GiB;
charge414.992seconds, exactERROR, six recovered artifacts. No Llama or long/test
control predictions exist. Keep the failure and compare the available256Gemma
rows' IDs, lengths, checkpoint hashes and probabilities (maxdifference<1e-5).
Do not claim numerical identity on missing controls. Disable DataLoader forks
(workers0) to address the observed lifecycle failure. No training/packing/weight
changes. The pipeline-fix service cap is1000seconds, admitted only after control
termination and recovery
and fresh total charges plus1000 fit the original2400second phase budget. If it
does not fit, wait for an explicitly admitted refreshed-free-quota phase.

Initial projection:25,000 hidden rows × measured uniform seconds/row ×1.2,
plus both cold starts and600seconds reserve. Review long-sequence timing and
coverage before accepting this extrapolation. Save all failures. No training or
independent quality claim is authorized by a passing runtime gate.

## One bounded partial-output recovery and warm measurement

Version3 reached its owned watchdog (exit-15, serviceERROR), charged895.742s;
all Gemma cohorts and Llama uniform256 are complete, Llama long/test missing.
Gemma uniform probabilities exactly match all256 recovered version2 rows (max
absolute difference0). The cold uniform projection is33795.914seconds: FAIL at
the unchanged32400second limit. Preserve this result. Total phase charge1321.354s.

Prepare one1000second recovery role before further compute. Embed only verified
small model-output artifacts/hashes from exact failed version3 in the private
notebook. Never relabel that versionCOMPLETE. Reuse its completed Gemma long/test
outputs only in public qualification, with identical IDs/checkpoints and byte
hashes. Hidden mode never receives/uses the recovery cache and performs full
model inference on its actual test text. No raw conversations or target labels
are embedded. Freshly run both uniform256 cohorts again and finish Llama long/test.

Before timing uniform inference, run a fixed eight-row warmup (first eight of the
already frozen uniform IDs) through each exact model. Charge measured warmup once
in startup, alongside actual cold loading; extrapolate warmed uniform256 seconds
per row with the original20%margin and600second reserve. This is one prospective
startup/steady-state measurement, not a lowered margin or parameter sweep.
Report the original failed projection beside the new one. Require ID/length
identity and max probability difference<1e-5 versus both saved uniform controls,
all source/CSV/device checks, and full Llama long stress coverage before submission.
If it still fails, no submission under this runtime recipe. Fresh admission must
keep1321.354s plus1000s within the original2400s phase cap.
