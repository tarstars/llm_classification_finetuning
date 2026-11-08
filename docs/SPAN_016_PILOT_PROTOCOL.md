# H56: fixed reward-expert pilot, frozen before rewards

This refines the source-stage plan before any reward output, weight acquisition
or GPU admission. The main change is the8Bpretrained reward expert replacing
Llama. The0.6Bhead is diagnostic only. No model/temperature/weight/route grid.

Use exactly the1024fresh later-vote groups from prepare_span_016_cohort.py:
seed42hash order over eligible cached prompt/session representatives, timestamps
strictly after2025-07-06T18:16:33UTC, excluding all v11learning/confirmation groups.
Winner is first queried after durable IDs are written. Keep exact ID order,
target/group files and source hashes. English, first-evaluation, single-turn,
identical user turns and swap-aware duplicate exclusions inherit the audited
Arena140ksource preparation. Compare its unchanged original human votes.
The deployed parser must also preserve multi-turn competition inputs, nulls and
missing turns. Log their incidence; never flatten all turns into an invented one.

Use each pinned checkpoint's native non-generation chat template, no extra system
prompt, trained scalar head and raw logits. Tokenize each A/Bconversation
independently. Cap each at2048tokens: retain the prefix and the final native
EOS/newline pair if truncated. Record full/retained lengths and token hashes.
Use native pad151654, right padding; verify score-head loading has no missing,
unexpected or mismatched keys.8B uses float16 over both T4s, contiguous18/18layer
split, no quantization or replacement head.0.6B uses float16. Native and repeated
controls, device use, nonfinite checks and package hashes precede admission.

Let d=rA-rB. Fit exactly softmax(beta*d,-beta*d,gamma0-gamma1*abs(d)), with bounds
beta/gamma1 in[0,10],gamma0 in[-20,20]. Fixed L-BFGS-B, analytic gradient,
start[.1,0,.1],maxiter1000,ftol1e-12,gtol1e-8. No absolute reward mean or sigmoid.
Five stratified prompt-group outer folds, seed42; three inner diagnostic folds,
seed43+outer index, no inner selection. Fit three parameters only on each outer
training fold and predict its held-out groups. Keep fold membership/coefficients.
Full1024fit is for deployment, never reported as held-out calibration quality.

Unchanged Gemma uses the author8192packing/context4096/fp16/image. Baseline H60
remains equal Gemma/Llama with its fixed10%work/20%row Gemma-order averaging.
Candidate is equal Gemma/reward, with the same JS/work routing algorithm using
the new expert's probabilities and summed A/Btoken work. On route use
.25Gemma-original+.25aligned-Gemma-swapped+.5reward; off route.5/.5.

For the1024quality cohort, both arms use the same full swapped-Gemma cache,
avoiding mismatched batch-composition comparisons. Report this explicitly.
Actual candidate runtime/retention is measured separately on the original512
uniform rows, with the actual routed reverse pass and full-data-fitted calibrator.
Canonical original256controls for unchanged Gemma/Llama must match1e-5; new
reward/native numerical controls are distinct from old-checkpoint identity.

Primary metric is paired cross-fitted replacement versus H60 multiclass log loss
on the1024out-of-time groups. Fixed5000group bootstrap resamples, seed4242; report
class/tie/length behavior and accuracy as secondary diagnostics. Initial
information gate: point loss reduction>=.005 and95%lower bound>-.005. It is an
information decision, not proof. If passed, a separately admitted follow-up may
confirm the ONE frozen mapping/recipe on fresh post-publication groups, with
point reduction>=.005 and lower bound>0 before official admission. Declare
confirmation IDs before their inference; no fitting on confirmation labels.
An initial result already meeting the stronger gate may proceed without a new
quality cohort if actual deployed controls/runtime all pass. Otherwise stop
without threshold relaxation or another mixture.0.6B cannot reject8B by itself.

Later votes postdate both checkpoints but remain a newer model/user population;
this is not competition CV or a leaderboard score forecast. Text could preexist
its new vote. A single frozen official outcome tests transfer; retain H60if worse.

Deployment may deterministically recompute the fixed1024reward calibration
scores and refit three parameters from those known external votes; this is
legitimate training, not hidden labels. Include that extra inference in hidden
cost. Prefer a separately frozen three-number calibration artifact if an exact
second source version is already admitted, with no public per-row prediction
cache in hidden mode. Never omit calibration time from the projection.

Source assets can be staged as one private Kaggle dataset from immutable HF
files, with original licensing/attribution and exact full weight hashes. This
does not admit GPU. Keep total staged payload<=18GiB, current free disk>=40GiB
before acquisition, no paid CPU/storage scaling, and original48hour deadline.
Public GPU limits remain3600initial/14400follow-up,18000total/max2versions/1official.
Before weights: finish and rehearse the exact serialized worker, not just syntax.
Before GPU: fresh complete quota/jobs/receipt cursors and pinned offline assets.
Actual25kprojection must be<32400s with20%margin/allcold/full80rowwarmup/other/
calibration overhead/600sreserve; owned watchdog and strict source/ID/token/swap/
CSV/simplex/two-T4 checks. No automatic retries after a timeout.
