# Frozen H72 GPU information stage

One6144conversation-token extra Gemma-original pass on the357exactly truncated
cases; unchanged old H60route/expert weights/other inputs. CPUtoken references
cover all cohorts and all three model roles. This information-only notebook
cannot be officially submitted. No target columns enter inference.

The native H60forward implementation is copied into span026_engine.py. Changes
are the explicit cohort allowlist, per-cohort processor budget, model-position
guard,30second aggregate progress and post-cohort checks. Model loading, sharding,
fp16/kernels/packing8192 and forward operations are unchanged. Only role m0 on
context_warmup/context_probe/changed_extended uses6144; every other pass is4096.

First measure the full57477-row text-only selector on actual Kaggle CPUs and
require exactly the frozen357IDs. In one Gemma process run old80warmup,
old256control, old512uniform,6k80warmup and6k64probe. The probe is seed45random
plus one maximum-length case, sorted IDs. Before any larger diagnostic cohort:

- Verify exact checkpoint records, IDs, lengths, CPUtoken fingerprints and the
  canonical old raw-probability control within1e-5.
- Require peak allocated memory on each T4to stay<=95%of device total, and all
  actual input positions within the retained8192limit. OOM/source/control failures
  remain errors; no batch or quantization fallback.
- Project using retained H60parent30912.486seconds, worse retained/current
  Gemma512time (including the prior276.1072s reference), positive parent startup
  slowdowns, full extra-pass seconds/row times357/57477times25000times1.2,
  all80new-mode warmup seconds and the WHOLEmeasured57477-row selector cost.
  The parent already includes oldcold/fullwarm/other/600reserve. Never divide
  elapsed two-T4wall time by2. This screen is not final deployment eligibility.

If projected time>=32400s or memory guard fails, save the failure and stop cleanly
before changed-row quality scoring or other model loads. Otherwise continue
Goldoriginal357andG6k357in the already-loaded Gemma process. Then Llama swapped
scores its old80warm/control256/changed357. Finally Goldreverse scores its old
80warm/control42(the exact H60routed controls)/changed357. Each saved cohort
passes CPUtokens/checkpoint/ID checks and each role's canonical raw control.

After terminal COMPLETE, retrieve/hash every selected artifact and independently
verify accepted source/image/offline inputs and all controls again. Only then
query original target labels for the four-output descriptive comparison.
Use the initial fixed parent/candidate on/off formulas, conservative per-row
gain bounds, original prevalence, normalized prompt groups,seed42/2000resamples
and per-class/slice metrics. The prospective amendment's fixed-0.005overall
gross-regression guard applies. Positive local gain is not required or treated
as independent evidence; no fitting or extra context/weight variants.

Actual full candidate qualification and a frozen official outcome remain separate
requirements. Initial<=7200two-T4s; qualification<=3600s; total<=10800/max2versions/
oneofficial. Original deadline/quota, full fresh admission, exact final-cell
rehearsals and durable mutation intents remain mandatory. Hidden mode will use
actual text inference with public control caches excluded.
