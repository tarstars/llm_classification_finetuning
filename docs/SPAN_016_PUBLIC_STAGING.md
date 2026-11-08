# H56 public staging addendum before GPU or reward outputs

The fixed1024quality cohort, native experts, three calibration parameters, mixture,
primary comparison, information thresholds and18000second phase cap are unchanged.
No model reward or new quality output has been inspected. Weight acquisition is
in progress, no GPU version has been accepted. This addendum supersedes only the
assumption that initial quality and complete target-runtime diagnostics must all
fit into the3600second first public session.

The original public worker scores the new1024source rows plus256canonical
controls and512original-runtime rows through multiple large models, and scores
all1024source rows through swapped Gemma for the matched comparison. V13measured
475.84seconds for just512base two-expert rows, plus separate startup/warmup/reverse
costs. Repeating unnecessary baseline passes while adding another two-conversation
reward expert risks exceeding the first-hour cap before useful evidence is saved.
This is a planning inference from actual legacy costs, not a new reward-speed
measurement. Native8Bthroughput remains unknown until the actual pilot.

The initial public version therefore uses --information-only:

- Score the full frozen1024source groups through both reward sizes and the matched
  Gemma/Llama/full-reverse quality comparison; no smaller primary-quality sample.
- Keep unchanged256canonical Gemma/Llama controls, full80row warmups, native reward
  controls, hardware/source checks and the visible-test CSV smoke.
- Defer the512original-distribution deployment-runtime cohort. Llama also omits
  unused test/runtime predictions; it is a public comparator only. Reward controls
  use the fixed identical-tensor native checks during warmup, avoiding an unused
  extra256-row pass.

The actual serialized public/hidden branches and final CSV bookkeeping are
rehearsed again before GPU. No hidden behavior changes: hidden mode performs real
Gemma/reward inference, including the known1024training-calibration pass, with
no hidden votes/public prediction cache. Model weights and input hashes are fixed.

The information-only version is **not eligible for an official submission** even
if its quality interval is positive, because target runtime is unmeasured. Only
if the fixed information gate passes may the one separately admitted follow-up
perform the actual512target-runtime pipeline and needed confirmation within
<=14400seconds. It must satisfy all original numerical/source/information and
conservative<32400second hidden-runtime gates before one official trial. No extra
GPU version, budget, recipe, threshold relaxation or automatic retry is authorized.
An engineering failure retains its service status and requires a distinct explicit
follow-up admission; an observation timeout never permits a duplicate mutation.

For each outer validation fold, compute all cohort probability-dependent routing
scores with that fold's training-only calibrator, then retain its held-out rows.
This keeps a held-out label outside its route through other rows' calibrators.
The frozen baseline uses its unchanged cohort route; both arms share the same
full reverse-Gemma cache. Save every fold route and its work fraction. The union
of held-out selected rows describes cross-validation rather than one deployed
cohort budget. Deployment uses one full-data-trained calibrator and the same
10%work/20%row rule. A regression test changes another fold's model and verifies
that the held-out fold's predictions and route remain unchanged. No reward output
or GPU result was seen before this validation correction.
