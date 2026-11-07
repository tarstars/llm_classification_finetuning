# H59: fixed-temperature confirmation qualifies one official candidate

Exact job arseniytrushin/llm-preference-span011-confirmation-calibration/1 completed
on2026-11-07. The1024groups were selected before labels and disjoint from all
1024learning prompt/session groups. **T=1.3161518671742531** was fitted only on
learning and committed before confirmation inputs were uploaded or scored.
Confirmation fitted no parameter and evaluated only the pre-frozen recipe.

| Confirmation metric | Baseline | Frozen temperature |
| --- | ---: | ---: |
| Multiclass log loss |1.04426491|1.03285107|
| Accuracy |0.472656|0.472656|
| Multiclass Brier score |0.628933|0.621776|
| Mean maximum probability |0.526468|0.483732|

Paired loss reduction **0.01141384**, group-bootstrap95%interval
**[0.00439561,0.01839702]**, passes the predeclared point>=0.001 and lower>0 rule.
Accuracy is unchanged by construction. The interval describes these later votes;
it does not predict competition log loss or rank. Earlier learning group-CV gain
0.00461556 remained uncertain,[-0.00245062,0.01240010]. It was only an information
gate; confirmation provides independent evidence for the frozen scalar.

## Domain and class review

All three class losses improved: A0.992060→0.991998, B1.018731→1.013972,
tie1.139048→1.105734. Most gain comes from ties. Class fractions A/B/tie are
0.3643/0.3428/0.2930 versus original training0.3491/0.3419/0.3090; the observed
class-prior difference is modest. No class weight or tie offset was fitted.

Seven of eight prepared-length bins improved. The3072–4096bin regressed0.02186
on only11rows; its small support does not justify fitting a route or changing T.
The neighboring4096+bin improved0.05724on8rows and is also noisy. The four main
bins containing916rows each improved. The48row code-keyword proxy gained0.04374;
the other976rows gained0.00982. Benefit is not confined to that proxy. No examples
were dropped or relabeled after scoring, and no new variant was selected.

These2025English first-evaluation single-turn pairs are harder than the competition
population. Matching label-free prepared lengths and similar class fractions does
not establish identical conditional preferences or transport of calibration.
The remaining transfer uncertainty is material; one official frozen submission is
the direct test. The independent-source result is not competition CV. No numerical
forecast of official score is claimed. If it regresses, preserve the existing best
0.83053 submission and record the negative result without tuning T to the ladder.

## Engineering and admission decision

All20retrieved confirmation artifacts were hash-verified. Both256row original
component controls again match exactly, maximum probability difference0. Frozen
checkpoint/source/input/IDs/swap/mixture/temperature/CSV and two T4 checks passed.
The deployment retains8192packing,4096conversation cap, fp16/author kernels, equal
original Gemma and swapped Llama weights, offline pinned Python3.10image and the
owned watchdog. Hidden mode reads actual test inputs, with no public prediction
cache, source cohort or labels. No additional expert pass is introduced.

The original512row target-runtime profile plus20%margin and the larger observed
startup across both sessions, then600seconds reserve, projects29849.450seconds
(8.292hours), under32400seconds. Confirmation charge1650.445seconds and learning
2436.757total **4087.202seconds/1.1353elapsed two-T4session hours**, within the
original21600second public cap. The original48hour deadline/quota period are
unchanged. No further research GPU runs while an official receipt is pending.

Decision: admit exactly the existing confirmation notebook/version at the frozen T
for one official submission, after fresh quota/jobs/source checks. Its seven exact
evidence hashes are in results/span_011/deployment_decision.json. No official
receipt exists at this report's preparation. The rank<=19goal remains active and
requires fresh actual leaderboard evidence after official scoring.

Evidence: results/span_011/confirmation_metrics.json,
results/span_011/calibration_confirmation_qualification.json,
results/span_011/calibration_confirmation_source_verification.json,
results/span_011/frozen_temperature.json and executed notebook23. Per-row outputs
and labels stay private under the project database.

Official submission **58091536** was accepted at2026-11-07 12:22:36UTC from the
exact qualified version. Initial status PENDING. See results/span_011/official_receipt.json
and official_submission_intent.json. No official score or rank improvement is
claimed yet. The controller observes that receipt, with no research GPU jobs
while it is pending; the frozen parameters and evidence remain unchanged.
