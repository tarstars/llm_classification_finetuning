# V5 results: model strength helps, approximating it remains uncertain

The source-model experiment completed all three focus tasks. **Known model
strength contains useful residual signal, and TF-IDF can infer source classes;
the primary practical preference selector did not reliably improve LoRA.**
Best official log loss remains **0.95481**. No submission or GPU job was launched.

| Stage | Actual finding | Decision |
| --- | --- | --- |
| E06: known-source diagnostic | Matched calibrated LoRA 0.953701 → oracle strength 0.940456; gain 0.013246, 95% interval [0.008968, 0.017515] | Passed information gate |
| E07: source attribution | TF-IDF 27.08% source-class accuracy / 2.938585 source loss; embedding head 9.95% / 3.704803; prior loss 3.822561 | Portable TF-IDF passed its gate |
| H45: practical preference | Nested selector 0.953834 versus calibration 0.953701; gain −0.000133, interval [−0.002038, 0.001660] | Adoption failed; retain official reference |

The fixed compact strength family separately achieved **0.952397**, gain
**0.001305 [0.000037, 0.002618]**. The full-posterior family was weaker; one inner
family choice erased this small overall advantage. We preserve the primary
selector result rather than adopting a family after seeing all outer scores.
A future frozen strength-only replication is justified, but not a claimed
deployable gain. These small gains do not establish a route to 0.8.

Evaluation uses the unchanged 45,985 training / 5,746 development split and 5,211
development prompt groups. Strength/source models train only on original training;
residual selection/calibration is nested within development prompt groups. Source
fits purge duplicate answers across fitting/evaluation boundaries, including 882
answers before final fitting. Confirmation was unused in v5. Development and prior
checkpoint selection were inspected, so this is exploratory. The older 0.954069
cross-fitted-temperature reference differs from the frozen-temperature 0.953971;
our primary matched comparator is the new nested recalibration 0.953701.

Total successful CPU experiment elapsed time is **1,748.212 seconds / 29.14 minutes**.
Including the conservative 60-second pre-fit failure allowance gives **1,808.212
seconds / 30.14 minutes**, below the 7.5-hour combined cap. Peak worker RSS is
3.701 GiB. Separate audit/notebook/preparation time is not represented as model
training time. The initial E06 serialization failure and corrected loader are
preserved; original data, folds and numerical gates were unchanged. GPU charge is
zero, with no external extraction or classifier retraining.

The source head's final 11,492-answer inference takes 14.009 seconds. The conditional
classifier-plus-source projection including conservative cold setup, 20% margin
and 600-second reserve is **19,245.25 seconds** for an assumed 25,000 hidden rows.
Local CPU speed and prior classifier timing are assumptions; a new offline Kaggle
profile is still required before deployment. Unknown-source and time-shift behavior
is not established by these same-source prompt-group splits.

See [E06](SPAN_005_E06.md), [E07](SPAN_005_E07.md), [H45](SPAN_005_H45.md),
[scientific audit](../results/span_005/scientific_audit.json),
[completion](SPAN_005_COMPLETION.md) and
[executed notebook16](../notebooks/16_backlog_v5_span.ipynb).
V5 is sealed. V6 has 37 tasks: 35 unfinished carryovers plus H46 frozen replication
and conditional E08 offline qualification. It has no focus or execution; v7 is
an empty draft. No v6 experiment is included.
