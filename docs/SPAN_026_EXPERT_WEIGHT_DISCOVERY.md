# Next-span proposal: one fixed expert-weight ablation

The H72pilot passes engineering but does not improve either local routing-state
loss. Its new information disproportionately changes the B/tie channels. This
does not establish the right expert weight or forecast a competition score.
The full H72qualification is running; nothing in this proposal changes it.

Review of completed work found that H61/H62already tested broader order routes
and H64stopped the missing-order teacher approach. Preserve those outcomes.
H37/H43tested the older weak classifier/prompted-judge combination. H59tested
only a shared temperature on the strong public ensemble and explicitly did not
fit expert weights; its failed transport remains a warning. The retained strong
ensemble still has equal total Gemma/Llama weights.

Queue a distinct, fixed60%Gemma/40%Llama expert prior for the next span. This is
a modest experimental perturbation, not a learned optimum or a claim that model
size determines quality. Keep T1and the incumbent's exact inputs, original H60
route, orientation handling and runtime path. Off-route use.6Goriginal+.4L;
on-route use.3Goriginal+.3alignedGreverse+.4L. If H72becomes the official
incumbent without achieving the goal, preserve its original-Gemma6kreplacement;
otherwise retain H60's4krecipe. Freeze that comparator before future execution.

First perform one descriptive matched-cache replay, original and already
inspected later votes when matched components exist, with paired group/class
diagnostics. No weight, temperature, length, route or source grid. Training
overlap and known later-domain transport failure exclude independent-CV or hidden
score claims. A prospectively frozen gross-regression guard may stop obvious
harm; advancement still needs actual all-role qualification and at most one
official outcome. The weight change adds no model pass.

This is planning only: no new-weight probabilities, losses or GPU execution have
been computed. A separate bounded span must activate and select the task first.
If the rank goal is achieved, do not execute it. Otherwise inherit the actual
remaining quota/deadline, preserve the best official recipe and close negative
results without fitting against leaderboard feedback.

Evidence: [H72pilot](SPAN_026_PILOT_RESULTS.md),
[H60official improvement](SPAN_013_RESULTS.md),
[H59temperature-only failure](SPAN_011_RESULTS.md),
[H61completed route experiment](SPAN_014_RESULTS.md).
