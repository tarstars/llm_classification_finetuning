# V26: context extension rejected; no official submission

H72did not improve local preference loss and failed the conservative deployment
runtime gate. Retain the legitimate H60official0.82987. Fresh complete board at
2026-11-08 19:43:55UTC: **23rd/310**, rank19/20cutoff0.82915,leader0.82692. The
rank<=19goal remains active and unachieved. No new official trial was admitted.

The one frozen change was an extra original-Gemma6144conversation-token pass on
truly truncated inputs, retaining old H60routing, weights, T1and other inputs.
All57477native4k/6kCPUcomparisons found357changed rows (0.6211%). Bcharacter
retention rose64.52%to88.60%;22pairs gained their first Btext. Extra whole-pass
work proxy was8.985%of original Gemma work. These are exposure/work diagnostics,
not semantic or quality guarantees. The deployable selector reproduced the exact
357IDs. Total source-preparation CPU626.642s/1200s; no completed audit was rerun.

## Quality and interpretation

The information pilot scored all357rows in four matched modes. Off-route loss
rose0.884017to0.897547; on-route0.880353to0.884853. Bclass loss/recall worsened,
while tie loss/recall improved. Weighted conservative gain-0.000234356 had paired
prompt-group95%[-0.000317813,-0.000152574]; upper routing bound+0.000122367.
Original votes overlap checkpoint training; these are descriptive diagnostics,
not independent CV or an official-score estimate. More retained text was not
enough to improve this local comparison. Changed length behavior and changed
human-preference evidence are hypotheses, not identified causal explanations.

The prospective pre-GPU amendment explicitly used a fixed-0.005gross-regression
guard, which passed, and did not require positive local gain. It permitted the
single actual full qualification. Neither the candidate nor the evaluation rule
was fitted after these results. See [pilot findings](SPAN_026_PILOT_RESULTS.md).

## Actual engineering outcome

Both exact private/offline/two-T4jobs completed. Pilot source
`arseniytrushin/llm-preference-span026-context-audit/1` yielded32verified files;
actual candidate `arseniytrushin/llm-preference-span026-context-candidate/1`
yielded41. All canonical raw probabilities are exact(maxdiff0), including
control/uniform/long/test on both base models and the reversed Gemma route.
IDs, native tokens/checkpoints/positions, original routes, mixtures, unchanged
rows, visible CSV and memory checks pass. Public controls remain excluded from
hidden inference; no target columns enter either worker.

The early screen projected32051.569s/8h54m. The actual qualification measured
Gemma512at290.599s, Llama512at213.454s and routed Gemma512at32.809s. Under the
frozen worse-timing/max-overhead/20%margin/600sreserve rules, its parent backbone
alone projects32668.060s. Extra6kpass920.036s, full new warmup297.747s and whole
selector121.837s produce **34007.680s/9h26m48s**, above32400. The early verified
floor32930.016s already established failure. Memory was within limits, peaking
70.12%/56.84%/69.93%by role. No faster-node retry or margin relaxation was used.

The frozen terminal auditor initially failed an overstrict bitwise check of
derived CPU JSmetadata:173/256values differed by at most2.78e-17, with route
IDs identical. NumPy1.26also reproduced this tiny cross-machine discrepancy.
The original auditor and failure are preserved. A separate rejected-only replay
uses the established H60derived-JS tolerance(rtol1e-12,atol1e-14), retaining
1e-5raw-probability controls and exact route/mixture checks. It verifies the
engineering outputs and confirms runtime rejection; it cannot authorize an
official submission. Qualification target labels were not read. See
[original failure](../results/span_026/frozen_qualification_audit_failure.json)
and [rejected-only replay](../results/span_026/candidate_qualification_replay.json).

## Resources and closure

Pilot charge5461.312s/7200s plus qualification2442.451s/3600s totals7903.763s
(2.1955elapsed two-T4hours), below10800s/max2versions. Exactly two accepted
versions and zero officials; all owned controllers terminal. Quota79188.346s
free, reservation0, paid scaling off, same Nov14anchor. The attempted early-stop
investigation found no exposed kernel-session ID; no cancellation with a guessed
ID was sent. The existing service/worker caps stayed active and the job completed.
The fresh complete inventory covers71owned notebooks with no nonterminal jobs;
the fully paginated competition history has no pending officials.

The exact goal is archived as GOAL_V26.md. H72is sealed as done/negative, with
final executed Notebook38and a passing closure audit. All102prior snapshots,
user07/12notebooks and their link are preserved. V27contains all37unfinished
carried items plus H73's fixed60/40expert prior and H74's native packing
prerequisite,39active items total; neither new task was executed in v26. Four
new final/start snapshot files bring the historical total to106. The slower
backbone demonstrates that merely removing6k
does not establish runtime eligibility. A new bounded span must first create
measured headroom with identical outputs before trying the fixed weight change.
Span closure does not complete the persistent rank goal.

[Coverage](SPAN_026_COVERAGE.md) · [Runtime rejection](SPAN_026_RUNTIME_REJECTION.md)
· [Notebook38](../notebooks/38_context_coverage.ipynb)
· [Final leaderboard](../results/span_026/final_leaderboard.json).
· [Closure audit](../results/span_026/closure_audit.json).
