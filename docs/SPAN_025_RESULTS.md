# V25: identical-only override fails; retain H60

The fixed published probabilities `[0.06,0.06,0.88]` worsen the measured
identical-answer stratum. All275original pairs were scored. Their loss is
**0.391508** under the rule versus **0.311360** for H60's off-route outcome and
**0.310603** for its on-route outcome. Stop the rule without a probability grid,
full deployment qualification or official submission.

| Original-data comparison | Measured result |
| --- | ---: |
| Identical pairs / full original rows |275 /57477 (0.478%)|
| Labels A / B / Tie |12 /15 /248|
| Population-weighted lower gain |−0.00039647|
| Population-weighted upper gain |−0.00037409|
| Lower-gain paired prompt-group95% |[−0.00066280,−0.00017682]|
| Required lower gain |greater than0.0005, with positive interval lower bound|

For fixed expert outputs, H60 is either `.5G+.5L` off route or
`.25G+.25swap(G)+.5L` on route, with Lalready class-aligned. Original/swapped
token sequences are identical for these identical strings. Taking the larger
true-class probability across the two outcomes supplies a conservative gain
bound. The upper bound is also negative. These are algebraic bounds on the
measured outputs, not universal numerical bounds over unknown GPU batch shapes.
No global route was guessed. All three classes contribute negative lower gain:
A−0.00008791, B−0.00012421, Tie−0.00018436.

These original votes overlap checkpoint training. The finding is a futility
result for this frozen rule, **not independent CV, a hidden-score forecast or
proof that human noise can be predicted on new identical pairs**. The model may
have learned these training examples. No alternate constant is fitted from them.

## Execution and checks

Exact source `arseniytrushin/llm-preference-span025-identical-audit/1` is
COMPLETE. Both256-row canonical controls reproduce H60 probabilities exactly:
maximum absolute difference0 for Gemma and Llama. All275IDs, lengths and token
fingerprints, checkpoint records, pinned Python3.10/Torch2.1.2/Transformers4.42.3,
twoT4devices, offline sources and accepted metadata pass. All15retrieved
artifact hashes are verified. The unchanged engine and scoring method were
frozen/pushed in52b4596before GPU or this quality query. FiveCPU tests and the
actual notebook success/failure/watchdog/Jupyter-startup rehearsals passed.

The worker exited0 after789.687seconds including setup; its inference
orchestration took774.814seconds. Kaggle charged **803.502two-T4session seconds**
(13.39minutes), below the1800second pilot cap. The service did not report
COMPLETE until15:53:18UTC despite the earlier completed worker/files. That
observation delay was not another GPU run and was not treated as failure.
Controller27732is terminalexit0. No follow-up or official submission was made.
Post-run quota is20,907.891used /87,092.109free seconds, reserved0/paidFalse;
the original Nov10deadline and Nov14quota anchor remain unchanged.

Earlier local preparation failures remain recorded: the missing staged Llama
tokenizer and modern-tokenizer control mismatch. The retained tokenizer was
downloaded once, hash-verified and checked with the native pinned runtime.
No model weight was acquired on the laptop. Notebook37 is executed with the
final failed gate. Heavy arrays remain under ROOT/runs/span_025.

## Next information source

The old H24question-cap hypothesis refers to the project's earlier JointTokenizer,
not this retained public model. The actual author ProcessorPAB concatenates
question, A and B in order, then truncates the conversation prefix at4096tokens.
Extra evidence can therefore be missing from the tail of B or later turns.
The existing full-length audit finds362near-cap rows, but its length threshold
is only a proxy; exact lost-content coverage remains unmeasured.

Queue a new bounded context-extension experiment linked to H23/H24, preserving
their old wording and results. First audit exact4k/6kinput changes without labels
using the native processor/runtime. If coverage and work support a practical
test, retain the H60route and all existing outputs, replacing only the original
Gemma component on actually truncated rows with a6144token original-order pass.
A separate frozen pilot must measure oldoriginal/reverseGemma, extendedGemma and
Llama on matched changed rows, with actual controls/runtime/memory. Bounds over
the two parent routing outcomes avoid guessing the global route. This is new
text evidence, not another constant on the rejected identical-answer rule.
Longer context has no measured gain here and may hurt or fail runtime.

The rank<=19goal remains active; best official0.82987. The fresh complete board
at16:08:15UTC confirms23/312, rank19/20cutoff0.82915 and leader0.82692.
Preserve98snapshots, archivedgoals and user07/12. Close H71 as a measured negative,
carry37unfinished tasks and select a bounded successor before more compute.

[Frozen plan](SPAN_025_PLAN.md) · [primary audit](../results/span_025/information_audit.json) ·
[terminal run](../results/span_025/jobs/identical_pilot/controller_state.json) ·
[complete board](../results/span_025/final_leaderboard.json) ·
[executed notebook37](../notebooks/37_identical_pair_audit.ipynb).
