# Runtime discoveries queued for v11

These are conditional future tasks, not extra v10 GPU admission. The current
packing pilot remains live and must finish/stop before span closure. Preserve
the original two-recipe and21600second public phase limits. The rank<=19 goal
continues if this span cannot produce an admitted successful submission.

## E16: measure packing on the same GPU and qualify cost before all passes

The completed8192-token pilot projects32998.845seconds, failing9h. Its Gemma
uniform512pass takes280.146seconds. The live12288-token pilot has now completed
that same cohort in312.999seconds, with control256at173.468versus138.147seconds.
This is a cross-session timing regression, not a controlled causal packing result.
Source equivalence and profile-ID records establish the intended single change;
GPU clock/thermal/session differences remain possible. Complete the live pilot
and inspect its final artifacts before concluding anything about the full recipe.

A future bounded engineering test should compare6144and8192token packing on the
same loaded model, identical IDs and full-batch warmup. Compare both models if
needed, record per-GPU memory/clocks/utilization, alternate the order once to
expose cache effects, and use exact token/prediction controls. Cap this profile
at30minutes; do not rerun the already slower12karm without a concrete need for a
matched control. Select only on runtime/memory/numerical identity, never labels.
Before a full four-orientation qualification, calculate the measured base-only
runtime lower bound; stop if it already violates the conservative nine-hour cap.
Keep20%margin and600seconds reserve. A changed numerical recipe needs matched
evaluation; preserve every failed identity gate rather than relaxing it.

## H58: keep model weights equal and average only Llama's answer order

The first pilot measured selected-uniform reverse costs32.714seconds for Gemma
and23.801seconds for Llama. On the same fixed route, test exactly one cheaper
rule: unrouted predictions remain(.5Gemma+.5Llama-swapped); routed predictions
become(.5Gemma+.25Llama-swapped+.25Llama-original), after class alignment.
This preserves the two experts' total.5/.5weights. It averages order only for
the cheaper expert, so it leaves Gemma's position sensitivity unresolved.

Removing the measured Gemma reverse rate and its cold/warmup cost from the first
projection yields a **forecast30945.410seconds /8.596hours**, still including20%
margin and600seconds reserve. This is a feasibility forecast, not a measured new
worker qualification. Model accuracy may regress and the rank gain is unknown.
Do not treat training-overlapping diagnostics as independent quality evidence.

In a later activated span, first replay this one fixed rule on cached outputs
for descriptive diagnostics, then prospectively freeze/rehearse a real three-pass
worker and remeasure the selected subset within a30minute pilot. No hidden cache,
mixture-weight fitting or route-cap search. Admit one official frozen candidate
only after all source/ID/prediction/device/CSV and measured runtime gates. Resolve
E16 first if current packing/numerical results make that prerequisite necessary.
