# Pre-GPU correction to the H02 pilot

The initial v31 draft increased the never-run size pilot to128accepted updates
on4096training rows, while retaining the5400-second session. That was too
optimistic without checking the saved training timing first. No GPU or new
quality evaluation ran under this draft. Its exact plan and goal are preserved
in SPAN_031_INITIAL_PLAN.md and SPAN_031_INITIAL_GOAL.md, and its completed
input preparation remains intact under results/span_031/preparation.json and
the heavy database. It is superseded, not a completed128-update experiment.

The actual historical Qwen3-4B continuation trained256nominal cycles in
5913.436seconds:23.099seconds/cycle. Applying that reference to128cycles
gives2956.718seconds, exceeding the old size-session driver's first-arm
training window1980seconds before development inference, model setup or8B.
This is a planning warning, not a causal throughput benchmark across nodes,
data mixes, or model sizes.8B throughput is still unmeasured. It does not
justify admitting the enlarged draft inside its unchanged90-minute budget.

Before any GPU execution, restore the originally prepared32accepted updates,
1024training rows,512development rows,48nominal-cycle ceiling and4warmup
updates. Both arms still start fresh and use the same fixed samples/packing.
Use the exact original unrun v8 sample IDs; preserve the unused expanded inputs.
The pilot measures memory, throughput and early learning; its small exposure
cannot establish full-training quality or an official-score forecast.

The5400-second public allocation, original span start/quota counter,
November14reset anchor, November10absolute deadline, no-official rule, raw
quality/memory/exposure gates and all prior charges remain unchanged. Preserve
the initial resource plan and record its hash in the active plan. A longer
training study requires a separately costed follow-up supported by actual8B
measurements. No gate is relaxed in response to a quality result.

[Timing evidence and revision](../results/span_031/protocol_revision.json).
