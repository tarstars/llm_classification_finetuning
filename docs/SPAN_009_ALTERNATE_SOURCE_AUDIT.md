# An accessible earlier BlackPearl checkpoint, not the final winning merge

Read-only audit2026-11-06 while official submission58068582 is pending. No GPU
execution or weight download; no change to the submitted model.

The README's final dataset sayoulala/v7-dpo-16bit-01234-8bit-all returns403.
However exact public notebook sayoulala/merge-gemma-model-step1/1 isCOMPLETE
and exposes ten output artifacts, including four safetensor shards. Its source
loads the adapter named v7-8-gemma-v7-64r128-ddgemma2-16bit-4bit905 and saves
v7-8-gemma-v7-64r128-ddgemma2-16bit-4bit905_merge. This is an earlier single
merged classifier, not the documented final five-fold GPTQ/DPO merge. The number
905 in a filename is not a verified quality measurement.

Only config.json (1096bytes) and model.safetensors.index.json (39128bytes) were
downloaded. Configuration declares Gemma2ForSequenceClassification, bfloat16,
42layers, hidden3584 and three labels. Aggregate evidence is in
results/span_009/blackpearl_parent_availability.json. Small source/config caches
stay under database/runs/span_009/research/sayoulala__merge-gemma-model-step1.
No signed download URLs are retained in project evidence.

Potential follow-up only if still relevant after the frozen ensemble's official
outcome: resolve training/label provenance and licenses, check exact input
serialization, assess runtime and distinctness before allocating a small pilot.
Original-train supervision precludes independent validation claims on overlapping
local rows. Accessibility alone does not justify a GPU run, a claimed gain, a new
leaderboard submission, or treating this checkpoint as the author's final winner.
