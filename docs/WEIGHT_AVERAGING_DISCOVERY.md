# A source-feasibility question for the next span

H79's joint binary comparator selects zero winner correction in every outer
fold. Its raw non-tie accuracy and tie-margin AUC are weaker than the existing
ensemble. Do not try another binary-model family or fitted margin rescue.
H76also preserves a negative official probability-weight change. A next step
should add a distinct mechanism without another production forward pass.

The primary [model-soups paper](https://arxiv.org/abs/2203.05482) studies averaging
compatible fine-tuned weights to improve accuracy at unchanged inference cost.
That result is motivation, not evidence of a gain here. Unlike a probability
mixture, a weight average produces one new model and requires matched validation.

A direct local configuration read on2026-11-09shows our current checkpoints
are full FP16states with no stored quantization configuration: m0is
Gemma2ForSequenceClassification based ongoogle/gemma-2-9b-it,42layers/3584hidden;
m3is LlamaForSequenceClassification based onRLHFlow/ArmoRM-Llama3-8B-v0.1,
32layers/4096hidden. **These two cannot be weight-averaged.** No compatible
second checkpoint has been established. Do not average across architectures,
packed quantization tensors, class/tokenizer mappings or unknown lineages.

Proposed separate source gate: establish whether a legitimately public second
competition classifier with the same exact Gemma2base/tokenizer/head semantics
and attributable training/checkpoint lineage exists. This is a CPU/network audit,
not a new model download or quality experiment. Inspect at most three named
primary-source candidates, preserve earlier401/403/unsupported-lineage stops,
and stop if no compatible accessible pair can be proved. Do not bypass gates or
substitute an arbitrary mirror. No availability, runtime or score claim is made.

Only after that gate, a separately frozen single equal-weight average could be
considered. It needs streaming tensor/dtype checks and class alignment, a matched
native information pilot, group uncertainty and current-incumbent/runtime
qualification. No coefficient grid or official-score-guided choice. Native
architecture and production pass count would stay the same, but numerical
predictions deliberately change; evaluate matched cohorts consistently rather
than relax a previously failed identity control or combine incompatible caches.

Original deadline, quota anchor, actual charges and official reserve persist.
Queue this source feasibility question for the next span; perform no weight
average or new GPU work in v36. Evidence: [H79result](SPAN_036_RESULTS.md),
[negative official weight trial](SPAN_030_RESULTS.md).
