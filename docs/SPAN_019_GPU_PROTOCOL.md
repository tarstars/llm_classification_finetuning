# H65: native replicated decoder pilot, then the frozen quality comparison

The retained expert's custom decoder cannot be reused unchanged: it has an MLP
head, concatenated gate/up projection and an attention branch that omits logit
softcapping. The mirror has a linear head and native Gemma2 attention. Use the
pinned image's Transformers4.42.3 eager decoder, query scalar224, softcap50,
separate gate/up projections, original norm/GELU and the unchanged head.

Choose the production backend prospectively, before any real model outputs:
two original GPTQ8 native replicas, one perT4. Declared weights total10.36GB,
below one T4's memory; actual activations/fit are an engineering gate, not yet
verified. Use the audited official AutoGPTQ0.7.1Torch fallback and load all1347keys
exactly for each replica. Compare a sequential GPU0reference against the same
packets assigned acrossGPU0/GPU1, plus a repeated parallel execution. Strict raw
logit/probability difference<=1e-5, head SHA256, finite/simplex and ID/token checks
remain. Keep the original shard bytes; no new quantization or head replacement.

This selects direct native execution after the tiny saved-checkpoint tests made
that path concrete. It supersedes the earlier prospective dense-expansion asset
path: CPU expansion remains a tested optional storage tool and its failed/layout
correction evidence stays preserved, but no18.5GBdense copy is needed in the
selected deployment. Source/CPU checks already passed; all quality gates, original
deadline, quota period and resource caps are unchanged. No quality outputs have
been inspected, and this is not another model/mixture/format variant. It avoids
converting model math just to reuse the retained expert's incompatible kernels.

Inputs remain the frozen formatter/turn join/BOS/truncate2000+EOS/pool. Batch in
stable length order, padded-token budget8192/max16rows; restore original ID order.
Remove redundant left padding while preserving each actual token's original
native absolute RoPEposition and source pool index, including literal pad tokens.
Masks describe added padding, not literal token0. Every sequence<=2001tokens is
below4096slidingwindow. Reference and production use identical tensors, positions,
masks and pool indices. Both use an explicit common FP32softmax of unchanged head
logits for valid probabilities; the card's F16softmax is a storage-precision
variant, not a fitted calibration alternative. Do not claim different batch
shapes or the literal padded single-row example are a numerical identity check.

Controls use first8frozen original control rows plus4fixed synthetic edge cases,
no winner labels. Independent native input/pool tests and actual saved tiny
original-GPTQ/dense/split tests run in Transformers4.42.3 on CPU. They do not prove
real model GPUagreement, memory or throughput. Public GPUcontrols must pass before
scoring other inputs. Record both devices' actual memory, rows and times.

Initial public version<=3600seconds is engineering only: native controls first,
then full80warmup and actualuniform512profile for both deployed experts, retained
Gemma control256andlong64stress, and a CSV on visible test3. No winner labels or
quality metrics are read by that job; source1024quality need not fit this first
engineering allowance. This first version cannot qualify an official submission.

Only after engineering/numerical/runtime gates pass, admit one followup within
the original14400second cap to complete the SAMEfrozen original512and later1024
comparison from ASSET_PROTOCOL.md. No model/input/ensemble/route/temperature search.
The actual H60later baseline needs its fixed routed reverse predictions; no
imputation. Strict quality thresholds/uncertainty stay unchanged. Repeat actual512/
full80warmup and canonical checks in the final qualified recipe. Conservative25k
cost includes20%per-row margin, worst comparable parent/current retained timings,
all production cold/warmup, verification/tokenization/other overhead and600reserve;
it must be<32400seconds. Native-reference-only public diagnostic time is charged
to the public budget but is not part of the deployed hidden execution path.

Privately stage only the original10.36GBweights/tokenizer, audited native source,
original attributions and a strict file manifest, with no extra archive or public
redistribution. No converted weights or hidden predictions in notebook outputs.
Before GPU, execute actual serialized worker success/failure/watchdog and public/
hidden branch rehearsals. Hidden inference reads only actual test text and pinned
assets, no training votes, later labels or public prediction caches. Fresh full
quota/jobs/pinned dependencies/all official cursors, paidFalse, same original48hour
period. At most2accepted public versions, one production quality recipe and one
frozen official after full qualification.
