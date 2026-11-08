# H65 source and CPU controls: candidate available, quality unmeasured

The pinned public mirror is structurally usable. Its source audit downloaded
22,430,531bytes in34.4seconds without a complete shard or model inference.
All1347tensor keys match the index:42layers,294GPTQ8projections and a finite
3x3584F16head. Head row norms are1.15285/1.15168/1.15073. This does not authenticate
the head's training, original author-run lineage or any reported winning score.
No new score estimate is available; retain official0.82987/rank23.

The mirror's effective native pad is0, overriding config1. Its two cards agree
on formatting. Eight synthetic checks reproduce exact token IDs and pooling,
including left padding, embedded special tokens and the possible2001sttoken
after2000-token truncation plus appendedEOS. Preserve the mirror's unusual
first-pad-minus-one semantics, rather than replacing them with a generic
last-nonpadding implementation. Joining competition turn lists with newlines is
the explicit application recipe; the card itself gives single-turn examples.

The first synthetic conversion test failed by0.015625in F16projection outputs.
Decoded weights were bit-identical to the official AutoGPTQv0.7.1fallback.
Restoring its physical input-by-output matrix layout gives exact agreement on
the same three seeded synthetic matrices. The first failure is preserved in
native_cpu_attempt_1.json; native_cpu_controls.json records the correction with
maximum difference0for all three. The1e-5tolerance was unchanged. Full model
GPU probability agreement is still unmeasured and is not inferred from this test.

The fixed prospective candidate is equal original-orientation retainedGemma and
mirror probabilities,T1,no new route/calibration/TTA. Exact originalcontrol256/
uniform512/long64and existing later1024IDs/hashes were frozen without reading
winner labels. Original votes overlap training; later votes are previously
inspected and differ in domain. The quality gates and bounded public protocol
were published before weights in docs/SPAN_019_ASSET_PROTOCOL.md. No candidate
outputs have been observed. Source and CPU checks admit only offline acquisition/
streaming expansion; no GPU or official is admitted yet. Preserve the original
deadline/quota period and all historical snapshots/user notebooks.

The range-reader tests reject ignored, incorrect and excessive responses before
consuming a weight body. Converter tests verify original head preservation and
known quantized values through actual serialized safetensors shards. No new
quantization or training occurs. Expansion must preserve the original fixed
GPTQ8levels and classifier head; originally floating modules keep their layout.
All heavy files remain under ROOT/runs/span_019. The source checkpoint's model
card and applicable Gemma terms remain attached; no separate derivative license
or independently authenticated mirror lineage is claimed.

Executed interim notebook: notebooks/31_classifier_mirror_audit.ipynb, two code
cells and zero errors. At09:01:56UTC, the acquisition's live exec handle35176was
still running, with742,391,808bytes of the first shard received. The committed
acquisition_live_observation.json is a dated observation, not completion evidence.
No GPU quota has been consumed by these source/asset stages. H65 remains running;
native whole-model controls, quality comparison and deployment runtime are pending.

At09:49:06UTC all three original shards completed with their pinned SHA256,
total10,361,181,040bytes. The native private asset package has16files/10.383GB;
hard links avoid duplicating its heavy files. The selected prospective GPUbackend
is two native replicas, one perT4, as specified in SPAN_019_GPU_PROTOCOL.md.
No dense conversion is needed for this selected deployment. The optional CPU
converter and its preserved layout failure/correction remain reproducible evidence.

Actual saved tiny original-GPTQ/dense/split tests and public/hidden orchestration
tests pass (5tests); two runtime-formula tests also pass. The real serialized
worker passed success/failure/watchdog, notebook startup and public/hidden branch
rehearsals. These do not establish real model GPUfit, agreement or quality.
Before GPU, the exact token/position/pool fingerprints were recorded for all
fixed cohorts. Uniform512mean662.14tokens; later1024mean685.23; all64long stress
rows reach2001tokens. No winner labels were read. There is still no new score.
