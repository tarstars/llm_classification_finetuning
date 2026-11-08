# V19 / H65: native classifier works, but the fixed ensemble is too slow

The accessible Gemma2paired classifier passed the source, input, head, GPU
numerical and retained canonical checks. Its conservative25,000row inference
projection is **34,774.17seconds /9.66hours**, above the32,400second limit.
The declared engineering stop applies: no quality followup, human-vote evaluation
or official submission. Checkpoint quality remains unmeasured.

The original rankgoal remains active. A fresh complete official board at
2026-11-08T11:58:54UTC confirms **23/311,0.82987**, rank19cutoff0.82915 and
leader0.82702. H60remains the deployed best.

## Frozen candidate and actual execution

Checkpoint `wath5/kgl_lmsys_pref_classif`, revision
`cd24909db06583ecf00e65b6919c6cc1531c9885`:42layers/1347tensor keys,
10,361,028,608declared tensor bytes and a finite3x3584F16linear head. Three
original GPTQ8shards were acquired/SHA256verified and privately staged once as
`arseniytrushin/llm-span019-native-classifier/1`. All remote payload file hashes
passed in the actual notebook. Mirror attribution does not authenticate the
original author-run lineage, head training or a claimed winning score. Original
Gemma terms/MIT native implementation attribution are retained; no public weight
redistribution occurred. Source details: [source results](SPAN_019_SOURCE_RESULTS.md).

The fixed candidate is .5retainedGemmaoriginal+.5mirrororiginal,T1, without a
mixture/temperature/format/route grid. Native four-space formatting joins turns
with newlines, uses BOS/truncate2000thenEOS (up to2001tokens), leftpad0and original
absolute positions/pooling. Original GPTQ8Torch execution uses the pinned eager
decoder with softcap50/queryscalar224, one identical replica perT4. Stable length
batching uses8192padded tokens/max16rows. The optional dense converter was not
deployed; its preserved first arithmetic failure and corrected layout tests
remain in the source/CPU evidence.

One public version was accepted and **COMPLETE**:
`arseniytrushin/llm-preference-span019-native-classifier/1`. The service normalized
the title to this canonical slug; the returned accepted reference was used for
every subsequent read. Its source cells match the committed notebook; internet
was disabled. Real hardware/runtime: two15GiBTeslaT4s, Python3.10.13,
Torch2.1.2, Transformers4.42.3. All28retrieved artifacts passed their hashes.

| Check / measurement | Actual result |
| --- | --- |
| Native versus two-replica raw logits,12rows | max difference0 |
| Native versus two-replica probabilities | max difference0 |
| Repeated parallel probabilities | max difference0 |
| Retained canonical control256probabilities | max difference0 |
| Source/head/class/ID/token/asset/hardware checks | passed |
| Mirror80warmup |42.2312s |
| Mirroruniform512, both replicas wall time |298.5762s |
| Mirrortokenization-inclusive uniform rate |0.585047s/row |
| Retaineduniform512 |276.1072s /0.539272s/row |
| Mirrorlong64at2001tokens |124.7767s |
| Mirrorpeak allocated /device |12.1162GiB |
| Setup |169.4898s |
| Retainedstartup/warmup/other |219.7961s |
| Mirrorstartup/warmup/other |55.3075s |
| Conservative25kprojection,20%margin+600sreserve |34,774.1668s |

The mirror measurement is already elapsed time for both replicas; dividing it
again by two would give an invalid runtime. Native-reference diagnostics were
charged to the public pilot and excluded from the hidden production path.
The actual512/full80warmup projection, not a short-prefix extrapolation, fails.
The initial pilot checked repeated probabilities; the prepared but unexecuted
followup additionally guards repeated raw logits. Do not claim that latter GPU
check ran in H65.

## Resources, scientific limits and next decision

Actual public charge: **1705.706two-T4seconds /28.43minutes**, below the3600second
initial cap. Source-first audit used22,430,531bytes/34.41seconds. Asset transfer
was CPU/network work. One public version, zero followups, zero new official
receipts. Terminal inventory:67owned notebooks, none active; all7official
receipts terminal. Quota18092.289used/89907.711free seconds, reserved0, paidFalse,
sameNov14refresh. OriginalNov10deadline and resource limits were not reset.

No model-quality metrics are available. The original512evaluation rows overlap
checkpoint training; later1024votes were previously inspected and differ in
domain. Neither would be independent competition CV. Later comparator routes
were prepared without reading votes (175source/42control rows), but actual source
reverse inference and complete quality comparison did not run. Cached values
were not imputed to fill those gaps. The prepared followup must remain unrun
because the initial engineering gate failed.

Queued **H66forv20**: one memory-conditional12288padded-token/max32-row candidate,
same checkpoint, input, math and ensemble. Under the stated conservative
static/dynamic allocation model, its estimated peak is13.6743GiB, or14.1743GiB
including512MiBuntracked-CUDA reserve, below15GiB. This is an admission estimate,
not actual GPUfit or speed evidence. About1.1564xmirror throughput would be
needed if all other costs stayed unchanged. One bounded test may establish that;
OOM, numerical failure, runtime or fixed quality failure stops it. No fallback
grid. Recompute all mirror cohorts at the new shape and report numerical drift.
See [discovery/proposed bounds](SPAN_019_BATCHING_DISCOVERY.md).

Evidence: [engineering qualification](../results/span_019/engineering_qualification.json),
[memory finding](../results/span_019/memory_discovery.json),
[terminal inventory](../results/span_019/terminal_inventory.json),
[official history](../results/span_019/terminal_official_history.json),
[fresh complete board](../results/span_019/final_leaderboard.json),
[executed notebook31](../notebooks/31_classifier_mirror_audit.ipynb).
