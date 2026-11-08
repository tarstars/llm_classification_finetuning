# H74 CPU packing proof

The native author collator preserves concatenated token streams, row order and
per-sequence positions under the fixed8192-to10240sum-token cap change.
The first full audit completed in15.527CPUelapsed seconds against600s; no
GPU/model probability/quality measurement or target-label read occurred.

| Saved full Gemma length population | 8192packing | 10240packing |
| --- | ---: | ---: |
| Rows | 57477 | 57477 |
| Input tokens | 53256651 | 53256651 |
| Microbatches | 7368 | 5903 |
| Mean tokens/microbatch | 7228.10 | 9021.96 |
| Maximum tokens/microbatch | 8192 | 10240 |
| Maximum individual input | 4330 | 4330 |

This full-population check uses synthetic integer payloads with actual saved
lengths, because the native grouping algorithm depends only on lengths. It is
not a claim that all original text was retokenized. Separately, actual native
inputs for all15old control/uniform/long/test/warm cohorts across3roles have
exact frozen CPUtoken fingerprints under both caps. A4330+4330+100boundary case
changes2microbatches to1without position changes; an intentionally sorted fixture
demonstrates why row-order checks are required. The native sort flag remains false.

There are19.883%fewer microbatches and unchanged total input work. Peak sum of
squared sequence lengths per microbatch rises29718244to38385164. These are
grouping/work proxies, not GPU speedup, memory-fit or quality guarantees.
Changed matrix shapes can alter numerical outputs, so actual raw<=1e-5controls,
memory and full runtime profiling remain mandatory before H73can proceed.

[Audit](../results/span_027/packing_cpu_audit.json)
· [Frozen program](../scripts/audit_span_027_packing.py)
· [GPU protocol](SPAN_027_GPU_PROTOCOL.md).
