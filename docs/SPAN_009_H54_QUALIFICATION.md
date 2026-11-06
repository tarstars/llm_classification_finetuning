# H54: public ensemble qualified for one official submission

Exact accepted version: arseniytrushin/llm-preference-span009-public-ensemble/4,
terminalCOMPLETE. This qualifies engineering/runtime, not generalization or the
rank30 objective. Official score is pending. Model provenance, overlap limitations,
source versions and frozen equal-weight recipe are in SPAN_009_PUBLIC_MODEL_PROTOCOL.md.

Independent audit: results/span_009/H54_qualification.json. Actual remote cells
match the frozen notebook; offline image and inputs match. Strict checkpoint
loading and both-device measurements pass. Recomputed aligned ensemble, IDs,
coverage and visible CSV pass. Both models' fresh uniform256 predictions exactly
match the saved controls (maxdifference0). Reused Gemma long/test outputs match
exact version3 byte-for-byte; that prior service remainsERROR. Serialized branch
tests establish that hidden mode receives no recovery cache and performs full
model inference on the actual hidden text.

| Measurement | Gemma2-9B | Llama3-8B |
| --- | ---: | ---: |
| Uniform256 warm seconds |126.844|94.925|
| Long64 seconds |187.534|116.340|
| Eight-row warmup seconds |22.248|16.712|
| Cold loading/verification seconds |170.673|182.717|
| Maximum observed input tokens |4330|4333|

Conservative projection:25,000rows ×0.866287seconds/row ×1.2 +392.350seconds
startup/warmup +600seconds reserve =**26,980.971seconds /7.495hours**, below
32,400seconds. The original cold-sample projection33,795.914seconds remains a
recordedFAIL; the bounded prospective warm measurement counts startup once,
retaining the same margin and reserve. Long cohorts cover the fixed64 largest
raw-character rows; original raw-length/token limits and all votes are unchanged.

Actual public charges include all failures:10.620+414.992+895.742+786.796 =
**2108.150seconds /35.136minutes**, below2400seconds. No paid scaling. No new
research GPU jobs during hidden evaluation. No local heldout quality claim:
public competition-trained checkpoints may have seen these local rows. The
executed notebook21 shows descriptive scores with this limitation explicitly.

Decision: admit one frozen exact-version code submission using the prepared
guarded controller. Persist its intent/receipt and observe the official terminal
score and fully paginated live leaderboard. A scored submission below the top30
does not complete the persistent user goal; continue with a bounded next phase.
