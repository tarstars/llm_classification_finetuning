# Preserve native packing while overlapping CPU preparation

H74's larger packing changed probabilities despite exact inputs/checkpoints:
Gemma uniform maximum difference 0.000491977 and Llama control 0.000352919.
The strict 0.00001 guard failed. Preserve that rejection and do not retry caps.
Its incomplete profile establishes neither deployment headroom nor quality.

The incumbent native loop prepares/tokenizes/collates each batch synchronously
on the CPU before launching its microbatches on the GPUs. Its author two-stage
GPU forward already overlaps GPU work. A bounded producer thread could prepare
the next CPU batch concurrently, keeping batch80, zero worker processes, the
exact original8192 grouping/order/tensors, fp16 and forward unchanged. This is a
CPU execution proposal, not another packing or numerical recipe. Old worker
fork crashes are a reason to avoid subprocess data workers; thread safety still
needs to be demonstrated.

Queue one prerequisite for v28, without executing it in v27. First inspect and
time the exact native processor/collator on the frozen cohorts within300CPU
seconds. Prove synchronous and prefetched streams, microbatch boundaries, IDs,
position resets and host tensors identical. A single producer owns CPU dataset
access; all CUDA calls remain on the consumer. Queue depth2, deterministic FIFO,
exception propagation and bounded teardown must be rehearsed, including failure
while the queue is full. Do not change source serialization or model loading.

If the CPU proof passes and fresh resources permit, allow one<=3600s actual
three-role engineering profile. Keep original8192packing, full80warmups, fixed
control/uniform512/long64/test cohorts, originalroute, <=0.00001 old raw controls
on every available cohort and <=95%each-T4memory. Record preparation/queue/forward
timings without double counting overlap; freeze instrumentation before admission.
Use measured end-to-end wall time and retained worst cold/fullwarm/other,
20%steady margin and600sreserve to require a conservative25kprojection<32400s.
The CPU speedup is unknown and may be insufficient. No queue/thread/cap grid,
model change, numerical relaxation, deadline/quota reset or faster-node retry.

If it qualifies, a later H73 contract may retain this original8192 execution path
and test its one still-uncomputed60/40prior. Amend that task's failed prerequisite
explicitly after v27 is sealed; preserve the10240failure and all older snapshots.
No local replay or official candidate is admitted merely by this discovery.

Evidence: [v27 result](SPAN_027_RESULTS.md),
[saved-probability diagnostics](../results/span_027/failure_description.json),
[native synchronous loop](../src/arena/span010_order.py).
