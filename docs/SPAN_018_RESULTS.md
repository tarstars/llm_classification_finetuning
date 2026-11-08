# V18/H64: teacher check fails; stop before feature inference

The single queued teacher does not beat actual H60on the fixed512rows. Loss is
0.6529882493for H60and0.6532469480for the teacher: reduction**-0.00025870**,
descriptive paired prompt-group95%[-0.00266273,+0.00216893]. It fails the frozen
strict>0.001minimum useful signal. The interval does not establish harm; it
does not justify source/feature/prototype GPU work under this contract either.
Close H64with this early-stop result, no head training, GPU or submission.

The teacher preserved the actual H60predictions exactly on its83selected rows
and used the equal four-orientation mean on the429remaining rows. Base Gemma/
Llama cached predictions match actual H60exactly in every public cohort; IDs,
lengths, route, source artifact hashes and parent mixture reconstruction pass.
The new teacher was fixed and pushed in b0d82e4before its one quality lookup.

| Cohort | Rows | Parent loss | Teacher loss | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Control |256|0.598234396|0.598817230|-0.000582834|
| Uniform |512|0.652988249|0.653246948|-0.000258699|
| Long |64|0.896484400|0.899929270|-0.003444870|

Uniform accuracy is75.20%versus74.80%; control accuracy rises slightly despite
worse loss. Original votes overlap checkpoint training and these cohorts have
been inspected historically. None of these are independent teacher advantage,
competition CV or official-score estimates. No alternate teacher, weight,
temperature or route is tested after this result. H15's deferred independent
teacher-provenance issue remains unresolved; teacher-function learnability was
not measured. A potentially accurate cheap approximation has no demonstrated
useful target here, so no surrogate is fitted.

Three teacher tests pass: retained-route invariance, correct opposite-role
alignment/swap equivariance, and invalid-route rejection. The exact scripts,
goal/resource contract and input/output hashes are recorded in
results/span_018/H64_teacher_replay.json. Per-row arrays remain under
ROOT/runs/span_018/cached_teacher. Keep the original quota period/caps and all
historical evidence; the unused allowance does not authorize a different teacher.

Retain official0.82987and the global rank<=19goal. Executed
notebooks/30_missing_order_teacher.ipynb records this CPU-only result. A fresh
complete inventory on Nov8 observes66owned notebooks, none active; quota remains
16386.583seconds used,91613.417seconds free, no reservations or paid scaling.
All7official receipts remain terminal. The complete08:28:36UTC leaderboard is
23/310, rank19cutoff0.82915, leader0.82702; no new official submission occurred.

Separate read-only planning found an accessible BlackPearl-family three-class
mirror; docs/SPAN_018_CHECKPOINT_DISCOVERY.md distinguishes its actual metadata
from unresolved provenance/quality. Queue the bounded follow-up for v19 without
reopening H64or H55. Preserve70prior snapshots/user notebooks07/12/link, archive
the exact v18contract and carry all37unfinished tasks into the next span.
