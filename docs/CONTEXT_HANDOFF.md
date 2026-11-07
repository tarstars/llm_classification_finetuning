# Active goal: above20th place (rank19 or better)

Updated2026-11-07 after the first v10 pilot. Read GOAL.md, PROJECT_POLICY.md,
docs/SPAN_010_PLAN.md and docs/SPAN_010_PACKED_PROTOCOL.md. The goal stays active
until a fresh full leaderboard shows team17244411 at rank<=19. Current best official
remains0.83053/rank30 (v9 submission58068582); last full read07:36UTC had311teams,
rank19/20cutoff0.82915. Ties at the cutoff do not prove above20. No v10 official
submission yet. The first goal turn made progress; the next verified the live
pilot and recovered terminal results. This is not a blocked condition.

## Current span and resources

V10 has40members, focus E15done/H57running; v11empty draft. H55/H56 unselected.
The persistent user goal authorizes this new span after completed rank30; old
rank30 stop rules are historical. Public phase21600s total, same quota period
and deadline2026-11-08T07:27:25UTC. Initial pilot charged2802.478s; second prospective
job<=5400s fits the remaining cap. At most two frozen recipes/two official
submissions; no paid scaling or external text API. Each hidden run needs measured
conservative<32400s with20%margin, all startup/warmup and600sreserve. Never lower
a gate to admit a failed result. All historical snapshots and user notebooks07/12
are preserved. No subagents. Scope searches to this project/named heavy folders.

## First pilot is terminal, not submitted

Exact arseniytrushin/llm-preference-span010-order-completion/1 COMPLETE;
local observer85261 finished.50files retrieved and hash-verified. Independent
H57_order_pilot_qualification.json: both256row base controls match v9 exactly
(maxdifference0); checkpoints/IDs/lengths/routes/four-way alignment/CSV/bothT4s pass.
Projection32998.845s (9.16635h) FAILS9h by598.845s. Preserve this version and failure.
See docs/SPAN_010_ORDER_PILOT_RESULTS.md. Source commit7e3be95, auxiliary9f2e11a.

Fixed route: Jensen-Shannon expert disagreement/work, workproxy=sum(n*max(1,n/2048)),
<=10%work and<=20%rows, deterministic ID tie-break. Equal four aligned predictions
on routed rows, original equal Gemma/Llama mean otherwise. Uniform512route83rows,
work9.9805%. Descriptive loss0.654242->0.653010 (gain.001232, group95%[-.000199,.002712]);
not independent: both models trained original votes. Do not fit weights to these
losses or official feedback. Float64 JS replay differed<6e-17 across libm/NumPy;
auditor uses1e-12relative/1e-14absolute for JS only, exact route indices and unchanged
1e-5prediction controls. The runtimeFAIL remains unchanged.

## Second packing-only recipe in preparation

New files span010_packed.py/build_span_010_packed.py/run_span_010_packed.py.
Only driver difference: packed microbatch tokens8192->12288. Models,4096context,
image/kernels, batch80, routing and weights unchanged. Separate Kaggle slug
arseniytrushin/llm-preference-span010-order-packed; roleorder_packed. Builder proves
all other driver bytes identical and exercises actual serialized cells/startup.
Three packed CPU orchestration fixtures and two leaderboard tests pass; initial
full121suite passed earlier. Recompute all controls/orientations/subset timings;
no cache. Freeze/commit new bundle before launch; never rebuild after push intent.
After admission, recover/observe the exact same accepted version. An observation
timeout does not authorize another GPU mutation. First frozen local files remain
unchanged, so historical source verification still works.

Audit command: scripts/audit_span_010_order.py --role order_packed (or --source-only).
It writes separate per-role qualifications and current H57_qualification.json;
never erase the first failed result. Official submission only after all gates,
using scripts/submit_span_010_order.py --role order_packed; later --observe-only
resumes the same receipt. It verifies frozen/committed source, exact completed
Kaggle version, CSV, fresh jobs/quota, separate intent and candidate count.
Do not submit the failed order_pilot recipe.

## Artifacts and preservation

Use .venv/bin/python. Heavy ROOT=/home/tarstars/database/llm_classification_finetuning;
private artifacts under runs/span_010, aggregate evidence under results/span_010.
Notebook22 is executed with measured first-pilot results and failed runtime;
refresh after the second attempt and official receipt. Builder supports both
attempt histories. Notebook link ~/notebooks/llm_classification_finetuning already
points to this repo/notebooks. Preserve dirty notebooks07/12; stage explicit paths.
Preservation baseline results/span_010/preservation.json hashes38snapshot files
and user notebooks. Backlog CLI controls status/history and generated Markdown.

GOAL.md now contains the above20contract. Exact old contract is docs/GOAL_V9.md;
old accepted Kaggle version4 and commit057370d preserve v9 (rank30/.83053).
Previous own-trained Qwen3-4B official.95481 is historical. Never rerun historical
builders against a different active version. GitHub tarstars/llm_classification_finetuning,
branchmain; commit/push important measured results and verify publication.
