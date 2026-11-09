# Pre-GPU revision: a two-pass candidate, not the over-budget three-pass parent

No new preference targets or GPU predictions have been read. The frozen native
input proof is unchanged:312affected rows and exact native retained-turn inputs.
The initial goal/plan/config are preserved under ROOT/runs/span_035/initial_proposal
with hashes in pre_gpu_design_revision.json. This correction changes the candidate
comparison/runtime plan before spending GPU, without resetting caps or deadline.

The matching three-role prefetch profile is32973.468seconds, above32400. Its
two-role counterpart is30766.762seconds. A rare input-retention change cannot
be assumed to rescue the three-role floor. Freeze the candidate as equal new-
policy Gemma original plus unchanged native Llama swapped, T1. There is no reverse
Gemma pass or router. The initial ablation reference is the unchanged SAME two-
expert recipe, whose existing official score is0.83053. Current H60(0.82987)
remains the incumbent and is reported separately; dropping its mirror is explicit.

The observed rank19cutoff0.82915is0.00138below that two-pass baseline. Require a
source information gain>=0.0015 with positive paired changed-prompt95lower bound
before qualification, rather than the initial three-pass gross-regression guard.
This is a practical predeclared information threshold, not a transfer coefficient
or forecast: original votes overlap checkpoint training. No local result has
selected this rule. Failure stops the recipe; no reverse pass is added back later.

Use ALL312newly affected rows, equal original/new-Gemma plus the same aligned
Llama caches, and fixed original population312/57477. Compare exact two-pass
mixtures on those rows and scale paired prompt-group bootstrap gains by that
fixed prevalence. This is a conditional affected-stratum interval, not an
unconditional population-sampling interval. Fixed-output off/on H60 diagnostics
may be shown, but do not establish full incumbent benefit, especially on unchanged
rows where H60retains a mirror. Actual full qualification and one legitimate
official result remain necessary.

Runtime qualification retains the matching two-role prefetch floors293.909023s
Gemma/213.175659s Llama per512rows, plus any larger actual candidate rates and
all cold/fullwarm/preprocessing/other costs,20%margin/600sreserve. Thus no faster
node or changed inputs can silently erase the known conservative baseline cost.
Both T4s/source/controls<=1e-5/memory<=95%/CSV and<32400s remain mandatory.
No extra GPU version, retry, numerical change, external service or cap increase.

Four training-only examples expose context-loss risk: one code-review final turn
refers to a discarded code block, and a Unity follow-up loses the initial setup.
Retained turns themselves are exact; these examples neither prove semantic harm
nor tune this frozen policy. Their limitations accompany the information pilot.
