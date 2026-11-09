# Independent critical review requested by the user

The user explicitly requests **Fable at high reasoning effort**, a critical review
of this project, and a substantive response to the criticisms. Review read-only;
do not modify files, launch experiments/GPU jobs, submit, commit, push or contact
anyone. Do not delegate further. Never walk HOME, ~/prj, Arc/FUSE mounts or
troll_farm. Scope searches to this project and explicitly named project artifacts.
Do not read credentials. No competition text sent to online model endpoints.

The actual objective is team17244411rank<=19on the official Kaggle
llm-classification-finetuning leaderboard after legitimate inference. Latest
observation: rank24/309, loss0.82987;19th-place cutoff0.82915. The objective is
unachieved. There are19.37GPU hours free before costs change, with a32400second
official reserve and original deadline2026-11-10T08:37:37.343197UTC. Check current
files, not a stale handoff, before assuming status. No live GPU job now.

Start with PROJECT_POLICY.md, docs/CONTEXT_HANDOFF.md, GOAL.md, BACKLOG.md and
docs/SPAN_039_PLAN.md. Critically examine the policy and experiment choices too;
do not treat agent-written gates, span structure or documentation as evidence
that the strategy is sound. The project has accumulated many completed negative
or blocked experiments and extensive bookkeeping while the score has plateaued.
Assess whether that work is aligned with the leaderboard objective and the user's
request to spend expensive model tokens carefully.

Key evidence to verify independently:
- Incumbent H60 and native public classifier pipeline; docs/SPAN_013* and
  results/span_013. Two-expert baseline0.83053versus incumbent0.82987 are distinct.
- H59temperature and H76expert-weight results, especially docs/SPAN_030_RESULTS.md
  and source-versus-official transfer. Do not recommend blind leaderboard tuning.
- Runtime/numerical controls and stopped variants: docs/SPAN_026_RESULTS.md,
  SPAN_027_RESULTS.md, SPAN_028_RESULTS.md, SPAN_029_RESULTS.md; code where useful.
  Are requirements and tolerances justified, or do they preclude valid new recipes?
- H78scalar and H79PairRM information screens: docs/SPAN_034_RESULTS.md,
  SPAN_036_RESULTS.md and their aggregate/fold records. H79all outer offsets are
  zero, so its[0,0]bootstrap interval is conditional on identical predictions.
- H80source feasibility audit: docs/SPAN_037_RESULTS.md. Distinguish actual
  incompatibility from unknown provenance and overly restrictive assumptions.
- H35arithmetic coverage: docs/SPAN_038_RESULTS.md, code/tests and coverage.json.
  Coverage is73/57477original versus1/11215later representatives; no blind
  reliability audit occurred. These are not certified whole-answer errors.
- Current H81 audit has committed original and corrected aggregate results. Original64and later48exact model IDs have zero intersection;
  the matched1024cohort still has zero IDoverlap despite near-identical token-length
  marginals. Its composite mismatch gate did NOT pass (only one nonidentity
  post-standardization SMD>.25). A constant-column floating-point artifact was corrected separately with four tests;
  the first result and unchanged gate outcome are preserved.
- Historical H46producer-strength replication failed its stated quality gate;
  E08is deferred. Review actual results before proposing it as already qualified.

Heavy root is /home/tarstars/database/llm_classification_finetuning. Do not scan
the entire database. Specific fold/prediction arrays live under runs/span_034,
span_036and span_039; inspect only what is needed. Normal Python .venv/bin/python,
PYTHONPATH=src:scripts. Read-only access is enough. Preserve user-dirty notebooks
07_error_poster.ipynb and12_backlog_v1_span.ipynb.

Deliver a candid review with:
1. Prioritized findings backed by concrete file/line or artifact evidence. Separate
   demonstrated errors, statistical limitations, strategic criticisms and hypotheses.
2. Assessment of validation leakage/exposure, population mismatch, selection/multiple
   testing, arbitrary gates, numerical identity/runtime claims and uncertainty.
3. Whether we are making useful progress or substituting safe paperwork for score
   improvements; identify work that should stop or be simplified.
4. Two or three feasible next actions ranked by expected chance of reaching rank19,
   with compute/time costs, validation and stopping criteria. Challenge the present
   research direction if warranted; avoid an unbounded new backlog.
5. What evidence could change your mind, and what conclusions we must not claim.

Do not merely endorse existing guardrails or invent measured gains. Be concise
enough to act on, thorough enough to expose substantive mistakes. Your final answer
will be saved verbatim and the primary agent must react point by point.


## Updated invocation and review scope
The user now explicitly authorizes Claude to perform this review. You are running
through Claude Code, requested model alias=fable, effort=high. You are in an isolated
read-only review bundle. These are byte-for-byte copies of project files whose
SHA256 and sizes are in MANIFEST.json; line numbers match the repository at commit
586dc4c. Only Read/Grep/Glob are enabled. Do not access paths outside this directory,
do not delegate and do not request raw text/credentials/network access. Sources or
artifacts absent from the bundle are unavailable: state that limitation explicitly.

Current H81 report, numeric correction and executed notebook are complete/pushed.
The review-selection blocker has been resolved by the user's Claude authorization.
The README is stale at v11 while active backlog is v39: evaluate that discrepancy.
Reports span001..039 let you audit the strategy across all work. Prioritize the
latest public-ensemble stage for actionable recommendations. Read selectively; no
need to consume every source. Produce your final review now, about 1800-3000 words,
with numbered findings and precise repository-relative file:line evidence.
Separate fixes that can be done immediately from experiments requiring a new
prospective protocol. Challenge arbitrary gates, but do not imply a failed historical
gate was passed or that leaderboard feedback is independent validation.
