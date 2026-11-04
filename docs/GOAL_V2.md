# Goal: execute a bounded backlog-v2 span and form backlog v3

Research span closed on **2026-11-04**: v2 is sealed, v3 is active with 37 tasks
and no selected focus, and v4 is an empty draft. See
[results](../docs/SPAN_002_RESULTS.md) and [completion evidence](../docs/SPAN_002_COMPLETION.md).
The contract below is preserved; do not execute v3 under this goal.

Work in `/home/tarstars/prj/llm_classification_finetuning`. Follow `AGENTS.md` and
`PROJECT_POLICY.md`. Act autonomously to test useful improvements to human-preference
classification, record honest outcomes and prepare the next version of the backlog.
The user authorized this goal on 2026-11-03. The completed first-span goal is
archived at `docs/GOAL_V1.md`; do not repeat its experiments.

This goal covers **one v2 execution span**, with focus **H40, E01, H03, H37 and
H42**. Prioritize cached calibration, valid training measurements and training
exposure; test a cached ensemble and pursue tie-preserving adaptation only when
its prerequisites and evidence justify it. Other v2 tasks remain available for
carryover. Discoveries enter v3 and do not expand the current span.

Use the previous span's default limits: **12 elapsed two-T4 session-hours total**
and **48 elapsed wall-clock hours**, starting at this goal's original activation.
Record that original time and fixed deadline on the first execution pass, using
the goal service's creation time when available. Preserve them on resumption.
Use the smaller of the GPU cap and verified available Kaggle quota after accounting
for existing jobs. Count setup, controls, probes, evaluation, failures, retries and
any submission execution. Two T4s for one hour count as one session-hour. Do not
use paid scaling or a later quota refresh to enlarge this span. Check current
runtime rules and quota before launch; individual jobs must fit the verified
session limit and a conservative seven-hour job cap.

1. **Resume safely and freeze the contract.** Read the policy, v1 report, backlog,
   goal state and existing span-002 artifacts. If v2 has already closed, verify its
   report, final snapshot and v3 handoff instead of rerunning it. Reconcile exact
   remote job/version state before retries; never duplicate a live job or reset
   the clock because an agent restarted.

   ```bash
   uv run python scripts/backlog.py check
   uv run python scripts/backlog.py versions
   uv run python scripts/backlog.py list --version 2 --focus
   uv run python scripts/backlog.py list --actionable --focus
   ```

   Before new measurements, write `docs/SPAN_002_PROTOCOL.md` and an aggregate plan
   under `results/span_002/`: original start/deadline, current quota and jobs, exact
   split/sample/group IDs and hashes, baseline/checkpoint identity, pinned code and
   configuration, maximum variants, controls, exposure, budget and stopping rules.
   Verify cached prediction IDs, labels and coverage before joining them. Reuse
   the attempt-002 split and frozen deployed classifier on common evaluation rows.

   Existing development and confirmation partitions are exploratory, not untouched.
   Keep confirmation out of this span. Use grouped inner/nested selection for
   calibrators and ensemble weights, with a cross-fitted temperature-only control.
   Report the historically calibrated deployment reference separately. In-sample
   training predictions are not OOF predictions. Predeclare any larger development
   evaluation before selecting a candidate.

2. **Execute a bounded research loop.** Read each selected task's full record,
   prepare a falsifiable experiment, update status through the CLI, execute and
   monitor it, and write results/evidence before moving on. Independent CPU work
   can proceed while remote jobs run. Initial tests are:

   - **H40:** on cached frozen and paired-model predictions, compare temperature
     alone with at most two low-capacity A/B-symmetric calibration extensions,
     including a tie intercept. Fit/select within grouped folds and report
     out-of-fold probabilities. Determine whether calibration explains the paired
     model's tie deficit. No GPU training is needed.
   - **E01:** directly log nominal cycles, accepted/skipped updates, scaler and
     scheduler state, processed versus accepted tokens and private row/class/group
     exposure. Verify a tiny forced-overflow and resume probe against actual
     parameter changes. GPU verification is capped at 15 minutes. Finish this
     prerequisite before longer training comparisons; do not rerun v1 for logging.
   - **H03:** inventory optimizer/scheduler/RNG checkpoints first. Preserve selected
     checkpoint 800 and distinguish it from final step 877. If exact state exists,
     test continuation toward one epoch with a fixed schedule and endpoints.
     Otherwise test at most one explicitly labeled fresh warm-start schedule if a
     useful controlled comparison fits the budget. Retain the original baseline
     and measure accepted exposure through E01. The previous 0.61 epochs does not
     prove undertraining; stop at a predeclared plateau/overfit rule.
   - **H37:** first cross-fit one nonnegative scalar blend of the frozen classifier
     and existing prompted-Gemma judge on matched cached rows, retaining both
     single-model controls. Do not train a new component under this task. Expand
     prediction coverage only after a promising result and verified runtime/quota.
   - **H42:** run only after H40 and E01. If calibration resolves the measured
     tradeoff, document that decision and stop without this GPU experiment.
     Otherwise compare at most two matched arms: paired CE + 0.1 symmetric KL,
     versus the same objective with a Bernoulli tie-probability anchor to the
     frozen, symmetric, uncalibrated teacher on training rows, default weight 1.0.
     Never feed a dev-fitted calibrator into training. Require benefit against
     both the paired control and deployed classifier; count teacher/control costs.

   Freeze task allocations before GPU launch and keep recovery inside the total
   cap. Prioritize informative pilots. Default useful-gain thresholds are 0.001
   absolute log loss for cheap postprocessing and 0.005 for training pilots;
   record any justified change before seeing new results. Report paired
   group-bootstrap uncertainty, overall/class/tie loss and recall, tie discrimination,
   order behavior, retention, accepted exposure, throughput, memory and charged
   compute. An inconclusive pilot does not prove success or failure. Adoption
   requires a larger prespecified comparison with positive paired evidence,
   practical gain and feasible deployment.

   Preserve negative results and failures. `done` means the declared experiment
   finished, including a negative outcome. `rejected` requires an evidenced
   validity/feasibility reason; `deferred` requires a concrete constraint and
   carryover recommendation. Link real evidence for every focus-item outcome.
   Optional competition submission is limited to one frozen candidate that passes
   adoption/runtime gates and fits the budget; submission is not required.

3. **Form the v3 draft from evidence.** Check existing tasks before adding a
   discovery. Queue only useful new tests with the observation/report, originating
   task, claim, gain estimate, risks, costs, dependencies and acceptance/stop rules.
   Use the CLI to preserve stable IDs and provenance:

   ```bash
   uv run python scripts/backlog.py template > /tmp/arena_v3_proposal.json
   # Fill in the actual evidence-grounded proposal before queueing it.
   uv run python scripts/backlog.py queue --item-file /tmp/arena_v3_proposal.json \
     --from H40 --note "Describe the actual discovery" \
     --evidence docs/SPAN_002_RESULTS.md
   uv run python scripts/backlog.py list --version next
   ```

   Do not create filler tasks or execute draft v3 work. Existing unfinished v2 work
   carries forward automatically when the span closes.

4. **Publish reproducible outcomes.** Produce `docs/SPAN_002_RESULTS.md`, aggregate
   metrics/decisions under `results/span_002/`, and executed notebook
   `notebooks/13_backlog_v2_span.ipynb`. Include exact variants/coverage, comparators,
   uncertainty, selection limitations, actual resource use, negative findings,
   adoption decisions and next steps. Keep text, per-row predictions, datasets,
   vectors, weights and heavy intermediates under
   `~/database/llm_classification_finetuning/runs/span_002` or existing heavy paths.
   Preserve the notebook link under `~/notebooks`.

   Update backlog history and generated views through the tooling, validate them
   and run checks appropriate to changes. Commit/push important milestones. Preserve
   unrelated user changes, including the kernel display-name edit in
   `notebooks/07_error_poster.ipynb`. Refresh project memory and context handoff.
   Distinguish local estimates from official scores; neither a gain nor 0.8 is
   guaranteed.

5. **Seal v2 and hand off v3.** Resolve every focus item with an evidenced outcome
   or justified deferral; finish or stop all jobs. Write the final report,
   recommendations and resource audit before advancing:

   ```bash
   uv run python scripts/backlog.py advance \
     --note "Summarize the measured v2 outcome and reason for closure" \
     --evidence docs/SPAN_002_RESULTS.md
   uv run python scripts/backlog.py list --version 3 --new
   uv run python scripts/backlog.py list --version 3 --carried
   uv run python scripts/backlog.py check
   ```

   Verify sealed v2 start/final snapshots, v3 discoveries plus unfinished carryovers,
   and the new empty v4 draft. Keep historical records and original v1 snapshots
   intact. Commit/push the handoff and **stop before executing v3**.

Success means a substantive, honestly evaluated v2 span with reproducible results
and a justified versioned v3 backlog. Positive score improvement is not required.
Identify budget-limited work as unfinished and carry it forward. If access or
resources prevent meaningful execution, record the blocker, complete independent
useful work and leave the goal unfinished rather than claiming success by
deferring everything. Do not erase outcomes or mark unexecuted experiments done.
