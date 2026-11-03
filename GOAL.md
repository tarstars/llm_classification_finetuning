# Goal: execute the first v1 research span and prepare backlog v2

Work in `/home/tarstars/prj/llm_classification_finetuning`. Follow `AGENTS.md` and
`PROJECT_POLICY.md`. Improve our understanding of how to reduce human-preference
classification log loss through controlled experiments and reproducible evidence.
Act autonomously within the stated scope and resource limits.

This goal covers **one execution span**. Its selected tasks are **E00, H01, H09,
H04 and H11** from backlog v1: the evaluation/resource contract followed by pilots
of a preference-trained backbone, swap consistency, broader LoRA coverage and a
learned symmetric tie head. The remaining v1 tasks stay available for later spans.
Discoveries made during this span belong to backlog v2 and do not expand this goal.

The default limits for this goal are **12 elapsed two-T4 session-hours** in total
and **48 elapsed wall-clock hours**, beginning when execution of the goal starts.
Use whichever GPU allowance is smaller: that cap or the verified currently available
Kaggle quota after accounting for existing jobs. Count controls, probes, failures,
retries and evaluation; two T4s running for one hour consume one session-hour in
this budget. Verify how Kaggle's reported quota units map to that allowance.
Use currently authorized resources; do not wait for a later quota refresh to extend
the span. These caps can be edited by the user before setting the goal.

1. **Resume safely and establish the contract.** Read the policy, backlog, version
   metadata, recent experiment reports and active job state. Check whether work for
   this span already exists before launching anything. If v1 has already closed,
   verify its final snapshot, report and v2 handoff rather than repeating the runs.

   ```bash
   uv run python scripts/backlog.py check
   uv run python scripts/backlog.py versions
   uv run python scripts/backlog.py list --version 1 --focus
   uv run python scripts/backlog.py list --actionable --focus
   ```

   Start with E00. Write `docs/SPAN_001_PROTOCOL.md` with the original start time,
   deadline, quota snapshot, remaining budget, selected task IDs, exact grouped
   evaluation protocol, matched controls, artifact locations, maximum variants,
   practical gain thresholds and stopping rules. Reuse these recorded limits on
   resumption; restarting the agent must not reset the clock or compute allowance.
   Inventory exact checkpoint/configuration hashes and prediction coverage. Existing
   development and confirmation data are inspected and exploratory.

2. **Execute the bounded research loop.** Repeatedly select the highest-value
   actionable task from the declared focus. Read its complete record. Prepare a
   falsifiable pilot, change its status with a useful note, execute and monitor it,
   evaluate against a matched comparator, and record the outcome before moving on.
   Do not launch duplicate jobs. Respect dependencies and include control costs.
   Start another task when independent useful work can proceed.

   Prioritize informative pilots over full follow-up training. A larger run is
   appropriate only when pilot evidence supports it and the remaining scope/budget
   permits it. Otherwise create a v2 follow-up. Report calibrated overall log loss,
   paired uncertainty, class/tie and order behavior, throughput, memory, retention
   and actual compute use. Separate model-relative improvements from improvements
   over our deployed classifier. Preserve failed and inconclusive findings.

   Use `done` when the declared experiment finishes, including a negative result.
   Use `rejected` for an evidenced validity/feasibility failure. Use `deferred` when
   worthwhile work must carry forward because of a documented constraint. Give
   every focus-item decision a real evidence reference. Label pilot coverage
   explicitly; completion of a pilot does not establish a full-training gain.

3. **Queue discoveries for v2.** Turn observations into concrete follow-up tasks
   only when they justify a useful next test. Include the claim, evidence, gain
   range, risks/mitigation, engineering and compute estimates, dependencies,
   acceptance/stopping criteria and a link to the originating task/report.

   ```bash
   uv run python scripts/backlog.py template > /tmp/arena_followup.json
   # Fill the template with the actual proposal and estimates before queueing it.
   uv run python scripts/backlog.py queue --item-file /tmp/arena_followup.json \
     --from H09 --note "Describe the actual observation motivating this follow-up" \
     --evidence docs/SPAN_001_RESULTS.md
   uv run python scripts/backlog.py list --version next
   ```

   Review the draft before adding a proposal to avoid duplicating existing work.
   The CLI assigns stable new IDs, records discovery during v1 and introduction in
   v2, and leaves the current span unchanged. Queue only real findings; do not create
   filler tasks to make v2 appear productive. Do not execute v2 tasks under this goal.

4. **Publish reproducible results.** Write `docs/SPAN_001_RESULTS.md`, per-task
   details as needed, aggregate machine-readable metrics and an executed notebook
   at `notebooks/12_backlog_v1_span.ipynb`. Include the baseline, exact tested
   variants, sample/exposure coverage, measured gains/uncertainty, actual resource
   use, negative findings and recommendations. Put raw text, predictions, datasets,
   weights and heavy intermediate files under
   `~/database/llm_classification_finetuning/runs/span_001` or their existing heavy
   artifact locations. Keep programs, reports, aggregate results and notebooks in Git.

   Update the backlog through its CLI, validate it, run focused checks appropriate
   to code changes, and commit/push important milestones. Preserve unrelated user
   changes. Do not describe an exploratory estimate as an official Kaggle score or
   assume that 0.8 is attainable from these pilots.

5. **Close the span and hand off v2.** When the focus items have evidenced outcomes
   and all jobs are finished or stopped, write the span's decision and next-version
   recommendations. Then seal v1 and create the v2 handoff:

   ```bash
   uv run python scripts/backlog.py advance \
     --note "Summarize the actual span outcome and the reason for closure" \
     --evidence docs/SPAN_001_RESULTS.md
   uv run python scripts/backlog.py list --version 2 --new
   uv run python scripts/backlog.py list --version 2 --carried
   uv run python scripts/backlog.py check
   ```

   `advance` preserves v1's final JSON/Markdown snapshots, activates v2 with queued
   discoveries and unfinished carried tasks, and opens an empty v3 draft. Completed
   tasks remain available as dependency/evidence records. Commit and push the handoff.
   **Stop after producing v2; a later goal selects its execution focus and budget.**

The finished deliverable is an honestly evaluated first span, preserved v1 history,
executed results and a justified v2 backlog. Positive model improvement is not
required for a useful research outcome. Budget-limited work must be identified as
unfinished and carried forward. If access/resources prevent meaningful progress,
record the blocker and remaining work, continue independent useful preparation,
and report the goal as unfinished rather than declaring success by deferring
everything. Do not erase outcomes or mark unexecuted experiments done.
