# Project policy: research backlog

This policy governs experiment selection, execution, reporting and project memory
for contributors and coding agents. Explicit user instructions take precedence.

Use the backlog to distinguish intended work, completed experiments and supported
conclusions. [backlog/hypotheses.json](backlog/hypotheses.json) is the source of
truth; [BACKLOG.md](BACKLOG.md) is generated. Use
[scripts/backlog.py](scripts/backlog.py) for status changes and history. Edit the
JSON for estimates, priorities, dependencies and experiment design, then regenerate
the Markdown. Preserve stable hypothesis IDs and previous outcomes. A backlog
version defines a research span; its `focus_ids` define that span's bounded
execution scope. [GOAL.md](GOAL.md) describes the current span and its limits;
completed contracts are preserved at [docs/GOAL_V1.md](docs/GOAL_V1.md),
[docs/GOAL_V2.md](docs/GOAL_V2.md), [docs/GOAL_V3.md](docs/GOAL_V3.md),
[docs/GOAL_V4.md](docs/GOAL_V4.md), [docs/GOAL_V5.md](docs/GOAL_V5.md),
[docs/GOAL_V6.md](docs/GOAL_V6.md),
and [docs/GOAL_V7.md](docs/GOAL_V7.md),
[docs/GOAL_V8.md](docs/GOAL_V8.md),
[docs/GOAL_V9.md](docs/GOAL_V9.md),
[docs/GOAL_V10.md](docs/GOAL_V10.md),
[docs/GOAL_V11.md](docs/GOAL_V11.md),
and [docs/GOAL_V12.md](docs/GOAL_V12.md),
and [docs/GOAL_V13.md](docs/GOAL_V13.md),
and [docs/GOAL_V14.md](docs/GOAL_V14.md),
and [docs/GOAL_V15.md](docs/GOAL_V15.md),
and [docs/GOAL_V16.md](docs/GOAL_V16.md),
[docs/GOAL_V17.md](docs/GOAL_V17.md),
[docs/GOAL_V18.md](docs/GOAL_V18.md),
[docs/GOAL_V19.md](docs/GOAL_V19.md),
[docs/GOAL_V20.md](docs/GOAL_V20.md),
and [docs/GOAL_V21.md](docs/GOAL_V21.md),
and [docs/GOAL_V22.md](docs/GOAL_V22.md),
and [docs/GOAL_V23.md](docs/GOAL_V23.md),
and [docs/GOAL_V24.md](docs/GOAL_V24.md),
and [docs/GOAL_V25.md](docs/GOAL_V25.md),
[docs/GOAL_V26.md](docs/GOAL_V26.md), and
[docs/GOAL_V27.md](docs/GOAL_V27.md), and
[docs/GOAL_V28.md](docs/GOAL_V28.md), and
[docs/GOAL_V29.md](docs/GOAL_V29.md), and
[docs/GOAL_V30.md](docs/GOAL_V30.md), and
[docs/GOAL_V31.md](docs/GOAL_V31.md).
A completed goal file does not authorize
executing its successor backlog; select a new bounded goal first.

| Status | Meaning |
| --- | --- |
| `planned` | Recorded, awaiting preparation or prerequisites |
| `ready` | Dependencies completed; test and resource budget prepared |
| `running` | Work is underway, including an active remote job |
| `done` | Experiment finished, including negative or inconclusive results |
| `deferred` | Waiting for resources or stronger evidence |
| `rejected` | Ruled out, with a recorded reason and supporting evidence |

1. **Choose actionable work.** Check the backlog before beginning an experiment.
   Complete prerequisites first, then weigh priority, plausible gain, risk and
   engineering/compute cost together. Sorting by gain uses the optimistic endpoint,
   not expected return. A high priority is not an instruction to run every item.

   ```bash
   uv run python scripts/backlog.py list --actionable
   uv run python scripts/backlog.py list --actionable --focus
   uv run python scripts/backlog.py versions
   uv run python scripts/backlog.py list --priority P1
   uv run python scripts/backlog.py show H09
   ```

   E00 establishes the evaluation and resource contract. Dependencies must be
   `done` before dependent work becomes `ready`, starts or is marked `done`.
   A rejected dependency does not unblock its dependents: reconsider their design,
   revise dependencies with an explanation, or defer/reject the dependent work.

2. **Prepare a bounded experiment before spending compute.** Record the baseline,
   exact split/sample IDs, grouping, seed, primary change, metric, maximum variants,
   training exposure, calibration, resource budget and advancement/stopping rules.
   Make one main change per initial ablation. Record source/configuration/checkpoint
   hashes and the locations of inputs and outputs.

   Exercise generated notebook subprocess cells through their final bookkeeping
   with a tiny CPU worker before GPU admission. Syntax compilation alone does not
   catch an undefined runtime name. For new templates, use the tested literal-role
   helper in `arena.notebook_worker`, and cover successful exit, failed exit and
   the owned-process watchdog. Preserve already accepted historical notebooks.
   A failed service status stays failed even when separately verified complete
   worker artifacts can be recovered without another GPU execution.

   Verify current Kaggle quota, active jobs, model availability, memory fit and
   submission runtime before launching GPU work.
   For a native inference server, check that the backend itself detects the requested
   GPUs and resolves its shared libraries: the span-002 native backend could not
   load CUDA despite allocated GPUs. `nvidia-smi` or PyTorch availability alone
   does not establish native-backend execution. Run a bounded real
   profile before large scoring. Preserve failed numerical-identity checks; if the
   numerical recipe changes, re-evaluate matched cohorts consistently rather than
   pooling incompatible cached outputs.
   Distinguish engineering person-hours,
   CPU elapsed hours and elapsed two-T4 session hours. Count pilot and follow-up
   separately, plus controls, OOF prediction generation and retries when required.
   Jobs exceeding a session limit need a tested resume path and sufficient quota.

   Gain ranges are subjective planning estimates, not measurements, confidence
   intervals or guarantees. Positive gain means lower log loss; regressions are
   possible. Overlapping gains cannot be added.

3. **Record work as it happens.** Update status when preparation completes, work
   starts, a meaningful milestone occurs, or work stops. Every update needs an
   informative note. Moving directly from `planned` to `running` is allowed when
   dependencies are satisfied and the experiment is prepared. For example, after
   E00 is completed:

   ```bash
   uv run python scripts/backlog.py update H09 --status running \
     --note "Starting the predeclared paired-order pilot"
   ```

   The CLI appends dated history and regenerates the Markdown. It updates tracking;
   experiment programs launch and monitor jobs separately. Include the run ID or
   artifact location in progress notes. Serialize backlog edits to preserve history.
   Within the active span, explain any reopened attempt and retain earlier results.
   After a span closes, create a new follow-up for further work on a completed task;
   the historical task and its snapshots remain preserved.

4. **Evaluate and close with evidence.** Compare candidates on identical rows with
   matched controls. Report overall multiclass log loss, paired uncertainty, class
   behavior and relevant retention/order/runtime diagnostics. Use group-aware
   evaluation and paired group-resampled intervals where applicable. Treat pilot
   advancement as an information decision; a small inconclusive pilot does not
   establish either success or failure. Follow the predeclared budget and stop rule.

   Existing development and confirmation partitions have been inspected and must
   be described as exploratory. Use grouped inner/nested selection to limit further
   selection bias and independent data when available. Fit calibrators, gates,
   retrieval examples and other learned components within the appropriate folds.
   In-sample training predictions are not OOF predictions. Keep target-derived
   fields and unavailable model identities out of deployed inputs. Preserve original
   evaluation votes, and freeze a candidate before a competition submission.

   Save a report containing configuration, comparator, results, uncertainty, actual
   resource use and the decision. Link the executed notebook and aggregate metrics.
   Only after the report exists, close the item with its actual path; for example:

   ```bash
   uv run python scripts/backlog.py update H09 --status done \
     --note "Pilot completed; report contains the measured outcome" \
     --evidence docs/H09_RESULTS.md
   ```

   `done` means the experiment finished; it does not mean the hypothesis improved
   the score. Keep negative and inconclusive findings. Use `rejected` for an idea
   ruled out by evidence or a documented validity/feasibility problem, and `deferred`
   for work that may become worthwhile later. Record the reason in either case.
   Closing as `done` or `rejected` requires a result/decision reference.

5. **Revise and version the backlog when evidence changes.** Update estimates, priorities,
   risks, dependencies and test design in the JSON, explaining the revision in the
   item. Keep forecasts separate from measurements in linked reports. Preserve
   existing identifiers rather than deleting or repurposing them. Git history retains
   earlier wording and estimates; status history retains decisions and outcomes.

   The active version has a fixed task membership and a selected focus. Keep new
   discoveries in the next-version draft so research cannot silently expand the
   current goal. Create a complete proposal from the CLI template, then queue it
   with its originating task and actual supporting observation/report:

   ```bash
   uv run python scripts/backlog.py template > /tmp/arena_followup.json
   # Fill the template with a complete, evidence-grounded proposal.
   uv run python scripts/backlog.py queue --item-file /tmp/arena_followup.json \
     --from H09 --note "State the actual discovery" --evidence docs/ACTUAL_REPORT.md
   uv run python scripts/backlog.py list --version next
   ```

   **User-requested planning before a span starts:** when the active version has
   just been formed but has never selected a focus or started preparation/work,
   an explicit user request may add tasks to that upcoming span with
   `queue --into-active`. Link a plan recording the request with `--evidence`.
   The CLI appends planned items and version-history amendments, preserving the
   original start snapshot and all earlier snapshots. Parents must originate in
   earlier versions; same-span prerequisites belong in `depends_on`. This does
   not select focus, launch work or reset a goal. After focus/preparation begins,
   use the ordinary next-version queue. The default fixed-membership rule remains.

   IDs continue as H41, H42, and so on for hypotheses, or E01 onward for
   prerequisites. `introduced_in`, `discovered_during` and `derived_from` preserve
   provenance. Draft tasks cannot become ready, run or be marked done before their
   version activates. Both [BACKLOG.md](BACKLOG.md) and the live
   [next-version view](backlog/NEXT.md) are generated from the registry.

   Before closing a span, resolve every focus item as `done`, `rejected` or
   justifiably `deferred`, with actual evidence; a focus deferral needs a linked
   decision. Finish or stop every running job/task. Produce the span report, then:

   ```bash
   uv run python scripts/backlog.py advance \
     --note "Describe the actual outcome" --evidence docs/ACTUAL_SPAN_REPORT.md
   uv run python scripts/backlog.py list --version 1
   uv run python scripts/backlog.py list --new
   uv run python scripts/backlog.py list --carried
   ```

   `advance` preserves final JSON/Markdown snapshots, activates the next version
   with new discoveries plus unfinished carried work, and opens another draft.
   Completed/rejected tasks remain in the registry as dependencies and evidence.
   Never discard unfinished work merely because it was not selected for the span.
   Do not run the next span automatically when a goal ends at the version boundary.

   Start and final snapshots live under `backlog/versions/vN.start.*` and
   `vN.final.*`; hashes protect their recorded contents. Historical CLI reads use
   final snapshots, so later carried-task updates do not rewrite earlier results.
   Preserve snapshot files. A later span's goal selects its focus before execution;
   changing focus requires an explained decision and no running tasks:

   ```bash
   # Example after v2 activates, using task IDs actually present there:
   # uv run python scripts/backlog.py focus H41 H23 --note "Explain this span's selection"
   ```

   ```bash
   uv run python scripts/backlog.py render
   uv run python scripts/backlog.py check
   ```

   Keep the original H01–H40 records for traceability. New IDs are supported by
   the tooling without changing the validation contract for each follow-up.

6. **Validate and publish important milestones.** Run `check` after updates. Run
   focused tests when changing tooling or experiment code; for backlog tooling:

   ```bash
   uv run python -m unittest discover -s tests -p test_backlog.py -v
   git diff --check
   ```

   Review the diff and commit/push the relevant JSON, generated Markdown, source,
   reports, aggregate results and notebooks together. Stage explicit paths and
   preserve unrelated user changes. Report the outcome, remaining uncertainty and
   commit when handing work back to the user.

   Store datasets, model weights, cached vectors, per-row predictions and heavy
   intermediate files under `~/database/llm_classification_finetuning`. Keep programs,
   Markdown project memory, aggregate metrics and notebooks in Git. Notebook files
   remain in the project's `notebooks/`, linked from
   `~/notebooks/llm_classification_finetuning`. Keep credentials out of source,
   notebook output, logs and commits.

The CLI validates record structure, IDs, dependency cycles, execution-version and
dependency gates, closure references, span focus resolution, snapshot hashes and
both generated Markdown views. Contributors must review scientific validity,
actual evidence contents, artifact provenance and available resources.

Scope file searches to this named project. Never recursively walk `$HOME`, `~/prj`,
the Arcadia/Arc FUSE checkouts, or
`/media/tarstars/medium_data/database/troll_farm`.
