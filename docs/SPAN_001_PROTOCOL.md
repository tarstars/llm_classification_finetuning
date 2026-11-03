# Span 001: evaluation and resource protocol

Frozen before new GPU quality measurements. Goal: `GOAL.md`; backlog v1 focus:
E00, H01, H09, H04, H11. This span produces bounded pilot evidence and a v2 handoff.
It does not promise a 0.8 score or a completed full-data training comparison.

The original goal start is **2026-11-03 10:06:28 UTC**; deadline is
**2026-11-05 10:06:28 UTC**. Resuming the agent preserves both. The total cap is
**43,200 elapsed two-T4 session-seconds**, including setup, probes, failures,
evaluation and retries. Initial Kaggle quota: used **40,562.245 s**, reserved
**0 s**, allowed **108,000 s**, remaining **67,437.755 s**; refresh November 7.
Paid scaling is disabled. Fifteen recent project kernels were queried through
the authenticated API and all were COMPLETE; the old controller PID was absent.

The previous prompted-judge submission 57909397 is now verified COMPLETE with
official loss **0.98653**; the supervised classifier remains **0.97025**. This
refresh changes no frozen predictions and does not establish a useful teacher.

Kaggle staff state that one hour on two T4s consumes one quota hour:
[official T4 announcement](https://www.kaggle.com/discussions/product-feedback/361104).
The current general [notebook documentation](https://www.kaggle.com/docs/notebooks)
describes twelve-hour CPU/GPU sessions; the competition-specific page is dynamic
and was not fully readable by the web fetch. Preserve the project's conservative
nine-hour competition guard. Our individual jobs are capped at three and seven
hours, below either bound. Verify observed quota changes after each job.

Admit at most **3 h** for the reward pilot and **7 h** for the joint training /
feature job, leaving **2 h** inside the goal cap for measured, justified recovery.
Before launch, check current quota and all recorded handles. Never duplicate a
live job. A failed push with uncertain acceptance must be reconciled by exact
kernel/version before retrying. No automatic retry of a failed GPU execution.
Persist launch intents, returned versions, status and actual quota use under
`~/database/llm_classification_finetuning/runs/span_001`.

The machine-readable plan and inventories are in `results/span_001/`; private row
IDs and detailed artifacts are in the heavy run directory. Configuration:
[`configs/span_001.json`](../configs/span_001.json). Preparation program:
[`scripts/prepare_span_001.py`](../scripts/prepare_span_001.py).

1. **Fixed data and comparators.** Preserve attempt-002 split IDs, SHA-256
   `066f7ab82a3f14464ce0ef67bfd8b8b34ed190679bfea5c1fea80abc3aa4ecd1`.
   Counts are 45,985 train, 5,746 development and 5,746 confirmation. Sample once
   with seed **20261118**: 1,024 development rows, 4,096 training rows for frozen
   features/head fitting and 2,048 training rows for short adaptation pilots.
   The samples are explicit IDs with a recorded digest. Prompt groups never cross
   train/development; IDs, model identities and target-derived EDA fields are excluded
   from model inputs. The two training samples may overlap; their purposes are
   recorded, and neither contains development groups.

   The frozen baseline is Qwen3-4B attempt 002, kernel
   `arseniytrushin/llm-preference-qwen3-4b-train-002/1`, adapter digest
   `8afc09d728698c7bcd02e47f9d427166b4b07b3644c848052cc66c9b4f08e39d`,
   temperature **0.7220597450036835** and aligned swap averaging. Its cached
   predictions cover all development rows and were checked against source IDs /
   labels. On this fixed pilot, frozen deployed loss is **0.9763498352** and
   accuracy **52.2461%**. Verify the adapter digest before loading it. Inventory
   includes source, selection, sample, split and prediction digests.

   Baseline adapters were initially missing from the local downloaded subset;
   they have now been downloaded and their digest matches. Exact optimizer
   continuation is not required: this span tests explicitly labeled fresh-optimizer
   adaptation on top of frozen effective baseline weights, not H03 continuation.

2. **Selection and uncertainty.** Development and confirmation were inspected in
   earlier work. All new development findings are exploratory. Do not evaluate
   confirmation for this span. Keep the fixed final development IDs out of every
   training/head-selection operation. Choose the low-capacity head's regularization
   from **0.01, 0.1, 1.0** using three prompt-grouped inner folds of the 4,096
   feature-fit training rows. Fit scalers/PCA within those folds, then refit on
   those training rows. A backbone trained previously on the full original train
   partition is not an OOF model for these inner rows; report this limitation.

   Report raw probabilities and five-fold group-cross-fitted temperature calibration
   on development for new candidates and matched controls, with temperature bounds
   **[0.5, 3.0]**. These bounds were made explicit before any new model probabilities
   were obtained; no quality result motivated the choice. Calibration fits exclude
   each reported fold; do not report all-row fitted calibration as held-out quality.
   Keep the original deployed baseline, whose temperature was fitted previously on
   development, as a separately labeled reference. Use paired prompt-group bootstrap
   with **2,000** replicates for overall loss reduction and order-disagreement
   changes. Do not count rows within one repeated group as independent.

   Predeclared useful pilot gain is **0.005 absolute log loss**. A favorable point
   estimate with an interval still allowing useful gain can justify a v2 follow-up;
   an inconclusive small pilot is not proof of no effect. Report all four arms and
   head controls, rather than choosing one noisy winner. Confidence intervals are
   descriptive across several hypotheses, without a multiple-testing guarantee.
   Adoption requires a larger prespecified comparison with positive paired evidence,
   deployment feasibility and honest disclosure of inspected data.

3. **H01: frozen preference reward backbone.** Candidate is
   [Skywork-Reward-V2-Qwen3-4B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B),
   revision `fd958fef475f323f4e6b195930e3dd918485c668`. Public metadata identifies
   Apache-2.0, no access gate, Qwen3ForSequenceClassification and one reward output.
   Its model card describes preference training and recommends conversation
   scoring without system prompts. Known benchmark overlap is not established;
   unknown pretraining overlap remains a limitation, not a clean-data claim.

   Use the official tokenizer/template, FP16 on each T4 and data-parallel frozen
   inference. First verify finite outputs, padding/template behavior, memory and
   throughput. Compare each answer independently. Retain the question/answer
   content selected by the baseline's token-budget policy, then wrap that content
   for reward scoring. Record retention, template overhead and any additional
   clipping. This tests an alternative scoring recipe, not the isolated causal
   effect of preference training: architecture use, precision and template differ.

   Score the fixed 4,096 feature-fit and 1,024 development rows. Fit a symmetric
   three-class probability mapping on training reward differences and symmetric
   score features; do not use a scalar reward as a three-class probability.
   Choose mapping capacity/regularization only within training groups. Evaluate
   loss, A/B/tie behavior, exact swap symmetry and projected offline inference cost.
   No full-test prediction or new competition submission is part of this pilot.

4. **H04/H09: controlled short training.** Keep the existing frozen adapter active
   and add a fresh trainable adapter/head, avoiding a quantized merge or changes to
   the frozen effective base. Test this composition on a tiny local model before
   GPU use; zero-step outputs must agree with the frozen baseline. Keep original
   adapter parameters frozen and its dropout inactive. Train a fresh optimizer in
   every arm, seed 20261118, LR **2e-5**, four warmup updates, cosine schedule,
   weight decay **0.01**, dropout **0.05**, and **64** optimizer updates. Stop a
   training arm at **2,700 s**; record truncated exposure rather than concealing it.

   Four arms: random-swap CE / attention; random-swap CE / attention+MLP;
   paired-view CE / attention; paired-view CE plus **0.1 symmetric KL** / attention.
   Attention-only fresh rank is 16. Broad coverage uses attention rank 5 and MLP
   rank 6, alpha twice each rank. For Qwen3-4B this is approximately a 1.25% lower
   LoRA parameter count; measure actual counts and require a difference below 2%.
   The classification head has identical initial weights and capacity in each arm.

   On two ranks with accumulation 8, random arms use two canonical rows per device
   microbatch; paired arms use one row with both views. Thus each optimizer update
   processes 32 views in either regime. Paired arms see half as many canonical-row
   exposures; H09's clean primary contrast is paired CE versus paired CE+KL, which
   matches both exposures and updates. Random/paired comparisons additionally
   describe this exposure tradeoff. Count actual retained tokens and elapsed time.
   H04's primary contrast is the two random arms at matched data/exposure and near
   matched trainable count. Fixed final updates are evaluated; no dev-selected
   checkpoint search. Report raw and swap-averaged predictions for every arm.

5. **H11: learned symmetric tie head on frozen comparison features.** Extract
   baseline pooled answer representations in both orientations on the same 4,096
   training and 1,024 development rows. Align A/B before averaging. Build signed
   features for direction and symmetric features for ties; fit a training-only
   16-component representation reduction. Compare a factorized learned tie/winner
   head against a coupled three-class head using the same features/capacity, plus
   a probability-only calibration control. Enforce/check swap equivariance.

   This is a frozen-feature architecture pilot, not an end-to-end tie-head fine-tune.
   A benefit of extra features does not establish that factorization helps: report
   both contrasts. Assess overall loss, tie discrimination/recall/precision and
   calibration; increased tie recall alone is insufficient. Queue supported
   end-to-end work for v2 rather than claiming it was performed.

Stop immediately on nonfinite optimization, label/view misalignment, invalid
probabilities or broken provenance. Profile before full scoring, keep resumable
artifacts where feasible, and respect remaining per-job and span ceilings. Record
each failure and any justified recovery explicitly. At the end, write the report,
execute notebook 12, queue grounded discoveries, resolve focus items with evidence,
seal v1 and produce v2 under the project policy.

Implementation event, before H01 quality measurements: the first reward job reached
ERROR at its first inference batch because the installed chat-template API returned
a `BatchEncoding` container. It consumed **170.830 quota seconds**. The corrected
code explicitly requests/extracts flat token IDs. A local check with the real pinned
tokenizer processed all **5,120** planned rows (maximum packed length **988**) and
confirmed the official template exists. A regression test covers the container
change. Recovery uses a separately recorded job/version, preserves the failed run,
and counts its quota within the original cap; sample/content policy is unchanged.
