# Forty hypotheses toward 0.8 log loss

These are proposed experiments, not measured gains or assurances that 0.8 is attainable.
Our best recorded official score is 0.97025. Reaching 0.8 would require a reduction
of 0.17025, about 17.5%; individual experiment gains cannot be assumed to add.
Hypothesis identifiers are stable so future results can be linked to them.

Evidence informing the priorities:
- The current classifier saw only 0.61 training epochs, but its late development
  losses plateaued. More epochs remain uncertain.
- Answer order changed 38.1% of raw decisions in the LoRA pilot. Random swap
  augmentation and inference swap averaging are already present; H09 adds an
  explicit paired consistency objective.
- The classifier loses content on 20% of development rows. A longer-context
  inference pilot showed only a small uncertain gain, so H23 is not an established cure.
- In 300 sealed blind judgments, the current assistant reached 56.3% accuracy
  against Gemma's 53.3%, an uncertain difference. A larger general judge is not
  assumed to solve the problem.
- Tie errors are substantial. Length and refusals carry some legitimate preference
  signal. ID/order diagnostics do not. Target-derived EDA metadata is excluded.
- Existing tie-intercept and structural-blend gains were small and uncertain,
  making H36 and H40 polishing experiments rather than likely 0.8 breakthroughs.

Historical context needs care. The winner of the **different, related 2024 LMSYS
competition** reported 0.876 public and 0.96898 private loss. Kaggle staff said
the private leaderboard would be produced by rerunning submissions on a new test
set. Those scores do not establish an achievable 0.8 on our split or the current
competition, and do not prove a 1.0 irreducible floor. Sources:
[author's first-place write-up](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629),
[Kaggle staff's closing-mechanics post](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524376).
The author's recipe provides precedent for preference fine-tuning, large-model
distillation and a compact quantized submission, rather than an isolated guarantee
for any component.

| ID | Hypothesis | First falsifiable test | Effort | Priority |
| --- | --- | --- | --- | --- |
| H01 **Preference-trained backbone** | A model already trained to rank human preferences transfers better than a general chat model. | Benchmark a public Skywork or ArmoRM checkpoint with a train-fitted three-class probability head. | medium | high |
| H02 **Larger supervised classifier** | An 8–9B classifier learns comparison details missed by our 4B model. | Fine-tune a larger backbone on exactly the same grouped split and compare at matched token exposure. | high | high |
| H03 **Complete training exposure** | Stopping at 0.61 epochs left useful supervised signal unlearned. | Continue from a complete optimizer checkpoint through one epoch; try a second only if development improves. | high | medium |
| H04 **Broader adapter coverage** | Attention-only LoRA cannot sufficiently change preference representations. | Add adapters to feed-forward projections while keeping the split, rank budget and training tokens controlled. | high | high |
| H05 **Better optimization** | The learning rate or batch regime caused an early optimization plateau. | Compare a small predeclared learning-rate/schedule grid on training-only pilots, then one full run. | high | medium |
| H06 **Response-aware pooling** | The final-token representation loses evidence distributed through both answers. | Pool token representations separately for A and B and learn their comparison. | high | medium |
| H07 **Bidirectional comparison attention** | A causal mask limits how earlier response tokens use evidence from the later response. | Test bidirectional attention during classification fine-tuning against the causal baseline. | high | medium |
| H08 **Less quantization error** | Four-bit quantization damages subtle preference distinctions. | Compare the same checkpoint at four-bit, eight-bit and higher precision on a fixed inference sample. | medium | low |
| H09 **Explicit swap consistency** | Random swapping alone does not remove answer-position sensitivity. | Train paired A/B and B/A views with a consistency penalty after aligning their class probabilities. | high | high |
| H10 **Shared response utilities** | Independent position-specific representations encourage spurious A/B preferences. | Score each question-response pair with shared weights and predict the winner from the utility difference. | high | high |
| H11 **Separate symmetric tie model** | Tie recognition needs different evidence from choosing A versus B. | Predict tie probability from symmetric pair features, then split the remaining probability between A and B. | medium | high |
| H12 **Soft targets for repeated votes** | Conflicting votes reflect uncertainty that hard one-hot labels erase. | Within training folds, aggregate repeated equivalent comparisons into empirical class distributions. | low | medium |
| H13 **Noise-robust training loss** | Some incorrect or idiosyncratic votes disproportionately distort training. | Compare ordinary cross-entropy with bounded or bootstrapped losses; keep the original labels for evaluation. | medium | medium |
| H14 **Criterion-level supervision** | A single winner label provides too little guidance about why an answer wins. | Add auxiliary prediction of factual consistency, instruction compliance and relevance using reliable training annotations. | high | medium |
| H15 **Teacher distillation** | A capable trained preference teacher supplies richer supervision than a single vote. | Generate teacher probabilities without teacher exposure to the evaluation groups, then distill a Kaggle-sized student. | very high | high |
| H16 **Refusal appropriateness** | The model confuses helpful restraint with unnecessary refusal. | Train a feature or auxiliary head that evaluates the request together with the refusal, instead of always penalizing refusals. | medium | medium |
| H17 **Additional public preference data** | Our 46K training comparisons do not cover enough preference situations. | Mix provenance-checked preference datasets into training and remove overlaps with known evaluation groups. | high | high |
| H18 **Controlled instruction counterexamples** | The model has insufficient experience with small but decisive instruction violations. | Create verified training pairs differing only in word count, required format, omitted items or similar constraints. | medium | medium |
| H19 **Verified factual counterexamples** | Training contains too little supervision for distinguishing fluent facts from fluent inventions. | Create pairs by modifying known facts in otherwise matched answers, with independent verification. | high | medium |
| H20 **Hard-example mining** | Broadly uncertain comparison regions need more attention than already easy cases. | Use out-of-fold training errors to sample difficult examples, without mining evaluation labels. | medium | medium |
| H21 **Balanced domain exposure** | Frequent conversational topics crowd out code, math and less frequent languages. | Compare domain-balanced training sampling with the original distribution and evaluate overall loss as well as domain slices. | medium | medium |
| H22 **Repeated-prompt reweighting** | Large repeated-prompt groups receive too much training influence. | Cap or inversely weight each training prompt group's total contribution. | low | medium |
| H23 **Longer-context training** | Truncation removes the evidence that distinguishes the answers. | Train and evaluate at 2,048–4,096 tokens, measuring paired gains and submission throughput. | high | medium |
| H24 **Adaptive question budget** | The fixed 256-token question cap removes essential user constraints. | Allocate question tokens according to task and turn structure rather than one fixed cap. | medium | high |
| H25 **Turn-aligned serialization** | Grouping all user turns before all answers obscures conversational dependencies. | Interleave each user turn with the corresponding candidate responses and compare against the current format. | medium | high |
| H26 **Hierarchical long-text comparison** | A single truncated pass cannot compare long answers reliably. | Encode chunks, aggregate their evidence, and train a final comparison head. | high | medium |
| H27 **Salient extractive retention** | Head-and-tail truncation often deletes decisive material in the middle. | Retain instruction-bearing spans, code definitions, conclusions and conflicting claims instead of only the ends. | medium | medium |
| H28 **Task-aware compression** | A short faithful representation can preserve more useful information than raw truncation. | Compress long inputs into task-relevant evidence and train the classifier on those summaries; audit compression errors. | high | medium |
| H29 **Human-preference rubric** | Our judge overweights ideal factual correctness relative to observed user preference. | Compare a voter-oriented prompt with the current impartial-quality prompt on a fixed development pilot. | low | medium |
| H30 **Retrieved training demonstrations** | A small number of similar labeled training examples clarify ambiguous preference boundaries. | Retrieve demonstrations only from training groups, then judge the new pair without its label. | medium | medium |
| H31 **Brief critique before verdict** | Immediate one-token judgment skips useful comparisons. | Allow a tightly bounded assessment before requesting the class; measure the added inference time. | medium | medium |
| H32 **Independent reference answer** | Judging candidates directly lets fluent errors anchor the evaluator. | First derive a short reference answer without seeing either candidate, then compare each candidate against it. | high | medium |
| H33 **Offline factual retrieval** | Missing factual knowledge prevents correct evaluation of biographies and niche questions. | Attach a trusted offline reference corpus and retrieve evidence without retrieving competition votes. | high | medium |
| H34 **Code execution evidence** | A judge can prefer plausible-looking programs that do not run. | For applicable tasks, obtain resource-limited execution or static-check results and add them as features. | high | medium |
| H35 **Symbolic math verification** | The model misses elementary algebra, arithmetic and constraint errors. | Use calculators or symbolic checks to score applicable answer claims and feed the results to the preference model. | medium | medium |
| H36 **Structural residual correction** | Length, similarity and formatting explain errors not captured by the neural model. | Fit a small residual classifier on out-of-fold neural probabilities plus text-derived structure features. | low | low |
| H37 **Diverse model ensemble** | Different model families make complementary errors. | Average or stack independently trained preference models using out-of-fold predictions and frozen weights. | high | high |
| H38 **Domain-specific experts** | One classifier cannot serve coding, factual QA, creative writing and multilingual comparison equally well. | Route examples to separately trained domain experts with a learned soft gate. | very high | medium |
| H39 **Budgeted strong-judge cascade** | Expensive judgment is useful on selected difficult examples rather than every row. | Use a trained gate to reserve the strong judge for uncertain cases, keeping projected full-test runtime below the limit. | medium | medium |
| H40 **Richer probability calibration** | A single temperature misses systematic tie and confidence differences. | Cross-fit vector or low-capacity conditional calibration, then freeze it; compare against the existing calibrated model. | low | low |

Effort is qualitative relative to our current setup, not a measured GPU-hour estimate.
H15 may need teacher compute beyond our two T4s. H07 requires attention-mask and
training-stack support. H23 and H26–H35 need explicit inference profiling. A public
checkpoint name does not establish Kaggle availability, license suitability, lack
of benchmark exposure, or runtime fit; verify those for the selected pilot.

The initial shortlist is **H01, H09, H11, H24, H25**, followed by **H02, H15, H17,
H37** if the compute and data are available. H01 can start as a small frozen-model
comparison with a learned three-class head; H09 targets a failure we directly measured.
This ordering is a judgment about expected information and cost, not a forecast gain.

Evaluation contract:
- Use the existing grouped partitions consistently for screening. Since both
  development and confirmation have been inspected, neither is now called untouched.
  Use nested grouped validation for selection, and a genuinely new evaluation set
  if one becomes available.
- Fit supervision, retrieval demonstrations, sampling weights and learned
  postprocessors on the appropriate training folds only. Audit additional preference
  data and checkpoints for known overlap and provenance.
- Predeclare the primary change, training exposure and budget. Compare on the same
  rows with paired loss intervals, tie/A/B metrics, truncation and runtime.
- Freeze a candidate before a competition submission. Keep the baseline and record
  unsuccessful experiments; do not tune through repeated leaderboard probing.
- H12 uses only training-fold vote aggregates. H14, H18 and H19 use verified or
  synthetic criteria without overwriting evaluation votes. Teacher confidence is
  not automatically human truth.
- IDs, model-name metadata unavailable in test, winner columns, canonical_label
  and other target-derived analysis fields stay out of deployed inputs.
- Preserve a valid normalized three-class probability distribution. Accuracy of
  71.2% is needed for 0.8 only within the user's constant-alpha rule; it is not a
  universal requirement for probabilistic classifiers.

Sources for technical precedents, distinct from our untested hypotheses:
- [Skywork reward-model repository](https://github.com/SkyworkAI/Skywork-Reward)
  and [Skywork Reward V2 paper](https://arxiv.org/abs/2507.01352): preference-trained
  reward backbones and preference-data curation.
- [Second-place LMSYS write-up](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685):
  grouped validation, additional deduplicated preference data and alternate input layouts.
- [Our EDA](../EDA_REPORT.md), [blind comparison](../ATTEMPT_004_RESULTS.md)
  and [target audit](../TARGET_AUDIT.md): local evidence and limits.

All forty entries currently have status **proposed**. No new hypothesis experiment
was run while assembling this register.

