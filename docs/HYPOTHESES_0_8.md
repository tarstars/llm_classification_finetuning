# Forty hypotheses toward 0.8 log loss

The revised list is the [ranked research backlog](../BACKLOG.md), backed by
[structured records](../backlog/hypotheses.json) and the project's
[backlog CLI](../scripts/backlog.py). It retains **H01–H40** and adds one evaluation
prerequisite, E00. Every hypothesis has a gain range, engineering and compute budget,
risk/mitigation, first test, acceptance/stop rules, dependencies and revision note.
Git history preserves the original register at commit `ee9cf20`.

The estimates are subjective planning ranges, not measured results or guarantees.
The largest corrections are that the final causal classification token already
sees both answers (H07), probability factorization alone cannot improve tie
recognition (H11), and conflicting repeated comparisons cover too few rows to
explain the main gap (H12). More training (H03), packing changes (H24/H25) and a
stronger general judge are plausible experiments, not established improvements.

The first model-pilot wave is **H01, H09, H04, H11**, after E00. Low-cost CPU audits
and calibration can accompany that work. No new model experiment was run while
creating the backlog.

```bash
uv run python scripts/backlog.py list --priority P1
uv run python scripts/backlog.py show H09
uv run python scripts/backlog.py check
```

Our best recorded official score is **0.97025**; reaching 0.8 would require a
reduction of **0.17025**. Overlapping experiment gains cannot be added. Neither an
irreducible loss floor of 1.0 nor target leakage as the explanation for all better
scores has been established by our [target audit](../TARGET_AUDIT.md).

Historical context needs care. The winner of the **different, related 2024 LMSYS
competition** reported 0.876 public and 0.96898 private loss. Kaggle staff described
rerunning submissions on a new private test set. These figures do not establish
an achievable 0.8 on our split or the current competition:
[author's write-up](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629),
[Kaggle staff's closing-mechanics post](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/524376).

Technical precedents include [Skywork reward models](https://github.com/SkyworkAI/Skywork-Reward),
the [Reward V2 paper](https://arxiv.org/abs/2507.01352), and the
[second-place LMSYS write-up](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685).
They are distinct from local evidence in our [EDA](../EDA_REPORT.md),
[blind comparison](../ATTEMPT_004_RESULTS.md) and [target audit](../TARGET_AUDIT.md).
The backlog specifies the evaluation safeguards and limits for each proposed use.
