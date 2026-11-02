# Browse the frozen-model errors

Open **http://127.0.0.1:8765/** while the local server is running. The standalone
file is `~/database/llm_classification_finetuning/eda/attempt_002/poster/site/index.html`.
It can also be opened directly in a browser. The HTML embeds its data and needs no
external scripts or services. Browser notes are tied to the browser origin; export
them before switching between the file and server or changing browsers.

The [executed result notebook](../notebooks/07_error_poster.ipynb) preserves aggregate
metrics, category comparisons and high-loss example IDs. The [summary](../results/error_poster_002/summary.json)
records coverage, provenance and the generated HTML checksum.

The gallery covers a fixed uniform sample of **2,048 training-partition rows**,
all **5,746 development rows**, and all **5,746 local confirmation rows**. The
training sample was chosen without inspecting losses, from 45,985 rows, seed
20261111. Highest-loss training cases are highest within that sample. Only 0.61
epochs were completed and per-row exposure was not logged, so training partition
membership is not proof that a particular example was seen by an optimizer step.

| Scored cohort | Rows | Log loss | Accuracy |
| --- | ---: | ---: | ---: |
| Training sample | 2,048 | 0.96566 | 55.27% |
| Development | 5,746 | 0.96883 | 52.73% |
| Local confirmation | 5,746 | 0.96236 | 53.06% |

There is no large training-loss advantage in this sample. The small differences
and partial training do not by themselves diagnose underfitting or overfitting.
Inference took 1,267.5 seconds on two Kaggle T4s; the full job, including setup and
artifact finalization, took about 26 minutes. The official score remains 0.97025.

“Held-out test (local)” means the confirmation fold. Kaggle hidden examples and
labels are unavailable. Confirmation had already been evaluated before this task;
its text is now inspected too, so it cannot be called untouched in future work.

The selected model, adapter and temperature are frozen. Ranking uses the probability
after both answer orders are aligned, averaged and calibrated. Training and
confirmation files are already calibrated; the saved development probabilities
receive the temperature once. There is no new training or competition submission.

Start with the suggested review links or choose a cohort and category. Each card
shows the observed vote, predicted choice, all three probabilities, individual
negative log likelihood, cohort loss rank, full question and answers. Amber spans
mark content omitted by the original-order token budget. “Exact input” reveals
decoded packed sequences in both orders. Multi-turn examples also have an aligned
turn view. Where raw probabilities are retained, the order diagnostic shows the
two aligned views. Longer-context predictions exist only for the development pilot.

Twenty overlapping categories include tie errors, similar answers, truncation,
multi-turn conversations, code/math/creative/transformation wording, refusals,
length imbalance, identical answers and order/context diagnostics. Task labels are
heuristics. The gallery is a deduplicated union of the top 12 cases per category
and cohort. Search and filters operate on this selected gallery; category metrics
use all scored rows, and explicitly state their population. Missing diagnostic
coverage must not be read as evidence that an example is order-invariant.

Observed votes can be subjective. Analyst notes separate a visible issue from a
proposed explanation, preserve uncertainty and suggest a controlled follow-up.
For example, an untruncated rewrite request without source text exposes a relevance
failure, whereas preference between two plausible overviews may reflect style.
The missing-input pattern also appears in a training example that invents a
combined program before either source script is supplied. Training notes also
cover a refusal preference and a high-confidence miss of a tie vote.
Neither selected examples nor heuristic tags establish the prevalence of a cause.
The [EDA report](../EDA_REPORT.md) provides the broader statistical evidence.

Write your own hypothesis and notes at the bottom of each card. Notes persist in
local browser storage and can be exported as JSON. Copy-link buttons support direct
links to examples; the print button prints the currently displayed cards.

To rebuild using existing heavy artifacts and start the server:

```bash
uv sync --extra train --extra eda --extra qa
uv run --extra train python scripts/build_error_poster.py
uv run python scripts/serve_error_poster.py
```

The local server binds only to `127.0.0.1`. Code, aggregate evidence, paraphrased
review notes and notebooks belong in Git. Full-text HTML, raw predictions, browser
screenshots and exported private notes belong under `~/database`.

The inference source is [poster_infer.py](../src/arena/poster_infer.py); its private
Kaggle package is in [attempt_002_poster_train](../kaggle/attempt_002_poster_train/).
Rebuilding that package does not run inference: pushing it would start another job
and spend GPU quota. The completed job's predictions are reusable.

Verification:

```bash
uv run python -m unittest discover -s tests -v
uv run --extra qa python scripts/check_error_poster.py
uv run python scripts/build_error_poster_notebook.py
```

The browser check covers loss ordering, filters, calibration consistency, omitted
text, pagination, deep links, saved notes, export and mobile overflow. Its screenshots
stay in the private artifact directory.
