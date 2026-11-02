# A single confidence for a hard judge prediction

On the fixed 512-row, 2,048-token development experiment, the prompted Gemma judge
chooses the correct class on 275 rows (53.7109%). Giving its chosen class a single
probability alpha and each other class `(1 - alpha) / 2` has a best fitted log loss
of **1.01124**, at **alpha = 0.537109375**. Each other class receives 0.2314453125.
The five-fold, prompt-group cross-fitted estimate is **1.01387**.

For accuracy `a`, the mean log loss is

```text
L(alpha) = -a ln(alpha) - (1-a) ln((1-alpha)/2)
alpha_optimal = a
L_optimal = -a ln(a) - (1-a) ln((1-a)/2)
```

The derivative vanishes at `alpha = a`; positive second derivative establishes
the minimum. Thus the optimum depends only on accuracy, not the class frequencies
or which incorrect class was chosen. This is a bound for one global alpha and
equal probabilities for the other two classes, with the hard predictions fixed.

| Hard prediction | Correct / total | Fitted alpha | Best fit on all rows | Cross-fitted log loss |
| --- | ---: | ---: | ---: | ---: |
| Original answer order | 275 / 512 | 0.53711 | 1.01124 | 1.01387 |
| Argmax after averaging both answer orders | 261 / 512 | 0.50977 | 1.03276 | 1.03704 |

The original-order cross-fitted estimate has a descriptive prompt-group bootstrap
95% interval of [0.97746, 1.04961]. Each fold fits alpha using only its training
groups. The same-row optimum is optimistic; these are local development estimates,
not official Kaggle scores. Development has also been used in earlier experiments.

The original raw label probabilities scored 1.42159. Temperature plus a tie
intercept scored 0.98076 under the same five-fold protocol. A constant alpha fixes
much of the raw overconfidence, but discards useful differences in confidence and
tie preference. No submission or ongoing judge inference configuration was changed.

On the above-chance branch (`a >= 1/3`), reaching 0.8 log loss with this rule
requires **71.1986% accuracy** even when alpha is optimal. The present hard
predictions cannot achieve 0.8 by fitting alpha alone.

The hard prediction here is the argmax of saved label probabilities, without new
inference. Changing the prompt or decoding procedure could change its accuracy;
that would require a separate evaluation.

Reproduce with `.venv/bin/python scripts/analyze_fixed_alpha.py`. This reads saved
predictions and group IDs under `~/database`, checks their configuration and labels,
and writes aggregate [results](../results/attempt_003/fixed_alpha_2048.json).
See the executed [notebook](../notebooks/09_fixed_judge_confidence.ipynb).
