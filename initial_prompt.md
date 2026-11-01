Participate in the Kaggle competition:
https://www.kaggle.com/competitions/llm-classification-finetuning/data

The main project directory is:
`/home/tarstars/prj/llm_classification_finetuning`

It is connected to GitHub. Carry out the experiments, document the findings, and commit and push important results.

First inspect the official problem statement, dataset, evaluation metric, and submission requirements. Build a reproducible NLP study comparing the following approaches.

Text vectorization:
1. TF-IDF.
2. An RNN trained on the competition texts. Tune and validate the recurrent model to obtain good quality, then use its learned representations to vectorize the data.
3. A contemporary pretrained embedding backend. You may use Eliza; consult earlier sessions for integration examples. I explicitly authorize sending the full competition dataset’s prompts and responses to Eliza for embedding extraction.

For each vectorization approach, compare:
- Gradient-boosted decision trees.
- Logistic regression.
- A neural network.

Include a baseline that always predicts the most frequent training class. Calculate its quality metrics alongside those of the learned models. Report the competition metric and useful classification metrics, including accuracy, balanced accuracy, macro F1, and Brier score. Explain how hard predictions are handled when calculating log loss. Also include a class-frequency probability baseline where useful.

Beyond these approaches, think about the problem independently and propose several additional methods. Clearly distinguish approaches actually tested from proposals for future experiments.

Use consistent validation partitions for meaningful comparisons, prevent data leakage, and document preprocessing, hyperparameters, model selection, limitations, and reproducibility. Distinguish local validation results from actual Kaggle leaderboard scores. Prepare a competition-compatible submission notebook and record its execution and submission status.

Environment and storage:
- You may use `uv` to install packages in this project directory.
- You may install the Kaggle package and any ML packages you need.
- Produce Jupyter notebooks containing the experiments, results, plots, and conclusions.
- Store notebooks in the project and create a link to their directory under `~/notebooks`.
- Keep datasets, model weights, cached embeddings, predictions, and other heavy intermediate files under `~/database`, preferably in a project-specific subdirectory.
- Use GitHub for programs, notebooks, small result summaries, and Markdown project memory. Do not commit heavy datasets or model artifacts.
- Keep credentials out of source code, notebook outputs, logs, and commits.

Filesystem constraint:
Never recursively walk the FUSE/network checkouts, including:
- `~/prj/arcadia`
- `~/prj/arc00` through `arc02`
- similarly named checkouts such as `~/prj/arc_00`, `arc_01`, etc.
- `/media/tarstars/medium_data/database/troll_farm`

Do not run broad recursive searches or directory walks over those mounts, `~/prj`, or `$HOME`. Commands such as `find`, `grep -r`, `rg --files`, `ls -R`, and `du` can get stuck there for hours and prevent the laptop from suspending. Restrict searches to explicitly named local project directories.

Carry the work through to concrete, reviewable results, keeping project memory and GitHub updated at important milestones.
