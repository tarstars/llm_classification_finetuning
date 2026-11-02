# Attempt 003: prompted general-model judge

The user requested an actual competition submission using a prompted strong model,
without fine-tuning or blending with our Qwen3-4B classifier. This is an inference-only
experiment. Model weights remain unchanged. Small probability calibration is reported
separately from raw zero-shot quality.

Primary candidate: official Gemma 4 31B-IT, QAT Q4_0 GGUF, Kaggle model version 2.
Runtime fallback: official Gemma 4 26B-A4B-IT in the same format/version, if the dense
model cannot meet the competition's nine-hour inference limit. The project has two
T4 GPUs. We must verify fit and throughput, not infer them solely from checkpoint size.

Use pinned llama.cpp revision `59695669f1924c81f421f2044c690ed6373bec79`, compiled for
CUDA architecture 75. A setup-only private kernel may download source. The compiled
binary/shared libraries become an attached artifact for internet-disabled evaluation
and submission notebooks. Model files are explicitly versioned Kaggle inputs.

The initial setup probe (kernel version 1, archived job `runtime_probe_failed_1`)
failed during CMake generation because Kaggle exposed `libcuda.so.1` without the
unversioned developer library expected by FindCUDAToolkit. The corrected probe
supplies the actual installed library found through `ldconfig` and named driver
directories. This failure produced no judge quality measurements.

The initial prompt is fixed in `configs/attempt_003_judge.json`. It asks for a human
preference judgment using correctness, instructions, relevance and groundedness, and
warns against position/verbosity shortcuts. It contains no demonstrations or selected
poster answers. Use Gemma's official non-thinking chat template, including its empty
thought-channel prefix. Score three single-token labels A/B/C with a restricted
grammar and unfiltered, temperature-one label probabilities. Do not treat a generated
JSON confidence estimate as a model likelihood. C maps to the competition's tie class.

Select a uniform, fixed 512-row development sample using seed 20261113, without
inspecting errors. Its 32-row prefix is a throughput profile. Labels never appear in
judge inputs. Development has been used previously; these results are exploratory,
not fresh confirmation. The inspected local confirmation fold is not used for new
selection. Freeze the selected model, packing, order averaging and calibration before
hidden-test inference.

Inputs preserve all three fields with explicit separators and conversation boundaries.
Use a 2,048-token total budget, a 384-token question cap, balanced answer allocation
with unused-budget redistribution, and head/tail truncation. Record retention and exact
packed lengths. Compare original-order and aligned swap-average predictions on the
same rows. A hardware profile determines whether both views fit the runtime budget.

Measure multiclass log loss, accuracy, per-class behavior, probability calibration,
position sensitivity, runtime, GPU memory and truncation. Compare our frozen Qwen3-4B
probabilities on exactly the same IDs. Report raw judge quality and group-cross-fitted
temperature/tie-intercept calibration; fit deployment calibration on the sampled dev
rows after evaluating the calibration protocol. Calibration is not language-model
fine-tuning. No claim that the sample score predicts 0.8 without uncertainty.

The initial runtime projection assumes 25,000 hidden rows and must be labeled as an
assumption. Allow a margin below nine hours. The submitted notebook reads the actual
test rows and produces a valid `submission.csv` for every one, using the frozen judge
configuration. Do not silently substitute another classifier or fill unfinished rows
with priors. Record and investigate failures before relaunching jobs.

Programs, executed result notebooks, aggregate evidence and project memory belong in
Git. Runtime binaries, model files, raw prompts, per-row predictions and logs belong
under `~/database/llm_classification_finetuning/runs/attempt_003` or private Kaggle
artifacts. Commit and push important milestones and the final result.

The resumable controller is `scripts/run_judge_pipeline.py --publish-results`.
Its state is `~/database/llm_classification_finetuning/runs/attempt_003/pipeline_state.json`.
It pins saved job versions, evaluates development, freezes a runtime-feasible judge,
validates offline prediction, submits once, monitors the official score, and refreshes
the report/notebook. A failed expensive job stops the controller for diagnosis.

Sources: [competition requirements](https://www.kaggle.com/c/llm-classification-finetuning),
[Gemma model card](https://huggingface.co/google/gemma-4-31B-it),
[llama.cpp server API](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
