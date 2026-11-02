# Attempt 003: prompted general-model judge

The user requested an actual competition submission using a prompted strong model,
without fine-tuning or blending with our Qwen3-4B classifier. This is an inference-only
experiment. Model weights remain unchanged. Small probability calibration is reported
separately from raw zero-shot quality.

The initial primary candidate was official Gemma 4 31B-IT, QAT Q4_0 GGUF, Kaggle
model version 2. Measured throughput ruled it out for the nine-hour limit. The main
candidate is official Gemma 4 26B-A4B-IT in the same format/version. The project has two
T4 GPUs. We must verify fit and throughput, not infer them solely from checkpoint size.

Use official llama.cpp CUDA 12.8 release `b11960`, revision
`8642541b255bfb5dc39cd351a178be09013a55d6`, checked against its published SHA-256.
A setup-only private kernel downloads it and verifies both models on T4s. The binary
and shared libraries become attached artifacts for internet-disabled evaluation
and submission notebooks. Model files are explicitly versioned Kaggle inputs.

The initial setup probe (kernel version 1, archived job `runtime_probe_failed_1`)
failed during CMake generation because Kaggle exposed `libcuda.so.1` without the
unversioned developer library expected by FindCUDAToolkit. The corrected probe
supplies the actual installed library found through `ldconfig` and named driver
directories. This failure produced no judge quality measurements. Version 2 built
successfully in 1,629.9 seconds and loaded both official models. An additional
checksum-verified official CUDA 12.8 binary (`b11960`) also loaded both models, and
is preserved as a faster setup alternative; the current profile uses compiled
runtime version 2. The main evaluation uses the official binary to avoid recompilation.

The first raw smoke-test response exposed that post-sampling top-N vocabulary scores
need not contain every label; the pipeline stopped before development evaluation.
The corrected scorer uses the common-label logit offset and conditional normalization
described below. Profile version 1 was prepared before explicit beginning-token
handling was corrected and is diagnostic only; its quality scores do not enter model
selection. The full development run repeats every sampled row with corrected encoding
and a synthetic preflight before real data.

The initial prompt is fixed in `configs/attempt_003_judge.json`. It asks for a human
preference judgment using correctness, instructions, relevance and groundedness, and
warns against position/verbosity shortcuts. It contains no demonstrations or selected
poster answers. Use Gemma's official non-thinking chat template, including its empty
thought-channel prefix. Score three single-token labels A/B/C with a restricted
grammar and temperature-one label probabilities conditioned on A/B/C. The server's
top-token list can contain non-label tokens despite the grammar, so all three label
logits receive an identical positive offset before retrieving their scores. Conditional
normalization cancels that common offset exactly; a regression test verifies the
identity. Missing label scores or invalid probabilities abort scoring. Do not treat a generated
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
packed lengths. Include the model's required beginning token exactly once, then send
the exact token-ID sequence. The server's formatted chat text can omit this token;
token-ID requests do not auto-insert it. A synthetic arithmetic/order preflight checks
the complete scoring path before real rows are evaluated. Compare original-order and aligned swap-average predictions on the
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

The initial four-slot timing diagnostic projected about 66 hours for 31B and 21 hours
for 26B with both orientations over 25,000 rows. The main evaluation therefore focuses
on 26B, with 16 concurrent slots, 4,096 logical batch tokens, 1,024 physical microbatch
tokens and sufficient HTTP workers. Prompt wording and the 2,048-token input limit
are unchanged. If no configuration meets the runtime margin, repeat the same fixed
512 rows at 1,024 tokens. Preserve the original report separately. This fallback is
triggered by runtime, not by inspecting hidden-test results. Deployment must consume
the resulting frozen configuration rather than the initial defaults.

Both development runs completed. The 1,024-token, single-order projection is
8.28 hours **including** a 20% margin and 600 seconds for loading/finalization.
The initial eight-hour selection guard stopped the controller, despite this fitting
the competition's nine-hour limit. Revise the selection guard to 8.5 hours using
the analyzer's explicit `--runtime-budget-seconds 30600` option. This changes only
the runtime decision, without changing the frozen prompt, packing, model or saved
predictions. Preserve the override in the report and selection; do not claim it
was predeclared. Both-order inference remains infeasible. No hidden-test results
have entered this decision.

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
