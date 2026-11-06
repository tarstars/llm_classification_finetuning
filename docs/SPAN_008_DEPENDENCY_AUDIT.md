# Model-list responses need full instance identity checks

The live call asking Kaggle for Qwen3's transformers/8b model versions returned
45entries across multiple variations and frameworks. Version number1 alone
therefore does not establish that the requested8B transformer checkpoint exists.
The shared controller currently checks the version number only. A regression
test using these response shapes should harden that helper in the next span;
accepted historical bundles remain unchanged during the current jobs.

For this span's conditional size pilot, a separate exact check requires all of
variationSlug=8b, framework=MODEL_FRAMEWORK_TRANSFORMERS andversionNumber=1.
It finds exactly one matching record, modelInstanceId301517/versionid363139.
The model is attached explicitly and the worker additionally verifies its
tokenizer vocabulary against the4B reference and records actual model-config and
packed-token hashes. Model-list response and exact filtered evidence are in
[source_readiness.json](../results/span_008/source_readiness.json) and
[H15_audit.json](../results/span_008/H15_audit.json).

This is a source-availability guard, not proof of memory fit or model quality.
