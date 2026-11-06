# E14: exact model-instance admission

The shared admission helper now matches owner, model, framework, variation and
version together, requires exactly one fully identified match, rejects conflicting
identity URLs and fails closed on potentially truncated listings. It no longer
accepts an unrelated model solely because its version number matches.

Three regression tests cover unrelated same-version entries, missing fields,
duplicates, conflicting URLs and unpinned sources; all pass. A read-only live
check in results/span_009/E14_live_admission.json resolves the intended
qwen-lm/qwen-3/transformers/8b/1 to instance301517/version363139. No GPU execution
or historical notebook modification. Future generated bundles freeze the changed
helper hash; historical accepted sources remain in their exact versions and Git.
