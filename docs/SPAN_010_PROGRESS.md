# V10: first pilot completed; packing-only variant prepared

The user goal remains rank19or better. Official best stays0.83053/rank30; no v10
submission yet. First public version arseniytrushin/llm-preference-span010-order-completion/1
is COMPLETE, charged2802.478seconds; all50retrieved files pass hashes. Exact256row
prediction controls have maximum difference0 for both models. Source, IDs,
checkpoint loading, swaps, route/mixture replay, CSV and both T4s pass.

Conservative runtime32998.845seconds FAILS the unchanged32400second limit. This
recipe is not submitted. Uniform training-overlapping loss improves by.001232,
with group95%[-.000199,.002712]; it is not independent quality evidence.
See SPAN_010_ORDER_PILOT_RESULTS.md and H57_order_pilot_qualification.json.

The second prospective recipe changes only packed microbatch budget8192->12288,
with byte-equivalence proof for everything else. Three packed CPU orchestration
fixtures, two goal/leaderboard tests and actual serialized worker/startup
rehearsals pass. All prediction controls,20%margin,600sreserve, nine-hour hidden
limit and original21600second phase cap remain unchanged. Its public job cap is
5400seconds. Freeze/commit before admission and preserve both exact receipts.

Notebook22 is executed with four code cells, two figures and the first measured
runtimeFAIL. Independent audit and submission programs now take an explicit
--role, so a failed qualification cannot be mistaken for a different recipe.
Never rebuild a source after its push intent, or re-submit after an ambiguous
receipt. Only a fresh complete official rank<=19completes the goal.

Second pilot is now RUNNING at exact returned source
arseniytrushin/llm-preference-span010-packed-order-completion/1, commit25d71c9.
Kaggle derived this slug from the title instead of the requested order-packed ID;
the controller follows the actual response and no second push was made.
Observer handle64908; recover/observe this exact version if observation expires.
