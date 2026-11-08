# E18: native host batches preserved under bounded CPU prefetch

The frozen method `a0c78a6` passed its one native CPU proof: **31.847 seconds**
within the 300-second cap, using the pinned Transformers4.42.3/tokenizers0.19.1
environment. Unified controller59216 terminated with exit0. Do not rerun or
overwrite its proof or preservation files.

All15actual old cohorts (Gemma, Llama-swapped, Gemma-reversed; warm-up and
control/uniform/long/visible test, including routed subsets) preserve exact
host tensors, token streams, row order, individual positions and **original8192
microbatch boundaries**. The producer owns native dataset/processor iteration,
uses FIFO queue2 and hands over only host tensors. Consumer hashing verifies the
same native collator output. No model weights were loaded and no labels read;
placeholder labels only satisfy the author dataset interface.

Local synchronous preparation took 2.468seconds for Gemma's512uniform rows,
2.291 for Llama and0.273 for83routed reversed-Gemma rows. These are laptop
preparation measurements during proof, not Kaggle savings or a deployment
forecast. The proof consumer hashes tensors instead of running a model.

Seven focused FIFO/failure/teardown tests pass. Six profile integration tests
also pass, including raw failure preservation, changed microbatch rejection,
unjoined-producer rejection, early stop before later model loading, route/CSV
replay and retained worse repeat timings. GPU inference, numerical equivalence,
memory and full runtime remain **unmeasured** for this execution change.

The GPU engine source is prepared but not admitted. It retains original8192
groups and author forward, records preparation/queue/forward-operation wall
durations, and checks complete host tensor hashes and boundaries against the
CPU proof. End-to-end wall time alone enters projection, without summing
overlapping preparation durations. Notebook source, actual final-cell/import/
guard/watchdog/Jupyter rehearsals, current quota/admission and an independent
terminal auditor must still be completed before any promotion or H73replay.

Heavy exact batch references: `~/database/llm_classification_finetuning/runs/`
`span_028/cpu_batch_references.json`. Aggregate proof and source/output hashes:
[cpu_audit.json](../results/span_028/cpu_audit.json). Preservation captured all
110historical snapshot files and unchanged user notebooks07/12 and their link.
H73 remains deferred and its60/40probabilities have not been computed.
