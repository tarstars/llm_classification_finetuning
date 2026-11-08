# V15: the earlier BlackPearl checkpoint is unsupported for deployment

H55 stops before GPU. Retain the official H60 score **0.82987**, rank23/312 at
2026-11-08 02:47UTC. The objective remains rank<=19; its current cutoff is0.82915.
No checkpoint quality, gain, uncertainty interval or runtime was measured here.

The bounded source audit refreshed the exact public
[earlier parent](https://www.kaggle.com/code/sayoulala/merge-gemma-model-step1/1).
It is COMPLETE, with ten artifacts; source/config/index hashes exactly match our
v9 observation. It declares a42-layer three-class Gemma2 classifier, a score head
and18,483,433,472weight bytes. Accessibility and the905filename establish no score.

The adapter named by that merge,
`sayoulala/v7-8-gemma-v7-64r128-ddgemma2-16bit-4bit905`, returns403 for metadata.
This is a separate new observation from the historical403 for the final winning
five-fold/DPO dataset, which was not retried. The exact earlier training run,
fold membership, class semantics and loaded state keys cannot be traced.

Pinned [author training code](https://github.com/shyoulala/LMSYS_BlackPearl/blob/05ac9b917a99aac43bfe0dbcb30e6e83a38688be/src/train_ce_model_nice_memory_grad_gemma2_double_16bit.py)
explicitly enables gradients for score parameters and saves them with LoRA keys.
Thus CAUSAL_LM wrapping without modules_to_save does **not** prove a random head.
The family preprocessing maps A/B/tie to0/1/2, but the earlier merge uses
strict=False and retains no incompatibility report. These family observations
do not prove that this particular adapter restored its correctly ordered head.

Native serialization also matters. The final family inference uses `User question`,
`Answer A`, `Answer B`, word caps256/700/700, balanced token truncation and appended
EOS. Its context is2000; published training slices1100tokens before appending EOS
and sets padding to EOS. The earlier merged config instead has pad0. No exact
earlier inference reference resolves the context/pooling difference. Reusing the
current checkpoint's format or choosing a class permutation from labels would
create an unsupported new model. The repository has no top-level license;
an Apache header on one training file does not establish the adapter's full lineage.

[Machine-readable decision](../results/span_015/H55_source_audit.json) separates
verified architecture/availability from five unresolved requirements. Only140,665
bytes of new small source/config/metadata evidence were cached, within50MiB and
the original60minute audit cap. No full weights, GPU jobs, quality variants,
external text inference or official submissions were used. Quota is unchanged.
The12894.122seconds already used in the quota period belong to earlier spans.

[Notebook27](../notebooks/27_blackpearl_source_audit.ipynb) records the source
requirements and actual stop decision. The global ranking goal remains active.
H56 modern preference heads and H62 coverage remain unselected in this span;
the next bounded selection must establish its own prospective contract.
