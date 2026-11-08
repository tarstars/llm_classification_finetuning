# V23: no qualifying new checkpoint in the audited releases

The bounded source audit found **no immediately usable different strong classifier**
in the inspected releases. This is a scoped negative result, not an exhaustive
claim about all public models. No weights, GPU inference or official submission
were used. The rank goal remains active at23/312, loss0.82987.

| Source | Verified observation | Decision |
| --- | --- | --- |
| Retained tascj author | Complete public listing:20datasets, onepage; LMSYS has only m0/m3, modules and packages. Pinned stage3 has m0/m3 only. Stage1 m1 is Gemma2-27B. | No additional final checkpoint identified; do not invent m1/m2 releases. |
| Third-place Raja Biswas | Public inference notebook v5 links a trained Gemma adapter v1 and runtime utilities v3. Adapter config saves classification_head/classifier/score, rank16/alpha32, and names its trained headless base. | Exact base returns401 on HF; exact-name Kaggle model search is empty. Stop this release, no substitute base. |
| Fourth-place historical solution | Primary repository now redirects to pairjudge. Original inference remains under competition/, with a gemma9b-ru mount; public exact-name dataset search is empty. | Artifact owner/version is unresolved; no arbitrary mirror or weight download. |
| New pairjudge example | Its own report explicitly distinguishes the small0.5B example from the historical medal model; reported heldout loss is1.05855 with swap averaging. | No evidence to prioritize it as a stronger full replacement. Different evaluation population prevents direct comparison. |

The third-place source is pinned at
`4007b8dfdb4c19899252962ff7f690e5f7038c86`; fourth-place source at
`f4d2a3fbd0497e28673010787612d3dac04e7668`. Retained tascj source remains
`83cd93d50b9283c18711e8c63e4e1c6399c7b9ce`. The actual third-place Kaggle runtime
model file differs from the GitHub reference (3,000 versus3,300bytes); it must not
be silently replaced. Neither source availability nor modules_to_save proves
actual trained tensor quality. No weights were read in this audit.

Two bounded HF catalog samples were also archived: the first100 LMSYS name
matches and top40 downloaded Gemma classification tags. These are partial search
results; unrelated routers/reward/scalar heads and gated models do not establish
an available three-class competition classifier. H55's prior403 was not retried.

## A concrete successor from the primary source

Fourth-place historical inference applies a fixed structural rule after model
inference. Empty-answer strings are exactly `[null]`, `[]`, `[ ]`, `[  ]`,
`[""]`, `["",""]`. It assigns `[.04,.88,.08]` for emptyA,
`[.88,.04,.08]` for emptyB, then `[.06,.06,.88]` for exact raw-string-identical
answers, in that order. Both-empty cases therefore follow the identical rule
when their raw strings match. These are published constants, not fitted here.

Queue a single H70 test against H60, keeping every other probability unchanged.
First freeze the exact rule, identify coverage without targets, and replay the
existing original512cache once. Original rows overlap public-checkpoint training,
so this is a futility screen only. Sparse coverage cannot establish transfer.
Any expanded stratified inference must be separately frozen, capped and admitted;
no threshold/probability/string-normalization grid or copied predictions. This
cheap source-derived lead does not reopen the rejected mirror.

The source archive is under ROOT/runs/span_023/source_audit; its final manifest
records hashes, elapsedtime and bytes. No prediction outputs from public notebooks
were retained or inspected; only their code/Markdown and metadata were used.
Preserve90historical snapshots, archived goals and usernotebooks07/12. Close H69,
carry37unfinished tasks and queue H70 for the next span under the unchanged goal.

Primary sources:
[tascj stage3](https://github.com/tascj/kaggle-lmsys-chatbot-arena/blob/83cd93d50b9283c18711e8c63e4e1c6399c7b9ce/configs/stage3/README.md),
[third-place training config](https://github.com/rbiswasfc/lmsys-arena/blob/4007b8dfdb4c19899252962ff7f690e5f7038c86/conf/r_gemma/conf_r_gemma_m205.yaml),
[author's inference](https://www.kaggle.com/code/conjuring92/lmsys-3rd-place-submission),
[fourth-place historical inference](https://github.com/DaoyuanLi2816/pairjudge/blob/f4d2a3fbd0497e28673010787612d3dac04e7668/competition/inference.ipynb).

Project evidence: [listing](../results/span_023/source_listing.json),
[inventories/access](../results/span_023/candidate_inventory.json),
[base search](../results/span_023/base_observation.json),
[final audit](../results/span_023/source_audit.json),
[executed notebook35](../notebooks/35_public_classifier_sources.ipynb).
