# Conditional next-span reward-model test

Planning only while submission58068582 runs; no new v9 recipe, GPU job, weights
download or external text service. The rank30 objective remains unchanged.

Primary sources: [Skywork paper, revised March2026](https://arxiv.org/abs/2507.01352),
[Qwen3-8B model card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B),
[0.6B card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B).
The authors describe26million curated preference pairs and a frozen scalar
reward per conversation. These are binary preference reward models, not our
three-class classifiers. Official configs confirm Qwen3ForSequenceClassification
with one output. Both checked repositories are ungated/Apache2.0; exact revisions
and card/config hashes are in results/span_009/reward_v2_source_planning.json.
Published benchmarks do not establish Kaggle quality or exclude training overlap.

## Why revisit H10

H10 proposed learning shared response utilities with10–24engineering hours and
12–28follow-up GPUhours. Released pretrained reward heads allow a much cheaper
initial test: freeze the encoder and learn only a small three-class calibration.
The source evidence changes feasibility, not the measured merit of H10. Preserve
its original record and queue a precise follow-up for v10.

Build two proper multi-turn histories from the same user turns: one with A's
responses, one with B's. Follow the card's chat-template rules, including no
extra system prompt. Export raw scalar logits, not a pipeline's sigmoid output;
do not replace the learned one-output head with a randomly initialized three-output
head. Record retained tokens, turn boundaries, exact IDs and source hashes.

Let d=r(question,A)-r(question,B). Start with exactly one symmetric mapping:

    P(A,B,tie)=softmax(beta*d, -beta*d, gamma0-gamma1*abs(d))
    beta>=0, gamma1>=0

Swapping answers exchanges A/B and preserves tie probability by construction.
Bradley-Terry training leaves a question-dependent score offset unidentified;
therefore omit the absolute reward mean initially. Ties/both-bad votes and direct
cross-answer interactions remain real weaknesses; this is a hypothesis, not an
argument that utilities dominate the current joint classifier.

## Bounded proposed test, admitted only in a future phase

After H54 is terminal and v10 activates, audit data lineage and exact offline
Kaggle availability. Do not claim independent validation if overlap is known;
report unresolved provenance explicitly. The current tascj classifier is known
to overlap original training votes, so local fusion gains cannot prove generalization.

At most two frozen size arms:0.6B as a cheap control and8B as the main quality arm.
A weak small arm alone cannot reject the larger arm. Use one fixed label-free
1024-row cohort, existing prompt groups/splits, seed42 and the mapping above.
Fit calibration in inner group folds; report cross-fitted loss, class behavior,
paired uncertainty, retained context and measured two-GPU cost. Budget initial
pilot<=one elapsed two-T4 hour only after fresh resource admission. A separately
admitted follow-up may score fixed training-calibration/dev/confirmation cohorts
within<=four additional hours; no automatic budget increase.

Retain the joint baseline. An8B utility expert would likely need to replace an
existing expert; adding it to all current inference may exceed nine hours. A
small utility expert might fit as an extra feature. Measure these costs before
any submission. Freeze one candidate only if evidence and runtime justify it;
never fit mixture weights to leaderboard feedback. If rank30 is already achieved,
do not launch this conditional work. No new GPU jobs while58068582 is pending.
