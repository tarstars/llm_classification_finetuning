# V32: refusal detector fails before preference fitting

H16 stopped at its prespecified blind detector gate. No conditional preference
correction was fitted, no H16 preference targets were loaded and no quality gain
was measured. This closes the fixed detector experiment, not the broader question
of whether refusal appropriateness can improve our strong ensemble.

## Coverage and blind review

The existing uncalibrated v11 strong-expert caches and their exact input/target
file hashes were verified without loading targets. Coverage passed the required
32 one-sided detections in each partition:

| Post-checkpoint source partition | Rows | A only | B only | Both | One-sided fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learning | 1024 | 41 | 47 | 24 | 8.59% |
| Confirmation | 1024 | 36 | 36 | 16 | 7.03% |

The deterministic blind audit contained24positive detections (12per side) and
8nondetections, drawn only from original training text. No preference labels were
included. Codex reviewed the first600characters, retaining ambiguity rather
than adjusting labels to the threshold. Among24positive detections:

- **17 clear opening refusals**:70.83% precision on this small audit.
- **2 ambiguous cases**: an answered wordplay riddle with a numerical caveat and
  an explanation of the liar paradox ending in an inability claim.
- **5 false positives**: compliance with a request not to mention a subject;
  generic physical visibility using “we”; quoted fictional dialogue; a riddle
  explanation about grasping an abstract moment; and language capabilities
  followed by a machine-versus-human caveat.

Even counting both ambiguous cases as correct yields **19/24=79.17%**, below the
required20/24. The detector gate fails. None of the8reviewed nondetections
contained an opening refusal, but this does not establish recall. There is one
reviewer and only32cases; these are descriptive audit results, not precise
population estimates or ground-truth appropriateness annotations.

## Decision and limits

Stop before target attachment, nested fitting or confirmation evaluation. No
regex/threshold/sample/model rescue is performed in v32. The planned global
refusal and style controls, conditional coefficient fits and quality baselines
were **not executed**. No new log loss or accuracy can be reported, and no
preference benefit or harm follows from detector precision alone.

The finding explains a concrete feature risk: first-person inability wording
does not reliably identify an assistant declining the user's request. A future
follow-up should distinguish assistant speech from quoted dialogue, generic
physical statements, compliance and capability description before learning a
preference correction. That follow-up needs a distinct frozen detector and new
blind training-only audit, retaining the failed detector and avoiding preference
label-based detector revision. Its quality is unmeasured.

The2025source cohorts were inspected by prior experiments and differ from the
competition. Their caches also lack H60's routed reverse-Gemma pass. Even a
successful future fit would require matched-current-pipeline transfer and actual
runtime/source qualification; it would not estimate an official score.

## Engineering, resources and evidence

The first preparation attempt86872failed on a row-sum assertion tighter than
float32 rounding:44/1024rows differed from1 by1.1920929e-7. Original source and
failure evidence were retained. The correction uses the unchanged original
pipeline probability validator (atol1e-5); predictions and scientific gates did
not change. Successful preparation36794exit0 took3.470seconds with0.991GiBRSS.
The failed first attempt did not instrument its elapsed time, so no exact summed
preparation CPU time is claimed. Two focused descriptor tests passed, including
swap antisymmetry and the opening-window boundary.

There were0newGPUversions,0officials,0externaltextcalls and0modeldownloads.
All prior GPU charges, the November14quota anchor and original November10deadline
remain in force. The terminal resource observation records fresh quota, previous
pilot status, complete official history and board; this CPU-only span does not
claim to have refreshed the full owned-notebook catalog. Future GPU work requires
fresh full admission. The rank<=19goal remains active and unachieved.

Fresh08:43UTC observation on November9: quota is unchanged at36651.064used/
71348.936free seconds,0reserved/paidFalse; previous pilotCOMPLETE and all8official
receipts terminal. Complete board remains23/309,best0.82987,19th/20th0.82915,
leader0.82674. No leaderboard improvement is established. H77 is queued for v33
with the actor-attribution discovery;36unfinished tasks carry forward alongside
it. Historical126snapshots, exact goals, user-edited notebooks and link are
preserved; v32closure adds4final/start snapshot files.

[Preparation](../results/span_032/preparation.json) ·
[Blind detector audit](../results/span_032/detector_audit.json) ·
[Preserved assertion failure](../results/span_032/preparation_assertion_failure.json) ·
[Executed Notebook44](../notebooks/44_refusal_detector_audit.ipynb) ·
[Resources](../results/span_032/resource_close.json).

Heavy inputs, per-row text descriptors, the32-case sample and its annotations
remain under `~/database/llm_classification_finetuning/runs/span_032`.
