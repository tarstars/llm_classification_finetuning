"""Render an evidence-based review of every v7 member without executing carryovers."""
from pathlib import Path
import json

REPO=Path(__file__).resolve().parents[1]
reviews={
'H02':('High upside, next funded training wave','Reprofile8B with the new exposure telemetry. Old8B and4B pilots used different batches/exposure, so they do not establish model-size quality. A full larger-model comparison needs more than this span\'s remaining quota.'),
'H05':('Retain a matched-schedule test','H49 changes exposure under one restarted schedule. Even a positive result will not isolate a learning-rate cause. Compare at most two schedules at matched accepted tokens before a larger run.'),
'H06':('Retain after the current backbone test','Last-token pooling and24.9% order disagreement motivate response-aware pooling, not proof it helps. Freeze representation and compare a small head or matched fine-tune; include extraction/OOF cost.'),
'H07':('Defer architecture replacement','Bidirectional comparison changes the pretrained causal representation and training cost. Prefer smaller pooling/serialization interventions before a new attention architecture.'),
'H08':('Keep a separate numerical experiment','Intentional precision changes need consistently rescored cohorts and calibration. They are different from E05\'s instability with unchanged weights/settings. Profile before promising speed or quality.'),
'H10':('Retain as a distinct modeling assumption','Shared response utility is efficient but may miss pair-specific ties and nontransitive preferences. Test against the joint classifier with equal data/exposure, not a scalar reward gap alone.'),
'H12':('Audit replicate support before soft labels','Repeated questions with different answers are not repeated votes on the same pair. Count exact paired replicas before smoothing; do not derive a noise floor from prompt repetition.'),
'H13':('Low priority without noise evidence','A high in-sample loss can mean a hard example rather than a wrong vote. Robust loss needs a fixed noise hypothesis and matched calibration; the current T<1 does not support a blanket overconfidence story.'),
'H14':('Defer auxiliary supervision engineering','Criterion labels require trustworthy provenance and held-out value beyond the primary CE. Teacher correctness and human preference are different targets; begin with a small verified criterion set.'),
'H15':('Conditional on a useful stable teacher','Distillation should follow a teacher/mixture gain on matched held-out rows and an affordable data-generation plan. E05 backend stability is necessary for this native teacher, not sufficient quality evidence.'),
'H16':('Retain a narrow refusal test','Separate appropriate refusals from generic refusal phrases. Use predefined slices and global calibration controls; lexical source/style correlations alone do not justify refusal-specific relabeling.'),
'H17':('High upside, data audit first','Public preference data may offer more than tiny source corrections. First verify competition permission, license, pair/prompt overlap, source shift and provenance; keep validation groups isolated. No data is acquired in this span.'),
'H18':('Retain small controlled counterexamples','Generate or curate changes with a known constraint violation and a reliable preferred response. Avoid changing several answer properties at once or assuming a teacher\'s preference is human ground truth.'),
'H19':('Retain verified factual subset','The poster contains factual mistakes, but their overall vote-loss contribution is unmeasured. Verify facts offline/provenance first and test a small controlled training intervention; do not replace human labels from unsupported recollection.'),
'H20':('Defer expensive OOF mining','Current training predictions are in-sample. Honest hard-example OOF mining can multiply backbone training cost; loss-based oversampling may amplify noisy votes. Account for those folds before spending GPU.'),
'H21':('Retain with fixed domain evaluation','Domain-loss differences are observational. Preserve the competition\'s overall weighting and inspect shift before reweighting; report whether a global gain survives all domains.'),
'H22':('Cheap preparation, matched training later','Prompt grouping already prevents a major split leak, but repeated prompts can still dominate exposure. Quantify effective prompt weights first, then change only the training sampler/weights at matched exposure.'),
'H23':('Revisit through H48 first','H48 expands the inference-only context test to every changed development input. This cannot establish a trained2K model\'s value; a positive result motivates matched-exposure long-context training, a negative one lowers its priority.'),
'H24':('Keep near the next packing wave','The request cap remains256tokens even at2K. Diagnose lost request constraints, then test adaptive allocation without changing total tokens. Answer-length confounding is not a causal packing result.'),
'H25':('Keep near the next packing wave','[TURN] boundaries survive parsing, but the fields are concatenated rather than turn-interleaved. Test aligned serialization on a predefined multiturn slice and full overall score at the same token budget.'),
'H26':('Defer hierarchy until cheaper retention tests','Chunking increases compute and may lose cross-chunk comparisons. H48, adaptive allocation and extractive retention offer cheaper information before hierarchical training.'),
'H27':('Retain cheap deterministic retention','Head/tail can omit a decisive middle segment. Compare a single fixed extractive rule at matched token count, with unchanged rows included and no answer-correctness oracle in selection.'),
'H28':('Defer generated compression','Summarization adds hallucination, lost constraints and substantial inference time. Prefer deterministic retention tests before a generator whose total hidden runtime may exceed the limit.'),
'H29':('Use a stable backend before prompt tuning','Freeze prompt candidates before held-out evaluation. Backend drift can masquerade as a prompt gain, and the300judged examples are already inspected. A better rubric does not supply missing factual knowledge automatically.'),
'H30':('Retain only with train-only retrieval','Exclude held prompts/pairs from demonstrations and include extra prompt tokens in timing/retention. A demonstration can improve format while crowding out the answers being judged.'),
'H31':('Defer until a small affordable route is justified','A critique adds generated tokens and may rationalize an incorrect choice. Compare a fixed short critique to direct judgment under stable serving and the full route budget.'),
'H32':('Defer expensive reference generation','Generating an ideal answer is much slower than scoring three label tokens and can introduce false facts. Require a small informative slice and bounded route before whole-data generation.'),
'H33':('Longer-term grounding work','Offline fact retrieval needs a licensed, reproducible index and coverage evidence. Test its relevance to preference loss, not just factual lookup accuracy; do not assume it solves ambiguous human votes.'),
'H34':('Subset feasibility before execution evidence','First estimate runnable-code coverage and static-check value. Any actual execution needs a constrained sandbox and strict time/memory limits; passing tests still does not determine every human preference.'),
'H35':('Retain a bounded symbolic subset','Use verifiable parsing/equivalence on a predefined mathematical subset. Report coverage and global impact; unsupported natural-language math judgments are not symbolic verification.'),
'H36':('Raise as a cheap future residual test','V6 style nearly matched inferred strength. A small nonlinear structural residual may be worth a CPU-only nested test, with classifier calibration/style controls and a very small fixed family; no in-sample backbone stacking.'),
'H38':('Defer multiple experts','Experts/gating require several fits and honest OOF predictions. No current evidence shows this beats one better general model at equal total compute; domain imbalance adds validation risk.'),
'H41':('Keep deferred after reward diagnostics','A scalar reward gap loses tie information; pooled features might recover it but need new consistent extraction. Compare with the stronger H03 reference and include extraction/OOF costs, not the old weaker comparator.'),
'H44':('Conditional work in this span','Fit only after E05 demonstrates a genuinely stable affordable recipe and a uniform fresh cache exists. Every calibration/routing stage stays inside grouped folds. Otherwise defer without fitting.'),
'E05':('Execute the skipped stability factor','V4 changed cache reuse and remained unstable. This span isolates client concurrency1/16 with fixed server/model/cache-off; preserve label-independent replay/order/restart guards.'),
'E08':('Keep deferred on the failed source gate','H46\'s tiny gain was inconclusive and style nearly matched it. H47 completion alone cannot unlock source deployment: a genuinely positive, separately frozen quality transfer must come first.'),
'H47':('Execute a bounded CPU target comparison','Test direct strength regression against identity-derived expected strength with fold-local rating targets and equal residual features. Do not let source-class accuracy stand in for preference quality.'),
'E09':('Framework foundation in this span','Fault-tested shared admission, mutation intents, bounded retrieval and immutable sources reduce expensive orchestration mistakes. Historical frozen programs stay intact.'),
'E10':('Actual training observability in this span','Track accepted exposure, skips, gradients, learning rate, scaler/scheduler, hardware and checkpoints. Audit both ranks; a live rank0 token counter is not a measured global token total.'),
'E11':('Theory and evidence synthesis in this span','Reproduce the tie/winner CE factorization and distinguish optimization, calibration, preference and factual correctness. Review every active member; forecasts remain unmeasured.'),
'H48':('Execute the full paired context cohort','Fresh short/long predictions cover all1152changed inputs, with a fixed batch8 memory/numerical guard and128-row checkpoints. Incomplete coverage cannot become an assumed overall gain.'),
'H49':('Primary score experiment in this span','One verified warm restart from step1438, up to320newcycles, fixed final endpoint and matched grouped calibration. Report exact added exposure and any budget/instability stopping.'),
'H50':('Conditional one-candidate ladder test','Predeclared priority is training, then context. Freeze before confirmation and require genuine quality, identity and runtime gates before one official evaluation; never select on leaderboard feedback.')}

r=json.loads((REPO/'backlog/hypotheses.json').read_text());version=next(v for v in r['versions'] if v['number']==7);items={i['id']:i for i in r['items']}
assert set(version['item_ids'])==set(reviews),'Every current task needs an explicit review'
rows=['# V7 review of skipped work and current focus','',
      'This review uses existing project evidence and the fixed v7 plan. Decisions are priorities and test designs, not measured gains. No unselected task is executed by this review. Existing estimate ranges remain subjective; fresh resource admission is required before a later span.', '',
      '| Task | Priority / current status | Decision | Evidence and next bounded test |','| --- | --- | --- | --- |']
for ident in version['item_ids']:
    item=items[ident];decision,note=reviews[ident]
    rows.append(f"| {ident}: {item['title']} | {item['priority']} / {item['status']} | {decision} | {note} |")
rows+=['','All43v7members are reviewed, including36unfinished carryovers and7new v7 items. H47 was queued at the v6 boundary; the six additional items came from this explicit planning request. The current focus is nine tasks.',
       '', 'Sources: [v7 plan](SPAN_007_PLAN.md), [theoretical guide](PIPELINE_THEORY.md), [v6](SPAN_006_RESULTS.md), [v5](SPAN_005_RESULTS.md), [v4](SPAN_004_RESULTS.md), [v3](SPAN_003_RESULTS.md), [H03 exposure](SPAN_002_H03.md), [EDA](../EDA_REPORT.md) and each item\'s evidence/history in the registry.']
(REPO/'docs/SPAN_007_CARRYOVER_REVIEW.md').write_text('\n'.join(rows)+'\n')
print('Reviewed',len(reviews),'tasks without executing unselected work')
