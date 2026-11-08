# V26 / H72: extra conversation evidence for the retained Gemma expert

The rank<=19goal remains active; retain H60official0.82987. Select H72only.
H71's fixed identical rule failed and stops. Old H23/H24remain historical
carryovers: their earlier tokenizer/training claims are not current-model facts.
Actual retained ProcessorPAB truncates the concatenated conversation prefix to
4096tokens before adding the unchanged judge/chat framing. It can omit B-tail
or later turns. No4k/6kcoverage or quality comparison has been made yet.

First perform one label-free CPUaudit, <=1200elapsed seconds, with the exact
native Transformers4.42.3/Tokenizers0.19.1and retained Gemma assets/processor.
Compare actual4096versus6144build_input token arrays on all57477originalrows;
verify old cached IDs/lengths and canonical256control fingerprint. Record exact
changedIDs, old/new lengths/fingerprints, model position limit, prefix field
coverage and token-work. Field coverage is character exposure in the conversation
prefix, not a measure of useful semantic evidence. Any reconstructed formatting
must equal the native conversation exactly. No target columns are read.

ONE6144variant, fixed before counting. Candidate adds a full original-order6k
Gemma pass only on actually changed/truncated rows. Keep the original H60route,
Gold/Llama/reverse inputs and expert weights; replace only the original-order
Gemma component. No recomputed global route, balanced reformatting, different
length thresholds or weights. Added work means the whole extra pass, not merely
the additional2048tokens. CPUadvancement requires at least64changedrows, at most
1%of originalrows, added nativeworkproxy<=15%of originalGemmaproxy and input
positions within model limits. These are feasibility screens, not runtime proof.

If supported, freeze up to384changedIDs (seed44, sorted IDs) and the actual pilot
worker/scoring method before any GPUoutputs or labels. Four matched outputs:
Goldoriginal, alignedGoldreverse, G6koriginal and alignedLlama. For fixed outputs:

    parent_off = .5Gold + .5L
    candidate_off = .5G6k + .5L
    parent_on = .25Gold + .25Goldreverse + .5L
    candidate_on = .25G6k + .25Goldreverse + .5L

Use minimum per-row gain over the two outcomes, original changed-row prevalence
weighting, seed42/2000pairedpromptgroups and class/retention diagnostics. Before
any GPUoutput, the prospective evaluation rule is amended in
SPAN_026_EVALUATION_AMENDMENT.md: local original-data loss is descriptive, with
a fixed gross-regression stop if conservative point gain is below-0.005. It is
not an independent promotion metric. Original training overlap and fixed-output
numerical-bound limitations remain; noCV/hidden-scoreforecast. The initial
CPUcontract/goal are preserved separately and anchored at278b9dc.

GPUis conditional, not admitted by this plan/CPUresult alone. Initial pilot
<=7200two-T4seconds, with an early real6kmemory/throughput probe and conservative
9hprojection before a larger quality cohort. Stop early on memory/runtime/source
failure. Canonical1e-5oldrecipe controls, exact newtoken fingerprints, pinned
offline sources/devices and complete artifact/terminal checks are mandatory.
Passing engineering and the fixed gross-regression guard permits one<=3600second full H60+extra-context
qualification, using worse retained/current runtime,20%margin, allcold/fullwarm/
other overhead and600sreserve, total<32400s. Then at most one frozenofficial.

Publicphase<=10800seconds/max2acceptedversions/oneofficial. No newweights,
conversion, training, external text service, GPUduringofficialpending or timeout
duplicate mutation. Inherit Nov10 08:37:37.343197UTCdeadline and Nov14quota anchor;
fresh complete quota/jobs/receipts/dependencies before each GPU. No alternative
context/routing grid after a failed gate.

Heavy ROOT/runs/span_026; aggregate/report/executednotebook38 inGit. Preserve102
snapshots, archives, user07/12 and notebooklink. CLItracking; newdiscoveries next
draft, unfinishedcarry. Source publication before measurement; span completion
is not achievement of the actual rankgoal.
