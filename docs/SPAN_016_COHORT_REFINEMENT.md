# H56 source refinement from label-free token evidence, before GPU

The initial unused1024later-vote groups average1785reward tokens per pair versus
756.5on the fixed512competition rows, a2.36xwork mismatch. They retain83.2%of
full tokens versus96.5%for the competition sample. This is measured CPU token
evidence, not a model-quality result. No reward, candidate loss or GPU output has
been inspected. The mismatch threatens both the first-hour budget and transport
of a calibration fitted to the longer newer conversations.

Preserve the initial cohort, token references and all source-stage observations.
The large33file model/runtime asset is immutable and its ongoing upload must
continue unchanged. Before any actual quality experiment, replace ONLY the
calibration/evaluation cohort with1024post-publication groups whose exact frozen
Gemma PAB input-length bins match the original competition training distribution.
Use the previously audited bin edges[0,512,768,1024,1536,2048,3072,4096,4500]and
1024row quotas[244,248,219,207,55,33,11,7], established in v11before this reward
experiment. No labels determine lengths, quotas or selection.

Eligible groups retain all original source filters and vote timestamps strictly
after2025-07-06T18:16:33UTC. Exclude the2048previously inspected v11groups and the
1024prepared but never scored v16groups. Order remaining representatives by
sha256(42:span016:lengthmatched:<group_id>); take each bin's fixed quota. Save the
complete selection and input hashes before querying its votes. Assert exact CPU
Gemma-token identity against the original256GPU controls before selection. Stop
on insufficient bin support; no quota adjustment or seed search.

Stage this small replacement input/vote collection as ONE additional private
dataset, separate from the immutable16.38GBweights/runtime upload. No GPU
version, compute cap or quality recipe is added. Keep the earlier calibration
files in the large asset untouched and unused. The actual GPU notebook must
verify the replacement hashes and use them in both public and hidden paths.
Archive the first cohort references; do not overwrite them or pool predictions.

The main8B/control0.6Bcheckpoints, native2048token-per-conversation recipe, equal
Gemma/reward mixture, routing, outer-fold isolation, statistical thresholds and
resource caps stay fixed. Freeze replacement IDs and rehearse the final worker
before GPU admission. All evaluation remains out-of-time source evidence, not
competition CV; matching lengths does not remove model/user/language/turn shifts.
The first version remains information-only and ineligible for official submission.
