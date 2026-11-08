# Future hypothesis: approximate missing answer-order predictions

H62's larger actual Gemma-order coverage fails its fixed information gate. E17
also cautions against fitting new human-vote calibration parameters on a later
population as though it were target CV. A distinct possible route is to learn a
cheap approximation to an expensive deterministic function of the retained
experts. This is a queued hypothesis, with no experiment or GPU admitted here.

Let G be Gemma-original and L the aligned Llama-swapped predictions. The base is
(G+L)/2. Its full four-orientation counterpart is

    P4 = (G + aligned_Gemma_swapped + L + aligned_Llama_original)/4.

V10has fixed full-orientation public caches, including m3_reverse_uniform.npz;
even its routed four-orientation pipeline failed the nine-hour runtime gate.
Whether a fixed P4teacher helps over the current H60is unmeasured. The first
future test must be a CPU-only matched-cache check: preserve H60on its existing
route and use P4only off-route. Freeze that single teacher before lookup, require
uniform512descriptive gain>0.001, report paired uncertainty/exposure limits, and
stop before feature inference if this minimum useful signal is absent.

If justified, audit whether existing forward passes can expose the pooled
classification states without changing raw probabilities or pipeline overlap.
Train one small frozen head to predict centered logit residuals
log(P4)-log(base), using those states/base probabilities and teacher outputs.
Human winner labels must not train this surrogate. Keep the deployed H60route;
apply the surrogate only off-route. No extra large-model pass or external API is
part of deployment. Feature capture, head family, training IDs/groups and all
selection/gain/runtime limits must be frozen in a separately selected span.

This differs from asserting an unproven stronger preference teacher in H15. A
held-out teacher-function loss can test approximation quality even when human
votes overlap backbone training. It does not establish independent human-vote
advantage, superior hidden predictions or a stronger teacher. Those claims need
separate appropriate evidence; inspected original labels are descriptive only.
Require fresh confirmation if human-vote diagnostics would decide promotion.

Risks: missing orientations may be unpredictable from one orientation's state;
small-sample high-dimensional fitting can overfit; feature copies can break GPU
overlap; trained-checkpoint exposure can exaggerate apparent human-vote gains.
Keep strict native/base1e-5controls and actual512deployment profiling, conservative
25kcost<32400seconds with20%margin/allcold/full80warmup/other/600reserve, no hidden
prediction cache and one frozen official outcome. Prototype or teacher failure
stops the attempt. No head/teacher/temperature/route grid or leaderboard fitting.

Evidence: SPAN_017_RESULTS.md, SPAN_010_RESULTS.md and the retained H60recipe.
This proposal neither reopens completed tasks nor changes v17's E17/H62focus.
