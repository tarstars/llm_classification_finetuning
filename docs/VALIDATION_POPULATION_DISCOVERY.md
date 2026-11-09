# Audit what our validation populations can tell us

H35's fixed arithmetic extractor has nonzero coverage73/57477on original rows
and1/11215on later public representatives. Any checked-claim coverage is247and18,
respectively. These outputs are not blind-certified errors. They nevertheless
show that a feature can have markedly different opportunity across the two
sources even before preference labels are inspected.

The later public cohort is English, single-turn and afterJuly2025. Original
competition examples span different turns, dates, languages and model families.
The public competition classifiers overlap original training votes; later votes
can be more independent but less representative. Neither source alone gives an
unqualified estimate of hidden competition performance. H76's previously positive
local weight result and negative official outcome are a separate warning about
transfer, not a reason to choose parameters from leaderboard feedback.

Queue a bounded v39audit before another model-family trial. First compare
target-free covariates: turns, lengths, math-request/extraction/context rates,
formatting and available model-generation metadata. Model identities may be used
for diagnostics only, never hidden-test inputs. Separate full-source comparisons
from existing length-matched selections and account for source eligibility.

Report standardized differences and overlap/support, including uncertainty where
sampling is involved. Do not infer that modern models are better at arithmetic
from this parser, or that a null modern-source feature helps the competition.
Do not reopen H35or widen its parser after inspecting cases. Do not change the
historical outcomes of H79/H78/H59/H76.

Only if the source audit establishes a material evaluation mismatch, separately
freeze any label/prediction analysis with identical recipes and explicit teacher
exposure. Original in-sample teacher predictions are not OOF; class mapping and
any fitted component remain fold-local. A new population is a new experiment,
not retrospective replacement of a failed result. No GPU, model/weight/prompt
search or official follows from this discovery alone. Preserve original resource
anchors and require useful prospective evidence before numerical qualification.

Evidence: [H35](SPAN_038_RESULTS.md), [H79](SPAN_036_RESULTS.md),
[H76official](SPAN_030_RESULTS.md).
