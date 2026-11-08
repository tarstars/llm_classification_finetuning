# Check the fixed classifier's usefulness before another backend experiment

H66's live complete512row mirror profile is267.815618533seconds, versus
H65's298.576158465seconds:1.11486xthroughput. The required retained parent rate is
276.107160295seconds/512rows. Thus even the lower bound

`25000 * 1.2 * (267.815618533 + 276.107160295) / 512 + 600 = 32470.47532seconds`

exceeds32400before all startup/warmup/tokenization/other costs. The declared
runtimegate cannot pass. The original accepted H66job is still running; retain
its source, terminal status and every actual canonical/numerical/memory failure
or success. Do not submit it or run another batch shape. Its quality remains
unmeasured because the current contract gates votes behind feasible runtime.

Queue a separate v21CPU-only information screen after H66closes. Use its completed,
hash-verified original512/later1024mirror predictions, matched retained Gemma and
actual full H60baseline (including source175reverse rows). ONEprimary candidate
remains .5retainedGemmaoriginal+.5mirrororiginal,T1, same new-shape outputs for
all cohorts. No fitting, new weight/temperature/route/class-map grid or imputation.
Do not alter H66's engineering stop or mark its source deployment eligible.

Evaluate the same predeclared quality criteria once: originaluniform reduction
>.001 versus H60; later1024reduction>=0 with seed42/2000paired-group95lower>-.01.
Report logloss/accuracy/Brier/class behavior and training/later-inspection/domain
limits. This CPU screen is a research usefulness decision, not deployment
qualification, independent CV or a competition-score forecast. Stop further
backend work if it fails. Budget<=300CPUseconds/zeroGPU/zeroofficial/zeronetwork.

Only a useful fixed predictor would justify a new bounded backend proposal.
An earlier option was deferred, not disproven: src/arena/dense_checkpoint.py can
stream the original GPTQ8tensors into denseF16, and span019_model.py already has
native-softcap stage1/stage2 and the corrected native matrix layout. The actual
saved tiny CPUcheckpoint test compares originalquantized/dense/split outputs.
A two-T4dense layer pipeline could avoid dequantizing every projection for each
batch while sharing one18.5GBmodel across both devices. Real GPUidentity, memory,
speed and conversion overhead are unmeasured. This is a distinct backend idea,
not another batch grid or permission to expand/upload weights now. Any such
future experiment needs its own source/code/resource contract and full native
matched-packet<=1e-5and actual512/full80warmup/runtime/quality qualification.
