# H02 input transport repair before any accepted GPU version

The first SaveKernel request returned HTTP400 at06:23:58UTC November9.
Controller82267 exited1. The full response body was not retained, so the
server's exact rejection reason is unknown. Read-only reconciliation at06:26
found no run for the requested notebook (HTTP404), no matching owned notebook,
no reservation, and unchanged32965.550seconds used. Preserve its original
intent, admission, source bundle, notebook and source in place. Never rerun that
controller or delete its intent. This was not a training failure or a measured
quality result, and no accepted version or GPU charge is established.

The SDK-normalized source measured2,179,346bytes. Kaggle documents source-size
rejections around1MB in its support forum, making the inline selected CSV a
plausible cause, not a confirmed explanation of this specific400:
[Kaggle support discussion](https://www.kaggle.com/product-feedback/553902).
The earlier16MiB local builder guard was insufficient for this transport.

Move the **same already frozen compressed CSV bytes** to one private Kaggle
dataset `arseniytrushin/llm-span031-size-inputs/1`. It contains only the1024train
and512development rows and their original training votes. Preserve original
competition terms, keep the asset private, and include no confirmation or
hidden votes, predictions or model identities. Use a binary suffix to avoid
automatic decompression. Upload once with a durable intent, then verify the
same version/private/ready state; never repeat an uncertain asset mutation.
The notebook checks its exact SHA256 before loading models.

Keep all model/data/split/ID/packing/seed/optimizer/update/memory/quality/runtime
rules unchanged. The compact notebook must pass actual serialized staging,
CPUTrainer, success/failure/watchdog and Jupyter rehearsals and both file and
SDK-normalized source guards below900,000bytes. Publish the new source and
input manifest before upload, and the concrete accepted asset/build evidence
before one new SaveKernel call. Capture any new HTTPerror message without
retrying or exposing request headers/source payloads.

Use a separate `size_pilot_compact` controller directory and notebook identifier
so the rejected intent is preserved. Before its single mutation, freshly
reconcile the original identifier again, require no accepted old version or
GPU charge, all official receipts terminal, no live owned job, exact private
asset version1 ready, and the original remaining quota/deadline allowance.
Across both transport attempts allow **at most one accepted public GPU version**
and **5400charged session-seconds total**, no official submission, same span
start/quota anchor/deadline and all previous charges. There is no numerical
retry, parameter search or expanded compute allocation. A further transport
rejection must be preserved and diagnosed from its captured response; do not
blindly submit again. No goal completion follows from transport recovery.

The first compact-controller call47887 then stopped during its read-only
inventory, before any compact save intent or mutation: the profile API returned
one `[Private Notebook]` row with no reference or ID and a2010placeholder date.
Explicit owner filtering did not recover it. The separate authenticated
`Your Work` catalog returned75canonical owned references, preserved all74
addressable profile references, and uniquely recovered the rejected embedded
notebook's identifier. Its run endpoint returns404; its private draft source
endpoint returns403 and is not needed or accessed further. Thus the earlier
inference that no notebook object existed was too broad: no run was found,
but a redacted private draft exists. No GPU charge or reservation is present.

The scoped inventory adapter resolves that exact entry through both complete
catalogs, then checks its run status directly and lets the existing inventory
check every remaining notebook. It drops no row. Unknown redactions, missing
references, unequal coverage, pagination, wrong ownership, or an actual run
prevent admission. Four boundary tests passed. Preserve the earlier compact
bundle and record a source amendment for this admission-only fix; notebook,
data, model, configuration, GPU allocation and numerical tests are unchanged.
