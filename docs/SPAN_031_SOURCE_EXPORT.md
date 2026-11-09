# Accepted H02 source verified through direct version evidence

The sole accepted GPU version is
`arseniytrushin/llm-preference-span031-size-compact/1`, accepted November9 at
07:14:42UTC and RUNNING by07:18. Controller1748 monitors this same version.
The numerical notebook has not changed since its tested compact build.

The original read-only source verifier2423 stopped on `kernel_sources`:
Kaggle's metadata export omitted `/1` from notebook and dataset references.
It preserved model version numbers, with only framework capitalization
different. Exact code cells and private/offline/GPU flags had already matched.
Preserve that verifier and its failed assertion; do not edit frozen source or
restart the accepted job to work around this metadata representation.

Independent verifier49482 exited0 at07:22:09UTC. It compared the same downloaded
accepted source cells, checked privacy/offline/GPU/model references, and queried
each parent directly. Both completed parent notebooks have current version1,
and the private input dataset has current version1. Thus the omitted suffixes
have only the required numbered version available; names alone were not used
as proof. A newer parent or dataset would have failed this verification.

[Actual source proof](../results/span_031/jobs/size_pilot_compact/source_verification.json)
records the direct observations and hashes. No numerical, input, resource,
memory, quality or runtime condition was changed. GPU fit, full32-update
exposure and development quality still require the post-run independent audit.
