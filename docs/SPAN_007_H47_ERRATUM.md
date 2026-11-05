# H47 fold-count documentation correction

At **2026-11-05 19:26:26 UTC**, while source fitting was still underway and before
any preference results, the original protocol's source-fold count was found to
be wrong. V5 has **three source folds**, of15,282/15,372/15,331training pairs.
Its separate preference evaluation has **five outer folds**, and both use three
inner folds in the new H47 procedure.

The committed executable loops over the saved source-fold values and was already
running exactly those three folds. It asserts equality of each source fitting
index set against the old E07 artifacts. No source row, group, rating target,
penalty grid, feature, fitted parameter or evaluation rule is changed. Final ridge
penalty remains the code's frozen median of the original source-fold selections,
therefore a median of three selections rather than the prose's claimed five.

Preserve the original protocol and contract, including their incorrect count,
as the pre-fit record. Read this erratum with them. The executable recipe and
original data/fold files take precedence over the mistaken prose. This is a
documentation correction, not a newly selected variant or a restart. The original
CPU clock and cap remain unchanged.

The adjacent [fold-schema proof](../results/span_007/H47_fold_schema.json) records
actual counts and hashes, so final auditing can distinguish source and preference
folds. Future contracts should derive counts from saved fold assignments rather
than repeat them manually in prose.
