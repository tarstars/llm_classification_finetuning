# Target access audit: loaded columns versus used labels

H49 reuses the frozen `read_competition` loader. It loads the CSV's columns into
memory, including the confirmation target columns. The training program then
selects labels only from the training partition, fixed development monitoring
rows and final development partition. Explicit ID splits use IDs and prompt
groups, not confirmation votes. No confirmation vote enters a model, loss,
calibrator, stopping rule or selection criterion.

The worker summary's inherited-style field `confirmation_targets_read=false`
should therefore be read as **confirmation targets not used**. It does not prove
those columns were never loaded. Preserve the original worker/source/summary,
and use the precise distinction in public results. No numerical recipe, labels,
weights or split is changed to correct this reporting distinction.

The accompanying target-isolation test changes confirmation votes in a fixture
and checks that the same frozen loader/split/data-selection path leaves training
and development labels/text unchanged. The actual source audit additionally
checks that only those three selected subframes feed losses in H49. This supports
non-use, without making an incorrect column-access claim.

New label-free prediction code selects only ID/request/answer columns. H47's
confirmation input Parquet physically has no target columns. H50 must freeze the
candidate before it explicitly evaluates confirmation votes. Both original
cohorts remain historically inspected; this audit creates no untouched data.
