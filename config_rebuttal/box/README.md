# Archived rebuttal configurations

`legacy_groups/` contains the previous numbered configuration groups exactly as
they existed before the Actor and AttributedGraph-Flickr consolidation.

The active configurations now live under:

- `../clean_heter/<dataset>/`
- `../featfree_heter/<smoother>/<dataset>/<backbone>/`
- `../figure1_heter/<dataset>/<backbone>/`

`suites/` contains superseded suite scripts. Their paths point into
`legacy_groups/` and the corresponding archived experiment outputs under
`rebuttal_experiments/box/legacy_groups/`.

Do not use archived configurations for new Actor or AttributedGraph-Flickr
runs. Use the active configuration tree and the root-level rebuttal suites.
