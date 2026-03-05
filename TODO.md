# TODO

Immediate next steps and near-term work items for the Alberta Framework.

## Step 2 — Remaining Work

- [ ] Neuron utility tracking (per-hidden-unit EMA of gradient magnitude)
- [ ] Feature generation and testing ("generate and test" mechanisms)
- [ ] Nonlinear feature discovery for streaming problems
- [ ] Comparison studies: MLPLearner across diverse non-stationarity types (drift, abrupt, periodic)
- [ ] AdaptiveObGD (Appendix B of Elsayed et al. 2024) — RMSProp-style second-moment normalization
- [ ] More bsuite sweep experiments and analysis (beyond catch/cartpole)

## Step 3 — GVF Predictions (Next Major Step)

- [ ] GVF specification type (cumulant, discount, policy)
- [ ] Stream TD(lambda) with MLP function approximation
- [ ] Horde architecture: many GVFs learning in parallel via MultiHeadMLPLearner
- [ ] Integration with ObGD bounding for stable GVF learning

## rlsecd Integration

- [ ] AF-2: Get permission from Edan Meyer to publish IDBD-MLP
- [ ] AF-2: IDBD-MLP 100k-event replay test in rlsecd (post permission + implementation)
- [ ] AF-2: IDBD-MLP full 1.6M log stability test
- [ ] Simplify rlsecd SecurityAgent to use framework checkpoint utilities
- [ ] Simplify rlsecd SecurityAgent to use framework config serialization
- [ ] Integrate `compute_feature_relevance` into rlsecd periodic reporting (60s interval)

## Infrastructure

- [ ] Update CHANGELOG.md with each release (moved from CLAUDE.md)
- [ ] Keep bsuite running on Python 3.13 via PYTHONPATH workaround
