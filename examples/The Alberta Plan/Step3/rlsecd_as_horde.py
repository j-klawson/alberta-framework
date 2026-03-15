"""rlsecd's 5 heads expressed as GVF demons.

Shows how rlsecd's existing MultiHeadMLPLearner configuration maps
directly to HordeLearner with 5 single-step prediction demons.

This script validates that wrapping in HordeLearner produces identical
results (all gamma=0), providing a migration path for rlsecd.

Usage:
    python "examples/The Alberta Plan/Step3/rlsecd_as_horde.py"
"""

import jax.numpy as jnp
import jax.random as jr

from alberta_framework import (
    DemonType,
    EMANormalizer,
    GVFSpec,
    HordeLearner,
    MultiHeadMLPLearner,
    ObGDBounding,
    Timer,
    create_horde_spec,
)

# rlsecd's 5 prediction heads as GVF demons
RLSECD_DEMONS = [
    GVFSpec(
        name="is_malicious",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=0,
    ),
    GVFSpec(
        name="attack_type",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=1,
    ),
    GVFSpec(
        name="severity",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=2,
    ),
    GVFSpec(
        name="confidence",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=3,
    ),
    GVFSpec(
        name="action_quality",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=4,
    ),
]


def main() -> None:
    """Validate HordeLearner equivalence with MultiHeadMLPLearner."""
    feature_dim = 26  # rlsecd's feature dimension
    n_heads = 5
    key = jr.key(42)
    k1, k2, k3 = jr.split(key, 3)

    # Shared config
    kwargs = dict(
        hidden_sizes=(128, 64),
        step_size=1.0,
        sparsity=0.9,
        normalizer=EMANormalizer(decay=0.99),
        bounder=ObGDBounding(kappa=2.0),
        use_layer_norm=True,
    )

    # Original MultiHeadMLPLearner (as rlsecd uses)
    multi = MultiHeadMLPLearner(n_heads=n_heads, **kwargs)

    # HordeLearner equivalent
    horde_spec = create_horde_spec(RLSECD_DEMONS)
    horde = HordeLearner(horde_spec=horde_spec, **kwargs)

    # Init with same key
    m_state = multi.init(feature_dim, k1)
    h_state = horde.init(feature_dim, k1)

    # Generate some test data
    obs = jr.normal(k2, (feature_dim,))
    targets = jr.normal(k3, (n_heads,))
    next_obs = jnp.zeros(feature_dim)  # doesn't matter for gamma=0

    # Compare predictions
    with Timer("MultiHead predict"):
        m_preds = multi.predict(m_state, obs)
    with Timer("Horde predict"):
        h_preds = horde.predict(h_state, obs)

    print("\nPrediction comparison:")
    for i, demon in enumerate(RLSECD_DEMONS):
        print(
            f"  {demon.name}: multi={float(m_preds[i]):.6f}, "
            f"horde={float(h_preds[i]):.6f}, "
            f"match={jnp.allclose(m_preds[i], h_preds[i])}"
        )

    # Compare updates
    m_result = multi.update(m_state, obs, targets)
    h_result = horde.update(h_state, obs, targets, next_obs)

    print("\nUpdate comparison:")
    for i, demon in enumerate(RLSECD_DEMONS):
        m_err = float(m_result.errors[i])
        h_err = float(h_result.td_errors[i])
        print(
            f"  {demon.name}: multi_err={m_err:.6f}, "
            f"horde_err={h_err:.6f}, "
            f"match={jnp.allclose(m_result.errors[i], h_result.td_errors[i])}"
        )

    all_match = jnp.allclose(m_preds, h_preds) and jnp.allclose(
        m_result.errors, h_result.td_errors
    )
    print(f"\nAll outputs match: {all_match}")

    # Show config
    print("\nHorde config:")
    config = horde.to_config()
    print(f"  Type: {config['type']}")
    print(f"  Demons: {len(config['horde_spec']['demons'])}")
    for d in config["horde_spec"]["demons"]:
        print(f"    - {d['name']} (gamma={d['gamma']}, type={d['demon_type']})")


if __name__ == "__main__":
    main()
