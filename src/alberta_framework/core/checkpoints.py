"""Checkpoint utilities for saving and loading learner state.

Provides ``save_checkpoint`` and ``load_checkpoint`` for persisting any
learner state (``LearnerState``, ``MLPLearnerState``, ``MultiHeadMLPState``,
``TDLearnerState``) to disk. Uses NumPy ``.npz`` for array data and JSON
for metadata and structural validation.

The caller provides a *template* state (from ``learner.init()``) to
``load_checkpoint`` so the treedef is known at load time. This avoids
serializing JAX treedefs directly, which is not supported.

Examples
--------
```python
import jax.random as jr
from alberta_framework import MultiHeadMLPLearner, save_checkpoint, load_checkpoint

learner = MultiHeadMLPLearner(n_heads=5, hidden_sizes=(64, 64))
state = learner.init(feature_dim=20, key=jr.key(42))

# Save
save_checkpoint(state, "agent.ckpt", metadata={"epoch": 1})

# Load (template provides treedef)
template = learner.init(feature_dim=20, key=jr.key(0))
loaded_state, meta = load_checkpoint(template, "agent.ckpt")
assert meta["epoch"] == 1
```
"""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Format version for future compatibility
_FORMAT_VERSION = 1


def save_checkpoint(
    state: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save learner state to disk.

    Flattens the state PyTree, saves array leaves to ``.npz``, and writes
    structural info + user metadata to a companion ``.json`` file.

    Args:
        state: Any learner state (LearnerState, MLPLearnerState,
            MultiHeadMLPState, TDLearnerState)
        path: Base path for the checkpoint (without extension).
            Creates ``{path}.npz`` and ``{path}.json``.
        metadata: Optional user metadata dict to store alongside
            the checkpoint (e.g. epoch, learner config, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    leaves, treedef = jax.tree.flatten(state)

    # Convert JAX arrays to numpy for saving
    np_leaves: dict[str, np.ndarray] = {}
    leaf_info: list[dict[str, Any]] = []
    for i, leaf in enumerate(leaves):
        key = f"leaf_{i}"
        if isinstance(leaf, (jax.Array, jnp.ndarray)):
            arr = np.asarray(leaf)
            np_leaves[key] = arr
            leaf_info.append({
                "key": key,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "type": "array",
            })
        elif isinstance(leaf, (int, float, bool, np.integer, np.floating)):
            # Scalar Python/numpy types stored as 0-d arrays
            arr = np.asarray(leaf)
            np_leaves[key] = arr
            leaf_info.append({
                "key": key,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "type": "scalar",
                "python_type": type(leaf).__name__,
            })
        else:
            # None or other non-array leaf
            leaf_info.append({
                "key": key,
                "type": "none" if leaf is None else "unknown",
                "value": None if leaf is None else repr(leaf),
            })

    # Save arrays
    np.savez(str(path.with_suffix(".npz")), **np_leaves)  # type: ignore[arg-type]

    # Save metadata JSON
    meta_dict: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "leaf_count": len(leaves),
        "leaves": leaf_info,
    }
    if metadata is not None:
        meta_dict["user_metadata"] = metadata

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta_dict, f, indent=2)


def load_checkpoint(
    state_template: Any,
    path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load checkpoint into a state matching the template's tree structure.

    The template state (from ``learner.init()``) provides the PyTree
    structure. Saved array leaves are loaded from ``.npz`` and
    unflattened using the template's treedef.

    Args:
        state_template: A state of the same type and structure as the
            saved state. Typically created via ``learner.init()`` with
            the same architecture.
        path: Base path for the checkpoint (without extension).
            Reads ``{path}.npz`` and ``{path}.json``.

    Returns:
        Tuple of ``(loaded_state, user_metadata)`` where ``user_metadata``
        is the dict passed to ``save_checkpoint``, or an empty dict if
        none was provided.

    Raises:
        FileNotFoundError: If checkpoint files don't exist
        ValueError: If leaf count doesn't match template
    """
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Checkpoint array file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata file not found: {json_path}")

    # Load metadata
    with open(json_path) as f:
        meta_dict = json.load(f)

    # Flatten template to get treedef
    template_leaves, treedef = jax.tree.flatten(state_template)
    saved_leaf_count = meta_dict["leaf_count"]

    if len(template_leaves) != saved_leaf_count:
        raise ValueError(
            f"Leaf count mismatch: template has {len(template_leaves)} leaves, "
            f"checkpoint has {saved_leaf_count} leaves. "
            f"Ensure the learner architecture matches the saved checkpoint."
        )

    # Load arrays
    npz_data = np.load(npz_path, allow_pickle=False)
    leaf_info = meta_dict["leaves"]

    loaded_leaves: list[Any] = []
    for i, info in enumerate(leaf_info):
        if info["type"] == "none":
            loaded_leaves.append(None)
        elif info["type"] in ("array", "scalar"):
            key = info["key"]
            arr = npz_data[key]
            if info["type"] == "scalar" and info.get("python_type") == "float":
                # Restore as Python float (for birth_timestamp, uptime_s)
                loaded_leaves.append(float(arr))
            elif info["type"] == "scalar" and info.get("python_type") == "int":
                loaded_leaves.append(int(arr))
            elif info["type"] == "scalar" and info.get("python_type") == "bool":
                loaded_leaves.append(bool(arr))
            else:
                loaded_leaves.append(jnp.array(arr))
        else:
            # Unknown type — use template leaf as fallback
            loaded_leaves.append(template_leaves[i])

    state = jax.tree.unflatten(treedef, loaded_leaves)
    user_metadata = meta_dict.get("user_metadata", {})

    return state, user_metadata
