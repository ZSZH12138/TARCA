"""Deterministic synthetic regime-switching data foundations."""

from .regimes import (
    RANDOM_STREAM_NAMES,
    RandomStream,
    build_regime_parameter_schedule,
    compute_stationary_distribution,
    make_unseen_parameter_shift,
    regime_persistence_statistics,
    resolve_regime_parameters,
    sample_regime_sequence,
    spawn_random_streams,
    validate_transition_matrix,
)

__all__ = (
    "RANDOM_STREAM_NAMES",
    "RandomStream",
    "build_regime_parameter_schedule",
    "compute_stationary_distribution",
    "make_unseen_parameter_shift",
    "regime_persistence_statistics",
    "resolve_regime_parameters",
    "sample_regime_sequence",
    "spawn_random_streams",
    "validate_transition_matrix",
)
