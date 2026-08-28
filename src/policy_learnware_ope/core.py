"""Small, NumPy-native contracts shared by the v0.4b OPE estimators.

The module is intentionally strict at the data and actor boundaries.  In
particular, a row number created *after* transition subsampling is not accepted
as a native timestep, dataset cuts are not silently converted to terminals,
and every policy action query carries explicit per-row keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


class DataValidationError(ValueError):
    """Raised when an OPE data contract cannot be established safely."""

    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class PolicySemantics(str, Enum):
    """Semantics of one candidate policy exposed by an actor provider."""

    DETERMINISTIC = "deterministic"
    STOCHASTIC_KEYED = "stochastic_keyed"


class EstimateStatus(str, Enum):
    """Machine-readable outcome for an estimator/candidate cell."""

    PASS = "PASS"
    NO_GO_EXISTING_LOG_DENSITY = "NO_GO_EXISTING_LOG_DENSITY"
    NO_GO_TARGET_POLICY_SEMANTICS = "NO_GO_TARGET_POLICY_SEMANTICS"
    NO_GO_BEHAVIOR_SUPPORT = "NO_GO_BEHAVIOR_SUPPORT"
    NO_GO_MISSING_NEXT_BEHAVIOR_ACTION = "NO_GO_MISSING_NEXT_BEHAVIOR_ACTION"
    AMBIGUOUS_TERMINATION = "AMBIGUOUS_TERMINATION"
    INVALID_DATA = "INVALID_DATA"
    FAILED = "FAILED"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class ValueEstimate:
    """One raw finite-horizon value estimate and its audit evidence."""

    method_id: str
    status: EstimateStatus | str
    value: float | None
    support: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must be non-empty")
        try:
            status = EstimateStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"unknown estimate status: {self.status!r}") from exc
        if status is EstimateStatus.PASS:
            if self.value is None or not np.isfinite(float(self.value)):
                raise ValueError("PASS requires a finite value")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError("a failed/no-go estimate must not publish a value")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "support", _frozen_mapping(self.support))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))
        object.__setattr__(self, "cost", _frozen_mapping(self.cost))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "status": self.status.value,
            "value": self.value,
            "support": _json_safe(self.support),
            "provenance": _json_safe(self.provenance),
            "cost": _json_safe(self.cost),
            "diagnostics": _json_safe(self.diagnostics),
        }


def _float_matrix(value: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] == 0:
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _float_vector(value: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _integer_vector(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} must be a one-dimensional integer array")
    array = np.array(raw, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _bool_vector(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} must be one-dimensional")
    if raw.dtype.kind == "b":
        array = raw.astype(bool, copy=True)
    elif raw.dtype.kind in "iu" and np.all((raw == 0) | (raw == 1)):
        array = raw.astype(bool)
    else:
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} must contain booleans only")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Rewarded transitions with an auditable native-time reconstruction.

    ``episode_offsets`` index contiguous episode groups in this batch.  Rows may
    be a native-index subset; in that case ``native_timestep`` must retain the
    index reconstructed *before* sampling and ``timestep_provenance`` must be
    ``"native_indices"``.  ``"episode_offsets"`` is reserved for unsampled,
    full episode groups and is checked against ``arange``.  Values such as
    ``"sample_ordinal"`` are deliberately rejected.

    ``truncation_reason`` is required for every true ``truncated`` row.  The
    accepted reasons are ``"horizon"``, ``"environment"``, and
    ``"dataset_cut"``; the last remains bootstrap-able.
    """

    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dataset_cut: np.ndarray
    native_timestep: np.ndarray
    episode_id: np.ndarray
    episode_offsets: np.ndarray
    timestep_provenance: str
    next_behavior_action: np.ndarray | None = None
    truncation_reason: np.ndarray | None = None
    source_digest: str | None = None

    def __post_init__(self) -> None:
        observation = _float_matrix(self.observation, "observation")
        action = _float_matrix(self.action, "action")
        reward = _float_vector(self.reward, "reward")
        next_observation = _float_matrix(self.next_observation, "next_observation")
        terminated = _bool_vector(self.terminated, "terminated")
        truncated = _bool_vector(self.truncated, "truncated")
        dataset_cut = _bool_vector(self.dataset_cut, "dataset_cut")
        native_timestep = _integer_vector(self.native_timestep, "native_timestep")
        episode_id = _integer_vector(self.episode_id, "episode_id")
        episode_offsets = _integer_vector(self.episode_offsets, "episode_offsets")

        n_rows = len(reward)
        if n_rows == 0:
            raise DataValidationError(EstimateStatus.INVALID_DATA.value, "transition batch is empty")
        row_arrays = {
            "observation": observation,
            "action": action,
            "next_observation": next_observation,
            "terminated": terminated,
            "truncated": truncated,
            "dataset_cut": dataset_cut,
            "native_timestep": native_timestep,
            "episode_id": episode_id,
        }
        for name, array in row_arrays.items():
            if len(array) != n_rows:
                raise DataValidationError(EstimateStatus.INVALID_DATA.value, f"{name} row count disagrees with reward")
        if observation.shape != next_observation.shape:
            raise DataValidationError(EstimateStatus.INVALID_DATA.value, "observation and next_observation shapes differ")
        if np.any(native_timestep < 0):
            raise DataValidationError(EstimateStatus.INVALID_DATA.value, "native_timestep must be non-negative")

        if len(episode_offsets) < 2 or episode_offsets[0] != 0 or episode_offsets[-1] != n_rows:
            raise DataValidationError(
                EstimateStatus.INVALID_DATA.value,
                "episode_offsets must start at zero and end at the transition count",
            )
        if np.any(np.diff(episode_offsets) <= 0):
            raise DataValidationError(EstimateStatus.INVALID_DATA.value, "episode_offsets must be strictly increasing")

        provenance = str(self.timestep_provenance)
        if provenance not in {"episode_offsets", "native_indices"}:
            raise DataValidationError(
                EstimateStatus.INVALID_DATA.value,
                "native timestep provenance must be episode_offsets or native_indices; sampled row ordinals are forbidden",
            )
        seen_ids: set[int] = set()
        for start, stop in zip(episode_offsets[:-1], episode_offsets[1:]):
            ids = episode_id[start:stop]
            if np.any(ids != ids[0]) or int(ids[0]) in seen_ids:
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "episode_id must form exactly one contiguous group per episode offset",
                )
            seen_ids.add(int(ids[0]))
            times = native_timestep[start:stop]
            if np.any(np.diff(times) <= 0):
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "native timesteps must be strictly increasing within each episode",
                )
            if provenance == "episode_offsets" and not np.array_equal(times, np.arange(stop - start)):
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "episode_offsets provenance requires full episodes with native t=0..length-1",
                )

        reasons: np.ndarray
        if self.truncation_reason is None:
            if np.any(truncated):
                raise DataValidationError(
                    EstimateStatus.AMBIGUOUS_TERMINATION.value,
                    "truncated rows require an explicit truncation_reason",
                )
            reasons = np.full(n_rows, "none", dtype="U11")
        else:
            raw_reasons = np.asarray(self.truncation_reason)
            if raw_reasons.ndim != 1 or len(raw_reasons) != n_rows:
                raise DataValidationError(EstimateStatus.INVALID_DATA.value, "truncation_reason must match transition rows")
            reasons = raw_reasons.astype("U16")
            allowed = np.isin(reasons, ["none", "horizon", "environment", "dataset_cut"])
            if not np.all(allowed):
                raise DataValidationError(EstimateStatus.AMBIGUOUS_TERMINATION.value, "unknown truncation reason")
            if np.any(truncated != (reasons != "none")):
                raise DataValidationError(
                    EstimateStatus.AMBIGUOUS_TERMINATION.value,
                    "truncated and truncation_reason disagree",
                )
        if np.any(terminated & truncated):
            raise DataValidationError(EstimateStatus.AMBIGUOUS_TERMINATION.value, "a row cannot be both terminated and truncated")
        # A plain dataset_cut flag need not also be a Gym-style truncation.
        # When both flags are true, however, the reason must identify the cut;
        # conversely that reason is invalid unless both flags are present.
        if np.any((reasons == "dataset_cut") != (truncated & dataset_cut)):
            raise DataValidationError(
                EstimateStatus.AMBIGUOUS_TERMINATION.value,
                "dataset_cut truncation reason must exactly match truncated & dataset_cut",
            )
        if np.any(dataset_cut & terminated):
            raise DataValidationError(
                EstimateStatus.AMBIGUOUS_TERMINATION.value,
                "artificial dataset cuts must not be labelled native termination",
            )
        for start, stop in zip(episode_offsets[:-1], episode_offsets[1:]):
            physical_stop = terminated[start:stop] | (reasons[start:stop] == "environment")
            if np.any(physical_stop[:-1]):
                raise DataValidationError(
                    EstimateStatus.AMBIGUOUS_TERMINATION.value,
                    "native termination/environment truncation must end its episode offset group",
                )

        next_behavior_action = None
        if self.next_behavior_action is not None:
            next_behavior_action = _float_matrix(self.next_behavior_action, "next_behavior_action")
            if next_behavior_action.shape != action.shape:
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "next_behavior_action must match action shape",
                )
        if self.source_digest is not None and not str(self.source_digest):
            raise DataValidationError(EstimateStatus.INVALID_DATA.value, "source_digest cannot be empty")

        reasons.setflags(write=False)
        for name, value in {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "terminated": terminated,
            "truncated": truncated,
            "dataset_cut": dataset_cut,
            "native_timestep": native_timestep,
            "episode_id": episode_id,
            "episode_offsets": episode_offsets,
            "timestep_provenance": provenance,
            "next_behavior_action": next_behavior_action,
            "truncation_reason": reasons,
        }.items():
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.reward)

    @property
    def episode_count(self) -> int:
        return len(self.episode_offsets) - 1

    def bootstrap_mask(self, horizon: int) -> np.ndarray:
        """Return the finite-horizon Bellman mask without terminalizing cuts."""

        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        horizon = int(horizon)
        if np.any(self.native_timestep >= horizon):
            raise DataValidationError(
                EstimateStatus.INVALID_DATA.value,
                "native_timestep lies outside the estimator horizon",
            )
        horizon_reason = self.truncation_reason == "horizon"
        if np.any(horizon_reason & (self.native_timestep != horizon - 1)):
            raise DataValidationError(
                EstimateStatus.AMBIGUOUS_TERMINATION.value,
                "horizon truncation does not occur at native t=H-1",
            )
        no_bootstrap = (
            self.terminated
            | (self.native_timestep == horizon - 1)
            | (self.truncation_reason == "environment")
            | horizon_reason
        )
        # dataset_cut (including a truncation explicitly identified as such)
        # intentionally does not enter no_bootstrap.
        mask = (~no_bootstrap).astype(np.float64)
        mask.setflags(write=False)
        return mask


@runtime_checkable
class CandidateActionProvider(Protocol):
    """A bound candidate actor with explicit policy and PRNG semantics."""

    policy_id: str
    semantics: PolicySemantics | str

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        """Sample one action per row using the supplied keys (never hidden RNG)."""


@runtime_checkable
class BehaviorDensityProvider(Protocol):
    """Conditional behavior density that supports arbitrary action queries."""

    density_id: str
    exact: bool

    def log_prob(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        native_timestep: np.ndarray,
    ) -> np.ndarray:
        """Return log mu(a|s,t) for every supplied, potentially new action."""


def policy_semantics(provider: CandidateActionProvider) -> PolicySemantics:
    try:
        return PolicySemantics(provider.semantics)
    except (AttributeError, ValueError) as exc:
        raise DataValidationError(
            EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
            "candidate provider must declare deterministic or stochastic_keyed semantics",
        ) from exc


def policy_id(provider: CandidateActionProvider) -> str:
    value = str(getattr(provider, "policy_id", ""))
    if not value:
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, "candidate provider lacks policy_id")
    return value


def validate_action_keys(keys: Any, n_rows: int) -> np.ndarray:
    """Validate scalar seeds or structured integer keys for a row batch."""

    raw = np.asarray(keys)
    if raw.ndim not in {1, 2} or len(raw) != n_rows or raw.dtype.kind not in "iu":
        raise DataValidationError(
            EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
            "every action query requires one explicit integer key/seed per row",
        )
    if raw.ndim == 2 and raw.shape[1] == 0:
        raise DataValidationError(EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value, "structured action keys cannot be empty")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise DataValidationError(EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value, "action keys/seeds must be non-negative")
    array = np.array(raw, dtype=np.uint64, copy=True)
    array.setflags(write=False)
    return array


def candidate_actions(
    provider: CandidateActionProvider,
    observations: np.ndarray,
    native_timestep: np.ndarray,
    *,
    keys: np.ndarray,
    require_deterministic: bool,
) -> np.ndarray:
    """Query a candidate actor while enforcing semantics, keys, and shape."""

    obs = _float_matrix(observations, "candidate observations")
    times = _integer_vector(native_timestep, "candidate native_timestep")
    if len(times) != len(obs):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, "candidate timestep count differs from observations")
    checked_keys = validate_action_keys(keys, len(obs))
    semantics = policy_semantics(provider)
    if require_deterministic and semantics is not PolicySemantics.DETERMINISTIC:
        raise DataValidationError(
            EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
            "this estimator only defines deterministic evaluation-policy targets",
        )
    try:
        action = provider.sample_actions(obs, times, keys=checked_keys)
    except TypeError as exc:
        raise DataValidationError(
            EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
            "candidate action provider must accept explicit keys",
        ) from exc
    result = _float_matrix(action, "candidate action")
    if len(result) != len(obs):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, "candidate action row count differs from observations")
    return result


def behavior_log_prob(
    provider: BehaviorDensityProvider,
    observations: np.ndarray,
    actions: np.ndarray,
    native_timestep: np.ndarray,
) -> np.ndarray:
    """Evaluate an exact conditional density without accepting silent clipping."""

    if not bool(getattr(provider, "exact", False)):
        raise DataValidationError(
            EstimateStatus.NO_GO_EXISTING_LOG_DENSITY.value,
            "behavior density is absent or not exact for the logged distribution",
        )
    obs = _float_matrix(observations, "density observations")
    act = _float_matrix(actions, "density actions")
    times = _integer_vector(native_timestep, "density native_timestep")
    if len(obs) != len(act) or len(obs) != len(times):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, "density query rows disagree")
    raw = np.asarray(provider.log_prob(obs, act, times), dtype=np.float64)
    if raw.shape != (len(obs),):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, "behavior log_prob must return one scalar per row")
    # +/- infinity is preserved here: callers distinguish impossible support
    # from a malformed (NaN) density calculation.
    if np.any(np.isnan(raw)):
        raise DataValidationError(EstimateStatus.INVALID_DATA.value, "behavior log_prob returned NaN")
    raw.setflags(write=False)
    return raw


__all__ = [
    "BehaviorDensityProvider",
    "CandidateActionProvider",
    "DataValidationError",
    "EstimateStatus",
    "PolicySemantics",
    "TransitionBatch",
    "ValueEstimate",
    "behavior_log_prob",
    "candidate_actions",
    "policy_id",
    "policy_semantics",
    "validate_action_keys",
]
