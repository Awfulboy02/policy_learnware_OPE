"""Finite-horizon NumPy FQE and kernel/importance-weighted FQE.

These implementations are deliberately compact reference estimators for the
companion repository and its synthetic acceptance tests.  They are real fitted
Bellman estimators, not placeholder score generators.  Both critics consume
``native_timestep / H`` explicitly and share the strict mask contract from
``TransitionBatch``.
"""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np

from .core import (
    BehaviorDensityProvider,
    CandidateActionProvider,
    DataValidationError,
    EstimateStatus,
    PolicySemantics,
    TransitionBatch,
    ValueEstimate,
    behavior_log_prob,
    candidate_actions,
    finite_horizon_method_id,
    finite_horizon_value_convention,
    policy_id,
    policy_semantics,
    validate_action_keys,
)
from .kmifqe import B20KMIFQETrainer, seed_from_keys


FH_FQE_METHOD_ID = "FH_FQE_G099_H1000"
FH_KMIFQE_METHOD_ID = "FH_KMIFQE_G099_H1000"


def _key_digest(keys: np.ndarray) -> str:
    return sha256(np.asarray(keys, dtype="<u8").tobytes()).hexdigest()


_ACTION_KEY_SCHEDULE = "sha256_domain_xor_u64_v1"


def _derived_action_key_schedule(
    keys: np.ndarray,
    *,
    sample_count: int,
    domain: str,
) -> np.ndarray:
    """Derive auditable common-random-number keys for one MC query panel."""

    base = np.asarray(keys, dtype=np.uint64)
    schedule = np.empty((sample_count, *base.shape), dtype=np.uint64)
    column_count = 1 if base.ndim == 1 else base.shape[1]
    for sample_index in range(sample_count):
        derived = np.array(base, copy=True)
        for column_index in range(column_count):
            token_bytes = sha256(
                (
                    f"{_ACTION_KEY_SCHEDULE}\0{domain}\0"
                    f"{sample_index}\0{column_index}"
                ).encode("utf-8")
            ).digest()[:8]
            token = np.frombuffer(token_bytes, dtype="<u8")[0]
            if derived.ndim == 1:
                derived ^= token
            else:
                # Structured JAX key words are uint32 values carried in the
                # common uint64 validation container.  Preserve that ABI.
                derived[:, column_index] ^= token & np.uint64(0xFFFFFFFF)
        schedule[sample_index] = derived
    schedule.setflags(write=False)
    return schedule


def _action_schedule_digest(domain: str, schedule: np.ndarray) -> str:
    digest = sha256()
    digest.update(b"fqe_action_key_schedule_digest_v1\0")
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(schedule.shape, dtype="<u8").tobytes())
    digest.update(np.asarray(schedule, dtype="<u8").tobytes())
    return digest.hexdigest()


def _aggregate_action_schedule_digests(**parts: str) -> str:
    digest = sha256()
    digest.update(b"fqe_action_key_schedule_aggregate_v1\0")
    for name, value in sorted(parts.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"=")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


class _QuadraticTimeFeatures:
    """Small ridge feature map with explicit state/action-time interactions."""

    def __init__(self, observation: np.ndarray, action: np.ndarray) -> None:
        self.observation_mean = np.mean(observation, axis=0)
        self.observation_scale = np.std(observation, axis=0)
        self.observation_scale[self.observation_scale < 1e-8] = 1.0
        self.action_mean = np.mean(action, axis=0)
        self.action_scale = np.std(action, axis=0)
        self.action_scale[self.action_scale < 1e-8] = 1.0

    def transform(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        normalized_time: np.ndarray,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        time = np.asarray(normalized_time, dtype=np.float64).reshape(-1, 1)
        if observation.ndim != 2 or observation.shape[1] != len(self.observation_mean):
            raise ValueError("observation shape differs from fitted critic ABI")
        if action.ndim != 2 or action.shape[1] != len(self.action_mean):
            raise ValueError("action shape differs from fitted critic ABI")
        if len(observation) != len(action) or len(observation) != len(time):
            raise ValueError("critic feature row counts disagree")
        state = (observation - self.observation_mean) / self.observation_scale
        act = (action - self.action_mean) / self.action_scale
        # This stays O(d) instead of creating a large all-pairs polynomial map.
        # Time enters directly and through interactions, so different remaining
        # horizons cannot collapse into an unconditioned Q(s,a).
        return np.concatenate(
            [
                np.ones((len(state), 1), dtype=np.float64),
                state,
                act,
                time,
                np.square(state),
                np.square(act),
                np.square(time),
                state * time,
                act * time,
            ],
            axis=1,
        )


class FiniteHorizonFQE:
    """Time-conditioned fitted Q evaluation with a NumPy ridge critic."""

    method_id = FH_FQE_METHOD_ID
    method_family = "FH_FQE"

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        ridge: float = 1e-6,
        max_iterations: int | None = None,
        tolerance: float = 1e-9,
        stochastic_action_samples: int = 8,
    ) -> None:
        if (
            isinstance(gamma, (bool, np.bool_))
            or not isinstance(gamma, (int, float, np.integer, np.floating))
            or not np.isfinite(float(gamma))
            or not 0.0 <= float(gamma) <= 1.0
        ):
            raise ValueError("gamma must lie in [0, 1]")
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        if float(ridge) <= 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and positive")
        if float(tolerance) <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive")
        if (
            isinstance(stochastic_action_samples, (bool, np.bool_))
            or int(stochastic_action_samples) != stochastic_action_samples
            or int(stochastic_action_samples) < 2
        ):
            raise ValueError("stochastic_action_samples must be an integer >= 2")
        if max_iterations is None:
            if 0.0 < float(gamma) < 1.0:
                contraction_iterations = int(
                    np.ceil(np.log(float(tolerance)) / np.log(float(gamma)))
                ) + 1
            else:
                contraction_iterations = int(horizon) + 1
            max_iterations = max(int(horizon) + 1, contraction_iterations, 50)
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.method_id = finite_horizon_method_id(
            self.method_family, self.gamma, self.horizon
        )
        self.ridge = float(ridge)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.stochastic_action_samples = int(stochastic_action_samples)
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self._features: _QuadraticTimeFeatures | None = None
        self._coefficient: np.ndarray | None = None
        self._candidate: CandidateActionProvider | None = None
        self._candidate_id: str | None = None
        self._candidate_semantics: PolicySemantics | None = None
        self._gate: tuple[EstimateStatus, str] | None = None
        self._support: dict[str, Any] = {}
        self._provenance: dict[str, Any] = {}
        self._cost: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {}
        self._fitted = False

    def fit(
        self,
        batch: TransitionBatch,
        candidate: CandidateActionProvider,
        *,
        fit_keys: np.ndarray,
    ) -> "FiniteHorizonFQE":
        """Fit one candidate-specific critic and return ``self``."""

        self._reset_fit_state()
        start = perf_counter()
        self._candidate = candidate
        try:
            if not isinstance(batch, TransitionBatch):
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "FQE requires core.TransitionBatch so native time and masks cannot bypass validation",
                )
            self._candidate_id = policy_id(candidate)
            keys = validate_action_keys(fit_keys, len(batch))
            semantics = policy_semantics(candidate)
            self._candidate_semantics = semantics
            self._base_provenance(batch, semantics)
            self._provenance["fit_key_digest"] = _key_digest(keys)
            mask = batch.bootstrap_mask(self.horizon)
            next_actions, next_schedule_digest = self._query_next_actions(
                batch, candidate, keys, mask, semantics
            )
            support_schedule_digest = self._measure_action_support(
                batch, candidate, keys, semantics
            )
            fit_schedule_digest = _aggregate_action_schedule_digests(
                next=next_schedule_digest,
                support=support_schedule_digest,
            )
            self._provenance.update(
                {
                    "fit_next_action_key_schedule_digest": next_schedule_digest,
                    "fit_support_action_key_schedule_digest": support_schedule_digest,
                    "fit_action_key_schedule_digest": fit_schedule_digest,
                }
            )
            self._diagnostics["fit_action_key_schedule_digest"] = fit_schedule_digest
            self._fit_ridge(
                batch,
                next_actions,
                mask,
                np.ones(len(batch), dtype=np.float64),
            )
        except DataValidationError as exc:
            try:
                status = EstimateStatus(exc.status)
            except ValueError:
                status = EstimateStatus.INVALID_DATA
            self._close_gate(status, exc.detail)
        finally:
            self._cost["fit_seconds"] = float(perf_counter() - start)
            self._cost["fit_transitions"] = int(len(batch)) if isinstance(batch, TransitionBatch) else 0
        return self

    def _base_provenance(self, batch: TransitionBatch, semantics: PolicySemantics) -> None:
        self._provenance = {
            "method_id": self.method_id,
            "implementation": "numpy_quadratic_time_ridge_fqe",
            "implementation_scope": "EXECUTABLE_SYNTHETIC_REFERENCE",
            "upstream_code_parity": "NOT_CLAIMED",
            "planned_neural_critic_port": "PENDING_REAL_ASSET_PHASE",
            "adaptation": "FINITE_HORIZON_PROTOCOL_ADAPTATION",
            "gamma": self.gamma,
            "horizon": self.horizon,
            "value_convention": finite_horizon_value_convention(
                self.gamma, self.horizon
            ),
            "time_input": "native_timestep/H",
            "target_time": "(native_timestep+1)/H",
            "mask_contract": {
                "native_termination": "no_bootstrap",
                "horizon_t=H-1": "no_bootstrap",
                "dataset_cut": "bootstrap",
                "ambiguous_truncation": "fail_closed",
            },
            "timestep_provenance": batch.timestep_provenance,
            "source_digest": batch.source_digest,
            "candidate_id": self._candidate_id,
            "candidate_semantics": semantics.value,
            "policy_expectation": (
                "EXACT_DETERMINISTIC"
                if semantics is PolicySemantics.DETERMINISTIC
                else "STOCHASTIC_KEYED_MONTE_CARLO_EXPECTATION"
            ),
            "action_expectation_samples": (
                1
                if semantics is PolicySemantics.DETERMINISTIC
                else self.stochastic_action_samples
            ),
            "action_key_schedule": (
                "caller_keys_exact"
                if semantics is PolicySemantics.DETERMINISTIC
                else _ACTION_KEY_SCHEDULE
            ),
        }

    def _close_gate(self, status: EstimateStatus, detail: str) -> None:
        self._gate = (status, str(detail))
        self._diagnostics["gate_detail"] = str(detail)
        self._fitted = True

    def _query_next_actions(
        self,
        batch: TransitionBatch,
        candidate: CandidateActionProvider,
        keys: np.ndarray,
        mask: np.ndarray,
        semantics: PolicySemantics,
    ) -> tuple[tuple[np.ndarray, ...], str]:
        active = mask.astype(bool)
        active_keys = keys[active]
        if semantics is PolicySemantics.DETERMINISTIC:
            schedule = np.expand_dims(active_keys, axis=0)
        else:
            schedule = _derived_action_key_schedule(
                active_keys,
                sample_count=self.stochastic_action_samples,
                domain="fit_next_action",
            )
        results: list[np.ndarray] = []
        for sample_keys in schedule:
            result = np.zeros_like(batch.action, dtype=np.float64)
            if np.any(active):
                queried = candidate_actions(
                    candidate,
                    batch.next_observation[active],
                    batch.native_timestep[active] + 1,
                    keys=sample_keys,
                    require_deterministic=(
                        semantics is PolicySemantics.DETERMINISTIC
                    ),
                )
                if queried.shape[1] != batch.action.shape[1]:
                    raise DataValidationError(
                        EstimateStatus.INVALID_DATA.value,
                        "candidate action width differs from logged action ABI",
                    )
                result[active] = queried
            results.append(result)
        sample_count = len(results)
        self._diagnostics.update(
            {
                "bellman_action_samples": sample_count,
                "bellman_action_query_count": (
                    sample_count if np.any(active) else 0
                ),
                "bellman_action_query_rows": int(sample_count * np.sum(active)),
            }
        )
        self._cost.update(
            {
                "bellman_actor_query_calls": (
                    sample_count if np.any(active) else 0
                ),
                "bellman_actor_query_rows": int(sample_count * np.sum(active)),
            }
        )
        return tuple(results), _action_schedule_digest("fit_next_action", schedule)

    def _fit_ridge(
        self,
        batch: TransitionBatch,
        next_actions: tuple[np.ndarray, ...],
        mask: np.ndarray,
        sample_weight: np.ndarray,
    ) -> None:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != (len(batch),) or not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise DataValidationError(EstimateStatus.INVALID_DATA.value, "ridge weights must be finite and positive")
        weight = weight / np.mean(weight)
        self._features = _QuadraticTimeFeatures(batch.observation, batch.action)
        current_x = self._features.transform(
            batch.observation,
            batch.action,
            batch.native_timestep / self.horizon,
        )
        if len(next_actions) == 1:
            next_x = self._features.transform(
                batch.next_observation,
                next_actions[0],
                (batch.native_timestep + 1) / self.horizon,
            )
        else:
            next_x = np.zeros_like(current_x)
            for next_action in next_actions:
                next_x += self._features.transform(
                    batch.next_observation,
                    next_action,
                    (batch.native_timestep + 1) / self.horizon,
                )
            next_x /= len(next_actions)
        normalizer = float(np.sum(weight))
        gram = (current_x.T @ (weight[:, None] * current_x)) / normalizer
        gram += self.ridge * np.eye(gram.shape[0], dtype=np.float64)
        try:
            inverse_gram = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            inverse_gram = np.linalg.pinv(gram, rcond=1e-12)
        coefficient = np.zeros(current_x.shape[1], dtype=np.float64)
        converged = False
        change = np.inf
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            target = batch.reward + self.gamma * mask * (next_x @ coefficient)
            rhs = (current_x.T @ (weight * target)) / normalizer
            updated = inverse_gram @ rhs
            if not np.all(np.isfinite(updated)) or np.linalg.norm(updated) > 1e14:
                raise DataValidationError(EstimateStatus.FAILED.value, "ridge FQE iteration diverged")
            change = float(np.max(np.abs(updated - coefficient)))
            coefficient = updated
            if change <= self.tolerance:
                converged = True
                break
        self._coefficient = coefficient
        prediction = current_x @ coefficient
        next_prediction = next_x @ coefficient
        residual = batch.reward + self.gamma * mask * next_prediction - prediction
        self._diagnostics.update(
            {
                "iterations": iterations,
                "converged": converged,
                "coefficient_delta": change,
                "bellman_residual_rmse": float(np.sqrt(np.average(np.square(residual), weights=weight))),
                "feature_dimension": int(current_x.shape[1]),
                "dataset_cut_rows": int(np.sum(batch.dataset_cut)),
                "bootstrap_rows": int(np.sum(mask)),
            }
        )
        self._cost.update(
            {
                "iterations": iterations,
                "feature_dimension": int(current_x.shape[1]),
                "linear_solves": 1,
            }
        )
        if not converged:
            self._diagnostics["failed_closed"] = True
            self._close_gate(
                EstimateStatus.NO_GO_FIT_CONVERGENCE,
                f"ridge FQE did not converge within {self.max_iterations} iterations",
            )
            return
        self._fitted = True

    def _measure_action_support(
        self,
        batch: TransitionBatch,
        candidate: CandidateActionProvider,
        keys: np.ndarray,
        semantics: PolicySemantics,
    ) -> str:
        if semantics is PolicySemantics.DETERMINISTIC:
            support_keys = keys ^ np.uint64(0xA5A5A5A5A5A5A5A5)
            schedule = np.expand_dims(support_keys, axis=0)
        else:
            schedule = _derived_action_key_schedule(
                keys,
                sample_count=self.stochastic_action_samples,
                domain="fit_support_action",
            )
        action_scale = np.std(batch.action, axis=0)
        action_scale[action_scale < 1e-6] = 1.0
        distances: list[np.ndarray] = []
        for sample_keys in schedule:
            current_action = candidate_actions(
                candidate,
                batch.observation,
                batch.native_timestep,
                keys=sample_keys,
                require_deterministic=(semantics is PolicySemantics.DETERMINISTIC),
            )
            if current_action.shape != batch.action.shape:
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "candidate action width differs from logged action ABI",
                )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                standardized_delta = (current_action - batch.action) / action_scale
                distance = np.linalg.norm(standardized_delta, axis=1)
            if not np.all(np.isfinite(standardized_delta)) or not np.all(
                np.isfinite(distance)
            ):
                raise DataValidationError(
                    EstimateStatus.FAILED.value,
                    "action-support distance became non-finite",
                )
            distances.append(distance)
        distance = np.concatenate(distances)
        schedule_digest = _action_schedule_digest("fit_support_action", schedule)
        self._support = {
            "kind": (
                "pointwise_logged_state_action_distance"
                if semantics is PolicySemantics.DETERMINISTIC
                else "monte_carlo_logged_state_action_distance"
            ),
            "mean_action_distance": float(np.mean(distance)),
            "p95_action_distance": float(np.quantile(distance, 0.95)),
            "max_action_distance": float(np.max(distance)),
            "support_rows": int(len(batch)),
            "support_action_draws": int(len(distance)),
            "action_expectation_samples": int(len(schedule)),
            "support_action_key_schedule_digest": schedule_digest,
        }
        self._diagnostics.update(
            {
                "support_action_samples": int(len(schedule)),
                "support_action_query_count": int(len(schedule)),
                "support_action_query_rows": int(len(distance)),
            }
        )
        self._cost.update(
            {
                "support_actor_query_calls": int(len(schedule)),
                "support_actor_query_rows": int(len(distance)),
            }
        )
        return schedule_digest

    def estimate(
        self,
        initial_observations: np.ndarray,
        *,
        keys: np.ndarray,
        initial_timestep: int | np.ndarray = 0,
    ) -> ValueEstimate:
        """Estimate the mean raw ``J_gamma,H`` over supplied initial states."""

        if not self._fitted:
            raise RuntimeError("fit must be called before estimate")
        start = perf_counter()
        observations = np.asarray(initial_observations, dtype=np.float64)
        if observations.ndim != 2 or len(observations) == 0 or not np.all(np.isfinite(observations)):
            raise ValueError("initial_observations must be a non-empty finite 2-D matrix")
        checked_keys = validate_action_keys(keys, len(observations))
        estimate_provenance = dict(self._provenance)
        estimate_provenance["estimate_key_digest"] = _key_digest(checked_keys)
        raw_times = np.asarray(initial_timestep)
        if raw_times.dtype.kind not in "iu" or raw_times.dtype.kind == "b":
            raise ValueError("initial_timestep must contain integers")
        if raw_times.ndim == 0:
            times = np.full(len(observations), int(raw_times), dtype=np.int64)
        elif raw_times.shape == (len(observations),):
            times = raw_times.astype(np.int64)
        else:
            raise ValueError("initial_timestep must be an integer scalar or one value per state")
        if np.any(times < 0) or np.any(times >= self.horizon):
            raise ValueError("initial_timestep lies outside the finite horizon")
        if self._gate is not None:
            status, detail = self._gate
            cost = dict(self._cost)
            cost["estimate_seconds"] = float(perf_counter() - start)
            diagnostics = dict(self._diagnostics)
            diagnostics["gate_detail"] = detail
            return ValueEstimate(
                method_id=self.method_id,
                status=status,
                value=None,
                support=self._support,
                provenance=estimate_provenance,
                cost=cost,
                diagnostics=diagnostics,
            )
        if self._candidate is None or self._features is None or self._coefficient is None:
            raise RuntimeError("fitted estimator state is incomplete")
        if self._candidate_semantics is PolicySemantics.DETERMINISTIC:
            schedule = np.expand_dims(checked_keys, axis=0)
        elif self._candidate_semantics is PolicySemantics.STOCHASTIC_KEYED:
            schedule = _derived_action_key_schedule(
                checked_keys,
                sample_count=self.stochastic_action_samples,
                domain="estimate_initial_action",
            )
        else:
            raise RuntimeError("fitted estimator lacks candidate policy semantics")
        sampled_values: list[np.ndarray] = []
        for sample_keys in schedule:
            actions = candidate_actions(
                self._candidate,
                observations,
                times,
                keys=sample_keys,
                require_deterministic=(
                    self._candidate_semantics is PolicySemantics.DETERMINISTIC
                ),
            )
            features = self._features.transform(
                observations, actions, times / self.horizon
            )
            sampled_values.append(features @ self._coefficient)
        value_panel = np.stack(sampled_values, axis=0)
        values = np.mean(value_panel, axis=0)
        estimate_schedule_digest = _action_schedule_digest(
            "estimate_initial_action", schedule
        )
        estimate_provenance.update(
            {
                "estimate_action_key_schedule_digest": estimate_schedule_digest,
                "estimate_action_samples": int(len(schedule)),
            }
        )
        estimate_seconds = float(perf_counter() - start)
        cost = dict(self._cost)
        cost["estimate_seconds"] = estimate_seconds
        cost["runtime_seconds"] = float(cost.get("fit_seconds", 0.0) + estimate_seconds)
        cost.update(
            {
                "estimate_actor_query_calls": int(len(schedule)),
                "estimate_actor_query_rows": int(len(schedule) * len(observations)),
                "actor_query_calls": int(
                    cost.get("bellman_actor_query_calls", 0)
                    + cost.get("support_actor_query_calls", 0)
                    + len(schedule)
                ),
                "actor_query_rows": int(
                    cost.get("bellman_actor_query_rows", 0)
                    + cost.get("support_actor_query_rows", 0)
                    + len(schedule) * len(observations)
                ),
            }
        )
        diagnostics = dict(self._diagnostics)
        diagnostics.update(
            {
                "initial_state_count": int(len(observations)),
                "initial_value_std": float(np.std(values)),
                "initial_action_samples": int(len(schedule)),
                "initial_action_query_count": int(len(schedule)),
                "initial_action_query_rows": int(len(schedule) * len(observations)),
                "initial_action_value_within_state_std_mean": float(
                    np.mean(np.std(value_panel, axis=0))
                ),
                "estimate_action_key_schedule_digest": estimate_schedule_digest,
            }
        )
        return ValueEstimate(
            method_id=self.method_id,
            status=EstimateStatus.PASS,
            value=float(np.mean(values)),
            support=self._support,
            provenance=estimate_provenance,
            cost=cost,
            diagnostics=diagnostics,
        )


class FiniteHorizonKMIFQE(FiniteHorizonFQE):
    """Finite-horizon B20 protocol adaptation with in-sample TD."""

    method_id = FH_KMIFQE_METHOD_ID
    method_family = "FH_KMIFQE"

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        ridge: float = 1e-6,
        max_iterations: int | None = None,
        tolerance: float = 3e-3,
        kernel_bandwidth: float | None = None,
        min_log_density: float = -50.0,
        min_ess_fraction: float = 0.01,
        critic_features: int = 32,
        eigenvalue_floor: float = 1e-6,
        metric_regularization: float = 0.1,
        bandwidth_floor: float = 1e-3,
        bandwidth_ceiling: float = 10.0,
        ratio_clip_min: float = 1e-3,
        ratio_clip_max: float = 2.0,
        target_density_floor: float = 1e-12,
        resample_size: int | None = None,
        target_update_interval: int = 1,
        critic_step_size: float = 0.1,
        probability_tolerance: float | None = 1.5e-2,
    ) -> None:
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            ridge=ridge,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if kernel_bandwidth is not None:
            raise ValueError(
                "fixed kernel_bandwidth is unavailable on the B20 protocol path; "
                "bandwidth is estimated from TD-MSE bias and variance"
            )
        if not np.isfinite(min_log_density):
            raise ValueError("min_log_density must be finite")
        if not 0.0 < float(min_ess_fraction) <= 1.0:
            raise ValueError("min_ess_fraction must lie in (0, 1]")
        self.min_log_density = float(min_log_density)
        self.min_ess_fraction = float(min_ess_fraction)
        # Validate the method-specific configuration once without retaining a
        # half-fitted trainer.  Every candidate gets a fresh trainer in fit().
        validated = B20KMIFQETrainer(
            gamma=self.gamma,
            horizon=self.horizon,
            ridge=self.ridge,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            hidden_features=critic_features,
            eigenvalue_floor=eigenvalue_floor,
            metric_regularization=metric_regularization,
            bandwidth_floor=bandwidth_floor,
            bandwidth_ceiling=bandwidth_ceiling,
            ratio_clip_min=ratio_clip_min,
            ratio_clip_max=ratio_clip_max,
            target_density_floor=target_density_floor,
            resample_size=resample_size,
            target_update_interval=target_update_interval,
            critic_step_size=critic_step_size,
            probability_tolerance=probability_tolerance,
        )
        self._kmifqe_config = {
            **validated.config,
            "min_log_density": self.min_log_density,
            "min_ess_fraction": self.min_ess_fraction,
        }
        self._trainer: B20KMIFQETrainer | None = None

    def fit(
        self,
        batch: TransitionBatch,
        candidate: CandidateActionProvider,
        *,
        behavior_density: BehaviorDensityProvider,
        fit_keys: np.ndarray,
    ) -> "FiniteHorizonKMIFQE":
        self._reset_fit_state()
        start = perf_counter()
        self._candidate = candidate
        try:
            if not isinstance(batch, TransitionBatch):
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "KMIFQE requires core.TransitionBatch so native time and masks cannot bypass validation",
                )
            self._candidate_id = policy_id(candidate)
            keys = validate_action_keys(fit_keys, len(batch))
            try:
                semantics = policy_semantics(candidate)
            except DataValidationError:
                semantics = PolicySemantics.STOCHASTIC_KEYED
            self._candidate_semantics = semantics
            self._base_provenance(batch, semantics)
            self._provenance["fit_key_digest"] = _key_digest(keys)
            self._provenance.pop("planned_neural_critic_port", None)
            self._provenance.update(
                {
                    "implementation": "numpy_nonlinear_B20_kernel_metric_in_sample_fqe",
                    "implementation_scope": "EXECUTABLE_METHOD_LEVEL_PROTOCOL_ADAPTATION",
                    "method_identity": "B20_PROTOCOL_ADAPTATION",
                    "scientific_role": "B20_PROTOCOL_ADAPTATION",
                    "official_parity": False,
                    "paper_benchmark_parity": False,
                    "official_code_commit": "070f121d29f05638221695690d5b0d1f0e2bf75b",
                    "faithfulness_scope": (
                        "B20 mechanisms on the project finite-horizon raw-value protocol"
                    ),
                    "density_id": str(getattr(behavior_density, "density_id", "")),
                    "density_exact": getattr(behavior_density, "exact", False) is True,
                    "kernel": "B20_local_Mahalanobis_Gaussian",
                    "metric_learning_scope": (
                        "candidate-specific target-Q action Hessian, eigen floor, det(A)=1"
                    ),
                    "bandwidth": "B20_Eq11_TD_MSE_bias_variance_estimator",
                    "importance_weight": "K(a_next_behavior,pi(s_next))/mu(a_next_behavior|s_next)",
                    "resampling": "replacement_with_probability_w_over_sum_w",
                    "td_bootstrap": "same_resampled_row_logged_adjacent_action",
                    "bias_correction": "mean_clipped_importance_weight",
                    "trainer_config": dict(self._kmifqe_config),
                    "remaining_drift": [
                        "fixed seeded tanh feature critic rather than official fully-trained 2x256 tanh network",
                        "damped analytic output-layer fit rather than official Adam critic update",
                        "seeded common-random replacement uniforms are remapped each iteration rather than drawing a fresh minibatch",
                        "one overall clipped kernel/mu ratio uses project range [1e-3,2] rather than the official wider per-dimension clipping semantics",
                        "finite-H native-time masks and raw J_gamma,H rather than normalized continuing value",
                        "dense per-row Hessian/metric panels are not production-scaled for OPS-DS million-row workloads",
                        "no MuJoCo/D4RL paper benchmark parity claim",
                    ],
                    "production_scalability_status": "NO_GO_OPS_DS_DENSE_HESSIAN_PANEL",
                }
            )
            if getattr(behavior_density, "exact", False) is not True:
                self._close_gate(
                    EstimateStatus.NO_GO_EXISTING_LOG_DENSITY,
                    "KMIFQE requires exact behavior density; clipped-Gaussian existing logs do not provide it",
                )
                return self
            if semantics is not PolicySemantics.DETERMINISTIC:
                self._provenance.update(
                    {
                        "policy_expectation": "NO_GO_UNSUPPORTED_STOCHASTIC_KEYED",
                        "action_expectation_samples": 0,
                    }
                )
                self._close_gate(
                    EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS,
                    "this KMIFQE adaptation requires a deterministic evaluation policy",
                )
                return self
            if batch.next_behavior_action is None:
                self._close_gate(
                    EstimateStatus.NO_GO_MISSING_NEXT_BEHAVIOR_ACTION,
                    "KMIFQE requires the original adjacent next behavior action, not the next sampled row",
                )
                return self
            mask = batch.bootstrap_mask(self.horizon)
            next_target_actions, _ = self._query_next_actions(
                batch, candidate, keys, mask, semantics
            )
            next_target_action = next_target_actions[0]
            self._measure_action_support(batch, candidate, keys, semantics)
            active = mask.astype(bool)
            if not np.any(active):
                self._close_gate(
                    EstimateStatus.NO_GO_BEHAVIOR_SUPPORT,
                    "KMIFQE requires at least one bootstrap-active transition",
                )
                return self
            next_times = batch.native_timestep[active] + 1
            logged_next_action = batch.next_behavior_action
            assert logged_next_action is not None
            row = np.arange(len(batch) - 1)
            contiguous = (
                active[:-1]
                & (batch.episode_id[:-1] == batch.episode_id[1:])
                & (batch.native_timestep[1:] == batch.native_timestep[:-1] + 1)
            )
            verified_rows = row[contiguous]
            verified_logged_action = logged_next_action[verified_rows]
            verified_successor_action = batch.action[verified_rows + 1]
            comparison_dtype = np.result_type(
                verified_logged_action.dtype, verified_successor_action.dtype
            )
            comparison_epsilon = float(np.finfo(comparison_dtype).eps)
            adjacency_rtol = 32.0 * comparison_epsilon
            reference_scale = max(
                1.0,
                float(np.max(np.abs(verified_successor_action)))
                if len(verified_rows)
                else 0.0,
            )
            adjacency_atol = adjacency_rtol * reference_scale
            absolute_drift = np.abs(
                verified_logged_action - verified_successor_action
            )
            max_abs_drift = (
                float(np.max(absolute_drift)) if len(verified_rows) else 0.0
            )
            relative_denominator = np.maximum(
                np.abs(verified_successor_action), adjacency_atol
            )
            max_rel_drift = (
                float(np.max(absolute_drift / relative_denominator))
                if len(verified_rows)
                else 0.0
            )
            self._diagnostics.update(
                {
                    "adjacent_action_comparison_dtype": str(comparison_dtype),
                    "adjacent_action_tolerance_rule": (
                        "allclose_atol=32*eps*max(1,max_abs_successor),rtol=32*eps"
                    ),
                    "adjacent_action_atol": adjacency_atol,
                    "adjacent_action_rtol": adjacency_rtol,
                    "adjacent_action_max_abs_drift": max_abs_drift,
                    "adjacent_action_max_rel_drift": max_rel_drift,
                }
            )
            if len(verified_rows) and not np.allclose(
                verified_logged_action,
                verified_successor_action,
                rtol=adjacency_rtol,
                atol=adjacency_atol,
                equal_nan=False,
            ):
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "next_behavior_action contradicts the contiguous logged successor action",
                )
            verified_count = int(len(verified_rows))
            unverified_count = int(np.sum(active)) - verified_count
            self._provenance["logged_adjacency_authority"] = (
                "CONTIGUOUS_ROWS_VERIFIED"
                if unverified_count == 0
                else "NO_GO_UNVERIFIED_EXPORTED_ADJACENCY"
            )
            if unverified_count:
                self._support.update(
                    {
                        "active_rows": int(np.sum(active)),
                        "verified_adjacent_rows": verified_count,
                        "unverified_exported_adjacent_rows": unverified_count,
                    }
                )
                self._close_gate(
                    EstimateStatus.NO_GO_MISSING_NEXT_BEHAVIOR_ACTION,
                    "bootstrap-active rows lack physically verified adjacent behavior actions and no external adjacency authority is implemented",
                )
                return self
            log_behavior_active = behavior_log_prob(
                behavior_density,
                batch.next_observation[active],
                logged_next_action[active],
                next_times,
            )
            # This second query proves that density is valid at arbitrary target
            # actions, not merely a stored log-probability replay column.
            log_target_active = behavior_log_prob(
                behavior_density,
                batch.next_observation[active],
                next_target_action[active],
                next_times,
            )
            finite_behavior = np.isfinite(log_behavior_active)
            finite_target = np.isfinite(log_target_active)
            supported_target = finite_target & (
                log_target_active >= self.min_log_density
            )
            density_support = {
                "active_rows": int(np.sum(active)),
                "verified_adjacent_rows": verified_count,
                "unverified_exported_adjacent_rows": unverified_count,
                "finite_logged_density_fraction": float(np.mean(finite_behavior)),
                "target_support_fraction": float(np.mean(supported_target)),
                "minimum_required_log_density": self.min_log_density,
            }
            if not np.all(finite_behavior) or not np.all(supported_target):
                self._support.update(density_support)
                self._close_gate(
                    EstimateStatus.NO_GO_BEHAVIOR_SUPPORT,
                    "logged adjacent actions or arbitrary target actions lie outside certified behavior support",
                )
                return self
            log_behavior = np.zeros(len(batch), dtype=np.float64)
            log_target = np.zeros(len(batch), dtype=np.float64)
            log_behavior[active] = log_behavior_active
            log_target[active] = log_target_active
            seed, seed_digest = seed_from_keys(keys)
            trainer = B20KMIFQETrainer(
                gamma=self.gamma,
                horizon=self.horizon,
                ridge=self.ridge,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                hidden_features=int(self._kmifqe_config["hidden_features"]),
                eigenvalue_floor=float(self._kmifqe_config["eigenvalue_floor"]),
                metric_regularization=float(self._kmifqe_config["metric_regularization"]),
                bandwidth_floor=float(self._kmifqe_config["bandwidth_floor"]),
                bandwidth_ceiling=float(self._kmifqe_config["bandwidth_ceiling"]),
                ratio_clip_min=float(self._kmifqe_config["ratio_clip"][0]),
                ratio_clip_max=float(self._kmifqe_config["ratio_clip"][1]),
                target_density_floor=float(self._kmifqe_config["target_density_floor"]),
                resample_size=self._kmifqe_config["resample_size"],
                target_update_interval=int(self._kmifqe_config["target_update_interval"]),
                critic_step_size=float(self._kmifqe_config["critic_step_size"]),
                probability_tolerance=float(
                    self._kmifqe_config["probability_tolerance"]
                ),
            ).fit(
                observation=batch.observation,
                action=batch.action,
                reward=batch.reward,
                next_observation=batch.next_observation,
                native_timestep=batch.native_timestep,
                mask=mask,
                target_next_action=next_target_action,
                logged_next_action=logged_next_action,
                log_behavior_density=log_behavior,
                log_target_density=log_target,
                seed=seed,
                seed_digest=seed_digest,
                min_ess_fraction=self.min_ess_fraction,
            )
            self._trainer = trainer
            action_distance_support = dict(self._support)
            self._support = {
                **action_distance_support,
                **density_support,
                **trainer.support,
                "target_log_density_min": float(np.min(log_target_active)),
                "target_log_density_mean": float(np.mean(log_target_active)),
                "behavior_log_density_min": float(np.min(log_behavior_active)),
            }
            self._diagnostics.update(trainer.diagnostics)
            self._cost.update(trainer.cost)
            if not trainer.support_ok:
                self._close_gate(
                    EstimateStatus.NO_GO_BEHAVIOR_SUPPORT,
                    (
                        f"importance-weight ESS fraction {trainer.support.get('ess_fraction', 0.0):.6g} "
                        f"is below {self.min_ess_fraction:.6g}"
                    ),
                )
                return self
            if not trainer.converged:
                self._close_gate(
                    EstimateStatus.NO_GO_FIT_CONVERGENCE,
                    f"B20 KMIFQE trainer did not converge within {self.max_iterations} iterations",
                )
                return self
            if trainer.feature_map is None or trainer.coefficient is None:
                raise DataValidationError(
                    EstimateStatus.FAILED.value,
                    "B20 KMIFQE trainer published incomplete critic state",
                )
            self._features = trainer.feature_map
            self._coefficient = trainer.coefficient.copy()
            self._fitted = True
        except FloatingPointError as exc:
            self._close_gate(EstimateStatus.FAILED, str(exc))
        except DataValidationError as exc:
            try:
                status = EstimateStatus(exc.status)
            except ValueError:
                status = EstimateStatus.INVALID_DATA
            self._close_gate(status, exc.detail)
        finally:
            self._cost["fit_seconds"] = float(perf_counter() - start)
            self._cost["fit_transitions"] = int(len(batch)) if isinstance(batch, TransitionBatch) else 0
        return self


# Short aliases used by the runner without introducing a method registry layer.
FHFQE = FiniteHorizonFQE
FHKMIFQE = FiniteHorizonKMIFQE


__all__ = [
    "FHFQE",
    "FHKMIFQE",
    "FH_FQE_METHOD_ID",
    "FH_KMIFQE_METHOD_ID",
    "FiniteHorizonFQE",
    "FiniteHorizonKMIFQE",
]
