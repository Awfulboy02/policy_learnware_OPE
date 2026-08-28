"""Finite-horizon NumPy FQE and kernel/importance-weighted FQE.

These implementations are deliberately compact reference estimators for the
companion repository and its synthetic acceptance tests.  They are real fitted
Bellman estimators, not placeholder score generators.  Both critics consume
``native_timestep / H`` explicitly and share the strict mask contract from
``TransitionBatch``.
"""

from __future__ import annotations

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
    policy_id,
    policy_semantics,
    validate_action_keys,
)


FH_FQE_METHOD_ID = "FH_FQE_G099_H1000"
FH_KMIFQE_METHOD_ID = "FH_KMIFQE_G099_H1000"


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

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        ridge: float = 1e-6,
        max_iterations: int | None = None,
        tolerance: float = 1e-9,
    ) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        if float(ridge) <= 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and positive")
        if max_iterations is None:
            max_iterations = max(int(horizon) + 1, 50)
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if float(tolerance) <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.ridge = float(ridge)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self._features: _QuadraticTimeFeatures | None = None
        self._coefficient: np.ndarray | None = None
        self._candidate: CandidateActionProvider | None = None
        self._candidate_id: str | None = None
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
            self._candidate_id = policy_id(candidate)
            keys = validate_action_keys(fit_keys, len(batch))
            semantics = policy_semantics(candidate)
            self._base_provenance(batch, semantics)
            if semantics is not PolicySemantics.DETERMINISTIC:
                self._close_gate(
                    EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS,
                    "FQE currently implements deterministic evaluation-policy Bellman targets only",
                )
                return self
            mask = batch.bootstrap_mask(self.horizon)
            next_action = self._query_next_actions(batch, candidate, keys, mask)
            self._measure_action_support(batch, candidate, keys)
            self._fit_ridge(batch, next_action, mask, np.ones(len(batch), dtype=np.float64))
        except DataValidationError as exc:
            try:
                status = EstimateStatus(exc.status)
            except ValueError:
                status = EstimateStatus.INVALID_DATA
            self._close_gate(status, exc.detail)
        finally:
            self._cost["fit_seconds"] = float(perf_counter() - start)
            self._cost["fit_transitions"] = int(len(batch))
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
            "value_convention": "raw_discounted_return",
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
    ) -> np.ndarray:
        active = mask.astype(bool)
        result = np.zeros_like(batch.action, dtype=np.float64)
        if np.any(active):
            queried = candidate_actions(
                candidate,
                batch.next_observation[active],
                batch.native_timestep[active] + 1,
                keys=keys[active],
                require_deterministic=True,
            )
            if queried.shape[1] != batch.action.shape[1]:
                raise DataValidationError(
                    EstimateStatus.INVALID_DATA.value,
                    "candidate action width differs from logged action ABI",
                )
            result[active] = queried
        return result

    def _fit_ridge(
        self,
        batch: TransitionBatch,
        next_action: np.ndarray,
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
        next_x = self._features.transform(
            batch.next_observation,
            next_action,
            (batch.native_timestep + 1) / self.horizon,
        )
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
            self._diagnostics["convergence_warning"] = "maximum iterations reached"
        self._fitted = True

    def _measure_action_support(
        self,
        batch: TransitionBatch,
        candidate: CandidateActionProvider,
        keys: np.ndarray,
    ) -> None:
        current_action = candidate_actions(
            candidate,
            batch.observation,
            batch.native_timestep,
            keys=keys ^ np.uint64(0xA5A5A5A5A5A5A5A5),
            require_deterministic=True,
        )
        if current_action.shape != batch.action.shape:
            raise DataValidationError(
                EstimateStatus.INVALID_DATA.value,
                "candidate action width differs from logged action ABI",
            )
        action_scale = np.std(batch.action, axis=0)
        action_scale[action_scale < 1e-6] = 1.0
        distance = np.linalg.norm((current_action - batch.action) / action_scale, axis=1)
        self._support = {
            "kind": "pointwise_logged_state_action_distance",
            "mean_action_distance": float(np.mean(distance)),
            "p95_action_distance": float(np.quantile(distance, 0.95)),
            "max_action_distance": float(np.max(distance)),
            "support_rows": int(len(distance)),
        }

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
        if np.isscalar(initial_timestep):
            times = np.full(len(observations), int(initial_timestep), dtype=np.int64)
        else:
            raw_times = np.asarray(initial_timestep)
            if raw_times.shape != (len(observations),) or raw_times.dtype.kind not in "iu":
                raise ValueError("initial_timestep must be an integer scalar or one value per state")
            times = raw_times.astype(np.int64)
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
                provenance=self._provenance,
                cost=cost,
                diagnostics=diagnostics,
            )
        if self._candidate is None or self._features is None or self._coefficient is None:
            raise RuntimeError("fitted estimator state is incomplete")
        actions = candidate_actions(
            self._candidate,
            observations,
            times,
            keys=checked_keys,
            require_deterministic=True,
        )
        features = self._features.transform(observations, actions, times / self.horizon)
        values = features @ self._coefficient
        estimate_seconds = float(perf_counter() - start)
        cost = dict(self._cost)
        cost["estimate_seconds"] = estimate_seconds
        cost["runtime_seconds"] = float(cost.get("fit_seconds", 0.0) + estimate_seconds)
        diagnostics = dict(self._diagnostics)
        diagnostics.update(
            {
                "initial_state_count": int(len(observations)),
                "initial_value_std": float(np.std(values)),
            }
        )
        return ValueEstimate(
            method_id=self.method_id,
            status=EstimateStatus.PASS,
            value=float(np.mean(values)),
            support=self._support,
            provenance=self._provenance,
            cost=cost,
            diagnostics=diagnostics,
        )


class FiniteHorizonKMIFQE(FiniteHorizonFQE):
    """Kernel-smoothed, importance-weighted finite-horizon FQE adaptation.

    For active Bellman rows, the deterministic target action is matched to the
    logged next behavior action using a Gaussian kernel and inverse exact
    behavior density.  The density is also queried at the arbitrary target
    action, both to establish the required API and to gate target support.
    """

    method_id = FH_KMIFQE_METHOD_ID

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        ridge: float = 1e-6,
        max_iterations: int | None = None,
        tolerance: float = 1e-9,
        kernel_bandwidth: float | None = None,
        min_log_density: float = -50.0,
        min_ess_fraction: float = 0.01,
    ) -> None:
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            ridge=ridge,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if kernel_bandwidth is not None and (kernel_bandwidth <= 0 or not np.isfinite(kernel_bandwidth)):
            raise ValueError("kernel_bandwidth must be finite and positive")
        if not np.isfinite(min_log_density):
            raise ValueError("min_log_density must be finite")
        if not 0.0 < float(min_ess_fraction) <= 1.0:
            raise ValueError("min_ess_fraction must lie in (0, 1]")
        self.kernel_bandwidth = None if kernel_bandwidth is None else float(kernel_bandwidth)
        self.min_log_density = float(min_log_density)
        self.min_ess_fraction = float(min_ess_fraction)

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
            self._candidate_id = policy_id(candidate)
            keys = validate_action_keys(fit_keys, len(batch))
            try:
                semantics = policy_semantics(candidate)
            except DataValidationError:
                semantics = PolicySemantics.STOCHASTIC_KEYED
            self._base_provenance(batch, semantics)
            self._provenance.update(
                {
                    "implementation": "numpy_kernel_importance_weighted_time_ridge_fqe",
                    "density_id": str(getattr(behavior_density, "density_id", "")),
                    "density_exact": bool(getattr(behavior_density, "exact", False)),
                    "kernel": "gaussian_on_standardized_next_action",
                    "metric_learning_scope": "diagonal_logged_action_standardization",
                    "full_B20_Hessian_metric_port": "PENDING_REAL_ASSET_PHASE",
                    "importance_weight": "K(a_next_behavior,pi(s_next))/mu(a_next_behavior|s_next)",
                }
            )
            if not bool(getattr(behavior_density, "exact", False)):
                self._close_gate(
                    EstimateStatus.NO_GO_EXISTING_LOG_DENSITY,
                    "KMIFQE requires exact behavior density; clipped-Gaussian existing logs do not provide it",
                )
                return self
            if semantics is not PolicySemantics.DETERMINISTIC:
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
            next_action = self._query_next_actions(batch, candidate, keys, mask)
            weight = self._kernel_importance_weights(
                batch,
                next_action,
                mask,
                behavior_density,
            )
            if weight is None:
                return self
            self._fit_ridge(batch, next_action, mask, weight)
        except DataValidationError as exc:
            try:
                status = EstimateStatus(exc.status)
            except ValueError:
                status = EstimateStatus.INVALID_DATA
            self._close_gate(status, exc.detail)
        finally:
            self._cost["fit_seconds"] = float(perf_counter() - start)
            self._cost["fit_transitions"] = int(len(batch))
        return self

    def _kernel_importance_weights(
        self,
        batch: TransitionBatch,
        next_action: np.ndarray,
        mask: np.ndarray,
        density: BehaviorDensityProvider,
    ) -> np.ndarray | None:
        active = mask.astype(bool)
        weight = np.ones(len(batch), dtype=np.float64)
        if not np.any(active):
            self._support = {
                "ess": float(len(batch)),
                "ess_fraction": 1.0,
                "active_rows": 0,
            }
            return weight
        next_times = batch.native_timestep[active] + 1
        behavior_action = batch.next_behavior_action[active]
        target_action = next_action[active]
        log_behavior = behavior_log_prob(
            density,
            batch.next_observation[active],
            behavior_action,
            next_times,
        )
        # This second call is intentional: exact density must support arbitrary
        # candidate actions, not merely replay a stored action log-prob column.
        log_target = behavior_log_prob(
            density,
            batch.next_observation[active],
            target_action,
            next_times,
        )
        finite_behavior = np.isfinite(log_behavior)
        finite_target = np.isfinite(log_target)
        supported_target = finite_target & (log_target >= self.min_log_density)
        if not np.all(finite_behavior) or not np.all(supported_target):
            self._support = {
                "active_rows": int(np.sum(active)),
                "finite_logged_density_fraction": float(np.mean(finite_behavior)),
                "target_support_fraction": float(np.mean(supported_target)),
                "minimum_required_log_density": self.min_log_density,
            }
            self._close_gate(
                EstimateStatus.NO_GO_BEHAVIOR_SUPPORT,
                "logged next actions or arbitrary candidate actions lie outside certified behavior support",
            )
            return None

        action_scale = np.std(behavior_action, axis=0)
        action_scale[action_scale < 1e-6] = 1.0
        standardized_delta = (behavior_action - target_action) / action_scale
        distance = np.linalg.norm(standardized_delta, axis=1)
        if self.kernel_bandwidth is None:
            positive = distance[distance > 1e-12]
            bandwidth = float(np.median(positive)) if len(positive) else 1.0
        else:
            bandwidth = self.kernel_bandwidth
        bandwidth = max(bandwidth, 1e-6)
        log_weight = -0.5 * np.square(distance / bandwidth) - log_behavior
        log_weight -= float(np.max(log_weight))
        active_weight = np.exp(np.clip(log_weight, -80.0, 0.0))
        if not np.all(np.isfinite(active_weight)) or float(np.sum(active_weight)) <= 0.0:
            self._close_gate(EstimateStatus.NO_GO_BEHAVIOR_SUPPORT, "kernel importance weights are degenerate")
            return None
        weight[active] = active_weight
        weight /= np.mean(weight)
        ess = float(np.square(np.sum(weight)) / np.sum(np.square(weight)))
        ess_fraction = ess / len(weight)
        self._support = {
            "ess": ess,
            "ess_fraction": ess_fraction,
            "active_rows": int(np.sum(active)),
            "kernel_bandwidth": bandwidth,
            "weight_min": float(np.min(weight)),
            "weight_max": float(np.max(weight)),
            "target_log_density_min": float(np.min(log_target)),
            "target_log_density_mean": float(np.mean(log_target)),
            "behavior_log_density_min": float(np.min(log_behavior)),
            "mean_action_distance": float(np.mean(distance)),
            "metric_conditioning": float(np.max(action_scale) / np.min(action_scale)),
        }
        if ess_fraction < self.min_ess_fraction:
            self._close_gate(
                EstimateStatus.NO_GO_BEHAVIOR_SUPPORT,
                f"importance-weight ESS fraction {ess_fraction:.6g} is below {self.min_ess_fraction:.6g}",
            )
            return None
        return weight


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
