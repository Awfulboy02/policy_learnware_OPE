"""Compact NumPy kernel-metric in-sample FQE trainer.

This module is a finite-horizon protocol adaptation of B20.  It implements the
mechanisms needed by the companion benchmark without claiming the official
PyTorch architecture or published benchmark numbers: a candidate-specific
nonlinear critic and target critic, local action Hessians, the B20 leading-order
MSE bandwidth estimator, determinant-normalized local metrics, clipped kernel
importance ratios, replacement resampling, and logged-adjacent-action TD.
"""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np


def seed_from_keys(keys: np.ndarray) -> tuple[int, str]:
    """Derive a stable NumPy seed and an audit digest from validated keys."""

    canonical = np.asarray(keys, dtype="<u8")
    digest = _array_digest(canonical)
    return int.from_bytes(bytes.fromhex(digest[:16]), "little"), digest


def _array_digest(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.dtype.str.encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _require_finite(name: str, *arrays: np.ndarray) -> None:
    if any(not np.all(np.isfinite(np.asarray(array))) for array in arrays):
        raise FloatingPointError(f"KMIFQE {name} became non-finite")


class _TanhCriticFeatures:
    """Fixed nonlinear features with an exact action Hessian.

    Only the output coefficients are fitted.  This is intentionally smaller
    than the official two-layer neural critic, while remaining genuinely
    nonlinear in state and action.  Action dimensions that never vary in the
    log receive zero hidden weights: their curvature is not identifiable and
    must not be invented by random features.
    """

    def __init__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        *,
        hidden_features: int,
        rng: np.random.Generator,
    ) -> None:
        with np.errstate(over="ignore", invalid="ignore"):
            self.observation_mean = np.mean(observation, axis=0)
            observation_std = np.std(observation, axis=0)
            self.action_mean = np.mean(action, axis=0)
            action_std = np.std(action, axis=0)
        _require_finite(
            "critic normalization",
            self.observation_mean,
            observation_std,
            self.action_mean,
            action_std,
        )
        self.observation_informative = observation_std >= 1e-8
        self.observation_scale = np.where(self.observation_informative, observation_std, 1.0)
        self.action_informative = action_std >= 1e-8
        self.action_scale = np.where(self.action_informative, action_std, 1.0)
        self.observation_dim = observation.shape[1]
        self.action_dim = action.shape[1]
        self.input_dim = self.observation_dim + self.action_dim + 1
        self.hidden_features = int(hidden_features)
        self.weight = rng.normal(
            scale=0.8 / np.sqrt(max(self.input_dim, 1)),
            size=(self.input_dim, self.hidden_features),
        )
        observation_slice = slice(0, self.observation_dim)
        action_slice = slice(
            self.observation_dim,
            self.observation_dim + self.action_dim,
        )
        self.weight[observation_slice] *= self.observation_informative[:, None]
        self.weight[action_slice] *= self.action_informative[:, None]
        self.bias = rng.uniform(-1.0, 1.0, size=self.hidden_features)

    @property
    def feature_dimension(self) -> int:
        return 1 + self.input_dim + self.hidden_features

    def _standardized_input(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        normalized_time: np.ndarray,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        time = np.asarray(normalized_time, dtype=np.float64).reshape(-1, 1)
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError("observation shape differs from fitted KMIFQE critic")
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action shape differs from fitted KMIFQE critic")
        if len(observation) != len(action) or len(action) != len(time):
            raise ValueError("KMIFQE critic row counts disagree")
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(action)):
            raise ValueError("KMIFQE critic inputs must be finite")
        state = (observation - self.observation_mean) / self.observation_scale
        act = (action - self.action_mean) / self.action_scale
        # Centering time keeps tanh features away from one-sided saturation.
        centered_time = 2.0 * time - 1.0
        standardized = np.concatenate([state, act, centered_time], axis=1)
        _require_finite("standardized critic input", standardized)
        return standardized

    def transform(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        normalized_time: np.ndarray,
    ) -> np.ndarray:
        standardized = self._standardized_input(observation, action, normalized_time)
        hidden = np.tanh(standardized @ self.weight + self.bias)
        return np.concatenate(
            [
                np.ones((len(standardized), 1), dtype=np.float64),
                standardized,
                hidden,
            ],
            axis=1,
        )

    def action_hessian(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        normalized_time: np.ndarray,
        coefficient: np.ndarray,
    ) -> np.ndarray:
        standardized = self._standardized_input(observation, action, normalized_time)
        hidden = np.tanh(standardized @ self.weight + self.bias)
        hidden_coefficient = np.asarray(coefficient, dtype=np.float64)[
            1 + self.input_dim :
        ]
        if hidden_coefficient.shape != (self.hidden_features,):
            raise ValueError("KMIFQE critic coefficient shape is invalid")
        action_weight = self.weight[
            self.observation_dim : self.observation_dim + self.action_dim
        ] / self.action_scale[:, None]
        second_derivative = (
            -2.0
            * hidden
            * (1.0 - np.square(hidden))
            * hidden_coefficient[None, :]
        )
        hessian = np.einsum(
            "nm,im,jm->nij",
            second_derivative,
            action_weight,
            action_weight,
            optimize=True,
        )
        hessian = 0.5 * (hessian + np.swapaxes(hessian, 1, 2))
        _require_finite("action Hessian", hessian)
        return hessian


class B20KMIFQETrainer:
    """Candidate-specific B20 mechanism trainer for the finite-horizon wrapper."""

    def __init__(
        self,
        *,
        gamma: float,
        horizon: int,
        ridge: float,
        max_iterations: int,
        tolerance: float,
        hidden_features: int = 32,
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
        probability_tolerance: float | None = None,
    ) -> None:
        integer_fields = {
            "hidden_features": hidden_features,
            "max_iterations": max_iterations,
            "target_update_interval": target_update_interval,
        }
        if resample_size is not None:
            integer_fields["resample_size"] = resample_size
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive_fields = {
            "ridge": ridge,
            "tolerance": tolerance,
            "eigenvalue_floor": eigenvalue_floor,
            "bandwidth_floor": bandwidth_floor,
            "bandwidth_ceiling": bandwidth_ceiling,
            "ratio_clip_min": ratio_clip_min,
            "ratio_clip_max": ratio_clip_max,
            "target_density_floor": target_density_floor,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(float(metric_regularization)) or float(metric_regularization) < 0.0:
            raise ValueError("metric_regularization must be finite and non-negative")
        if not np.isfinite(float(critic_step_size)) or not 0.0 < float(critic_step_size) <= 1.0:
            raise ValueError("critic_step_size must lie in (0, 1]")
        if probability_tolerance is None:
            probability_tolerance = tolerance
        if not np.isfinite(float(probability_tolerance)) or not 0.0 < float(
            probability_tolerance
        ) <= 2.0:
            raise ValueError("probability_tolerance must lie in (0, 2]")
        if bandwidth_floor > bandwidth_ceiling:
            raise ValueError("bandwidth_floor cannot exceed bandwidth_ceiling")
        if ratio_clip_min > ratio_clip_max:
            raise ValueError("ratio_clip_min cannot exceed ratio_clip_max")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.ridge = float(ridge)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.hidden_features = int(hidden_features)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.metric_regularization = float(metric_regularization)
        self.bandwidth_floor = float(bandwidth_floor)
        self.bandwidth_ceiling = float(bandwidth_ceiling)
        self.ratio_clip_min = float(ratio_clip_min)
        self.ratio_clip_max = float(ratio_clip_max)
        self.target_density_floor = float(target_density_floor)
        self.resample_size = None if resample_size is None else int(resample_size)
        self.target_update_interval = int(target_update_interval)
        self.critic_step_size = float(critic_step_size)
        self.probability_tolerance = float(probability_tolerance)
        self.feature_map: _TanhCriticFeatures | None = None
        self.coefficient: np.ndarray | None = None
        self.target_coefficient: np.ndarray | None = None
        self.converged = False
        self.support_ok = False
        self.support: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}
        self.cost: dict[str, Any] = {}

    @property
    def config(self) -> dict[str, Any]:
        return {
            "critic": "fixed_tanh_features_with_fitted_output",
            "hidden_features": self.hidden_features,
            "ridge": self.ridge,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "eigenvalue_floor": self.eigenvalue_floor,
            "metric_regularization": self.metric_regularization,
            "bandwidth_floor": self.bandwidth_floor,
            "bandwidth_ceiling": self.bandwidth_ceiling,
            "ratio_clip": [self.ratio_clip_min, self.ratio_clip_max],
            "target_density_floor": self.target_density_floor,
            "resample_size": self.resample_size,
            "target_update_interval": self.target_update_interval,
            "critic_step_size": self.critic_step_size,
            "probability_tolerance": self.probability_tolerance,
        }

    def _local_metric(
        self, hessian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _require_finite("local-metric Hessian input", hessian)
        rows, action_dim, _ = hessian.shape
        metric = np.empty_like(hessian)
        transform = np.empty_like(hessian)
        hessian_eigenvalues = np.empty((rows, action_dim), dtype=np.float64)
        metric_eigenvalues = np.empty((rows, action_dim), dtype=np.float64)
        for row in range(rows):
            eigenvalue, eigenvector = np.linalg.eigh(hessian[row])
            hessian_eigenvalues[row] = eigenvalue
            positive = eigenvalue > self.eigenvalue_floor
            negative = eigenvalue < -self.eigenvalue_floor
            positive_count = int(np.sum(positive))
            negative_count = int(np.sum(negative))
            shaped = np.zeros(action_dim, dtype=np.float64)
            if positive_count:
                shaped[positive] = positive_count * eigenvalue[positive]
            if negative_count:
                shaped[negative] = -negative_count * eigenvalue[negative]
            active_max = float(np.max(shaped)) if np.any(shaped > 0.0) else 1.0
            shaped = np.maximum(shaped, self.eigenvalue_floor)
            shaped += self.metric_regularization * active_max
            # alpha=det(M)^(-1/d) from B20 Proposition 3.
            shaped /= float(np.exp(np.mean(np.log(shaped))))
            metric_eigenvalues[row] = shaped
            metric[row] = (eigenvector * shaped[None, :]) @ eigenvector.T
            # L=U sqrt(M) satisfies A=L L^T and det(A)=1.
            transform[row] = eigenvector * np.sqrt(shaped)[None, :]
        _require_finite(
            "local metric", metric, transform, hessian_eigenvalues, metric_eigenvalues
        )
        return metric, transform, hessian_eigenvalues, metric_eigenvalues

    def _bandwidth(
        self,
        *,
        current_feature: np.ndarray,
        reward: np.ndarray,
        mask: np.ndarray,
        target_candidate_feature: np.ndarray,
        hessian: np.ndarray,
        log_target_density: np.ndarray,
        coefficient: np.ndarray,
        target_coefficient: np.ndarray,
        active: np.ndarray,
    ) -> tuple[float, float, float, bool]:
        action_dim = hessian.shape[1]
        active_count = int(np.sum(active))
        laplacian = np.trace(hessian, axis1=1, axis2=2)
        bias_vector = 0.5 * self.gamma * np.mean(
            laplacian[:, None] * current_feature[active], axis=0
        )
        _require_finite("bandwidth bias estimate", bias_vector)
        bias_squared = float(np.sum(np.square(bias_vector)))
        target_q = target_candidate_feature[active] @ target_coefficient
        current_q = current_feature[active] @ coefficient
        td_target_action = reward[active] + self.gamma * mask[active] * target_q - current_q
        parameter_gradient_squared = np.sum(np.square(current_feature[active]), axis=1)
        density = np.maximum(np.exp(log_target_density[active]), self.target_density_floor)
        _require_finite(
            "bandwidth variance inputs",
            target_q,
            current_q,
            td_target_action,
            parameter_gradient_squared,
            density,
        )
        gaussian_squared_integral = float((4.0 * np.pi) ** (-action_dim / 2.0))
        variance_constant = gaussian_squared_integral * float(
            np.mean(np.square(td_target_action) * parameter_gradient_squared / density)
        )
        regularized = False
        denominator = bias_squared
        if denominator < np.finfo(np.float64).eps:
            denominator = np.finfo(np.float64).eps
            regularized = True
        if not np.isfinite(variance_constant) or variance_constant < 0.0:
            raise FloatingPointError("B20 bandwidth variance estimate is non-finite")
        if variance_constant == 0.0:
            raw = 0.0
        else:
            raw = (
                action_dim
                * variance_constant
                / (4.0 * active_count * denominator)
            ) ** (1.0 / (action_dim + 4.0))
        if not np.isfinite(raw):
            raise FloatingPointError("B20 bandwidth estimate is non-finite")
        bandwidth = float(np.clip(raw, self.bandwidth_floor, self.bandwidth_ceiling))
        return bandwidth, bias_squared, variance_constant, regularized

    def _importance_weights(
        self,
        *,
        metric: np.ndarray,
        target_action: np.ndarray,
        logged_action: np.ndarray,
        log_behavior_density: np.ndarray,
        bandwidth: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        action_dim = target_action.shape[1]
        delta = target_action - logged_action
        mahalanobis_squared = np.einsum(
            "ni,nij,nj->n", delta, metric, delta, optimize=True
        )
        log_kernel = (
            -0.5 * mahalanobis_squared / (bandwidth**2)
            - action_dim * np.log(bandwidth)
            - 0.5 * action_dim * np.log(2.0 * np.pi)
        )
        _require_finite("kernel geometry", mahalanobis_squared, log_kernel)
        log_ratio = log_kernel - log_behavior_density
        raw = np.exp(np.clip(log_ratio, -745.0, 709.0))
        clipped = np.clip(raw, self.ratio_clip_min, self.ratio_clip_max)
        clip_fraction = float(
            np.mean((raw < self.ratio_clip_min) | (raw > self.ratio_clip_max))
        )
        if not np.all(np.isfinite(clipped)) or float(np.sum(clipped)) <= 0.0:
            raise FloatingPointError("kernel importance ratios are degenerate")
        return clipped, clip_fraction, mahalanobis_squared

    def fit(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        native_timestep: np.ndarray,
        mask: np.ndarray,
        target_next_action: np.ndarray,
        logged_next_action: np.ndarray,
        log_behavior_density: np.ndarray,
        log_target_density: np.ndarray,
        seed: int,
        seed_digest: str,
        min_ess_fraction: float,
    ) -> "B20KMIFQETrainer":
        rng = np.random.default_rng(seed)
        active = np.asarray(mask, dtype=np.float64) > 0.0
        active_count = int(np.sum(active))
        if active_count == 0:
            raise ValueError("KMIFQE requires at least one bootstrap-active transition")
        self.feature_map = _TanhCriticFeatures(
            observation,
            action,
            hidden_features=self.hidden_features,
            rng=rng,
        )
        current_feature = self.feature_map.transform(
            observation, action, native_timestep / self.horizon
        )
        target_candidate_feature = self.feature_map.transform(
            next_observation,
            target_next_action,
            (native_timestep + 1) / self.horizon,
        )
        logged_next_feature = self.feature_map.transform(
            next_observation,
            logged_next_action,
            (native_timestep + 1) / self.horizon,
        )
        coefficient = np.zeros(current_feature.shape[1], dtype=np.float64)
        target_coefficient = coefficient.copy()
        active_indices = np.flatnonzero(active)
        inactive_indices = np.flatnonzero(~active)
        draw_count = self.resample_size or active_count
        # Fixed uniforms make every iteration deterministic while probabilities
        # still update with h, Hessian, and Q.  Sampling remains with replacement.
        replacement_uniform = rng.random(draw_count)
        bandwidth_history: list[float] = []
        td_loss_history: list[float] = []
        coefficient_delta_history: list[float] = []
        prediction_delta_history: list[float] = []
        bias_history: list[float] = []
        variance_history: list[float] = []
        metric_variation_history: list[float] = []
        ess_history: list[float] = []
        clip_fraction_history: list[float] = []
        mean_weight_history: list[float] = []
        probability_l1_delta_history: list[float | None] = []
        bandwidth_relative_delta_history: list[float | None] = []
        metric_relative_delta_history: list[float | None] = []
        target_lag_history: list[float] = []
        q_tolerance_history: list[float] = []
        first_two_resample_index_digests: list[str] = []
        first_two_td_target_means: list[float] = []
        regularized_bandwidth_updates = 0
        target_updates = 0
        hessian_updates = 0
        linear_solves = 0
        metric_bandwidth_seconds = 0.0
        resampling_seconds = 0.0
        td_update_seconds = 0.0
        stable_rounds = 0
        last_metric = np.broadcast_to(
            np.eye(action.shape[1], dtype=np.float64),
            (active_count, action.shape[1], action.shape[1]),
        ).copy()
        last_transform = last_metric.copy()
        last_hessian_eigenvalues = np.zeros((active_count, action.shape[1]))
        last_metric_eigenvalues = np.ones((active_count, action.shape[1]))
        last_resampled = np.empty(0, dtype=np.int64)
        last_td_target = np.empty(0, dtype=np.float64)
        last_weights = np.ones(active_count, dtype=np.float64)
        last_mahalanobis = np.zeros(active_count, dtype=np.float64)
        last_unique_fraction = 0.0
        last_duplicate_count = 0
        last_loss: float | None = None
        convergence_iteration = 0
        previous_probability: np.ndarray | None = None
        previous_bandwidth: float | None = None
        previous_metric: np.ndarray | None = None

        for iteration in range(1, self.max_iterations + 1):
            phase = perf_counter()
            hessian = self.feature_map.action_hessian(
                next_observation[active],
                target_next_action[active],
                (native_timestep[active] + 1) / self.horizon,
                target_coefficient,
            )
            (
                metric,
                transform,
                hessian_eigenvalues,
                metric_eigenvalues,
            ) = self._local_metric(hessian)
            bandwidth, bias_squared, variance_constant, regularized = self._bandwidth(
                current_feature=current_feature,
                reward=reward,
                mask=mask,
                target_candidate_feature=target_candidate_feature,
                hessian=hessian,
                log_target_density=log_target_density,
                coefficient=coefficient,
                target_coefficient=target_coefficient,
                active=active,
            )
            if regularized:
                regularized_bandwidth_updates += 1
            mean_metric = np.mean(metric, axis=0)
            metric_variation = float(
                np.mean(np.linalg.norm(metric - mean_metric[None, :, :], axis=(1, 2)))
            )
            metric_bandwidth_seconds += perf_counter() - phase

            phase = perf_counter()
            weights, clip_fraction, mahalanobis_squared = self._importance_weights(
                metric=metric,
                target_action=target_next_action[active],
                logged_action=logged_next_action[active],
                log_behavior_density=log_behavior_density[active],
                bandwidth=bandwidth,
            )
            ess = float(np.square(np.sum(weights)) / np.sum(np.square(weights)))
            ess_fraction = ess / active_count
            probability = weights / np.sum(weights)
            probability_delta = (
                None
                if previous_probability is None
                else float(np.sum(np.abs(probability - previous_probability)))
            )
            bandwidth_delta = (
                None
                if previous_bandwidth is None
                else float(
                    abs(bandwidth - previous_bandwidth) / (1.0 + abs(previous_bandwidth))
                )
            )
            metric_delta = (
                None
                if previous_metric is None
                else float(
                    np.max(np.linalg.norm(metric - previous_metric, axis=(1, 2)))
                    / (1.0 + np.max(np.linalg.norm(previous_metric, axis=(1, 2))))
                )
            )
            local_resampled = np.searchsorted(
                np.cumsum(probability), replacement_uniform, side="right"
            )
            local_resampled = np.minimum(local_resampled, active_count - 1)
            resampled = active_indices[local_resampled]
            unique_count = len(np.unique(local_resampled))
            unique_fraction = float(unique_count / draw_count)
            duplicate_count = int(draw_count - unique_count)
            resampling_seconds += perf_counter() - phase

            bandwidth_history.append(bandwidth)
            bias_history.append(bias_squared)
            variance_history.append(variance_constant)
            metric_variation_history.append(metric_variation)
            ess_history.append(ess_fraction)
            clip_fraction_history.append(clip_fraction)
            probability_l1_delta_history.append(probability_delta)
            bandwidth_relative_delta_history.append(bandwidth_delta)
            metric_relative_delta_history.append(metric_delta)
            if len(first_two_resample_index_digests) < 2:
                first_two_resample_index_digests.append(
                    _array_digest(resampled.astype("<i8"))
                )
            previous_probability = probability.copy()
            previous_bandwidth = bandwidth
            previous_metric = metric.copy()
            last_metric = metric
            last_transform = transform
            last_hessian_eigenvalues = hessian_eigenvalues
            last_metric_eigenvalues = metric_eigenvalues
            last_resampled = resampled
            last_weights = weights
            last_mahalanobis = mahalanobis_squared
            last_unique_fraction = unique_fraction
            last_duplicate_count = duplicate_count
            hessian_updates += 1

            sampled_rows = np.concatenate([resampled, inactive_indices])

            if ess_fraction < min_ess_fraction:
                self.support_ok = False
                convergence_iteration = iteration
                break
            self.support_ok = True

            phase = perf_counter()
            sampled_mask = mask[sampled_rows]
            td_target = reward[sampled_rows] + self.gamma * sampled_mask * (
                logged_next_feature[sampled_rows] @ target_coefficient
            )
            sampled_feature = current_feature[sampled_rows]
            mean_weight = float(np.mean(weights))
            normalizer = float(len(sampled_rows))
            gram = mean_weight * (sampled_feature.T @ sampled_feature) / normalizer
            gram += self.ridge * np.eye(sampled_feature.shape[1], dtype=np.float64)
            rhs = mean_weight * (sampled_feature.T @ td_target) / normalizer
            try:
                solution = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                solution = np.linalg.pinv(gram, rcond=1e-12) @ rhs
            linear_solves += 1
            updated = coefficient + self.critic_step_size * (solution - coefficient)
            _require_finite("critic linear system", gram, rhs, solution, updated)
            if not np.all(np.isfinite(updated)) or np.linalg.norm(updated) > 1e14:
                raise FloatingPointError("KMIFQE critic update diverged")
            prediction_delta = float(
                np.max(np.abs(current_feature @ (updated - coefficient)))
            )
            prediction = sampled_feature @ updated
            td_loss = mean_weight * float(np.mean(np.square(prediction - td_target)))
            coefficient_delta = float(np.max(np.abs(updated - coefficient)))
            target_lag = float(
                np.max(np.abs(current_feature @ (updated - target_coefficient)))
            )
            q_scale = 1.0 + float(np.max(np.abs(current_feature @ updated)))
            q_tolerance = self.tolerance * q_scale
            _require_finite(
                "TD update",
                td_target,
                prediction,
                prediction_delta,
                td_loss,
                coefficient_delta,
                target_lag,
                q_tolerance,
            )
            coefficient = updated
            if iteration == 1 or iteration % self.target_update_interval == 0:
                target_coefficient = coefficient.copy()
                target_updates += 1
            td_update_seconds += perf_counter() - phase
            td_loss_history.append(td_loss)
            mean_weight_history.append(mean_weight)
            coefficient_delta_history.append(coefficient_delta)
            prediction_delta_history.append(prediction_delta)
            target_lag_history.append(target_lag)
            q_tolerance_history.append(q_tolerance)
            if len(first_two_td_target_means) < 2:
                first_two_td_target_means.append(float(np.mean(td_target)))
            last_td_target = td_target
            last_loss = td_loss

            minimum_iterations = max(3, self.horizon + 1)
            # Output-layer coefficients are not identifiable in a redundant
            # fixed-feature basis; convergence is therefore tested in Q-space,
            # not by requiring arbitrary null-space coefficients to stop moving.
            if (
                iteration >= minimum_iterations
                and target_updates >= 2
                and prediction_delta <= q_tolerance
                and target_lag <= q_tolerance
                and probability_delta is not None
                and probability_delta <= self.probability_tolerance
            ):
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= 2:
                self.converged = True
                self.support_ok = True
                convergence_iteration = iteration
                break
        else:
            convergence_iteration = self.max_iterations

        self.coefficient = coefficient
        self.target_coefficient = target_coefficient
        metric_abs = np.abs(last_hessian_eigenvalues)
        hessian_condition = np.max(metric_abs, axis=1) / np.maximum(
            np.min(metric_abs, axis=1), self.eigenvalue_floor
        )
        metric_condition = np.max(last_metric_eigenvalues, axis=1) / np.min(
            last_metric_eigenvalues, axis=1
        )
        off_diagonal = last_metric - np.eye(action.shape[1])[None, :, :] * np.diagonal(
            last_metric, axis1=1, axis2=2
        )[:, None, :]
        final_ess = ess_history[-1] if ess_history else 0.0
        self.support = {
            "kind": "B20_kernel_metric_importance_resampling",
            "active_rows": active_count,
            "ess": float(final_ess * active_count),
            "ess_fraction": float(final_ess),
            "kernel_bandwidth": float(bandwidth_history[-1]),
            "active_weight_min": float(np.min(last_weights)),
            "active_weight_max": float(np.max(last_weights)),
            "active_weight_mean": float(np.mean(last_weights)),
            "clip_fraction": float(clip_fraction_history[-1]),
            "unique_resampled_fraction": last_unique_fraction,
            "replacement_duplicate_count": last_duplicate_count,
            "mean_mahalanobis_squared": float(np.mean(last_mahalanobis)),
        }
        self.diagnostics = {
            "protocol": "B20_PROTOCOL_ADAPTATION",
            "bandwidth_source": "B20_EQ11_TD_MSE_BIAS_VARIANCE_ESTIMATOR",
            "bandwidth_history": bandwidth_history,
            "bandwidth_update_count": len(bandwidth_history),
            "bandwidth_regularized_update_count": regularized_bandwidth_updates,
            "alternating_update_count": len(bandwidth_history),
            "resampling_refresh_policy": (
                "every iteration recomputes Q/Hessian/h/L/probabilities and maps a "
                "seeded common-random uniform panel to replacement indices"
            ),
            "bias_constant_squared_history": bias_history,
            "variance_constant_history": variance_history,
            "hessian_eigenvalue_min": float(np.min(last_hessian_eigenvalues)),
            "hessian_eigenvalue_max": float(np.max(last_hessian_eigenvalues)),
            "hessian_abs_condition_p95": float(np.quantile(hessian_condition, 0.95)),
            "hessian_near_zero_fraction": float(
                np.mean(metric_abs <= self.eigenvalue_floor)
            ),
            "hessian_effective_rank_mean": float(
                np.mean(np.sum(metric_abs > self.eigenvalue_floor, axis=1))
            ),
            "metric_condition_p95": float(np.quantile(metric_condition, 0.95)),
            "metric_determinant_max_error": float(
                np.max(np.abs(np.linalg.det(last_metric) - 1.0))
            ),
            "metric_state_variation": float(metric_variation_history[-1]),
            "metric_state_variation_history": metric_variation_history,
            "metric_off_diagonal_frobenius_mean": float(
                np.mean(np.linalg.norm(off_diagonal, axis=(1, 2)))
            ),
            "local_metric_rows": active_count,
            "ess_fraction_history": ess_history,
            "clip_fraction_history": clip_fraction_history,
            "replacement_resampling": True,
            "replacement_draw_count": draw_count,
            "replacement_duplicate_count": last_duplicate_count,
            "unique_resampled_fraction": last_unique_fraction,
            "resample_index_sha256": _array_digest(last_resampled.astype("<i8")),
            "resample_index_sha256_first_two": first_two_resample_index_digests,
            "bootstrap_action_source": "logged_adjacent_next_behavior_action",
            "logged_adjacent_action_sha256": _array_digest(logged_next_action[active]),
            "last_td_target_mean": (
                float(np.mean(last_td_target)) if len(last_td_target) else None
            ),
            "last_td_target_std": (
                float(np.std(last_td_target)) if len(last_td_target) else None
            ),
            "td_loss": last_loss,
            "td_loss_history": td_loss_history,
            "td_target_mean_first_two": first_two_td_target_means,
            "coefficient_delta_history": coefficient_delta_history,
            "prediction_delta_history": prediction_delta_history,
            "target_lag_history": target_lag_history,
            "q_tolerance_history": q_tolerance_history,
            "probability_l1_delta_history": probability_l1_delta_history,
            "bandwidth_relative_delta_history": bandwidth_relative_delta_history,
            "metric_relative_delta_history": metric_relative_delta_history,
            "convergence_criterion": (
                "two consecutive full alternating updates after at least two target "
                "syncs: Q update and target lag <= tolerance*(1+max_abs_Q), and "
                "replacement probability L1 delta <= probability_tolerance; h/metric deltas are "
                "diagnostic because they may be non-identifiable when probabilities agree"
            ),
            "requested_tolerance": self.tolerance,
            "probability_l1_tolerance": self.probability_tolerance,
            "final_q_tolerance": q_tolerance_history[-1] if q_tolerance_history else None,
            "iterations": convergence_iteration,
            "converged": self.converged,
            "target_update_count": target_updates,
            "hessian_update_count": hessian_updates,
            "critic_parameter_l2": float(np.linalg.norm(coefficient)),
            "target_critic_parameter_l2": float(np.linalg.norm(target_coefficient)),
            "feature_dimension": current_feature.shape[1],
            "seed": int(seed),
            "fit_key_digest": seed_digest,
            "failed_closed": not self.converged,
            "mean_weight_bias_correction": float(np.mean(last_weights)),
            "mean_weight_history": mean_weight_history,
            "critic_objective": "mean_clipped_importance_weight*MSE + ridge*parameter_L2",
            "critic_step_rule": "configured damping toward the regularized objective solution",
            "ratio_normalization": "p=w/sum(w); TD loss multiplied by mean(w)",
            "metric_transform_frobenius_mean": float(
                np.mean(np.linalg.norm(last_transform, axis=(1, 2)))
            ),
        }
        self.cost = {
            "metric_bandwidth_seconds": float(metric_bandwidth_seconds),
            "replacement_resampling_seconds": float(resampling_seconds),
            "td_update_seconds": float(td_update_seconds),
            "linear_solves": linear_solves,
            "target_updates": target_updates,
            "hessian_updates": hessian_updates,
        }
        return self


__all__ = ["B20KMIFQETrainer", "seed_from_keys"]
