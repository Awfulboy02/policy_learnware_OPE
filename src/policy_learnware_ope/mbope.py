"""Small, executable model-based OPE estimators for the v0.4b protocol.

The implementations in this module intentionally optimize for auditability and
small synthetic/feasibility runs.  They are not claims of reproducing the
upstream papers at their original scale:

* ``DOPE_STYLE_MB_FF_G099_H1000`` is a project-defined feed-forward model.
  DOPE supplies benchmark inspiration only; it is not a "DOPE algorithm".
* ``AR_MBOPE_G099_H1000`` uses an explicit fixed-order autoregressive
  factorization, teacher forcing at fit time, and sequential generation.
* ``ETM_MBOPE_G099_H1000`` fits a conditional contrastive energy and samples
  it with Langevin dynamics.  It is not the feed-forward model under a new ID.

All policy calls happen inside learned-model rollouts.  Every call carries one
explicit key per row, including calls to actors whose semantics are declared
deterministic.  Native timesteps enter every transition feature; compressed
subsample ordinals are rejected rather than silently treated as time.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .core import (
    DataValidationError,
    EstimateStatus,
    TransitionBatch,
    ValueEstimate,
    candidate_actions,
    finite_horizon_method_id,
    finite_horizon_value_convention,
    policy_id,
    policy_semantics,
)


DOPE_STYLE_MB_FF_ID = "DOPE_STYLE_MB_FF_G099_H1000"
AR_MBOPE_ID = "AR_MBOPE_G099_H1000"
ETM_MBOPE_ID = "ETM_MBOPE_G099_H1000"


def _value_convention(gamma: float, horizon: int) -> str:
    return finite_horizon_value_convention(gamma, horizon)


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n, d)")
    return array


@dataclass(frozen=True)
class _BatchData:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dataset_cut: np.ndarray
    native_timestep: np.ndarray
    environment_terminal: np.ndarray
    valid: np.ndarray
    timestep_provenance: str
    physical_membership_sha256: str
    source_digest: str | None


def _membership_digest(batch: TransitionBatch) -> str:
    digest = sha256()
    for name in (
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "dataset_cut",
        "native_timestep",
        "episode_id",
        "episode_offsets",
        "truncation_reason",
    ):
        value = np.asarray(getattr(batch, name))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(batch.timestep_provenance.encode("utf-8"))
    digest.update(str(batch.source_digest or "").encode("utf-8"))
    return digest.hexdigest()


def _extract_batch(batch: TransitionBatch, horizon: int) -> _BatchData:
    if not isinstance(batch, TransitionBatch):
        raise DataValidationError(
            EstimateStatus.INVALID_DATA.value,
            "model-based estimators require a core.TransitionBatch so physical "
            "membership and termination semantics cannot bypass validation",
        )
    # Reuse the canonical finite-horizon validation even though a transition
    # model does not consume the returned Bellman mask.  This catches invalid
    # horizon truncations while preserving dataset cuts as non-terminals.
    batch.bootstrap_mask(horizon)
    observation = batch.observation
    action = batch.action
    reward = batch.reward
    next_observation = batch.next_observation
    terminated = batch.terminated
    truncated = batch.truncated
    dataset_cut = batch.dataset_cut
    native_timestep = batch.native_timestep
    reasons = batch.truncation_reason
    # Dataset cuts are artificial membership boundaries, not terminals.  Their
    # observed (s,a,r,s') rows remain valid for a transition model.  Native
    # terminations and environment truncations stop model rollouts; time-limit
    # truncations are handled solely by the configured finite horizon.
    environment_terminal = terminated | (reasons == "environment")
    valid = np.ones(len(batch), dtype=bool)
    return _BatchData(
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
        dataset_cut,
        native_timestep,
        environment_terminal,
        valid,
        batch.timestep_provenance,
        _membership_digest(batch),
        batch.source_digest,
    )


def _mix64(value: int) -> int:
    """SplitMix64 finalizer, expressed with Python ints to avoid overflow warnings."""

    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _key_array(value: Any, name: str, *, rows: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    elif raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers; implicit numeric casts are forbidden")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError(f"{name} must contain non-negative integers")
    if rows is not None and len(raw) != rows:
        raise ValueError(f"{name} must provide exactly one uint64 key per row")
    if len(raw) == 0:
        raise ValueError(f"{name} must not be empty")
    return np.asarray(raw, dtype=np.uint64)


def _seed_from_keys(keys: np.ndarray, salt: int) -> int:
    accumulator = _mix64(salt ^ len(keys))
    for position, key in enumerate(np.asarray(keys, dtype=np.uint64)):
        accumulator = _mix64(accumulator ^ int(key) ^ _mix64(position))
    return accumulator


def _key_digest(keys: np.ndarray) -> str:
    return sha256(np.asarray(keys, dtype="<u8").tobytes()).hexdigest()


@dataclass
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, *, floor: float = 1e-6) -> "_Standardizer":
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale < floor, 1.0, scale)
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def _ridge(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    gram = features.T @ features
    regularizer = np.eye(gram.shape[0], dtype=float) * float(penalty)
    # Do not regularize the explicit intercept.
    regularizer[0, 0] = 0.0
    right = features.T @ targets
    try:
        return np.linalg.solve(gram + regularizer, right)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + regularizer) @ right


class _RandomFeatureRidge:
    """One hidden-layer feed-forward map with a ridge-fitted output head."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, ridge: float, rng: np.random.Generator):
        self.weight = rng.normal(scale=1.0 / np.sqrt(max(input_dim, 1)), size=(input_dim, hidden_dim))
        self.bias = rng.uniform(-np.pi, np.pi, size=hidden_dim)
        self.output_dim = int(output_dim)
        self.ridge = float(ridge)
        self.coefficient: np.ndarray | None = None

    def features(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(inputs @ self.weight + self.bias)
        return np.concatenate([np.ones((len(inputs), 1)), inputs, hidden], axis=1)

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> "_RandomFeatureRidge":
        self.coefficient = _ridge(self.features(inputs), targets, self.ridge)
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.coefficient is None:
            raise RuntimeError("ridge head is not fitted")
        return self.features(inputs) @ self.coefficient


def _transition_inputs(
    observation: np.ndarray,
    action: np.ndarray,
    native_timestep: np.ndarray,
    horizon: int,
) -> np.ndarray:
    time = np.asarray(native_timestep, dtype=float).reshape(-1, 1) / float(horizon)
    return np.concatenate([observation, action, time], axis=1)


class _TerminationHead:
    """Small learned environment-termination head; truncations are never positives."""

    def __init__(self, ridge: float, hidden_dim: int = 16):
        self.ridge = float(ridge)
        self.hidden_dim = int(hidden_dim)
        self.scaler: _Standardizer | None = None
        self.model: _RandomFeatureRidge | None = None
        self.constant: float | None = None

    def fit(self, inputs: np.ndarray, terminated: np.ndarray, rng: np.random.Generator) -> None:
        labels = np.asarray(terminated, dtype=float).reshape(-1, 1)
        if np.all(labels == labels[0]):
            self.constant = float(labels[0, 0])
            return
        self.scaler = _Standardizer.fit(inputs)
        normalized = self.scaler.transform(inputs)
        self.model = _RandomFeatureRidge(
            normalized.shape[1], 1, self.hidden_dim, self.ridge, rng
        ).fit(normalized, labels)

    def probability(self, inputs: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(inputs), self.constant, dtype=float)
        if self.scaler is None or self.model is None:
            raise RuntimeError("termination head is not fitted")
        raw = self.model.predict(self.scaler.transform(inputs)).reshape(-1)
        return np.clip(raw, 0.0, 1.0)


@dataclass
class _FFMember:
    model: _RandomFeatureRidge
    residual_std: np.ndarray


class _FeedForwardDeltaModel:
    def __init__(self, *, members: int, hidden_dim: int, ridge: float):
        if members <= 0:
            raise ValueError("members must be positive")
        self.member_count = int(members)
        self.hidden_dim = int(hidden_dim)
        self.ridge = float(ridge)
        self.x_scaler: _Standardizer | None = None
        self.y_scaler: _Standardizer | None = None
        self.members: list[_FFMember] = []
        self.diagnostics: dict[str, Any] = {}

    def fit(self, inputs: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> None:
        self.x_scaler = _Standardizer.fit(inputs)
        self.y_scaler = _Standardizer.fit(targets)
        x = self.x_scaler.transform(inputs)
        y = self.y_scaler.transform(targets)
        predictions: list[np.ndarray] = []
        for _ in range(self.member_count):
            indices = rng.integers(0, len(x), size=len(x))
            model = _RandomFeatureRidge(
                x.shape[1], y.shape[1], self.hidden_dim, self.ridge, rng
            ).fit(x[indices], y[indices])
            fitted = model.predict(x)
            residual_std = np.maximum(np.std(y - fitted, axis=0), 1e-4)
            self.members.append(_FFMember(model, residual_std))
            predictions.append(self.y_scaler.inverse(fitted))
        stacked = np.stack(predictions, axis=0)
        mean_prediction = np.mean(stacked, axis=0)
        self.diagnostics = {
            "train_mse": float(np.mean((mean_prediction - targets) ** 2)),
            "ensemble_prediction_spread": float(np.mean(np.std(stacked, axis=0))),
            "ensemble_members": self.member_count,
            "target_parameterization": "delta_state_then_reward",
            "feed_forward_head": "random_tanh_features_plus_ridge",
        }

    def sample(
        self,
        inputs: np.ndarray,
        member_index: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.x_scaler is None or self.y_scaler is None or not self.members:
            raise RuntimeError("feed-forward model is not fitted")
        x = self.x_scaler.transform(inputs)
        output = np.empty((len(x), len(self.y_scaler.mean)), dtype=float)
        for index in np.unique(member_index):
            rows = np.flatnonzero(member_index == index)
            member = self.members[int(index)]
            normalized = member.model.predict(x[rows])
            normalized += rng.normal(size=normalized.shape) * member.residual_std
            output[rows] = self.y_scaler.inverse(normalized)
        return output


@dataclass
class _ARMember:
    heads: list[_RandomFeatureRidge]
    residual_std: np.ndarray


class _AutoregressiveDeltaModel:
    """p(delta_s, r | s, a, t) with a fixed left-to-right factorization."""

    def __init__(self, *, members: int, hidden_dim: int, ridge: float):
        self.member_count = int(members)
        self.hidden_dim = int(hidden_dim)
        self.ridge = float(ridge)
        self.x_scaler: _Standardizer | None = None
        self.y_scaler: _Standardizer | None = None
        self.members: list[_ARMember] = []
        self.diagnostics: dict[str, Any] = {}

    def fit(self, inputs: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> None:
        self.x_scaler = _Standardizer.fit(inputs)
        self.y_scaler = _Standardizer.fit(targets)
        x = self.x_scaler.transform(inputs)
        y = self.y_scaler.transform(targets)
        teacher_predictions: list[np.ndarray] = []
        free_predictions: list[np.ndarray] = []
        for _ in range(self.member_count):
            indices = rng.integers(0, len(x), size=len(x))
            heads: list[_RandomFeatureRidge] = []
            teacher = np.empty_like(y)
            free = np.empty_like(y)
            residual_std = np.empty(y.shape[1], dtype=float)
            for dimension in range(y.shape[1]):
                train_features = np.concatenate(
                    [x[indices], y[indices, :dimension]], axis=1
                )
                head = _RandomFeatureRidge(
                    train_features.shape[1], 1, self.hidden_dim, self.ridge, rng
                ).fit(train_features, y[indices, dimension : dimension + 1])
                teacher_features = np.concatenate([x, y[:, :dimension]], axis=1)
                teacher[:, dimension] = head.predict(teacher_features).reshape(-1)
                free_features = np.concatenate([x, free[:, :dimension]], axis=1)
                free[:, dimension] = head.predict(free_features).reshape(-1)
                residual_std[dimension] = max(
                    float(np.std(y[:, dimension] - teacher[:, dimension])), 1e-4
                )
                heads.append(head)
            self.members.append(_ARMember(heads, residual_std))
            teacher_predictions.append(self.y_scaler.inverse(teacher))
            free_predictions.append(self.y_scaler.inverse(free))
        teacher_mean = np.mean(np.stack(teacher_predictions), axis=0)
        free_mean = np.mean(np.stack(free_predictions), axis=0)
        self.diagnostics = {
            "teacher_forced_mse": float(np.mean((teacher_mean - targets) ** 2)),
            "free_running_one_step_mse": float(np.mean((free_mean - targets) ** 2)),
            "teacher_forced_mse_by_dimension": np.mean(
                (teacher_mean - targets) ** 2, axis=0
            ).tolist(),
            "free_running_mse_by_dimension": np.mean(
                (free_mean - targets) ** 2, axis=0
            ).tolist(),
            "factorization_order": [
                *[f"delta_state[{index}]" for index in range(targets.shape[1] - 1)],
                "reward",
            ],
            "generation": "fixed_order_sequential",
            "training": "teacher_forcing",
            "ensemble_members": self.member_count,
            "target_parameterization": "delta_state_then_reward",
        }

    def sample(
        self,
        inputs: np.ndarray,
        member_index: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.x_scaler is None or self.y_scaler is None or not self.members:
            raise RuntimeError("autoregressive model is not fitted")
        x = self.x_scaler.transform(inputs)
        result = np.empty((len(x), len(self.y_scaler.mean)), dtype=float)
        for index in np.unique(member_index):
            rows = np.flatnonzero(member_index == index)
            member = self.members[int(index)]
            generated = np.empty((len(rows), len(member.heads)), dtype=float)
            for dimension, head in enumerate(member.heads):
                features = np.concatenate([x[rows], generated[:, :dimension]], axis=1)
                mean = head.predict(features).reshape(-1)
                generated[:, dimension] = mean + rng.normal(
                    scale=member.residual_std[dimension], size=len(rows)
                )
            result[rows] = self.y_scaler.inverse(generated)
        return result


class _ContrastiveEnergyModel:
    """B22-inspired conditional RFF energy with Langevin contrastive fitting.

    The trainable energy head is deliberately small: only ``theta`` is
    optimized while the random Fourier basis and conditional center are fixed.
    This keeps the method NumPy-only, but still permits an exact analytic VJP
    of the B22 gradient penalty into every trainable energy parameter.  It is a
    method-level protocol adaptation, not the upstream four-layer MLP.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        ridge: float,
        energy_features: int,
        negatives: int,
        contrastive_steps: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        temperature: float,
        training_langevin_steps: int,
        training_step_size_initial: float,
        training_step_size_final: float,
        training_noise_scale: float,
        training_gradient_clip: float,
        training_drift_clip: float,
        training_sample_clip: float,
        gradient_penalty_margin: float,
        gradient_penalty_weight: float,
        langevin_steps: int,
        langevin_step_size: float,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.ridge = float(ridge)
        self.energy_features = int(energy_features)
        self.negatives = int(negatives)
        self.contrastive_steps = int(contrastive_steps)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.training_langevin_steps = int(training_langevin_steps)
        self.training_step_size_initial = float(training_step_size_initial)
        self.training_step_size_final = float(training_step_size_final)
        self.training_noise_scale = float(training_noise_scale)
        self.training_gradient_clip = float(training_gradient_clip)
        self.training_drift_clip = float(training_drift_clip)
        self.training_sample_clip = float(training_sample_clip)
        self.gradient_penalty_margin = float(gradient_penalty_margin)
        self.gradient_penalty_weight = float(gradient_penalty_weight)
        self.langevin_steps = int(langevin_steps)
        self.langevin_step_size = float(langevin_step_size)
        if min(
            self.energy_features,
            self.negatives,
            self.contrastive_steps,
            self.epochs,
            self.batch_size,
            self.training_langevin_steps,
            self.langevin_steps,
        ) <= 0:
            raise ValueError("energy feature, negative, optimization, and Langevin counts must be positive")
        self.x_scaler: _Standardizer | None = None
        self.y_scaler: _Standardizer | None = None
        self.center: _RandomFeatureRidge | None = None
        self.residual_scale: np.ndarray | None = None
        self.rff_weight: np.ndarray | None = None
        self.rff_bias: np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    @staticmethod
    def _require_finite(stage: str, *values: Any) -> None:
        if any(not np.isfinite(np.asarray(value, dtype=float)).all() for value in values):
            raise FloatingPointError(f"ETM {stage} became non-finite")

    def _phi(self, x: np.ndarray, residual: np.ndarray) -> np.ndarray:
        if self.rff_weight is None or self.rff_bias is None:
            raise RuntimeError("energy features are not initialized")
        joint = np.concatenate([x, residual], axis=-1)
        return np.sqrt(2.0 / self.energy_features) * np.cos(
            joint @ self.rff_weight + self.rff_bias
        )

    def _energy_candidates(self, x: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.center is None or self.residual_scale is None or self.theta is None:
            raise RuntimeError("energy model is not fitted")
        center = self.center.predict(x)
        residual = (candidates - center[:, None, :]) / self.residual_scale
        x_repeated = np.broadcast_to(x[:, None, :], (*residual.shape[:2], x.shape[1]))
        phi = self._phi(
            x_repeated.reshape(-1, x.shape[1]),
            residual.reshape(-1, residual.shape[-1]),
        ).reshape(len(x), candidates.shape[1], -1)
        energy = 0.5 * np.sum(residual**2, axis=2) + phi @ self.theta
        return energy, phi

    def _nce_loss(self, energy: np.ndarray) -> tuple[float, np.ndarray]:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = -energy / self.temperature
            maximum = np.max(logits, axis=1, keepdims=True)
            exponent = np.exp(logits - maximum)
            probabilities = exponent / np.sum(exponent, axis=1, keepdims=True)
            loss = float(
                np.mean(
                    energy[:, 0] / self.temperature
                    + maximum[:, 0]
                    + np.log(np.sum(exponent, axis=1))
                )
            )
        return loss, probabilities

    def _energy_gradient(
        self, x: np.ndarray, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return dE/dy and RFF angles for a batched candidate panel."""

        if (
            self.center is None
            or self.residual_scale is None
            or self.rff_weight is None
            or self.rff_bias is None
            or self.theta is None
        ):
            raise RuntimeError("energy model is not fitted")
        center = self.center.predict(x)
        residual = (candidates - center[:, None, :]) / self.residual_scale
        x_repeated = np.broadcast_to(x[:, None, :], (*residual.shape[:2], x.shape[1]))
        joint = np.concatenate([x_repeated, residual], axis=-1)
        angle = joint @ self.rff_weight + self.rff_bias
        target_weight = self.rff_weight[x.shape[1] :, :]
        feature_scale = np.sqrt(2.0 / self.energy_features)
        gradient_residual = residual - feature_scale * (
            np.sin(angle) * self.theta
        ) @ target_weight.T
        return gradient_residual / self.residual_scale, angle

    def _gradient_penalty(
        self, x: np.ndarray, candidates: np.ndarray
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        """Evaluate B22 Eq. (22) and its exact VJP into the RFF head.

        Equation (22) sums penalties across the positive and generated samples
        for one condition; this implementation then averages those per-condition
        sums over the mini-batch.  The upstream release instead averages across
        candidates, a documented scale difference of ``1 / (K + 1)``.
        """

        if self.residual_scale is None or self.rff_weight is None:
            raise RuntimeError("energy model is not fitted")
        gradient_y, angle = self._energy_gradient(x, candidates)
        gradient_norm = np.linalg.norm(gradient_y, axis=-1)
        excess = np.maximum(gradient_norm - self.gradient_penalty_margin, 0.0)
        penalty = excess**2

        # g = (r - c * (sin(z) * theta) B.T) / scale, hence
        # J_theta(g)^T v = -c * sin(z) * ((v / scale) B).
        safe_norm = np.maximum(gradient_norm, 1e-12)
        direction = gradient_y / safe_norm[..., None]
        target_weight = self.rff_weight[x.shape[1] :, :]
        feature_scale = np.sqrt(2.0 / self.energy_features)
        vjp = -feature_scale * np.sin(angle) * (
            (direction / self.residual_scale) @ target_weight
        )
        sample_gradient = 2.0 * excess[..., None] * vjp
        parameter_gradient = np.mean(np.sum(sample_gradient, axis=1), axis=0)
        summary = {
            "loss": float(np.mean(np.sum(penalty, axis=1))),
            "gradient_norm_mean": float(np.mean(gradient_norm)),
            "gradient_norm_max": float(np.max(gradient_norm)),
            "active_rate": float(np.mean(excess > 0.0)),
        }
        return summary["loss"], parameter_gradient, summary

    def _training_step_size(self, step: int) -> float:
        if self.training_langevin_steps == 1:
            return self.training_step_size_initial
        fraction = float(step) / float(self.training_langevin_steps - 1)
        return self.training_step_size_final + (
            self.training_step_size_initial - self.training_step_size_final
        ) * (1.0 - fraction) ** 2

    def _langevin_negatives(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate conditional training negatives using the release-code chain."""

        if self.center is None:
            raise RuntimeError("energy model is not fitted")
        output_dim = self.center.output_dim
        negative = rng.uniform(
            -1.0, 1.0, size=(len(x), self.negatives, output_dim)
        )
        initial = negative.copy()
        initial_energy, _ = self._energy_candidates(x, initial)
        self._require_finite("Langevin initialization", initial, initial_energy)
        for step in range(self.training_langevin_steps):
            gradient, _ = self._energy_gradient(x, negative)
            self._require_finite("Langevin gradient", gradient)
            gradient = np.clip(
                gradient,
                -self.training_gradient_clip,
                self.training_gradient_clip,
            )
            noise = rng.normal(size=negative.shape)
            drift = self._training_step_size(step) * (
                0.5 * gradient + self.training_noise_scale * noise
            )
            drift = np.clip(
                drift, -self.training_drift_clip, self.training_drift_clip
            )
            negative = np.clip(
                negative - drift,
                -self.training_sample_clip,
                self.training_sample_clip,
            )
            self._require_finite("Langevin update", drift, negative)
            # NumPy has no autograd tape: using the value in the next iteration
            # exactly realizes the upstream detach-after-each-step convention.
        final_energy, _ = self._energy_candidates(x, negative)
        self._require_finite("Langevin final energy", final_energy)
        trace = {
            "start_energy_mean": float(np.mean(initial_energy)),
            "end_energy_mean": float(np.mean(final_energy)),
            "mean_displacement": float(
                np.mean(np.linalg.norm(negative - initial, axis=-1))
            ),
            "start_sample_mean": float(np.mean(initial)),
            "start_sample_std": float(np.std(initial)),
            "end_sample_mean": float(np.mean(negative)),
            "end_sample_std": float(np.std(negative)),
        }
        return negative, trace

    def fit(self, inputs: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> None:
        fit_started = perf_counter()
        self.x_scaler = _Standardizer.fit(inputs)
        target_scale = np.max(np.abs(targets), axis=0)
        target_scale = np.where(target_scale < 1e-12, 1.0, target_scale)
        self.y_scaler = _Standardizer(np.zeros_like(target_scale), target_scale)
        x = self.x_scaler.transform(inputs)
        y = self.y_scaler.transform(targets)
        self.center = _RandomFeatureRidge(
            x.shape[1], y.shape[1], self.hidden_dim, self.ridge, rng
        ).fit(x, y)
        center = self.center.predict(x)
        residual = y - center
        self.residual_scale = np.maximum(np.std(residual, axis=0), 0.15)
        joint_dim = x.shape[1] + y.shape[1]
        self.rff_weight = rng.normal(
            scale=1.0 / np.sqrt(max(joint_dim, 1)),
            size=(joint_dim, self.energy_features),
        )
        self.rff_bias = rng.uniform(-np.pi, np.pi, size=self.energy_features)
        self.theta = np.zeros(self.energy_features, dtype=float)
        first_loss: float | None = None
        last_loss = float("nan")
        last_gp_loss = float("nan")
        first_moment = np.zeros_like(self.theta)
        second_moment = np.zeros_like(self.theta)
        negative_sampling_seconds = 0.0
        gradient_penalty_seconds = 0.0
        backward_seconds = 0.0
        nce_seconds = 0.0
        optimizer_updates = 0
        epochs_entered = 0
        condition_rows = 0
        chain_start_energy: list[float] = []
        chain_end_energy: list[float] = []
        chain_displacement: list[float] = []
        grad_norm_means: list[float] = []
        grad_norm_maxes: list[float] = []
        penalty_active_rates: list[float] = []
        latest_trace: dict[str, Any] = {}

        for epoch in range(self.epochs):
            permutation = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                if optimizer_updates >= self.contrastive_steps:
                    break
                rows = permutation[start : start + self.batch_size]
                x_batch = x[rows]
                y_batch = y[rows]

                negative_started = perf_counter()
                negatives, latest_trace = self._langevin_negatives(x_batch, rng)
                negative_sampling_seconds += perf_counter() - negative_started
                candidates = np.concatenate([y_batch[:, None, :], negatives], axis=1)

                nce_started = perf_counter()
                energy, phi = self._energy_candidates(x_batch, candidates)
                loss, probability = self._nce_loss(energy)
                self._require_finite("InfoNCE", energy, phi, loss, probability)
                nce_gradient = np.mean(
                    (
                        phi[:, 0]
                        - np.sum(probability[:, :, None] * phi, axis=1)
                    )
                    / self.temperature,
                    axis=0,
                )
                self._require_finite("InfoNCE gradient", nce_gradient)
                nce_seconds += perf_counter() - nce_started

                gp_started = perf_counter()
                gp_loss, gp_gradient, gp_summary = self._gradient_penalty(
                    x_batch, candidates
                )
                self._require_finite(
                    "gradient penalty",
                    gp_loss,
                    gp_gradient,
                    list(gp_summary.values()),
                )
                gradient_penalty_seconds += perf_counter() - gp_started

                backward_started = perf_counter()
                gradient = (
                    nce_gradient
                    + self.gradient_penalty_weight * gp_gradient
                    + self.ridge * self.theta
                )
                self._require_finite("parameter gradient", gradient)
                optimizer_updates += 1
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    first_moment = 0.9 * first_moment + 0.1 * gradient
                    second_moment = 0.999 * second_moment + 0.001 * gradient**2
                    corrected_first = first_moment / (1.0 - 0.9**optimizer_updates)
                    corrected_second = second_moment / (1.0 - 0.999**optimizer_updates)
                self._require_finite(
                    "Adam state",
                    first_moment,
                    second_moment,
                    corrected_first,
                    corrected_second,
                )
                self.theta -= self.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
                self._require_finite("parameter update", self.theta)
                backward_seconds += perf_counter() - backward_started

                if first_loss is None:
                    first_loss = loss
                last_loss = loss
                last_gp_loss = gp_loss
                condition_rows += len(rows)
                chain_start_energy.append(latest_trace["start_energy_mean"])
                chain_end_energy.append(latest_trace["end_energy_mean"])
                chain_displacement.append(latest_trace["mean_displacement"])
                grad_norm_means.append(gp_summary["gradient_norm_mean"])
                grad_norm_maxes.append(gp_summary["gradient_norm_max"])
                penalty_active_rates.append(gp_summary["active_rate"])
            epochs_entered = epoch + 1
            if optimizer_updates >= self.contrastive_steps:
                break
        if optimizer_updates == 0 or first_loss is None:
            raise RuntimeError("ETM training produced no optimizer update")

        # Evaluate a fresh conditional Langevin panel after the final update.
        panel_rows = np.arange(min(len(y), self.batch_size))
        panel_x = x[panel_rows]
        panel_y = y[panel_rows]
        panel_negatives, panel_trace = self._langevin_negatives(panel_x, rng)
        candidates = np.concatenate([panel_y[:, None, :], panel_negatives], axis=1)
        energy, _ = self._energy_candidates(panel_x, candidates)
        final_loss, _ = self._nce_loss(energy)
        energy_gap = float(np.mean(energy[:, 1:]) - np.mean(energy[:, 0]))
        self._require_finite("final contrastive panel", energy, final_loss, energy_gap)
        self.diagnostics = {
            "conditional_energy": "quadratic_base_plus_learned_joint_RFF_energy",
            "contrastive_objective": "InfoNCE_with_conditional_Langevin_negatives",
            "training_negative_sampler": "conditional_batched_Langevin",
            "shuffle_or_replay_negatives_used": False,
            "negative_initialization": "uniform[-1,1]_normalized_target",
            "contrastive_steps": self.contrastive_steps,
            "optimizer_update_count": optimizer_updates,
            "configured_epochs": self.epochs,
            "epochs_entered": epochs_entered,
            "batch_size": self.batch_size,
            "contrastive_loss_initial": float(first_loss),
            "contrastive_loss_last_training_batch": float(last_loss),
            "contrastive_loss_final_panel": final_loss,
            "positive_negative_energy_gap": energy_gap,
            "negative_candidates": self.negatives,
            "energy_features": self.energy_features,
            "softmax_temperature": self.temperature,
            "training_langevin_steps": self.training_langevin_steps,
            "training_step_size_initial": self.training_step_size_initial,
            "training_step_size_final": self.training_step_size_final,
            "training_step_schedule": "official_release_polynomial_power_2",
            "training_langevin_noise_scale": self.training_noise_scale,
            "training_gradient_clip": self.training_gradient_clip,
            "training_drift_clip": self.training_drift_clip,
            "training_sample_clip": self.training_sample_clip,
            "training_chain_start_energy_mean": float(np.mean(chain_start_energy)),
            "training_chain_end_energy_mean": float(np.mean(chain_end_energy)),
            "training_chain_mean_displacement": float(np.mean(chain_displacement)),
            "final_panel_chain_start_energy": panel_trace["start_energy_mean"],
            "final_panel_chain_end_energy": panel_trace["end_energy_mean"],
            "final_panel_chain_mean_displacement": panel_trace["mean_displacement"],
            "langevin_negative_chain_count": condition_rows * self.negatives,
            "langevin_negative_step_count": condition_rows
            * self.negatives
            * self.training_langevin_steps,
            "gradient_penalty_formula": "sum_j_relu(norm_dE_dtarget_minus_margin)^2",
            "gradient_penalty_vjp": "exact_analytic_into_trainable_RFF_theta",
            "gradient_penalty_reduction": "sum_candidates_then_mean_batch",
            "gradient_penalty_margin": self.gradient_penalty_margin,
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "gradient_penalty_last": float(last_gp_loss),
            "gradient_norm_mean": float(np.mean(grad_norm_means)),
            "gradient_norm_max": float(np.max(grad_norm_maxes)),
            "gradient_penalty_active_rate": float(np.mean(penalty_active_rates)),
            "gradient_penalty_evaluation_count": condition_rows
            * (self.negatives + 1),
            "negative_sampling_seconds": negative_sampling_seconds,
            "nce_seconds": nce_seconds,
            "gradient_penalty_seconds": gradient_penalty_seconds,
            "analytic_backward_seconds": backward_seconds,
            "energy_fit_seconds": perf_counter() - fit_started,
            "theta_l2": float(np.linalg.norm(self.theta)),
            "actual_training_config": {
                "hidden_dim": self.hidden_dim,
                "ridge": self.ridge,
                "energy_features": self.energy_features,
                "generated_negative_chains_per_condition": self.negatives,
                "positive_candidates_per_condition": 1,
                "maximum_optimizer_updates": self.contrastive_steps,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "optimizer": "Adam",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "softmax_temperature": self.temperature,
                "target_normalization": "zero_mean_max_abs_box",
                "target_scale_floor_rule": "max_abs_below_1e-12_uses_1.0",
                "residual_scale_floor": 0.15,
                "training_langevin_steps": self.training_langevin_steps,
                "training_step_size_initial": self.training_step_size_initial,
                "training_step_size_final": self.training_step_size_final,
                "training_step_schedule": "polynomial_power_2",
                "training_noise_scale": self.training_noise_scale,
                "training_gradient_clip": self.training_gradient_clip,
                "training_drift_clip": self.training_drift_clip,
                "training_sample_clip": self.training_sample_clip,
                "gradient_penalty_margin": self.gradient_penalty_margin,
                "gradient_penalty_weight": self.gradient_penalty_weight,
                "gradient_penalty_reduction": "sum_candidates_then_mean_batch",
                "inference_langevin_steps": self.langevin_steps,
                "inference_langevin_step_size": self.langevin_step_size,
                "inference_noise_scale": 1.0,
                "inference_sample_clip": 10.0,
            },
            "paper_protocol_reference": "B22_2024_Eq22_Eq23_Table2",
            "official_source_commit": "2a2c780c0da074b6e7733a3cb6b40b2444452de6",
            "remaining_upstream_drift": [
                "fixed_random_Fourier_basis_and_ridge_center_replace_four_layer_trainable_MLP",
                "training chain selects the official release polynomial schedule rather than paper Eq23 epsilon-squared drift",
                "no_claim_of_official_benchmark_numerical_parity",
            ],
            "inference_sampler": "Langevin",
            "langevin_steps": self.langevin_steps,
            "langevin_step_size": self.langevin_step_size,
            "target_parameterization": "delta_state_then_reward",
        }

    def sample(
        self,
        inputs: np.ndarray,
        member_index: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del member_index  # Energy model is a single conditional density, not an ensemble.
        required = (
            self.x_scaler,
            self.y_scaler,
            self.center,
            self.residual_scale,
            self.rff_weight,
            self.rff_bias,
            self.theta,
        )
        if any(item is None for item in required):
            raise RuntimeError("energy model is not fitted")
        assert self.x_scaler is not None
        assert self.y_scaler is not None
        assert self.center is not None
        assert self.residual_scale is not None
        assert self.rff_weight is not None
        assert self.rff_bias is not None
        assert self.theta is not None
        x = self.x_scaler.transform(inputs)
        center = self.center.predict(x)
        y = center + rng.normal(scale=self.residual_scale, size=center.shape)
        y_weight = self.rff_weight[x.shape[1] :, :]
        feature_scale = np.sqrt(2.0 / self.energy_features)
        for _ in range(self.langevin_steps):
            residual = (y - center) / self.residual_scale
            joint = np.concatenate([x, residual], axis=1)
            angle = joint @ self.rff_weight + self.rff_bias
            energy_gradient_residual = residual - feature_scale * (
                np.sin(angle) * self.theta
            ) @ y_weight.T
            gradient_y = energy_gradient_residual / self.residual_scale
            noise = rng.normal(size=y.shape)
            y -= 0.5 * self.langevin_step_size**2 * gradient_y
            y += self.langevin_step_size * noise
            y = np.clip(y, -10.0, 10.0)
        return self.y_scaler.inverse(y)


def _sample_actor(
    actor: Any,
    observations: np.ndarray,
    native_timestep: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    if not hasattr(actor, "sample_actions"):
        raise TypeError("candidate actor must implement sample_actions(..., keys=...)")
    actions = candidate_actions(
        actor,
        observations,
        native_timestep,
        keys=np.asarray(keys, dtype=np.uint64),
        require_deterministic=False,
    )
    actions = _matrix(actions, "actor actions")
    if len(actions) != len(observations) or not np.isfinite(actions).all():
        raise ValueError("actor returned invalid actions")
    return actions


def _make_value_estimate(**fields: Any) -> Any:
    return ValueEstimate(**fields)


class _BaseMBOPE:
    method_family = "BASE_MBOPE"
    identity: Mapping[str, Any] = {}
    member_count = 1

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        rollouts_per_initial: int = 4,
        ridge: float = 1e-4,
        termination_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        learn_termination: bool = False,
    ) -> None:
        if (
            isinstance(gamma, (bool, np.bool_))
            or not isinstance(gamma, (int, float, np.integer, np.floating))
            or not np.isfinite(float(gamma))
            or not 0.0 <= float(gamma) <= 1.0
        ):
            raise ValueError("gamma must be in [0, 1]")
        if (
            isinstance(horizon, (bool, np.bool_))
            or int(horizon) != horizon
            or int(horizon) <= 0
            or isinstance(rollouts_per_initial, (bool, np.bool_))
            or int(rollouts_per_initial) != rollouts_per_initial
            or int(rollouts_per_initial) <= 0
        ):
            raise ValueError("horizon and rollouts_per_initial must be positive integers")
        if (
            isinstance(ridge, (bool, np.bool_))
            or not isinstance(ridge, (int, float, np.integer, np.floating))
            or not np.isfinite(float(ridge))
            or float(ridge) <= 0.0
        ):
            raise ValueError("ridge must be finite and positive")
        if not isinstance(learn_termination, (bool, np.bool_)):
            raise ValueError("learn_termination must be boolean")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.method_id = finite_horizon_method_id(
            self.method_family, self.gamma, self.horizon
        )
        self.rollouts_per_initial = int(rollouts_per_initial)
        self.ridge = float(ridge)
        if termination_function is not None and learn_termination:
            raise ValueError("choose a known termination function or learned termination, not both")
        self.termination_function = termination_function
        self.learn_termination = bool(learn_termination)
        # Learned termination is a scientifically distinct method variant and
        # must never be silently published under one of the five frozen IDs.
        if self.learn_termination:
            self.method_id = f"{self.method_id}_TLEARNED"
        self._termination_mode = (
            "learned"
            if self.learn_termination
            else "known_function"
            if self.termination_function is not None
            else "horizon_only"
        )
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self._actor: Any = None
        self._model: Any = None
        self._termination: _TerminationHead | None = None
        self._fit_diagnostics: dict[str, Any] = {}
        self._fit_seconds = 0.0
        self._fit_rows = 0
        self._fit_key_digest = ""
        self._observation_dim = 0
        self._action_dim = 0
        self._behavior_action_mean: np.ndarray | None = None
        self._behavior_action_scale: np.ndarray | None = None
        self._timestep_provenance = ""
        self._physical_membership_sha256 = ""
        self._source_digest: str | None = None

    def _build_model(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    def fit(
        self, batch: TransitionBatch, candidate: Any, *, fit_keys: Any
    ) -> "_BaseMBOPE":
        # A failed refit must invalidate the previous estimator instead of
        # leaving a mixture of old provenance and a newly overwritten model.
        self._reset_fit_state()
        started = perf_counter()
        data = _extract_batch(batch, self.horizon)
        policy_id(candidate)
        policy_semantics(candidate)
        fit_key_array = _key_array(fit_keys, "fit_keys")
        rng = np.random.default_rng(_seed_from_keys(fit_key_array, 0x4D424F5045))
        valid = data.valid
        inputs = _transition_inputs(
            data.observation[valid],
            data.action[valid],
            data.native_timestep[valid],
            self.horizon,
        )
        targets = np.concatenate(
            [
                data.next_observation[valid] - data.observation[valid],
                data.reward[valid, None],
            ],
            axis=1,
        )
        model = self._build_model(rng)
        model.fit(inputs, targets, rng)
        native_environment_termination = bool(np.any(data.environment_terminal[valid]))
        termination: _TerminationHead | None = None
        if self.learn_termination:
            termination = _TerminationHead(self.ridge)
            termination.fit(inputs, data.environment_terminal[valid], rng)
        elif self.termination_function is None and native_environment_termination:
            from .core import DataValidationError, EstimateStatus

            raise DataValidationError(
                EstimateStatus.AMBIGUOUS_TERMINATION.value,
                "native environment termination requires a shared known functional or "
                "a separately identified learn_termination=True variant",
            )
        behavior_action_mean = np.mean(data.action[valid], axis=0)
        behavior_action_scale = np.std(data.action[valid], axis=0)
        behavior_action_scale[behavior_action_scale < 1e-6] = 1.0
        fit_rows = int(np.sum(valid))
        fit_key_digest = _key_digest(fit_key_array)
        fit_seconds = perf_counter() - started
        fit_diagnostics = {
            **model.diagnostics,
            "method_identity": dict(self.identity),
            "native_timestep_used": True,
            "timestep_provenance": data.timestep_provenance,
            "dataset_cut_rows_treated_as_nonterminal": int(np.sum(data.dataset_cut & valid)),
            "environment_terminal_rows": int(np.sum(data.environment_terminal & valid)),
            "truncation_rows_not_relabeled_terminal": int(np.sum(data.truncated & valid)),
            "termination_contract": self._termination_mode,
            "learned_termination_method_suffix": self.learn_termination,
            "fit_rows": fit_rows,
            "physical_membership_sha256": data.physical_membership_sha256,
            "actor_queries_during_fit": 0,
        }

        # Publish fitted state only after every gate and derived field above
        # has succeeded.
        self._model = model
        self._termination = termination
        self._actor = candidate
        self._observation_dim = data.observation.shape[1]
        self._action_dim = data.action.shape[1]
        self._behavior_action_mean = behavior_action_mean
        self._behavior_action_scale = behavior_action_scale
        self._fit_rows = fit_rows
        self._fit_key_digest = fit_key_digest
        self._timestep_provenance = data.timestep_provenance
        self._physical_membership_sha256 = data.physical_membership_sha256
        self._source_digest = data.source_digest
        self._fit_seconds = fit_seconds
        self._fit_diagnostics = fit_diagnostics
        return self

    def estimate(
        self,
        initial_observations: Any,
        *,
        keys: Any,
        initial_timestep: int | Sequence[int] | np.ndarray = 0,
        candidate: Any | None = None,
    ) -> Any:
        if self._model is None:
            raise RuntimeError("fit must be called before estimate")
        actor = self._actor if candidate is None else candidate
        if actor is None:
            raise ValueError("a candidate actor is required")
        actor_id = policy_id(actor)
        actor_semantics = policy_semantics(actor).value
        initial = _matrix(initial_observations, "initial_observations")
        if initial.shape[1] != self._observation_dim or not np.isfinite(initial).all():
            raise ValueError("initial observation ABI differs from fitted data")
        root_keys = _key_array(keys, "keys", rows=len(initial))
        initial_time_raw = np.asarray(initial_timestep)
        if initial_time_raw.dtype.kind not in "iu" or initial_time_raw.dtype.kind == "b":
            raise ValueError("initial_timestep must contain integers")
        if initial_time_raw.ndim == 0:
            initial_time = np.full(len(initial), int(initial_time_raw), dtype=np.int64)
        else:
            if initial_time_raw.ndim != 1:
                raise ValueError("initial_timestep must be scalar or one-dimensional")
            initial_time = initial_time_raw.astype(np.int64, copy=False)
        if len(initial_time) != len(initial):
            raise ValueError("initial_timestep must be scalar or match initial observations")
        if np.any(initial_time < 0) or np.any(initial_time >= self.horizon):
            raise ValueError("initial_timestep lies outside the configured finite horizon")
        started = perf_counter()
        repeats = self.rollouts_per_initial
        state = np.repeat(initial, repeats, axis=0)
        timestep = np.repeat(initial_time, repeats)
        rollout_roots = np.asarray(
            [
                _mix64(int(key) ^ _mix64(replica + 0x524F4C4C))
                for key in root_keys
                for replica in range(repeats)
            ],
            dtype=np.uint64,
        )
        rng = np.random.default_rng(_seed_from_keys(rollout_roots, 0x4D4F44454C))
        member_index = np.asarray(
            [_mix64(int(key) ^ 0x454E53454D424C45) % self.member_count for key in rollout_roots],
            dtype=np.int64,
        )
        returns = np.zeros(len(state), dtype=float)
        discounts = np.ones(len(state), dtype=float)
        lengths = np.zeros(len(state), dtype=np.int64)
        alive = timestep < self.horizon
        terminal_count = 0
        actor_rows = 0
        actor_calls = 0
        action_z_sum = 0.0
        action_z_count = 0
        action_z_max = 0.0
        maximum_absolute_state = float(np.max(np.abs(state)))
        while np.any(alive):
            rows = np.flatnonzero(alive)
            action_keys = np.asarray(
                [
                    _mix64(int(rollout_roots[row]) ^ _mix64(int(timestep[row]) + 0x414354))
                    for row in rows
                ],
                dtype=np.uint64,
            )
            actions = _sample_actor(actor, state[rows], timestep[rows], action_keys)
            if actions.shape[1] != self._action_dim:
                raise ValueError("actor action ABI differs from fitted data")
            if self._behavior_action_mean is None or self._behavior_action_scale is None:
                raise RuntimeError("behavior action support summary is missing")
            action_z = np.linalg.norm(
                (actions - self._behavior_action_mean) / self._behavior_action_scale,
                axis=1,
            )
            action_z_sum += float(np.sum(action_z))
            action_z_count += len(action_z)
            action_z_max = max(action_z_max, float(np.max(action_z)))
            inputs = _transition_inputs(state[rows], actions, timestep[rows], self.horizon)
            generated = self._model.sample(inputs, member_index[rows], rng)
            if generated.shape != (len(rows), self._observation_dim + 1):
                raise RuntimeError("learned model returned an invalid transition shape")
            delta = generated[:, : self._observation_dim]
            reward = generated[:, -1]
            if not np.isfinite(generated).all():
                raise RuntimeError("learned model rollout became non-finite")
            if self._termination_mode == "learned":
                assert self._termination is not None
                environment_terminal = self._termination.probability(inputs) >= 0.5
            elif self._termination_mode == "known_function":
                assert self.termination_function is not None
                environment_terminal = np.asarray(
                    self.termination_function(state[rows] + delta, timestep[rows] + 1),
                    dtype=bool,
                ).reshape(-1)
                if len(environment_terminal) != len(rows):
                    raise ValueError("known termination function must return one boolean per row")
            else:
                environment_terminal = np.zeros(len(rows), dtype=bool)
            returns[rows] += discounts[rows] * reward
            discounts[rows] *= self.gamma
            state[rows] += delta
            timestep[rows] += 1
            lengths[rows] += 1
            maximum_absolute_state = max(
                maximum_absolute_state, float(np.max(np.abs(state[rows])))
            )
            terminal_count += int(np.sum(environment_terminal))
            actor_rows += len(rows)
            actor_calls += 1
            alive[rows] = ~environment_terminal & (timestep[rows] < self.horizon)
        estimate_seconds = perf_counter() - started
        diagnostics = {
            **self._fit_diagnostics,
            "rollout_return_std": float(np.std(returns)),
            "rollout_return_min": float(np.min(returns)),
            "rollout_return_max": float(np.max(returns)),
            "rollout_count": len(returns),
            "mean_rollout_length": float(np.mean(lengths)),
            "max_rollout_length": int(np.max(lengths)),
            "learned_environment_termination_count": terminal_count,
            "maximum_absolute_model_state": maximum_absolute_state,
            "actor_calls": actor_calls,
            "actor_rows": actor_rows,
            "actor_queries_only_inside_learned_model": True,
            "action_keys_explicit_for_every_row": True,
            "rollout_key_digest": _key_digest(root_keys),
            "runtime_seconds": float(self._fit_seconds + estimate_seconds),
        }
        support = {
            "behavior_density_required": False,
            "kind": "learned_model_rollout_global_behavior_action_z",
            "actor_semantics": actor_semantics,
            "rollout_action_z_mean": action_z_sum / max(action_z_count, 1),
            "rollout_action_z_max": action_z_max,
            "support_rows": action_z_count,
        }
        provenance = {
            "candidate_id": actor_id,
            "implementation": "policy_learnware_ope.mbope",
            "fit_key_digest": self._fit_key_digest,
            "native_timestep_provenance": self._timestep_provenance,
            "physical_membership_sha256": self._physical_membership_sha256,
            "source_digest": self._source_digest,
            "value_convention": _value_convention(self.gamma, self.horizon),
            "scientific_role": self.identity["scientific_role"],
            "upstream_parity_claim": "NONE",
        }
        cost = {
            "fit_seconds": float(self._fit_seconds),
            "estimate_seconds": float(estimate_seconds),
            "fit_transitions": self._fit_rows,
            "model_rollout_steps": actor_rows,
            "actor_queries": actor_rows,
        }
        return _make_value_estimate(
            method_id=self.method_id,
            status="PASS",
            value=float(np.mean(returns)),
            support=support,
            provenance=provenance,
            cost=cost,
            diagnostics=diagnostics,
        )


class DOPEStyleMBFFEstimator(_BaseMBOPE):
    """Project-defined feed-forward MB-OPE; DOPE is benchmark inspiration only."""

    method_family = "DOPE_STYLE_MB_FF"
    identity = {
        "family": "feed_forward_model_based_ope",
        "scientific_role": "PROJECT_DEFINED_REFERENCE",
        "project_defined": True,
        "implementation_scope": "EXECUTABLE_PROJECT_REFERENCE",
        "full_2x256_SiLU_diagonal_NLL_port": False,
        "dope_is_benchmark_inspiration_not_algorithm": True,
        "upstream_parity_claimed": False,
        "distinct_from_families": ["AR_MBOPE", "ETM_MBOPE"],
    }

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        rollouts_per_initial: int = 4,
        ridge: float = 1e-4,
        ensemble_members: int = 5,
        hidden_dim: int = 48,
        termination_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        learn_termination: bool = False,
    ) -> None:
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            rollouts_per_initial=rollouts_per_initial,
            ridge=ridge,
            termination_function=termination_function,
            learn_termination=learn_termination,
        )
        if (
            isinstance(ensemble_members, (bool, np.bool_))
            or not isinstance(ensemble_members, (int, np.integer))
            or int(ensemble_members) <= 0
            or isinstance(hidden_dim, (bool, np.bool_))
            or not isinstance(hidden_dim, (int, np.integer))
            or int(hidden_dim) <= 0
        ):
            raise ValueError("ensemble_members and hidden_dim must be positive integers")
        self.member_count = int(ensemble_members)
        self.hidden_dim = int(hidden_dim)

    def _build_model(self, rng: np.random.Generator) -> _FeedForwardDeltaModel:
        del rng
        return _FeedForwardDeltaModel(
            members=self.member_count, hidden_dim=self.hidden_dim, ridge=self.ridge
        )


class ARMBOPEEstimator(_BaseMBOPE):
    """Independent B06-inspired autoregressive MB-OPE project adaptation."""

    method_family = "AR_MBOPE"
    identity = {
        "family": "fixed_order_autoregressive_model_based_ope",
        "scientific_role": "PROJECT_METHOD_LEVEL_ADAPTATION_PROXY",
        "project_adaptation": True,
        "implementation_scope": "EXECUTABLE_PROJECT_REFERENCE_PROXY",
        "upstream_parity_claimed": False,
        "training": "teacher_forcing",
        "generation": "sequential_per_dimension",
        "distinct_from_families": ["DOPE_STYLE_MB_FF", "ETM_MBOPE"],
    }

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        rollouts_per_initial: int = 4,
        ridge: float = 1e-4,
        ensemble_members: int = 3,
        hidden_dim: int = 32,
        termination_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        learn_termination: bool = False,
    ) -> None:
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            rollouts_per_initial=rollouts_per_initial,
            ridge=ridge,
            termination_function=termination_function,
            learn_termination=learn_termination,
        )
        if (
            isinstance(ensemble_members, (bool, np.bool_))
            or not isinstance(ensemble_members, (int, np.integer))
            or int(ensemble_members) <= 0
            or isinstance(hidden_dim, (bool, np.bool_))
            or not isinstance(hidden_dim, (int, np.integer))
            or int(hidden_dim) <= 0
        ):
            raise ValueError("ensemble_members and hidden_dim must be positive integers")
        self.member_count = int(ensemble_members)
        self.hidden_dim = int(hidden_dim)

    def _build_model(self, rng: np.random.Generator) -> _AutoregressiveDeltaModel:
        del rng
        return _AutoregressiveDeltaModel(
            members=self.member_count, hidden_dim=self.hidden_dim, ridge=self.ridge
        )


class ETMMBOPEEstimator(_BaseMBOPE):
    """B22 protocol adaptation with Langevin training and rollout."""

    method_family = "ETM_MBOPE"
    identity = {
        "family": "contrastive_energy_transition_model_ope",
        "scientific_role": "PROJECT_ETM_PROTOCOL_ADAPTATION",
        "project_adaptation": True,
        "implementation_scope": "EXECUTABLE_METHOD_LEVEL_PROTOCOL_ADAPTATION",
        "upstream_parity_claimed": False,
        "fit": "conditional_Langevin_InfoNCE_plus_exact_RFF_gradient_penalty_VJP",
        "generation": "Langevin_energy_sampling",
        "upstream_source_commit": "2a2c780c0da074b6e7733a3cb6b40b2444452de6",
        "distinct_from_families": ["DOPE_STYLE_MB_FF", "AR_MBOPE"],
    }
    member_count = 1

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        horizon: int = 1000,
        rollouts_per_initial: int = 4,
        ridge: float = 1e-4,
        hidden_dim: int = 32,
        energy_features: int = 64,
        negatives: int = 4,
        contrastive_steps: int = 80,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.02,
        temperature: float = 1.0,
        training_langevin_steps: int = 5,
        training_step_size_initial: float = 0.1,
        training_step_size_final: float = 1e-3,
        training_noise_scale: float = 0.5,
        training_gradient_clip: float = 10.0,
        training_drift_clip: float = 0.5,
        training_sample_clip: float = 1.1,
        gradient_penalty_margin: float = 5.0,
        gradient_penalty_weight: float = 1.0,
        langevin_steps: int = 20,
        langevin_step_size: float = 0.04,
        termination_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        learn_termination: bool = False,
    ) -> None:
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            rollouts_per_initial=rollouts_per_initial,
            ridge=ridge,
            termination_function=termination_function,
            learn_termination=learn_termination,
        )
        positive_integers = {
            "hidden_dim": hidden_dim,
            "energy_features": energy_features,
            "negatives": negatives,
            "contrastive_steps": contrastive_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "training_langevin_steps": training_langevin_steps,
            "langevin_steps": langevin_steps,
        }
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in positive_integers.values()
        ):
            raise ValueError("ETM dimensions and step counts must be positive integers")
        for name, value in {
            "learning_rate": learning_rate,
            "temperature": temperature,
            "training_step_size_initial": training_step_size_initial,
            "training_step_size_final": training_step_size_final,
            "training_gradient_clip": training_gradient_clip,
            "training_drift_clip": training_drift_clip,
            "training_sample_clip": training_sample_clip,
            "langevin_step_size": langevin_step_size,
        }.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {
            "training_noise_scale": training_noise_scale,
            "gradient_penalty_margin": gradient_penalty_margin,
            "gradient_penalty_weight": gradient_penalty_weight,
        }.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if float(training_step_size_final) > float(training_step_size_initial):
            raise ValueError(
                "training_step_size_final must not exceed training_step_size_initial"
            )
        self.hidden_dim = int(hidden_dim)
        self.energy_features = int(energy_features)
        self.negatives = int(negatives)
        self.contrastive_steps = int(contrastive_steps)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.training_langevin_steps = int(training_langevin_steps)
        self.training_step_size_initial = float(training_step_size_initial)
        self.training_step_size_final = float(training_step_size_final)
        self.training_noise_scale = float(training_noise_scale)
        self.training_gradient_clip = float(training_gradient_clip)
        self.training_drift_clip = float(training_drift_clip)
        self.training_sample_clip = float(training_sample_clip)
        self.gradient_penalty_margin = float(gradient_penalty_margin)
        self.gradient_penalty_weight = float(gradient_penalty_weight)
        self.langevin_steps = int(langevin_steps)
        self.langevin_step_size = float(langevin_step_size)

    def _build_model(self, rng: np.random.Generator) -> _ContrastiveEnergyModel:
        del rng
        return _ContrastiveEnergyModel(
            hidden_dim=self.hidden_dim,
            ridge=self.ridge,
            energy_features=self.energy_features,
            negatives=self.negatives,
            contrastive_steps=self.contrastive_steps,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            training_langevin_steps=self.training_langevin_steps,
            training_step_size_initial=self.training_step_size_initial,
            training_step_size_final=self.training_step_size_final,
            training_noise_scale=self.training_noise_scale,
            training_gradient_clip=self.training_gradient_clip,
            training_drift_clip=self.training_drift_clip,
            training_sample_clip=self.training_sample_clip,
            gradient_penalty_margin=self.gradient_penalty_margin,
            gradient_penalty_weight=self.gradient_penalty_weight,
            langevin_steps=self.langevin_steps,
            langevin_step_size=self.langevin_step_size,
        )


def make_model_based_estimator(method_id: str, **kwargs: Any) -> _BaseMBOPE:
    # The CLI needs one selector; keeping the three-class map local avoids a
    # second public registry and redundant exact-ID class aliases.
    estimator_types = {
        DOPE_STYLE_MB_FF_ID: DOPEStyleMBFFEstimator,
        AR_MBOPE_ID: ARMBOPEEstimator,
        ETM_MBOPE_ID: ETMMBOPEEstimator,
    }
    try:
        estimator = estimator_types[method_id]
    except KeyError as exc:
        raise ValueError(f"unknown model-based method: {method_id}") from exc
    return estimator(**kwargs)


__all__ = [
    "AR_MBOPE_ID",
    "ARMBOPEEstimator",
    "DOPE_STYLE_MB_FF_ID",
    "DOPEStyleMBFFEstimator",
    "ETM_MBOPE_ID",
    "ETMMBOPEEstimator",
    "make_model_based_estimator",
]
