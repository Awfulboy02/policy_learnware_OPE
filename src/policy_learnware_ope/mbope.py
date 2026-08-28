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


DOPE_STYLE_MB_FF_ID = "DOPE_STYLE_MB_FF_G099_H1000"
AR_MBOPE_ID = "AR_MBOPE_G099_H1000"
ETM_MBOPE_ID = "ETM_MBOPE_G099_H1000"
VALUE_CONVENTION = "J_gamma=0.99_H=1000_raw"


def _value_convention(gamma: float, horizon: int) -> str:
    return f"J_gamma={float(gamma):.6g}_H={int(horizon)}_raw"


def _field(batch: Any, singular: str, plural: str | None = None) -> Any:
    if hasattr(batch, singular):
        return getattr(batch, singular)
    if plural is not None and hasattr(batch, plural):
        return getattr(batch, plural)
    raise ValueError(f"transition batch is missing {singular}")


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n, d)")
    return array


def _vector(value: Any, name: str, *, dtype: Any = float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
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


def _extract_batch(batch: Any, horizon: int) -> _BatchData:
    observation = _matrix(_field(batch, "observation", "observations"), "observation")
    action = _matrix(_field(batch, "action", "actions"), "action")
    reward = _vector(_field(batch, "reward", "rewards"), "reward")
    next_observation = _matrix(
        _field(batch, "next_observation", "next_observations"), "next_observation"
    )
    terminated = _vector(_field(batch, "terminated"), "terminated", dtype=bool)
    truncated = _vector(_field(batch, "truncated"), "truncated", dtype=bool)
    dataset_cut = _vector(_field(batch, "dataset_cut"), "dataset_cut", dtype=bool)
    native_timestep = _vector(
        _field(batch, "native_timestep"), "native_timestep", dtype=np.int64
    )
    size = len(observation)
    arrays = (
        action,
        reward,
        next_observation,
        terminated,
        truncated,
        dataset_cut,
        native_timestep,
    )
    if any(len(array) != size for array in arrays):
        raise ValueError("transition fields have inconsistent row counts")
    if next_observation.shape != observation.shape:
        raise ValueError("next_observation must match observation shape")
    if size == 0:
        raise ValueError("transition batch is empty")
    provenance = str(getattr(batch, "timestep_provenance", ""))
    if provenance not in {"episode_offsets", "native_indices"}:
        raise ValueError(
            "native_timestep provenance must be episode_offsets or native_indices; "
            "compressed/subsample ordinals and unknown provenance fail closed"
        )
    if np.any(native_timestep < 0):
        raise ValueError("native_timestep must be non-negative")
    finite = (
        np.isfinite(observation).all(axis=1)
        & np.isfinite(action).all(axis=1)
        & np.isfinite(reward)
        & np.isfinite(next_observation).all(axis=1)
    )
    if np.any(native_timestep >= int(horizon)):
        raise ValueError("native_timestep lies outside the configured finite horizon")
    reasons = np.asarray(
        getattr(batch, "truncation_reason", np.full(size, "none")), dtype=str
    ).reshape(-1)
    if len(reasons) != size:
        raise ValueError("truncation_reason must match transition rows")
    # Dataset cuts are artificial membership boundaries, not terminals.  Their
    # observed (s,a,r,s') rows remain valid for a transition model.  Native
    # terminations and environment truncations stop model rollouts; time-limit
    # truncations are handled solely by the configured finite horizon.
    environment_terminal = terminated | (reasons == "environment")
    valid = finite
    if not np.any(valid):
        raise ValueError("no valid native finite-horizon transitions remain")
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
        provenance,
    )


def _mix64(value: int) -> int:
    """SplitMix64 finalizer, expressed with Python ints to avoid overflow warnings."""

    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _keys(value: Any, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint64)
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)
    if len(array) != rows:
        raise ValueError(f"{name} must provide exactly one uint64 key per row")
    return array


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
    """Conditional RFF energy with NCE fitting and Langevin generation."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        ridge: float,
        energy_features: int,
        negatives: int,
        contrastive_steps: int,
        learning_rate: float,
        langevin_steps: int,
        langevin_step_size: float,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.ridge = float(ridge)
        self.energy_features = int(energy_features)
        self.negatives = int(negatives)
        self.contrastive_steps = int(contrastive_steps)
        self.learning_rate = float(learning_rate)
        self.langevin_steps = int(langevin_steps)
        self.langevin_step_size = float(langevin_step_size)
        if min(self.energy_features, self.negatives, self.contrastive_steps, self.langevin_steps) <= 0:
            raise ValueError("energy feature, negative, optimization, and Langevin counts must be positive")
        self.x_scaler: _Standardizer | None = None
        self.y_scaler: _Standardizer | None = None
        self.center: _RandomFeatureRidge | None = None
        self.residual_scale: np.ndarray | None = None
        self.rff_weight: np.ndarray | None = None
        self.rff_bias: np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

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

    @staticmethod
    def _nce_loss(energy: np.ndarray) -> tuple[float, np.ndarray]:
        logits = -energy
        maximum = np.max(logits, axis=1, keepdims=True)
        exponent = np.exp(logits - maximum)
        probabilities = exponent / np.sum(exponent, axis=1, keepdims=True)
        loss = float(
            np.mean(energy[:, 0] + maximum[:, 0] + np.log(np.sum(exponent, axis=1)))
        )
        return loss, probabilities

    def fit(self, inputs: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> None:
        self.x_scaler = _Standardizer.fit(inputs)
        self.y_scaler = _Standardizer.fit(targets)
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
        first_moment = np.zeros_like(self.theta)
        second_moment = np.zeros_like(self.theta)
        for step in range(1, self.contrastive_steps + 1):
            candidates = np.empty((len(y), self.negatives + 1, y.shape[1]), dtype=float)
            candidates[:, 0] = y
            for negative in range(self.negatives):
                permutation = rng.permutation(len(y))
                candidates[:, negative + 1] = y[permutation]
                # Prevent fixed points from becoming mislabeled negatives.
                same = permutation == np.arange(len(y))
                if np.any(same):
                    candidates[same, negative + 1] += rng.normal(
                        scale=0.5, size=(int(np.sum(same)), y.shape[1])
                    )
            energy, phi = self._energy_candidates(x, candidates)
            loss, probability = self._nce_loss(energy)
            if first_loss is None:
                first_loss = loss
            # d(E_pos + logsumexp(-E))/d theta
            gradient = np.mean(
                phi[:, 0] - np.sum(probability[:, :, None] * phi, axis=1), axis=0
            )
            gradient += self.ridge * self.theta
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            self.theta -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            last_loss = loss
        # Evaluate a fresh, deterministic-size contrastive panel after the update.
        candidates = np.empty((len(y), self.negatives + 1, y.shape[1]), dtype=float)
        candidates[:, 0] = y
        for negative in range(self.negatives):
            candidates[:, negative + 1] = y[rng.permutation(len(y))]
        energy, _ = self._energy_candidates(x, candidates)
        final_loss, _ = self._nce_loss(energy)
        energy_gap = float(np.mean(energy[:, 1:]) - np.mean(energy[:, 0]))
        self.diagnostics = {
            "conditional_energy": "quadratic_base_plus_learned_joint_RFF_energy",
            "contrastive_objective": "InfoNCE_with_replay_negatives",
            "contrastive_steps": self.contrastive_steps,
            "contrastive_loss_initial": float(first_loss),
            "contrastive_loss_last_training_batch": float(last_loss),
            "contrastive_loss_final_panel": final_loss,
            "positive_negative_energy_gap": energy_gap,
            "negative_candidates": self.negatives,
            "energy_features": self.energy_features,
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


def _actor_id(actor: Any) -> str:
    return str(getattr(actor, "policy_id", getattr(actor, "candidate_id", "unknown")))


def _actor_semantics(actor: Any) -> str:
    semantics = getattr(actor, "semantics", getattr(actor, "policy_semantics", "unspecified"))
    return str(getattr(semantics, "value", semantics))


def _sample_actor(
    actor: Any,
    observations: np.ndarray,
    native_timestep: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    if hasattr(actor, "sample_actions"):
        from .core import candidate_actions

        actions = candidate_actions(
            actor,
            observations,
            native_timestep,
            keys=np.asarray(keys, dtype=np.uint64),
            require_deterministic=False,
        )
    elif hasattr(actor, "actions"):
        actions = actor.actions(
            _actor_id(actor),
            observations,
            action_keys=np.asarray(keys, dtype=np.uint64),
            require_deterministic=False,
        )
    else:
        raise TypeError("candidate actor must implement sample_actions(..., keys=...)")
    actions = _matrix(actions, "actor actions")
    if len(actions) != len(observations) or not np.isfinite(actions).all():
        raise ValueError("actor returned invalid actions")
    return actions


def _make_value_estimate(**fields: Any) -> Any:
    # Import lazily so this module stays importable while the tiny companion
    # repository is assembled in parallel.
    from .core import ValueEstimate

    return ValueEstimate(**fields)


class _BaseMBOPE:
    method_id = "BASE_MBOPE"
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
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if horizon <= 0 or rollouts_per_initial <= 0:
            raise ValueError("horizon and rollouts_per_initial must be positive")
        self.gamma = float(gamma)
        self.horizon = int(horizon)
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

    def _build_model(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    def fit(self, batch: Any, candidate: Any, *, fit_keys: Any) -> "_BaseMBOPE":
        started = perf_counter()
        data = _extract_batch(batch, self.horizon)
        from .core import policy_id, policy_semantics

        policy_id(candidate)
        policy_semantics(candidate)
        fit_key_array = np.asarray(fit_keys, dtype=np.uint64).reshape(-1)
        if len(fit_key_array) == 0:
            raise ValueError("fit_keys must contain at least one explicit uint64 key")
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
        self._model = self._build_model(rng)
        self._model.fit(inputs, targets, rng)
        native_environment_termination = bool(np.any(data.environment_terminal[valid]))
        if self.learn_termination:
            self._termination = _TerminationHead(self.ridge)
            self._termination.fit(inputs, data.environment_terminal[valid], rng)
        elif self.termination_function is None and native_environment_termination:
            from .core import DataValidationError, EstimateStatus

            raise DataValidationError(
                EstimateStatus.AMBIGUOUS_TERMINATION.value,
                "native environment termination requires a shared known functional or "
                "a separately identified learn_termination=True variant",
            )
        self._actor = candidate
        self._observation_dim = data.observation.shape[1]
        self._action_dim = data.action.shape[1]
        self._behavior_action_mean = np.mean(data.action[valid], axis=0)
        self._behavior_action_scale = np.std(data.action[valid], axis=0)
        self._behavior_action_scale[self._behavior_action_scale < 1e-6] = 1.0
        self._fit_rows = int(np.sum(valid))
        self._fit_key_digest = _key_digest(fit_key_array)
        self._timestep_provenance = data.timestep_provenance
        self._fit_seconds = perf_counter() - started
        self._fit_diagnostics = {
            **self._model.diagnostics,
            "method_identity": dict(self.identity),
            "native_timestep_used": True,
            "timestep_provenance": data.timestep_provenance,
            "dataset_cut_rows_treated_as_nonterminal": int(np.sum(data.dataset_cut & valid)),
            "environment_terminal_rows": int(np.sum(data.environment_terminal & valid)),
            "truncation_rows_not_relabeled_terminal": int(np.sum(data.truncated & valid)),
            "termination_contract": self._termination_mode,
            "learned_termination_method_suffix": self.learn_termination,
            "fit_rows": self._fit_rows,
            "actor_queries_during_fit": 0,
        }
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
        initial = _matrix(initial_observations, "initial_observations")
        if initial.shape[1] != self._observation_dim or not np.isfinite(initial).all():
            raise ValueError("initial observation ABI differs from fitted data")
        root_keys = _keys(keys, len(initial), "keys")
        initial_time = np.asarray(initial_timestep, dtype=np.int64)
        if initial_time.ndim == 0:
            initial_time = np.full(len(initial), int(initial_time), dtype=np.int64)
        else:
            initial_time = initial_time.reshape(-1)
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
            "actor_semantics": _actor_semantics(actor),
            "rollout_action_z_mean": action_z_sum / max(action_z_count, 1),
            "rollout_action_z_max": action_z_max,
            "support_rows": action_z_count,
        }
        provenance = {
            "candidate_id": _actor_id(actor),
            "implementation": "policy_learnware_ope.mbope",
            "fit_key_digest": self._fit_key_digest,
            "native_timestep_provenance": self._timestep_provenance,
            "value_convention": _value_convention(self.gamma, self.horizon),
            "production_method_convention": VALUE_CONVENTION,
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

    method_id = DOPE_STYLE_MB_FF_ID
    identity = {
        "family": "feed_forward_model_based_ope",
        "project_defined": True,
        "implementation_scope": "EXECUTABLE_SYNTHETIC_REFERENCE",
        "full_2x256_SiLU_diagonal_NLL_port": False,
        "dope_is_benchmark_inspiration_not_algorithm": True,
        "distinct_from": [AR_MBOPE_ID, ETM_MBOPE_ID],
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
        self.member_count = int(ensemble_members)
        self.hidden_dim = int(hidden_dim)

    def _build_model(self, rng: np.random.Generator) -> _FeedForwardDeltaModel:
        del rng
        return _FeedForwardDeltaModel(
            members=self.member_count, hidden_dim=self.hidden_dim, ridge=self.ridge
        )


class ARMBOPEEstimator(_BaseMBOPE):
    """Independent B06-inspired autoregressive MB-OPE project adaptation."""

    method_id = AR_MBOPE_ID
    identity = {
        "family": "fixed_order_autoregressive_model_based_ope",
        "project_adaptation": True,
        "implementation_scope": "EXECUTABLE_SYNTHETIC_REFERENCE",
        "official_B06_architecture_parity": False,
        "training": "teacher_forcing",
        "generation": "sequential_per_dimension",
        "distinct_from": [DOPE_STYLE_MB_FF_ID, ETM_MBOPE_ID],
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
        self.member_count = int(ensemble_members)
        self.hidden_dim = int(hidden_dim)

    def _build_model(self, rng: np.random.Generator) -> _AutoregressiveDeltaModel:
        del rng
        return _AutoregressiveDeltaModel(
            members=self.member_count, hidden_dim=self.hidden_dim, ridge=self.ridge
        )


class ETMMBOPEEstimator(_BaseMBOPE):
    """Contrastive conditional-energy transition model with Langevin rollout."""

    method_id = ETM_MBOPE_ID
    identity = {
        "family": "contrastive_energy_transition_model_ope",
        "project_adaptation": True,
        "implementation_scope": "EXECUTABLE_SYNTHETIC_REFERENCE",
        "official_ETM_architecture_parity": False,
        "fit": "InfoNCE",
        "generation": "Langevin_energy_sampling",
        "distinct_from": [DOPE_STYLE_MB_FF_ID, AR_MBOPE_ID],
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
        learning_rate: float = 0.02,
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
        self.hidden_dim = int(hidden_dim)
        self.energy_features = int(energy_features)
        self.negatives = int(negatives)
        self.contrastive_steps = int(contrastive_steps)
        self.learning_rate = float(learning_rate)
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
            learning_rate=self.learning_rate,
            langevin_steps=self.langevin_steps,
            langevin_step_size=self.langevin_step_size,
        )


# Exact method-ID aliases are convenient for a deliberately tiny CLI/registry.
DOPE_STYLE_MB_FF_G099_H1000 = DOPEStyleMBFFEstimator
AR_MBOPE_G099_H1000 = ARMBOPEEstimator
ETM_MBOPE_G099_H1000 = ETMMBOPEEstimator

MODEL_BASED_ESTIMATORS = {
    DOPE_STYLE_MB_FF_ID: DOPEStyleMBFFEstimator,
    AR_MBOPE_ID: ARMBOPEEstimator,
    ETM_MBOPE_ID: ETMMBOPEEstimator,
}


def make_model_based_estimator(method_id: str, **kwargs: Any) -> _BaseMBOPE:
    try:
        estimator = MODEL_BASED_ESTIMATORS[method_id]
    except KeyError as exc:
        raise ValueError(f"unknown model-based method: {method_id}") from exc
    return estimator(**kwargs)


__all__ = [
    "AR_MBOPE_G099_H1000",
    "AR_MBOPE_ID",
    "ARMBOPEEstimator",
    "DOPE_STYLE_MB_FF_G099_H1000",
    "DOPE_STYLE_MB_FF_ID",
    "DOPEStyleMBFFEstimator",
    "ETM_MBOPE_G099_H1000",
    "ETM_MBOPE_ID",
    "ETMMBOPEEstimator",
    "MODEL_BASED_ESTIMATORS",
    "make_model_based_estimator",
]
