from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_ope.core import PolicySemantics, TransitionBatch
from policy_learnware_ope.mbope import (
    DOPE_STYLE_MB_FF_ID,
    ARMBOPEEstimator,
    DOPEStyleMBFFEstimator,
    ETMMBOPEEstimator,
    make_model_based_estimator,
)


class LinearActor:
    policy_id = "toy-linear"
    semantics = PolicySemantics.DETERMINISTIC

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        checked_keys = np.asarray(keys, dtype=np.uint64)
        assert checked_keys.shape == (len(observations),)
        self.calls.append((native_timestep.copy(), checked_keys.copy()))
        return -0.25 * observations[:, :1] + 0.02 * native_timestep[:, None]


class KeyedLinearActor(LinearActor):
    policy_id = "toy-keyed-linear"
    semantics = PolicySemantics.STOCHASTIC_KEYED

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        base = super().sample_actions(observations, native_timestep, keys=keys)
        key_array = np.asarray(keys, dtype=np.uint64)
        jitter = ((key_array % 1009).astype(float) / 1008.0 - 0.5) * 0.02
        return base + jitter[:, None]


class ActionsOnlyActor:
    """An unbound provider shape that must not bypass the bound actor ABI."""

    policy_id = "toy-actions-only"
    semantics = PolicySemantics.DETERMINISTIC

    def __init__(self) -> None:
        self.native_times: list[np.ndarray] = []

    def actions(
        self,
        candidate_id: str,
        observations: np.ndarray,
        *,
        native_timestep: np.ndarray,
        action_keys: np.ndarray,
        require_deterministic: bool = False,
    ) -> np.ndarray:
        assert candidate_id == self.policy_id
        assert not require_deterministic
        assert np.asarray(action_keys).dtype.kind in "iu"
        self.native_times.append(np.asarray(native_timestep).copy())
        return -0.25 * observations[:, :1] + 0.02 * native_timestep[:, None]


def _linear_batch(*, episodes: int = 72, horizon: int = 5, seed: int = 7) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    observation: list[list[float]] = []
    action: list[list[float]] = []
    reward: list[float] = []
    next_observation: list[list[float]] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    dataset_cut: list[bool] = []
    native_timestep: list[int] = []
    episode_id: list[int] = []
    reasons: list[str] = []
    for episode in range(episodes):
        state = float(rng.uniform(-1.5, 1.5))
        for timestep in range(horizon):
            behavior_action = float(rng.uniform(-1.0, 1.0))
            next_state = 0.65 * state + 0.30 * behavior_action + 0.05 * timestep
            observed_reward = 0.8 + 0.20 * state - 0.10 * behavior_action + 0.05 * timestep
            is_dataset_cut = episode < 3 and timestep == 2
            is_horizon = timestep == horizon - 1
            observation.append([state])
            action.append([behavior_action])
            reward.append(observed_reward)
            next_observation.append([next_state])
            terminated.append(False)
            truncated.append(is_dataset_cut or is_horizon)
            dataset_cut.append(is_dataset_cut)
            native_timestep.append(timestep)
            episode_id.append(episode)
            reasons.append("dataset_cut" if is_dataset_cut else "horizon" if is_horizon else "none")
            state = next_state
    rows = episodes * horizon
    return TransitionBatch(
        observation=np.asarray(observation),
        action=np.asarray(action),
        reward=np.asarray(reward),
        next_observation=np.asarray(next_observation),
        terminated=np.asarray(terminated),
        truncated=np.asarray(truncated),
        dataset_cut=np.asarray(dataset_cut),
        native_timestep=np.asarray(native_timestep, dtype=np.int64),
        episode_id=np.asarray(episode_id, dtype=np.int64),
        episode_offsets=np.arange(0, rows + 1, horizon, dtype=np.int64),
        timestep_provenance="episode_offsets",
        truncation_reason=np.asarray(reasons),
        source_digest="toy-linear-v1",
    )


def _true_value(initial: np.ndarray, *, horizon: int, gamma: float) -> float:
    values = []
    for initial_state in initial[:, 0]:
        state = float(initial_state)
        total = 0.0
        discount = 1.0
        for timestep in range(horizon):
            action = -0.25 * state + 0.02 * timestep
            total += discount * (0.8 + 0.20 * state - 0.10 * action + 0.05 * timestep)
            state = 0.65 * state + 0.30 * action + 0.05 * timestep
            discount *= gamma
        values.append(total)
    return float(np.mean(values))


def _estimators() -> list[object]:
    common = {"horizon": 5, "rollouts_per_initial": 12, "ridge": 1e-4}
    return [
        DOPEStyleMBFFEstimator(**common, ensemble_members=3, hidden_dim=24),
        ARMBOPEEstimator(**common, ensemble_members=2, hidden_dim=18),
        ETMMBOPEEstimator(
            **common,
            hidden_dim=24,
            energy_features=48,
            negatives=3,
            contrastive_steps=45,
            learning_rate=0.01,
            langevin_steps=10,
            langevin_step_size=0.025,
        ),
    ]


@pytest.mark.parametrize("estimator", _estimators(), ids=lambda item: item.method_id)
def test_each_model_based_line_really_fits_and_estimates_known_finite_horizon(estimator: object) -> None:
    batch = _linear_batch()
    actor = LinearActor()
    estimator.fit(batch, actor, fit_keys=np.asarray([11, 29], dtype=np.uint64))
    assert actor.calls == []  # A model fit must not query the evaluation policy.

    initial = np.asarray([[-1.0], [-0.25], [0.5], [1.2]])
    estimate = estimator.estimate(
        initial,
        keys=np.asarray([101, 202, 303, 404], dtype=np.uint64),
    )
    expected = _true_value(initial, horizon=5, gamma=0.99)

    assert estimate.status.value == "PASS"
    assert np.isfinite(estimate.value)
    assert abs(estimate.value - expected) < 1.25
    assert estimate.method_id == estimator.method_id
    assert estimate.method_id.endswith("_G099_H5")
    assert "H1000" not in estimate.method_id
    assert estimate.provenance["value_convention"] == "J_gamma=0.99_H=5_raw"
    assert estimate.provenance["upstream_parity_claim"] == "NONE"
    assert estimate.provenance["physical_membership_sha256"]
    assert (
        estimate.provenance["physical_membership_sha256"]
        == estimate.diagnostics["physical_membership_sha256"]
    )
    assert estimate.diagnostics["native_timestep_used"] is True
    assert estimate.diagnostics["dataset_cut_rows_treated_as_nonterminal"] == 3
    assert estimate.diagnostics["truncation_rows_not_relabeled_terminal"] == 75
    assert estimate.diagnostics["actor_queries_during_fit"] == 0
    assert estimate.diagnostics["actor_queries_only_inside_learned_model"] is True
    assert estimate.diagnostics["action_keys_explicit_for_every_row"] is True
    assert estimate.diagnostics["max_rollout_length"] == 5
    assert estimate.support["kind"] == "learned_model_rollout_global_behavior_action_z"
    assert estimate.support["support_rows"] > 0
    assert np.isfinite(estimate.support["rollout_action_z_mean"])
    assert actor.calls and all(len(times) == len(keys) for times, keys in actor.calls)


def test_method_identity_and_faithfulness_diagnostics_are_distinct() -> None:
    batch = _linear_batch(episodes=48)
    initial = np.asarray([[-0.5], [0.75]])
    results = {}
    for position, estimator in enumerate(_estimators()):
        actor = LinearActor()
        estimator.fit(batch, actor, fit_keys=np.asarray([900 + position], dtype=np.uint64))
        results[type(estimator)] = estimator.estimate(
            initial, keys=np.asarray([77, 88], dtype=np.uint64)
        )

    feed_forward_result = results[DOPEStyleMBFFEstimator]
    feed_forward = feed_forward_result.diagnostics
    assert feed_forward_result.provenance["scientific_role"] == "PROJECT_DEFINED_REFERENCE"
    assert feed_forward["method_identity"]["dope_is_benchmark_inspiration_not_algorithm"] is True
    assert feed_forward["method_identity"]["upstream_parity_claimed"] is False
    assert feed_forward["feed_forward_head"] == "random_tanh_features_plus_ridge"

    autoregressive_result = results[ARMBOPEEstimator]
    autoregressive = autoregressive_result.diagnostics
    assert (
        autoregressive_result.provenance["scientific_role"]
        == "PROJECT_METHOD_LEVEL_ADAPTATION_PROXY"
    )
    assert autoregressive["training"] == "teacher_forcing"
    assert autoregressive["generation"] == "fixed_order_sequential"
    assert autoregressive["factorization_order"] == ["delta_state[0]", "reward"]
    assert np.isfinite(autoregressive["teacher_forced_mse"])
    assert np.isfinite(autoregressive["free_running_one_step_mse"])

    energy_result = results[ETMMBOPEEstimator]
    energy = energy_result.diagnostics
    assert (
        energy_result.provenance["scientific_role"]
        == "PROJECT_CONTRASTIVE_ENERGY_ADAPTATION_PROXY"
    )
    assert energy["contrastive_objective"] == "InfoNCE_with_replay_negatives"
    assert energy["inference_sampler"] == "Langevin"
    assert energy["contrastive_steps"] == 45
    assert energy["langevin_steps"] == 10
    assert np.isfinite(energy["positive_negative_energy_gap"])


def test_model_based_configuration_and_failed_refit_fail_closed() -> None:
    with pytest.raises(ValueError, match="ridge"):
        DOPEStyleMBFFEstimator(ridge=-1e-3)
    with pytest.raises(ValueError, match="learning_rate"):
        ETMMBOPEEstimator(learning_rate=0.0)
    with pytest.raises(ValueError, match="langevin_step_size"):
        ETMMBOPEEstimator(langevin_step_size=0.0)

    estimator = DOPEStyleMBFFEstimator(
        horizon=5,
        rollouts_per_initial=2,
        ensemble_members=2,
        hidden_dim=12,
    ).fit(
        _linear_batch(episodes=24),
        LinearActor(),
        fit_keys=np.asarray([7], dtype=np.uint64),
    )
    episodes = 24
    length = 3
    rows = episodes * length
    terminal = np.tile(np.asarray([False, False, True]), episodes)
    terminal_batch = TransitionBatch(
        observation=np.zeros((rows, 1)),
        action=np.zeros((rows, 1)),
        reward=np.ones(rows),
        next_observation=np.zeros((rows, 1)),
        terminated=terminal,
        truncated=np.zeros(rows, dtype=bool),
        dataset_cut=np.zeros(rows, dtype=bool),
        native_timestep=np.tile(np.arange(length), episodes),
        episode_id=np.repeat(np.arange(episodes), length),
        episode_offsets=np.arange(0, rows + 1, length),
        timestep_provenance="episode_offsets",
    )
    with pytest.raises(ValueError, match="termination|episode offset"):
        estimator.fit(
            terminal_batch,
            LinearActor(),
            fit_keys=np.asarray([8], dtype=np.uint64),
        )
    with pytest.raises(RuntimeError, match="fit must be called"):
        estimator.estimate(np.zeros((1, 1)), keys=np.asarray([1], dtype=np.uint64))


def test_keyed_stochastic_actor_is_reproducible_and_bad_timestep_fails_closed() -> None:
    batch = _linear_batch(episodes=40)
    actor = KeyedLinearActor()
    estimator = make_model_based_estimator(
        DOPE_STYLE_MB_FF_ID,
        horizon=5,
        rollouts_per_initial=6,
        ensemble_members=2,
        hidden_dim=16,
    )
    estimator.fit(batch, actor, fit_keys=np.asarray([1234], dtype=np.uint64))
    initial = np.asarray([[-0.2], [0.4]])
    keys = np.asarray([55, 66], dtype=np.uint64)
    first = estimator.estimate(initial, keys=keys)
    second = estimator.estimate(initial, keys=keys)
    assert estimator.method_id == "DOPE_STYLE_MB_FF_G099_H5"
    assert first.value == second.value
    assert first.support["actor_semantics"] == "stochastic_keyed"
    assert first.diagnostics["rollout_key_digest"] == second.diagnostics["rollout_key_digest"]
    assert (
        first.provenance["physical_membership_sha256"]
        == second.provenance["physical_membership_sha256"]
    )

    with pytest.raises(ValueError, match="core.TransitionBatch"):
        DOPEStyleMBFFEstimator(horizon=5).fit(
            object(), actor, fit_keys=np.asarray([1], dtype=np.uint64)
        )


def test_native_environment_termination_stops_learned_rollout() -> None:
    episodes = 64
    length = 3
    state = np.zeros((episodes * length, 1), dtype=float)
    action = np.linspace(-1.0, 1.0, episodes * length)[:, None]
    timestep = np.tile(np.arange(length), episodes)
    terminal = timestep == length - 1
    batch = TransitionBatch(
        observation=state,
        action=action,
        reward=np.ones(episodes * length),
        next_observation=state,
        terminated=terminal,
        truncated=np.zeros(episodes * length, dtype=bool),
        dataset_cut=np.zeros(episodes * length, dtype=bool),
        native_timestep=timestep,
        episode_id=np.repeat(np.arange(episodes), length),
        episode_offsets=np.arange(0, episodes * length + 1, length),
        timestep_provenance="episode_offsets",
    )
    actor = LinearActor()
    estimator = DOPEStyleMBFFEstimator(
        horizon=5,
        rollouts_per_initial=8,
        ensemble_members=2,
        hidden_dim=16,
        learn_termination=True,
    ).fit(batch, actor, fit_keys=np.asarray([919], dtype=np.uint64))
    estimate = estimator.estimate(np.zeros((3, 1)), keys=np.asarray([4, 5, 6]))
    assert estimate.method_id == "DOPE_STYLE_MB_FF_G099_H5_TLEARNED"
    assert estimate.diagnostics["termination_contract"] == "learned"
    assert estimate.diagnostics["learned_environment_termination_count"] == 24
    assert estimate.diagnostics["mean_rollout_length"] == pytest.approx(3.0)
    assert estimate.value == pytest.approx(1.0 + 0.99 + 0.99**2, abs=0.08)


def test_method_ids_keys_and_bound_actor_abi_are_strict() -> None:
    assert DOPEStyleMBFFEstimator().method_id == DOPE_STYLE_MB_FF_ID
    assert ARMBOPEEstimator(gamma=1.0, horizon=7).method_id == "AR_MBOPE_G1000_H7"
    assert ETMMBOPEEstimator(gamma=0.995, horizon=9).method_id == "ETM_MBOPE_G0995_H9"

    batch = _linear_batch(episodes=32)
    actor = LinearActor()
    estimator = DOPEStyleMBFFEstimator(
        horizon=5, rollouts_per_initial=2, ensemble_members=2, hidden_dim=12
    )
    for invalid in (np.asarray([1.5]), np.asarray([-1]), np.asarray([True])):
        with pytest.raises(ValueError, match="integers|non-negative"):
            estimator.fit(batch, actor, fit_keys=invalid)

    estimator.fit(batch, actor, fit_keys=np.asarray([7], dtype=np.uint64))
    for invalid in (np.asarray([2.5]), np.asarray([-2]), np.asarray([False])):
        with pytest.raises(ValueError, match="integers|non-negative"):
            estimator.estimate(np.asarray([[0.0]]), keys=invalid)

    estimate = estimator.estimate(np.asarray([[0.0]]), keys=np.asarray([2]))
    assert estimate.status.value == "PASS"
    assert actor.calls

    with pytest.raises(TypeError, match="sample_actions"):
        estimator.estimate(
            np.asarray([[0.0]]),
            keys=np.asarray([2]),
            candidate=ActionsOnlyActor(),
        )

    class MissingSemanticsActor:
        policy_id = "missing-semantics"

        def sample_actions(self, observations, native_timestep, *, keys):
            del native_timestep, keys
            return np.zeros((len(observations), 1))

    with pytest.raises(ValueError, match="declare deterministic"):
        estimator.estimate(
            np.asarray([[0.0]]),
            keys=np.asarray([2]),
            candidate=MissingSemanticsActor(),
        )
