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
        == "PROJECT_ETM_PROTOCOL_ADAPTATION"
    )
    assert (
        energy["contrastive_objective"]
        == "InfoNCE_with_conditional_Langevin_negatives"
    )
    assert energy["training_negative_sampler"] == "conditional_batched_Langevin"
    assert energy["shuffle_or_replay_negatives_used"] is False
    assert energy["gradient_penalty_vjp"] == "exact_analytic_into_trainable_RFF_theta"
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
    with pytest.raises(ValueError, match="training_noise_scale"):
        ETMMBOPEEstimator(training_noise_scale=-0.1)
    with pytest.raises(ValueError, match="step_size_final"):
        ETMMBOPEEstimator(
            training_step_size_initial=0.01,
            training_step_size_final=0.02,
        )

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
    assert first.value == pytest.approx(second.value, rel=1e-9, abs=1e-10)
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


def _small_etm(**overrides: object) -> ETMMBOPEEstimator:
    config: dict[str, object] = {
        "horizon": 5,
        "rollouts_per_initial": 3,
        "hidden_dim": 12,
        "energy_features": 24,
        "negatives": 2,
        "contrastive_steps": 8,
        "epochs": 4,
        "batch_size": 25,
        "learning_rate": 0.008,
        "temperature": 1.0,
        "training_langevin_steps": 4,
        "training_step_size_initial": 0.08,
        "training_step_size_final": 0.01,
        "training_noise_scale": 0.25,
        "training_gradient_clip": 5.0,
        "training_drift_clip": 0.3,
        "training_sample_clip": 1.1,
        "gradient_penalty_margin": 0.0,
        "gradient_penalty_weight": 0.2,
        "langevin_steps": 4,
        "langevin_step_size": 0.02,
    }
    config.update(overrides)
    return ETMMBOPEEstimator(**config)


def test_etm_training_negatives_are_conditional_langevin_and_seed_reproducible() -> None:
    batch = _linear_batch(episodes=20, seed=19)
    first = _small_etm().fit(
        batch, LinearActor(), fit_keys=np.asarray([123, 456], dtype=np.uint64)
    )
    second = _small_etm().fit(
        batch, LinearActor(), fit_keys=np.asarray([123, 456], dtype=np.uint64)
    )
    first_diagnostics = first._fit_diagnostics
    second_diagnostics = second._fit_diagnostics

    assert first._model is not None and second._model is not None
    np.testing.assert_allclose(
        first._model.theta, second._model.theta, rtol=1e-9, atol=1e-10
    )
    assert first_diagnostics["optimizer_update_count"] == 8
    assert first_diagnostics["training_langevin_steps"] == 4
    assert first_diagnostics["batch_size"] == 25
    assert first_diagnostics["configured_epochs"] == 4
    assert first_diagnostics["epochs_entered"] == 2
    assert first_diagnostics["langevin_negative_chain_count"] > 0
    assert first_diagnostics["langevin_negative_step_count"] > 0
    assert first_diagnostics["training_chain_mean_displacement"] > 0.0
    assert first_diagnostics["final_panel_chain_mean_displacement"] > 0.0
    training_energy_scale = max(
        1.0,
        abs(first_diagnostics["training_chain_start_energy_mean"]),
        abs(first_diagnostics["training_chain_end_energy_mean"]),
    )
    panel_energy_scale = max(
        1.0,
        abs(first_diagnostics["final_panel_chain_start_energy"]),
        abs(first_diagnostics["final_panel_chain_end_energy"]),
    )
    assert (
        first_diagnostics["training_chain_mean_absolute_energy_change"]
        > np.finfo(np.float64).eps * training_energy_scale
    )
    assert (
        first_diagnostics["final_panel_chain_mean_absolute_energy_change"]
        > np.finfo(np.float64).eps * panel_energy_scale
    )
    config = first_diagnostics["actual_training_config"]
    assert config["target_normalization"] == "zero_mean_max_abs_box"
    assert config["optimizer"] == "Adam"
    for key in (
        "negative_sampling_seconds",
        "gradient_penalty_seconds",
        "analytic_backward_seconds",
        "energy_fit_seconds",
    ):
        assert np.isfinite(first_diagnostics[key])
        assert first_diagnostics[key] >= 0.0

    model = first._model
    assert model is not None and model.x_scaler is not None and model.y_scaler is not None
    raw_targets = np.concatenate(
        [
            batch.next_observation - batch.observation,
            batch.reward[:, None],
        ],
        axis=1,
    )
    box_targets = model.y_scaler.transform(raw_targets)
    np.testing.assert_allclose(
        model.y_scaler.mean,
        0.0,
        rtol=0.0,
        atol=np.finfo(model.y_scaler.mean.dtype).eps,
    )
    assert np.max(np.abs(box_targets)) <= 1.0 + 1e-12
    raw_inputs = np.concatenate(
        [
            batch.observation,
            batch.action,
            batch.native_timestep[:, None] / 5.0,
        ],
        axis=1,
    )
    normalized = model.x_scaler.transform(raw_inputs)
    left, left_trace = model._langevin_negatives(
        normalized[[0, 1]], np.random.default_rng(991)
    )
    right, right_trace = model._langevin_negatives(
        normalized[[-2, -1]], np.random.default_rng(991)
    )
    # Equal seeds give the same uniform starts and noise.  Different conditions
    # still produce different chain endpoints, proving conditional generation.
    assert left_trace["start_sample_mean"] == pytest.approx(
        right_trace["start_sample_mean"], rel=1e-9, abs=1e-10
    )
    assert left_trace["start_sample_std"] == pytest.approx(
        right_trace["start_sample_std"], rel=1e-9, abs=1e-10
    )
    assert not np.allclose(left, right)
    assert left_trace["mean_displacement"] > 0.0

    initial = np.asarray([[-0.4], [0.6]])
    rollout_keys = np.asarray([71, 72], dtype=np.uint64)
    first_value = first.estimate(initial, keys=rollout_keys).value
    second_value = second.estimate(initial, keys=rollout_keys).value
    assert np.isfinite(first_value)
    assert first_value == pytest.approx(second_value, rel=1e-9, abs=1e-10)


def test_etm_non_finite_training_fails_before_publishing_model() -> None:
    batch = _linear_batch(episodes=4, seed=29)
    estimator = _small_etm(
        contrastive_steps=1,
        epochs=1,
        batch_size=20,
    ).fit(batch, LinearActor(), fit_keys=np.asarray([81], dtype=np.uint64))
    estimator.temperature = 1e-320
    with pytest.raises(FloatingPointError, match="non-finite"):
        estimator.fit(batch, LinearActor(), fit_keys=np.asarray([82], dtype=np.uint64))
    with pytest.raises(RuntimeError, match="fit must be called"):
        estimator.estimate(np.asarray([[0.0]]), keys=np.asarray([3], dtype=np.uint64))

    overflowing_adam = _small_etm(
        contrastive_steps=1,
        epochs=1,
        batch_size=20,
        gradient_penalty_weight=1e200,
    )
    with pytest.raises(FloatingPointError, match="Adam state"):
        overflowing_adam.fit(
            batch, LinearActor(), fit_keys=np.asarray([83], dtype=np.uint64)
        )
    with pytest.raises(RuntimeError, match="fit must be called"):
        overflowing_adam.estimate(
            np.asarray([[0.0]]), keys=np.asarray([4], dtype=np.uint64)
        )


def test_etm_gradient_penalty_vjp_matches_finite_difference_and_changes_update() -> None:
    batch = _linear_batch(episodes=20, seed=23)
    keys = np.asarray([771], dtype=np.uint64)
    without_penalty = _small_etm(gradient_penalty_weight=0.0).fit(
        batch, LinearActor(), fit_keys=keys
    )
    with_penalty = _small_etm(gradient_penalty_weight=0.2).fit(
        batch, LinearActor(), fit_keys=keys
    )
    with_diagnostics = with_penalty._fit_diagnostics
    assert with_diagnostics["gradient_penalty_last"] > 0.0
    assert 0.0 < with_diagnostics["gradient_penalty_active_rate"] <= 1.0
    assert without_penalty._model is not None and with_penalty._model is not None
    assert not np.allclose(
        without_penalty._model.theta,
        with_penalty._model.theta,
        rtol=1e-8,
        atol=1e-10,
    )

    model = with_penalty._model
    assert model is not None and model.x_scaler is not None and model.y_scaler is not None
    assert model.theta is not None
    raw_inputs = np.concatenate(
        [
            batch.observation[:3],
            batch.action[:3],
            batch.native_timestep[:3, None] / 5.0,
        ],
        axis=1,
    )
    x = model.x_scaler.transform(raw_inputs)
    positive = np.concatenate(
        [
            batch.next_observation[:3] - batch.observation[:3],
            batch.reward[:3, None],
        ],
        axis=1,
    )
    positive = model.y_scaler.transform(positive)
    generated, _ = model._langevin_negatives(x, np.random.default_rng(4567))
    candidates = np.concatenate([positive[:, None, :], generated], axis=1)
    _, analytic_gradient, _ = model._gradient_penalty(x, candidates)

    baseline = model.theta.copy()
    epsilon = 1e-6
    for index in range(3):
        model.theta = baseline.copy()
        model.theta[index] += epsilon
        plus, _, _ = model._gradient_penalty(x, candidates)
        model.theta = baseline.copy()
        model.theta[index] -= epsilon
        minus, _, _ = model._gradient_penalty(x, candidates)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        assert analytic_gradient[index] == pytest.approx(
            finite_difference, rel=2e-5, abs=2e-6
        )
    model.theta = baseline
