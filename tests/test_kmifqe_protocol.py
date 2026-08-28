from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_ope.core import EstimateStatus, PolicySemantics, TransitionBatch
from policy_learnware_ope.fqe import FiniteHorizonKMIFQE
from policy_learnware_ope.kmifqe import B20KMIFQETrainer, seed_from_keys


class CurvedActor:
    policy_id = "curved-target"
    semantics = PolicySemantics.DETERMINISTIC

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        assert np.asarray(keys).dtype == np.uint64
        time = np.asarray(native_timestep, dtype=float)
        return np.column_stack(
            [
                0.18 * np.tanh(observations[:, 0]) + 0.02 * time,
                -0.14 * np.sin(observations[:, 1]) - 0.01 * time,
            ]
        )


class ExactUnitGaussian:
    density_id = "two-dimensional-unit-gaussian-exact"
    exact = True

    def log_prob(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        native_timestep: np.ndarray,
    ) -> np.ndarray:
        del observations, native_timestep
        return -0.5 * np.sum(np.square(actions), axis=1) - np.log(2.0 * np.pi)


def _curved_batch(*, episodes: int = 48, horizon: int = 4) -> TransitionBatch:
    rng = np.random.default_rng(20240828)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    next_actions: list[np.ndarray] = []
    times: list[int] = []
    episode_ids: list[int] = []
    for episode in range(episodes):
        state = rng.uniform(-1.2, 1.2, size=2)
        logged_actions = rng.normal(scale=0.55, size=(horizon + 1, 2))
        for time in range(horizon):
            action = logged_actions[time]
            next_state = np.asarray(
                [
                    0.62 * state[0] + 0.24 * action[0] - 0.08 * action[1],
                    0.55 * state[1] + 0.18 * action[1] + 0.07 * state[0] * action[0],
                ]
            )
            reward = (
                0.7
                + 0.12 * state[0]
                - 0.08 * state[1]
                + (0.30 + 0.16 * np.tanh(state[0])) * action[0] ** 2
                - (0.22 + 0.12 * np.cos(state[1])) * action[1] ** 2
                + 0.15 * state[0] * action[0] * action[1]
            )
            observations.append(state.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            next_observations.append(next_state.copy())
            next_actions.append(logged_actions[time + 1].copy())
            times.append(time)
            episode_ids.append(episode)
            state = next_state
    rows = episodes * horizon
    return TransitionBatch(
        observation=np.asarray(observations),
        action=np.asarray(actions),
        reward=np.asarray(rewards),
        next_observation=np.asarray(next_observations),
        terminated=np.zeros(rows, dtype=bool),
        truncated=np.zeros(rows, dtype=bool),
        dataset_cut=np.zeros(rows, dtype=bool),
        native_timestep=np.asarray(times, dtype=np.int64),
        episode_id=np.asarray(episode_ids, dtype=np.int64),
        episode_offsets=np.arange(0, rows + 1, horizon, dtype=np.int64),
        timestep_provenance="episode_offsets",
        next_behavior_action=np.asarray(next_actions),
        source_digest="curved-B20-mechanism-fixture-v1",
    )


def _fit(batch: TransitionBatch) -> FiniteHorizonKMIFQE:
    return FiniteHorizonKMIFQE(
        gamma=0.99,
        horizon=4,
        ridge=2e-5,
        max_iterations=200,
        tolerance=3e-3,
        critic_features=28,
        min_ess_fraction=0.005,
        target_update_interval=1,
        critic_step_size=0.1,
    ).fit(
        batch,
        CurvedActor(),
        behavior_density=ExactUnitGaussian(),
        fit_keys=np.arange(len(batch), dtype=np.uint64) + 71,
    )


def test_b20_mechanisms_are_executed_and_seeded_reproducibly() -> None:
    batch = _curved_batch()
    first = _fit(batch).estimate(
        batch.observation[::37], keys=np.arange(len(batch.observation[::37]), dtype=np.uint64)
    )
    second = _fit(batch).estimate(
        batch.observation[::37], keys=np.arange(len(batch.observation[::37]), dtype=np.uint64)
    )

    assert first.status is EstimateStatus.PASS
    assert np.isfinite(first.value)
    assert first.value == pytest.approx(second.value, rel=1e-9, abs=1e-10)
    assert first.method_id == "FH_KMIFQE_G099_H4"
    assert first.provenance["scientific_role"] == "B20_PROTOCOL_ADAPTATION"
    assert first.provenance["official_parity"] is False
    assert first.provenance["paper_benchmark_parity"] is False

    diagnostics = first.diagnostics
    assert diagnostics["bandwidth_source"] == "B20_EQ11_TD_MSE_BIAS_VARIANCE_ESTIMATOR"
    assert diagnostics["bandwidth_update_count"] >= 2
    assert np.ptp(diagnostics["bandwidth_history"]) > 1e-6
    assert diagnostics["alternating_update_count"] == diagnostics["iterations"]
    assert diagnostics["bandwidth_update_count"] == diagnostics["iterations"]
    assert diagnostics["hessian_update_count"] == diagnostics["iterations"]
    assert diagnostics["target_update_count"] >= 2
    assert diagnostics["metric_state_variation"] > 1e-8
    assert diagnostics["metric_off_diagonal_frobenius_mean"] > 1e-8
    assert diagnostics["metric_determinant_max_error"] < 1e-8
    assert np.isfinite(diagnostics["hessian_abs_condition_p95"])
    assert diagnostics["replacement_resampling"] is True
    assert diagnostics["replacement_duplicate_count"] > 0
    assert diagnostics["unique_resampled_fraction"] < 1.0
    assert diagnostics["bootstrap_action_source"] == "logged_adjacent_next_behavior_action"
    assert first.support["verified_adjacent_rows"] == first.support["active_rows"]
    assert diagnostics["td_loss"] >= 0.0
    assert diagnostics["resample_index_sha256"] == second.diagnostics[
        "resample_index_sha256"
    ]
    np.testing.assert_allclose(
        diagnostics["bandwidth_history"],
        second.diagnostics["bandwidth_history"],
        rtol=1e-9,
        atol=1e-10,
    )
    assert diagnostics["target_critic_parameter_l2"] == pytest.approx(
        second.diagnostics["target_critic_parameter_l2"], rel=1e-9, abs=1e-10
    )
    assert diagnostics["prediction_delta_history"][-1] <= diagnostics["final_q_tolerance"]
    assert diagnostics["target_lag_history"][-1] <= diagnostics["final_q_tolerance"]
    assert diagnostics["probability_l1_delta_history"][-1] <= diagnostics[
        "probability_l1_tolerance"
    ]
    assert np.isfinite(diagnostics["bandwidth_relative_delta_history"][-1])
    assert np.isfinite(diagnostics["metric_relative_delta_history"][-1])

    action_dimension = batch.action.shape[1]
    active_count = first.support["active_rows"]
    expected_bandwidth = []
    for bias_squared, variance in zip(
        diagnostics["bias_constant_squared_history"],
        diagnostics["variance_constant_history"],
        strict=True,
    ):
        denominator = max(bias_squared, np.finfo(np.float64).eps)
        raw = (
            action_dimension * variance / (4.0 * active_count * denominator)
        ) ** (1.0 / (action_dimension + 4.0))
        expected_bandwidth.append(np.clip(raw, 1e-3, 10.0))
    np.testing.assert_allclose(
        diagnostics["bandwidth_history"], expected_bandwidth, rtol=1e-10, atol=1e-12
    )


def test_logged_adjacent_action_changes_real_in_sample_bootstrap() -> None:
    batch = _curved_batch()
    assert batch.next_behavior_action is not None
    mask = batch.bootstrap_mask(4)
    target_next_action = np.zeros_like(batch.action)
    keys = np.arange(len(batch), dtype=np.uint64) + 900
    seed, digest = seed_from_keys(keys)

    def train(logged_next_action: np.ndarray) -> B20KMIFQETrainer:
        log_behavior = -0.5 * np.sum(np.square(logged_next_action), axis=1)
        return B20KMIFQETrainer(
            gamma=0.99,
            horizon=4,
            ridge=2e-5,
            max_iterations=2,
            tolerance=1e-3,
            hidden_features=28,
            target_update_interval=1,
            critic_step_size=0.1,
        ).fit(
            observation=batch.observation,
            action=batch.action,
            reward=batch.reward,
            next_observation=batch.next_observation,
            native_timestep=batch.native_timestep,
            mask=mask,
            target_next_action=target_next_action,
            logged_next_action=logged_next_action,
            log_behavior_density=log_behavior,
            log_target_density=np.zeros(len(batch)),
            seed=seed,
            seed_digest=digest,
            min_ess_fraction=0.005,
        )

    baseline = train(batch.next_behavior_action)
    perturbed = train(-batch.next_behavior_action)

    # With pi(s')=0 and a symmetric density, a' and -a' have identical
    # kernel weights through iteration two.  The discrete sampled rows are
    # therefore identical, isolating the logged a' contribution to TD.
    assert baseline.diagnostics["resample_index_sha256_first_two"] == (
        perturbed.diagnostics["resample_index_sha256_first_two"]
    )
    np.testing.assert_allclose(
        baseline.diagnostics["bandwidth_history"],
        perturbed.diagnostics["bandwidth_history"],
        rtol=1e-9,
        atol=1e-10,
    )
    assert not np.isclose(
        baseline.diagnostics["td_target_mean_first_two"][1],
        perturbed.diagnostics["td_target_mean_first_two"][1],
        rtol=1e-7,
        atol=1e-9,
    )


def test_mean_weight_bias_correction_changes_regularized_critic_not_sampling() -> None:
    batch = _curved_batch(episodes=8)
    mask = batch.bootstrap_mask(4)
    target_next_action = np.zeros_like(batch.action)
    logged_next_action = np.zeros_like(batch.action)
    keys = np.arange(len(batch), dtype=np.uint64) + 1200
    seed, digest = seed_from_keys(keys)

    def train(log_behavior_density: np.ndarray) -> B20KMIFQETrainer:
        return B20KMIFQETrainer(
            gamma=0.99,
            horizon=4,
            ridge=2e-5,
            max_iterations=1,
            tolerance=1e-3,
            hidden_features=20,
            critic_step_size=0.5,
        ).fit(
            observation=batch.observation,
            action=batch.action,
            reward=batch.reward,
            next_observation=batch.next_observation,
            native_timestep=batch.native_timestep,
            mask=mask,
            target_next_action=target_next_action,
            logged_next_action=logged_next_action,
            log_behavior_density=log_behavior_density,
            log_target_density=np.zeros(len(batch)),
            seed=seed,
            seed_digest=digest,
            min_ess_fraction=0.005,
        )

    baseline = train(np.zeros(len(batch)))
    doubled = train(np.full(len(batch), -np.log(2.0)))
    assert baseline.diagnostics["resample_index_sha256"] == doubled.diagnostics[
        "resample_index_sha256"
    ]
    assert doubled.diagnostics["mean_weight_history"][0] == pytest.approx(
        2.0 * baseline.diagnostics["mean_weight_history"][0]
    )
    assert baseline.coefficient is not None and doubled.coefficient is not None
    assert not np.allclose(baseline.coefficient, doubled.coefficient, rtol=1e-9, atol=1e-12)
