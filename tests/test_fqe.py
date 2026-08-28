from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from policy_learnware_ope.core import (
    DataValidationError,
    EstimateStatus,
    PolicySemantics,
    TransitionBatch,
    finite_horizon_method_id,
    finite_horizon_value_convention,
    validate_action_keys,
)
from policy_learnware_ope.fqe import FiniteHorizonFQE, FiniteHorizonKMIFQE


@dataclass
class ConstantActor:
    value: float = 0.0
    semantics: PolicySemantics = PolicySemantics.DETERMINISTIC
    policy_id: str = "constant-policy"
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        self.calls.append((observations.copy(), native_timestep.copy(), keys.copy()))
        return np.full((len(observations), 1), self.value, dtype=np.float64)


@dataclass
class GaussianDensity:
    exact: bool = True
    density_id: str = "unit-gaussian-exact"
    support_limit: float | None = None
    calls: list[np.ndarray] = field(default_factory=list)

    def log_prob(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        native_timestep: np.ndarray,
    ) -> np.ndarray:
        del observations, native_timestep
        self.calls.append(actions.copy())
        result = -0.5 * np.sum(np.square(actions), axis=1)
        if self.support_limit is not None:
            result[np.any(np.abs(actions) > self.support_limit, axis=1)] = -np.inf
        return result


class UnknownClippedDensity(GaussianDensity):
    exact = False
    density_id = "legacy-clipped-gaussian-unknown"

    def __init__(self) -> None:
        super().__init__(exact=False, density_id=self.density_id)

    def log_prob(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("an inexact density must be rejected before log_prob")


def known_mdp_batch(
    *,
    episodes: int = 32,
    horizon: int = 3,
    include_next_behavior_action: bool = True,
) -> TransitionBatch:
    """A one-state/action MDP with reward one and exact finite value."""

    n_rows = episodes * horizon
    native_timestep = np.tile(np.arange(horizon, dtype=np.int64), episodes)
    episode_id = np.repeat(np.arange(episodes, dtype=np.int64), horizon)
    offsets = np.arange(0, n_rows + 1, horizon, dtype=np.int64)
    action = np.zeros((n_rows, 1), dtype=np.float64)
    return TransitionBatch(
        observation=np.zeros((n_rows, 1), dtype=np.float64),
        action=action,
        reward=np.ones(n_rows, dtype=np.float64),
        next_observation=np.zeros((n_rows, 1), dtype=np.float64),
        terminated=np.zeros(n_rows, dtype=bool),
        truncated=np.zeros(n_rows, dtype=bool),
        dataset_cut=np.zeros(n_rows, dtype=bool),
        native_timestep=native_timestep,
        episode_id=episode_id,
        episode_offsets=offsets,
        timestep_provenance="episode_offsets",
        next_behavior_action=action.copy() if include_next_behavior_action else None,
        source_digest="synthetic-known-mdp",
    )


def test_fqe_recovers_known_finite_horizon_value_and_uses_native_time() -> None:
    batch = known_mdp_batch()
    actor = ConstantActor()
    estimator = FiniteHorizonFQE(
        gamma=0.99,
        horizon=3,
        ridge=1e-10,
        max_iterations=20,
        tolerance=1e-11,
    ).fit(batch, actor, fit_keys=np.arange(len(batch), dtype=np.uint64))

    result = estimator.estimate(
        np.zeros((8, 1)),
        keys=np.arange(100, 108, dtype=np.uint64),
    )

    assert result.status is EstimateStatus.PASS
    assert result.method_id == "FH_FQE_G099_H3"
    assert result.value == pytest.approx(1.0 + 0.99 + 0.99**2, abs=2e-6)
    assert result.provenance["time_input"] == "native_timestep/H"
    assert result.provenance["target_time"] == "(native_timestep+1)/H"
    assert result.diagnostics["converged"] is True
    assert result.diagnostics["bootstrap_rows"] == 2 * batch.episode_count
    assert result.support["kind"] == "pointwise_logged_state_action_distance"
    assert result.support["support_rows"] == len(batch)
    assert result.to_dict()["status"] == "PASS"
    # Both Bellman and initial-state queries received explicit integer keys.
    assert len(actor.calls) == 3
    assert actor.calls[0][2].dtype == np.uint64
    assert actor.calls[-1][2].shape == (8,)


def test_bootstrap_mask_distinguishes_cut_termination_and_horizon() -> None:
    batch = TransitionBatch(
        observation=np.zeros((3, 1)),
        action=np.zeros((3, 1)),
        reward=np.zeros(3),
        next_observation=np.zeros((3, 1)),
        terminated=np.array([False, True, False]),
        truncated=np.zeros(3, dtype=bool),
        dataset_cut=np.array([True, False, False]),
        native_timestep=np.array([0, 1, 2]),
        episode_id=np.array([10, 11, 12]),
        episode_offsets=np.array([0, 1, 2, 3]),
        timestep_provenance="native_indices",
    )

    np.testing.assert_array_equal(batch.bootstrap_mask(3), [1.0, 0.0, 0.0])


def test_transition_batch_rejects_sample_ordinals_and_ambiguous_truncation() -> None:
    kwargs = dict(
        observation=np.zeros((2, 1)),
        action=np.zeros((2, 1)),
        reward=np.zeros(2),
        next_observation=np.zeros((2, 1)),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        dataset_cut=np.zeros(2, dtype=bool),
        native_timestep=np.array([0, 1]),
        episode_id=np.array([0, 0]),
        episode_offsets=np.array([0, 2]),
    )
    with pytest.raises(DataValidationError, match="sampled row ordinals are forbidden"):
        TransitionBatch(**kwargs, timestep_provenance="sample_ordinal")

    ambiguous = dict(kwargs)
    ambiguous["truncated"] = np.array([False, True])
    with pytest.raises(DataValidationError) as caught:
        TransitionBatch(**ambiguous, timestep_provenance="episode_offsets")
    assert caught.value.status == EstimateStatus.AMBIGUOUS_TERMINATION.value


def test_horizon_reason_must_match_native_horizon() -> None:
    batch = TransitionBatch(
        observation=np.zeros((1, 1)),
        action=np.zeros((1, 1)),
        reward=np.zeros(1),
        next_observation=np.zeros((1, 1)),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.ones(1, dtype=bool),
        dataset_cut=np.zeros(1, dtype=bool),
        native_timestep=np.array([0]),
        episode_id=np.array([0]),
        episode_offsets=np.array([0, 1]),
        timestep_provenance="native_indices",
        truncation_reason=np.array(["horizon"]),
    )
    with pytest.raises(DataValidationError) as caught:
        batch.bootstrap_mask(3)
    assert caught.value.status == EstimateStatus.AMBIGUOUS_TERMINATION.value


def test_cut_reason_flags_are_consistent_and_short_native_episode_is_valid() -> None:
    common = dict(
        observation=np.zeros((1, 1)),
        action=np.zeros((1, 1)),
        reward=np.zeros(1),
        next_observation=np.zeros((1, 1)),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.ones(1, dtype=bool),
        dataset_cut=np.ones(1, dtype=bool),
        native_timestep=np.array([0]),
        episode_id=np.array([0]),
        episode_offsets=np.array([0, 1]),
        timestep_provenance="native_indices",
    )
    with pytest.raises(DataValidationError) as caught:
        TransitionBatch(**common, truncation_reason=np.array(["horizon"]))
    assert caught.value.status == EstimateStatus.AMBIGUOUS_TERMINATION.value
    cut = TransitionBatch(**common, truncation_reason=np.array(["dataset_cut"]))
    np.testing.assert_array_equal(cut.bootstrap_mask(3), [1.0])

    # An episode that genuinely terminates before the configured maximum H is
    # still a complete native episode: its local indices remain 0..length-1.
    early = TransitionBatch(
        observation=np.zeros((5, 1)),
        action=np.zeros((5, 1)),
        reward=np.zeros(5),
        next_observation=np.zeros((5, 1)),
        terminated=np.array([False, True, False, False, True]),
        truncated=np.zeros(5, dtype=bool),
        dataset_cut=np.zeros(5, dtype=bool),
        native_timestep=np.array([0, 1, 0, 1, 2]),
        episode_id=np.array([4, 4, 9, 9, 9]),
        episode_offsets=np.array([0, 2, 5]),
        timestep_provenance="episode_offsets",
    )
    np.testing.assert_array_equal(early.bootstrap_mask(5), [1.0, 0.0, 1.0, 1.0, 0.0])


def test_short_zero_based_group_cannot_impersonate_full_h1000_episode() -> None:
    rows = 64
    compressed = TransitionBatch(
        observation=np.zeros((rows, 1)),
        action=np.zeros((rows, 1)),
        reward=np.zeros(rows),
        next_observation=np.zeros((rows, 1)),
        terminated=np.zeros(rows, dtype=bool),
        truncated=np.zeros(rows, dtype=bool),
        dataset_cut=np.zeros(rows, dtype=bool),
        native_timestep=np.arange(rows),
        episode_id=np.zeros(rows, dtype=np.int64),
        episode_offsets=np.asarray([0, rows]),
        timestep_provenance="episode_offsets",
    )
    with pytest.raises(DataValidationError, match="compressed 0..N-1") as caught:
        compressed.bootstrap_mask(1000)
    assert caught.value.status == EstimateStatus.INVALID_DATA.value


def test_fqe_rejects_duck_typed_batch_that_bypasses_transition_validation() -> None:
    class DuckBatch:
        def __len__(self):
            return 3

    actor = ConstantActor()
    for estimator in (FiniteHorizonFQE(horizon=3), FiniteHorizonKMIFQE(horizon=3)):
        if isinstance(estimator, FiniteHorizonKMIFQE):
            estimator.fit(
                DuckBatch(),  # type: ignore[arg-type]
                actor,
                behavior_density=GaussianDensity(),
                fit_keys=np.arange(3, dtype=np.uint64),
            )
        else:
            estimator.fit(
                DuckBatch(),  # type: ignore[arg-type]
                actor,
                fit_keys=np.arange(3, dtype=np.uint64),
            )
        result = estimator.estimate(
            np.zeros((1, 1)), keys=np.asarray([1], dtype=np.uint64)
        )
        assert result.status is EstimateStatus.INVALID_DATA
        assert result.value is None


def test_fqe_fails_closed_for_stochastic_candidate_and_requires_keys() -> None:
    batch = known_mdp_batch(episodes=2)
    stochastic = ConstantActor(semantics=PolicySemantics.STOCHASTIC_KEYED)
    estimator = FiniteHorizonFQE(horizon=3).fit(
        batch,
        stochastic,
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    result = estimator.estimate(np.zeros((1, 1)), keys=np.array([7], dtype=np.uint64))
    assert result.status is EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS
    assert result.value is None
    assert stochastic.calls == []

    invalid_keys = FiniteHorizonFQE(horizon=3).fit(
        batch,
        ConstantActor(),
        fit_keys=np.arange(len(batch), dtype=float),
    )
    invalid_result = invalid_keys.estimate(
        np.zeros((1, 1)),
        keys=np.array([1], dtype=np.uint64),
    )
    assert invalid_result.status is EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS

    with pytest.raises(DataValidationError):
        validate_action_keys(np.array([1.0]), 1)
    with pytest.raises(DataValidationError):
        validate_action_keys(np.array([-1], dtype=np.int64), 1)


def test_method_id_uses_actual_protocol_and_nonconvergence_fails_closed() -> None:
    assert FiniteHorizonFQE(horizon=5).method_id == "FH_FQE_G099_H5"
    assert FiniteHorizonKMIFQE(horizon=5).method_id == "FH_KMIFQE_G099_H5"
    assert FiniteHorizonFQE(gamma=1.0, horizon=5).method_id == "FH_FQE_G1000_H5"
    assert FiniteHorizonFQE().method_id == "FH_FQE_G099_H1000"
    with pytest.raises(ValueError, match="gamma"):
        FiniteHorizonFQE(gamma=True)
    precise_gamma = 0.123456789
    assert finite_horizon_method_id("AR_MBOPE", precise_gamma, 5) == "AR_MBOPE_G0123456789_H5"
    assert (
        finite_horizon_value_convention(precise_gamma, 5)
        == "J_gamma=0.123456789_H=5_raw"
    )

    batch = known_mdp_batch(episodes=2)
    estimators = [
        FiniteHorizonFQE(
            horizon=3,
            ridge=1e-10,
            max_iterations=1,
            tolerance=1e-15,
        ).fit(
            batch,
            ConstantActor(),
            fit_keys=np.arange(len(batch), dtype=np.uint64),
        ),
        FiniteHorizonKMIFQE(
            horizon=3,
            ridge=1e-10,
            max_iterations=1,
            tolerance=1e-15,
        ).fit(
            batch,
            ConstantActor(),
            behavior_density=GaussianDensity(),
            fit_keys=np.arange(len(batch), dtype=np.uint64),
        ),
    ]
    for estimator in estimators:
        result = estimator.estimate(
            np.zeros((1, 1)), keys=np.array([9], dtype=np.uint64)
        )
        assert result.status is EstimateStatus.NO_GO_FIT_CONVERGENCE
        assert result.value is None
        assert result.diagnostics["converged"] is False
        assert result.diagnostics["failed_closed"] is True

    fitted = FiniteHorizonFQE(horizon=3).fit(
        batch,
        ConstantActor(),
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    for invalid_timestep in (True, 1.9):
        with pytest.raises(ValueError, match="integers"):
            fitted.estimate(
                np.zeros((1, 1)),
                keys=np.asarray([1], dtype=np.uint64),
                initial_timestep=invalid_timestep,
            )


def test_kmifqe_recovers_known_value_and_queries_arbitrary_density() -> None:
    batch = known_mdp_batch()
    actor = ConstantActor(value=0.25)
    density = GaussianDensity()
    estimator = FiniteHorizonKMIFQE(
        gamma=0.99,
        horizon=3,
        ridge=1e-10,
        max_iterations=20,
        tolerance=1e-11,
    ).fit(
        batch,
        actor,
        behavior_density=density,
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    result = estimator.estimate(np.zeros((4, 1)), keys=np.arange(4, dtype=np.uint64))

    assert result.status is EstimateStatus.PASS
    assert result.method_id == "FH_KMIFQE_G099_H3"
    assert result.value == pytest.approx(1.0 + 0.99 + 0.99**2, abs=2e-6)
    assert result.support["ess"] == pytest.approx(2 * batch.episode_count)
    assert result.support["ess_fraction"] == pytest.approx(1.0)
    assert result.provenance["method_identity"] == "KMIFQE_PROJECT_ADAPTATION"
    assert result.provenance["official_parity"] is False
    # One exact-density query for logged a' and another for pi(s').
    assert len(density.calls) == 2
    np.testing.assert_array_equal(density.calls[0], 0.0)
    np.testing.assert_array_equal(density.calls[1], 0.25)


def test_kmifqe_existing_density_and_next_action_gates() -> None:
    batch = known_mdp_batch(episodes=2)
    actor = ConstantActor()
    unknown = UnknownClippedDensity()
    estimator = FiniteHorizonKMIFQE(horizon=3).fit(
        batch,
        actor,
        behavior_density=unknown,
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    result = estimator.estimate(np.zeros((1, 1)), keys=np.array([1], dtype=np.uint64))
    assert result.status is EstimateStatus.NO_GO_EXISTING_LOG_DENSITY
    assert result.value is None

    truthy_string = GaussianDensity()
    truthy_string.exact = "false"  # type: ignore[assignment]
    misleading = FiniteHorizonKMIFQE(horizon=3).fit(
        batch,
        actor,
        behavior_density=truthy_string,
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    misleading_result = misleading.estimate(
        np.zeros((1, 1)), keys=np.array([8], dtype=np.uint64)
    )
    assert misleading_result.status is EstimateStatus.NO_GO_EXISTING_LOG_DENSITY
    assert truthy_string.calls == []

    without_next_action = known_mdp_batch(episodes=2, include_next_behavior_action=False)
    missing = FiniteHorizonKMIFQE(horizon=3).fit(
        without_next_action,
        actor,
        behavior_density=GaussianDensity(),
        fit_keys=np.arange(len(without_next_action), dtype=np.uint64),
    )
    missing_result = missing.estimate(
        np.zeros((1, 1)),
        keys=np.array([2], dtype=np.uint64),
    )
    assert missing_result.status is EstimateStatus.NO_GO_MISSING_NEXT_BEHAVIOR_ACTION

    stochastic = FiniteHorizonKMIFQE(horizon=3).fit(
        batch,
        ConstantActor(semantics=PolicySemantics.STOCHASTIC_KEYED),
        behavior_density=GaussianDensity(),
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    stochastic_result = stochastic.estimate(
        np.zeros((1, 1)),
        keys=np.array([4], dtype=np.uint64),
    )
    assert stochastic_result.status is EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS


def test_kmifqe_fails_closed_outside_behavior_support() -> None:
    batch = known_mdp_batch(episodes=2)
    estimator = FiniteHorizonKMIFQE(horizon=3).fit(
        batch,
        ConstantActor(value=2.0),
        behavior_density=GaussianDensity(support_limit=1.0),
        fit_keys=np.arange(len(batch), dtype=np.uint64),
    )
    result = estimator.estimate(np.zeros((1, 1)), keys=np.array([3], dtype=np.uint64))

    assert result.status is EstimateStatus.NO_GO_BEHAVIOR_SUPPORT
    assert result.value is None
    assert result.support["target_support_fraction"] == 0.0


def test_kmifqe_ess_gate_uses_only_active_bellman_rows() -> None:
    rows = 100
    terminal = np.ones(rows, dtype=bool)
    terminal[:2] = False
    next_behavior_action = np.zeros((rows, 1))
    next_behavior_action[1, 0] = 1.0
    batch = TransitionBatch(
        observation=np.zeros((rows, 1)),
        action=np.zeros((rows, 1)),
        reward=np.ones(rows),
        next_observation=np.zeros((rows, 1)),
        terminated=terminal,
        truncated=np.zeros(rows, dtype=bool),
        dataset_cut=np.zeros(rows, dtype=bool),
        native_timestep=np.zeros(rows, dtype=np.int64),
        episode_id=np.arange(rows, dtype=np.int64),
        episode_offsets=np.arange(rows + 1, dtype=np.int64),
        timestep_provenance="native_indices",
        next_behavior_action=next_behavior_action,
    )

    class SkewedDensity(GaussianDensity):
        def log_prob(self, observations, actions, native_timestep):
            del observations, native_timestep
            return np.where(actions[:, 0] > 0.5, 20.0, 0.0)

    estimator = FiniteHorizonKMIFQE(
        horizon=2,
        min_ess_fraction=0.9,
    ).fit(
        batch,
        ConstantActor(),
        behavior_density=SkewedDensity(),
        fit_keys=np.arange(rows, dtype=np.uint64),
    )
    result = estimator.estimate(
        np.zeros((1, 1)), keys=np.asarray([9], dtype=np.uint64)
    )
    assert result.status is EstimateStatus.NO_GO_BEHAVIOR_SUPPORT
    assert result.support["active_rows"] == 2
    assert result.support["ess_fraction"] == pytest.approx(0.5, abs=1e-6)
