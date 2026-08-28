"""Single command-line runner for synthetic acceptance and read-only census."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import RawDeltaTask5Adapter, SealedRawOperator, census_real_assets, sha256_file
from .benchmark import export_metrics, join_oracle_and_score, seal_ranking
from .core import EstimateStatus, PolicySemantics, TransitionBatch, ValueEstimate
from .fqe import (
    FH_FQE_METHOD_ID,
    FH_KMIFQE_METHOD_ID,
    FiniteHorizonFQE,
    FiniteHorizonKMIFQE,
)
from .mbope import (
    AR_MBOPE_ID,
    DOPE_STYLE_MB_FF_ID,
    ETM_MBOPE_ID,
    make_model_based_estimator,
)


TOY_CONTEXT = "synthetic_linear_native_time_v1"
TOY_HORIZON = 5
TOY_GAMMA = 0.99


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _ToyActor:
    policy_id: str
    gain: float
    bias: float
    semantics: PolicySemantics = PolicySemantics.DETERMINISTIC

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(observations, dtype=float)
        times = np.asarray(native_timestep, dtype=float)
        checked_keys = np.asarray(keys)
        if checked_keys.shape != (len(obs),) or checked_keys.dtype.kind not in "iu":
            raise ValueError("toy actor requires one explicit integer key per row")
        action = self.gain * obs[:, 0] + self.bias + 0.015 * times
        return action[:, None]


class _ExactGaussianDensity:
    density_id = "synthetic_unsquashed_gaussian_sigma_0.8"
    exact = True
    sigma = 0.8

    def log_prob(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        native_timestep: np.ndarray,
    ) -> np.ndarray:
        del observations, native_timestep
        action = np.asarray(actions, dtype=float)
        return -0.5 * np.sum((action / self.sigma) ** 2, axis=1) - action.shape[1] * np.log(
            self.sigma * np.sqrt(2.0 * np.pi)
        )


def _toy_batch(seed: int, *, episodes: int = 48) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    observation: list[list[float]] = []
    action: list[list[float]] = []
    reward: list[float] = []
    next_observation: list[list[float]] = []
    next_behavior_action: list[list[float]] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    dataset_cut: list[bool] = []
    timestep: list[int] = []
    episode_id: list[int] = []
    reasons: list[str] = []
    for episode in range(episodes):
        state = float(rng.uniform(-1.25, 1.25))
        behavior_actions = rng.normal(scale=_ExactGaussianDensity.sigma, size=TOY_HORIZON + 1)
        for native_t in range(TOY_HORIZON):
            behavior_action = float(behavior_actions[native_t])
            next_state = 0.62 * state + 0.28 * behavior_action + 0.035 * native_t
            observed_reward = 1.0 + 0.18 * state - 0.12 * behavior_action - 0.025 * behavior_action**2
            is_horizon = native_t == TOY_HORIZON - 1
            observation.append([state])
            action.append([behavior_action])
            reward.append(observed_reward)
            next_observation.append([next_state])
            next_behavior_action.append([float(behavior_actions[native_t + 1])])
            terminated.append(False)
            truncated.append(is_horizon)
            dataset_cut.append(False)
            timestep.append(native_t)
            episode_id.append(episode)
            reasons.append("horizon" if is_horizon else "none")
            state = next_state
    rows = episodes * TOY_HORIZON
    digest_payload = np.concatenate(
        [np.asarray(observation).reshape(-1), np.asarray(action).reshape(-1), np.asarray(reward)]
    )
    return TransitionBatch(
        observation=np.asarray(observation),
        action=np.asarray(action),
        reward=np.asarray(reward),
        next_observation=np.asarray(next_observation),
        terminated=np.asarray(terminated),
        truncated=np.asarray(truncated),
        dataset_cut=np.asarray(dataset_cut),
        native_timestep=np.asarray(timestep, dtype=np.int64),
        episode_id=np.asarray(episode_id, dtype=np.int64),
        episode_offsets=np.arange(0, rows + 1, TOY_HORIZON, dtype=np.int64),
        timestep_provenance="episode_offsets",
        next_behavior_action=np.asarray(next_behavior_action),
        truncation_reason=np.asarray(reasons),
        source_digest=sha256(np.asarray(digest_payload, dtype="<f8").tobytes()).hexdigest(),
    )


def _toy_candidates() -> list[_ToyActor]:
    return [
        _ToyActor(f"toy-policy-{index}", gain=-0.18 + 0.06 * index, bias=-0.30 + 0.15 * index)
        for index in range(5)
    ]


def _toy_initial_states() -> np.ndarray:
    return np.asarray([[-1.0], [-0.65], [-0.25], [0.0], [0.3], [0.7], [1.1]], dtype=float)


def _toy_oracle(candidates: Sequence[_ToyActor], initial_states: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate_index, candidate in enumerate(candidates):
        totals = np.zeros(len(initial_states), dtype=float)
        state = initial_states.copy()
        discount = 1.0
        for native_t in range(TOY_HORIZON):
            keys = np.arange(len(state), dtype=np.uint64) + candidate_index * 1000 + native_t * 100
            action = candidate.sample_actions(
                state,
                np.full(len(state), native_t, dtype=np.int64),
                keys=keys,
            )[:, 0]
            totals += discount * (
                1.0 + 0.18 * state[:, 0] - 0.12 * action - 0.025 * action**2
            )
            state[:, 0] = 0.62 * state[:, 0] + 0.28 * action + 0.035 * native_t
            discount *= TOY_GAMMA
        result[candidate.policy_id] = float(np.mean(totals))
    return result


def _combined_diagnostics(estimate: ValueEstimate) -> dict[str, Any]:
    diagnostics = dict(estimate.diagnostics)
    diagnostics.update(dict(estimate.support))
    diagnostics.update(dict(estimate.cost))
    if "runtime_seconds" not in diagnostics:
        diagnostics["runtime_seconds"] = float(
            diagnostics.get("fit_seconds", 0.0) + diagnostics.get("estimate_seconds", 0.0)
        )
    return diagnostics


def _toy_method_scope() -> dict[str, dict[str, str]]:
    return {
        FH_FQE_METHOD_ID: {
            "status": "TOY_MVP_PASS",
            "scope": "finite-horizon NumPy ridge FQE; project adaptation, not upstream reproduction",
        },
        FH_KMIFQE_METHOD_ID: {
            "status": "TOY_MVP_PASS",
            "scope": "exact-density kernel/importance feasibility implementation; full learned Hessian metric parity pending",
        },
        ETM_MBOPE_ID: {
            "status": "TOY_MVP_PASS",
            "scope": "compact RFF contrastive energy plus Langevin sampler; not official ETM architecture parity",
        },
        DOPE_STYLE_MB_FF_ID: {
            "status": "TOY_MVP_PASS",
            "scope": "project-defined residual-Gaussian random-feature ridge ensemble; DOPE is not an algorithm",
        },
        AR_MBOPE_ID: {
            "status": "TOY_MVP_PASS",
            "scope": "B06-inspired fixed-order teacher-forced/sequential autoregressive model",
        },
        "RAW_DELTA_TASK5": {
            "status": "TOY_ADAPTER_PARITY_PASS",
            "scope": "sealed-output delegation only; no Raw-RKME mathematics copied into this repository",
        },
    }


def run_toy(output: str | Path, *, seed: int = 7) -> dict[str, Any]:
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    batch = _toy_batch(seed)
    candidates = _toy_candidates()
    initial = _toy_initial_states()
    density = _ExactGaussianDensity()
    estimates: dict[str, dict[str, ValueEstimate]] = {}

    for method_id, estimator_type in (
        (FH_FQE_METHOD_ID, FiniteHorizonFQE),
        (FH_KMIFQE_METHOD_ID, FiniteHorizonKMIFQE),
    ):
        method_estimates: dict[str, ValueEstimate] = {}
        for candidate_index, candidate in enumerate(candidates):
            estimator = estimator_type(
                gamma=TOY_GAMMA,
                horizon=TOY_HORIZON,
                ridge=1e-7,
                max_iterations=100,
            )
            fit_keys = np.arange(len(batch), dtype=np.uint64) + candidate_index * 10_000
            if method_id == FH_KMIFQE_METHOD_ID:
                estimator.fit(batch, candidate, behavior_density=density, fit_keys=fit_keys)
            else:
                estimator.fit(batch, candidate, fit_keys=fit_keys)
            estimate_keys = np.arange(len(initial), dtype=np.uint64) + candidate_index * 100_000
            method_estimates[candidate.policy_id] = estimator.estimate(initial, keys=estimate_keys)
        estimates[method_id] = method_estimates

    model_kwargs: dict[str, dict[str, Any]] = {
        DOPE_STYLE_MB_FF_ID: {"ensemble_members": 3, "hidden_dim": 24},
        AR_MBOPE_ID: {"ensemble_members": 2, "hidden_dim": 20},
        ETM_MBOPE_ID: {
            "hidden_dim": 24,
            "energy_features": 48,
            "negatives": 3,
            "contrastive_steps": 40,
            "learning_rate": 0.01,
            "langevin_steps": 10,
            "langevin_step_size": 0.025,
        },
    }
    for method_index, method_id in enumerate(
        (ETM_MBOPE_ID, DOPE_STYLE_MB_FF_ID, AR_MBOPE_ID)
    ):
        estimator = make_model_based_estimator(
            method_id,
            gamma=TOY_GAMMA,
            horizon=TOY_HORIZON,
            rollouts_per_initial=12,
            ridge=1e-4,
            **model_kwargs[method_id],
        ).fit(
            batch,
            candidates[0],
            fit_keys=np.asarray([seed + 100 + method_index], dtype=np.uint64),
        )
        method_estimates = {}
        for candidate_index, candidate in enumerate(candidates):
            estimate_keys = np.arange(len(initial), dtype=np.uint64) + candidate_index * 100_000
            method_estimates[candidate.policy_id] = estimator.estimate(
                initial,
                keys=estimate_keys,
                candidate=candidate,
            )
        estimates[method_id] = method_estimates

    candidate_tasks = {candidate.policy_id: "TOY_TASK" for candidate in candidates}
    raw_fixture_path = destination / "delegated_raw_fixture.json"
    raw_fixture_scores = {
        candidate.policy_id: abs(candidate.gain + 0.06) + abs(candidate.bias - 0.15)
        for candidate in candidates
    }
    _write_json(
        raw_fixture_path,
        {
            "schema": "synthetic-frozen-raw-output.v1",
            "context_id": TOY_CONTEXT,
            "scores": raw_fixture_scores,
            "synthetic_fixture_only": True,
        },
    )
    raw_adapter = RawDeltaTask5Adapter(
        SealedRawOperator(raw_fixture_path, sha256_file(raw_fixture_path))
    )
    raw_scores = raw_adapter.score(
        context_id=TOY_CONTEXT,
        task_id="TOY_TASK",
        candidate_tasks=candidate_tasks,
        query_artifact="synthetic://raw-delta-query",
        membership_digest=batch.source_digest or "",
    )

    seals: dict[str, str] = {}
    raw_seal = seal_ranking(
        destination / "seals" / "RAW_DELTA_TASK5.json",
        method_id="RAW_DELTA_TASK5",
        context_id=TOY_CONTEXT,
        score_kind="compatibility",
        scores=raw_scores,
        provenance={
            "synthetic_fixture_only": True,
            "delegated_artifact_sha256": sha256_file(raw_fixture_path),
            "membership_digest": batch.source_digest,
        },
        higher_is_better=False,
    )
    seals["RAW_DELTA_TASK5"] = str(raw_seal.path)

    for method_id, method_estimates in estimates.items():
        scores = {candidate_id: estimate.value for candidate_id, estimate in method_estimates.items()}
        statuses = {
            candidate_id: estimate.status.value for candidate_id, estimate in method_estimates.items()
        }
        diagnostics = {
            candidate_id: _combined_diagnostics(estimate)
            for candidate_id, estimate in method_estimates.items()
        }
        seal = seal_ranking(
            destination / "seals" / f"{method_id}.json",
            method_id=method_id,
            context_id=TOY_CONTEXT,
            score_kind="value",
            scores=scores,
            statuses=statuses,
            diagnostics=diagnostics,
            provenance={
                "synthetic_fixture_only": True,
                "dataset_digest": batch.source_digest,
                "native_timestep_provenance": batch.timestep_provenance,
                "gamma": TOY_GAMMA,
                "horizon": TOY_HORIZON,
                "stage": "pre_join_sealed",
            },
            value_convention=f"toy_J_gamma={TOY_GAMMA}_H={TOY_HORIZON}_raw",
        )
        seals[method_id] = str(seal.path)

    # Synthetic oracle evaluation is deliberately invoked only after every
    # method score and ranking has been sealed.
    oracle_values = _toy_oracle(candidates, initial)
    metrics = [
        join_oracle_and_score(seal_path, oracle_values=oracle_values)
        for _, seal_path in sorted(seals.items())
    ]
    metric_paths = export_metrics(
        metrics,
        json_path=destination / "metrics.json",
        csv_path=destination / "metrics.csv",
    )
    result = {
        "schema": "policy-learnware.toy-run.v1",
        "status": "TOY_MVP_PASS"
        if all(
            estimate.status is EstimateStatus.PASS
            for method_estimates in estimates.values()
            for estimate in method_estimates.values()
        )
        else "TOY_MVP_FAILED",
        "synthetic_fixture_only": True,
        "context_id": TOY_CONTEXT,
        "gamma": TOY_GAMMA,
        "horizon": TOY_HORIZON,
        "candidate_count": len(candidates),
        "transition_count": len(batch),
        "method_scope": _toy_method_scope(),
        "estimates": {
            method_id: {
                candidate_id: estimate.to_dict()
                for candidate_id, estimate in method_estimates.items()
            }
            for method_id, method_estimates in estimates.items()
        },
        "raw_scores": raw_scores,
        "oracle_values_after_seal": oracle_values,
        "metrics": metrics,
        "artifacts": metric_paths,
        "ranking_seals": seals,
        "real_asset_training_started": False,
    }
    _write_json(destination / "run.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-learnware-ope")
    commands = parser.add_subparsers(dest="command", required=True)
    toy = commands.add_parser("toy", help="run all methods on a synthetic finite-horizon fixture")
    toy.add_argument("--output", required=True, type=Path)
    toy.add_argument("--seed", type=int, default=7)
    census = commands.add_parser("census", help="perform a read-only real-asset adequacy census")
    census.add_argument("--dataset", required=True, type=Path)
    census.add_argument("--oracle", type=Path)
    census.add_argument("--density-manifest", type=Path)
    census.add_argument("--actor-authority", type=Path)
    census.add_argument("--horizon", type=int, default=1000)
    census.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "toy":
        result = run_toy(args.output, seed=args.seed)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "run": str(Path(args.output).resolve() / "run.json"),
                    "methods": sorted(result["method_scope"]),
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "TOY_MVP_PASS" else 1
    report = census_real_assets(
        dataset_path=args.dataset,
        oracle_path=args.oracle,
        density_manifest_path=args.density_manifest,
        actor_authority_path=args.actor_authority,
        horizon=args.horizon,
    )
    if args.output:
        _write_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
