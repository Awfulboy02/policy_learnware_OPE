"""Single command-line runner for synthetic acceptance and read-only census."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import tomllib
from typing import Any, Sequence

import numpy as np


ARTIFACTS_ROOT_ENV = "RL_LEARNWARE_ARTIFACTS_ROOT"
SYSTEM_GIT = "/usr/bin/git"
RELOCATION_MANIFEST_SHA256 = "81e726c297c78ebc110df017e06e6fb56de73face39371198635299f931bfed9"
RELOCATION_MANIFEST_SCHEMA = "rl-learnware-relocation/v1"


def _git_clean_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def _reject_symlink_components(path: Path, where: str) -> None:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        raise ValueError(f"{where} must be an absolute path")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{where} path component is not auditable: {current}") from exc
        if current.is_symlink():
            raise ValueError(f"{where} must not contain a symlink component: {current}")


def _validate_relocation_manifest(root: Path) -> None:
    manifest = root / "relocation_manifest.json"
    _reject_symlink_components(manifest, "fallback relocation manifest")
    try:
        raw = manifest.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fallback artifacts root lacks the published relocation manifest") from exc
    if sha256(raw).hexdigest() != RELOCATION_MANIFEST_SHA256:
        raise ValueError("fallback relocation manifest digest differs from the published manifest")
    if not isinstance(payload, dict) or set(payload) != {"schema", "mappings"}:
        raise ValueError("fallback relocation manifest fields differ")
    if payload["schema"] != RELOCATION_MANIFEST_SCHEMA or not isinstance(payload["mappings"], list):
        raise ValueError("fallback relocation manifest schema differs")
    required = {"kind", "source", "target", "content_manifest_sha256", "file_count", "total_bytes", "role", "access_class", "status"}
    optional = {"completeness", "known_missing"}
    if any(not isinstance(row, dict) or not required <= set(row) or set(row) - required - optional for row in payload["mappings"]):
        raise ValueError("fallback relocation manifest mapping fields differ")

from .adapters import (
    FROZEN_V03_COMMIT,
    FROZEN_V03_TREE,
    RAW_FIXTURE_METHOD_ID,
    RAW_QUERY_SCHEMA,
    RAW_REQUEST_SCHEMA,
    RAW_RESPONSE_SCHEMA,
    RawDeltaTask5Adapter,
    SealedRawOperator,
    census_real_assets,
    sha256_file,
)
from .benchmark import (
    ORACLE_MANIFEST_SCHEMA,
    candidate_set_digest,
    export_metrics,
    join_oracle_and_score,
    oracle_manifest_digest,
    seal_ranking,
)
from .core import (
    EstimateStatus,
    PolicySemantics,
    TransitionBatch,
    ValueEstimate,
    finite_horizon_value_convention,
)
from .fqe import FiniteHorizonFQE, FiniteHorizonKMIFQE
from .mbope import (
    AR_MBOPE_ID,
    DOPE_STYLE_MB_FF_ID,
    ETM_MBOPE_ID,
    make_model_based_estimator,
)


TOY_CONTEXT = "synthetic_linear_native_time_v1"
TOY_HORIZON = 5
TOY_GAMMA = 0.99
TOY_VALUE_CONVENTION = finite_horizon_value_convention(TOY_GAMMA, TOY_HORIZON)
V04B_PLAN_SHA256 = "5fb35cc2ee4c27afd411f77f0c2813088b6d6ab901f8910f442ed5b231e1719e"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _payload_digest(payload: Any) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _write_canonical_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(payload))


def _containing_git_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
        bare_markers = (
            (candidate / "HEAD").is_file()
            and (candidate / "objects").is_dir()
            and (candidate / "refs").is_dir()
            and (candidate / "config").is_file()
        )
        if bare_markers:
            try:
                bare = subprocess.run(
                    [
                        SYSTEM_GIT,
                        f"--git-dir={candidate}",
                        "config",
                        "--bool",
                        "--get",
                        "core.bare",
                    ],
                    check=False,
                    env=_git_clean_env(),
                    text=True,
                    capture_output=True,
                )
            except OSError:
                return candidate.resolve()
            if bare.returncode != 0 or bare.stdout.strip() == "true":
                return candidate.resolve()
    return None


def _verified_source_checkout() -> Path | None:
    """Return this package's Git root only for the canonical source layout."""

    lexical_file = Path(__file__).absolute()
    try:
        _reject_symlink_components(lexical_file, "package source")
    except ValueError:
        return None
    current_file = lexical_file.resolve()
    candidate = current_file.parents[2]
    expected_cli = candidate / "src" / "policy_learnware_ope" / "cli.py"
    pyproject = candidate / "pyproject.toml"
    try:
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project_name = metadata["project"]["name"]
        same_cli = expected_cli.samefile(current_file)
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None
    if project_name != "policy-learnware-ope" or not same_cli:
        return None
    if not (candidate / ".git").is_dir():
        return None
    commands = (
        [SYSTEM_GIT, "rev-parse", "--show-toplevel"],
        [SYSTEM_GIT, "rev-parse", "--verify", "HEAD^{commit}"],
        [SYSTEM_GIT, "ls-files", "--error-unmatch", "pyproject.toml"],
        [SYSTEM_GIT, "diff", "--quiet", "HEAD", "--", "pyproject.toml"],
        [SYSTEM_GIT, "diff", "--cached", "--quiet", "HEAD", "--", "pyproject.toml"],
        [SYSTEM_GIT, "status", "--porcelain", "--", "pyproject.toml", "src/policy_learnware_ope"],
    )
    try:
        results = [subprocess.run(command, cwd=candidate, env=_git_clean_env(), check=False, text=True, capture_output=True) for command in commands]
    except (OSError, ValueError):
        return None
    if any(result.returncode != 0 for result in results):
        return None
    if results[-1].stdout:
        return None
    try:
        top = Path(results[0].stdout.strip()).resolve(strict=True)
    except (OSError, ValueError):
        return None
    return candidate if top == candidate.resolve() else None


def _installed_package_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    try:
        package_version = distribution_version("policy-learnware-ope")
        status = "INSTALLED_IMMUTABLE_CONTENT"
    except PackageNotFoundError:
        package_version = "UNAVAILABLE"
        status = "UNVERIFIED_PACKAGE_LAYOUT"
    files = sorted(path for path in package_root.rglob("*.py") if path.is_file())
    manifest = {
        "schema": "policy-learnware.installed-python-tree.v1",
        "distribution": "policy-learnware-ope",
        "version": package_version,
        "files": [
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    tree_digest = _payload_digest(manifest)
    return {
        "commit": f"PACKAGE_CONTENT_SHA256:{tree_digest}",
        "tree": tree_digest,
        "worktree_status": status,
        "package_name": "policy-learnware-ope",
        "package_version": package_version,
    }


def _resolve_artifact_path(
    path: str | Path,
    *,
    artifacts_root: str | Path | None = None,
) -> Path:
    """Resolve one CLI artifact path without inheriting a foreign checkout."""

    requested = Path(path).expanduser()
    if requested.is_absolute():
        _reject_symlink_components(requested, "artifact path")
        return requested.resolve()
    root_value = artifacts_root
    if root_value is None:
        if ARTIFACTS_ROOT_ENV in os.environ:
            root_value = os.environ[ARTIFACTS_ROOT_ENV]
            if not str(root_value).strip():
                raise ValueError(f"{ARTIFACTS_ROOT_ENV} must not be empty")
    if root_value is None:
        checkout = _verified_source_checkout()
        if checkout is None:
            raise ValueError(
                "relative artifact path requires --artifacts-root or "
                "RL_LEARNWARE_ARTIFACTS_ROOT outside a verified source checkout"
            )
        root = checkout.parent / "artifacts"
        _reject_symlink_components(root, "fallback artifacts root")
        _validate_relocation_manifest(root)
    else:
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            raise ValueError("artifacts root must be an absolute path")
    _reject_symlink_components(root, "artifacts root")
    root = root.resolve()
    lexical = root / requested
    _reject_symlink_components(lexical, "artifact path")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("relative artifact path escapes artifacts root") from error
    return resolved


def _guard_output_location(path: str | Path) -> Path:
    """Reject experiment writes into any Git repository."""

    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ValueError("guarded output path must already be absolute")
    _reject_symlink_components(requested, "output path")
    destination = requested.resolve()
    containing_repo = _containing_git_root(destination)
    if containing_repo is not None:
        raise PermissionError(
            f"refusing to write into a Git repository: {containing_repo}"
        )
    return destination


def _implementation_identity(commit_override: str | None = None) -> dict[str, Any]:
    if commit_override is not None:
        return {
            "commit": str(commit_override),
            "tree": "CALLER_SUPPLIED",
            "worktree_status": "CALLER_SUPPLIED",
        }
    repository = _verified_source_checkout()
    if repository is None:
        return _installed_package_identity()
    try:
        commit = subprocess.run(
            [SYSTEM_GIT, "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
            env=_git_clean_env(),
        ).stdout.strip()
        tree = subprocess.run(
            [SYSTEM_GIT, "rev-parse", "HEAD^{tree}"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
            env=_git_clean_env(),
        ).stdout.strip()
        worktree = subprocess.run(
            [SYSTEM_GIT, "status", "--porcelain"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
            env=_git_clean_env(),
        ).stdout
        return {
            "commit": commit,
            "tree": tree,
            "worktree_status": "CLEAN" if not worktree else "DIRTY",
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "UNAVAILABLE",
            "tree": "UNAVAILABLE",
            "worktree_status": "UNVERIFIED",
        }


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
    source_digest = _payload_digest(
        {
            "schema": "policy-learnware.synthetic-transition-batch.v1",
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "terminated": terminated,
            "truncated": truncated,
            "dataset_cut": dataset_cut,
            "native_timestep": timestep,
            "episode_id": episode_id,
            "episode_offsets": list(range(0, rows + 1, TOY_HORIZON)),
        }
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
        source_digest=source_digest,
    )


def _raw_membership_digest(batch: TransitionBatch) -> str:
    """Bind the reward-free physical transition view consumed by Raw."""

    return _payload_digest(
        {
            "schema": "policy-learnware.raw-physical-membership.v1",
            "fields": {
                "observation": batch.observation.tolist(),
                "action": batch.action.tolist(),
                "next_observation": batch.next_observation.tolist(),
                "native_timestep": batch.native_timestep.tolist(),
                "episode_offsets": batch.episode_offsets.tolist(),
            },
        }
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
    return diagnostics


def _without_volatile_fields(value: Any) -> Any:
    volatile_tokens = (
        "runtime",
        "wallclock",
        "elapsed",
        "duration",
        "latency",
        "fitseconds",
        "estimateseconds",
    )
    if isinstance(value, dict):
        stable: dict[str, Any] = {}
        for key, item in value.items():
            normalized = "".join(character for character in str(key).casefold() if character.isalnum())
            if any(token in normalized for token in volatile_tokens) or normalized.endswith(
                ("seconds", "milliseconds", "microseconds", "nanoseconds")
            ):
                continue
            stable[str(key)] = _without_volatile_fields(item)
        return stable
    if isinstance(value, list):
        return [_without_volatile_fields(item) for item in value]
    return value


def _stable_estimate(estimate: ValueEstimate) -> dict[str, Any]:
    payload = _without_volatile_fields(estimate.to_dict())
    payload["cost"] = {"reported_separately": "runtime.json"}
    return payload


def _toy_method_scope(method_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    scopes: dict[str, dict[str, Any]] = {}
    for method_id in method_ids:
        if method_id.startswith("FH_FQE_"):
            scopes[method_id] = {
                "status": "TOY_MVP_PASS",
                "scope": (
                    "finite-horizon NumPy ridge FQE; project adaptation, "
                    "not upstream reproduction"
                ),
                "official_paper_parity": False,
            }
        elif method_id.startswith("FH_KMIFQE_"):
            scopes[method_id] = {
                "status": "TOY_MVP_PASS",
                "scope": (
                    "KMIFQE project adaptation with an exact-density fixture; "
                    "full learned-Hessian reference parity is deferred"
                ),
                "scientific_role": "PROJECT_ADAPTATION",
                "official_paper_parity": False,
            }
        elif method_id.startswith("ETM_MBOPE_"):
            scopes[method_id] = {
                "status": "TOY_MVP_PASS",
                "scope": "compact contrastive-energy/Langevin project proxy",
                "scientific_role": "PROJECT_CONTRASTIVE_ENERGY_ADAPTATION_PROXY",
                "official_paper_parity": False,
            }
        elif method_id.startswith("DOPE_STYLE_MB_FF_"):
            scopes[method_id] = {
                "status": "TOY_MVP_PASS",
                "scope": (
                    "project-defined residual-Gaussian random-feature ridge ensemble; "
                    "DOPE is not an algorithm"
                ),
                "scientific_role": "PROJECT_DEFINED_REFERENCE",
                "official_paper_parity": False,
            }
        elif method_id.startswith("AR_MBOPE_"):
            scopes[method_id] = {
                "status": "TOY_MVP_PASS",
                "scope": "B06-inspired fixed-order teacher-forced/sequential project proxy",
                "scientific_role": "PROJECT_METHOD_LEVEL_ADAPTATION_PROXY",
                "official_paper_parity": False,
            }
        else:
            raise ValueError(f"unknown toy method identity: {method_id}")
    scopes[RAW_FIXTURE_METHOD_ID] = {
        "status": "TOY_FIXTURE_PASS",
        "scope": "reward-free sealed-response adapter fixture; no production Raw/RKME operator is connected",
        "scientific_role": "FIXTURE_ONLY",
        "official_paper_parity": False,
        "production_status": "NO_GO_RAW_OPERATOR_AUTHORITY",
    }
    return scopes


def run_toy(
    output: str | Path,
    *,
    seed: int = 7,
    implementation_commit: str | None = None,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    destination = _guard_output_location(
        _resolve_artifact_path(output, artifacts_root=artifacts_root)
    )
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"toy output exists and is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        raise FileExistsError(
            f"toy output directory must be empty; refusing partial overwrite: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    implementation = _implementation_identity(implementation_commit)
    implementation_commit = implementation["commit"]
    model_specs: list[tuple[str, str, dict[str, Any]]] = [
        (
            ETM_MBOPE_ID,
            "ETM_MBOPE",
            {
                "hidden_dim": 24,
                "energy_features": 48,
                "negatives": 3,
                "contrastive_steps": 40,
                "learning_rate": 0.01,
                "langevin_steps": 10,
                "langevin_step_size": 0.025,
            },
        ),
        (
            DOPE_STYLE_MB_FF_ID,
            "DOPE_STYLE_MB_FF",
            {"ensemble_members": 3, "hidden_dim": 24},
        ),
        (
            AR_MBOPE_ID,
            "AR_MBOPE",
            {"ensemble_members": 2, "hidden_dim": 20},
        ),
    ]
    config: dict[str, Any] = {
        "schema": "policy-learnware.toy-config.v2",
        "seed": seed,
        "context_id": TOY_CONTEXT,
        "gamma": TOY_GAMMA,
        "horizon": TOY_HORIZON,
        "episodes": 48,
        "candidate_count": 5,
        "fqe": {
            "ridge": 1e-7,
            "max_iterations": 2500,
            "tolerance": 1e-8,
        },
        "model_based": {family: kwargs for _, family, kwargs in model_specs},
        "rollouts_per_initial": 12,
    }
    config_digest = _payload_digest(config)
    batch = _toy_batch(seed)
    candidates = _toy_candidates()
    initial = _toy_initial_states()
    density = _ExactGaussianDensity()
    estimates: dict[str, dict[str, ValueEstimate]] = {}
    runtime_by_method: dict[str, float] = {}

    for estimator_type in (FiniteHorizonFQE, FiniteHorizonKMIFQE):
        method_started = perf_counter()
        method_estimates: dict[str, ValueEstimate] = {}
        actual_method_id: str | None = None
        for candidate_index, candidate in enumerate(candidates):
            estimator = estimator_type(
                gamma=TOY_GAMMA,
                horizon=TOY_HORIZON,
                ridge=1e-7,
                max_iterations=2500,
                tolerance=1e-8,
            )
            if actual_method_id is None:
                actual_method_id = estimator.method_id
            elif estimator.method_id != actual_method_id:
                raise RuntimeError("one estimator family produced inconsistent method IDs")
            fit_keys = np.arange(len(batch), dtype=np.uint64) + candidate_index * 10_000
            if estimator_type is FiniteHorizonKMIFQE:
                estimator.fit(batch, candidate, behavior_density=density, fit_keys=fit_keys)
            else:
                estimator.fit(batch, candidate, fit_keys=fit_keys)
            estimate_keys = np.arange(len(initial), dtype=np.uint64) + candidate_index * 100_000
            method_estimates[candidate.policy_id] = estimator.estimate(initial, keys=estimate_keys)
        if actual_method_id is None:
            raise RuntimeError("toy fixture has no candidates")
        estimates[actual_method_id] = method_estimates
        runtime_by_method[actual_method_id] = perf_counter() - method_started

    for method_index, (selector_id, _family, model_kwargs) in enumerate(model_specs):
        method_started = perf_counter()
        estimator = make_model_based_estimator(
            selector_id,
            gamma=TOY_GAMMA,
            horizon=TOY_HORIZON,
            rollouts_per_initial=12,
            ridge=1e-4,
            **model_kwargs,
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
        estimates[estimator.method_id] = method_estimates
        runtime_by_method[estimator.method_id] = perf_counter() - method_started

    candidate_tasks = {candidate.policy_id: "TOY_TASK" for candidate in candidates}
    candidate_ids = sorted(candidate_tasks)
    raw_membership_digest = _raw_membership_digest(batch)
    query_payload = {
        "schema": RAW_QUERY_SCHEMA,
        "context_id": TOY_CONTEXT,
        "membership_digest": raw_membership_digest,
        "fields": {
            "observation": batch.observation.tolist(),
            "action": batch.action.tolist(),
            "next_observation": batch.next_observation.tolist(),
            "native_timestep": batch.native_timestep.tolist(),
            "episode_offsets": batch.episode_offsets.tolist(),
        },
    }
    raw_query_path = destination / "raw_query.reward_free.json"
    _write_canonical_json(raw_query_path, query_payload)
    query_artifact_digest = sha256_file(raw_query_path)
    raw_fixture_scores = {
        candidate.policy_id: abs(candidate.gain + 0.06) + abs(candidate.bias - 0.15)
        for candidate in candidates
    }
    raw_request_binding = {
        "request_schema": RAW_REQUEST_SCHEMA,
        "method_id": RAW_FIXTURE_METHOD_ID,
        "task_id": "TOY_TASK",
        "context_id": TOY_CONTEXT,
        "candidate_ids": candidate_ids,
        "query": {
            "schema": RAW_QUERY_SCHEMA,
            "artifact_sha256": query_artifact_digest,
            "fields": [
                "observation",
                "action",
                "next_observation",
                "native_timestep",
                "episode_offsets",
            ],
            "forbidden_fields": ["reward", "oracle", "candidate_action"],
        },
        "membership_digest": raw_membership_digest,
    }
    raw_request_sha256 = sha256(
        json.dumps(raw_request_binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw_fixture_path = destination / "raw_fixture.response.json"
    _write_json(
        raw_fixture_path,
        {
            "schema": RAW_RESPONSE_SCHEMA,
            "request_binding": raw_request_binding,
            "request_sha256": raw_request_sha256,
            "scores": raw_fixture_scores,
            "synthetic_fixture_only": True,
        },
    )
    raw_adapter = RawDeltaTask5Adapter(
        SealedRawOperator(raw_fixture_path, sha256_file(raw_fixture_path)),
        method_id=RAW_FIXTURE_METHOD_ID,
    )
    raw_started = perf_counter()
    raw_scores = raw_adapter.score(
        context_id=TOY_CONTEXT,
        task_id="TOY_TASK",
        candidate_tasks=candidate_tasks,
        query_artifact_digest=query_artifact_digest,
        membership_digest=raw_membership_digest,
    )
    runtime_by_method[RAW_FIXTURE_METHOD_ID] = perf_counter() - raw_started

    seals: dict[str, dict[str, str]] = {}
    raw_seal = seal_ranking(
        destination / "seals" / f"{RAW_FIXTURE_METHOD_ID}.json",
        method_id=RAW_FIXTURE_METHOD_ID,
        context_id=TOY_CONTEXT,
        score_kind="compatibility",
        scores=raw_scores,
        provenance={
            "synthetic_fixture_only": True,
            "scientific_role": "FIXTURE_ONLY",
            "official_paper_parity": False,
            "response_artifact_sha256": sha256_file(raw_fixture_path),
            "query_artifact_sha256": query_artifact_digest,
            "membership_digest": raw_membership_digest,
            "seed": seed,
            "implementation_commit": implementation_commit,
            "implementation_tree": implementation["tree"],
            "implementation_worktree_status": implementation["worktree_status"],
            "config_sha256": config_digest,
        },
        higher_is_better=False,
        value_convention=TOY_VALUE_CONVENTION,
    )
    seals[RAW_FIXTURE_METHOD_ID] = {
        "path": f"seals/{RAW_FIXTURE_METHOD_ID}.json",
        "sha256": raw_seal.digest,
    }

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
                "seed": seed,
                "implementation_commit": implementation_commit,
                "implementation_tree": implementation["tree"],
                "implementation_worktree_status": implementation["worktree_status"],
                "config_sha256": config_digest,
                "official_paper_parity": False,
            },
            value_convention=TOY_VALUE_CONVENTION,
        )
        seals[method_id] = {
            "path": f"seals/{method_id}.json",
            "sha256": seal.digest,
        }

    # Synthetic oracle evaluation is deliberately invoked only after every
    # method score and ranking has been sealed.
    oracle_values = _toy_oracle(candidates, initial)
    oracle_manifest = {
        "schema": ORACLE_MANIFEST_SCHEMA,
        "context_id": TOY_CONTEXT,
        "candidate_values": oracle_values,
        "candidate_set_digest": candidate_set_digest(candidate_ids),
        "value_convention": TOY_VALUE_CONVENTION,
    }
    expected_oracle_digest = oracle_manifest_digest(oracle_manifest)
    _write_canonical_json(destination / "oracle_manifest.synthetic.json", oracle_manifest)
    metrics: list[dict[str, Any]] = []
    for method_id, seal_ref in sorted(seals.items()):
        metric = join_oracle_and_score(
            destination / seal_ref["path"],
            expected_seal_digest=seal_ref["sha256"],
            oracle_manifest=oracle_manifest,
            expected_oracle_digest=expected_oracle_digest,
        )
        # Wall-clock measurements are attached only after the stable seal has
        # been authenticated and joined.
        metric["runtime_seconds"] = runtime_by_method[method_id]
        metrics.append(metric)
    metric_paths = export_metrics(
        metrics,
        json_path=destination / "metrics.json",
        csv_path=destination / "metrics.csv",
    )
    runtime_payload = {
        "schema": "policy-learnware.runtime.v1",
        "outside_ranking_seal": True,
        "method_runtime_seconds": runtime_by_method,
    }
    _write_json(destination / "runtime.json", runtime_payload)
    stable_estimates = {
        method_id: {
            candidate_id: _stable_estimate(estimate)
            for candidate_id, estimate in method_estimates.items()
        }
        for method_id, method_estimates in estimates.items()
    }
    reproducibility_payload = {
        "seed": seed,
        "config_sha256": config_digest,
        "estimates": stable_estimates,
        "raw_scores": raw_scores,
        "ranking_seal_sha256": {
            method_id: seal_ref["sha256"] for method_id, seal_ref in sorted(seals.items())
        },
    }
    method_scope = _toy_method_scope(sorted(estimates))
    for method_id, method_estimates in estimates.items():
        if any(estimate.status is not EstimateStatus.PASS for estimate in method_estimates.values()):
            method_scope[method_id]["status"] = "TOY_MVP_FAILED"
    result = {
        "schema": "policy-learnware.toy-run.v2",
        "status": "TOY_MVP_PASS"
        if all(
            estimate.status is EstimateStatus.PASS
            for method_estimates in estimates.values()
            for estimate in method_estimates.values()
        )
        else "TOY_MVP_FAILED",
        "synthetic_fixture_only": True,
        "seed": seed,
        "implementation_commit": implementation_commit,
        "implementation": implementation,
        "config": config,
        "config_sha256": config_digest,
        "reproducibility_sha256": _payload_digest(reproducibility_payload),
        "context_id": TOY_CONTEXT,
        "gamma": TOY_GAMMA,
        "horizon": TOY_HORIZON,
        "candidate_count": len(candidates),
        "transition_count": len(batch),
        "method_scope": method_scope,
        "estimates": stable_estimates,
        "raw_scores": raw_scores,
        "oracle_values_after_seal": oracle_values,
        "oracle_manifest_sha256": expected_oracle_digest,
        "metrics": metrics,
        "artifacts": {
            "metrics_json": "metrics.json",
            "metrics_json_sha256": metric_paths["json_sha256"],
            "metrics_csv": "metrics.csv",
            "metrics_csv_sha256": metric_paths["csv_sha256"],
            "runtime": "runtime.json",
            "raw_query": "raw_query.reward_free.json",
            "raw_query_sha256": query_artifact_digest,
            "raw_membership_sha256": raw_membership_digest,
            "raw_response": "raw_fixture.response.json",
            "raw_response_sha256": sha256_file(raw_fixture_path),
            "synthetic_oracle_manifest": "oracle_manifest.synthetic.json",
        },
        "ranking_seals": seals,
        "real_asset_training_started": False,
        "production_raw_status": "NO_GO_RAW_OPERATOR_AUTHORITY",
    }
    _write_json(destination / "run.json", result)
    return result


def build_real_preflight(*, implementation_commit: str | None = None) -> dict[str, Any]:
    """Return a stable frozen-fact gate without claiming a live census join."""

    implementation = _implementation_identity(implementation_commit)
    implementation_commit = implementation["commit"]
    config = {
        "schema": "policy-learnware.real-preflight-config.v1",
        "primary_value_convention": "J_gamma=0.99_H=1000_raw",
        "frozen_v03_commit": FROZEN_V03_COMMIT,
        "frozen_v03_tree": FROZEN_V03_TREE,
        "plan_sha256": V04B_PLAN_SHA256,
        "asset_mode": "READ_ONLY",
    }
    required_gates = {
        "actor_authority": {
            "status": "NO_GO",
            "code": "NO_GO_ACTOR_AUTHORITY",
            "reason": "no executable actor/repository authority checker is connected",
        },
        "discounted_oracle": {
            "status": "NO_GO",
            "code": "NO_GO_ORACLE_DISCOUNTED_VALUE",
            "reason": "existing oracle exposes episodic returns, not per-step rewards bound to J_0.99,H1000",
        },
        "exact_behavior_density": {
            "status": "NO_GO",
            "code": "NO_GO_EXISTING_LOG_DENSITY",
            "reason": "existing clipped-Gaussian logs have no verified arbitrary-action exact density",
        },
    }
    return {
        "schema": "policy-learnware.real-preflight.v1",
        "status": "NO_GO",
        "seed": None,
        "implementation_commit": implementation_commit,
        "implementation": implementation,
        "config": config,
        "config_sha256": _payload_digest(config),
        "required_gates": required_gates,
        "raw_adapter": {
            "status": "NO_GO",
            "code": "NO_GO_RAW_OPERATOR_AUTHORITY",
            "reason": "no digest-locked Raw-Delta/RKME export from frozen v03 is connected",
        },
        "method_readiness": {
            "FH_FQE_G099_H1000": "NO_GO_ACTOR_AUTHORITY",
            "FH_KMIFQE_G099_H1000": "NO_GO_EXISTING_LOG_DENSITY",
            "ETM_MBOPE_G099_H1000": "NO_GO_ACTOR_AUTHORITY",
            "DOPE_STYLE_MB_FF_G099_H1000": "NO_GO_ACTOR_AUTHORITY",
            "AR_MBOPE_G099_H1000": "NO_GO_ACTOR_AUTHORITY",
            "RAW_DELTA_TASK5_PROJECT_ADAPTER": "NO_GO_RAW_OPERATOR_AUTHORITY",
        },
        "production_training_started": False,
        "asset_mutation_started": False,
        "provenance": {
            "evidence_scope": "frozen-fact pre-asset gate",
            "live_census_artifact_joined": False,
            "capabilities_are_not_inferred_from_truthy_manifest_fields": True,
            "official_paper_parity_claimed": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-learnware-ope")
    commands = parser.add_subparsers(dest="command", required=True)
    toy = commands.add_parser("toy", help="run all methods on a synthetic finite-horizon fixture")
    toy.add_argument("--artifacts-root", type=Path)
    toy.add_argument("--output", required=True, type=Path)
    toy.add_argument("--seed", type=int, default=7)
    census = commands.add_parser("census", help="perform a read-only real-asset adequacy census")
    census.add_argument("--artifacts-root", type=Path)
    census.add_argument("--dataset", required=True, type=Path)
    census.add_argument("--oracle", type=Path)
    census.add_argument("--density-manifest", type=Path)
    census.add_argument("--actor-authority", type=Path)
    census.add_argument("--expected-dataset-digest")
    census.add_argument("--expected-oracle-digest")
    census.add_argument("--expected-density-manifest-digest")
    census.add_argument("--expected-actor-authority-digest")
    census.add_argument("--horizon", type=int, default=1000)
    census.add_argument("--output", type=Path)
    preflight = commands.add_parser(
        "real-preflight",
        help="emit the stable fail-closed production readiness decision",
    )
    preflight.add_argument("--artifacts-root", type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "toy":
        output = _resolve_artifact_path(args.output, artifacts_root=args.artifacts_root)
        result = run_toy(output, seed=args.seed)
        run_path = output / "run.json"
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "run": str(run_path),
                    "run_sha256": sha256_file(run_path),
                    "methods": sorted(result["method_scope"]),
                    "ranking_seal_sha256": {
                        method_id: seal_ref["sha256"]
                        for method_id, seal_ref in sorted(result["ranking_seals"].items())
                    },
                    "oracle_manifest_sha256": result["oracle_manifest_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "TOY_MVP_PASS" else 1
    if args.command == "real-preflight":
        report = build_real_preflight()
        output_path = _guard_output_location(
            _resolve_artifact_path(args.output, artifacts_root=args.artifacts_root)
        )
        _write_canonical_json(output_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "artifact": str(output_path),
                    "sha256": sha256_file(output_path),
                },
                sort_keys=True,
            )
        )
        return 2
    dataset = _resolve_artifact_path(args.dataset, artifacts_root=args.artifacts_root)
    oracle = (
        _resolve_artifact_path(args.oracle, artifacts_root=args.artifacts_root)
        if args.oracle is not None
        else None
    )
    density_manifest = (
        _resolve_artifact_path(args.density_manifest, artifacts_root=args.artifacts_root)
        if args.density_manifest is not None
        else None
    )
    actor_authority = (
        _resolve_artifact_path(args.actor_authority, artifacts_root=args.artifacts_root)
        if args.actor_authority is not None
        else None
    )
    report = census_real_assets(
        dataset_path=dataset,
        oracle_path=oracle,
        density_manifest_path=density_manifest,
        actor_authority_path=actor_authority,
        horizon=args.horizon,
        expected_dataset_digest=args.expected_dataset_digest,
        expected_oracle_digest=args.expected_oracle_digest,
        expected_density_manifest_digest=args.expected_density_manifest_digest,
        expected_actor_authority_digest=args.expected_actor_authority_digest,
    )
    implementation = _implementation_identity()
    census_config = {
        "schema": "policy-learnware.real-asset-census-config.v1",
        "horizon": args.horizon,
        "expected_dataset_sha256": args.expected_dataset_digest,
        "expected_oracle_sha256": args.expected_oracle_digest,
        "expected_density_manifest_sha256": args.expected_density_manifest_digest,
        "expected_actor_authority_sha256": args.expected_actor_authority_digest,
        "oracle_supplied": oracle is not None,
        "density_manifest_supplied": density_manifest is not None,
        "actor_authority_supplied": actor_authority is not None,
        "asset_mode": "READ_ONLY",
    }
    report.update(
        {
            "seed": None,
            "implementation_commit": implementation["commit"],
            "implementation": implementation,
            "config": census_config,
            "config_sha256": _payload_digest(census_config),
            "provenance": {
                "input_paths_recorded": False,
                "asset_mode": "READ_ONLY",
                "plan_sha256": V04B_PLAN_SHA256,
                "capabilities_require_executable_checkers": True,
            },
        }
    )
    if args.output:
        output = _resolve_artifact_path(args.output, artifacts_root=args.artifacts_root)
        _write_canonical_json(_guard_output_location(output), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
