"""Single command-line runner for synthetic acceptance and read-only census."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import tempfile
import tomllib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .adapters import (
    FROZEN_V03_COMMIT,
    FROZEN_V03_TREE,
    GateClosed,
    RAW_FIXTURE_METHOD_ID,
    RAW_PROJECT_METHOD_ID,
    RAW_QUERY_SCHEMA,
    RAW_REQUEST_SCHEMA,
    RAW_RESPONSE_SCHEMA,
    RAW_SCORE_SEMANTICS,
    RawDeltaTask5Adapter,
    SealedRawOperator,
    census_real_assets,
    execute_frozen_raw_query,
    sha256_file,
)
from .benchmark import (
    ORACLE_MANIFEST_SCHEMA,
    candidate_set_digest,
    export_metrics,
    join_oracle_and_score,
    load_ranking_seal,
    oracle_manifest_digest,
    seal_ranking,
)
from .core import (
    DataValidationError,
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
from .real import (
    ActorAuthority,
    FrozenFPOActor,
    export_existing_log,
    export_reward_free_query,
    load_export,
)


TOY_CONTEXT = "synthetic_linear_native_time_v1"
TOY_HORIZON = 5
TOY_GAMMA = 0.99
TOY_VALUE_CONVENTION = finite_horizon_value_convention(TOY_GAMMA, TOY_HORIZON)
V04B_PLAN_SHA256 = "5fb35cc2ee4c27afd411f77f0c2813088b6d6ab901f8910f442ed5b231e1719e"
REAL_SMOKE_CONFIG_SCHEMA = "policy-learnware.ope.real-smoke-config.v1"
REAL_SMOKE_STAGE_SCHEMA = "policy-learnware.ope.real-smoke-stage.v1"
REAL_SMOKE_RUN_SCHEMA = "policy-learnware.ope.real-smoke-run.v1"


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


def _common_random_keys(
    *,
    seed: int,
    method_id: str,
    context_id: str,
    phase: str,
    rows: int,
) -> np.ndarray:
    """Derive one candidate-independent CRN panel from frozen identities."""

    root = int.from_bytes(
        sha256(
            _canonical_bytes(
                {
                    "schema": "policy-learnware.common-random-key.v1",
                    "seed": seed,
                    "method_id": method_id,
                    "context_id": context_id,
                    "phase": phase,
                }
            )
        ).digest()[:8],
        "big",
    )
    return np.arange(rows, dtype=np.uint64) ^ np.uint64(root)


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
                        "git",
                        f"--git-dir={candidate}",
                        "config",
                        "--bool",
                        "--get",
                        "core.bare",
                    ],
                    check=False,
                    text=True,
                    capture_output=True,
                )
            except OSError:
                return candidate.resolve()
            if bare.returncode != 0:
                return candidate.resolve()
            if bare.returncode == 0 and bare.stdout.strip() == "true":
                return candidate.resolve()
    return None


def _verified_source_checkout() -> Path | None:
    """Return this package's Git root only for the canonical src layout."""

    current_file = Path(__file__).resolve()
    candidate = current_file.parents[2]
    expected_cli = candidate / "src" / "policy_learnware_ope" / "cli.py"
    pyproject = candidate / "pyproject.toml"
    try:
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project_name = metadata["project"]["name"]
        same_cli = expected_cli.samefile(current_file)
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None
    if not (candidate / ".git").exists() or project_name != "policy-learnware-ope":
        return None
    return candidate if same_cli else None


def _installed_package_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    try:
        package_version = distribution_version("policy-learnware-ope")
        status = "INSTALLED_IMMUTABLE_CONTENT"
    except PackageNotFoundError:
        package_version = "UNAVAILABLE"
        status = "UNVERIFIED_PACKAGE_LAYOUT"
    files = sorted(
        path for path in package_root.rglob("*.py") if path.is_file()
    )
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


def _guard_output_location(path: str | Path) -> Path:
    """Reject writes into any Git repository other than this companion."""

    destination = Path(path).resolve()
    containing_repo = _containing_git_root(destination)
    companion_repo = _verified_source_checkout()
    if containing_repo is not None and containing_repo != companion_repo:
        raise PermissionError(
            f"refusing to write into a different Git repository: {containing_repo}"
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
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        worktree = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
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


def _toy_oracle(
    candidates: Sequence[_ToyActor], initial_states: np.ndarray, *, seed: int
) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate in candidates:
        totals = np.zeros(len(initial_states), dtype=float)
        state = initial_states.copy()
        discount = 1.0
        for native_t in range(TOY_HORIZON):
            keys = _common_random_keys(
                seed=seed,
                method_id="TOY_ORACLE_AFTER_SEAL",
                context_id=TOY_CONTEXT,
                phase=f"oracle_rollout_t{native_t}",
                rows=len(state),
            )
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
    # Preserve deterministic work counts (iterations, transitions, actor
    # queries, linear solves, and similar) while keeping wall-clock timing out
    # of the reproducibility payload and ranking seal.
    stable_cost = dict(payload.get("cost", {}))
    stable_cost["timing_artifact"] = "runtime.json"
    payload["cost"] = stable_cost
    return payload


def _real_smoke_exact_keys(
    value: Any, expected: set[str], where: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GateClosed(
            "NO_GO_REAL_SMOKE_CONFIG",
            f"{where} fields differ from the real-smoke schema",
        )
    return value


def _real_smoke_digest(value: Any, where: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", f"{where} must be SHA-256")
    return text


def _read_real_smoke_config(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    expected_sha256 = _real_smoke_digest(expected_sha256, "config digest")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "config is unreadable") from exc
    if sha256(raw).hexdigest() != expected_sha256:
        raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "config digest mismatch")
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "config is not JSON") from exc
    config = dict(
        _real_smoke_exact_keys(
            config,
            {"schema", "protocol", "dataset", "actors", "raw", "fqe", "mbff"},
            "config",
        )
    )
    if config["schema"] != REAL_SMOKE_CONFIG_SCHEMA:
        raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "config schema differs")
    protocol = _real_smoke_exact_keys(
        config["protocol"],
        {
            "context_id",
            "task_id",
            "seed",
            "gamma",
            "horizon",
            "budget",
            "split_seed",
            "track",
        },
        "protocol",
    )
    if (
        not isinstance(protocol["context_id"], str)
        or not protocol["context_id"]
        or not isinstance(protocol["task_id"], str)
        or not protocol["task_id"]
        or isinstance(protocol["seed"], bool)
        or protocol["seed"] != 1
        or protocol["gamma"] != 0.99
        or protocol["horizon"] != 1000
        or protocol["budget"] != 24
        or protocol["split_seed"] != 40401
        or protocol["track"] != "development"
    ):
        raise GateClosed(
            "NO_GO_REAL_SMOKE_CONFIG", "protocol differs from the frozen B24 smoke"
        )
    dataset = _real_smoke_exact_keys(
        config["dataset"], {"p0_census_path", "p0_census_sha256", "bank_path"}, "dataset"
    )
    _real_smoke_digest(dataset["p0_census_sha256"], "P0 census digest")
    actors = _real_smoke_exact_keys(
        config["actors"],
        {"fpo_checkout", "policy_repo_checkout", "candidates"},
        "actors",
    )
    candidates = actors["candidates"]
    if not isinstance(candidates, Mapping) or len(candidates) != 5:
        raise GateClosed(
            "NO_GO_REAL_SMOKE_CONFIG", "actors must contain exactly five candidates"
        )
    for candidate_id, record in candidates.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "candidate ID is invalid")
        record = _real_smoke_exact_keys(
            record,
            {"authority_path", "authority_sha256", "bundle_dir"},
            f"actor {candidate_id}",
        )
        _real_smoke_digest(record["authority_sha256"], f"actor {candidate_id} authority")
    raw_config = _real_smoke_exact_keys(
        config["raw"],
        {
            "authority_path",
            "authority_sha256",
            "repo_root",
            "raw_view_root",
            "asset_census_path",
            "raw_adapter_path",
            "block_size",
        },
        "raw",
    )
    _real_smoke_digest(raw_config["authority_sha256"], "Raw authority digest")
    if raw_config["block_size"] != 2048:
        raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "Raw block_size is frozen at 2048")
    _real_smoke_exact_keys(
        config["fqe"],
        {"ridge", "max_iterations", "tolerance", "stochastic_action_samples"},
        "fqe",
    )
    mbff = _real_smoke_exact_keys(
        config["mbff"],
        {
            "ridge",
            "rollouts_per_initial",
            "ensemble_members",
            "hidden_dim",
            "termination_mode",
        },
        "mbff",
    )
    if mbff["termination_mode"] != "horizon_only":
        raise GateClosed(
            "NO_GO_REAL_SMOKE_CONFIG", "smoke MB-FF requires horizon_only termination"
        )
    return config, expected_sha256


def _real_smoke_p0_identity(
    config: Mapping[str, Any]
) -> tuple[list[str], str, dict[str, Any]]:
    dataset = config["dataset"]
    census_path = Path(dataset["p0_census_path"])
    expected = dataset["p0_census_sha256"]
    if sha256_file(census_path) != expected:
        raise GateClosed("NO_GO_ASSET_ABI", "P0 census digest mismatch")
    try:
        census = json.loads(census_path.read_text(encoding="utf-8"))
        protocol = config["protocol"]
        candidate_set = census["freeze"]["candidate_sets"][protocol["task_id"]]
        candidate_ids = candidate_set["candidate_ids"]
        membership = census["freeze"]["memberships"][protocol["context_id"]]
        smoke = census["freeze"]["smoke"]
        smoke_context = smoke["contexts"][protocol["task_id"]]
        bank_rows = census["asset_facts"]["banks"]["full_rows"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GateClosed("NO_GO_ASSET_ABI", "P0 census lacks smoke identity") from exc
    if (
        census.get("schema") != "policy-learnware.ope.p0-live-census.v1"
        or not isinstance(candidate_ids, list)
        or len(candidate_ids) != 5
        or candidate_ids != sorted(set(candidate_ids))
        or set(candidate_ids) != set(config["actors"]["candidates"])
        or membership.get("task_id") != protocol["task_id"]
    ):
        raise GateClosed("NO_GO_ASSET_ABI", "P0 TASK_5/context binding differs")
    matches = [
        row
        for row in bank_rows
        if row.get("context_id") == protocol["context_id"]
    ]
    if (
        len(matches) != 1
        or matches[0].get("task_id") != protocol["task_id"]
        or matches[0].get("role") != "development_query"
        or matches[0].get("status") != "PASS"
    ):
        raise GateClosed("NO_GO_ASSET_ABI", "P0 bank/context binding differs")
    bank = matches[0]
    if (
        smoke.get("budget_episodes") != 24
        or smoke.get("seed") != 1
        or smoke.get("status") != "NO_GO_PRE_ORACLE_SMOKE_BRIDGES_MISSING"
        or smoke_context.get("context_id") != protocol["context_id"]
        or smoke_context.get("dataset_digest") != bank.get("dataset_digest")
        or smoke_context.get("candidate_membership_digest")
        != candidate_set.get("membership_digest")
        or smoke_context.get("fit_membership_digest")
        != membership.get("fit_membership_digest")
        or smoke_context.get("validation_membership_digest")
        != membership.get("validation_membership_digest")
        or smoke_context.get("s0_membership_digest")
        != membership.get("s0_membership_digest")
        or set(smoke_context.get("methods", ()))
        != {
            "RAW_DELTA_TASK5",
            "FH_FQE_G099_H1000",
            "DOPE_STYLE_MB_FF_G099_H1000",
        }
    ):
        raise GateClosed("NO_GO_ASSET_ABI", "P0 frozen smoke cell differs")
    return list(candidate_ids), str(matches[0]["bank_sha256"]), dict(membership)


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _real_smoke_stage(
    root: Path,
    name: str,
    *,
    config_sha256: str,
    resume: bool,
    build: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    destination = root / name
    if destination.exists():
        if not resume:
            raise FileExistsError(f"real-smoke stage already exists: {destination}")
        try:
            stage = json.loads((destination / "stage.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateClosed("NO_GO_ASSET_ABI", f"{name} stage is unreadable") from exc
        if (
            not isinstance(stage, Mapping)
            or stage.get("schema") != REAL_SMOKE_STAGE_SCHEMA
            or stage.get("stage") != name
            or stage.get("config_sha256") != config_sha256
            or not isinstance(stage.get("artifacts"), Mapping)
        ):
            raise GateClosed("NO_GO_ASSET_ABI", f"{name} stage identity mismatch")
        for relative, digest in stage["artifacts"].items():
            artifact = destination / str(relative)
            if (
                Path(str(relative)).is_absolute()
                or ".." in Path(str(relative)).parts
                or artifact.is_symlink()
                or not artifact.is_file()
                or sha256_file(artifact) != digest
            ):
                raise GateClosed("NO_GO_ASSET_ABI", f"{name} stage artifact mismatch")
        return dict(stage)

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.partial-", dir=root))
    payload = build(temporary)
    if "artifacts" not in payload or not isinstance(payload["artifacts"], Mapping):
        raise RuntimeError(f"{name} stage builder omitted artifacts")
    stage = {
        "schema": REAL_SMOKE_STAGE_SCHEMA,
        "stage": name,
        "config_sha256": config_sha256,
        **payload,
    }
    _write_canonical_json(temporary / "stage.json", stage)
    os.rename(temporary, destination)
    return stage


def _raw_subprocess_entry(
    arguments: dict[str, Any], sender: Any
) -> None:  # pragma: no cover - exercised through the parent boundary
    try:
        sys.dont_write_bytecode = True
        sender.send({"ok": True, "response": execute_frozen_raw_query(**arguments)})
    except Exception as exc:
        sender.send(
            {
                "ok": False,
                "status": getattr(exc, "status", "NO_GO_RAW_PARITY"),
                "detail": str(exc),
                "exception_type": type(exc).__name__,
            }
        )
    finally:
        sender.close()


def _execute_raw_isolated(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen Raw checkout in a clean interpreter, away from actor imports."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_raw_subprocess_entry, args=(arguments, sender))
    process.start()
    sender.close()
    process.join(timeout=1800)
    if process.is_alive():
        process.terminate()
        process.join()
        raise GateClosed("NO_GO_RAW_PARITY", "isolated Raw execution timed out")
    try:
        result = receiver.recv()
    except EOFError as exc:
        raise GateClosed(
            "NO_GO_RAW_PARITY",
            f"isolated Raw execution exited without evidence (rc={process.exitcode})",
        ) from exc
    finally:
        receiver.close()
    if process.exitcode != 0 or not isinstance(result, Mapping) or not result.get("ok"):
        raise GateClosed(
            str(result.get("status", "NO_GO_RAW_PARITY"))
            if isinstance(result, Mapping)
            else "NO_GO_RAW_PARITY",
            str(result.get("detail", "isolated Raw execution failed"))
            if isinstance(result, Mapping)
            else "isolated Raw execution failed",
        )
    response = result.get("response")
    if not isinstance(response, Mapping):
        raise GateClosed("NO_GO_RAW_PARITY", "isolated Raw response is malformed")
    return dict(response)


def run_real_smoke(
    config_path: str | Path,
    output: str | Path,
    *,
    expected_config_sha256: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Raw + FQE + MB-FF and stop after oracle-blind ranking seals."""

    config, config_sha256 = _read_real_smoke_config(
        config_path, expected_config_sha256
    )
    candidate_ids, bank_sha256, frozen_membership = _real_smoke_p0_identity(config)
    protocol = config["protocol"]
    actors_config = config["actors"]
    raw_config = config["raw"]
    implementation = _implementation_identity()
    if implementation["worktree_status"] not in {
        "CLEAN",
        "INSTALLED_IMMUTABLE_CONTENT",
    }:
        raise GateClosed(
            "NO_GO_IMPLEMENTATION_IDENTITY",
            "real smoke requires a clean checkout or immutable installed package",
        )
    normalized_config = {
        "schema": "policy-learnware.ope.real-smoke-path-free-config.v1",
        "protocol": dict(protocol),
        "protocol_correction": {
            "planning_text_semantics": "DETERMINISTIC_DEPLOYMENT",
            "frozen_asset_semantics": "PER_STEP_STOCHASTIC_KEYED",
            "deterministic_true_effect": "FEATHER_NOISE_DISABLED_ACTION_STILL_PRNG_SAMPLED",
            "fqe_mb_policy_expectation": "COMMON_RANDOM_KEYED_MONTE_CARLO",
            "kmifqe_existing_fpo_status": "NO_GO_TARGET_POLICY_SEMANTICS",
        },
        "dataset": {
            "p0_census_sha256": config["dataset"]["p0_census_sha256"],
            "bank_sha256": bank_sha256,
            "membership_protocol": "ope-existing-log-membership-v1",
        },
        "candidate_ids": candidate_ids,
        "candidate_set_sha256": candidate_set_digest(candidate_ids),
        "actor_authority_sha256": {
            candidate_id: actors_config["candidates"][candidate_id][
                "authority_sha256"
            ]
            for candidate_id in candidate_ids
        },
        "raw": {
            "authority_sha256": raw_config["authority_sha256"],
            "block_size": raw_config["block_size"],
            "score_semantics": RAW_SCORE_SEMANTICS,
        },
        "fqe": dict(config["fqe"]),
        "mbff": dict(config["mbff"]),
    }
    normalized_config_sha256 = _payload_digest(normalized_config)
    destination = _guard_output_location(output)
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"real-smoke output is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(
            f"real-smoke output is non-empty; pass --resume to verify it: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema": "policy-learnware.ope.real-smoke-config-lock.v1",
        "input_config_sha256": config_sha256,
        "path_free_config_sha256": normalized_config_sha256,
        "path_free_config": normalized_config,
        "implementation": implementation,
    }
    lock_path = destination / "config.lock.json"
    if lock_path.exists():
        if lock_path.read_bytes() != _canonical_bytes(lock):
            raise GateClosed("NO_GO_REAL_SMOKE_CONFIG", "resume config lock differs")
    else:
        _write_canonical_json(lock_path, lock)

    final_path = destination / "run.json"
    if final_path.exists():
        if not resume:
            raise FileExistsError(f"real-smoke run already exists: {final_path}")
        try:
            final = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateClosed("NO_GO_ASSET_ABI", "existing run artifact is unreadable") from exc
        if (
            final.get("schema") != REAL_SMOKE_RUN_SCHEMA
            or final.get("config_sha256") != config_sha256
            or final.get("path_free_config_sha256") != normalized_config_sha256
        ):
            raise GateClosed("NO_GO_ASSET_ABI", "existing run identity mismatch")
        stage_references = final.get("stages")
        if not isinstance(stage_references, Mapping) or set(stage_references) != {
            "data",
            "raw",
            "fqe",
            "mbff",
        }:
            raise GateClosed("NO_GO_ASSET_ABI", "existing run stage inventory differs")
        verified_stages: dict[str, dict[str, Any]] = {}
        for stage_name in ("data", "raw", "fqe", "mbff"):
            if not (destination / stage_name).is_dir():
                raise GateClosed("NO_GO_ASSET_ABI", f"{stage_name} stage is absent")
            verified_stages[stage_name] = _real_smoke_stage(
                destination,
                stage_name,
                config_sha256=config_sha256,
                resume=True,
                build=lambda _path: {},
            )
            reference = stage_references[stage_name]
            expected_path = f"{stage_name}/stage.json"
            if (
                not isinstance(reference, Mapping)
                or reference.get("path") != expected_path
                or sha256_file(destination / expected_path) != reference.get("sha256")
            ):
                raise GateClosed("NO_GO_ASSET_ABI", f"{stage_name} stage seal differs")
        expected_rankings: dict[str, list[str]] = {}
        expected_seal_references: dict[str, dict[str, str]] = {}
        for stage_name in ("raw", "fqe", "mbff"):
            stage = verified_stages[stage_name]
            method_id = stage.get("method_id")
            seal_digest = stage.get("seal_sha256")
            seal_relative = f"{stage_name}/ranking.seal.json"
            if (
                not isinstance(method_id, str)
                or not method_id
                or stage.get("artifacts", {}).get("ranking.seal.json")
                != seal_digest
            ):
                raise GateClosed(
                    "NO_GO_ASSET_ABI", f"{stage_name} ranking authority differs"
                )
            try:
                sealed = load_ranking_seal(
                    destination / seal_relative,
                    expected_seal_digest=seal_digest,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise GateClosed(
                    "NO_GO_ASSET_ABI", f"{stage_name} ranking seal is invalid"
                ) from exc
            sealed_candidate_ids = [
                row["candidate_id"] for row in sealed.payload["rows"]
            ]
            if (
                sealed.payload["method_id"] != method_id
                or sealed.payload["context_id"] != protocol["context_id"]
                or sealed_candidate_ids != candidate_ids
                or sealed.payload["candidate_set_digest"]
                != candidate_set_digest(candidate_ids)
                or sealed.payload["ranking"] != stage.get("ranking")
            ):
                raise GateClosed(
                    "NO_GO_ASSET_ABI", f"{stage_name} ranking binding differs"
                )
            expected_rankings[method_id] = list(sealed.payload["ranking"])
            expected_seal_references[method_id] = {
                "path": seal_relative,
                "sha256": seal_digest,
            }
        if len(expected_rankings) != 3:
            raise GateClosed("NO_GO_ASSET_ABI", "ranking method identities collide")
        expected_all_pass = bool(
            verified_stages["raw"].get("status") == "PASS"
            and verified_stages["fqe"].get("all_candidates_pass") is True
            and verified_stages["mbff"].get("all_candidates_pass") is True
        )
        expected_data_summary = {
            "export_manifest_sha256": verified_stages["data"][
                "export_manifest_sha256"
            ],
            "fit_membership_sha256": verified_stages["data"][
                "fit_membership_sha256"
            ],
            "validation_membership_sha256": verified_stages["data"][
                "validation_membership_sha256"
            ],
            "s0_membership_sha256": verified_stages["data"][
                "s0_membership_sha256"
            ],
            "query_sha256": verified_stages["data"]["query_sha256"],
        }
        if (
            final.get("status")
            != ("SEALED_PRE_ORACLE" if expected_all_pass else "INCOMPLETE_PRE_ORACLE")
            or final.get("metrics_status")
            != ("WAITING_ORACLE" if expected_all_pass else "NOT_READY")
            or final.get("context_id") != protocol["context_id"]
            or final.get("task_id") != protocol["task_id"]
            or final.get("seed") != protocol["seed"]
            or final.get("gamma") != protocol["gamma"]
            or final.get("horizon") != protocol["horizon"]
            or final.get("candidate_ids") != candidate_ids
            or final.get("data") != expected_data_summary
            or final.get("rankings") != expected_rankings
            or final.get("ranking_seals") != expected_seal_references
        ):
            raise GateClosed("NO_GO_ASSET_ABI", "existing run summary differs")
        runtime_reference = final.get("runtime")
        runtime_path = destination / "runtime.json"
        if (
            not isinstance(runtime_reference, Mapping)
            or runtime_reference.get("path") != "runtime.json"
            or runtime_path.is_symlink()
            or not runtime_path.is_file()
            or sha256_file(runtime_path) != runtime_reference.get("sha256")
        ):
            raise GateClosed("NO_GO_ASSET_ABI", "existing run runtime differs")
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateClosed("NO_GO_ASSET_ABI", "existing runtime is unreadable") from exc
        expected_runtime_artifacts = {
            name: stage["artifacts"]["runtime.json"]
            for name, stage in verified_stages.items()
        }
        if (
            runtime.get("schema") != "policy-learnware.ope.real-smoke-runtime.v1"
            or runtime.get("stage") != "run"
            or runtime.get("config_sha256") != config_sha256
            or runtime.get("path_free_config_sha256") != normalized_config_sha256
            or runtime.get("stage_runtime_artifacts") != expected_runtime_artifacts
            or final.get("shared_setup", {}).get("actor_count")
            != runtime.get("actor_count")
            or final.get("shared_setup", {}).get("actor_setup_wall_seconds")
            != runtime.get("actor_setup_wall_seconds")
            or final.get("shared_setup", {}).get("actor_setup_peak_rss_bytes")
            != runtime.get("actor_setup_peak_rss_bytes")
        ):
            raise GateClosed("NO_GO_ASSET_ABI", "existing runtime identity differs")
        for candidate_id in candidate_ids:
            record = actors_config["candidates"][candidate_id]
            authority = ActorAuthority.from_json(
                record["authority_path"],
                expected_sha256=record["authority_sha256"],
                census_path=config["dataset"]["p0_census_path"],
                expected_census_sha256=config["dataset"]["p0_census_sha256"],
                context_id=protocol["context_id"],
                candidate_id=candidate_id,
            )
            evidence = final.get("actor_evidence", {}).get(candidate_id, {})
            if (
                evidence.get("authority_sha256") != authority.authority_sha256
                or evidence.get("bundle_digest") != authority.bundle_digest
            ):
                raise GateClosed("NO_GO_ACTOR_AUTHORITY", "existing actor evidence differs")
        return final

    invocation_started = perf_counter()

    def build_data(stage_dir: Path) -> dict[str, Any]:
        started = perf_counter()
        exported = export_existing_log(
            config["dataset"]["p0_census_path"],
            expected_census_sha256=config["dataset"]["p0_census_sha256"],
            context_id=protocol["context_id"],
            bank_path=config["dataset"]["bank_path"],
            output_dir=stage_dir / "transitions",
        )
        manifest = exported["manifest"]
        if (
            manifest["context"]["context_id"] != protocol["context_id"]
            or manifest["context"]["task_id"] != protocol["task_id"]
            or manifest["source"]["bank_sha256"] != bank_sha256
            or manifest["membership_protocol"]["fit_episode_count"] != 24
            or manifest["membership_protocol"]["fit_rows_per_episode"] != 64
            or manifest["membership_protocol"]["split_seed"] != 40401
        ):
            raise GateClosed("NO_GO_ASSET_ABI", "exported smoke identity differs")
        query = export_reward_free_query(
            stage_dir / "transitions",
            expected_manifest_sha256=exported["manifest_sha256"],
            output_path=stage_dir / "raw-query.npz",
        )
        runtime = {
            "schema": "policy-learnware.ope.real-smoke-runtime.v1",
            "stage": "data",
            "wall_seconds": perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        _write_canonical_json(stage_dir / "runtime.json", runtime)
        return {
            "artifacts": {
                "raw-query.npz": query["artifact_sha256"],
                "runtime.json": sha256_file(stage_dir / "runtime.json"),
                **{
                    f"transitions/{split}.npz": manifest["splits"][split][
                        "file_sha256"
                    ]
                    for split in ("fit", "validation", "s0")
                },
                "transitions/manifest.json": exported["manifest_sha256"],
            },
            "bank_sha256": bank_sha256,
            "export_manifest_sha256": exported["manifest_sha256"],
            "fit_membership_sha256": manifest["splits"]["fit"][
                "membership_digest"
            ],
            "validation_membership_sha256": manifest["splits"]["validation"][
                "membership_digest"
            ],
            "s0_membership_sha256": manifest["splits"]["s0"][
                "membership_digest"
            ],
            "query_sha256": query["artifact_sha256"],
            "transition_count": query["transition_count"],
        }

    data_stage = _real_smoke_stage(
        destination,
        "data",
        config_sha256=config_sha256,
        resume=resume,
        build=build_data,
    )
    if (
        data_stage["fit_membership_sha256"]
        != frozen_membership.get("fit_membership_digest")
        or data_stage["validation_membership_sha256"]
        != frozen_membership.get("validation_membership_digest")
        or data_stage["s0_membership_sha256"]
        != frozen_membership.get("s0_membership_digest")
    ):
        raise GateClosed("NO_GO_ASSET_ABI", "P0/export memberships differ")

    value_convention = finite_horizon_value_convention(
        protocol["gamma"], protocol["horizon"]
    )
    candidate_tasks = {candidate_id: protocol["task_id"] for candidate_id in candidate_ids}
    common_provenance = {
        "seed": protocol["seed"],
        "implementation_commit": implementation["commit"],
        "implementation_tree": implementation["tree"],
        "implementation_worktree_status": implementation["worktree_status"],
        "config_sha256": config_sha256,
        "path_free_config_sha256": normalized_config_sha256,
        "p0_census_sha256": config["dataset"]["p0_census_sha256"],
        "bank_sha256": bank_sha256,
        "export_manifest_sha256": data_stage["export_manifest_sha256"],
        "fit_membership_sha256": data_stage["fit_membership_sha256"],
        "candidate_set_sha256": candidate_set_digest(candidate_ids),
        "actor_authority_sha256": normalized_config["actor_authority_sha256"],
        "gamma": protocol["gamma"],
        "horizon": protocol["horizon"],
        "stage": "PRE_JOIN_SEALED",
        "development_only": True,
    }

    def build_raw(stage_dir: Path) -> dict[str, Any]:
        started = perf_counter()
        placeholder = SealedRawOperator(
            stage_dir / "response.json",
            "0" * 64,
            expected_authority_digest=raw_config["authority_sha256"],
        )
        request = RawDeltaTask5Adapter(
            placeholder, method_id=RAW_PROJECT_METHOD_ID
        ).request(
            context_id=protocol["context_id"],
            task_id=protocol["task_id"],
            candidate_tasks=candidate_tasks,
            query_artifact_digest=data_stage["query_sha256"],
            membership_digest=data_stage["fit_membership_sha256"],
        )
        response = _execute_raw_isolated(
            {
                "authority_path": raw_config["authority_path"],
                "expected_authority_sha256": raw_config["authority_sha256"],
                "repo_root": raw_config["repo_root"],
                "raw_view_root": raw_config["raw_view_root"],
                "asset_census_path": raw_config["asset_census_path"],
                "raw_adapter_path": raw_config["raw_adapter_path"],
                "query_path": destination / "data" / "raw-query.npz",
                "expected_query_sha256": data_stage["query_sha256"],
                "request": request,
                "block_size": raw_config["block_size"],
            }
        )
        if response.get("score_semantics") != RAW_SCORE_SEMANTICS:
            raise GateClosed("NO_GO_RAW_PARITY", "isolated Raw semantics differ")
        _write_canonical_json(stage_dir / "response.json", response)
        response_sha256 = sha256_file(stage_dir / "response.json")
        adapter = RawDeltaTask5Adapter(
            SealedRawOperator(
                stage_dir / "response.json",
                response_sha256,
                expected_authority_digest=raw_config["authority_sha256"],
            ),
            method_id=RAW_PROJECT_METHOD_ID,
        )
        scores = adapter.score(
            context_id=protocol["context_id"],
            task_id=protocol["task_id"],
            candidate_tasks=candidate_tasks,
            query_artifact_digest=data_stage["query_sha256"],
            membership_digest=data_stage["fit_membership_sha256"],
        )
        _write_canonical_json(stage_dir / "scores.json", scores)
        seal = seal_ranking(
            stage_dir / "ranking.seal.json",
            method_id=RAW_PROJECT_METHOD_ID,
            context_id=protocol["context_id"],
            score_kind="compatibility",
            scores=scores,
            diagnostics={
                candidate_id: {"score_semantics": RAW_SCORE_SEMANTICS}
                for candidate_id in candidate_ids
            },
            provenance={
                **common_provenance,
                "scientific_role": "PROJECT_RAW_DELTA_TASK5_ADAPTER",
                "query_sha256": data_stage["query_sha256"],
                "response_sha256": response_sha256,
                "raw_authority_sha256": raw_config["authority_sha256"],
                "score_semantics": RAW_SCORE_SEMANTICS,
                "official_paper_parity": False,
            },
            higher_is_better=True,
            value_convention=value_convention,
        )
        runtime = {
            "schema": "policy-learnware.ope.real-smoke-runtime.v1",
            "stage": "raw",
            "wall_seconds": perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        _write_canonical_json(stage_dir / "runtime.json", runtime)
        return {
            "artifacts": {
                "ranking.seal.json": seal.digest,
                "response.json": response_sha256,
                "runtime.json": sha256_file(stage_dir / "runtime.json"),
                "scores.json": sha256_file(stage_dir / "scores.json"),
            },
            "method_id": RAW_PROJECT_METHOD_ID,
            "ranking": list(seal.payload["ranking"]),
            "selected_candidate_id": seal.payload["selected_candidate_id"],
            "seal_sha256": seal.digest,
            "status": "PASS",
        }

    raw_stage = _real_smoke_stage(
        destination,
        "raw",
        config_sha256=config_sha256,
        resume=resume,
        build=build_raw,
    )

    actor_setup_started = perf_counter()
    actors: dict[str, FrozenFPOActor] = {}
    for candidate_id in candidate_ids:
        record = actors_config["candidates"][candidate_id]
        authority = ActorAuthority.from_json(
            record["authority_path"],
            expected_sha256=record["authority_sha256"],
            census_path=config["dataset"]["p0_census_path"],
            expected_census_sha256=config["dataset"]["p0_census_sha256"],
            context_id=protocol["context_id"],
            candidate_id=candidate_id,
        )
        if authority.task_id != protocol["task_id"]:
            raise GateClosed("NO_GO_ACTOR_AUTHORITY", "actor task differs")
        actors[candidate_id] = FrozenFPOActor(
            authority,
            bundle_dir=record["bundle_dir"],
            fpo_checkout=actors_config["fpo_checkout"],
            policy_repo_checkout=actors_config["policy_repo_checkout"],
        )
    actor_setup_wall_seconds = perf_counter() - actor_setup_started
    actor_setup_peak_rss_bytes = _peak_rss_bytes()

    fit_batch = load_export(
        destination / "data" / "transitions",
        "fit",
        expected_manifest_sha256=data_stage["export_manifest_sha256"],
    )
    s0_batch = load_export(
        destination / "data" / "transitions",
        "s0",
        expected_manifest_sha256=data_stage["export_manifest_sha256"],
    )

    def build_value_stage(
        stage_dir: Path,
        *,
        family: str,
    ) -> dict[str, Any]:
        started = perf_counter()
        estimates: dict[str, ValueEstimate] = {}
        if family == "fqe":
            method_id: str | None = None
            for candidate_id in candidate_ids:
                estimator = FiniteHorizonFQE(
                    gamma=protocol["gamma"],
                    horizon=protocol["horizon"],
                    **config["fqe"],
                )
                method_id = estimator.method_id if method_id is None else method_id
                if estimator.method_id != method_id:
                    raise RuntimeError("FQE method identity changed across candidates")
                estimator.fit(
                    fit_batch,
                    actors[candidate_id],
                    fit_keys=_common_random_keys(
                        seed=protocol["seed"],
                        method_id=estimator.method_id,
                        context_id=protocol["context_id"],
                        phase="fit_transition_rows",
                        rows=len(fit_batch),
                    ),
                )
                estimates[candidate_id] = estimator.estimate(
                    s0_batch.observation,
                    initial_timestep=s0_batch.native_timestep,
                    keys=_common_random_keys(
                        seed=protocol["seed"],
                        method_id=estimator.method_id,
                        context_id=protocol["context_id"],
                        phase="estimate_s0_rows",
                        rows=len(s0_batch),
                    ),
                )
            scientific_role = "FINITE_HORIZON_PROTOCOL_ADAPTATION"
        else:
            estimator = make_model_based_estimator(
                DOPE_STYLE_MB_FF_ID,
                gamma=protocol["gamma"],
                horizon=protocol["horizon"],
                **{
                    key: value
                    for key, value in config["mbff"].items()
                    if key != "termination_mode"
                },
            )
            method_id = estimator.method_id
            estimator.fit(
                fit_batch,
                actors[candidate_ids[0]],
                fit_keys=_common_random_keys(
                    seed=protocol["seed"],
                    method_id=method_id,
                    context_id=protocol["context_id"],
                    phase="transition_model_fit",
                    rows=1,
                ),
            )
            for candidate_id in candidate_ids:
                estimates[candidate_id] = estimator.estimate(
                    s0_batch.observation,
                    initial_timestep=s0_batch.native_timestep,
                    keys=_common_random_keys(
                        seed=protocol["seed"],
                        method_id=method_id,
                        context_id=protocol["context_id"],
                        phase="estimate_s0_rows",
                        rows=len(s0_batch),
                    ),
                    candidate=actors[candidate_id],
                )
            scientific_role = "PROJECT_DEFINED_REFERENCE"
        assert method_id is not None
        stable = {
            candidate_id: _stable_estimate(estimates[candidate_id])
            for candidate_id in candidate_ids
        }
        _write_canonical_json(stage_dir / "estimates.json", stable)
        scores = {
            candidate_id: estimates[candidate_id].value for candidate_id in candidate_ids
        }
        statuses = {
            candidate_id: estimates[candidate_id].status.value
            for candidate_id in candidate_ids
        }
        diagnostics = {
            candidate_id: {
                **stable[candidate_id]["support"],
                **stable[candidate_id]["diagnostics"],
            }
            for candidate_id in candidate_ids
        }
        seal = seal_ranking(
            stage_dir / "ranking.seal.json",
            method_id=method_id,
            context_id=protocol["context_id"],
            score_kind="value",
            scores=scores,
            statuses=statuses,
            diagnostics=diagnostics,
            provenance={
                **common_provenance,
                "scientific_role": scientific_role,
                "s0_membership_sha256": data_stage["s0_membership_sha256"],
                "policy_semantics": "STOCHASTIC_KEYED_MONTE_CARLO_EXPECTATION",
                "candidate_independent_common_random_panel": True,
                "official_paper_parity": False,
            },
            higher_is_better=True,
            value_convention=value_convention,
        )
        runtime = {
            "schema": "policy-learnware.ope.real-smoke-runtime.v1",
            "stage": family,
            "wall_seconds": perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        _write_canonical_json(stage_dir / "runtime.json", runtime)
        all_pass = all(status == EstimateStatus.PASS.value for status in statuses.values())
        return {
            "all_candidates_pass": all_pass,
            "artifacts": {
                "estimates.json": sha256_file(stage_dir / "estimates.json"),
                "ranking.seal.json": seal.digest,
                "runtime.json": sha256_file(stage_dir / "runtime.json"),
            },
            "method_id": method_id,
            "ranking": list(seal.payload["ranking"]),
            "selected_candidate_id": seal.payload["selected_candidate_id"],
            "seal_sha256": seal.digest,
            "statuses": statuses,
        }

    fqe_stage = _real_smoke_stage(
        destination,
        "fqe",
        config_sha256=config_sha256,
        resume=resume,
        build=lambda stage_dir: build_value_stage(stage_dir, family="fqe"),
    )
    mbff_stage = _real_smoke_stage(
        destination,
        "mbff",
        config_sha256=config_sha256,
        resume=resume,
        build=lambda stage_dir: build_value_stage(stage_dir, family="mbff"),
    )

    actor_evidence = {
        candidate_id: {
            "authority_sha256": actors[candidate_id].authority.authority_sha256,
            "bundle_digest": actors[candidate_id].authority.bundle_digest,
            "semantics": actors[candidate_id].semantics.value,
            "initialization_parity_status": actors[candidate_id].parity["status"],
            "same_key_replay_status": actors[candidate_id].parity[
                "same_key_replay"
            ]["status"],
            "different_key_sensitivity_status": actors[candidate_id].parity[
                "different_key_sensitivity"
            ]["status"],
            "final_read_only_verification": dict(actors[candidate_id].verify_unchanged()),
        }
        for candidate_id in candidate_ids
    }
    all_pass = bool(
        raw_stage.get("status") == "PASS"
        and fqe_stage.get("all_candidates_pass") is True
        and mbff_stage.get("all_candidates_pass") is True
    )
    stage_runtime_artifacts = {
        name: stage["artifacts"]["runtime.json"]
        for name, stage in (
            ("data", data_stage),
            ("raw", raw_stage),
            ("fqe", fqe_stage),
            ("mbff", mbff_stage),
        )
    }
    proposed_runtime = {
        "schema": "policy-learnware.ope.real-smoke-runtime.v1",
        "stage": "run",
        "config_sha256": config_sha256,
        "path_free_config_sha256": normalized_config_sha256,
        "invocation_wall_seconds": perf_counter() - invocation_started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "actor_count": len(actors),
        "actor_setup_wall_seconds": actor_setup_wall_seconds,
        "actor_setup_peak_rss_bytes": actor_setup_peak_rss_bytes,
        "stage_runtime_artifacts": stage_runtime_artifacts,
    }
    runtime_path = destination / "runtime.json"
    if runtime_path.exists():
        if not resume or runtime_path.is_symlink() or not runtime_path.is_file():
            raise FileExistsError(f"real-smoke runtime conflict: {runtime_path}")
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateClosed("NO_GO_ASSET_ABI", "resume runtime is unreadable") from exc
        if (
            runtime.get("schema") != proposed_runtime["schema"]
            or runtime.get("stage") != "run"
            or runtime.get("config_sha256") != config_sha256
            or runtime.get("path_free_config_sha256") != normalized_config_sha256
            or runtime.get("actor_count") != len(actors)
            or runtime.get("stage_runtime_artifacts") != stage_runtime_artifacts
            or not isinstance(runtime.get("actor_setup_wall_seconds"), (int, float))
        ):
            raise GateClosed("NO_GO_ASSET_ABI", "resume runtime identity differs")
    else:
        runtime = proposed_runtime
        _write_canonical_json(runtime_path, runtime)
    run = {
        "schema": REAL_SMOKE_RUN_SCHEMA,
        "status": "SEALED_PRE_ORACLE" if all_pass else "INCOMPLETE_PRE_ORACLE",
        "metrics_status": "WAITING_ORACLE" if all_pass else "NOT_READY",
        "development_only": True,
        "oracle_accessed": False,
        "environment_accessed": False,
        "config_sha256": config_sha256,
        "path_free_config_sha256": normalized_config_sha256,
        "implementation": implementation,
        "context_id": protocol["context_id"],
        "task_id": protocol["task_id"],
        "seed": protocol["seed"],
        "gamma": protocol["gamma"],
        "horizon": protocol["horizon"],
        "candidate_ids": candidate_ids,
        "actor_evidence": actor_evidence,
        "shared_setup": {
            "actor_count": runtime["actor_count"],
            "actor_setup_wall_seconds": runtime["actor_setup_wall_seconds"],
            "actor_setup_peak_rss_bytes": runtime["actor_setup_peak_rss_bytes"],
            "runtime_artifact": "runtime.json",
        },
        "protocol_correction": normalized_config["protocol_correction"],
        "stages": {
            name: {
                "path": f"{name}/stage.json",
                "sha256": sha256_file(destination / name / "stage.json"),
            }
            for name in ("data", "raw", "fqe", "mbff")
        },
        "data": {
            "export_manifest_sha256": data_stage["export_manifest_sha256"],
            "fit_membership_sha256": data_stage["fit_membership_sha256"],
            "validation_membership_sha256": data_stage[
                "validation_membership_sha256"
            ],
            "s0_membership_sha256": data_stage["s0_membership_sha256"],
            "query_sha256": data_stage["query_sha256"],
        },
        "ranking_seals": {
            raw_stage["method_id"]: {
                "path": "raw/ranking.seal.json",
                "sha256": raw_stage["seal_sha256"],
            },
            fqe_stage["method_id"]: {
                "path": "fqe/ranking.seal.json",
                "sha256": fqe_stage["seal_sha256"],
            },
            mbff_stage["method_id"]: {
                "path": "mbff/ranking.seal.json",
                "sha256": mbff_stage["seal_sha256"],
            },
        },
        "rankings": {
            raw_stage["method_id"]: raw_stage["ranking"],
            fqe_stage["method_id"]: fqe_stage["ranking"],
            mbff_stage["method_id"]: mbff_stage["ranking"],
        },
        "scientific_status": {
            "FH_KMIFQE_G099_H1000": "NO_GO_EXISTING_LOG_DENSITY_AND_TARGET_POLICY_SEMANTICS",
            "ETM_MBOPE_G099_H1000": "NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT",
            "discounted_value_join": "WAITING_ORACLE",
            "exact_density_panel": "INCOMPLETE_REQUIRED_DENSITY_PANEL",
        },
        "runtime": {
            "path": "runtime.json",
            "sha256": sha256_file(runtime_path),
        },
    }
    _write_canonical_json(final_path, run)
    return run


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
                    "B20 protocol adaptation with nonlinear candidate critic, local "
                    "Hessian metric, estimated bandwidth, replacement resampling, "
                    "and logged-adjacent-action TD"
                ),
                "scientific_role": "B20_PROTOCOL_ADAPTATION",
                "official_paper_parity": False,
                "production_status": "NO_GO_OPS_DS_DENSE_HESSIAN_PANEL",
            }
        elif method_id.startswith("ETM_MBOPE_"):
            scopes[method_id] = {
                "status": "TOY_MVP_PASS",
                "scope": (
                    "B22 protocol adaptation with conditional training-time Langevin "
                    "negatives and an exact RFF gradient-penalty VJP"
                ),
                "scientific_role": "PROJECT_ETM_PROTOCOL_ADAPTATION",
                "official_paper_parity": False,
                "production_status": "NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT",
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
) -> dict[str, Any]:
    destination = _guard_output_location(output)
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
                "epochs": 12,
                "batch_size": 48,
                "learning_rate": 0.01,
                "temperature": 1.0,
                "training_langevin_steps": 5,
                "training_step_size_initial": 0.1,
                "training_step_size_final": 0.001,
                "training_noise_scale": 0.5,
                "training_gradient_clip": 10.0,
                "training_drift_clip": 0.5,
                "training_sample_clip": 1.1,
                "gradient_penalty_margin": 5.0,
                "gradient_penalty_weight": 1.0,
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
    model_common = {
        "ridge": 1e-4,
        "rollouts_per_initial": 12,
        "termination_mode": "horizon_only",
    }
    config: dict[str, Any] = {
        "schema": "policy-learnware.toy-config.v3",
        "seed": seed,
        "context_id": TOY_CONTEXT,
        "gamma": TOY_GAMMA,
        "horizon": TOY_HORIZON,
        "episodes": 48,
        "candidate_count": 5,
        "common_random_key_derivation": (
            "sha256(seed,method_id,context_id,phase)[0:8]_xor_row_or_s0_index"
        ),
        "fqe": {
            "ridge": 1e-7,
            "max_iterations": 2500,
            "tolerance": 1e-8,
        },
        "kmifqe": {
            "ridge": 1e-7,
            "max_iterations": 200,
            "tolerance": 3e-3,
            "critic_features": 32,
            "eigenvalue_floor": 1e-6,
            "metric_regularization": 0.1,
            "bandwidth_floor": 1e-3,
            "bandwidth_ceiling": 10.0,
            "ratio_clip_min": 1e-3,
            "ratio_clip_max": 2.0,
            "target_density_floor": 1e-12,
            "target_update_interval": 1,
            "critic_step_size": 0.1,
            "probability_tolerance": 0.015,
            "min_log_density": -50.0,
            "min_ess_fraction": 0.01,
            "resample_size": None,
        },
        "model_based": {family: kwargs for _, family, kwargs in model_specs},
        "model_common": model_common,
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
        for candidate in candidates:
            estimator_config = (
                config["kmifqe"]
                if estimator_type is FiniteHorizonKMIFQE
                else config["fqe"]
            )
            estimator = estimator_type(
                gamma=TOY_GAMMA,
                horizon=TOY_HORIZON,
                **estimator_config,
            )
            if actual_method_id is None:
                actual_method_id = estimator.method_id
            elif estimator.method_id != actual_method_id:
                raise RuntimeError("one estimator family produced inconsistent method IDs")
            fit_keys = _common_random_keys(
                seed=seed,
                method_id=estimator.method_id,
                context_id=TOY_CONTEXT,
                phase="fit_transition_rows",
                rows=len(batch),
            )
            if estimator_type is FiniteHorizonKMIFQE:
                estimator.fit(batch, candidate, behavior_density=density, fit_keys=fit_keys)
            else:
                estimator.fit(batch, candidate, fit_keys=fit_keys)
            estimate_keys = _common_random_keys(
                seed=seed,
                method_id=estimator.method_id,
                context_id=TOY_CONTEXT,
                phase="estimate_s0_rows",
                rows=len(initial),
            )
            method_estimates[candidate.policy_id] = estimator.estimate(initial, keys=estimate_keys)
        if actual_method_id is None:
            raise RuntimeError("toy fixture has no candidates")
        estimates[actual_method_id] = method_estimates
        runtime_by_method[actual_method_id] = perf_counter() - method_started

    model_fit_actor = min(candidates, key=lambda candidate: candidate.policy_id)
    for selector_id, _family, model_kwargs in model_specs:
        method_started = perf_counter()
        estimator = make_model_based_estimator(
            selector_id,
            gamma=TOY_GAMMA,
            horizon=TOY_HORIZON,
            rollouts_per_initial=model_common["rollouts_per_initial"],
            ridge=model_common["ridge"],
            **model_kwargs,
        )
        estimator.fit(
            batch,
            model_fit_actor,
            fit_keys=_common_random_keys(
                seed=seed,
                method_id=estimator.method_id,
                context_id=TOY_CONTEXT,
                phase="transition_model_fit",
                rows=1,
            ),
        )
        method_estimates = {}
        for candidate in candidates:
            estimate_keys = _common_random_keys(
                seed=seed,
                method_id=estimator.method_id,
                context_id=TOY_CONTEXT,
                phase="estimate_s0_rows",
                rows=len(initial),
            )
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
    ranking_semantics: dict[str, dict[str, Any]] = {}
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
    ranking_semantics[RAW_FIXTURE_METHOD_ID] = {
        "candidate_set_digest": raw_seal.payload["candidate_set_digest"],
        "ranking": list(raw_seal.payload["ranking"]),
        "selected_candidate_id": raw_seal.payload["selected_candidate_id"],
        "statuses": {
            row["candidate_id"]: row["status"] for row in raw_seal.payload["rows"]
        },
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
        ranking_semantics[method_id] = {
            "candidate_set_digest": seal.payload["candidate_set_digest"],
            "ranking": list(seal.payload["ranking"]),
            "selected_candidate_id": seal.payload["selected_candidate_id"],
            "statuses": {
                row["candidate_id"]: row["status"] for row in seal.payload["rows"]
            },
        }

    # Synthetic oracle evaluation is deliberately invoked only after every
    # method score and ranking has been sealed.
    oracle_values = _toy_oracle(candidates, initial, seed=seed)
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
    runtime_digest = sha256_file(destination / "runtime.json")
    stable_estimates = {
        method_id: {
            candidate_id: _stable_estimate(estimate)
            for candidate_id, estimate in method_estimates.items()
        }
        for method_id, method_estimates in estimates.items()
    }
    reproducibility_payload = {
        "schema": "policy-learnware.reproducibility-identity.v2",
        "seed": seed,
        "config_sha256": config_digest,
        "implementation": implementation,
        "dataset_digest": batch.source_digest,
        "candidate_ids": candidate_ids,
        "ranking_semantics": ranking_semantics,
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
            "runtime_sha256": runtime_digest,
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
        "method_blockers": {
            "FH_KMIFQE_G099_H1000": [
                {
                    "status": "NO_GO",
                    "code": "NO_GO_OPS_DS_DENSE_HESSIAN_PANEL",
                    "reason": (
                        "the current per-row dense Hessian/metric materialization is not "
                        "qualified for the million-row OPS-DS panel"
                    ),
                }
            ],
            "ETM_MBOPE_G099_H1000": [
                {
                    "status": "NO_GO",
                    "code": "NO_GO_ETM_INFERENCE_PROTOCOL_ALIGNMENT",
                    "reason": (
                        "the project inference Langevin initialization/noise/clipping/step "
                        "protocol is not aligned with either the B22 paper or official release"
                    ),
                }
            ],
        },
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
    toy.add_argument("--output", required=True, type=Path)
    toy.add_argument("--seed", type=int, default=7)
    census = commands.add_parser("census", help="perform a read-only real-asset adequacy census")
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
    preflight.add_argument("--output", required=True, type=Path)
    real_smoke = commands.add_parser(
        "real-smoke",
        help="run Raw, FQE, and MB-FF through oracle-blind ranking seals",
    )
    real_smoke.add_argument("--config", required=True, type=Path)
    real_smoke.add_argument("--expected-config-sha256", required=True)
    real_smoke.add_argument("--output", required=True, type=Path)
    real_smoke.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "toy":
        result = run_toy(args.output, seed=args.seed)
        run_path = Path(args.output).resolve() / "run.json"
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
        output_path = _guard_output_location(args.output)
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
    if args.command == "real-smoke":
        try:
            report = run_real_smoke(
                args.config,
                args.output,
                expected_config_sha256=args.expected_config_sha256,
                resume=args.resume,
            )
        except (GateClosed, DataValidationError) as exc:
            print(
                json.dumps(
                    {"status": exc.status, "detail": exc.detail}, sort_keys=True
                )
            )
            return 2
        run_path = Path(args.output).resolve() / "run.json"
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "metrics_status": report["metrics_status"],
                    "run": str(run_path),
                    "run_sha256": sha256_file(run_path),
                    "ranking_seal_sha256": {
                        method_id: reference["sha256"]
                        for method_id, reference in sorted(
                            report["ranking_seals"].items()
                        )
                    },
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "SEALED_PRE_ORACLE" else 2
    report = census_real_assets(
        dataset_path=args.dataset,
        oracle_path=args.oracle,
        density_manifest_path=args.density_manifest,
        actor_authority_path=args.actor_authority,
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
        "oracle_supplied": args.oracle is not None,
        "density_manifest_supplied": args.density_manifest is not None,
        "actor_authority_supplied": args.actor_authority is not None,
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
        _write_canonical_json(_guard_output_location(args.output), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
