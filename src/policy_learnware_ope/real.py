"""Digest-locked bridges to frozen v03 banks and FPO actors.

The v03 ``.npz`` files remain read-only.  This module validates one bank
against the caller-pinned P0 census, reconstructs native episode/time identity
from ``episode_offsets``, and publishes the frozen B=24 fit/validation/s0 views
as a no-clobber export.  The actor bridge consumes caller-pinned authority and
keeps historical FPO sampling explicitly keyed.  This module contains no Raw
or oracle logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import importlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping
import zipfile

import numpy as np

from .adapters import sha256_file
from .core import (
    DataValidationError,
    EstimateStatus,
    PolicySemantics,
    TransitionBatch,
)


P0_CENSUS_SCHEMA = "policy-learnware.ope.p0-live-census.v1"
EXPORT_SCHEMA = "policy-learnware.ope.existing-log-export.v1"
MEMBERSHIP_PROTOCOL = "ope-existing-log-membership-v1"
ACTOR_AUTHORITY_SCHEMA = "policy-learnware.ope.actor-authority.v1"
FROZEN_FPO_PROVIDER_SCHEMA = "policy-learnware.ope.frozen-fpo-actor.v1"
FROZEN_FPO_COMMIT = "418c2554f7cd22d52e14c07d951280929d73bf2f"
FROZEN_FPO_TREE = "54bc61908de03282897eb05ef0cc027202d2d1a7"
_SAME_BACKEND_ATOL = 1.0e-6
_SAME_BACKEND_RTOL = 1.0e-6
_CROSS_BACKEND_RAW_ATOL = 7.0e-2
_CROSS_BACKEND_ENV_ATOL = 3.0e-2
_CROSS_BACKEND_RTOL = 1.0e-5
_CROSS_BACKEND_EVIDENCE = {
    "version": "v04a-selected-market-f32-m2cpu-v2",
    "dtype": "float32",
    "scope": "stored-golden compatibility only; values do not enter ranking",
    "v03_evidence_aggregate_sha256": (
        "7892455fae56637dbc44c0bdd969cfc7c7182ec67af5a4db3b71b1d961911089"
    ),
    "m2_cpu_evidence_sha256": (
        "50ac5e13b021a415ab251f51672fabb61a334e79ae25f94e95d63c35a8f9fc46"
    ),
}
SPLIT_SEED = 40401
EPISODE_COUNT = 32
HORIZON = 1000
FIT_EPISODES = 24
VALIDATION_EPISODES = 4
S0_EPISODES = 4
FIT_ROWS_PER_EPISODE = 64

_REQUIRED_BANK_KEYS = {
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "episode_offsets",
}
_SPLITS = ("fit", "validation", "s0")
_ACTOR_AUTHORITY_TOKEN = object()


def _invalid(detail: str) -> DataValidationError:
    return DataValidationError(EstimateStatus.INVALID_DATA.value, detail)


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _invalid(f"value is not canonical-JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise _invalid(f"{name} must be a lowercase SHA-256 digest")
    return text


def _hash_rank(
    namespace: str,
    context_id: str,
    episode_id: int,
    split_seed: int,
    native_timestep: int | None = None,
) -> str:
    record: dict[str, Any] = {
        "context_id": context_id,
        "episode_id": int(episode_id),
        "namespace": namespace,
        "split_seed": int(split_seed),
    }
    if native_timestep is not None:
        record["native_timestep"] = int(native_timestep)
    return _digest(record)


def _physical_rows(
    context_id: str,
    bank_sha256: str,
    episode_ids: list[int],
    split: str,
    split_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        if split == "fit":
            interior = sorted(
                range(1, HORIZON - 1),
                key=lambda timestep: _hash_rank(
                    "fit-native-t-v1",
                    context_id,
                    episode_id,
                    split_seed,
                    timestep,
                ),
            )[: FIT_ROWS_PER_EPISODE - 2]
            timesteps = sorted([0, *interior, HORIZON - 1])
        elif split == "validation":
            timesteps = range(HORIZON)
        elif split == "s0":
            timesteps = (0,)
        else:  # pragma: no cover - all callers validate split first
            raise ValueError(split)
        rows.extend(
            {
                "bank_sha256": bank_sha256,
                "context_id": context_id,
                "episode_id": int(episode_id),
                "native_timestep": int(timestep),
                "row_index": int(episode_id * HORIZON + timestep),
            }
            for timestep in timesteps
        )
    return rows


def freeze_membership(
    context_id: str,
    bank_sha256: str,
    *,
    split_seed: int = SPLIT_SEED,
) -> dict[str, dict[str, Any]]:
    """Rebuild the frozen r2 physical memberships for one 32x1000 bank."""

    context_id = str(context_id)
    if not context_id:
        raise _invalid("context_id must be non-empty")
    bank_sha256 = _require_sha256(bank_sha256, "bank_sha256")
    if isinstance(split_seed, bool) or int(split_seed) != split_seed:
        raise _invalid("split_seed must be an integer")
    split_seed = int(split_seed)
    episode_order = sorted(
        range(EPISODE_COUNT),
        key=lambda episode_id: _hash_rank(
            "episode-split-v1",
            context_id,
            episode_id,
            split_seed,
        ),
    )
    episodes = {
        "fit": episode_order[:FIT_EPISODES],
        "validation": episode_order[
            FIT_EPISODES : FIT_EPISODES + VALIDATION_EPISODES
        ],
        "s0": episode_order[-S0_EPISODES:],
    }
    result: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        rows = _physical_rows(
            context_id,
            bank_sha256,
            episodes[split],
            split,
            split_seed,
        )
        result[split] = {
            "episode_ids": episodes[split],
            "membership_digest": _digest(rows),
            "rows": rows,
        }
    return result


def _load_census(
    census_path: Path,
    expected_census_sha256: str,
    context_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_census_sha256 = _require_sha256(
        expected_census_sha256, "expected_census_sha256"
    )
    actual_census_sha256 = sha256_file(census_path)
    if actual_census_sha256 != expected_census_sha256:
        raise _invalid(
            "P0 census digest mismatch: "
            f"expected {expected_census_sha256}, got {actual_census_sha256}"
        )
    try:
        census = json.loads(census_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _invalid(f"P0 census is unreadable: {exc}") from exc
    if not isinstance(census, dict) or census.get("schema") != P0_CENSUS_SCHEMA:
        raise _invalid("P0 census schema is missing or unsupported")
    try:
        bank_rows = census["asset_facts"]["banks"]["full_rows"]
        protocol = census["freeze"]["physical_membership_protocol"]
        split_seed = census["freeze"]["split_seed"]
        frozen = census["freeze"]["memberships"][context_id]
    except (KeyError, TypeError) as exc:
        raise _invalid(f"P0 census lacks required bank/membership evidence: {exc}") from exc
    matches = [
        row
        for row in bank_rows
        if isinstance(row, dict) and row.get("context_id") == context_id
    ]
    if len(matches) != 1:
        raise _invalid("P0 census must contain exactly one row for context_id")
    row = matches[0]
    if row.get("status") != "PASS":
        raise _invalid("P0 census bank row is not PASS")
    if protocol.get("version") != MEMBERSHIP_PROTOCOL:
        raise _invalid("P0 census membership protocol is not the frozen r2 protocol")
    if split_seed != SPLIT_SEED:
        raise _invalid("P0 census split seed differs from the frozen r2 seed")
    if not isinstance(frozen, dict):
        raise _invalid("P0 census frozen membership is malformed")
    return census, row, frozen


def _bool_array(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise _invalid(f"{name} must be one-dimensional")
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=True)
    if raw.dtype.kind in "iu" and np.all((raw == 0) | (raw == 1)):
        return raw.astype(bool)
    raise _invalid(f"{name} must contain booleans only")


def _source_snapshot(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise _invalid(f"source bank is unreadable: {exc}") from exc
    return sha256_file(path), int(stat.st_mtime_ns), int(stat.st_size)


def _read_and_validate_bank(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(_REQUIRED_BANK_KEYS.difference(archive.files))
            if missing:
                raise _invalid(f"source bank lacks required arrays: {missing}")
            arrays = {name: np.array(archive[name], copy=True) for name in _REQUIRED_BANK_KEYS}
    except DataValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise _invalid(f"source bank cannot be loaded safely: {exc}") from exc

    observations = arrays["observation"]
    actions = arrays["action"]
    rewards = arrays["reward"]
    next_observations = arrays["next_observation"]
    terminated = _bool_array(arrays["terminated"], "terminated")
    truncated = _bool_array(arrays["truncated"], "truncated")
    offsets_raw = np.asarray(arrays["episode_offsets"])
    if offsets_raw.ndim != 1 or offsets_raw.dtype.kind not in "iu":
        raise _invalid("episode_offsets must be a one-dimensional integer array")
    offsets = offsets_raw.astype(np.int64, copy=True)
    expected_offsets = np.arange(EPISODE_COUNT + 1, dtype=np.int64) * HORIZON
    if not np.array_equal(offsets, expected_offsets):
        raise _invalid("episode_offsets do not describe the frozen 32x1000 bank")
    row_count = EPISODE_COUNT * HORIZON
    if observations.ndim != 2 or observations.shape[0] != row_count or observations.shape[1] == 0:
        raise _invalid("observation has the wrong 32x1000 row shape")
    if actions.ndim != 2 or actions.shape[0] != row_count or actions.shape[1] == 0:
        raise _invalid("action has the wrong 32x1000 row shape")
    if next_observations.shape != observations.shape:
        raise _invalid("next_observation shape disagrees with observation")
    if rewards.shape != (row_count,):
        raise _invalid("reward has the wrong 32x1000 row shape")
    for name, array in {
        "observation": observations,
        "action": actions,
        "reward": rewards,
        "next_observation": next_observations,
    }.items():
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise _invalid(f"{name} contains a non-finite or non-numeric value")
    if terminated.shape != (row_count,) or truncated.shape != (row_count,):
        raise _invalid("termination arrays have the wrong 32x1000 row shape")
    if np.any(terminated & truncated):
        raise _invalid("a source row cannot be both terminated and truncated")
    done = (terminated | truncated).reshape(EPISODE_COUNT, HORIZON)
    if np.any(done[:, :-1]) or not np.all(done[:, -1]):
        raise _invalid("termination flags contradict the frozen episode offsets")
    if sha256_file(path) != expected_sha256:
        raise _invalid("source bank changed while it was being validated")
    arrays.update(
        {
            "terminated": terminated,
            "truncated": truncated,
            "episode_offsets": offsets,
        }
    )
    return arrays


def _validate_frozen_membership(
    context_id: str,
    bank_sha256: str,
    frozen: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    membership = freeze_membership(context_id, bank_sha256, split_seed=SPLIT_SEED)
    for split in _SPLITS:
        episode_key = f"{split}_episode_ids"
        digest_key = f"{split}_membership_digest"
        if list(frozen.get(episode_key, [])) != membership[split]["episode_ids"]:
            raise _invalid(f"P0 census {split} episode membership does not reproduce")
        if frozen.get(digest_key) != membership[split]["membership_digest"]:
            raise _invalid(f"P0 census {split} membership digest does not reproduce")
    expected_counts = {
        "fit_transition_count": FIT_EPISODES * FIT_ROWS_PER_EPISODE,
        "validation_transition_count": VALIDATION_EPISODES * HORIZON,
        "s0_state_count": S0_EPISODES,
    }
    for key, expected in expected_counts.items():
        if frozen.get(key) != expected:
            raise _invalid(f"P0 census {key} is inconsistent with the r2 protocol")
    return membership


def census_bank(
    census_path: str | Path,
    *,
    expected_census_sha256: str,
    context_id: str,
    bank_path: str | Path,
) -> dict[str, Any]:
    """Validate one bank and its r2 memberships without writing an artifact."""

    census_path = Path(census_path)
    bank_path = Path(bank_path)
    _, row, frozen = _load_census(census_path, expected_census_sha256, context_id)
    expected_bank_sha256 = _require_sha256(row.get("bank_sha256", ""), "census bank_sha256")
    before = _source_snapshot(bank_path)
    if before[0] != expected_bank_sha256:
        raise _invalid("source bank digest disagrees with the pinned P0 census")
    arrays = _read_and_validate_bank(bank_path, expected_bank_sha256)
    after = _source_snapshot(bank_path)
    if after != before:
        raise _invalid("source bank hash, mtime, or size changed during census")
    if row.get("transition_count") != EPISODE_COUNT * HORIZON or row.get("episode_count") != EPISODE_COUNT:
        raise _invalid("P0 census bank dimensions disagree with the frozen protocol")
    if row.get("observation_dim") != int(arrays["observation"].shape[1]):
        raise _invalid("P0 census observation dimension disagrees with the bank")
    if row.get("action_dim") != int(arrays["action"].shape[1]):
        raise _invalid("P0 census action dimension disagrees with the bank")
    membership = _validate_frozen_membership(context_id, expected_bank_sha256, frozen)
    return {
        "status": "PASS",
        "context_id": context_id,
        "task_id": row.get("task_id"),
        "role": row.get("role"),
        "dataset_digest": row.get("dataset_digest"),
        "bank_sha256": expected_bank_sha256,
        "episode_count": EPISODE_COUNT,
        "horizon": HORIZON,
        "observation_dim": int(arrays["observation"].shape[1]),
        "action_dim": int(arrays["action"].shape[1]),
        "membership_digests": {
            split: membership[split]["membership_digest"] for split in _SPLITS
        },
        "source_hash_unchanged": True,
        "source_mtime_unchanged": True,
    }


def _split_arrays(
    source: Mapping[str, np.ndarray],
    rows: list[dict[str, Any]],
    split: str,
    bank_sha256: str,
) -> dict[str, np.ndarray]:
    source_rows = np.asarray([row["row_index"] for row in rows], dtype=np.int64)
    episode_id = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
    native_timestep = np.asarray(
        [row["native_timestep"] for row in rows], dtype=np.int64
    )
    per_episode = {"fit": FIT_ROWS_PER_EPISODE, "validation": HORIZON, "s0": 1}[split]
    episode_count = {"fit": FIT_EPISODES, "validation": VALIDATION_EPISODES, "s0": S0_EPISODES}[split]
    episode_offsets = np.arange(episode_count + 1, dtype=np.int64) * per_episode
    next_behavior_action = np.array(source["action"][source_rows], copy=True)
    next_behavior_action_valid = native_timestep < HORIZON - 1
    next_behavior_action[next_behavior_action_valid] = source["action"][
        source_rows[next_behavior_action_valid] + 1
    ]
    terminated = np.asarray(source["terminated"][source_rows], dtype=bool)
    truncated = np.asarray(source["truncated"][source_rows], dtype=bool)
    truncation_reason = np.full(len(rows), "none", dtype="U16")
    truncation_reason[truncated] = "horizon"
    dataset_cut = np.full(len(rows), split == "s0", dtype=bool)
    return {
        "action": np.asarray(source["action"][source_rows]),
        "dataset_cut": dataset_cut,
        "episode_id": episode_id,
        "episode_offsets": episode_offsets,
        "membership_digest": np.asarray(_digest(rows)),
        "native_timestep": native_timestep,
        "next_behavior_action": next_behavior_action,
        "next_behavior_action_valid": next_behavior_action_valid,
        "next_observation": np.asarray(source["next_observation"][source_rows]),
        "observation": np.asarray(source["observation"][source_rows]),
        "reward": np.asarray(source["reward"][source_rows]),
        "source_digest": np.asarray(bank_sha256),
        "source_row_index": source_rows,
        "terminated": terminated,
        "timestep_provenance": np.asarray("native_indices"),
        "truncated": truncated,
        "truncation_reason": truncation_reason,
    }


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100444 << 16
            archive.writestr(info, buffer.getvalue())


def _batch_from_arrays(arrays: Mapping[str, np.ndarray]) -> TransitionBatch:
    return TransitionBatch(
        observation=arrays["observation"],
        action=arrays["action"],
        reward=arrays["reward"],
        next_observation=arrays["next_observation"],
        terminated=arrays["terminated"],
        truncated=arrays["truncated"],
        dataset_cut=arrays["dataset_cut"],
        native_timestep=arrays["native_timestep"],
        episode_id=arrays["episode_id"],
        episode_offsets=arrays["episode_offsets"],
        timestep_provenance=str(np.asarray(arrays["timestep_provenance"]).item()),
        next_behavior_action=arrays["next_behavior_action"],
        truncation_reason=arrays["truncation_reason"],
        source_digest=str(np.asarray(arrays["source_digest"]).item()),
    )


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise _invalid(f"exported transition file is unreadable: {exc}") from exc


def export_existing_log(
    census_path: str | Path,
    *,
    expected_census_sha256: str,
    context_id: str,
    bank_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one immutable B24 export after all source checks pass.

    The output directory must not exist and its parent must already exist.
    Construction and validation occur in a sibling temporary directory; the
    completed directory is renamed into place only after a final source
    hash/mtime check.
    """

    census_path = Path(census_path)
    bank_path = Path(bank_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"no-clobber output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output_dir.parent}")
    census, row, frozen = _load_census(
        census_path, expected_census_sha256, context_id
    )
    expected_bank_sha256 = _require_sha256(row.get("bank_sha256", ""), "census bank_sha256")
    source_before = _source_snapshot(bank_path)
    if source_before[0] != expected_bank_sha256:
        raise _invalid("source bank digest disagrees with the pinned P0 census")
    source = _read_and_validate_bank(bank_path, expected_bank_sha256)
    if row.get("transition_count") != EPISODE_COUNT * HORIZON or row.get("episode_count") != EPISODE_COUNT:
        raise _invalid("P0 census bank dimensions disagree with the frozen protocol")
    if row.get("observation_dim") != int(source["observation"].shape[1]) or row.get("action_dim") != int(source["action"].shape[1]):
        raise _invalid("P0 census feature dimensions disagree with the source bank")
    membership = _validate_frozen_membership(context_id, expected_bank_sha256, frozen)
    if _source_snapshot(bank_path) != source_before:
        raise _invalid("source bank hash, mtime, or size changed during validation")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    published = False
    try:
        split_manifest: dict[str, Any] = {}
        for split in _SPLITS:
            arrays = _split_arrays(
                source,
                membership[split]["rows"],
                split,
                expected_bank_sha256,
            )
            batch = _batch_from_arrays(arrays)
            filename = f"{split}.npz"
            path = temporary / filename
            _write_deterministic_npz(path, arrays)
            reloaded = _read_npz(path)
            _batch_from_arrays(reloaded)
            if not np.array_equal(
                reloaded["source_row_index"], arrays["source_row_index"]
            ):
                raise _invalid(f"{split} export row identity changed during serialization")
            split_manifest[split] = {
                "dataset_cut_count": int(np.sum(batch.dataset_cut)),
                "episode_ids": membership[split]["episode_ids"],
                "episode_count": batch.episode_count,
                "file": filename,
                "file_sha256": sha256_file(path),
                "membership_digest": membership[split]["membership_digest"],
                "physical_adjacent_action_count": int(
                    np.sum(arrays["next_behavior_action_valid"])
                ),
                "terminal_boundary_filler_count": int(
                    np.sum(~arrays["next_behavior_action_valid"])
                ),
                "transition_count": len(batch),
            }
        manifest = {
            "census": {
                "schema": census["schema"],
                "sha256": _require_sha256(
                    expected_census_sha256, "expected_census_sha256"
                ),
            },
            "context": {
                "context_id": context_id,
                "dataset_digest": row.get("dataset_digest"),
                "role": row.get("role"),
                "task_id": row.get("task_id"),
            },
            "membership_protocol": {
                "fit_episode_count": FIT_EPISODES,
                "fit_rows_per_episode": FIT_ROWS_PER_EPISODE,
                "horizon": HORIZON,
                "s0_episode_count": S0_EPISODES,
                "split_seed": SPLIT_SEED,
                "validation_episode_count": VALIDATION_EPISODES,
                "version": MEMBERSHIP_PROTOCOL,
            },
            "schema": EXPORT_SCHEMA,
            "source": {
                "bank_name": bank_path.name,
                "bank_sha256": expected_bank_sha256,
                "hash_unchanged": True,
                "mtime_unchanged": True,
                "read_only_projection": True,
            },
            "splits": split_manifest,
            "status": "PASS",
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_bytes(manifest))
        for path in temporary.iterdir():
            path.chmod(0o444)
        if _source_snapshot(bank_path) != source_before:
            raise _invalid("source bank hash, mtime, or size changed during export")
        if output_dir.exists():
            raise FileExistsError(f"no-clobber output already exists: {output_dir}")
        os.rename(temporary, output_dir)
        published = True
        manifest_sha256 = sha256_file(output_dir / "manifest.json")
        return {
            "status": "PASS",
            "output_dir": str(output_dir),
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
        }
    finally:
        if not published and temporary.exists():
            for path in temporary.iterdir():
                path.chmod(0o600)
            shutil.rmtree(temporary)


def load_export(
    output_dir: str | Path,
    split: str,
    *,
    expected_manifest_sha256: str,
) -> TransitionBatch:
    """Load a split only when a caller-pinned manifest and file digest match."""

    if split not in _SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise _invalid("export manifest digest mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _invalid(f"export manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPORT_SCHEMA:
        raise _invalid("export manifest schema is missing or unsupported")
    try:
        split_record = manifest["splits"][split]
        bank_sha256 = _require_sha256(
            manifest["source"]["bank_sha256"], "manifest bank_sha256"
        )
        context_id = str(manifest["context"]["context_id"])
    except (KeyError, TypeError) as exc:
        raise _invalid(f"export manifest lacks required fields: {exc}") from exc
    filename = split_record.get("file")
    if filename != f"{split}.npz":
        raise _invalid("export split filename is not canonical")
    path = output_dir / filename
    if sha256_file(path) != split_record.get("file_sha256"):
        raise _invalid("export split file digest mismatch")
    arrays = _read_npz(path)
    required = {
        "action",
        "dataset_cut",
        "episode_id",
        "episode_offsets",
        "membership_digest",
        "native_timestep",
        "next_behavior_action",
        "next_behavior_action_valid",
        "next_observation",
        "observation",
        "reward",
        "source_digest",
        "source_row_index",
        "terminated",
        "timestep_provenance",
        "truncated",
        "truncation_reason",
    }
    if set(arrays) != required:
        raise _invalid("export split arrays do not match the frozen schema")
    if str(np.asarray(arrays["source_digest"]).item()) != bank_sha256:
        raise _invalid("export split source digest disagrees with manifest")
    source_rows = np.asarray(arrays["source_row_index"])
    episode_ids = np.asarray(arrays["episode_id"])
    native_times = np.asarray(arrays["native_timestep"])
    if source_rows.ndim != 1 or episode_ids.shape != source_rows.shape or native_times.shape != source_rows.shape:
        raise _invalid("export physical row identity arrays are malformed")
    expected_source_rows = episode_ids * HORIZON + native_times
    if not np.array_equal(source_rows, expected_source_rows):
        raise _invalid("export row indices disagree with native episode/time identity")
    rows = [
        {
            "bank_sha256": bank_sha256,
            "context_id": context_id,
            "episode_id": int(episode_id),
            "native_timestep": int(native_timestep),
            "row_index": int(row_index),
        }
        for episode_id, native_timestep, row_index in zip(
            episode_ids, native_times, source_rows, strict=True
        )
    ]
    membership_digest = _digest(rows)
    if membership_digest != split_record.get("membership_digest") or membership_digest != str(np.asarray(arrays["membership_digest"]).item()):
        raise _invalid("export physical membership digest mismatch")
    valid = np.asarray(arrays["next_behavior_action_valid"])
    if valid.dtype.kind != "b" or not np.array_equal(valid, native_times < HORIZON - 1):
        raise _invalid("export adjacent-action availability mask is inconsistent")
    expected_cut = np.full(len(source_rows), split == "s0", dtype=bool)
    if not np.array_equal(np.asarray(arrays["dataset_cut"]), expected_cut):
        raise _invalid("export dataset-cut semantics disagree with split role")
    batch = _batch_from_arrays(arrays)
    if len(batch) != split_record.get("transition_count") or batch.episode_count != split_record.get("episode_count"):
        raise _invalid("export batch counts disagree with manifest")
    return batch


def export_reward_free_query(
    output_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the fit membership as the exact reward-free Raw query ABI."""

    output_dir = Path(output_dir)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"no-clobber query already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"query parent does not exist: {output_path.parent}")
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    manifest_path = output_dir / "manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise _invalid("export manifest digest mismatch while building Raw query")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fit_record = manifest["splits"]["fit"]
        membership_digest = _require_sha256(
            fit_record["membership_digest"], "fit membership_digest"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise _invalid(f"export manifest lacks the fit membership: {exc}") from exc

    batch = load_export(
        output_dir,
        "fit",
        expected_manifest_sha256=expected_manifest_sha256,
    )
    arrays = {
        "action": batch.action,
        "episode_offsets": batch.episode_offsets,
        "membership_digest": np.asarray(membership_digest),
        "native_timestep": batch.native_timestep,
        "next_observation": batch.next_observation,
        "observation": batch.observation,
    }
    _write_deterministic_npz(output_path, arrays)
    output_path.chmod(0o444)
    return {
        "artifact_sha256": sha256_file(output_path),
        "fields": sorted(arrays),
        "membership_digest": membership_digest,
        "transition_count": len(batch),
    }


def _strict_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise _invalid(
            f"{where} fields differ from the frozen schema: observed={observed}"
        )
    return value


def _git_oid(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise _invalid(f"{name} must be a lowercase 40-hex Git object ID")
    return text


def _reject_authority_leakage(value: Any, where: str = "authority") -> None:
    """Keep actor authority path-free and incapable of carrying labels/envs."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if "oracle" in normalized or normalized in {
                "env",
                "environment",
                "env_path",
                "env_root",
                "environment_path",
                "environment_root",
            }:
                raise _invalid(f"{where} must not carry environment/oracle authority")
            _reject_authority_leakage(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_authority_leakage(item, f"{where}[{index}]")
        return
    if isinstance(value, str) and Path(value).is_absolute():
        raise _invalid(f"{where} must not embed an absolute path")


def _candidate_record(
    census: Mapping[str, Any], context_id: str, candidate_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        bank_rows = census["asset_facts"]["banks"]["full_rows"]
        candidate_sets = census["freeze"]["candidate_sets"]
    except (KeyError, TypeError) as exc:
        raise _invalid(f"P0 census lacks candidate authority evidence: {exc}") from exc
    contexts = [
        row
        for row in bank_rows
        if isinstance(row, dict) and row.get("context_id") == context_id
    ]
    if len(contexts) != 1:
        raise _invalid("P0 census must contain exactly one candidate context row")
    context = contexts[0]
    task_id = context.get("task_id")
    try:
        candidate_set = candidate_sets[task_id]
        records = candidate_set["records"]
    except (KeyError, TypeError) as exc:
        raise _invalid(f"P0 census lacks the context task candidate set: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise _invalid("P0 candidate record set is empty or malformed")
    if candidate_set.get("membership_digest") != _digest(records):
        raise _invalid("P0 candidate-set membership digest does not reproduce")
    candidate_ids = [record.get("opaque_learnware_id") for record in records]
    if candidate_set.get("candidate_ids") != candidate_ids:
        raise _invalid("P0 candidate ID list disagrees with its frozen records")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("opaque_learnware_id") == candidate_id
    ]
    if len(matches) != 1:
        raise _invalid("candidate_id is absent or duplicated in the P0 TASK_5 block")
    record = matches[0]
    if record.get("task_id") != task_id:
        raise _invalid("P0 candidate task differs from the target context task")
    return context, record


@dataclass(frozen=True, slots=True)
class ActorAuthority:
    """Caller-pinned, path-free authority for one frozen FPO candidate."""

    candidate_id: str
    task_id: str
    observation_dim: int
    action_dim: int
    execution_abi: Mapping[str, Any]
    bundle_digest: str
    bundle_manifest_sha256: str
    runtime_digest: str
    fpo_commit: str
    fpo_tree: str
    fpo_head_tree_digest: str
    fpo_source_digest: str
    fpo_source_file_count: int
    policy_repo_commit: str
    policy_repo_tree: str
    authority_sha256: str
    census_sha256: str
    _verification_token: object = field(repr=False, compare=False)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        census_path: str | Path,
        expected_census_sha256: str,
        context_id: str,
        candidate_id: str,
    ) -> "ActorAuthority":
        authority_path = Path(path)
        expected_sha256 = _require_sha256(expected_sha256, "expected actor authority SHA-256")
        if sha256_file(authority_path) != expected_sha256:
            raise _invalid("actor authority digest mismatch")
        try:
            payload = json.loads(authority_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _invalid(f"actor authority is unreadable: {exc}") from exc
        _reject_authority_leakage(payload)
        payload = _strict_keys(
            payload,
            {
                "schema",
                "provider_schema",
                "candidate_id",
                "bundle_digest",
                "bundle_manifest",
                "execution_abi",
                "observation_dim",
                "action_dim",
                "runtime_digest",
                "fpo_checkout",
                "policy_repo_checkout",
                "loader",
            },
            "actor authority",
        )
        if payload["schema"] != ACTOR_AUTHORITY_SCHEMA:
            raise _invalid("actor authority schema is unsupported")
        if payload["provider_schema"] != FROZEN_FPO_PROVIDER_SCHEMA:
            raise _invalid("actor authority provider schema is unsupported")
        if payload["candidate_id"] != candidate_id or not str(candidate_id):
            raise _invalid("actor authority candidate_id differs from the caller")
        manifest = _strict_keys(
            payload["bundle_manifest"], {"schema", "sha256"}, "bundle_manifest"
        )
        if manifest["schema"] != "policy-learnware.policy-bundle.v0":
            raise _invalid("actor authority bundle manifest schema is unsupported")
        fpo = _strict_keys(
            payload["fpo_checkout"],
            {
                "commit",
                "tree",
                "head_tree_digest",
                "source_digest",
                "source_file_count",
            },
            "fpo_checkout",
        )
        policy_repo = _strict_keys(
            payload["policy_repo_checkout"],
            {"commit", "tree"},
            "policy_repo_checkout",
        )
        loader = _strict_keys(
            payload["loader"], {"module", "function", "runtime_only"}, "loader"
        )
        if loader != {
            "module": "policy_learnware_v0.policy.loader",
            "function": "load_policy",
            "runtime_only": True,
        }:
            raise _invalid("actor authority does not bind the runtime-only frozen loader")
        observation_dim = payload["observation_dim"]
        action_dim = payload["action_dim"]
        for value, name in (
            (observation_dim, "observation_dim"),
            (action_dim, "action_dim"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _invalid(f"actor authority {name} must be a positive integer")

        census_path = Path(census_path)
        expected_census_sha256 = _require_sha256(
            expected_census_sha256, "expected_census_sha256"
        )
        if sha256_file(census_path) != expected_census_sha256:
            raise _invalid("P0 census digest mismatch while binding actor authority")
        try:
            census = json.loads(census_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _invalid(f"P0 census is unreadable: {exc}") from exc
        if not isinstance(census, dict) or census.get("schema") != P0_CENSUS_SCHEMA:
            raise _invalid("P0 census schema is missing or unsupported")
        context, record = _candidate_record(census, context_id, candidate_id)
        execution_abi = payload["execution_abi"]
        if not isinstance(execution_abi, Mapping) or dict(execution_abi) != record.get("execution_abi"):
            raise _invalid("actor authority execution ABI differs from the P0 candidate record")
        bundle_digest = _require_sha256(payload["bundle_digest"], "bundle_digest")
        manifest_sha256 = _require_sha256(
            manifest["sha256"], "bundle_manifest.sha256"
        )
        if bundle_digest != record.get("bundle_digest") or manifest_sha256 != bundle_digest:
            raise _invalid("actor authority bundle/manifest digest differs from P0")
        if observation_dim != context.get("observation_dim") or action_dim != context.get("action_dim"):
            raise _invalid("actor authority dimensions differ from the P0 context ABI")
        if execution_abi.get("action_transform_id") != "tanh":
            raise _invalid("frozen FPO provider requires the P0 tanh action transform")
        policy_commit = _git_oid(policy_repo["commit"], "policy repo commit")
        policy_tree = _git_oid(policy_repo["tree"], "policy repo tree")
        fpo_commit = _git_oid(fpo["commit"], "FPO commit")
        fpo_tree = _git_oid(fpo["tree"], "FPO tree")
        if fpo_commit != FROZEN_FPO_COMMIT or fpo_tree != FROZEN_FPO_TREE:
            raise _invalid("actor authority differs from the unique archived FPO checkout")
        source_file_count = fpo["source_file_count"]
        if (
            isinstance(source_file_count, bool)
            or not isinstance(source_file_count, int)
            or source_file_count <= 0
        ):
            raise _invalid("FPO source_file_count must be a positive integer")
        return cls(
            candidate_id=candidate_id,
            task_id=str(context.get("task_id")),
            observation_dim=int(observation_dim),
            action_dim=int(action_dim),
            execution_abi=MappingProxyType(dict(execution_abi)),
            bundle_digest=bundle_digest,
            bundle_manifest_sha256=manifest_sha256,
            runtime_digest=_require_sha256(payload["runtime_digest"], "runtime_digest"),
            fpo_commit=fpo_commit,
            fpo_tree=fpo_tree,
            fpo_head_tree_digest=_require_sha256(
                fpo["head_tree_digest"], "FPO head_tree_digest"
            ),
            fpo_source_digest=_require_sha256(
                fpo["source_digest"], "FPO source_digest"
            ),
            fpo_source_file_count=source_file_count,
            policy_repo_commit=policy_commit,
            policy_repo_tree=policy_tree,
            authority_sha256=expected_sha256,
            census_sha256=expected_census_sha256,
            _verification_token=_ACTOR_AUTHORITY_TOKEN,
        )


def _absolute_directory(path: str | Path, name: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise _invalid(f"{name} must be an absolute, non-symlink directory")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise _invalid(f"{name} does not exist: {exc}") from exc
    if not resolved.is_dir():
        raise _invalid(f"{name} is not a directory")
    prohibited = {"env", "environment", "environments", "oracle", "oracles"}
    if any(part.casefold() in prohibited for part in resolved.parts):
        raise _invalid(f"{name} must not resolve through an environment/oracle path")
    return resolved


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise _invalid(f"cannot verify Git checkout {root.name}: {detail}")
    return completed.stdout.strip()


def _verify_checkout(
    path: str | Path,
    *,
    expected_commit: str,
    expected_tree: str,
    name: str,
) -> Path:
    root = _absolute_directory(path, name)
    git_marker = root / ".git"
    if git_marker.is_symlink() or not git_marker.exists():
        raise _invalid(f"{name} lacks a local, non-symlink .git authority")
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise _invalid(f"{name} is not a Git worktree")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise _invalid(f"{name} is not the root of its Git worktree")
    if _git(root, "rev-parse", "HEAD") != expected_commit:
        raise _invalid(f"{name} HEAD differs from actor authority")
    if _git(root, "rev-parse", "HEAD^{tree}") != expected_tree:
        raise _invalid(f"{name} tree differs from actor authority")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise _invalid(f"{name} worktree is not clean")
    return root


def _checkout_source_identity(root: Path) -> dict[str, Any]:
    """Hash both the tracked Git description and live source bytes."""

    completed = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-rz", "--full-tree", "HEAD"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _invalid(f"cannot enumerate FPO source tree: {detail}")
    head_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            header, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise _invalid("cannot parse FPO Git tree") from exc
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or kind != "blob" or mode not in {
            "100644",
            "100755",
        }:
            raise _invalid(f"FPO tree contains an unsafe tracked entry: {relative!r}")
        path = root.joinpath(*pure.parts)
        try:
            metadata = path.lstat()
            data = path.read_bytes()
        except OSError as exc:
            raise _invalid(f"cannot read tracked FPO source {relative!r}: {exc}") from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise _invalid(f"tracked FPO source is not a regular file: {relative!r}")
        actual_mode = "100755" if metadata.st_mode & 0o111 else "100644"
        if actual_mode != mode:
            raise _invalid(f"tracked FPO source mode differs from HEAD: {relative!r}")
        head_records.append(
            {"mode": mode, "object": object_id, "path": relative}
        )
        source_records.append(
            {
                "bytes": len(data),
                "mode": actual_mode,
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if not head_records:
        raise _invalid("FPO source tree is empty")
    return {
        "head_tree_digest": _digest(head_records),
        "source_digest": _digest(source_records),
        "source_file_count": len(source_records),
    }


def _read_json_object(path: Path, where: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise _invalid(f"{where} is absent or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _invalid(f"{where} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise _invalid(f"{where} must be one JSON object")
    return value


def _verify_bundle(bundle_dir: str | Path, authority: ActorAuthority) -> Path:
    root = _absolute_directory(bundle_dir, "bundle_dir")
    manifest_path = root / "bundle_manifest.json"
    if sha256_file(manifest_path) != authority.bundle_manifest_sha256:
        raise _invalid("bundle manifest bytes differ from actor authority")
    manifest = _read_json_object(manifest_path, "bundle manifest")
    _strict_keys(
        manifest,
        {
            "schema",
            "complete",
            "created_at",
            "algorithm",
            "task",
            "seed",
            "outer_iteration",
            "environment_steps",
            "files",
        },
        "bundle manifest",
    )
    if (
        manifest["schema"] != "policy-learnware.policy-bundle.v0"
        or manifest["complete"] is not True
        or manifest["algorithm"] != "fpo"
        or manifest["task"] != authority.task_id
    ):
        raise _invalid("bundle manifest protocol/task differs from actor authority")
    files = manifest["files"]
    expected_files = {
        "actor.npz",
        "golden_io.npz",
        "obs_stats.npz",
        "policy_spec.json",
        "provenance.json",
    }
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise _invalid("bundle payload inventory differs from policy-bundle.v0")
    if {path.name for path in root.iterdir()} != expected_files | {"bundle_manifest.json"}:
        raise _invalid("bundle directory contains unmanifested entries")
    for filename in sorted(expected_files):
        metadata = _strict_keys(files[filename], {"bytes", "sha256"}, f"bundle {filename}")
        expected_size = metadata["bytes"]
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise _invalid(f"bundle {filename} byte count is invalid")
        expected_digest = _require_sha256(metadata["sha256"], f"bundle {filename} SHA-256")
        payload = root / filename
        if payload.is_symlink() or not payload.is_file():
            raise _invalid(f"bundle payload is absent or symlinked: {filename}")
        if payload.stat().st_size != expected_size or sha256_file(payload) != expected_digest:
            raise _invalid(f"bundle payload differs from manifest: {filename}")
    spec = _read_json_object(root / "policy_spec.json", "policy_spec.json")
    if (
        spec.get("observation_size") != authority.observation_dim
        or spec.get("action_size") != authority.action_dim
    ):
        raise _invalid("bundle policy_spec dimensions differ from actor authority")
    provenance = _read_json_object(root / "provenance.json", "provenance.json")
    if (
        provenance.get("fpo_commit") != authority.fpo_commit
        or provenance.get("runtime_digest") != authority.runtime_digest
    ):
        raise _invalid("bundle runtime provenance differs from actor authority")
    return root


def _load_policy_runtime(
    *,
    policy_repo_checkout: Path,
    bundle_dir: Path,
    fpo_checkout: Path,
    authority: ActorAuthority,
) -> Any:
    """Import only the loader from the verified old policy checkout."""

    source_root = (policy_repo_checkout / "src").resolve()
    loader_path = source_root / "policy_learnware_v0" / "policy" / "loader.py"
    if not loader_path.is_file() or loader_path.is_symlink():
        raise _invalid("verified old policy checkout lacks its frozen loader")
    for name, module in tuple(sys.modules.items()):
        if name != "policy_learnware_v0" and not name.startswith("policy_learnware_v0."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            Path(origin).resolve().relative_to(source_root)
        except ValueError as exc:
            raise _invalid(
                f"cached module {name!r} came from another policy checkout"
            ) from exc
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    # Both checkouts are frozen inputs.  The runner process stays bytecode-off
    # after this point so lazy policy/FPO imports cannot create __pycache__.
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module("policy_learnware_v0.policy.loader")
        loader = getattr(module, "load_policy")
    except (ImportError, AttributeError) as exc:
        raise _invalid(f"frozen policy loader is unavailable: {exc}") from exc
    origin = getattr(module, "__file__", None)
    if origin is None or Path(origin).resolve() != loader_path.resolve():
        raise _invalid("frozen policy loader was imported from another checkout")
    try:
        return loader(
            bundle_dir,
            fpo_root=fpo_checkout,
            expected_fpo_commit=authority.fpo_commit,
            expected_runtime_digest=authority.runtime_digest,
            runtime_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _invalid(f"frozen FPO runtime loader failed: {exc}") from exc


def _jax_modules() -> tuple[Any, Any]:
    try:
        return importlib.import_module("jax"), importlib.import_module("jax.numpy")
    except ImportError as exc:
        raise _invalid("frozen FPO action sampling requires JAX") from exc


def _uint64_key_data(keys: np.ndarray) -> np.ndarray:
    """Map u64 seeds or checked ``[N,2]`` words to JAX key data."""

    values = np.asarray(keys)
    if values.dtype.kind not in "iu" or values.ndim not in {1, 2}:
        raise DataValidationError(
            EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
            "FrozenFPOActor requires uint64 seeds or two integer key words per row",
        )
    if values.dtype.kind == "i" and np.any(values < 0):
        raise DataValidationError(
            EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
            "FrozenFPOActor key words must be non-negative",
        )
    if values.ndim == 1:
        if values.dtype != np.dtype(np.uint64):
            raise DataValidationError(
                EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
                "scalar FrozenFPOActor keys must have uint64 dtype",
            )
        high = (values >> np.uint64(32)).astype(np.uint32)
        low = (values & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        result = np.column_stack((high, low)).astype(np.uint32, copy=False)
    else:
        if values.shape[1] != 2 or np.any(values > np.iinfo(np.uint32).max):
            raise DataValidationError(
                EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
                "structured FrozenFPOActor key words must fit uint32 exactly",
            )
        result = values.astype(np.uint32, copy=True)
    result.setflags(write=False)
    return result


def _max_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    delta = np.abs(actual64 - expected64)
    scale = np.maximum(
        np.maximum(np.abs(actual64), np.abs(expected64)),
        np.finfo(np.float32).eps,
    )
    return (
        float(np.max(delta, initial=0.0)),
        float(np.max(delta / scale, initial=0.0)),
    )


def _golden_io(bundle_dir: Path, authority: ActorAuthority) -> dict[str, np.ndarray]:
    try:
        with np.load(bundle_dir / "golden_io.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "observation",
                "prng_key_data",
                "raw_action",
                "environment_action",
            }:
                raise _invalid("golden_io numerical inventory differs from policy-bundle.v0")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise _invalid(f"golden_io is unreadable: {exc}") from exc
    observation = arrays["observation"]
    key_data = arrays["prng_key_data"]
    raw_action = arrays["raw_action"]
    environment_action = arrays["environment_action"]
    if (
        observation.shape != (8, authority.observation_dim)
        or raw_action.shape != (8, authority.action_dim)
        or environment_action.shape != raw_action.shape
        or observation.dtype != np.dtype(np.float32)
        or raw_action.dtype != np.dtype(np.float32)
        or environment_action.dtype != np.dtype(np.float32)
        or key_data.shape != (2,)
        or key_data.dtype != np.dtype(np.uint32)
    ):
        raise _invalid("golden_io shape/dtype differs from the frozen FPO ABI")
    if not all(
        np.all(np.isfinite(value))
        for value in (observation, raw_action, environment_action)
    ):
        raise _invalid("golden_io contains non-finite values")
    if not np.allclose(
        environment_action,
        np.tanh(raw_action),
        atol=_SAME_BACKEND_ATOL,
        rtol=_SAME_BACKEND_RTOL,
    ):
        raise _invalid("golden_io environment action is not tanh(raw_action)")
    return arrays


def _compile_action_sampler(native_state: Any, jax: Any, jnp: Any) -> Any:
    def sample_one(observation: Any, key: Any) -> Any:
        raw_action, _ = native_state.sample_action(
            observation, key, deterministic=True
        )
        return jnp.tanh(raw_action)

    return jax.jit(jax.vmap(sample_one))


def _initial_actor_parity(
    native_state: Any,
    bundle_dir: Path,
    authority: ActorAuthority,
) -> tuple[dict[str, Any], Any]:
    """Replay the locked FPO probe and bind the compiled action path."""

    golden = _golden_io(bundle_dir, authority)
    observation = golden["observation"]
    expected_raw = golden["raw_action"]
    expected_environment = golden["environment_action"]
    key_data = golden["prng_key_data"]
    jax, jnp = _jax_modules()
    wrap = getattr(jax.random, "wrap_key_data", None)
    device_key_data = jnp.asarray(key_data, dtype=jnp.uint32)
    base_key = wrap(device_key_data) if wrap is not None else device_key_data
    device_observation = jnp.asarray(observation, dtype=jnp.float32)
    try:
        raw_first, _ = native_state.sample_action(
            device_observation, base_key, deterministic=True
        )
        raw_second, _ = native_state.sample_action(
            device_observation, base_key, deterministic=True
        )
        actual_raw = np.asarray(jax.device_get(raw_first))
        repeated_raw = np.asarray(jax.device_get(raw_second))
        actual_environment = np.asarray(jax.device_get(jnp.tanh(raw_first)))
        repeated_environment = np.asarray(jax.device_get(jnp.tanh(raw_second)))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise _invalid(f"frozen FPO golden replay failed: {exc}") from exc
    if (
        actual_raw.shape != expected_raw.shape
        or repeated_raw.shape != expected_raw.shape
        or actual_environment.shape != expected_environment.shape
        or repeated_environment.shape != expected_environment.shape
        or actual_raw.dtype != np.dtype(np.float32)
        or repeated_raw.dtype != np.dtype(np.float32)
        or actual_environment.dtype != np.dtype(np.float32)
        or repeated_environment.dtype != np.dtype(np.float32)
        or not all(
            np.all(np.isfinite(value))
            for value in (
                actual_raw,
                repeated_raw,
                actual_environment,
                repeated_environment,
            )
        )
    ):
        raise _invalid("frozen FPO golden replay returned an invalid action tensor")

    replay_raw_abs, replay_raw_rel = _max_errors(actual_raw, repeated_raw)
    replay_env_abs, replay_env_rel = _max_errors(
        actual_environment, repeated_environment
    )
    if not (
        np.allclose(
            actual_raw,
            repeated_raw,
            atol=_SAME_BACKEND_ATOL,
            rtol=_SAME_BACKEND_RTOL,
        )
        and np.allclose(
            actual_environment,
            repeated_environment,
            atol=_SAME_BACKEND_ATOL,
            rtol=_SAME_BACKEND_RTOL,
        )
    ):
        raise _invalid("same-key frozen FPO replay is not reproducible")
    transform_abs, transform_rel = _max_errors(
        actual_environment, np.tanh(actual_raw)
    )
    if not np.allclose(
        actual_environment,
        np.tanh(actual_raw),
        atol=_SAME_BACKEND_ATOL,
        rtol=_SAME_BACKEND_RTOL,
    ):
        raise _invalid("frozen FPO deployment action is not tanh(raw_action)")

    try:
        different_key = jax.random.fold_in(base_key, np.uint32(0x5A17C0DE))
        different_raw, _ = native_state.sample_action(
            device_observation, different_key, deterministic=True
        )
        different_raw = np.asarray(jax.device_get(different_raw))
        different_environment = np.asarray(jax.device_get(jnp.tanh(different_raw)))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise _invalid(f"frozen FPO different-key sensitivity failed: {exc}") from exc
    if (
        different_raw.shape != actual_raw.shape
        or different_environment.shape != actual_environment.shape
        or not np.all(np.isfinite(different_raw))
        or not np.all(np.isfinite(different_environment))
    ):
        raise _invalid("frozen FPO different-key sensitivity returned invalid actions")
    different_raw_abs, different_raw_rel = _max_errors(actual_raw, different_raw)
    different_env_abs, different_env_rel = _max_errors(
        actual_environment, different_environment
    )
    different_rows = int(
        np.sum(np.any(actual_environment != different_environment, axis=1))
    )
    if different_rows == 0:
        raise _invalid("frozen FPO actions are insensitive to a changed PRNG key")

    raw_abs, raw_rel = _max_errors(actual_raw, expected_raw)
    env_abs, env_rel = _max_errors(actual_environment, expected_environment)
    exact_golden = bool(
        np.allclose(
            actual_raw,
            expected_raw,
            atol=_SAME_BACKEND_ATOL,
            rtol=_SAME_BACKEND_RTOL,
        )
        and np.allclose(
            actual_environment,
            expected_environment,
            atol=_SAME_BACKEND_ATOL,
            rtol=_SAME_BACKEND_RTOL,
        )
    )
    if not (
        np.allclose(
            actual_raw,
            expected_raw,
            atol=_CROSS_BACKEND_RAW_ATOL,
            rtol=_CROSS_BACKEND_RTOL,
        )
        and np.allclose(
            actual_environment,
            expected_environment,
            atol=_CROSS_BACKEND_ENV_ATOL,
            rtol=_CROSS_BACKEND_RTOL,
        )
    ):
        raise _invalid("frozen FPO action drift exceeds the v0.4a compatibility envelope")

    compiled_sampler = _compile_action_sampler(native_state, jax, jnp)
    count = min(2, len(observation))
    indices = jnp.asarray(np.arange(count, dtype=np.uint32), dtype=jnp.uint32)
    try:
        row_keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
            indices
        )
        scalar_actions = []
        for index in range(count):
            raw_action, _ = native_state.sample_action(
                device_observation[index], row_keys[index], deterministic=True
            )
            scalar_actions.append(np.asarray(jax.device_get(jnp.tanh(raw_action))))
        scalar_array = np.stack(scalar_actions)
        compiled_first = np.asarray(
            jax.device_get(compiled_sampler(device_observation[:count], row_keys))
        )
        compiled_second = np.asarray(
            jax.device_get(compiled_sampler(device_observation[:count], row_keys))
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise _invalid(f"frozen FPO scalar/compiled parity failed: {exc}") from exc
    if (
        scalar_array.shape != (count, authority.action_dim)
        or compiled_first.shape != scalar_array.shape
        or compiled_second.shape != scalar_array.shape
        or scalar_array.dtype != np.dtype(np.float32)
        or compiled_first.dtype != np.dtype(np.float32)
        or compiled_second.dtype != np.dtype(np.float32)
        or not all(
            np.all(np.isfinite(value))
            for value in (scalar_array, compiled_first, compiled_second)
        )
    ):
        raise _invalid("frozen FPO scalar/compiled parity returned invalid actions")
    compiled_abs, compiled_rel = _max_errors(compiled_first, scalar_array)
    compiled_replay_abs, compiled_replay_rel = _max_errors(
        compiled_first, compiled_second
    )
    if not (
        np.allclose(
            compiled_first,
            scalar_array,
            atol=_SAME_BACKEND_ATOL,
            rtol=_SAME_BACKEND_RTOL,
        )
        and np.allclose(
            compiled_first,
            compiled_second,
            atol=_SAME_BACKEND_ATOL,
            rtol=_SAME_BACKEND_RTOL,
        )
    ):
        raise _invalid("frozen FPO scalar/compiled actions disagree")

    golden_status = "PASS" if exact_golden else "WARNING_CROSS_BACKEND_COMPATIBLE"
    return (
        {
            "status": golden_status,
            "golden_replay": {
                "status": golden_status,
                "sample_count": int(len(observation)),
                "raw_max_abs_error": raw_abs,
                "raw_max_relative_error": raw_rel,
                "environment_max_abs_error": env_abs,
                "environment_max_relative_error": env_rel,
                "exact_atol": _SAME_BACKEND_ATOL,
                "exact_rtol": _SAME_BACKEND_RTOL,
                "compatibility_raw_atol": _CROSS_BACKEND_RAW_ATOL,
                "compatibility_environment_atol": _CROSS_BACKEND_ENV_ATOL,
                "compatibility_rtol": _CROSS_BACKEND_RTOL,
                "evidence": dict(_CROSS_BACKEND_EVIDENCE),
            },
            "same_key_replay": {
                "status": "PASS",
                "raw_exact": bool(np.array_equal(actual_raw, repeated_raw)),
                "environment_exact": bool(
                    np.array_equal(actual_environment, repeated_environment)
                ),
                "raw_max_abs_error": replay_raw_abs,
                "raw_max_relative_error": replay_raw_rel,
                "environment_max_abs_error": replay_env_abs,
                "environment_max_relative_error": replay_env_rel,
                "atol": _SAME_BACKEND_ATOL,
                "rtol": _SAME_BACKEND_RTOL,
            },
            "different_key_sensitivity": {
                "status": "PASS",
                "changed_row_count": different_rows,
                "sample_count": int(len(observation)),
                "raw_max_abs_difference": different_raw_abs,
                "raw_max_relative_difference": different_raw_rel,
                "environment_max_abs_difference": different_env_abs,
                "environment_max_relative_difference": different_env_rel,
            },
            "scalar_compiled": {
                "status": "PASS",
                "sample_count": count,
                "max_abs_error": compiled_abs,
                "max_relative_error": compiled_rel,
                "compiled_replay_exact": bool(
                    np.array_equal(compiled_first, compiled_second)
                ),
                "compiled_replay_max_abs_error": compiled_replay_abs,
                "compiled_replay_max_relative_error": compiled_replay_rel,
                "atol": _SAME_BACKEND_ATOL,
                "rtol": _SAME_BACKEND_RTOL,
            },
            "action_transform": {
                "status": "PASS",
                "max_abs_error": transform_abs,
                "max_relative_error": transform_rel,
                "atol": _SAME_BACKEND_ATOL,
                "rtol": _SAME_BACKEND_RTOL,
            },
        },
        compiled_sampler,
    )


class FrozenFPOActor:
    """Keyed FPO actor whose historical ``deterministic=True`` still samples."""

    semantics = PolicySemantics.STOCHASTIC_KEYED

    def __init__(
        self,
        authority: ActorAuthority,
        *,
        bundle_dir: str | Path,
        fpo_checkout: str | Path,
        policy_repo_checkout: str | Path,
    ) -> None:
        if not isinstance(authority, ActorAuthority):
            raise TypeError("authority must be an ActorAuthority")
        if authority._verification_token is not _ACTOR_AUTHORITY_TOKEN:
            raise _invalid("ActorAuthority must be loaded from its caller-pinned JSON")
        supplied_fpo = _absolute_directory(fpo_checkout, "fpo_checkout")
        verified_fpo = _verify_checkout(
            supplied_fpo,
            expected_commit=authority.fpo_commit,
            expected_tree=authority.fpo_tree,
            name="fpo_checkout",
        )
        fpo_source = _checkout_source_identity(verified_fpo)
        if fpo_source != {
            "head_tree_digest": authority.fpo_head_tree_digest,
            "source_digest": authority.fpo_source_digest,
            "source_file_count": authority.fpo_source_file_count,
        }:
            raise _invalid("live archived FPO source digest differs from actor authority")
        verified_policy_repo = _verify_checkout(
            policy_repo_checkout,
            expected_commit=authority.policy_repo_commit,
            expected_tree=authority.policy_repo_tree,
            name="policy_repo_checkout",
        )
        verified_bundle = _verify_bundle(bundle_dir, authority)
        policy = _load_policy_runtime(
            policy_repo_checkout=verified_policy_repo,
            bundle_dir=verified_bundle,
            fpo_checkout=verified_fpo,
            authority=authority,
        )
        post_fpo = _verify_checkout(
            verified_fpo,
            expected_commit=authority.fpo_commit,
            expected_tree=authority.fpo_tree,
            name="fpo_checkout",
        )
        post_policy_repo = _verify_checkout(
            verified_policy_repo,
            expected_commit=authority.policy_repo_commit,
            expected_tree=authority.policy_repo_tree,
            name="policy_repo_checkout",
        )
        if post_fpo != verified_fpo or post_policy_repo != verified_policy_repo:
            raise _invalid("frozen checkout identity changed during policy load")
        if _checkout_source_identity(post_fpo) != fpo_source:
            raise _invalid("archived FPO source changed during policy load")
        if _verify_bundle(verified_bundle, authority) != verified_bundle:
            raise _invalid("frozen bundle identity changed during policy load")
        native_state = getattr(policy, "native_state", None)
        if native_state is None or not callable(getattr(native_state, "sample_action", None)):
            raise _invalid("frozen loader did not expose native_state.sample_action")
        if (
            getattr(policy, "observation_dim", None) != authority.observation_dim
            or getattr(policy, "action_dim", None) != authority.action_dim
        ):
            raise _invalid("loaded policy dimensions differ from actor authority")
        loaded_digest = getattr(policy, "bundle_digest", None)
        if loaded_digest != authority.bundle_digest:
            raise _invalid("loaded policy bundle digest differs from actor authority")
        parity, compiled_sampler = _initial_actor_parity(
            native_state, verified_bundle, authority
        )
        # Sampling can trigger lazy imports; prove that parity itself left every
        # frozen input unchanged before exposing the provider.
        _verify_checkout(
            verified_fpo,
            expected_commit=authority.fpo_commit,
            expected_tree=authority.fpo_tree,
            name="fpo_checkout",
        )
        _verify_checkout(
            verified_policy_repo,
            expected_commit=authority.policy_repo_commit,
            expected_tree=authority.policy_repo_tree,
            name="policy_repo_checkout",
        )
        if _checkout_source_identity(verified_fpo) != fpo_source:
            raise _invalid("archived FPO source changed during actor parity")
        _verify_bundle(verified_bundle, authority)
        self.policy_id = authority.candidate_id
        self.observation_dim = authority.observation_dim
        self.action_dim = authority.action_dim
        self.authority = authority
        self._bundle_dir = verified_bundle
        self._fpo_checkout = verified_fpo
        self._policy_repo_checkout = verified_policy_repo
        self._fpo_source_identity = MappingProxyType(dict(fpo_source))
        self._native_state = native_state
        self._compiled_sampler = compiled_sampler
        self._parity = parity
        self._last_key_ledger: Mapping[str, Any] | None = None

    @property
    def parity(self) -> Mapping[str, Any]:
        return json.loads(json.dumps(self._parity))

    @property
    def provenance(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "provider_schema": FROZEN_FPO_PROVIDER_SCHEMA,
                "candidate_id": self.policy_id,
                "semantics": PolicySemantics.STOCHASTIC_KEYED.value,
                "historical_deterministic_argument": True,
                "historical_deterministic_argument_is_sampling_free": False,
                "uint64_to_jax_key_data": "[high32,low32]",
                "action_transform": "tanh",
                "authority_sha256": self.authority.authority_sha256,
                "census_sha256": self.authority.census_sha256,
                "bundle_digest": self.authority.bundle_digest,
                "initialization_parity": self.parity,
            }
        )

    def verify_unchanged(self) -> Mapping[str, Any]:
        """Recheck frozen inputs once at runner completion."""

        _verify_checkout(
            self._fpo_checkout,
            expected_commit=self.authority.fpo_commit,
            expected_tree=self.authority.fpo_tree,
            name="fpo_checkout",
        )
        _verify_checkout(
            self._policy_repo_checkout,
            expected_commit=self.authority.policy_repo_commit,
            expected_tree=self.authority.policy_repo_tree,
            name="policy_repo_checkout",
        )
        if _checkout_source_identity(self._fpo_checkout) != dict(
            self._fpo_source_identity
        ):
            raise _invalid("archived FPO source changed after provider construction")
        _verify_bundle(self._bundle_dir, self.authority)
        return MappingProxyType(
            {
                "status": "PASS",
                "candidate_id": self.policy_id,
                "bundle_digest": self.authority.bundle_digest,
                "fpo_commit": self.authority.fpo_commit,
                "fpo_tree": self.authority.fpo_tree,
                "fpo_source_digest": self.authority.fpo_source_digest,
                "policy_repo_commit": self.authority.policy_repo_commit,
                "policy_repo_tree": self.authority.policy_repo_tree,
                "initialization_parity_status": self._parity["status"],
            }
        )

    def _sampler(self) -> tuple[Any, Any, Any]:
        jax, jnp = _jax_modules()
        if self._compiled_sampler is None:
            self._compiled_sampler = _compile_action_sampler(
                self._native_state, jax, jnp
            )
        return self._compiled_sampler, jax, jnp

    @property
    def last_key_ledger(self) -> Mapping[str, Any] | None:
        return self._last_key_ledger

    @staticmethod
    def _key_chain(
        keys: np.ndarray, jax: Any, jnp: Any
    ) -> tuple[np.ndarray, Any, np.ndarray]:
        key_data = _uint64_key_data(keys)
        wrapped = getattr(jax.random, "wrap_key_data", None)
        device_key_data = jnp.asarray(key_data, dtype=jnp.uint32)
        device_keys = wrapped(device_key_data) if wrapped is not None else device_key_data
        try:
            next_keys = jax.vmap(lambda key: jax.random.split(key, 2)[1])(
                device_keys
            )
            next_key_data = np.asarray(
                jax.device_get(jax.random.key_data(next_keys)), dtype=np.uint32
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise _invalid(f"cannot derive the frozen FPO next-key ledger: {exc}") from exc
        if next_key_data.shape != key_data.shape:
            raise _invalid("JAX next-key ledger has an invalid shape")
        next_key_data.setflags(write=False)
        return key_data, device_keys, next_key_data

    def next_action_keys(self, keys: np.ndarray) -> np.ndarray:
        """Return ``split(key, 2)[1]`` as explicit uint32 ``[N,2]`` data."""

        jax, jnp = _jax_modules()
        _, _, next_key_data = self._key_chain(keys, jax, jnp)
        return next_key_data

    def sample_actions(
        self,
        observations: np.ndarray,
        native_timestep: np.ndarray,
        *,
        keys: np.ndarray,
    ) -> np.ndarray:
        observations_array = np.asarray(observations)
        times = np.asarray(native_timestep)
        if (
            observations_array.ndim != 2
            or observations_array.shape[1] != self.observation_dim
            or observations_array.dtype.kind not in "fiu"
            or not np.all(np.isfinite(observations_array))
        ):
            raise _invalid("candidate observations differ from the frozen FPO ABI")
        if (
            times.ndim != 1
            or len(times) != len(observations_array)
            or times.dtype.kind not in "iu"
            or (times.dtype.kind == "i" and np.any(times < 0))
        ):
            raise _invalid("candidate native_timestep is malformed")
        sampler, jax, jnp = self._sampler()
        key_data, device_keys, next_key_data = self._key_chain(keys, jax, jnp)
        if len(key_data) != len(observations_array):
            raise DataValidationError(
                EstimateStatus.NO_GO_TARGET_POLICY_SEMANTICS.value,
                "FrozenFPOActor key count differs from observations",
            )
        try:
            actions = np.asarray(
                jax.device_get(
                    sampler(
                        jnp.asarray(observations_array, dtype=jnp.float32),
                        device_keys,
                    )
                )
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _invalid(f"frozen FPO action sampling failed: {exc}") from exc
        if (
            actions.shape != (len(observations_array), self.action_dim)
            or not np.all(np.isfinite(actions))
            or np.any(actions < -1.0)
            or np.any(actions > 1.0)
        ):
            raise _invalid("frozen FPO sampler returned an invalid action tensor")
        ledger = {
            "mapping": "scalar_u64->[high32,low32];structured_words=checked_uint32",
            "input_key_data": key_data.tolist(),
            "next_key_data": next_key_data.tolist(),
            "transition": "jax.random.split(key,2)[1]",
        }
        self._last_key_ledger = MappingProxyType(ledger)
        return actions.astype(np.float64, copy=False)
