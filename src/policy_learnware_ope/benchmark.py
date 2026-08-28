"""Oracle-blind ranking seals and post-seal OPE/selection metrics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _assert_oracle_free(value: Any, *, path: str = "provenance") -> None:
    forbidden = ("oracle", "true_value", "target_return", "episode_return")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in forbidden):
                raise ValueError(f"oracle-bearing field cannot enter pre-oracle seal: {path}.{key}")
            _assert_oracle_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_oracle_free(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class RankingSeal:
    path: Path
    digest: str
    payload: Mapping[str, Any]


def seal_ranking(
    path: str | Path,
    *,
    method_id: str,
    context_id: str,
    score_kind: str,
    scores: Mapping[str, float | None],
    statuses: Mapping[str, str] | None = None,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any],
    higher_is_better: bool = True,
    value_convention: str = "J_gamma=0.99_H=1000_raw",
) -> RankingSeal:
    """Seal estimates without accepting an oracle argument or oracle fields."""

    if score_kind not in {"value", "compatibility"}:
        raise ValueError("score_kind must be value or compatibility")
    if not method_id or not context_id or not scores:
        raise ValueError("method_id, context_id, and scores are required")
    _assert_oracle_free(provenance)
    statuses = dict(statuses or {})
    diagnostics = dict(diagnostics or {})
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(scores):
        raw_score = scores[candidate_id]
        score = None if raw_score is None else float(raw_score)
        if score is not None and not np.isfinite(score):
            raise ValueError(f"non-finite score for {candidate_id}")
        status = statuses.get(candidate_id, "PASS" if score is not None else "INCOMPLETE")
        if score is None and status == "PASS":
            raise ValueError("PASS rows require a finite score")
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "score": score,
                "status": str(status),
                "diagnostics": _json_safe(diagnostics.get(candidate_id, {})),
            }
        )
    successful = [row for row in rows if row["score"] is not None and row["status"] == "PASS"]
    successful.sort(
        key=lambda row: (
            -row["score"] if higher_is_better else row["score"],
            row["candidate_id"],
        )
    )
    payload: dict[str, Any] = {
        "schema": "policy-learnware.ranking-seal.v1",
        "method_id": method_id,
        "context_id": context_id,
        "score_kind": score_kind,
        "value_convention": str(value_convention) if score_kind == "value" else None,
        "higher_is_better": bool(higher_is_better),
        "ranking": [row["candidate_id"] for row in successful],
        "selected_candidate_id": successful[0]["candidate_id"] if successful else None,
        "rows": rows,
        "provenance": _json_safe(dict(provenance)),
    }
    digest = _payload_digest(payload)
    envelope = {"payload": payload, "sha256": digest}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_bytes(envelope))
    return RankingSeal(destination, digest, payload)


def load_ranking_seal(path: str | Path) -> RankingSeal:
    source = Path(path)
    envelope = json.loads(source.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    digest = envelope.get("sha256")
    if not isinstance(payload, dict) or digest != _payload_digest(payload):
        raise ValueError("ranking seal digest mismatch")
    return RankingSeal(source, digest, payload)


def _average_ranks(values: Sequence[float], *, descending: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array if descending else array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    position = 0
    while position < len(array):
        stop = position + 1
        while stop < len(array) and array[order[stop]] == array[order[position]]:
            stop += 1
        ranks[order[position:stop]] = (position + 1 + stop) / 2.0
        position = stop
    return ranks


def _rank_correlations(estimated: Sequence[float], oracle: Sequence[float]) -> tuple[float | None, float | None]:
    if len(estimated) < 2:
        return None, None
    est_ranks = _average_ranks(estimated, descending=True)
    oracle_ranks = _average_ranks(oracle, descending=True)
    if np.std(est_ranks) == 0 or np.std(oracle_ranks) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(est_ranks, oracle_ranks)[0, 1])
    concordant = discordant = est_ties = oracle_ties = 0
    for left in range(len(estimated)):
        for right in range(left + 1, len(estimated)):
            est_delta = np.sign(estimated[left] - estimated[right])
            oracle_delta = np.sign(oracle[left] - oracle[right])
            if est_delta == 0 and oracle_delta == 0:
                continue
            if est_delta == 0:
                est_ties += 1
            elif oracle_delta == 0:
                oracle_ties += 1
            elif est_delta == oracle_delta:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + est_ties) * (concordant + discordant + oracle_ties)
    )
    kendall = None if denominator == 0 else float((concordant - discordant) / denominator)
    return spearman, kendall


def join_oracle_and_score(
    seal_path: str | Path,
    *,
    oracle_values: Mapping[str, float],
    secondary_values: Mapping[str, float] | None = None,
    winner_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Join oracle values only after verifying the immutable ranking seal."""

    seal = load_ranking_seal(seal_path)
    rows = list(seal.payload["rows"])
    candidate_ids = [row["candidate_id"] for row in rows]
    if set(candidate_ids) != set(oracle_values):
        raise ValueError("oracle candidate set differs from sealed candidate set")
    oracle = {key: float(value) for key, value in oracle_values.items()}
    if not np.isfinite(list(oracle.values())).all():
        raise ValueError("oracle values must be finite")
    best_value = max(oracle.values())
    winners = sorted(
        key for key, value in oracle.items() if best_value - value <= float(winner_tolerance)
    )
    selected = seal.payload["selected_candidate_id"]
    successful = [row for row in rows if row["status"] == "PASS" and row["score"] is not None]
    common_ids = [row["candidate_id"] for row in successful]
    estimated = [float(row["score"]) for row in successful]
    oracle_common = [oracle[candidate_id] for candidate_id in common_ids]
    oriented_estimated = (
        estimated if seal.payload["higher_is_better"] else [-value for value in estimated]
    )
    spearman, kendall = _rank_correlations(oriented_estimated, oracle_common)
    if seal.payload["score_kind"] == "value" and common_ids:
        errors = np.asarray(estimated) - np.asarray(oracle_common)
        value_mae: float | None = float(np.mean(np.abs(errors)))
        value_rmse: float | None = float(np.sqrt(np.mean(errors**2)))
        value_bias: float | None = float(np.mean(errors))
    else:
        value_mae = value_rmse = value_bias = None
    diagnostics = [row.get("diagnostics", {}) for row in rows]
    ess_values = [float(item["ess"]) for item in diagnostics if item.get("ess") is not None]
    runtimes = [
        float(item["runtime_seconds"])
        for item in diagnostics
        if item.get("runtime_seconds") is not None
    ]
    selected_secondary = None
    if secondary_values is not None and selected is not None:
        if set(secondary_values) != set(candidate_ids):
            raise ValueError("secondary-value candidate set differs from seal")
        selected_secondary = float(secondary_values[selected])
    value_convention = seal.payload.get("value_convention")
    selected_primary = None if selected is None else oracle[selected]
    return {
        "schema": "policy-learnware.oracle-join.v1",
        "ranking_seal_sha256": seal.digest,
        "method_id": seal.payload["method_id"],
        "context_id": seal.payload["context_id"],
        "score_kind": seal.payload["score_kind"],
        "value_convention": value_convention,
        "selected_candidate_id": selected,
        "oracle_winner_ids": winners,
        "hit_at_1": None if selected is None else int(selected in winners),
        "regret_at_1": None if selected is None else float(best_value - oracle[selected]),
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "value_mae": value_mae,
        "value_rmse": value_rmse,
        "value_bias": value_bias,
        "selected_primary_value": selected_primary,
        "selected_J_gamma_099_H1000": (
            selected_primary
            if value_convention == "J_gamma=0.99_H=1000_raw"
            else None
        ),
        "selected_J_norm": selected_secondary,
        "ess_min": min(ess_values) if ess_values else None,
        "ess_mean": float(np.mean(ess_values)) if ess_values else None,
        "runtime_seconds": float(sum(runtimes)),
        "successful_candidates": len(successful),
        "candidate_count": len(rows),
        "status_coverage": float(len(successful) / len(rows)),
        "failures": {
            row["candidate_id"]: row["status"] for row in rows if row["status"] != "PASS"
        },
        "provenance": seal.payload["provenance"],
    }


def export_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, str]:
    """Write stable JSON and a compact flat summary CSV."""

    safe_records = [_json_safe(dict(record)) for record in records]
    json_payload = {"schema": "policy-learnware.metrics.v1", "records": safe_records}
    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_bytes(_canonical_bytes(json_payload))
    columns = [
        "method_id",
        "context_id",
        "score_kind",
        "value_convention",
        "value_mae",
        "value_rmse",
        "value_bias",
        "hit_at_1",
        "regret_at_1",
        "spearman",
        "kendall_tau_b",
        "ess_min",
        "ess_mean",
        "runtime_seconds",
        "status_coverage",
        "ranking_seal_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in safe_records:
        writer.writerow({key: "" if record.get(key) is None else record.get(key) for key in columns})
    csv_destination.write_text(buffer.getvalue(), encoding="utf-8")
    return {
        "json": str(json_destination),
        "json_sha256": sha256(json_destination.read_bytes()).hexdigest(),
        "csv": str(csv_destination),
        "csv_sha256": sha256(csv_destination.read_bytes()).hexdigest(),
    }


__all__ = [
    "RankingSeal",
    "export_metrics",
    "join_oracle_and_score",
    "load_ranking_seal",
    "seal_ranking",
]
