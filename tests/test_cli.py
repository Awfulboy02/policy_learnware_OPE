from __future__ import annotations

import json
from pathlib import Path

from policy_learnware_ope.benchmark import load_ranking_seal
from policy_learnware_ope.cli import main, run_toy


EXPECTED_METHODS = {
    "FH_FQE_G099_H1000",
    "FH_KMIFQE_G099_H1000",
    "ETM_MBOPE_G099_H1000",
    "DOPE_STYLE_MB_FF_G099_H1000",
    "AR_MBOPE_G099_H1000",
    "RAW_DELTA_TASK5",
}


def test_toy_runner_fits_all_methods_seals_then_exports_metrics(tmp_path: Path):
    result = run_toy(tmp_path / "toy", seed=17)

    assert result["status"] == "TOY_MVP_PASS"
    assert set(result["method_scope"]) == EXPECTED_METHODS
    assert result["real_asset_training_started"] is False
    assert len(result["metrics"]) == len(EXPECTED_METHODS)
    for method_id, seal_path in result["ranking_seals"].items():
        seal = load_ranking_seal(seal_path)
        assert seal.payload["method_id"] == method_id
        assert "oracle_values" not in seal.payload
        assert len(seal.payload["ranking"]) == 5
    raw_metrics = next(row for row in result["metrics"] if row["method_id"] == "RAW_DELTA_TASK5")
    assert raw_metrics["value_mae"] is None
    assert raw_metrics["value_rmse"] is None
    for method_id, rows in result["estimates"].items():
        assert len(rows) == 5
        assert {row["status"] for row in rows.values()} == {"PASS"}, method_id
    assert Path(result["artifacts"]["json"]).is_file()
    assert Path(result["artifacts"]["csv"]).is_file()


def test_cli_smoke_prints_machine_readable_summary(tmp_path: Path, capsys):
    output = tmp_path / "cli-toy"
    assert main(["toy", "--output", str(output), "--seed", "19"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "TOY_MVP_PASS"
    assert set(printed["methods"]) == EXPECTED_METHODS
    assert Path(printed["run"]).is_file()
