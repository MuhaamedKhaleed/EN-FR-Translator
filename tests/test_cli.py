import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from enfr_translator.cli import app

runner = CliRunner()


def test_no_implicit_ml_import() -> None:
    code = "import sys;import enfr_translator.cli;assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["translate", "--help"]).exit_code == 0


def test_config_data(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("source,target\nHello,Bonjour\nThanks,Merci\nBye,Au revoir\n")
    config = tmp_path / ".env"
    config.write_text(f"PARALLEL_CSV={path}\n")
    result = runner.invoke(app, ["check-data", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "pairs": 3,
        "train_pairs": 2,
        "validation_pairs": 1,
    }
    config.write_text("PARALLEL_CSV=/missing\n")
    assert (
        runner.invoke(
            app, ["check-data", "--data", str(path), "--config", str(config)]
        ).exit_code
        == 0
    )


def test_expected_errors(tmp_path: Path) -> None:
    for args in [
        [],
        ["--config", str(tmp_path / "missing")],
        ["--data", str(tmp_path / "missing")],
    ]:
        result = runner.invoke(app, ["check-data", *args])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
    config = tmp_path / ".env"
    config.write_text("OTHER=none\n")
    assert runner.invoke(app, ["check-data", "--config", str(config)]).exit_code == 2
    assert runner.invoke(app, ["translate", str(tmp_path), "hello"]).exit_code == 2
