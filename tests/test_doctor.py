from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scripts.doctor as doctor_cli  # noqa: E402

import tarca.stage0.diagnostics as diagnostics  # noqa: E402
from tarca.stage0.models import CheckResult, DoctorReport  # noqa: E402


def _passing_report() -> DoctorReport:
    return DoctorReport(
        results=(
            CheckResult(
                name="test.component",
                status="PASS",
                details={"nested": {"values": [1, 2]}},
            ),
        )
    )


def test_report_models_are_deeply_immutable_and_serialize_to_new_plain_objects() -> None:
    result = _passing_report().results[0]

    with pytest.raises(TypeError):
        result.details["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.details["nested"]["new"] = "value"  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.details["nested"]["values"].append(3)  # type: ignore[union-attr]

    first = _passing_report().to_dict()
    second = _passing_report().to_dict()
    first["results"][0]["details"]["nested"]["values"].append(3)

    assert first != second
    assert second["schema_version"] == "1.0"
    json.dumps(second)


def test_system_checks_cover_required_components_and_clean_write_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = set(PROJECT_ROOT.glob(".tarca-doctor-*"))
    monkeypatch.setattr(diagnostics.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diagnostics.torch.cuda, "device_count", lambda: 0)

    results = diagnostics.check_system(PROJECT_ROOT)
    by_name = {result.name: result for result in results}

    assert {
        "system.os",
        "system.python",
        "system.cpu",
        "system.memory",
        "system.gpu",
        "system.cuda",
        "system.disk",
        "system.project_write",
        "system.git",
        "system.uv",
    } <= by_name.keys()
    assert by_name["system.gpu"].status == "SKIP"
    assert by_name["system.cuda"].status == "SKIP"
    assert by_name["system.project_write"].status == "PASS"
    assert set(PROJECT_ROOT.glob(".tarca-doctor-*")) == before


def test_system_check_isolates_one_component_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_memory() -> CheckResult:
        raise OSError("forced memory probe failure")

    monkeypatch.setattr(diagnostics, "_check_memory", broken_memory)

    by_name = {result.name: result for result in diagnostics.check_system(PROJECT_ROOT)}

    assert by_name["system.memory"].status == "FAIL"
    assert by_name["system.memory"].details["exception_type"] == "OSError"
    assert by_name["system.cpu"].status == "PASS"
    assert by_name["system.disk"].status == "PASS"


def test_numeric_checks_cover_dtypes_cpu_matmul_and_nonfinite_detection() -> None:
    by_name = {result.name: result for result in diagnostics.check_numeric()}

    assert {
        "numeric.numpy.float32",
        "numeric.numpy.float64",
        "numeric.torch.float32",
        "numeric.torch.float64",
        "numeric.cpu_matmul",
        "numeric.finite_detection",
    } == by_name.keys()
    assert all(result.status == "PASS" for result in by_name.values())
    assert by_name["numeric.cpu_matmul"].details["device"] == "cpu"
    assert by_name["numeric.finite_detection"].details["numpy_detected"] is True
    assert by_name["numeric.finite_detection"].details["torch_detected"] is True


def test_run_diagnostics_isolates_component_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_pot() -> CheckResult:
        raise RuntimeError("forced POT failure")

    monkeypatch.setattr(diagnostics, "check_pot", broken_pot)
    report = diagnostics.run_diagnostics(PROJECT_ROOT)
    by_name = {result.name: result for result in report.results}

    failure = by_name["numeric.pot"]
    assert failure.status == "FAIL"
    assert failure.details["exception_type"] == "RuntimeError"
    assert failure.details["message"] == "forced POT failure"
    assert "pytest tests/test_pot_smoke.py" in (failure.remediation or "")
    assert "numeric.torch_hook" in by_name


def test_project_write_failure_preserves_original_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_probe(*_args: object, **_kwargs: object) -> object:
        raise PermissionError("write probe denied")

    monkeypatch.setattr(diagnostics.tempfile, "NamedTemporaryFile", denied_probe)

    result = diagnostics._safe_result(
        "system.project_write",
        "Check project permissions.",
        lambda: diagnostics._check_project_write(PROJECT_ROOT),
    )

    assert result.status == "FAIL"
    assert result.details["exception_type"] == "PermissionError"
    assert result.details["message"] == "write probe denied"
    assert result.remediation == "Check project permissions."


def test_pyvene_import_failure_is_a_core_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyvene", None)

    result = diagnostics.check_pyvene()

    assert result.status == "FAIL"
    assert result.details["exception_type"] == "ModuleNotFoundError"
    assert result.remediation is not None


def test_pyvene_import_notices_are_captured_instead_of_polluting_cli_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    class FakeIntervention:
        def __init__(self, *, embed_dim: int) -> None:
            assert embed_dim == 3

        def __call__(self, _base: object, source: object) -> object:
            return source

    fake_pyvene = SimpleNamespace(
        VanillaIntervention=FakeIntervention,
        IntervenableConfig=object,
        IntervenableModel=object,
    )

    def noisy_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "pyvene":
            print("optional stdout notice")
            print("optional stderr notice", file=sys.stderr)
            return fake_pyvene
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.delitem(sys.modules, "pyvene", raising=False)
    monkeypatch.setattr(builtins, "__import__", noisy_import)
    monkeypatch.setattr(diagnostics.importlib.metadata, "version", lambda _name: "test")

    result = diagnostics.check_pyvene()
    captured = capsys.readouterr()

    assert result.status == "PASS"
    assert captured.out == ""
    assert captured.err == ""
    assert result.details["import_notices"] == {
        "stdout": "optional stdout notice",
        "stderr": "optional stderr notice",
    }


def test_renderers_include_schema_details_and_remediation() -> None:
    report = DoctorReport(
        results=(
            CheckResult(
                name="numeric.forced",
                status="FAIL",
                details={"exception_type": "ValueError", "message": "bad value"},
                remediation="Run the focused numeric check.",
            ),
        )
    )

    rendered = diagnostics.render_markdown(report)
    payload = json.loads(diagnostics.report_to_json(report))

    assert "# TARCA Stage 0 Doctor Report" in rendered
    assert "Overall status: **FAIL**" in rendered
    assert "numeric.forced" in rendered
    assert '"exception_type": "ValueError"' in rendered
    assert "Run the focused numeric check." in rendered
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["FAIL"] == 1
    assert payload["results"][0]["remediation"] == "Run the focused numeric check."


def test_doctor_cli_writes_json_and_markdown_atomically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    json_path = tmp_path / "nested" / "doctor.json"
    markdown_path = tmp_path / "nested" / "doctor.md"

    exit_code = doctor_cli.main(
        ["--json", str(json_path), "--markdown", str(markdown_path)],
        runner=lambda _root: _passing_report(),
    )

    assert exit_code == 0
    assert "Overall status: **PASS**" in capsys.readouterr().out
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert "test.component" in markdown_path.read_text(encoding="utf-8")
    assert not list((tmp_path / "nested").glob(".tarca-doctor-*"))


def test_doctor_cli_supports_summary_only_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = doctor_cli.main([], runner=lambda _root: _passing_report())

    assert exit_code == 0
    assert "test.component" in capsys.readouterr().out


def test_doctor_cli_returns_one_for_forced_core_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = DoctorReport(
        results=(
            CheckResult(
                name="system.forced",
                status="FAIL",
                details={"exception_type": "OSError", "message": "forced"},
                remediation="Restore the required system component.",
            ),
        )
    )

    exit_code = doctor_cli.main(
        ["--json", str(tmp_path / "failed.json")],
        runner=lambda _root: report,
    )

    assert exit_code == 1
    assert "system.forced" in capsys.readouterr().out
    assert (
        json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))["summary"]["FAIL"] == 1
    )
