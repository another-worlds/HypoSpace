from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from diagnostics import (
    DIAGNOSTICS_VERSION,
    DiagnosticsReport,
    run_diagnostics,
)

_EXPECTED_SUBSYSTEMS = {
    "config",
    "preprocessor",
    "hierarchy",
    "kernel_library",
    "extractor",
    "semantic",
    "mechanistic",
    "full_pipeline",
    "nnsight",
    "pyvene",
}

_VALID_STATUSES = {"ok", "degraded", "failed", "skipped"}


@pytest.fixture(scope="module")
def report() -> DiagnosticsReport:
    return run_diagnostics()


def test_run_diagnostics_returns_report(report):
    assert isinstance(report, DiagnosticsReport)
    assert report.schema_version == DIAGNOSTICS_VERSION
    assert report.overall_status in {"ok", "degraded", "failed"}


def test_report_is_json_serializable(report):
    raw = json.dumps(dataclasses.asdict(report), ensure_ascii=False)
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert "probes" in parsed
    assert "dependencies" in parsed


def test_all_probes_present(report):
    found = {p.subsystem for p in report.probes}
    assert found == _EXPECTED_SUBSYSTEMS


def test_probe_statuses_are_valid_values(report):
    for probe in report.probes:
        assert probe.status in _VALID_STATUSES, f"{probe.subsystem} has invalid status '{probe.status}'"


def test_latency_ms_nonnegative(report):
    for probe in report.probes:
        assert probe.latency_ms >= 0.0, f"{probe.subsystem} has negative latency"


def test_no_uncaught_exception():
    # run_diagnostics() must never raise
    run_diagnostics()


def test_dependencies_all_present_in_report(report):
    found = {d.name for d in report.dependencies}
    assert found == {"torch", "nnsight", "pyvene", "diskcache"}


def test_dependency_status_is_bool(report):
    for dep in report.dependencies:
        assert isinstance(dep.available, bool), f"{dep.name}.available is not bool"


def _get_probe(report: DiagnosticsReport, subsystem: str):
    return next(p for p in report.probes if p.subsystem == subsystem)


def test_config_probe_passes(report):
    probe = _get_probe(report, "config")
    assert probe.status == "ok", f"config probe errors: {probe.errors}"


def test_preprocessor_probe_passes(report):
    probe = _get_probe(report, "preprocessor")
    assert probe.status == "ok", f"preprocessor probe errors: {probe.errors}"


def test_hierarchy_probe_degraded_not_failed(report):
    # hierarchy always emits a warning about top_k=0 silent promotion → degraded
    probe = _get_probe(report, "hierarchy")
    assert probe.status in {"ok", "degraded"}, f"hierarchy probe errors: {probe.errors}"
    assert probe.errors == [], f"hierarchy probe should have no errors: {probe.errors}"


def test_kernel_library_probe_passes(report):
    probe = _get_probe(report, "kernel_library")
    assert probe.status in {"ok", "degraded"}, f"kernel_library probe errors: {probe.errors}"
    assert probe.errors == [], f"kernel_library probe should have no errors: {probe.errors}"


def test_extractor_probe_passes(report):
    probe = _get_probe(report, "extractor")
    assert probe.status in {"ok", "degraded"}, f"extractor probe errors: {probe.errors}"
    assert probe.errors == [], f"extractor probe should have no errors: {probe.errors}"


def test_semantic_probe_passes(report):
    probe = _get_probe(report, "semantic")
    assert probe.status == "ok", f"semantic probe errors: {probe.errors}"


def test_mechanistic_probe_is_degraded(report):
    # mechanistic always emits the stub warning → degraded
    probe = _get_probe(report, "mechanistic")
    assert probe.status == "degraded", f"mechanistic probe status={probe.status}, errors={probe.errors}"
    assert probe.errors == [], f"mechanistic probe should have no errors: {probe.errors}"
    assert any("stub" in w for w in probe.warnings)


def test_full_pipeline_probe_passes(report):
    probe = _get_probe(report, "full_pipeline")
    assert probe.status == "ok", f"full_pipeline probe errors: {probe.errors}"


def test_nnsight_probe_skipped_when_absent(report):
    probe = _get_probe(report, "nnsight")
    try:
        import nnsight  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        assert probe.status == "skipped"


def test_pyvene_probe_skipped_when_absent(report):
    probe = _get_probe(report, "pyvene")
    try:
        import pyvene  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        assert probe.status == "skipped"


def test_overall_status_degraded_not_failed(report):
    # mechanistic always warns → overall at least "degraded", never "failed" on a working install
    assert report.overall_status in {"ok", "degraded"}, f"overall_status unexpectedly 'failed': check probe errors"


def test_api_diagnostics_method():
    from api import HypoSpaceAPI
    api = HypoSpaceAPI()
    result = api.diagnostics()
    assert isinstance(result, DiagnosticsReport)


def test_check_details_have_required_fields(report):
    for probe in report.probes:
        for check in probe.checks:
            assert check.name, f"{probe.subsystem} has a CheckDetail with empty name"
            assert isinstance(check.passed, bool), f"{probe.subsystem}/{check.name}: passed is not bool"


@pytest.mark.parametrize("subsystem", [
    "config", "preprocessor", "hierarchy", "kernel_library",
    "extractor", "semantic", "mechanistic", "full_pipeline",
])
def test_non_optional_probe_not_skipped(subsystem, report):
    probe = _get_probe(report, subsystem)
    assert probe.status != "skipped", f"{subsystem} probe should never be skipped"


def test_cli_diagnostics_flag():
    result = subprocess.run(
        [sys.executable, "main.py", "--diagnostics"],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
    )
    assert result.returncode == 0, f"--diagnostics exited {result.returncode}: {result.stderr}"
    parsed = json.loads(result.stdout)
    assert "overall_status" in parsed
    assert "probes" in parsed
