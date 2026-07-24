"""Contracts for the approved public/local publication boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"
README_PATH = PROJECT_ROOT / "README.md"
STAGE0_SCOPE_PATH = PROJECT_ROOT / "docs" / "stage0_scope.md"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "stage0" / "STAGE0_IMPLEMENTATION_REPORT.md"
CURATED_EVIDENCE = (
    "artifacts/stage0/STAGE0_IMPLEMENTATION_REPORT.md",
    "artifacts/stage0/third_party_commits.json",
    "artifacts/stage0/reference_smoke/plot/result_summary.md",
    "artifacts/stage0/reference_smoke/diroca/result_summary.md",
)
REPRESENTATIVE_LOCAL_ONLY_PATHS = (
    ".superpowers/sdd/internal-plan.md",
    "docs/superpowers/plans/internal-plan.md",
    "docs/MANUAL_VERIFICATION_STAGE0.md",
    "artifacts/stage0/command_log.json",
    "artifacts/stage0/unapproved-evidence.txt",
    "artifacts/stage0/reference_smoke/plot/status.json",
    "artifacts/stage0/reference_smoke/plot/unapproved-evidence.txt",
    "artifacts/stage0/reference_smoke/diroca/unapproved-evidence.txt",
)
REQUIRED_IGNORE_RULES = (
    ".superpowers/",
    "docs/superpowers/",
    "docs/MANUAL_VERIFICATION_STAGE0.md",
    "artifacts/stage0/*",
)
ALLOWED_UNIGNORE_PARENT_DIRECTORIES = frozenset(
    {
        "artifacts/",
        "artifacts/stage0/",
        "artifacts/stage0/reference_smoke/",
        "artifacts/stage0/reference_smoke/plot/",
        "artifacts/stage0/reference_smoke/diroca/",
    }
)
PUBLIC_TEXT_FILES = (
    README_PATH,
    STAGE0_SCOPE_PATH,
    REPORT_PATH,
    PROJECT_ROOT / "Makefile",
    *(PROJECT_ROOT / path for path in CURATED_EVIDENCE[1:]),
)
MACHINE_FINGERPRINTS = (
    r"C:\Users\DELL",
    "C:/Users/DELL",
    r"AppData\Local\Temp",
    "AppData/Local/Temp",
    r"D:\software\MyAnaconda\envs\tarca-stage0",
    "D:/software/MyAnaconda/envs/tarca-stage0",
)


def _check_ignored(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "check-ignore", "--no-index", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_gitignore_excludes_local_release_material_and_raw_stage0_evidence() -> None:
    ignored = set(GITIGNORE_PATH.read_text(encoding="utf-8").splitlines())

    assert set(REQUIRED_IGNORE_RULES) <= ignored, (
        "The publication boundary must ignore internal workflow material, the local manual, "
        "and raw Stage 0 evidence."
    )


def test_gitignore_behaviorally_keeps_only_curated_evidence_public() -> None:
    for path in CURATED_EVIDENCE:
        result = _check_ignored(path)
        assert result.returncode == 1, (
            f"Curated public evidence must not be ignored: {path}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )

    for path in REPRESENTATIVE_LOCAL_ONLY_PATHS:
        result = _check_ignored(path)
        assert result.returncode == 0, (
            f"Local-only material must be ignored: {path}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )


def test_gitignore_unignores_exactly_four_curated_evidence_files() -> None:
    unignore_rules = {
        line[1:].lstrip("/")
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith(("!artifacts/", "!/artifacts/"))
    }
    expected_files = set(CURATED_EVIDENCE)
    unexpected_files = unignore_rules - expected_files - ALLOWED_UNIGNORE_PARENT_DIRECTORIES

    assert expected_files <= unignore_rules, (
        "Each approved curated evidence file must have an explicit unignore rule."
    )
    assert not unexpected_files, (
        "Only the four curated evidence files may be unignored; parent-directory "
        f"exceptions are permitted. Unexpected rules: {sorted(unexpected_files)}"
    )


def test_public_text_does_not_contain_machine_fingerprints() -> None:
    for path in PUBLIC_TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        for fingerprint in MACHINE_FINGERPRINTS:
            assert fingerprint not in text, f"{path} leaks machine fingerprint: {fingerprint}"


def test_public_release_does_not_add_a_license_file() -> None:
    license_files = [
        path
        for path in PROJECT_ROOT.iterdir()
        if path.is_file()
        and (path.name.lower() == "license" or path.name.lower().startswith("license."))
    ]
    assert not license_files, f"No LICENSE file is approved for this release: {license_files}"
