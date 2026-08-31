from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tarca.stage1b.source_capsules import SourceCapsuleReceipt, SourceCapsuleSource
from tarca.stage2.config import load_stage2_config
from tarca.stage2.sources import build_stage2_source_capsule

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_stage2_capsule_builder_passes_exact_four_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[tuple[str, ...]] = []

    def fake_build(sources, cache_root, output_path, runner):  # type: ignore[no-untyped-def]
        captured.append(tuple(source.source_id for source in sources))
        entries = tuple(
            SourceCapsuleSource(
                source_id=source.source_id,
                repository_url=source.repository_url,
                commit=source.commit,
                authorization_id=source.authorization_id,
                tree_sha256="a" * 64,
                asset_sha256=tuple(
                    (asset.relative_path, asset.sha256) for asset in source.assets
                ),
                bundle_path=f"bundles/{source.source_id}.bundle",
                bundle_sha256="b" * 64,
            )
            for source in sources
        )
        return SourceCapsuleReceipt("2.0.0", "c" * 64, "d" * 64, entries)

    monkeypatch.setattr("tarca.stage2.sources.build_source_capsule", fake_build)
    config = load_stage2_config(Path("configs/stage2/stage2_v1.yaml"))

    receipt = build_stage2_source_capsule(
        config,
        tmp_path / "cache",
        tmp_path / "stage2-v1-official-sources.tar.gz",
    )

    assert captured == [
        ("dlinear", "itransformer", "patchtst", "scoring_rules_l96")
    ]
    assert tuple(item.source_id for item in receipt.sources) == captured[0]


def test_stage2_source_clis_expose_explicit_offline_transfer_options() -> None:
    expected = {
        "materialize_stage2_sources.py": ("--config", "--cache-root", "--source-id"),
        "package_stage2_source_capsule.py": ("--config", "--cache-root", "--output"),
        "import_stage2_source_capsule.py": (
            "--config",
            "--cache-root",
            "--capsule",
            "--receipt",
        ),
    }
    for script_name, options in expected.items():
        completed = subprocess.run(
            (sys.executable, str(REPOSITORY_ROOT / "scripts" / script_name), "--help"),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert all(option in completed.stdout for option in options)

