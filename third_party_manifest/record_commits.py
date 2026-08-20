from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tarca.stage0.sources import load_sources_manifest  # noqa: E402


def main() -> int:
    manifest = load_sources_manifest(REPO_ROOT / "third_party_manifest/sources.yaml")
    summary = {
        "schema_version": manifest.schema_version,
        "verification_date": manifest.verification_date.isoformat(),
        "sources": [
            {
                "source_id": source.source_id,
                "repository_url": source.repository_url,
                "default_branch": source.default_branch,
                "commit": source.commit,
                "package_name": source.package_name,
                "package_version": source.package_version,
                "release_tag": source.release_tag,
                "release_commit": source.release_commit,
                "license_status": source.license_status,
                "allowed_action": source.allowed_action,
            }
            for source in manifest.sources
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
