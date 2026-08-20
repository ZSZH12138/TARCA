from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tarca.stage0.sources import (  # noqa: E402
    ThirdPartySource,
    audit_dependency_bindings,
    load_sources_manifest,
    verify_remote_release_bindings,
)


def _github_coordinates(repository_url: str) -> tuple[str, str]:
    owner, repository = repository_url.removeprefix("https://github.com/").split("/", 1)
    return owner, repository


def _check_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "TARCA-Stage0/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}: {url}")
    return url


def _resolve_release_tag(source: ThirdPartySource) -> str:
    if source.release_tag is None:
        raise ValueError(f"source has no release tag: {source.source_id}")
    repository = f"{source.repository_url}.git"
    direct_ref = f"refs/tags/{source.release_tag}"
    peeled_ref = f"{direct_ref}^{{}}"
    completed = subprocess.run(
        ["git", "ls-remote", repository, direct_ref, peeled_ref],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    refs = {
        ref: commit.lower()
        for line in completed.stdout.splitlines()
        if len(parts := line.split()) == 2
        for commit, ref in (parts,)
    }
    resolved = refs.get(peeled_ref, refs.get(direct_ref, ""))
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"release tag could not be resolved: {source.source_id}")
    return resolved


def _remote_urls(manifest: object) -> list[str]:
    urls: list[str] = []
    for source in manifest.sources:  # type: ignore[attr-defined]
        owner, repository = _github_coordinates(source.repository_url)
        urls.append(f"https://github.com/{owner}/{repository}/commit/{source.commit}")
        if source.license_status != "UNKNOWN":
            urls.append(
                f"https://raw.githubusercontent.com/{owner}/{repository}/"
                f"{source.commit}/{source.license_file}"
            )
        if source.release_tag is not None:
            urls.append(f"https://github.com/{owner}/{repository}/tree/{source.release_tag}")
            urls.append(f"https://github.com/{owner}/{repository}/commit/{source.release_commit}")
    return sorted(set(urls))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen third-party source references.")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Verify declared commits, release tags, and license files against GitHub.",
    )
    args = parser.parse_args()
    try:
        manifest = load_sources_manifest(REPO_ROOT / "third_party_manifest/sources.yaml")
        binding_summary = audit_dependency_bindings(REPO_ROOT, manifest)
        remote_urls = _remote_urls(manifest) if args.network else []
        if remote_urls:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(_check_url, remote_urls))
        remote_release_binding_count = (
            verify_remote_release_bindings(manifest, _resolve_release_tag) if args.network else 0
        )
        payload = {
            "status": "PASS",
            "mode": "REMOTE_VERIFIED" if args.network else "STATIC_METADATA_VALIDATED",
            "network_used": args.network,
            "source_count": len(manifest.sources),
            "dependency_count": sum(
                source.allowed_action == "DEPENDENCY" for source in manifest.sources
            ),
            "unknown_license_count": sum(
                source.license_status == "UNKNOWN" for source in manifest.sources
            ),
            "remote_url_check_count": len(remote_urls),
            "remote_release_binding_count": remote_release_binding_count,
            **binding_summary,
        }
        exit_code = 0
    except Exception as exc:
        payload = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
