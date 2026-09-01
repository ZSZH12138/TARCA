from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import IO, Literal, Self, cast

from pydantic import field_validator, model_validator

from tarca.contracts import (
    Sha256Hash,
    StrictContractModel,
    canonical_json_bytes,
    canonical_json_hash,
    sha256_file,
)
from tarca.e02.config import load_e02_config
from tarca.stage2.freeze import verify_frozen_stage2_suite


class E02ServerHandoffError(RuntimeError):
    """Raised when the frozen Stage 2 handoff cannot be restored safely."""


class E02ServerHandoff(StrictContractModel):
    schema_version: Literal["tarca-e02-server-handoff-v1"]
    complete_archive_filename: str
    complete_archive_sha256: Sha256Hash
    stage2_freeze_receipt_sha256: Sha256Hash
    stage2_freeze_receipt_file_sha256: Sha256Hash
    stage2_manifest_file_sha256: Sha256Hash
    e02_scientific_config_sha256: Sha256Hash
    required_members: tuple[str, ...]

    @field_validator("required_members", mode="before")
    @classmethod
    def _members_are_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _identity_is_complete(self) -> Self:
        if not self.complete_archive_filename.endswith(".tar.gz"):
            raise ValueError("complete archive filename must end in .tar.gz")
        if not self.required_members or len(self.required_members) != len(
            set(self.required_members)
        ):
            raise ValueError("required archive members must be nonempty and unique")
        for member in self.required_members:
            path = _safe_member_name(member)
            if path.parts[:2] != ("artifacts", "stage2"):
                raise ValueError("required archive members must belong to Stage 2 artifacts")
        return self


def load_e02_server_handoff(path: Path) -> E02ServerHandoff:
    try:
        return E02ServerHandoff.model_validate_json(path.resolve().read_text(encoding="utf-8"))
    except Exception as error:
        raise E02ServerHandoffError(f"E02 server handoff is invalid: {error}") from error


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "." in path.parts
        or "\x00" in name
        or "\\" in name
    ):
        raise E02ServerHandoffError("complete archive contains an unsafe path")
    return path


def _destination(root: Path, member: PurePosixPath) -> Path:
    destination = (root / Path(*member.parts)).resolve()
    if destination == root or root not in destination.parents:
        raise E02ServerHandoffError("complete archive destination escapes the repository")
    return destination


def _stream_hash(source: IO[bytes]) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _atomic_stream(destination: Path, source: IO[bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".e02-stage2-restore-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(destination: Path, value: dict[str, object]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".e02-stage2-receipt-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_receipt(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise E02ServerHandoffError("existing Stage 2 restore receipt is invalid") from error
    if not isinstance(value, dict):
        raise E02ServerHandoffError("existing Stage 2 restore receipt is invalid")
    unsigned = dict(value)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if receipt_sha256 != canonical_json_hash(unsigned):
        raise E02ServerHandoffError("existing Stage 2 restore receipt is invalid")
    return cast(dict[str, object], value)


def load_stage2_restore_receipt(
    path: Path, handoff: E02ServerHandoff
) -> dict[str, object]:
    receipt = _read_receipt(path.resolve())
    if any(
        (
            receipt.get("schema_version") != "tarca-e02-stage2-restore-v1",
            receipt.get("status") != "RESTORED",
            receipt.get("complete_archive_filename") != handoff.complete_archive_filename,
            receipt.get("complete_archive_sha256") != handoff.complete_archive_sha256,
            receipt.get("stage2_freeze_receipt_sha256")
            != handoff.stage2_freeze_receipt_sha256,
            receipt.get("e02_scientific_config_sha256")
            != handoff.e02_scientific_config_sha256,
            receipt.get("formal_tasks_executed") != 0,
            receipt.get("scientific_results_visible") is not False,
        )
    ):
        raise E02ServerHandoffError("Stage 2 restore receipt does not match the handoff")
    return receipt


def load_server_bundle_verification_receipt(path: Path) -> dict[str, object]:
    receipt = _read_receipt(path.resolve())
    if any(
        (
            receipt.get("schema_version")
            != "tarca-e02-server-bundle-verification-v1",
            receipt.get("status") != "VERIFIED",
            not isinstance(receipt.get("server_bundle_sha256"), str),
            not isinstance(receipt.get("verified_file_count"), int),
            receipt.get("formal_tasks_executed") != 0,
        )
    ):
        raise E02ServerHandoffError("server bundle verification receipt is invalid")
    return receipt


def verify_server_bundle_checkout(
    repository_root: Path, server_bundle: Path
) -> dict[str, object]:
    root = repository_root.resolve()
    bundle = server_bundle.resolve()
    if not bundle.is_file():
        raise E02ServerHandoffError("server bundle is missing")
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            if any(
                member.issym()
                or member.islnk()
                or (not member.isfile() and not member.isdir())
                for member in members
            ):
                raise E02ServerHandoffError("server bundle contains a non-regular member")
            for member in members:
                _safe_member_name(member.name)
            manifests = tuple(
                member
                for member in members
                if member.isfile() and member.name == "SHA256SUMS.json"
            )
            if len(manifests) != 1:
                raise E02ServerHandoffError("server bundle must contain one SHA256SUMS manifest")
            source = archive.extractfile(manifests[0])
            if source is None:
                raise E02ServerHandoffError("server bundle manifest cannot be read")
            manifest = json.load(source)
    except E02ServerHandoffError:
        raise
    except Exception as error:
        raise E02ServerHandoffError("server bundle cannot be verified") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "tarca-stage2-bundle-v1"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise E02ServerHandoffError("server bundle manifest is invalid")
    sums = cast(dict[str, object], manifest["files"])
    if not sums:
        raise E02ServerHandoffError("server bundle manifest has no files")
    for relative, expected in sums.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise E02ServerHandoffError("server bundle manifest entries are invalid")
        path = _destination(root, _safe_member_name(relative))
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise E02ServerHandoffError(f"server bundle checkout file drifted: {relative}")
    receipt_payload: dict[str, object] = {
        "schema_version": "tarca-e02-server-bundle-verification-v1",
        "status": "VERIFIED",
        "server_bundle_sha256": sha256_file(bundle),
        "verified_file_count": len(sums),
        "formal_tasks_executed": 0,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": canonical_json_hash(receipt_payload),
    }
    receipt_path = root / "artifacts/e02/runtime/server_bundle_verification_receipt.json"
    if receipt_path.is_file():
        existing = load_server_bundle_verification_receipt(receipt_path)
        if existing != receipt:
            raise E02ServerHandoffError("server bundle verification identity drifted")
        return existing
    _atomic_json(receipt_path, receipt)
    return receipt


def restore_stage2_complete_archive(
    repository_root: Path,
    archive_path: Path,
    handoff_path: Path,
) -> dict[str, object]:
    root = repository_root.resolve()
    archive_file = archive_path.resolve()
    handoff = load_e02_server_handoff(handoff_path)
    if archive_file.name != handoff.complete_archive_filename:
        raise E02ServerHandoffError("complete archive filename does not match the handoff")
    if not archive_file.is_file() or sha256_file(archive_file) != handoff.complete_archive_sha256:
        raise E02ServerHandoffError("complete archive SHA-256 does not match the handoff")
    config_path = root / "configs/e02/e02_v1.yaml"
    if not config_path.is_file() or (
        load_e02_config(config_path).scientific_hash()
        != handoff.e02_scientific_config_sha256
    ):
        raise E02ServerHandoffError("E02 scientific configuration does not match the handoff")

    member_hashes: dict[str, str] = {}
    with tarfile.open(archive_file, "r:gz") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        for member in members:
            path = _safe_member_name(member.name)
            if member.name in names:
                raise E02ServerHandoffError("complete archive contains duplicate members")
            names.add(member.name)
            if path.parts[0] not in {"artifacts", "logs"}:
                raise E02ServerHandoffError(
                    "complete archive contains an unexpected top-level path"
                )
            if path.parts[0] == "artifacts" and path.parts[:2] != ("artifacts", "stage2"):
                raise E02ServerHandoffError("complete archive contains non-Stage 2 artifacts")
            if member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                raise E02ServerHandoffError("complete archive contains a non-regular member")
        if set(handoff.required_members) - names:
            raise E02ServerHandoffError("complete archive is missing a required member")

        stage2_files = tuple(
            member
            for member in members
            if member.isfile()
            and _safe_member_name(member.name).parts[:2] == ("artifacts", "stage2")
        )
        for member in stage2_files:
            source = archive.extractfile(member)
            if source is None:
                raise E02ServerHandoffError("complete archive member cannot be read")
            digest = _stream_hash(source)
            member_hashes[member.name] = digest
            destination = _destination(root, _safe_member_name(member.name))
            if destination.exists():
                if not destination.is_file() or destination.is_symlink():
                    raise E02ServerHandoffError("complete archive would overwrite non-file state")
                if sha256_file(destination) != digest:
                    raise E02ServerHandoffError("complete archive would overwrite drifted state")

        for member in stage2_files:
            destination = _destination(root, _safe_member_name(member.name))
            if destination.is_file():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise E02ServerHandoffError("complete archive member cannot be read")
            _atomic_stream(destination, source)
            if sha256_file(destination) != member_hashes[member.name]:
                raise E02ServerHandoffError("restored Stage 2 artifact SHA-256 does not match")

    freeze = verify_frozen_stage2_suite(root / "artifacts/stage2")
    freeze_receipt_path = root / "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json"
    manifest_path = root / "artifacts/stage2/frozen/v1/stage2_manifest.json"
    if any(
        (
            freeze.receipt_sha256 != handoff.stage2_freeze_receipt_sha256,
            sha256_file(freeze_receipt_path) != handoff.stage2_freeze_receipt_file_sha256,
            sha256_file(manifest_path) != handoff.stage2_manifest_file_sha256,
        )
    ):
        raise E02ServerHandoffError("restored Stage 2 frozen identity does not match")

    receipt_payload: dict[str, object] = {
        "schema_version": "tarca-e02-stage2-restore-v1",
        "status": "RESTORED",
        "complete_archive_filename": archive_file.name,
        "complete_archive_sha256": handoff.complete_archive_sha256,
        "stage2_freeze_receipt_sha256": freeze.receipt_sha256,
        "e02_scientific_config_sha256": handoff.e02_scientific_config_sha256,
        "restored_file_count": len(member_hashes),
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": canonical_json_hash(receipt_payload),
    }
    receipt_path = root / "artifacts/e02/runtime/stage2_restore_receipt.json"
    if receipt_path.is_file():
        existing = load_stage2_restore_receipt(receipt_path, handoff)
        if existing != receipt:
            raise E02ServerHandoffError("existing Stage 2 restore receipt identity drifted")
        return existing
    _atomic_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="TARCA E02 v1 server handoff verifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-bundle")
    verify.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify.add_argument("--server-bundle", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--repository-root", type=Path, default=Path.cwd())
    restore.add_argument("--stage2-archive", type=Path, required=True)
    restore.add_argument(
        "--handoff", type=Path, default=Path("configs/e02/e02_server_handoff_v1.json")
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "verify-bundle":
        result = verify_server_bundle_checkout(
            arguments.repository_root, arguments.server_bundle
        )
    else:
        result = restore_stage2_complete_archive(
            arguments.repository_root, arguments.stage2_archive, arguments.handoff
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
