"""Helpers for tying calibration, allocation, and quantization artifacts together."""

from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping


CALIBRATION_FIELDS = ("dataset", "nsamples", "seqlen", "seed")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(artifact_path: str | Path) -> Path:
    """Return the sidecar path without changing the legacy artifact name."""
    return Path(f"{artifact_path}.metadata.json")


def load_optional_metadata(
    artifact_path: str | Path,
    *,
    artifact_name: str,
) -> dict[str, Any] | None:
    """Load a sidecar when present, but keep pre-sidecar artifacts usable."""
    sidecar = metadata_path(artifact_path)
    if not sidecar.exists():
        warnings.warn(
            f"{artifact_name} has no metadata sidecar at {sidecar}; "
            "accepting it as a legacy artifact without provenance validation.",
            stacklevel=2,
        )
        return None
    with sidecar.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata object in {sidecar}.")
    expected_hash = metadata.get("artifact_sha256")
    if expected_hash is not None:
        actual_hash = file_sha256(artifact_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{artifact_name} content does not match {sidecar}: "
                f"expected SHA-256 {expected_hash}, got {actual_hash}."
            )
    return metadata


def write_metadata(artifact_path: str | Path, metadata: Mapping[str, Any]) -> Path:
    sidecar = metadata_path(artifact_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    if Path(artifact_path).exists():
        payload["artifact_sha256"] = file_sha256(artifact_path)
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return sidecar


def validate_calibration_artifacts(
    artifacts: Iterable[tuple[str, Mapping[str, Any] | None]],
    expected: Mapping[str, Any],
) -> str | None:
    """Validate known calibration fields and hashes across related artifacts.

    Missing metadata is tolerated for compatibility. Once metadata exists,
    mismatches are errors so a stale artifact cannot be consumed silently.
    """
    input_hashes: dict[str, str] = {}
    for artifact_name, metadata in artifacts:
        if metadata is None:
            continue
        calibration = metadata.get("calibration")
        if not isinstance(calibration, Mapping):
            warnings.warn(
                f"{artifact_name} metadata has no calibration object; "
                "skipping provenance validation for this legacy format.",
                stacklevel=2,
            )
            continue
        mismatches = []
        for field in CALIBRATION_FIELDS:
            expected_value = expected.get(field)
            actual_value = calibration.get(field)
            if actual_value is None:
                if expected_value is not None:
                    warnings.warn(
                        f"{artifact_name} metadata has no {field!r}; "
                        "accepting the incomplete legacy metadata.",
                        stacklevel=2,
                    )
                continue
            if expected_value is not None and actual_value != expected_value:
                mismatches.append(
                    f"{field}: expected {expected_value!r}, got {actual_value!r}"
                )
        if mismatches:
            raise ValueError(
                f"Calibration metadata mismatch for {artifact_name}: "
                + "; ".join(mismatches)
            )
        input_hash = calibration.get("input_ids_sha256")
        if input_hash:
            input_hashes[artifact_name] = str(input_hash)

    if len(set(input_hashes.values())) > 1:
        details = ", ".join(
            f"{name}={value}" for name, value in input_hashes.items()
        )
        raise ValueError(f"Calibration token hashes disagree across artifacts: {details}")
    return next(iter(input_hashes.values()), None)
