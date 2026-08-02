"""Download immutable downstream sources and record file-level provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


USER_AGENT = "CFNN-downstream-validation/1.0"


def _request(url: str, headers: dict | None = None):
    request_headers = {"User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    return urllib.request.Request(url, headers=request_headers)


def _read_json_url(url: str) -> dict:
    with urllib.request.urlopen(_request(url), timeout=60) as response:
        return json.load(response)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_provider_checksum(
    path: Path, provider_checksum: str | None, algorithm: str | None = None
) -> None:
    """Fail closed when a provider publishes a checksum for a downloaded file."""
    if not provider_checksum:
        return
    expected = str(provider_checksum).strip().lower()
    if ":" in expected:
        declared_algorithm, expected = expected.split(":", 1)
        if algorithm and algorithm.lower() != declared_algorithm:
            raise ValueError(
                f"provider checksum algorithm mismatch for {path.name}: "
                f"{algorithm!r} != {declared_algorithm!r}"
            )
        algorithm = declared_algorithm
    algorithm = (algorithm or "md5").lower()
    try:
        digest = hashlib.new(algorithm)
    except ValueError as error:
        raise ValueError(f"unsupported provider checksum algorithm: {algorithm}") from error
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().lower() != expected:
        raise ValueError(f"provider checksum mismatch for {path.name}")


def _zenodo_license(record: dict) -> str | list:
    metadata = record.get("metadata", {})
    license_value = metadata.get("license")
    if isinstance(license_value, dict):
        return license_value.get("id") or license_value.get("title") or license_value
    if license_value:
        return license_value
    return metadata.get("rights") or []


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else None
    try:
        with urllib.request.urlopen(_request(url, headers), timeout=120) as response:
            status = getattr(response, "status", None) or response.getcode()
            mode = "ab" if offset and status == 206 else "wb"
            with temporary.open(mode) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        if temporary.stat().st_size == 0:
            raise ValueError(f"downloaded empty file from {url}")
        temporary.replace(destination)
    except BaseException:
        raise


def _zenodo_files(specification: dict) -> tuple[list[dict], dict]:
    metadata = _read_json_url(specification["url"])
    files = []
    for item in metadata.get("files", []):
        content_url = item.get("links", {}).get("content") or item.get("links", {}).get("self")
        if not content_url:
            raise ValueError(f"Zenodo file lacks content URL: {item.get('key')}")
        files.append({
            "filename": item["key"],
            "url": content_url,
            "expected_size": int(item["size"]),
            "provider_checksum": item.get("checksum"),
        })
    if not files:
        raise ValueError(f"Zenodo record {specification['record_id']} has no files")
    record_metadata = metadata.get("metadata", {})
    return files, {
        "record_id": int(specification["record_id"]),
        "doi": metadata.get("doi"),
        "title": record_metadata.get("title"),
        "creators": [
            creator.get("name") for creator in record_metadata.get("creators", [])
            if creator.get("name")
        ],
        "license": _zenodo_license(metadata),
        "updated": metadata.get("updated"),
    }


def _figshare_files(specification: dict) -> tuple[list[dict], dict]:
    """Resolve exactly one declared Figshare article version."""
    metadata = _read_json_url(specification["api_url"])
    expected_version = int(specification["version"])
    actual_version = metadata.get("version")
    if actual_version != expected_version:
        raise ValueError(
            "Figshare version mismatch: "
            f"expected {expected_version}, received {actual_version!r}"
        )
    license_name = metadata.get("license", {}).get("name")
    expected_license = specification.get("license")
    if expected_license and license_name != expected_license:
        raise ValueError(
            "Figshare license mismatch: "
            f"expected {expected_license!r}, received {license_name!r}"
        )
    files = []
    for item in metadata.get("files", []):
        download_url = item.get("download_url")
        if not download_url:
            raise ValueError(f"Figshare file lacks download URL: {item.get('name')}")
        files.append({
            "filename": item["name"],
            "url": download_url,
            "expected_size": int(item["size"]),
            "provider_checksum": item.get("computed_md5") or item.get("supplied_md5"),
            "provider_checksum_algorithm": "md5",
            "file_id": int(item["id"]),
        })
    if not files:
        raise ValueError(
            f"Figshare article {specification['article_id']} has no downloadable files"
        )
    required_filename = specification.get("expected_data_filename")
    if required_filename and required_filename not in {item["filename"] for item in files}:
        raise ValueError(
            f"Figshare article {specification['article_id']} is missing "
            f"required file {required_filename!r}"
        )
    expected_data = specification.get("expected_data_file")
    if expected_data:
        data_file = next(
            item for item in files if item["filename"] == required_filename
        )
        expected_values = {
            "file_id": int(expected_data["file_id"]),
            "expected_size": int(expected_data["size"]),
            "provider_checksum": expected_data["figshare_md5"],
        }
        for key, expected_value in expected_values.items():
            if data_file.get(key) != expected_value:
                raise ValueError(
                    f"Figshare pinned file metadata mismatch for {required_filename}: "
                    f"{key} expected {expected_value!r}, received "
                    f"{data_file.get(key)!r}"
                )
    return files, {
        "article_id": int(specification["article_id"]),
        "version": expected_version,
        "title": metadata.get("title"),
        "authors": [
            author.get("full_name") for author in metadata.get("authors", [])
            if author.get("full_name")
        ],
        "license": license_name,
        "api_url": specification["api_url"],
        "url_public_html": metadata.get("url_public_html"),
        "modified_date": metadata.get("modified_date"),
    }


def _direct_files(specification: dict) -> tuple[list[dict], dict]:
    filename = specification.get("filename")
    if not filename:
        filename = Path(urllib.parse.unquote(urllib.parse.urlparse(specification["url"]).path)).name
    if not filename:
        raise ValueError(f"cannot derive filename from {specification['url']}")
    return [{"filename": filename, "url": specification["url"]}], {
        "dataset_id": specification.get("dataset_id"),
        "license": specification.get("license"),
    }


def download_sources(source_config: Path, raw_root: Path, *, metadata_only: bool = False) -> dict:
    source_config = Path(source_config)
    raw_root = Path(raw_root)
    specifications = json.loads(source_config.read_text())
    raw_root.mkdir(parents=True, exist_ok=True)
    old_manifest_path = raw_root / "source_manifest.json"
    old_files = {}
    if old_manifest_path.exists():
        old_manifest = json.loads(old_manifest_path.read_text())
        old_files = {item["path"]: item for item in old_manifest.get("files", [])}

    source_rows = []
    file_rows = []
    for source_name, specification in specifications.items():
        provider = specification["provider"]
        if provider == "zenodo":
            files, provider_metadata = _zenodo_files(specification)
        elif provider == "figshare":
            files, provider_metadata = _figshare_files(specification)
        elif provider in {"uci", "direct"}:
            files, provider_metadata = _direct_files(specification)
        else:
            raise ValueError(f"unsupported source provider: {provider}")
        source_rows.append({
            "name": source_name,
            "provider": provider,
            "role": specification["role"],
            "url": specification.get("url", specification.get("api_url")),
            "file_count": len(files),
            **provider_metadata,
        })
        for item in files:
            relative_path = Path(source_name) / item["filename"]
            destination = raw_root / relative_path
            if metadata_only:
                file_rows.append({
                    "source": source_name,
                    "path": relative_path.as_posix(),
                    "url": item["url"],
                    "expected_size": item.get("expected_size"),
                    "provider_checksum": item.get("provider_checksum"),
                    "status": "metadata_only",
                })
                continue
            if destination.exists():
                digest = _sha256(destination)
                prior = old_files.get(relative_path.as_posix())
                if prior and prior.get("sha256") != digest:
                    raise ValueError(f"existing raw file checksum changed: {relative_path}")
            else:
                _download(item["url"], destination)
                digest = _sha256(destination)
            expected_size = item.get("expected_size")
            if expected_size is not None and destination.stat().st_size != expected_size:
                raise ValueError(f"download size mismatch for {relative_path}")
            _verify_provider_checksum(
                destination,
                item.get("provider_checksum"),
                item.get("provider_checksum_algorithm"),
            )
            file_rows.append({
                "source": source_name,
                "path": relative_path.as_posix(),
                "url": item["url"],
                "size": destination.stat().st_size,
                "sha256": digest,
                "provider_checksum": item.get("provider_checksum"),
                "status": "verified",
            })

    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata_only": bool(metadata_only),
        "source_config": source_config.resolve().as_posix(),
        "source_config_sha256": _sha256(source_config),
        "sources": source_rows,
        "files": file_rows,
    }
    output_name = "source_metadata_manifest.json" if metadata_only else "source_manifest.json"
    (raw_root / output_name).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "downstream_sources.json")
    parser.add_argument("--raw-root", type=Path, default=root / "downstream_data" / "raw")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    manifest = download_sources(args.config, args.raw_root, metadata_only=args.metadata_only)
    print(json.dumps({
        "metadata_only": manifest["metadata_only"],
        "sources": len(manifest["sources"]),
        "files": len(manifest["files"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
