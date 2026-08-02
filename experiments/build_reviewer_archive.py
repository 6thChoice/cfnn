"""Build a deterministic, checksummed reviewer archive from approved evidence roots."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import tarfile
from pathlib import Path


DIRECT_FILES = (
    "experiment_refine/budget_matching.py",
    "experiment_refine/build_confirmatory_artifacts.py",
    "experiment_refine/build_reviewer_archive.py",
    "experiment_refine/confirmatory_ablation_benchmark.py",
    "experiment_refine/confirmatory_artifacts.py",
    "experiment_refine/confirmatory_benchmark.py",
    "experiment_refine/confirmatory_models.py",
    "experiment_refine/confirmatory_protocol.py",
    "experiment_refine/confirmatory_statistics.py",
    "experiment_refine/confirmatory_training.py",
    "experiment_refine/confirmatory_config.json",
    "experiment_refine/confirmatory_ablation_config.json",
    "experiment_refine/confirmatory_results_schema.json",
    "experiment_refine/energy_efficiency.csv",
    "experiment_refine/export_confirmatory_datasets.py",
    "experiment_refine/fair_baselines.py",
    "experiment_refine/nmr_data.py",
    "experiment_refine/requirements-lock.txt",
    "experiment_refine/requirements-server-freeze.txt",
    "experiment_refine/run_confirmatory_server.sh",
)

DIRECTORIES = (
    "experiment_refine/confirmatory_results_sharded",
    "experiment_refine/confirmatory_supplement_sharded",
    "experiment_refine/confirmatory_ablation_results",
    "experiment_refine/confirmatory_shared_inventory",
    "experiment_refine/confirmatory_datasets",
    "experiment_refine/docs",
    "experiment_refine/tests",
    "experiment_refine/paper/derived",
    "experiment_refine/paper/figures",
    "experiment_refine/paper/scripts",
)

PAPER_SUFFIXES = {".tex", ".bib", ".bst", ".cls", ".md", ".pdf"}


def _approved(path: Path) -> bool:
    lowered = [part.lower() for part in path.parts]
    return (
        not any("invalid" in part for part in lowered)
        and "archive" not in lowered
        and not any(part.startswith("notion_") for part in lowered)
        and "__pycache__" not in lowered
        and path.suffix not in {".pyc", ".pyo"}
    )


def collect_files(repo_root: Path) -> list[Path]:
    files = {repo_root / relative for relative in DIRECT_FILES if (repo_root / relative).is_file()}
    for relative in DIRECTORIES:
        directory = repo_root / relative
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file() and _approved(path))
    paper = repo_root / "experiment_refine" / "paper"
    if paper.is_dir():
        files.update(
            path for path in paper.iterdir()
            if path.is_file() and path.suffix in PAPER_SUFFIXES and _approved(path)
        )
    return sorted(path for path in files if _approved(path))


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mode = 0o644
    return info


def build_archive(repo_root: Path, output: Path) -> int:
    repo_root, output = repo_root.resolve(), output.resolve()
    files = collect_files(repo_root)
    checksums = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.relative_to(repo_root).as_posix()}")
    checksum_bytes = ("\n".join(checksums) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    data = path.read_bytes()
                    archive.addfile(
                        _tar_info(path.relative_to(repo_root).as_posix(), len(data)),
                        io.BytesIO(data),
                    )
                archive.addfile(_tar_info("SHA256SUMS", len(checksum_bytes)), io.BytesIO(checksum_bytes))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = build_archive(Path(args.repo_root), Path(args.output))
    print(f"wrote {count} approved files to {args.output}")


if __name__ == "__main__":
    main()
