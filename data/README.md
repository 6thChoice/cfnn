# Data

This package keeps raw data out of Git. Source metadata and checksums are retained so reviewers can reproduce the download state.

Use:

```bash
python data/download_downstream_datasets.py --config data/downstream_sources.json --raw-root data/raw
```

The downloader writes:

- `data/raw/source_manifest.json` for downloaded and verified files.
- `data/raw/source_metadata_manifest.json` when `--metadata-only` is used.

The manuscript's measured-response tasks use Zenodo records 7767046 and 14175959.
