"""
Prepare ChefBot recipe data for local ingest.

Priority:
  1. dataset/recipes.json already present → validate & exit
  2. CHEFBOT_DATASET_URL / --url → download into dataset/recipes.json
  3. --from-sample (default fallback) → copy tracked sample_recipes.jsonl
     to dataset/recipes.json for a runnable demo corpus

Examples:
  python -u prepare_dataset.py
  python -u prepare_dataset.py --from-sample
  python -u prepare_dataset.py --url "https://example.com/recipes.jsonl"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

DATASET_DIR = Path("dataset")
TARGET = DATASET_DIR / "recipes.json"
SAMPLE = DATASET_DIR / "sample_recipes.jsonl"
DEFAULT_URL_ENV = "CHEFBOT_DATASET_URL"


def _count_jsonl(path: Path, *, max_lines: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            json.loads(line)
            count += 1
            if max_lines is not None and count >= max_lines:
                break
    return count


def validate_recipes_file(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < 32:
        raise ValueError(f"{path} looks empty ({size} bytes)")
    # Full 60k+ file can be large; validate structure on first 20 lines only,
    # then count quickly without re-parsing every object if huge.
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        if not first:
            raise ValueError(f"{path} is empty")
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list) or not payload:
                raise ValueError(f"{path} JSON array is empty or invalid")
            required = {"recipe_title", "ingredients", "directions"}
            sample = payload[0]
            if not required.issubset(sample.keys()):
                raise ValueError(
                    f"{path} objects missing keys {required - set(sample.keys())}"
                )
            return len(payload)

    # JSONL path
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if line_no <= 20:
                for key in ("recipe_title", "ingredients", "directions"):
                    if key not in obj:
                        raise ValueError(f"Line {line_no} missing '{key}'")
            count += 1
    if count < 1:
        raise ValueError(f"{path} has no recipe rows")
    return count


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading dataset from {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - user-provided URL by design
    count = validate_recipes_file(tmp)
    tmp.replace(dest)
    print(f"Downloaded OK ({count} recipes, {dest.stat().st_size:,} bytes)")


def copy_sample(dest: Path) -> None:
    if not SAMPLE.exists():
        raise FileNotFoundError(
            f"Tracked sample missing: {SAMPLE}. Re-clone the repo or restore it."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE, dest)
    count = validate_recipes_file(dest)
    print(
        f"Installed demo corpus from {SAMPLE} → {dest} "
        f"({count} recipes). For the full ~62k dump, place recipes.json "
        f"manually or pass --url / set {DEFAULT_URL_ENV}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ChefBot recipe dataset")
    parser.add_argument(
        "--url",
        default=(os.getenv(DEFAULT_URL_ENV, "") or "").strip().strip('"'),
        help=f"HTTP(S) URL to recipes JSON/JSONL (or set {DEFAULT_URL_ENV})",
    )
    parser.add_argument(
        "--from-sample",
        action="store_true",
        help="Force copy dataset/sample_recipes.jsonl → dataset/recipes.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dataset/recipes.json if it already exists",
    )
    args = parser.parse_args(argv)

    if TARGET.exists() and not args.force and not args.from_sample and not args.url:
        try:
            count = validate_recipes_file(TARGET)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR existing {TARGET} failed validation: {exc}", file=sys.stderr)
            return 1
        print(f"Dataset ready: {TARGET} ({count} recipes)")
        return 0

    if args.from_sample or (not args.url and not TARGET.exists()):
        if TARGET.exists() and not args.force and args.from_sample:
            print(f"ERROR {TARGET} exists. Re-run with --force to overwrite.")
            return 1
        copy_sample(TARGET)
        return 0

    if args.url:
        if TARGET.exists() and not args.force:
            print(f"ERROR {TARGET} exists. Re-run with --force to overwrite.")
            return 1
        try:
            download_url(args.url, TARGET)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR download failed: {exc}", file=sys.stderr)
            return 1
        return 0

    print(
        "ERROR: No dataset source. Use --from-sample, --url, or place "
        f"{TARGET} manually.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
