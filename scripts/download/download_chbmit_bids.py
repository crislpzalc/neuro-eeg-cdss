from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


RECORD_ID = "10259996"
DOI = "10.5281/zenodo.10259996"
FILENAME = "BIDS_CHB-MIT.zip"
EXPECTED_MD5 = "df22b26fb8bf3db837c1bdbf205e99b7"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = DATA_DIR / FILENAME
EXTRACT_DIR = DATA_DIR / "chbmit_bids"


def compute_md5(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def run_zenodo_get(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "zenodo_get",
        RECORD_ID,
        "-g",
        FILENAME,
        "-o",
        str(output_dir),
    ]

    print("[INFO] Downloading dataset with zenodo_get...")
    print("[INFO] Command:", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError("zenodo_get failed. Install it with:\n  pip install zenodo-get")

    if not ZIP_PATH.exists():
        raise RuntimeError(f"Expected ZIP file not found after download: {ZIP_PATH}")


def safe_extract(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = extract_to / member
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(extract_to.resolve())):
                raise RuntimeError(f"Unsafe path detected in ZIP: {member}")

        print("[INFO] Extracting ZIP...")
        zf.extractall(extract_to)
        print("[OK] Extraction completed.")


def normalize_extracted_structure(extract_dir: Path) -> None:
    if (extract_dir / "dataset_description.json").exists():
        return

    children = list(extract_dir.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        return

    nested_root = children[0]
    if not (nested_root / "dataset_description.json").exists():
        return

    print(f"[INFO] Detected nested folder: {nested_root.name}")
    print("[INFO] Flattening directory structure...")

    for item in nested_root.iterdir():
        target = extract_dir / item.name
        if target.exists():
            raise RuntimeError(f"Target already exists: {target}")
        shutil.move(str(item), str(target))

    nested_root.rmdir()
    print("[OK] Structure normalized.")


def validate_bids_root(extract_dir: Path) -> None:
    dataset_description = extract_dir / "dataset_description.json"
    participants_tsv = extract_dir / "participants.tsv"
    subjects = sorted(p for p in extract_dir.glob("sub-*") if p.is_dir())

    if not dataset_description.exists():
        raise RuntimeError("Missing dataset_description.json after extraction.")
    if not participants_tsv.exists():
        raise RuntimeError("Missing participants.tsv after extraction.")
    if not subjects:
        raise RuntimeError("No sub-* directories found after extraction.")

    print("[OK] Basic BIDS validation passed:")
    print("  - dataset_description.json found")
    print("  - participants.tsv found")
    print(f"  - subjects detected: {len(subjects)}")
    print(f"  - first subjects: {[p.name for p in subjects[:5]]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract CHB-MIT BIDS dataset from Zenodo."
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists() and not args.force_extract:
        print(f"[INFO] Dataset already exists at: {EXTRACT_DIR}")
        validate_bids_root(EXTRACT_DIR)
        return 0

    if args.force_extract and EXTRACT_DIR.exists():
        print(f"[INFO] Removing existing dataset: {EXTRACT_DIR}")
        shutil.rmtree(EXTRACT_DIR)

    if args.force_download and ZIP_PATH.exists():
        print(f"[INFO] Removing existing ZIP: {ZIP_PATH}")
        ZIP_PATH.unlink()

    if not ZIP_PATH.exists():
        run_zenodo_get(DATA_DIR)
    else:
        print(f"[INFO] ZIP already exists: {ZIP_PATH}")

    if not args.skip_md5:
        print("[INFO] Verifying MD5 checksum...")
        md5 = compute_md5(ZIP_PATH)
        print(f"[INFO] MD5: {md5}")
        if md5 != EXPECTED_MD5:
            raise RuntimeError(f"MD5 mismatch. Expected {EXPECTED_MD5}, got {md5}")
        print("[OK] MD5 verified.")

    safe_extract(ZIP_PATH, EXTRACT_DIR)
    normalize_extracted_structure(EXTRACT_DIR)
    validate_bids_root(EXTRACT_DIR)

    if not args.keep_zip:
        print(f"[INFO] Removing ZIP file: {ZIP_PATH}")
        ZIP_PATH.unlink(missing_ok=True)

    print("\n[OK] Dataset ready at:")
    print(f"  {EXTRACT_DIR}")
    print(f"[INFO] Source DOI: {DOI}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
