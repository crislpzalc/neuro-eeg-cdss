from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("data/raw/chbmit_bids")
OUTPUT_PATH = Path("data/manifests/manifest.parquet")


def parse_bids_filename(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("_")

    entities = {
        "subject": None,
        "session": None,
        "task": None,
        "run": None,
        "suffix": None,
    }

    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            if key == "sub":
                entities["subject"] = f"sub-{value}"
            elif key == "ses":
                entities["session"] = f"ses-{value}"
            elif key == "task":
                entities["task"] = value
            elif key == "run":
                entities["run"] = value
        else:
            entities["suffix"] = part

    return entities


def main():
    rows = []

    for eeg_file in sorted(DATASET_ROOT.rglob("*_eeg.edf")):
        meta = parse_bids_filename(eeg_file)

        rows.append(
            {
                "subject": meta["subject"],
                "session": meta["session"],
                "task": meta["task"],
                "run": meta["run"],
                "filename": eeg_file.name,
                "path": str(eeg_file),
            }
        )

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("[OK] Manifest created")
    print(df.head())
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject'].nunique()}")


if __name__ == "__main__":
    main()
