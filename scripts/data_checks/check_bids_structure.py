from pathlib import Path

DATASET_ROOT = Path("data/raw/chbmit_bids")


def main():
    assert DATASET_ROOT.exists(), "Dataset not found"

    assert (DATASET_ROOT / "dataset_description.json").exists()
    assert (DATASET_ROOT / "participants.tsv").exists()

    subjects = list(DATASET_ROOT.glob("sub-*"))
    assert len(subjects) > 0, "No subjects found"

    print("[OK] BIDS structure valid")
    print(f"Subjects: {len(subjects)}")


if __name__ == "__main__":
    main()
