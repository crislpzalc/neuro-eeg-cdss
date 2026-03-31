from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids


DATASET_ROOT = Path("data/raw/chbmit_bids")


def parse_bids_entities(eeg_file: Path) -> dict:
    stem = eeg_file.stem
    parts = stem.split("_")

    entities = {}
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
        else:
            entities["suffix"] = part

    return entities


def main():
    eeg_files = sorted(DATASET_ROOT.rglob("*_eeg.edf"))
    assert eeg_files, "No EEG files found"

    eeg_file = eeg_files[0]
    entities = parse_bids_entities(eeg_file)

    bids_path = BIDSPath(
        root=DATASET_ROOT,
        subject=entities.get("sub"),
        session=entities.get("ses"),
        task=entities.get("task"),
        acquisition=entities.get("acq"),
        run=entities.get("run"),
        processing=entities.get("proc"),
        recording=entities.get("recording"),
        datatype="eeg",
        suffix="eeg",
        extension=".edf",
    )

    raw = read_raw_bids(bids_path=bids_path)

    annotations = raw.annotations

    print(f"[OK] Annotations inspected for: {eeg_file.name}")
    print(f"Number of annotations: {len(annotations)}")

    for i in range(min(10, len(annotations))):
        print(
            f"{i}: onset={annotations.onset[i]}, "
            f"duration={annotations.duration[i]}, "
            f"description={annotations.description[i]}"
        )


if __name__ == "__main__":
    main()
