from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids


DATASET_ROOT = Path("data/raw/chbmit_bids")


def parse_bids_entities(eeg_file: Path) -> dict:
    stem = eeg_file.stem  # filename without extension
    parts = stem.split("_")

    entities = {}
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
        else:
            # suffix like "eeg"
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

    print("[OK] Recording loaded successfully")
    print(f"File: {eeg_file.name}")
    print(raw)
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']}")
    print(f"First 10 channels: {raw.ch_names[:10]}")


if __name__ == "__main__":
    main()
