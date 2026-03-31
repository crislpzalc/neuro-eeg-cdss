from pathlib import Path

from neuro_eeg_cdss.preprocessing.dataset_builder import (
    derive_events_tsv_path_from_eeg_path,
)


def test_derive_events_tsv_path_from_eeg_path():
    eeg_path = Path("data/raw/chbmit_bids/sub-01/ses-01/eeg/sub-01_ses-01_task-chb01_01_eeg.edf")

    events_path = derive_events_tsv_path_from_eeg_path(eeg_path)

    assert events_path.name == "sub-01_ses-01_task-chb01_01_events.tsv"
