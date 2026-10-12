"""Shared EEG preprocessing pipeline.

This is the ONE place where raw EEG becomes model-ready epochs.

Both training (via SheffieldLoader) and prediction (via PredictionThread)
must go through this module so the model always sees features computed from
the same preprocessing chain it was trained on.

Pipeline:
    read_raw_eeglab
      -> bandpass filter (BANDPASS_LOW_HZ .. BANDPASS_HIGH_HZ)
      -> fixed-length epochs (EPOCH_DURATION_S each)
      -> capped at MAX_EPOCHS_PER_SUBJECT
"""
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np

from src.utils.config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    EPOCH_DURATION_S,
    MAX_EPOCHS_PER_SUBJECT,
)


def filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply the standard bandpass filter in place and return the raw."""
    raw.filter(
        BANDPASS_LOW_HZ,
        BANDPASS_HIGH_HZ,
        fir_design="firwin",
        verbose=False,
    )
    return raw


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    epoch_duration: float = EPOCH_DURATION_S,
    max_epochs: int = MAX_EPOCHS_PER_SUBJECT,
) -> np.ndarray:
    """Segment raw EEG into fixed-length epochs.

    Returns:
        np.ndarray of shape (n_epochs, n_channels, n_times), capped at
        ``max_epochs``.
    """
    events = mne.make_fixed_length_events(
        raw, duration=epoch_duration, overlap=0.0
    )
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=epoch_duration,
        baseline=None,
        preload=True,
        verbose=False,
    )
    data = epochs.get_data(copy=False)
    return data[: min(len(data), max_epochs)]


def prepare_from_raw(
    raw: mne.io.BaseRaw,
    epoch_duration: float = EPOCH_DURATION_S,
    max_epochs: int = MAX_EPOCHS_PER_SUBJECT,
) -> Tuple[List[np.ndarray], List[str], float]:
    """Filter + epoch an already-loaded Raw object.

    Use this when you already have a Raw (e.g. inside SheffieldLoader).

    Returns:
        (list_of_epochs, channel_names, sampling_rate)
    """
    filter_raw(raw)
    epochs_data = epochs_from_raw(raw, epoch_duration, max_epochs)
    epoch_list = [epochs_data[i] for i in range(len(epochs_data))]
    return epoch_list, list(raw.ch_names), float(raw.info["sfreq"])


def prepare_from_file(
    file_path: Path,
    epoch_duration: float = EPOCH_DURATION_S,
    max_epochs: int = MAX_EPOCHS_PER_SUBJECT,
) -> Tuple[List[np.ndarray], List[str], float, np.ndarray]:
    """Load a .set file from disk and run the full pipeline.

    Use this in the prediction path.

    Returns:
        (list_of_epochs, channel_names, sampling_rate, raw_data)
        where raw_data is the filtered [n_channels, n_times] array for plotting.
    """
    raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
    filter_raw(raw)

    # Grab the filtered data BEFORE epoching so the plot widget can show it.
    raw_data = raw.get_data()

    epochs_data = epochs_from_raw(raw, epoch_duration, max_epochs)
    epoch_list = [epochs_data[i] for i in range(len(epochs_data))]

    return epoch_list, list(raw.ch_names), float(raw.info["sfreq"]), raw_data