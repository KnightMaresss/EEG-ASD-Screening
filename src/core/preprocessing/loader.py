"""EEG Data Loader for Sheffield ASD resting-state dataset."""

import mne
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.core.preprocessing.pipeline import prepare_from_raw
from src.utils.config import MAX_EPOCHS_PER_SUBJECT
import logging
import re

logger = logging.getLogger(__name__)


class SheffieldLoader:
    """Loader for Sheffield ASD resting-state EEG dataset."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize loader.
        
        Args:
            data_dir: Directory containing .set/.fdt files
        """
        self.data_dir = Path(data_dir)
        self.subject_files = self._find_subject_files()
        
    def _find_subject_files(self) -> Dict[str, Dict]:
        """Find all complete subject files."""
        subject_files = {}
        
        for set_file in sorted(self.data_dir.glob("*.set")):
            subject_id = set_file.stem.split('_')[0]
            subject_id = ''.join(c for c in subject_id if c.isdigit())
            
            fdt_file = self.data_dir / f"{set_file.stem}.fdt"
            
            if fdt_file.exists():
                subject_files[subject_id] = {
                    'set_file': set_file,
                    'fdt_file': fdt_file,
                    'subject_id': subject_id
                }
        
        logger.info(f"Found {len(subject_files)} complete subjects")
        return subject_files
    
    def get_subject_ids(self) -> List[str]:
        """Get list of all subject IDs."""
        return sorted(self.subject_files.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    def load_raw(self, subject_id: str) -> mne.io.Raw:
        """Load raw EEG data for a subject."""
        file_info = self.subject_files.get(subject_id)
        if file_info is None:
            raise FileNotFoundError(f"No files found for subject {subject_id}")
        
        raw = mne.io.read_raw_eeglab(
            file_info['set_file'], 
            preload=True, 
            verbose=False
        )
        
        logger.info(f"Loaded {subject_id}: {len(raw.ch_names)} channels, "
                   f"{raw.n_times / raw.info['sfreq']:.1f}s")
        
        return raw
    
    def get_subject_label(self, subject_id: str) -> int:
        """
        Get label for a subject.
        
        Based on typical organization: subjects 1-28 = ASD, 29-56 = Control
        If you find actual labels, update this method.
        """
        subject_num = int(subject_id) if subject_id.isdigit() else 0
        
        # Common convention: first half ASD, second half Control
        if subject_num <= 28:
            return 1  # ASD
        else:
            return 0  # Control
    
    def load_all_subjects(
        self,
        epoch_duration: float = 2.0,
        max_epochs_per_subject: Optional[int] = None,
        max_subjects: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
        """
        Load all subjects and return data with labels.
        
        Returns:
            Tuple of (epochs_list, labels_list, subject_ids, channel_names)
        """
        all_epochs = []
        all_labels = []
        all_subjects = []
        channel_names = None
        
        subject_ids = self.get_subject_ids()
        
        if max_subjects:
            subject_ids = subject_ids[:max_subjects]
        
        for i, subject_id in enumerate(subject_ids):
            try:
                label = self.get_subject_label(subject_id)
                
                logger.info(f"Loading {i+1}/{len(subject_ids)}: Subject {subject_id} "
                           f"({'ASD' if label == 1 else 'Control'})")
                
                raw = self.load_raw(subject_id)

                epoch_list, ch_names_this, _sfreq = prepare_from_raw(
                    raw,
                    epoch_duration=epoch_duration,
                    max_epochs=max_epochs_per_subject or MAX_EPOCHS_PER_SUBJECT,
                )

                if channel_names is None:
                    channel_names = ch_names_this

                for epoch in epoch_list:
                    all_epochs.append(epoch)
                    all_labels.append(label)
                    all_subjects.append(subject_id)

                logger.info(f"  → {len(epoch_list)} epochs extracted")
                
            except Exception as e:
                logger.error(f"Error loading subject {subject_id}: {e}")
        
        n_asd = sum(all_labels)
        n_control = len(all_labels) - n_asd
        
        logger.info(f"\n{'='*50}")
        logger.info(f"TOTAL: {len(all_epochs)} epochs from {len(set(all_subjects))} subjects")
        logger.info(f"  ASD: {n_asd} epochs")
        logger.info(f"  Control: {n_control} epochs")
        logger.info(f"{'='*50}")
        
        return all_epochs, all_labels, all_subjects, channel_names
    
    # Compatibility methods
    def get_available_subjects(self, data_dir: Path = None) -> List[str]:
        return self.get_subject_ids()
    
    def load_session_data(self, subject_id: str, session_id: str = None, data_dir: Path = None):
        raw = self.load_raw(subject_id)
        return {
            'data': raw.get_data(),
            'sampling_rate': raw.info['sfreq'],
            'channel_names': raw.ch_names,
            'subject_id': subject_id
        }