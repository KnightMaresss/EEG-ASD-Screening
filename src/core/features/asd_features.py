"""Feature extraction for ASD resting-state EEG."""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ASDFeatureExtractor:
    """Extract spectral features for ASD detection."""
    
    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    
    def __init__(self, sfreq: float = 512.0):
        self.sfreq = sfreq
        
    def compute_band_power(self, data: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """Compute power in a frequency band."""
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=min(256, data.shape[1]))
        low_idx = np.argmax(freqs >= band[0])
        high_idx = np.argmax(freqs >= band[1]) if band[1] <= freqs[-1] else -1
        return np.mean(psd[:, low_idx:high_idx], axis=1)
    
    def extract_features(self, epoch: np.ndarray, ch_names: List[str]) -> Dict[str, float]:
        """Extract features from a single epoch - simplified version."""
        features = {}
        
        # 1. Mean band powers across all channels (robust to channel names!)
        for band_name, band_range in self.BANDS.items():
            powers = self.compute_band_power(epoch, band_range)
            features[f'mean_{band_name}_power'] = np.mean(powers)
            features[f'std_{band_name}_power'] = np.std(powers)
            features[f'max_{band_name}_power'] = np.max(powers)
        
        # 2. Theta/Beta ratio (key ASD biomarker from PDF)
        theta_power = np.mean(self.compute_band_power(epoch, self.BANDS['theta']))
        beta_power = np.mean(self.compute_band_power(epoch, self.BANDS['beta']))
        features['theta_beta_ratio'] = theta_power / (beta_power + 1e-10)
        
        # 3. Alpha/Theta ratio
        alpha_power = np.mean(self.compute_band_power(epoch, self.BANDS['alpha']))
        features['alpha_theta_ratio'] = alpha_power / (theta_power + 1e-10)
        
        # 4. Try to find frontal channels for frontal theta
        frontal_patterns = ['Fp', 'AF', 'F', 'frontal']
        frontal_indices = []
        for i, ch in enumerate(ch_names):
            if any(pattern in ch for pattern in frontal_patterns):
                frontal_indices.append(i)
        
        if frontal_indices:
            frontal_data = epoch[frontal_indices, :]
            frontal_theta = self.compute_band_power(frontal_data, self.BANDS['theta'])
            features['frontal_theta_mean'] = np.mean(frontal_theta)
        else:
            features['frontal_theta_mean'] = features['mean_theta_power']
        
        # 5. Global statistics
        features['global_mean'] = float(np.mean(epoch))
        features['global_std'] = float(np.std(epoch))
        features['global_max'] = float(np.max(np.abs(epoch)))
        
        return features
    
    def extract_all_features(
        self, 
        epochs: List[np.ndarray], 
        ch_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from all epochs."""
        all_features = []
        
        logger.info(f"Extracting features from {len(epochs)} epochs...")
        
        for i, epoch in enumerate(epochs):
            try:
                features = self.extract_features(epoch, ch_names)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Error on epoch {i}: {e}")
                # Skip problematic epochs
                continue
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(epochs)} epochs")
        
        if not all_features:
            raise ValueError("No features extracted! Check your data.")
        
        # Get consistent feature names from first successful epoch
        feature_names = list(all_features[0].keys())
        
        # Build feature matrix
        X_list = []
        for f_dict in all_features:
            row = []
            for name in feature_names:
                val = f_dict.get(name, 0.0)
                row.append(float(val) if val is not None else 0.0)
            X_list.append(row)
        
        X = np.array(X_list)
        
        logger.info(f"Feature matrix: {X.shape}")
        logger.info(f"Features: {feature_names}")
        
        return X, feature_names