"""Model management - save, load, check availability."""

import joblib
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage ML models for the application."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.models_dir / "p300_classifier.joblib"
        self.scaler_path = self.models_dir / "scaler.joblib"
        self.metadata_path = self.models_dir / "model_metadata.joblib"
        
    def model_exists(self) -> bool:
        """Check if trained model exists."""
        return self.model_path.exists() and self.scaler_path.exists()
    
    def save_model(self, model, scaler, metadata: dict = None) -> None:
        """
        Save model, scaler, and metadata to disk.
        
        Args:
            model: Trained scikit-learn model
            scaler: Fitted StandardScaler
            metadata: Optional dictionary with model info
        """
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        
        if metadata:
            joblib.dump(metadata, self.metadata_path)
            
        logger.info(f"Model saved to {self.models_dir}")
    
    def load_model(self) -> Tuple:
        """
        Load saved model and scaler.
        
        Returns:
            Tuple of (model, scaler, metadata)
        
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        if not self.model_exists():
            raise FileNotFoundError(f"No trained model found in {self.models_dir}")
        
        model = joblib.load(self.model_path)
        scaler = joblib.load(self.scaler_path)
        
        metadata = None
        if self.metadata_path.exists():
            metadata = joblib.load(self.metadata_path)
        
        logger.info(f"Model loaded from {self.models_dir}")
        return model, scaler, metadata
    
    def get_model_info(self) -> Optional[dict]:
        """Get model metadata if available."""
        if self.metadata_path.exists():
            return joblib.load(self.metadata_path)
        return None