"""Background QThread workers for training and prediction.

Kept separate from the UI so `main_window.py` stays focused on wiring.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.core.features.asd_features import ASDFeatureExtractor
from src.core.models.model_manager import ModelManager
from src.core.preprocessing.loader import SheffieldLoader
from src.core.preprocessing.pipeline import prepare_from_file
from src.utils.config import (
    CV_FOLDS,
    DEFAULT_SFREQ,
    EPOCH_DURATION_S,
    MAX_EPOCHS_PER_SUBJECT,
    MODELS_DIR,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainingThread(QThread):
    """Train the ASD vs Control classifier off the UI thread."""

    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = Path(data_dir)

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            # Surface *any* failure to the UI instead of silently dying.
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        log = self.log_signal.emit

        log("=" * 60)
        log("STARTING ASD vs CONTROL TRAINING")
        log("=" * 60)

        log("\n📂 Loading Sheffield dataset...")
        self.progress_signal.emit(10)

        loader = SheffieldLoader(self.data_dir)
        epochs, labels, _subjects, ch_names = loader.load_all_subjects(
            epoch_duration=EPOCH_DURATION_S,
            max_epochs_per_subject=MAX_EPOCHS_PER_SUBJECT,
        )
        log(f"   Total epochs: {len(epochs)}")
        log(f"   ASD epochs: {sum(labels)}")
        log(f"   Control epochs: {len(labels) - sum(labels)}")

        log("\n🔬 Extracting spectral features...")
        self.progress_signal.emit(30)

        extractor = ASDFeatureExtractor(sfreq=DEFAULT_SFREQ)
        X, feature_names = extractor.extract_all_features(epochs, ch_names)
        y = np.array(labels)

        log(f"   Features extracted: {X.shape[1]} features")
        log(f"   Total samples: {X.shape[0]}")

        log("\n📏 Scaling features...")
        self.progress_signal.emit(50)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        log("\n🤖 Training Random Forest...")
        self.progress_signal.emit(60)

        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        log(f"\n📊 Performing {CV_FOLDS}-fold cross-validation...")
        self.progress_signal.emit(75)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

        log(f"   CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
        log(f"   Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        log("\n🎯 Training final model...")
        self.progress_signal.emit(85)
        model.fit(X_scaled, y)

        log("\n💾 Saving model...")
        self.progress_signal.emit(95)

        metadata = {
            "accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_asd": int(sum(labels)),
            "n_control": int(len(labels) - sum(labels)),
            "feature_names": feature_names,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "RandomForest_ASD_vs_Control",
            "dataset": "Sheffield_Resting_State",
        }

        ModelManager(MODELS_DIR).save_model(model, scaler, metadata)

        log("\n" + "=" * 60)
        log("✅ TRAINING COMPLETE!")
        log("=" * 60)
        log(f"   Accuracy: {cv_scores.mean():.1%}")
        log(f"   Model saved to: {MODELS_DIR}")

        self.progress_signal.emit(100)
        self.finished_signal.emit(
            {
                "model": model,
                "scaler": scaler,
                "accuracy": float(cv_scores.mean()),
                "metadata": metadata,
            }
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictionThread(QThread):
    """Run inference on a single EEG file off the UI thread.

    NOTE (temporary): preprocessing here is still *inline* — the same
    load → filter → epoch steps as before. In Step 4 we replace this with a
    shared `prepare_epochs()` call so training and prediction use the exact
    same pipeline. Until then, behavior is identical to the old code.
    """

    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        log = self.log_signal.emit

        log("Loading model...")
        self.progress_signal.emit(10)

        model_manager = ModelManager(MODELS_DIR)
        if not model_manager.model_exists():
            raise FileNotFoundError(
                "No trained model found. Please train the model first."
            )

        model, scaler, metadata = model_manager.load_model()
        if metadata and metadata.get("accuracy") is not None:
            log(f"Model loaded: Accuracy={metadata['accuracy']:.1%}")

        log(f"Loading EEG file: {Path(self.file_path).name}")
        self.progress_signal.emit(30)

        epoch_list, ch_names, sfreq, raw_data = prepare_from_file(self.file_path)
        n_epochs = len(epoch_list)

        log(f"Data shape: {len(ch_names)} channels, {raw_data.shape[1]} samples")
        log(f"Processing {n_epochs} epochs...")

        self.progress_signal.emit(50)

        log("Extracting features...")
        extractor = ASDFeatureExtractor(sfreq=sfreq)
        X, _feature_names = extractor.extract_all_features(epoch_list, ch_names)
        X_scaled = scaler.transform(X)

        self.progress_signal.emit(70)

        log("Making predictions...")
        probabilities = model.predict_proba(X_scaled)
        asd_prob = float(np.mean(probabilities[:, 1]))
        control_prob = float(np.mean(probabilities[:, 0]))

        self.progress_signal.emit(90)

        if asd_prob > 0.5:
            label = "ASD DETECTED"
            confidence = asd_prob
        else:
            label = "NEUROTYPICAL"
            confidence = control_prob

        results = {
            "prediction": label,
            "confidence": confidence,
            "asd_probability": asd_prob,
            "control_probability": control_prob,
            "data": raw_data,
            "ch_names": ch_names,
            "sfreq": sfreq,
            "n_epochs_processed": n_epochs,
        }

        log("\n✅ Prediction complete!")
        log(f"Result: {label}")
        log(f"Confidence: {confidence:.1%}")
        log(f"ASD Probability: {asd_prob:.1%}")
        log(f"Control Probability: {control_prob:.1%}")

        self.progress_signal.emit(100)
        self.finished_signal.emit(results)