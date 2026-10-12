"""Main application window for EEG ASD Screening.

Keeps only UI wiring and persistence. Threads live in `threads.py`,
design tokens/QSS live in `theme.py`, preprocessing lives in
`core/preprocessing/pipeline.py`.
"""
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Add src to path so `src.*` imports work when running this file directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database.db_manager import DatabaseManager
from src.core.models.model_manager import ModelManager
from src.gui.theme import Theme, load_stylesheet
from src.gui.threads import PredictionThread, TrainingThread
from src.gui.widgets.eeg_plot_widget import EEGPlotWidget
from src.utils.config import (
    DATABASE_PATH,
    MODELS_DIR,
    PLOT_CHANNEL_COUNT,
    PLOT_SAMPLE_COUNT,
    RAW_DATA_DIR,
)


class MainWindow(QMainWindow):
    """Top-level window. Builds tabs, wires signals, persists results."""

    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager(MODELS_DIR)
        self.db_manager = DatabaseManager(DATABASE_PATH)

        self.prediction_thread = None
        self.training_thread = None

        self.init_ui()
        self.check_for_saved_model()
        self.load_history_from_database()

    # ------------------------------------------------------------------ UI
    def init_ui(self) -> None:
        self.setWindowTitle("ASD Screening Tool - EEG Analysis v2.0")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(load_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_predict_tab(), "Predict")
        self.tabs.addTab(self.create_train_tab(), "Train Model")
        self.tabs.addTab(self.create_history_tab(), "History")
        layout.addWidget(self.tabs)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # ------------------------------------------------------------ Helpers
    @staticmethod
    def _make_card(title: str = None, caption: str = None):
        """Return (frame, content_layout) styled as a premium card."""
        card = QFrame()
        card.setObjectName("Card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        if title:
            header = QHBoxLayout()
            header.setSpacing(10)
            title_lbl = QLabel(title)
            title_lbl.setObjectName("CardTitle")
            header.addWidget(title_lbl)
            header.addStretch()
            if caption:
                sub = QLabel(caption)
                sub.setObjectName("CardCaption")
                header.addWidget(sub)
            outer.addLayout(header)

        return card, outer

    def log(self, message: str) -> None:
        """Append a message to the pipeline log on the Predict tab."""
        self.log_output.append(message)

    # =================================================================
    # PREDICT TAB
    # =================================================================
    def create_predict_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Page header
        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel("ASD Screening")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Resting-state EEG classification · Sheffield protocol")
        subtitle.setObjectName("PageSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_predict_left_column())
        splitter.addWidget(self._build_predict_right_column())
        splitter.setSizes([360, 840])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)
        return widget

    def _build_predict_left_column(self) -> QScrollArea:
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # --- File card ---
        file_card, file_layout = self._make_card("INPUT FILE", "EEG · .SET")
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet(
            f"padding: 10px 12px; background-color: {Theme.BG_CARD_ALT};"
            f"border: 1px solid {Theme.BORDER}; border-radius: 6px;"
            f"color: {Theme.TEXT_MUTED}; font-size: 12px;"
        )
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        browse_btn = QPushButton("Browse Files")
        browse_btn.setObjectName("GhostButton")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        layout.addWidget(file_card)

        # --- Model status card ---
        status_card, status_layout = self._make_card("MODEL", "STATUS")
        self.model_status_label = QLabel("Checking…")
        self.model_status_label.setObjectName("MutedLabel")
        self.model_status_label.setWordWrap(True)
        status_layout.addWidget(self.model_status_label)
        layout.addWidget(status_card)

        # --- Analyze button ---
        self.analyze_btn = QPushButton("ANALYZE FOR ASD")
        self.analyze_btn.setObjectName("PrimaryButton")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.setMinimumHeight(54)
        self.analyze_btn.clicked.connect(self.analyze_eeg)
        layout.addWidget(self.analyze_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Result card ---
        result_card, result_layout = self._make_card("RESULT")
        self.results_label = QLabel("Awaiting analysis…")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet(
            f"color: {Theme.TEXT_MUTED}; font-size: 13px; padding: 20px 6px;"
        )
        result_layout.addWidget(self.results_label)
        layout.addWidget(result_card)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.setWidget(content)
        return scroll

    def _build_predict_right_column(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        plot_card, plot_layout = self._make_card("EEG SIGNAL", "8 CHANNELS · 10 s")
        self.plot_widget = EEGPlotWidget()
        self.plot_widget.setMinimumHeight(280)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.plot_widget)
        layout.addWidget(plot_card, stretch=3)

        log_card, log_layout = self._make_card("PIPELINE LOG")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(120)
        self.log_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_card, stretch=1)

        return panel

    # ---------------------------------------------------- Predict actions
    def browse_file(self) -> None:
        default_dir = RAW_DATA_DIR / "sheffield_asd_eeg"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG File",
            str(default_dir) if default_dir.exists() else str(RAW_DATA_DIR),
            "SET files (*.set);;All files (*.*)",
        )
        if file_path:
            self.file_label.setText(file_path)
            self.log(f"Selected: {Path(file_path).name}")

    def analyze_eeg(self) -> None:
        file_path = self.file_label.text()

        if file_path == "No file selected":
            self.log("⚠️ Please select an EEG file first.")
            return

        if not self.model_manager.model_exists():
            self.log("❌ No trained ASD model found. Please train first.")
            return

        # Guard against double-click while a run is in flight
        if self.prediction_thread and self.prediction_thread.isRunning():
            self.log("⚠️ A prediction is already running.")
            return

        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_output.clear()
        self.log("=" * 60)
        self.log("ASD SCREENING ANALYSIS")
        self.log("=" * 60)

        self.prediction_thread = PredictionThread(file_path)
        self.prediction_thread.log_signal.connect(self.log)
        self.prediction_thread.progress_signal.connect(self.progress_bar.setValue)
        self.prediction_thread.finished_signal.connect(self.on_prediction_complete)
        self.prediction_thread.error_signal.connect(self.on_prediction_error)
        self.prediction_thread.start()

    def on_prediction_error(self, message: str) -> None:
        self.log(f"❌ Prediction failed: {message}")
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Prediction failed")

    def on_prediction_complete(self, results: dict) -> None:
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        is_asd = results["prediction"] == "ASD DETECTED"
        accent = Theme.DANGER if is_asd else Theme.SUCCESS
        accent_dim = Theme.DANGER_DIM if is_asd else Theme.SUCCESS_DIM
        verdict = "ASD Detected" if is_asd else "Neurotypical"
        tagline = (
            "Elevated spectral markers consistent with ASD profile"
            if is_asd
            else "No significant ASD spectral markers detected"
        )

        self.results_label.setText(f"""
            <div style="text-align:center;">
                <div style="
                    display:inline-block;
                    padding:4px 12px;
                    border-radius:20px;
                    background:{accent_dim};
                    color:{accent};
                    font-size:11px;
                    font-weight:700;
                    letter-spacing:1.5px;
                    margin-bottom:10px;
                ">CLASSIFICATION</div>

                <div style="
                    font-size:22px;
                    font-weight:800;
                    color:{accent};
                    letter-spacing:0.5px;
                    margin-bottom:4px;
                ">{verdict}</div>

                <div style="
                    font-size:11px;
                    color:{Theme.TEXT_MUTED};
                    margin-bottom:16px;
                ">{tagline}</div>

                <table style="width:100%; margin:0 auto; border-collapse:collapse;">
                  <tr>
                    <td style="padding:8px 6px; color:{Theme.TEXT_MUTED}; font-size:12px;">Confidence</td>
                    <td style="padding:8px 6px; color:{Theme.TEXT}; font-weight:700; text-align:right;">
                        {results['confidence']:.1%}
                    </td>
                  </tr>
                  <tr style="border-top:1px solid {Theme.BORDER};">
                    <td style="padding:8px 6px; color:{Theme.TEXT_MUTED}; font-size:12px;">ASD probability</td>
                    <td style="padding:8px 6px; color:{Theme.DANGER}; font-weight:700; text-align:right;">
                        {results['asd_probability']:.1%}
                    </td>
                  </tr>
                  <tr style="border-top:1px solid {Theme.BORDER};">
                    <td style="padding:8px 6px; color:{Theme.TEXT_MUTED}; font-size:12px;">Neurotypical</td>
                    <td style="padding:8px 6px; color:{Theme.SUCCESS}; font-weight:700; text-align:right;">
                        {results['control_probability']:.1%}
                    </td>
                  </tr>
                  <tr style="border-top:1px solid {Theme.BORDER};">
                    <td style="padding:8px 6px; color:{Theme.TEXT_MUTED}; font-size:12px;">Epochs processed</td>
                    <td style="padding:8px 6px; color:{Theme.TEXT}; font-weight:700; text-align:right;">
                        {results['n_epochs_processed']}
                    </td>
                  </tr>
                </table>
            </div>
        """)

        if "data" in results:
            sample_data = results["data"][:PLOT_CHANNEL_COUNT, :PLOT_SAMPLE_COUNT]
            self.plot_widget.plot_raw_eeg(
                sample_data, results["sfreq"], results["ch_names"][:PLOT_CHANNEL_COUNT]
            )

        self.add_to_history(results)
        self.status_bar.showMessage(
            f"Analysis complete · {results['prediction']} · {results['confidence']:.1%}"
        )

    # =================================================================
    # TRAIN TAB
    # =================================================================
    def create_train_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("ASD Model Training")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        info_card, info_layout = self._make_card(
            "TRAINING PIPELINE", "SHEFFIELD · RANDOM FOREST"
        )
        info_label = QLabel("""
        <p><b>Dataset:</b> Sheffield Resting-State EEG</p>
        <p><b>Subjects:</b> 56 (28 ASD, 28 Control)</p>
        <p><b>Features:</b> Spectral power (delta, theta, alpha, beta),
        theta/beta ratio, alpha asymmetry</p>
        <p><b>Model:</b> Random Forest Classifier with 5-fold cross-validation</p>
        <p><b>Training time:</b> Approximately 3-5 minutes</p>
        """)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_card)

        self.train_btn = QPushButton("🚀 START TRAINING")
        self.train_btn.setObjectName("TrainButton")
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        layout.addWidget(self.train_progress)

        layout.addWidget(QLabel("Training Log:"))
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log)

        layout.addStretch()
        return widget

    def train_model(self) -> None:
        if self.training_thread and self.training_thread.isRunning():
            self.train_log.append("⚠️ Training already in progress.")
            return

        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_log.clear()

        data_dir = RAW_DATA_DIR / "sheffield_asd_eeg"
        if not data_dir.exists():
            self.train_log.append("❌ Sheffield dataset not found!")
            self.train_log.append(f"   Expected at: {data_dir}")
            self.train_btn.setEnabled(True)
            self.train_progress.setVisible(False)
            return

        self.training_thread = TrainingThread(data_dir)
        self.training_thread.log_signal.connect(self.train_log.append)
        self.training_thread.progress_signal.connect(self.train_progress.setValue)
        self.training_thread.finished_signal.connect(self.on_training_complete)
        self.training_thread.error_signal.connect(self.on_training_error)
        self.training_thread.start()

    def on_training_error(self, message: str) -> None:
        self.train_log.append(f"❌ Training failed: {message}")
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.status_bar.showMessage("Training failed")

    def on_training_complete(self, results: dict) -> None:
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.status_bar.showMessage(
            f"✅ Training complete! Accuracy: {results['accuracy']:.1%}"
        )
        self.check_for_saved_model()

    # =================================================================
    # HISTORY TAB
    # =================================================================
    def create_history_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Prediction History")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        stats_card, stats_layout = self._make_card("SUMMARY")
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        layout.addWidget(stats_card)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["Date/Time", "File", "Prediction", "Confidence", "ASD Prob", "Control Prob"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.history_table)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Clear History")
        clear_btn.setObjectName("GhostButton")
        clear_btn.clicked.connect(self.clear_history)
        btn_layout.addWidget(clear_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("GhostButton")
        refresh_btn.clicked.connect(self.reload_history)
        btn_layout.addWidget(refresh_btn)
        layout.addLayout(btn_layout)
        return widget

    def reload_history(self) -> None:
        """Reload history from the database (clears the table first)."""
        self.history_table.setRowCount(0)
        self.load_history_from_database()

    def load_history_from_database(self) -> None:
        predictions = self.db_manager.get_all_predictions()
        for pred in predictions:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(pred["timestamp"]))
            self.history_table.setItem(row, 1, QTableWidgetItem(pred["filename"]))
            self.history_table.setItem(row, 2, QTableWidgetItem(pred["prediction"]))
            self.history_table.setItem(
                row, 3, QTableWidgetItem(f"{pred['confidence']:.1%}")
            )
            self.history_table.setItem(
                row, 4, QTableWidgetItem(f"{pred['asd_probability']:.1%}")
            )
            self.history_table.setItem(
                row, 5, QTableWidgetItem(f"{pred['control_probability']:.1%}")
            )
        self.update_history_stats()

    def update_history_stats(self) -> None:
        stats = self.db_manager.get_statistics()
        self.stats_label.setText(
            f"<b>Total Predictions:</b> {stats['total']} | "
            f"<b>ASD Detected:</b> "
            f"<span style='color:{Theme.DANGER};'>{stats['asd_count']}</span> | "
            f"<b>Neurotypical:</b> "
            f"<span style='color:{Theme.SUCCESS};'>{stats['control_count']}</span> | "
            f"<b>Avg Confidence:</b> {stats['avg_confidence']:.1%}"
        )

    def add_to_history(self, results: dict) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = Path(self.file_label.text()).name

        self.history_table.insertRow(0)
        self.history_table.setItem(0, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(0, 1, QTableWidgetItem(filename))
        self.history_table.setItem(0, 2, QTableWidgetItem(results["prediction"]))
        self.history_table.setItem(
            0, 3, QTableWidgetItem(f"{results['confidence']:.1%}")
        )
        self.history_table.setItem(
            0, 4, QTableWidgetItem(f"{results['asd_probability']:.1%}")
        )
        self.history_table.setItem(
            0, 5, QTableWidgetItem(f"{results['control_probability']:.1%}")
        )

        metadata = self.model_manager.get_model_info()
        db_results = {
            "filename": filename,
            "file_path": self.file_label.text(),
            "prediction": results["prediction"],
            "confidence": results["confidence"],
            "asd_probability": results["asd_probability"],
            "control_probability": results["control_probability"],
            "epochs_processed": results.get("n_epochs_processed", 0),
            "model_accuracy": metadata.get("accuracy", 0.0) if metadata else 0.0,
        }
        self.db_manager.save_prediction(db_results)
        self.update_history_stats()
        self.log("💾 Saved to database")

    def clear_history(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to delete all prediction history?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.history_table.setRowCount(0)
            count = self.db_manager.clear_all_predictions()
            self.update_history_stats()
            self.log(f"🗑️ Deleted {count} predictions from database")

    # =================================================================
    # MODEL STATUS
    # =================================================================
    def check_for_saved_model(self) -> None:
        if self.model_manager.model_exists():
            try:
                _model, _scaler, metadata = self.model_manager.load_model()
                if metadata and "accuracy" in metadata:
                    self.status_bar.showMessage(
                        f"✅ ASD Model loaded! Accuracy: {metadata['accuracy']:.1%}"
                    )
                else:
                    self.status_bar.showMessage("✅ ASD Model loaded successfully!")
                self.log("✅ Pre-trained ASD model loaded successfully!")
                if metadata:
                    self.log(f"   Model type: {metadata.get('model_type', 'N/A')}")
                    self.log(
                        f"   Accuracy: {metadata.get('accuracy', 0):.3f}"
                    )
                    self.log(f"   Samples: {metadata.get('n_samples', 'N/A')}")
                    self.log(
                        f"   ASD/Control: "
                        f"{metadata.get('n_asd', 'N/A')}/"
                        f"{metadata.get('n_control', 'N/A')}"
                    )
            except Exception as exc:
                self.status_bar.showMessage(f"⚠️ Error loading model: {exc}")
                self.log(f"⚠️ Could not load saved model: {exc}")
        else:
            self.status_bar.showMessage("ℹ️ No ASD model found. Go to 'Train Model' tab.")
            self.log("ℹ️ No trained ASD model found.")
            self.log("   Please train the model first.")

        self.update_model_status()

    def update_model_status(self) -> None:
        if self.model_manager.model_exists():
            metadata = self.model_manager.get_model_info()
            if metadata:
                self.model_status_label.setText(f"""
                    <div style="line-height:1.6;">
                      <span style="color:{Theme.SUCCESS}; font-weight:700;">● Ready</span>
                      <span style="color:{Theme.TEXT_DIM};"> · RandomForest</span><br>
                      <span style="color:{Theme.TEXT_MUTED};">Accuracy</span>
                      <span style="color:{Theme.TEXT}; font-weight:700;">
                          {metadata.get('accuracy', 0):.1%}
                      </span><br>
                      <span style="color:{Theme.TEXT_MUTED};">Cohort</span>
                      <span style="color:{Theme.TEXT};">
                          {metadata.get('n_asd', '–')} ASD / {metadata.get('n_control', '–')} CTL
                      </span>
                    </div>
                """)
            else:
                self.model_status_label.setText(
                    f"<span style='color:{Theme.SUCCESS}; font-weight:700;'>● Ready</span>"
                )
        else:
            self.model_status_label.setText(
                f"<span style='color:{Theme.DANGER}; font-weight:700;'>● Not trained</span><br>"
                f"<span style='color:{Theme.TEXT_MUTED}; font-size:12px;'>"
                f"Go to Train Model tab</span>"
            )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()