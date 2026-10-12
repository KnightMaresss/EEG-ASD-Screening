"""Matplotlib widget for EEG visualization in PySide6."""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class MplCanvas(FigureCanvas):
    """Matplotlib canvas embedded in Qt."""
    
    def __init__(self, parent=None, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2d2d2d')
        super().__init__(self.fig)
        self.setParent(parent)
        
    def clear(self):
        """Clear the figure."""
        self.fig.clear()


class EEGPlotWidget(QWidget):
    """Widget for displaying EEG plots."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        # Create canvas
        self.canvas = MplCanvas(self, width=8, height=5, dpi=100)
        
        # Add toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Style
        plt.style.use('dark_background')
        
    def plot_erp(self, erp_data: dict, channel_names: list = None):
        """
        Plot Event-Related Potentials.
        
        Args:
            erp_data: Dictionary with target_erp, nontarget_erp, time_points
            channel_names: List of channel names
        """
        self.canvas.clear()
        
        target_erp = erp_data['target_erp']
        nontarget_erp = erp_data['nontarget_erp']
        time_points = erp_data['time_points']
        
        n_channels = target_erp.shape[0]
        
        if channel_names is None:
            channel_names = [f'Channel {i+1}' for i in range(n_channels)]
        
        # Create subplots for each channel
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        for ch in range(n_channels):
            ax = self.canvas.fig.add_subplot(n_rows, n_cols, ch + 1)
            ax.set_facecolor('#1e1e1e')
            
            # Plot target and non-target ERPs
            ax.plot(time_points, target_erp[ch, :], '#ff4444', label='Target', linewidth=1.5)
            ax.plot(time_points, nontarget_erp[ch, :], '#4488ff', label='Non-Target', linewidth=1.5)
            
            # Add vertical line at stimulus onset
            ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
            
            # Highlight P300 window
            ax.axvspan(0.250, 0.500, alpha=0.2, color='yellow')
            
            ax.set_title(channel_names[ch], fontsize=10, color='white')
            ax.set_xlabel('Time (s)', fontsize=8, color='white')
            ax.set_ylabel('Amplitude (µV)', fontsize=8, color='white')
            ax.tick_params(labelsize=8, colors='white')
            
            if ch == 0:
                ax.legend(fontsize=8, facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def plot_raw_eeg(self, data: np.ndarray, sampling_rate: int, channel_names: list = None):
        """
        Plot raw EEG data with improved visibility.
        
        Args:
            data: EEG data [channels x time]
            sampling_rate: Sampling rate in Hz
            channel_names: List of channel names
        """
        self.canvas.clear()
        
        n_channels = data.shape[0]
        n_samples = data.shape[1]
        time_points = np.arange(n_samples) / sampling_rate
        
        if channel_names is None:
            channel_names = [f'Channel {i+1}' for i in range(n_channels)]
        
        n_rows = min(n_channels, 8)  # Show up to 8 channels
        
        # Create subplots with spacing
        self.canvas.fig.subplots_adjust(hspace=0.4, left=0.12, right=0.95, top=0.92, bottom=0.08)
        
        # Convert to microvolts for better readability
        data_uv = data * 1e6
        
        for ch in range(n_rows):
            ax = self.canvas.fig.add_subplot(n_rows, 1, ch + 1)
            ax.set_facecolor('#1e1e1e')
            
            # Plot EEG with better visibility
            ax.plot(time_points[:5000], data_uv[ch, :5000], '#00ff88', linewidth=0.8)
            
            # Add channel name and units
            ax.set_ylabel(f'{channel_names[ch]}\n(µV)', fontsize=8, color='white')
            
            # Add horizontal grid
            ax.grid(True, alpha=0.2, color='white')
            
            # Set y-limits based on data variation
            std_val = np.std(data_uv[ch, :])
            ax.set_ylim(-3 * std_val, 3 * std_val)
            
            # Add zero line
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
            
            ax.tick_params(labelsize=7, colors='white')
            
            if ch < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=9, color='white')
        
        self.canvas.fig.suptitle('Resting-State EEG Signal', fontsize=12, color='white', fontweight='bold')
        self.canvas.draw()
        
    def plot_feature_importance(self, feature_names: list, importance_scores: np.ndarray, top_n: int = 10):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            top_n: Number of top features to show
        """
        self.canvas.clear()
        
        # Sort features by importance
        indices = np.argsort(importance_scores)[-top_n:]
        
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        y_pos = np.arange(len(indices))
        bars = ax.barh(y_pos, importance_scores[indices], color='#4CAF50')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8, color='white')
        ax.set_xlabel('Importance', fontsize=10, color='white')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, axis='x')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def plot_confusion_matrix(self, cm: np.ndarray, classes: list):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            classes: Class names
        """
        self.canvas.clear()
        
        import seaborn as sns
        
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    annot_kws={'color': 'white', 'fontsize': 12},
                    cbar_kws={'ticks_color': 'white'})
        
        ax.set_xlabel('Predicted', fontsize=10, color='white')
        ax.set_ylabel('Actual', fontsize=10, color='white')
        ax.set_title('Confusion Matrix', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        
        # Fix colorbar text color
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def plot_band_powers(self, band_powers: dict, title: str = "Frequency Band Powers"):
        """
        Plot frequency band powers.
        
        Args:
            band_powers: Dictionary with band names and power values
            title: Plot title
        """
        self.canvas.clear()
        
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        bands = list(band_powers.keys())
        powers = list(band_powers.values())
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#f44336', '#9C27B0']
        
        bars = ax.bar(bands, powers, color=colors[:len(bands)])
        ax.set_ylabel('Power', fontsize=10, color='white')
        ax.set_title(title, fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add value labels on bars
        for bar, power in zip(bars, powers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{power:.3f}', ha='center', va='bottom', fontsize=8, color='white')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()