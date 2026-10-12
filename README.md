# EEG ASD Screening Tool

A desktop application that classifies resting-state EEG recordings as
**ASD (Autism Spectrum Disorder)** or **neurotypical** using spectral
features extracted from the signal and a Random Forest classifier.

Built with Python, PySide6, MNE-Python, and scikit-learn.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Building a Standalone Executable](#building-a-standalone-executable)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This tool provides an end-to-end pipeline for ASD screening from
resting-state EEG data. It includes:

- A **training workflow** that loads the Sheffield dataset, extracts
  spectral features, trains a Random Forest classifier, and persists
  the model to disk.
- A **prediction workflow** that loads a single `.set` EEG file, runs
  it through the same preprocessing pipeline used during training, and
  outputs a classification with confidence scores.
- A **local history** of all predictions stored in a SQLite database
  so results can be reviewed later.

The application is built around a single shared preprocessing pipeline
(`src/core/preprocessing/pipeline.py`) to guarantee that the model sees
identical feature distributions during training and inference — a
common source of silent errors in EEG-based ML projects.

---

## Features

### Training
- Loads 56-subject Sheffield dataset (28 ASD, 28 Control)
- Segments continuous EEG into fixed-length epochs (2 s each)
- Applies a 1–50 Hz bandpass filter
- Extracts 19 spectral features per epoch:
  - Mean/std/max power in delta, theta, alpha, beta bands
  - Theta/Beta ratio
  - Alpha/Theta ratio
  - Frontal theta power
  - Global mean, std, max amplitude
- Trains a Random Forest (100 trees, max depth 10)
- 5-fold stratified cross-validation
- Saves model, scaler, and metadata to `models/`

### Prediction
- Loads any EEGLAB `.set` file
- Applies the exact same filter + epoch + feature pipeline as training
- Outputs ASD vs. Neurotypical classification
- Reports per-class confidence and probability
- Plots the first 8 channels × 10 s of EEG
- Saves result to local SQLite database

### UI
- Three-tab layout: **Predict**, **Train Model**, **History**
- Dark clinical theme with design tokens centralized in `theme.py`
- Background threads for both training and prediction
  (UI never freezes)
- Live log console for every stage of the pipeline
- Result card with color-coded verdict (red = ASD, green = neurotypical)
- Prediction history table with summary statistics

---

## Screenshots

> Screenshots live in `assets/screenshots/`. See the
> [Where to add screenshots](#where-to-add-screenshots) section below
> for details on placing your own.

### Prediction tab
![Prediction tab](assets/screenshots/predict-tab.png)

### Training tab
![Training tab](assets/screenshots/train-tab.png)

### History tab
![History tab](assets/screenshots/history-tab.png)

### Sample EEG plot
![EEG plot](assets/screenshots/eeg-plot.png)

### Result card
![Result card](assets/screenshots/result-card.png)

---

## Requirements

### Software
- **Python 3.10 or newer** (tested on 3.11)
- **pip** (bundled with Python)
- **Git** (to clone the repo)
- **Windows 10/11**, **macOS 12+**, or **Linux** (tested on Ubuntu 22.04)

### Python dependencies

Installed automatically via `pip install -r requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `PySide6` | Qt6 desktop UI framework |
| `mne` | EEG/MEG data loading and signal processing |
| `numpy` | Numerical arrays |
| `scipy` | Signal processing (Welch PSD) |
| `scikit-learn` | Random Forest, cross-validation, scaling |
| `matplotlib` | EEG plotting widget |
| `joblib` | Model serialization |
| `pandas` | Tabular manipulation (transitive) |
| `seaborn` | Figure generation for paper/reports |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/KnightMares/eeg-asd-screening.git
cd eeg-asd-screening