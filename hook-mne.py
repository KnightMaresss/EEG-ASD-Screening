# hook-mne.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all MNE data files
datas = collect_data_files('mne')

# Collect all MNE submodules
hiddenimports = collect_submodules('mne')

# Add specific hidden imports
hiddenimports += [
    'mne.io',
    'mne.io.eeglab',
    'mne.preprocessing',
    'mne.epochs',
    'mne.viz',
    'mne.channels',
    'mne.utils',
    'mne.defaults',
    'lazy_loader',
]