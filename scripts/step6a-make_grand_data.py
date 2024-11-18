#%%
import sys
sys.path.append('..')

import os
import os.path as op
import mne
from tqdm import tqdm

import sys
sys.path.append('..')
from src.rsa import pre

DATA_DIR = '../../NOD-MEEG_upload'
RES_DATA_DIR = '../../NOD-MEEG_results/data'
GRAND_EPO_DIR = op.join(RES_DATA_DIR, 'grand_epochs')

EPOCH_DIR = op.join('derivatives', 'preprocessed', 'epochs')
MEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-MEG', EPOCH_DIR)
EEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-EEG', EPOCH_DIR)

os.makedirs(GRAND_EPO_DIR, exist_ok=True)
#%%
meg_epoch_files = {
    filename.split('-')[1][:2]: op.join(MEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(MEG_EPOCH_ROOT))
}
eeg_epoch_files = {
    filename.split('-')[1][:2]: op.join(EEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(EEG_EPOCH_ROOT))
}
#%% make grand m/eeg epochs
meg_epo:mne.Epochs = pre.make_grand_epo(
    list(meg_epoch_files.values()), 
    'info_with_data', 
    align_to = 0,
    n_jobs = 30)

eeg_epo:mne.Epochs = pre.make_grand_epo(
    list(eeg_epoch_files.values()),
    'info_with_data',
    align_to = 0,
    n_jobs = 30)

meg_epo.save(op.join(GRAND_EPO_DIR, 'meg_grand_epo.fif'))
eeg_epo.save(op.join(GRAND_EPO_DIR, 'eeg_grand_epo.fif'))

# meg_epo:mne.Epochs = mne.read_epochs(op.join(GRAND_EPO_DIR, 'meg_grand_epo.fif'))
# eeg_epo:mne.Epochs = mne.read_epochs(op.join(GRAND_EPO_DIR, 'eeg_grand_epo.fif'))
#%% make m/eeg class rdm
meta = meg_epo.metadata
sorted_meta = meta[meta['task'] == 'ImageNet'].sort_values(
    by=['stim_is_animate', 'super_class', 'class_id']
    )
sorted_class = sorted_meta['class_id'].unique()

meg_evos = pre.make_grand_evos(
    epochs = meg_epo,
    conditions = sorted_class,
    key = 'class_id',
    n_jobs = 4
)

eeg_evos = pre.make_grand_evos(
    epochs = eeg_epo,
    conditions = sorted_class,
    key = 'class_id',
    n_jobs = 4
)

for name, evos in zip(['eeg', 'meg'], [eeg_evos, meg_evos]):
    joblib.dump(evos, op.join(RES_DATA_DIR, 'class_evokeds', f'{name}_evos.pkl'))

#%%
