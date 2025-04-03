# %%
from __future__ import annotations

import os
import os.path as op
import sys

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp
from tqdm import tqdm

from src.rsa import pre
sys.path.append('..')


sys.path.append('..')

font_path = '../assets/Helvetica.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

FONT_SIZE = 12
COLORMAP_NAME = 'Spectral'

DATA_DIR = '../../NOD-MEEG_upload'
FIG_DIR = '../../NOD-MEEG_results/figs'
RES_DATA_DIR = '../../NOD-MEEG_results/data'
GRAND_EPO_DIR = op.join(RES_DATA_DIR, 'grand_epochs')

EPOCH_DIR = op.join('derivatives', 'preprocessed', 'epochs')
MEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-MEG', EPOCH_DIR)
EEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-EEG', EPOCH_DIR)

SOIs = ['O', 'T', 'OT', 'all']
CORR_METRIC = 'pearson'
# %%
meg_evos = joblib.load(
    op.join(RES_DATA_DIR, 'class_evokeds', 'meg_evos.pkl'),
)
eeg_evos = joblib.load(
    op.join(RES_DATA_DIR, 'class_evokeds', 'eeg_evos.pkl'),
)

meta_stim = pd.read_csv(
    f'{DATA_DIR}/NOD-stimulus/meta.csv',
    low_memory=False,
)
clss = [evo.comment for evo in meg_evos]

sup_clss_map = {}
for i in clss:
    sup_clss_map.setdefault(
        meta_stim[meta_stim['class_id'] == i]['super_class'].unique()[0], [],
    ).append(i)
# %%
meg_rdms = pre.generate_rdms(
    evos=meg_evos,
    sois=SOIs,
    is_spatiotemporal=False,
    n_jobs=2,
    metric=CORR_METRIC,
)

eeg_rdms = pre.generate_rdms(
    evos=eeg_evos,
    sois=SOIs,
    is_spatiotemporal=False,
    n_jobs=2,
    metric=CORR_METRIC,
)

meg_rdm = pre.generate_rdms(
    evos=meg_evos,
    sois=SOIs,
    is_spatiotemporal=True,
    n_jobs=2,
    metric=CORR_METRIC,
)

eeg_rdm = pre.generate_rdms(
    evos=eeg_evos,
    sois=SOIs,
    is_spatiotemporal=True,
    n_jobs=2,
    metric=CORR_METRIC,
)
# %%
ref_rdm = joblib.load(op.join('..', 'assets', 'ref_rdms.pkl'))

all_rdms = {
    '1000map30': sup_clss_map,
    'meg_spatial': meg_rdms,
    'eeg_spatial': eeg_rdms,
    'meg_spatiotemporal': meg_rdm,
    'eeg_spatiotemporal': eeg_rdm,
    'vtc': ref_rdm['vtc'],
    'wordnet': ref_rdm['wordnet'],
}

joblib.dump(all_rdms, op.join(RES_DATA_DIR, 'RDMs.pkl'))
# %%
