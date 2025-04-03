# %%
from __future__ import annotations

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mne
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from mne.io import BaseRaw
from mne_bids import find_matching_paths
from mne_bids import read_raw_bids


SUB = '01'
RUN = '01'
SES = 'ImageNet01'

ROOT = '../../NOD-MEEG_upload'
CLEAN_ROOT = 'derivatives/preprocessed/raw'
CLEAN_MROOT = f'{ROOT}/NOD-MEG/{CLEAN_ROOT}'
CLEAN_EROOT = f'{ROOT}/NOD-EEG/{CLEAN_ROOT}'

SAVE_DIR = '../../NOD-MEEG_results'
FIG_DIR = f'{SAVE_DIR}/figs'

FONTSIZE = 12
COLORMAP = 'Spectral'
NCOLOR = 290  # one channel one color
FONT_PATH = '../assets/Helvetica.ttc'


fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
# %%


def align_raw(
    raw: BaseRaw,
    paras: dict[str, float],
) -> BaseRaw:

    raw = raw.copy().load_data()

    dtype = 'meg' if 'meg' in raw else 'eeg' if 'eeg' in raw else None
    if dtype == 'meg':
        raw = raw.pick_types(meg='mag', ref_meg=False)
    elif dtype == 'eeg':
        raw = raw.pick_types(eeg=True)

    raw.resample(paras['sfreq'])
    raw.filter(paras['hfreq'], paras['lfreq'])

    return raw


# %% initialize data

rawMEG = read_raw_bids(
    find_matching_paths(
        subjects=SUB,
        root=f'{ROOT}/NOD-MEG',
        extensions='.fif',
        runs=RUN,
        sessions=SES,
    )[0],
)

rawEEG = read_raw_bids(
    find_matching_paths(
        subjects=SUB,
        root=f'{ROOT}/NOD-EEG',
        extensions='.set',
        runs=RUN,
        sessions=SES,
    )[0],
)

cleanMEG = mne.io.read_raw_fif(
    f'{CLEAN_MROOT}/sub-{SUB}_ses-{SES}_task-ImageNet_run-{RUN}_meg_clean.fif',
)
cleanEEG = mne.io.read_raw_fif(
    f'{CLEAN_EROOT}/sub-{SUB}_ses-{SES}_task-ImageNet_run-{RUN}_eeg_clean.fif',
)

# %% align MEG/EEG's sampling frequency, band pass for comparison and compute PSD
paras = {
    'lfreq': cleanMEG.info['lowpass'],
    'hfreq': cleanMEG.info['highpass'],
    'sfreq': cleanMEG.info['sfreq'],
}

rawMEG = align_raw(rawMEG, paras)
rawEEG = align_raw(rawEEG, paras)

raw_psdMEG = rawMEG.compute_psd()
raw_psdEEG = rawEEG.compute_psd()

clean_psdMEG = cleanMEG.compute_psd()
clean_psdEEG = cleanEEG.compute_psd()
# %% plot PSD
cmap_raw = cm.get_cmap('Spectral_r', NCOLOR)
cmap_clean = cm.get_cmap('Spectral', NCOLOR)
colors_raw = [cmap_raw(i / (NCOLOR - 1)) for i in range(NCOLOR)]
colors_clean = [cmap_clean(i / (NCOLOR - 1)) for i in range(NCOLOR)]


formatter = ticker.FuncFormatter(
    lambda x, _: f"{x:.1f}" if x == 0.1 else f"{int(x)}",
)


def plot_psd_all_channels(
    psd: mne.time_frequency.Spectrum,  # 直接传入PSD对象
    ax: plt.Axes,
    colors: list = colors,
):

    psd_data = psd.get_data()
    freqs = psd.freqs

    psd_data_db = 10 * np.log10(psd_data)  # PSD -> dB (10 * log10(PSD))
    psd_data_db_mean = np.mean(psd_data_db, axis=1, keepdims=True)
    psd_data_db -= psd_data_db_mean  # 白化PSD

    for i in range(psd_data_db.shape[0]):
        ax.plot(
            freqs, psd_data_db[i, :],
            color=colors[i],
            alpha=0.2,
            linewidth=0.5,
        )


plt.close('all')
fig = plt.figure(figsize=(12, 3), dpi=600)

gs = GridSpec(2, 2)
ax3 = fig.add_subplot(gs[1:, :1])
ax4 = fig.add_subplot(gs[1:, 1:])
ax1 = fig.add_subplot(gs[:1, :1])
ax2 = fig.add_subplot(gs[:1, 1:])

plot_psd_all_channels(raw_psdMEG, ax=ax1, colors=colors_raw)
plot_psd_all_channels(raw_psdEEG, ax=ax2, colors=colors_clean)
plot_psd_all_channels(clean_psdMEG, ax=ax3, colors=colors_clean)
plot_psd_all_channels(clean_psdEEG, ax=ax4, colors=colors_raw)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0.1, 110)
    ax.set_xticks([0.1, 20, 40, 60, 80, 100])  # 设置x轴刻度
    ax.set_ylim(-10, 40)
    ax.xaxis.set_major_formatter(formatter)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ax in [ax1, ax2]:
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
    if ax in [ax3, ax4]:
        ax.set_xlabel('Frequency (Hz)')
    if ax in [ax1, ax3]:
        ax.set_ylabel(r'fT$^2$/Hz (dB)')
    if ax in [ax2, ax4]:
        ax.set_ylabel(r'$\mu$V$^2$/Hz (dB)', labelpad=0)

label_font = FONTSIZE+2
ax1.text(
    0.95, 0.95, 'MEG-raw', transform=ax1.transAxes, fontsize=label_font,
    verticalalignment='top', horizontalalignment='right',
)
ax2.text(
    0.95, 0.95, 'EEG-raw', transform=ax2.transAxes, fontsize=label_font,
    verticalalignment='top', horizontalalignment='right',
)
ax3.text(
    0.95, 0.95, 'MEG-cleaned', transform=ax3.transAxes, fontsize=label_font,
    verticalalignment='top', horizontalalignment='right',
)
ax4.text(
    0.95, 0.95, 'EEG-cleaned', transform=ax4.transAxes, fontsize=label_font,
    verticalalignment='top', horizontalalignment='right',
)

# plt.tight_layout()
plt.show()
fig.savefig(f'{FIG_DIR}/psd.svg', dpi=600, bbox_inches='tight')
# %%
