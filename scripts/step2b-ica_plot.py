
#%%
from mne_bids import find_matching_paths, read_raw_bids
import mne
from mne.io import BaseRaw
import sys
sys.path.append('..')
from src import viz
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


SUB = '01'
RUN = '01'
SES = 'ImageNet01'

ROOT = '../../NOD-MEEG_upload'
CLEAN_ROOT = 'derivatives/preprocessed/raw'
CLEAN_MROOT = f'{ROOT}/NOD-MEG/{CLEAN_ROOT}'
CLEAN_EROOT = f'{ROOT}/NOD-EEG/{CLEAN_ROOT}'

SAVE_DIR = '../../NOD-MEEG_results'
FIG_DIR = f'{SAVE_DIR}/figs'

FONT_PATH = '../assets/Helvetica.ttc'
T_MIN = 50
T_MAX = 65

fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
palette = sns.color_palette(palette='tab20', n_colors=20,desat=1)
colors  = [palette[2] ,palette[6] ,palette[4]]
#%% io
rawMEG = read_raw_bids(
    find_matching_paths(subjects=SUB, 
                        root=f'{ROOT}/NOD-MEG',
                        extensions='.fif',
                        runs=RUN,
                        sessions=SES,
                        )[0]
)

rawEEG = read_raw_bids(
    find_matching_paths(subjects=SUB, 
                        root=f'{ROOT}/NOD-EEG',
                        extensions='.set',
                        runs=RUN,
                        sessions=SES,
                        )[0]
)

cleanMEG = mne.io.read_raw_fif(
    f'{CLEAN_MROOT}/sub-{SUB}_ses-{SES}_task-ImageNet_run-{RUN}_meg_clean.fif'
    )
cleanEEG = mne.io.read_raw_fif(
    f'{CLEAN_EROOT}/sub-{SUB}_ses-{SES}_task-ImageNet_run-{RUN}_eeg_clean.fif'
    )

#%% align MEG/EEG's sampling frequency, band pass for comparison and compute PSD
paras = {
    'lfreq': cleanMEG.info['lowpass'],
    'hfreq': cleanMEG.info['highpass'],
    'sfreq': cleanMEG.info['sfreq'],
}

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

rawMEG = align_raw(rawMEG, paras)
rawEEG = align_raw(rawEEG, paras)

#%% 
# To illustrate artifact components, here we extremely decompose the raw data into 100 components.
# This way, artifacts will be very clear, but there might be multiple same artifacts in one run.
# This is a controversial issue, so this approach is only for demonstrat the artifact topomap.
rawMEG_ = rawMEG.copy().filter(1, 40).resample(20)
ica = mne.preprocessing.ICA(n_components=100, 
                            method = 'infomax',
                            random_state=97,
                            )
ica.fit(rawMEG_)
#%%

def component_(ica, comp):
    return np.dot(
        ica.mixing_matrix_[:, comp].T, 
        ica.pca_components_[:ica.n_components_])

def sources_(ica, raw, comp, tlim=(20, 35)):
    ica_ts = ica.get_sources(raw)._data
    mean = np.mean(ica_ts, axis=1, keepdims=True)
    std = np.std(ica_ts, axis=1, keepdims=True)
    ica_ts = (ica_ts - mean) / std
    time_slice = slice(*raw.time_as_index(tlim))
    return ica_ts[comp][time_slice], raw.times[time_slice]

def extract_raw_data(raw, tlim=(20, 35), picks=None):
    raw_copy = raw.copy().pick(picks)
    time_slice = slice(*raw_copy.time_as_index(tlim))
    data = raw_copy.get_data()[:, time_slice]
    
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data = (data - mean) / std
    times = raw_copy.times[time_slice]
    
    return data, times

def _get_meg_pos(epo):
    ch_name_picks = mne.pick_channels_regexp(
        epo.ch_names, regexp='M[LRZ]...')
    type_picks = mne.pick_types(
        epo.info, meg=True)
    picks = np.intersect1d(ch_name_picks, type_picks)
    intercorr_chn = [epo.ch_names[idx][:5] for idx in picks]
    original_meg_layout = mne.channels.find_layout(epo.info, ch_type='meg')
    exclude_list = [x for x in original_meg_layout.names if x[:5] not in intercorr_chn]
    meg_layout = mne.channels.find_layout(epo.info, ch_type='meg', exclude=exclude_list)
    pos = meg_layout.pos[:, :2]
    layout_pos = pos - pos.mean(axis=0)
    return layout_pos

def make_main_plot(T_MIN, 
                   T_MAX, 
                   fontsize, 
                   ica, 
                   raw4plot, 
                   arti_data, 
                   colors, 
                   pre_data, 
                   pos_data, 
                   Epre_data, 
                   Epos_data, 
                   picks_ch, 
                   Epicks_ch):
    plt.close('all')
    gx, gy, fx, fy = 6, 12, 12, 3
    gs = GridSpec(gx, gy, wspace=1, hspace=2)
    fig = plt.figure(figsize=(fx, fy), dpi=600)

    # ICA components
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[2:4, 0:2])
    ax3 = fig.add_subplot(gs[4:6, 0:2])
    # ICA sources
    ax4 = fig.add_subplot(gs[0:2, 2:5])
    ax5 = fig.add_subplot(gs[2:4, 2:5])
    ax6 = fig.add_subplot(gs[4:6, 2:5])
    # raw data
    ax7 = fig.add_subplot(gs[0:3, 5:])
    ax8 = fig.add_subplot(gs[3:6, 5:])

    # ICA components
    for ax, (name, comp) in zip([ax1, ax2, ax3], arti_data.items()):
        mne.viz.plot_topomap(component_(ica, comp), _get_meg_pos(raw4plot),
                             cmap='RdBu_r', sphere=(0, 0, 0, 0.5), axes=ax, sensors=False,
                             res=200, contours=0, show=False)

        ax.set_title(name, va='center', ha='center', pad=-20)
        ax.set_adjustable('datalim')
        pos = ax.get_position()
        new_pos = [0.16, pos.y0-0.05, pos.width, pos.height]
        ax.set_position(new_pos)

    # ICA sources
    for artifact, ax in zip(arti_data.keys(), [ax4, ax5, ax6]):
        ica_artifact = sources_(ica, raw4plot, arti_data[artifact], tlim=(T_MIN, T_MAX))
        ax.plot(ica_artifact[1] - T_MIN, ica_artifact[0], color=colors[0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(0, T_MAX - T_MIN)
        ax.set_xticks(np.arange(0, T_MAX - T_MIN + 5, 5))
        if ax != ax6:
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
        else:
            ax.set_xlabel('Time (s)', fontsize=fontsize)

    # raw data
    offset = 10
    axes = [ax7, ax8]
    for meg_data, eeg_data, ax in zip([pre_data, pos_data], [Epre_data, Epos_data], axes):
        # MEG
        for i in range(len(meg_data[0])):
            ax.plot(meg_data[1] - T_MIN, meg_data[0][i] + i * offset, color=colors[1] if ax == ax7 else colors[2])
            ax.text(meg_data[1][-1] - T_MIN, meg_data[0][i][-1] + i * offset + 3,
                    f'MEG-{picks_ch[i].split("-")[0]}', va='center', ha='right', color='black', fontsize=fontsize-4)
        # EEG
        eeg_offset = len(meg_data[0]) * offset
        for i in range(len(eeg_data[0])):
            ax.plot(eeg_data[1] - T_MIN, eeg_data[0][i] + eeg_offset + i * offset, color=colors[1] if ax == ax7 else colors[2])
            ax.text(eeg_data[1][-1] - T_MIN, eeg_data[0][i][-1] + eeg_offset + i * offset + 4,
                    f'EEG-{Epicks_ch[i]}', va='center', ha='right', color='black', fontsize=fontsize-4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(0, T_MAX - T_MIN)
        ax.set_xticks(np.arange(0, T_MAX - T_MIN + 5, 5))
        if ax != ax8:
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
        else:
            ax.set_xlabel('Time (s)', fontsize=fontsize)

    ax7.set_title('Raw data', pad=0, fontsize=fontsize)
    ax8.set_title('Cleaned data', pad=0, fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    return fig

artifacts = {
    'eye blink':67,
    'eye movement':0,
    'heart beat':63,
}
picks_ch = ['MLF14-4504', 'MRF14-4504']
Epicks_ch = ['Fp1', 'Fp2']

ch_indx = [rawMEG.ch_names.index(ch) for ch in picks_ch]
Ech_indx = [rawEEG.ch_names.index(ch) for ch in Epicks_ch]

pre_data = extract_raw_data(rawMEG, picks=ch_indx, tlim=(T_MIN, T_MAX))
pos_data = extract_raw_data(cleanMEG, picks=ch_indx, tlim=(T_MIN, T_MAX))
Epre_data = extract_raw_data(rawEEG, picks=Ech_indx, tlim=(T_MIN, T_MAX))
Epos_data = extract_raw_data(cleanEEG, picks=Ech_indx, tlim=(T_MIN, T_MAX))

fig = make_main_plot(T_MIN, 
               T_MAX, 
               fontsize = 12, 
               ica = ica, 
               raw4plot = rawMEG_, 
               arti_data = artifacts, 
               colors = colors,
               pre_data = pre_data,
               pos_data = pos_data,
               Epre_data = Epre_data,
               Epos_data = Epos_data,
               picks_ch = picks_ch,
               Epicks_ch = Epicks_ch)


fig.savefig(f'{FIG_DIR}/ica.svg',dpi=600, bbox_inches = 'tight')
#%%
