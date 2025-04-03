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
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp
from tqdm import tqdm

from src.decoding import cls_2epo
sys.path.append('..')


font_path = '../assets/Helvetica.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

FONT_SIZE = 12
COLORMAP_NAME = 'Spectral'

DATA_DIR = '../../NOD-MEEG_upload'
FIG_DIR = '../../NOD-MEEG_results/figs'
RES_DATA_DIR = '../../NOD-MEEG_results/data'
EPOCH_DIR = op.join('derivatives', 'preprocessed', 'epochs')
MEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-MEG', EPOCH_DIR)
EEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-EEG', EPOCH_DIR)
# %%


def make_2epos(
    epo_files: dict,
    subject: str,
    to_select: str,
) -> list[mne.Epochs]:
    epos = mne.read_epochs(epo_files[subject])
    ani_epo = epos[epos.metadata[to_select] == True]
    inani_epo = epos[epos.metadata[to_select] == False]
    if len(ani_epo) == 0 or len(inani_epo) == 0:
        ani_epo = epos[epos.metadata[to_select] == 'True']
        inani_epo = epos[epos.metadata[to_select] == 'False']
    return [ani_epo, inani_epo]


def svm_all_sub(
    epo_files: dict[str, str],
    n_sample: int = 1000,
    cv: int = 10,
    SOI: str = 'OT',
    metric: str = 'accuracy',
    n_jobs: int = 16,
    C: float = 0.5,
    tol: float = 0.01,
    to_select: str = 'stim_is_animate',
    pca: bool = False,
) -> dict[str, np.ndarray]:

    scores = {}
    for sub in tqdm(epo_files.keys(), desc='Decoding subject'):
        epos = make_2epos(epo_files, sub, to_select)
        score = cls_2epo(
            epos,
            n_sample=n_sample,
            cv=cv,
            SOI=SOI,
            metric=metric,
            n_jobs=n_jobs,
            C=C,
            tol=tol,
            pca=pca,
            plot=False,
        )
        scores[sub] = score
    return scores


def make_combined_plot(
    np_Mscore,
    np_Escore,
    times,
    cmap='Spectral',
    fontsize=FONT_SIZE,
    title: str = 'Animacy Decoding Accuracy (%)',
    vmin=40,
    vmax=70,
) -> plt.Figure:
    times = times * 1000
    mean_Mscores = np.mean(np_Mscore, axis=0)*100
    mean_Escores = np.mean(np_Escore, axis=0)*100
    Msig_times = times[diff_test_1sample(np_Mscore, np_Mscore.shape[1])[2]]
    Esig_times = times[diff_test_1sample(np_Escore, np_Escore.shape[1])[2]]
    print(
        f'MEG has {len(Msig_times)} sig times, EEG has {len(Esig_times)} sig times',
    )
    Minitil_sig_t = Msig_times[0]
    Einitil_sig_t = Esig_times[0]
    Mmax_performance_t = times[np.argmax(mean_Mscores)]
    Emax_performance_t = times[np.argmax(mean_Escores)]
    print(
        f'MEG intial_sig_time:{Minitil_sig_t},max_performance{np.max(mean_Mscores)} at {Mmax_performance_t}ms',
    )
    print(
        f'EEG intial_sig_time:{Einitil_sig_t},max_performance{np.max(mean_Escores)} at {Emax_performance_t}ms',
    )

    plt.close('all')
    gs = GridSpec(6, 6)
    fig = plt.figure(figsize=(7, 3))

    ax1 = fig.add_subplot(gs[:, :4])
    ax2 = fig.add_subplot(gs[:3, 4:])
    ax3 = fig.add_subplot(gs[3:, 4:])

    # ax1.set_yticks([50, 55, 60])
    ax1.set_xlim([-100, 800])
    ax1.set_xticks([0, 250, 500, 750])

    ax1.axhline(50, color='k', linestyle='--')
    ax1.plot(times, mean_Mscores, label='MEG', color=colors[0], lw=2)
    ax1.plot(times, mean_Escores, label='EEG', color=colors[1], lw=2)

    min_y = min(np.min(mean_Mscores), np.min(mean_Escores))
    max_y = max(np.max(mean_Mscores), np.max(mean_Escores))
    scatter_space = (max_y - 47)/12

    yticks = np.arange(np.floor(min_y / 5) * 5, np.ceil(max_y / 5) * 5 + 1, 5)
    ax1.set_yticks(yticks)
    scatter_y = np.linspace(50-scatter_space, 50, 5)
    meg_y = scatter_y[2]
    eeg_y = scatter_y[0]

    ax1.scatter(
        Msig_times, [meg_y] * len(Msig_times),
        color=colors[0], marker='o', s=10, alpha=0.35,
    )  # 修改这里
    ax1.scatter(
        Esig_times, [eeg_y] * len(Esig_times),
        color=colors[1], marker='o', s=10, alpha=0.35,
    )  # 修改这里

    ax1.set_xlabel('Time (ms)', fontsize=fontsize)
    ax1.set_ylabel(title, fontsize=fontsize)
    ax1.legend(loc='upper left', fontsize=fontsize)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for i, (decoaccuracy, ax, title) in enumerate(zip([np_Mscore, np_Escore], [ax2, ax3], ['MEG', 'EEG'])):
        decoaccuracy = decoaccuracy * 100
        im = ax.imshow(
            decoaccuracy, aspect='auto', cmap='coolwarm',
            extent=[times[0], times[-1], 0.5, decoaccuracy.shape[0] + 0.5],
            interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax,
        )
        # ax.set_yticks([])
        # ax.set_yticklabels([])
        if title == 'EEG':
            ax.set_xticks([0, 250, 500, 750])
            ax.set_xticklabels([0, 250, 500, 750], fontsize=fontsize-2)
            ax.set_xlabel('Time (ms)', fontsize=fontsize)
            eeg_subs = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 24, 26, 27, 29, 30,
            ]
            ax.set_yticks(np.arange(1, len(eeg_subs) + 1, 3))
            ax.set_yticklabels(
                eeg_subs[::3], fontsize=fontsize-5.5, rotation=0, ha='right',
            )
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.set_yticks(np.arange(1, decoaccuracy.shape[0] + 1, 3))
            ax.set_yticklabels(
                np.arange(1, decoaccuracy.shape[0] + 1, 3),
                fontsize=fontsize-5.5, rotation=0, ha='right',
            )
        ax.set_ylabel(f'{title} Subject', fontsize=fontsize)

    # Adjust the position and size of the colorbar
    cbar_ax = fig.add_axes([1, 0.2, 0.01, 0.75])
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Animacy Decoding Accuracy (%)', fontsize=fontsize)
    plt.tight_layout()
    return fig


def diff_test_1sample(scores, n_timepoints, baseline=0.5, alpha=0.01, tail='greater'):
    """
    Perform a differential test using single-sample t-test for each time point.

    Parameters
    ----------
    scores : ndarray
        Array of shape (n_samples, n_timepoints) containing the scores.
    n_timepoints : int
        Number of time points.
    baseline : float, optional
        Baseline value for the t-test. Defaults to 0.5.
    alpha : float, optional
        Significance level for the t-test. Defaults to 0.01.
    tail : str, optional
        Type of t-test ('two-sided', 'greater', 'less'). Defaults to 'greater'.

    Returns
    -------
    tuple
        A tuple containing the t-values, p-values, and indices of significant time points.
    """
    t_values = []
    p_values = []
    for i in range(n_timepoints):
        t, p = ttest_1samp(scores[:, i], baseline)
        if tail == 'greater':
            p = p / 2 if t > 0 else 1 - p / 2
        elif tail == 'less':
            p = p / 2 if t < 0 else 1 - p / 2

        t_values.append(t)
        p_values.append(p)

    t_values = np.array(t_values)
    p_values = np.array(p_values)

    # Bonferroni correction
    corrected_alpha = alpha / n_timepoints
    significant_timepoints = np.where(p_values < corrected_alpha)[0]
    print(f'corrected p: {corrected_alpha}')

    return t_values, p_values, significant_timepoints


# %% decoding
meg_epoch_files = {
    filename.split('-')[1][:2]: op.join(MEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(MEG_EPOCH_ROOT))
}
eeg_epoch_files = {
    filename.split('-')[1][:2]: op.join(EEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(EEG_EPOCH_ROOT))
}

# %%

ani_meg_scores = svm_all_sub(
    meg_epoch_files,
    to_select='stim_is_animate',
    n_sample=1000,
)
ani_eeg_scores = svm_all_sub(
    eeg_epoch_files,
    to_select='stim_is_animate',
    n_sample=1000,
)
scores_animate = {
    'MEG': ani_meg_scores,
    'EEG': ani_eeg_scores,
}

face_meg_scores = svm_all_sub(
    meg_epoch_files,
    to_select='stim_is_face',
    # use the minimum number of trials, for number of face trials is usually small
    n_sample='min',
)
face_eeg_scores = svm_all_sub(
    eeg_epoch_files,
    to_select='stim_is_face',
    n_sample='min',
)  # use the minimum number of trials, for number of face trials is usually small
scores_face = {
    'MEG': face_meg_scores,
    'EEG': face_eeg_scores,
}

joblib.dump(scores_animate, op.join(RES_DATA_DIR, 'decoding_animacy.pkl'))
joblib.dump(scores_face, op.join(RES_DATA_DIR, 'decoding_face.pkl'))
# %% plotting
scores_animate = joblib.load(op.join(RES_DATA_DIR, 'decoding_animacy.pkl'))
scores_face = joblib.load(op.join(RES_DATA_DIR, 'decoding_face.pkl'))

colormap = 'Spectral'
cmap = plt.get_cmap(colormap, 10)
colors = [cmap(i / (10 - 1)) for i in range(10)]
lower_color = colors[int(len(colors) / 4)]
upper_color = colors[int(3 * len(colors) / 4)]
colors = [lower_color, upper_color]

ani_fig = make_combined_plot(
    np_Mscore=np.array(list(scores_animate['MEG'].values())),
    np_Escore=np.array(list(scores_animate['EEG'].values())),
    times=mne.read_epochs(meg_epoch_files['01']).times,
    cmap=COLORMAP_NAME,
    title='Animacy Decoding Accuracy (%)',
    vmin=45,
    vmax=65,
)

face_fig = make_combined_plot(
    np_Mscore=np.array(list(scores_face['MEG'].values())),
    np_Escore=np.array(list(scores_face['EEG'].values())),
    times=mne.read_epochs(meg_epoch_files['01']).times,
    cmap=COLORMAP_NAME,
    title='Face Decoding Accuracy (%)',
    vmin=35,
    vmax=70,
)

# %%
ani_fig.savefig(
    op.join(FIG_DIR, 'decoding_animacy.svg'),
    dpi=600,
    bbox_inches='tight',
)
face_fig.savefig(
    op.join(FIG_DIR, 'decoding_face.svg'),
    dpi=600,
    bbox_inches='tight',
)
# %%
