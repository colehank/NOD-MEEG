# %%
from __future__ import annotations

import os
import os.path as op
import sys

import gif
import joblib
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from joblib import delayed
from joblib import Parallel
from matplotlib import font_manager as fm
from matplotlib import image as mpimg
from matplotlib import patches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from src.rsa import corr
from src.viz import plot_rdm
sys.path.append('..')


gif.options.matplotlib['dpi'] = 200
gif.options.matplotlib['bbox_inches'] = 'tight'
# %%
font_path = '../assets/Helvetica.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

FIG_DIR = '../../NOD-MEEG_results/figs'
RES_DATA_DIR = '../../NOD-MEEG_results/data'
FONT_SIZE = 12

SOIs = ['O', 'T', 'OT', 'all']
N_ITER = 5000
N_JOBS = 64
SOI = 'OT'
corr_METRIC = 'spearman'
cmap = 'seismic'
rdms_dir = f'{FIG_DIR}/rdms'
os.makedirs(rdms_dir, exist_ok=True)
rdms = joblib.load(op.join(RES_DATA_DIR, 'RDMs.pkl'))
# %% plot gif of RDMs


@gif.frame
def generate_frame(t, rdms, superclass_map, vmin, vmax, cmap, title):
    time = t / 250 * 1000 - 100  # 转换为毫秒
    plot_rdm.rdm_with_class(
        rdm=rdms[t],  # 选择当前时间步的RDM
        superclass_map=superclass_map,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
    )
    plt.suptitle(f'{title}: {time:.0f} ms', fontsize=FONT_SIZE)


def generate_frames(rdms, superclass_map, vmin, vmax, cmap, title):
    frames = []
    for t in tqdm(range(len(rdms)), desc='Generating frames'):
        frame = generate_frame(
            t, rdms, superclass_map,
            vmin, vmax, cmap, title,
        )  # 每个帧生成一次
        frames.append(frame)
    return frames


modalities = ['meg', 'eeg']
conditions = ['OT', 'all']
frames_dict = {}

for modality in modalities:
    for condition in conditions:
        key = f'{modality}_frames_{condition}'
        frames_dict[key] = generate_frames(
            rdms[f'{modality}_spatial'][condition],
            rdms['1000map30'],
            -1, 1,
            cmap,
            f'{modality.upper()}({condition})',
        )

meg_frames_OT = frames_dict['meg_frames_OT']
meg_frames_all = frames_dict['meg_frames_all']
eeg_frames_OT = frames_dict['eeg_frames_OT']
eeg_frames_all = frames_dict['eeg_frames_all']


for frames, name in zip(
    [meg_frames_OT, meg_frames_all, eeg_frames_OT, eeg_frames_all],
    ['meg_OT', 'meg_all', 'eeg_OT', 'eeg_all'],
):
    gif.save(frames, f'{rdms_dir}/{name}.gif', duration=100)
# %% plot vtc rdm
vtc_rdm = rdms['vtc']
fig = plot_rdm.rdm_with_class(
    rdm=vtc_rdm,
    superclass_map=rdms['1000map30'],
    vmin=-1,
    vmax=1,
    cmap=cmap,
    show_colorbar=True,
)
plt.savefig(f'{rdms_dir}/vtc.png', dpi=600, bbox_inches='tight')
# %% do rsa
rsa = corr.RSA(
    input_type='one2n',
    rdm1=vtc_rdm,
    rdm2=np.array(rdms['meg_spatial'][SOI]),
    n_jobs=N_JOBS,
    n_iter=N_ITER,
    alpha=.05,
)
corr_mv, ci_mv = rsa.rsa(
    corr_method=corr_METRIC,
    sig_method='bootstrap',
)

rsa = corr.RSA(
    input_type='one2n',
    rdm1=vtc_rdm,
    rdm2=np.array(rdms['eeg_spatial'][SOI]),
    n_jobs=N_JOBS,
    n_iter=N_ITER,
    alpha=.05,
)

corr_ev, ci_ev = rsa.rsa(
    corr_method=corr_METRIC,
    sig_method='bootstrap',
)

rsa = corr.RSA(
    input_type='n2n',
    rdm1=np.array(rdms['meg_spatial'][SOI]),
    rdm2=np.array(rdms['eeg_spatial'][SOI]),
    n_jobs=N_JOBS,
    n_iter=N_ITER,
    alpha=.05,
)

corr_me, ci_me = rsa.rsa(
    corr_method=corr_METRIC,
    sig_method='bootstrap',
)


corrs = {
    'meg_vtc': {'corr': corr_mv, 'ci': np.array(ci_mv)},
    'eeg_vtc': {'corr': corr_ev, 'ci': np.array(ci_ev)},
    'meg_eeg': {'corr': corr_me, 'ci': np.array(ci_me)},
}

joblib.dump(
    corrs, op.join(RES_DATA_DIR, 'corrs.pkl'),
)


# %% plot rdm4fusion
corrs = joblib.load(op.join(RES_DATA_DIR, 'corrs.pkl'))

# max_corr_meg = np.max(corrs['meg_vtc']['corr'])
# max_rdm_meg = np.argmax(corrs['meg_vtc']['corr'])
# rdm_meg = rdms['meg_spatial']['OT'][max_rdm_meg]

# max_corr_eeg = np.max(corrs['eeg_vtc']['corr'])
# max_rdm_eeg = np.argmax(corrs['eeg_vtc']['corr'])
# rdm_eeg = rdms['eeg_spatial']['OT'][max_rdm_eeg]

rdm_meg = rdms['meg_spatial']['OT'][150]
rdm_eeg = rdms['eeg_spatial']['OT'][150]
vmin_meg, vmax_meg = -1, 1
vmin_eeg, vmax_eeg = -1, 1


max_meg = plot_rdm.rdm_with_class(
    rdm=rdm_meg,
    superclass_map=rdms['1000map30'],
    vmin=vmin_meg,
    vmax=vmax_meg,
    cmap=cmap,
    show_colorbar=False,
)


max_eeg = plot_rdm.rdm_with_class(
    rdm=rdm_eeg,
    superclass_map=rdms['1000map30'],
    vmin=vmin_eeg,
    vmax=vmax_eeg,
    cmap=cmap,
    show_colorbar=False,
)

ini_meg = plot_rdm.rdm_with_class(
    rdm=rdms['meg_spatial']['OT'][0],
    superclass_map=rdms['1000map30'],
    vmin=vmin_meg,
    vmax=vmax_meg,
    cmap=cmap,
    show_colorbar=False,
)

ini_eeg = plot_rdm.rdm_with_class(
    rdm=rdms['eeg_spatial']['OT'][0],
    superclass_map=rdms['1000map30'],
    vmin=vmin_eeg,
    vmax=vmax_eeg,
    cmap=cmap,
    show_colorbar=False,
)

end_meg = plot_rdm.rdm_with_class(
    rdm=rdms['meg_spatial']['OT'][-1],
    superclass_map=rdms['1000map30'],
    vmin=vmin_meg,
    vmax=vmax_meg,
    cmap=cmap,
    show_colorbar=False,
)

end_eeg = plot_rdm.rdm_with_class(
    rdm=rdms['eeg_spatial']['OT'][-1],
    superclass_map=rdms['1000map30'],
    vmin=vmin_eeg,
    vmax=vmax_eeg,
    cmap=cmap,
    show_colorbar=False,
)

for fig, name in zip(
    [max_meg, max_eeg, ini_meg, ini_eeg, end_meg, end_eeg],
    ['max_meg', 'max_eeg', 'ini_meg', 'ini_eeg', 'end_meg', 'end_eeg'],
):
    fig.savefig(f'{rdms_dir}/{name}.svg', dpi=600, bbox_inches='tight')

# %% plot colorbar
fig, ax = plt.subplots(figsize=(0.15, 4), dpi=600)
norm = plt.Normalize(vmin=-1, vmax=1)
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap='seismic'), cax=ax, orientation='vertical',
    ticks=[-1, -0.5, 0, 0.5, 1],
)
cbar.set_label('Dissimilarity', fontsize=FONT_SIZE)
# plt.savefig(f'{rdms_dir}/colorbar_seismic.svg', dpi=600, bbox_inches='tight')

# %% main fig
fontsize = FONT_SIZE

time_points = np.linspace(-0.1, 0.8, 226) * 1000
colormap = 'Spectral'
cmap_ = plt.get_cmap(colormap, 10)
colors = [cmap_(i / (10 - 1)) for i in range(10)]
lower_color = colors[int(len(colors) / 4)]
upper_color = colors[int(3 * len(colors) / 4)]
colors = [lower_color, upper_color]

corrs = joblib.load(op.join(RES_DATA_DIR, 'corrs.pkl'))
conditions = {
    'MEG vs. VTC': {
        'corrs': corrs['meg_vtc']['corr'],
        'ci': corrs['meg_vtc']['ci'],
        'color': colors[0],
    },
    'EEG vs. VTC': {
        'corrs': corrs['eeg_vtc']['corr'],
        'ci': corrs['eeg_vtc']['ci'],
        'color': colors[1],
    },
    'MEG vs. EEG': {
        'corrs': corrs['meg_eeg']['corr'],
        'ci': corrs['meg_eeg']['ci'],
        'color': 'grey',
    },
}

ax1fig = f'{rdms_dir}/meeg_rdm4plot.png'
ax2fig = f'{rdms_dir}/vtc.png'
# plt.close('all')
# fig = plt.figure(figsize=(12, 4), dpi=600)
# gs = GridSpec(4, 12, figure=fig, wspace=2, hspace=2)
# ax1 = fig.add_subplot(gs[2:, :5])
# ax2 = fig.add_subplot(gs[:2, 1:4])
# ax3 = fig.add_subplot(gs[:, 5:])

plt.close('all')
fig = plt.figure(figsize=(12, 6), dpi=600)
gs = GridSpec(6, 12, figure=fig, wspace=4, hspace=2)
ax1 = fig.add_subplot(gs[3:, :5])
ax2 = fig.add_subplot(gs[:3, :5])
ax3 = fig.add_subplot(gs[1:5, 5:])

ax1.set_title('MEG/EEG RDMs', fontsize=fontsize)
ax1.imshow(mpimg.imread(ax1fig))
ax1.axis('off')
# ax1.text(0.5, -0.02, 'Time', fontsize=fontsize-4, ha='center', va='center', transform=ax1.transAxes)

ax2.set_title('VTC RDM', fontsize=fontsize)
img = mpimg.imread(ax2fig)
ax2.imshow(img)
ax2.axis('off')

ax3.set_xlim(-100, 800)
ax3.set_ylim(-0.01, 0.2)
ax3.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

for label, cond in conditions.items():
    alpha = 0.5 if label == 'MEG vs. EEG' else 1
    ax3.plot(
        time_points,
        cond['corrs'],
        color=cond['color'],
        label=label,
        lw=1,
        alpha=alpha,
        linestyle='-',
    )
    ax3.fill_between(
        time_points,
        cond['ci'][:, 0],
        cond['ci'][:, 1],
        color=cond['color'],
        alpha=.5,
    )

    max_corr = np.max(cond['corrs'])
    max_time = time_points[np.argmax(cond['corrs'])]
    ci = cond['ci'][np.argmax(cond['corrs'])]
    print(f'{label} corr_max: {max_corr:.3f} at {max_time:3f} ms, ci: {ci}')

ax3.set_xlabel('Time (ms)', fontsize=fontsize)
ax3.set_ylabel('Spearman r', fontsize=fontsize)
ax3.axhline(0, color='k', linestyle='--')
ax3.legend(loc='upper left', fontsize=fontsize - 1)

# 添加大括号
x1, y1 = ax1.transAxes.transform((.35, 1.54))
x2, y2 = ax1.transAxes.transform((.35, .2))
x3, y3 = ax3.transAxes.transform((-.49, .3))

con1 = patches.ConnectionPatch(
    xyA=(x1, y1), xyB=(x3, y3), coordsA='figure pixels', coordsB='figure pixels',
    arrowstyle='-', linewidth=1, color='black', connectionstyle='arc3,rad=0',
)
con2 = patches.ConnectionPatch(
    xyA=(x2, y2), xyB=(x3, y3), coordsA='figure pixels', coordsB='figure pixels',
    arrowstyle='-', linewidth=1, color='black', connectionstyle='arc3,rad=0',
)

fig.add_artist(con1)
fig.add_artist(con2)
# plt.show()
plt.savefig(f'{FIG_DIR}/fusion.svg', dpi=600)

# %% plot superclass colors


def plot_superclass_colors(superclass_map, ax=None):
    superclasses = list(superclass_map.keys())
    n_superclasses = len(superclasses)
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')

    new_colors = np.vstack((tab20b.colors, tab20c.colors[:10]))
    custom_tab30 = ListedColormap(new_colors)
    superclass_colors = [custom_tab30(i) for i in range(n_superclasses)]
    color_dict = {
        superclass: superclass_colors[i]
        for i, superclass in enumerate(superclasses)
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 1.25), dpi=600)

    n_cols = 10  # 每行10个超类
    n_rows = 3   # 3行

    for idx, (superclass, color) in enumerate(color_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        rect = plt.Rectangle((col, n_rows - 1 - row), 1, 1, color=color)
        ax.add_patch(rect)

        superclass = 'geology' if superclass == 'geological_formation' else superclass
        text = ax.text(
            col + 0.5, n_rows - 1 - row + 0.5, superclass, ha='center', va='center',
            fontsize=FONT_SIZE+4, color='black', weight='bold',
        )

        # 添加白色轮廓效果
        text.set_path_effects([
            path_effects.Stroke(
                linewidth=4, foreground='white',
            ), path_effects.Normal(),
        ])

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis('off')
    plt.tight_layout()

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


fig, ax = plt.subplots(figsize=(12, 1.25), dpi=600)
plot_superclass_colors(rdms['1000map30'], ax)
fig.savefig(f'{FIG_DIR}/RDM_superclass.svg', dpi=600, bbox_inches='tight')
# %%
