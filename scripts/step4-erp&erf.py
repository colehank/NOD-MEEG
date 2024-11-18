"""
Script to load MEG and EEG epoch data, compute evoked responses, and plot them using matplotlib.
"""
#%%
import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager as fm

font_path = '../assets/Helvetica.ttc'  
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

FONT_SIZE = 12
EXAMPLE_SUBJECTS = [f'{i:02}' for i in range(1, 3)]
COLORMAP_NAME = 'Spectral'

DATA_DIR = '../../NOD-MEEG_upload'
FIG_DIR = '../../NOD-MEEG_results/figs'
MEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-MEG', 'derivatives', 'preprocessed', 'epochs')
EEG_EPOCH_ROOT = op.join(DATA_DIR, 'NOD-EEG', 'derivatives', 'preprocessed', 'epochs')

meg_epoch_files = {
    filename.split('-')[1][:2]: op.join(MEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(MEG_EPOCH_ROOT))
}
eeg_epoch_files = {
    filename.split('-')[1][:2]: op.join(EEG_EPOCH_ROOT, filename)
    for filename in sorted(os.listdir(EEG_EPOCH_ROOT))
}

# Load epoch data and compute evoked responses for example subjects
example_epochs = {}
example_evoked = {}
for subject in EXAMPLE_SUBJECTS:
    meg_epochs = mne.read_epochs(meg_epoch_files[subject])
    eeg_epochs = mne.read_epochs(eeg_epoch_files[subject])
    example_epochs[subject] = [meg_epochs, eeg_epochs]
    example_evoked[subject] = [epochs.average() for epochs in example_epochs[subject]]

data_list = []
for subject in example_evoked:
    for evoked in example_evoked[subject]:
        data_list.append(evoked.data)

#%%
def plot_evoked(
    data_list: list,
    times: np.ndarray,
    colors: list,
    alpha: float = 0.2,
    fontsize: int = 12,
    linewidth: float = 0.4,
) -> plt.Figure:
    """
    Plot evoked responses.

    Parameters:
    - data_list: List of numpy arrays containing the data to plot.
    - times: Array of time points corresponding to the data.
    - colors: List of colors for the plot lines.
    - alpha: Transparency level of the plot lines.
    - fontsize: Font size for text in the plots.
    - linewidth: Width of the plot lines.

    Returns:
    - fig: The matplotlib Figure object containing the plots.
    """
    num_plots = len(data_list) + 1  # Additional plot for the '...' marker
    fig = plt.figure(figsize=(6, 3), dpi=600)
    gs = GridSpec(num_plots, 1, height_ratios=[1] * (num_plots - 1) + [0.5])
    axes = [fig.add_subplot(gs[i]) for i in range(num_plots)]

    for i, ax in enumerate(axes):
        if i < len(data_list):
            n_channels = data_list[i].shape[0]
            for channel_idx in range(n_channels):
                ax.plot(
                    times,
                    data_list[i][channel_idx],
                    color=colors[channel_idx],
                    alpha=alpha,
                    linewidth=linewidth,
                )
            # Remove spines and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticks(np.arange(-100, 900, 100))
            ax.set_xlim(-100, 800)

            # Add labels for MEG/EEG and subject number
            label_positions = {0: 0.9, 1: 0.7, 2: 0.6, 3: 0.55}
            label = 'EEG' if n_channels < 100 else 'MEG'
            subject_number = i//2+1
            y_adjust = 0.15 if i in [0,1] else 0
            
            ax.text(
                1,
                label_positions.get(i, 0),
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                va='center',
                ha='right',
            )
            
            
            ax.text(
                0,
                label_positions.get(i, 0) - y_adjust,
                f'S{subject_number}',
                transform=ax.transAxes,
                fontsize=fontsize,
                va='center',
                ha='left',
            )
        else:
            # Empty plot for the '...' marker
            ax.plot(times, np.zeros_like(times), alpha=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel('Time (ms)')
            ax.get_yaxis().set_visible(False)
            ax.set_xticks(np.arange(-100, 900, 100))
            ax.set_xlim(-100, 800)

    # Add vertical line for onset
    baseline_x = 0.11
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False
    )
    plt.axvline(baseline_x, 0.1, color='black', linestyle='--', linewidth=1)

    # Add '...' text to indicate continuation
    fig.text(
        baseline_x + 0.0475, 0.285, '...', va='center', fontsize=fontsize + 10
    )

    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)
    plt.show()
    return fig

# Prepare data for plotting
times = example_evoked['01'][0].times * 1000  # Convert times to milliseconds
num_colors = data_list[0].shape[0]
colormap = cm.get_cmap(COLORMAP_NAME, num_colors)
colors = [colormap(i / (num_colors - 1)) for i in range(num_colors)]

# Plot the evoked responses
alpha = 0.4
linewidth = 0.8
fig = plot_evoked(
    data_list, times, colors, alpha=alpha, fontsize=FONT_SIZE, linewidth=linewidth
)

# Save the figure
fig.savefig(op.join(FIG_DIR, 'erf&erp.svg'), dpi=600, bbox_inches='tight')
# 
#%%
