#%%
import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = '../../NOD-MEEG_upload'
EEG_ROOT = op.join(ROOT, 'NOD-EEG', 'events')
MEG_ROOT = op.join(ROOT, 'NOD-MEG', 'events')

SAVE_DIR = '../../NOD-MEEG_results'
FIG_DIR = f'{SAVE_DIR}/figs'
FONT_SIZE = 12
#%%
def get_files(root_dir):
    """Get a dictionary mapping subject IDs to file paths in the specified directory.

    Parameters:
    ----------
    root_dir : str
        The directory containing the CSV files.

    Returns:
    ----------
    dict
        A dictionary mapping subject IDs to file paths.
    """
    files = {}
    for filename in sorted(os.listdir(root_dir)):
        if filename.endswith('.csv'):
            subject_id = filename.split('_')[0][-2:]
            files[subject_id] = op.join(root_dir, filename)
    return files

def extract_sub_accuracy(file_path):
    """Extract accuracy per session for a given subject file.

    Parameters:
    ----------
    file_path : str
        Path to the subject's data file.
        
    Returns:
    ----------
    dict
        A dictionary with session names as keys and accuracies as values.
    """
    event_data = pd.read_csv(file_path)
    subject_accuracy = {}
    for session in event_data['session'].unique():
        session_data = event_data[event_data['session'] == session]
        responses = pd.to_numeric(session_data['resp_is_right'], errors='coerce').values
        accuracy = np.mean(responses)
        subject_accuracy[session] = accuracy
    return subject_accuracy

def extract_accuracy(files_dict):
    accuracies = {}
    for subject_id in files_dict:
        accuracies[subject_id] = extract_sub_accuracy(files_dict[subject_id])
    return accuracies

def make_main_plot(plot_dict, fontsize=12):
    """Create a scatter plot of accuracies per subject and session.

    Parameters:
    ----------
    plot_dict : dict
        A dictionary with subject IDs as keys and dictionaries of session accuracies as values.
    fontsize : int
        Font size for the plot.

    Returns:
    ----------
    matplotlib.figure.Figure
        The figure object.
    """
    # Close all existing plots
    plt.close('all')

    # Define markers for EEG and MEG
    marker_dict = {
        'EEG': '^',  # Triangle up
        'MEG': 's'   # Square
    }

    # Get session names from the first subject
    session_names = list(next(iter(plot_dict.values())).keys())

    # Define colors for sessions
    colors = sns.color_palette(palette='Spectral', n_colors=5)
    del colors[2]  # Remove the third color to have 4 colors
    color_dict = {session_name: colors[i] for i, session_name in enumerate(session_names)}

    offset = 0.4  # Offset for x-axis positions

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 3), dpi=600)

    subject_positions = {}  # Store x-axis positions for each subject
    subject_labels = []     # Store subject labels for x-axis ticks

    legend_labels = set()   # Keep track of labels added to the legend

    for subject, sessions in plot_dict.items():
        datatype = subject[:-2]   # 'EEG' or 'MEG'
        subject_num = int(subject[-2:])  # Subject number

        # Determine x-axis position based on subject number
        if subject_num <= 10:
            subject_pos = subject_num * 2 - 1
        else:
            subject_pos = 19 + (subject_num - 10) * 1.5

        if subject_num not in subject_positions:
            subject_positions[subject_num] = subject_pos
            subject_labels.append(subject_num)

        # Adjust position based on data type (EEG or MEG)
        subject_actual_pos = subject_positions[subject_num] + (offset if datatype == 'EEG' else -offset)

        marker_style = marker_dict[datatype]

        for session_name, accuracy in sessions.items():
            color = color_dict[session_name]

            # Prepare label for legend
            label = None
            legend_label = f'{datatype} - {session_name}'
            if legend_label not in legend_labels:
                label = legend_label
                legend_labels.add(legend_label)

            # Plot the data point
            ax.scatter(
                subject_actual_pos, accuracy,
                facecolors='none', edgecolors=color, marker=marker_style,
                s=50, linewidths=3, alpha=1.0, label=label
            )

    # Set x-axis ticks and labels
    unique_subject_positions = sorted(subject_positions.values())
    ax.set_xticks(unique_subject_positions)
    ax.set_xticklabels(subject_labels)

    # Set labels and title
    ax.set_xlabel('Subject', fontsize=fontsize)
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    sns.despine(fig=fig, ax=ax)

    # Adjust legend position to avoid overlapping data points
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.65, 1.1),
        ncol=4,
        fontsize=fontsize-2,
    )

    plt.tight_layout()
    plt.show()

    return fig

#%%
EEG_FILES = get_files(EEG_ROOT)
MEG_FILES = get_files(MEG_ROOT)
EEG_accu = extract_accuracy(EEG_FILES)
MEG_accu = extract_accuracy(MEG_FILES)

plot_dict = {f'MEG{sub}': MEG_accu[sub] for sub in MEG_accu}
plot_dict.update({f'EEG{sub}': EEG_accu[sub] for sub in EEG_accu})

fig = make_main_plot(plot_dict, fontsize=FONT_SIZE)
fig.savefig(f'{FIG_DIR}/accuracy.svg', dpi=600, bbox_inches='tight')
#%%
