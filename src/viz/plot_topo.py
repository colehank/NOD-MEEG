from __future__ import annotations

import matplotlib.pyplot as plt
import mne

from ..utils import get_soi_picks


def plot_sensors(
    inst: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
    sois: list[str] = ['O', 'P', 'C', 'F', 'T'],
) -> None:
    """Plot the sensors of the SOI.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.Epochs | mne.Evoked
        The data object.
    sois : list[str]
        The SOIs.
    """
    full_name = {
        'F': 'Frontal',
        'C': 'Central',
        'P': 'Parietal',
        'O': 'Occipital',
        'T': 'Temporal',
    }
    groups = {}
    for soi in sois:
        picks = get_soi_picks(inst, soi)
        groups[soi] = picks
    ch_groups = list(groups.values())
    colors = plt.cm.get_cmap('Set2', len(ch_groups))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for soi, color in zip(sois, colors(range(len(ch_groups)))):
        ax.scatter([], [], color=color, label=full_name[soi])
    ax.legend(loc='best', framealpha=0.9)

    mne.viz.plot_sensors(
        info=inst.info,
        ch_groups=ch_groups,
        linewidth=1,
        pointsize=40,
        cmap=colors,
        axes=ax,
        to_sphere=True,
        show=False,
    )

    return fig
