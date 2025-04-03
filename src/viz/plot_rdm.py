from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


def rdm_with_class(
    rdm: np.ndarray,
    superclass_map: dict[str, list[str]],
    vmin: float,
    vmax: float,
    show_colorbar: bool = True,
    cmap: str = 'coolwarm',
) -> matplotlib.figure.Figure:
    # class colors
    sup_clss = list(superclass_map.keys())
    n_superclasses = len(sup_clss)
    tab20b = matplotlib.colormaps.get_cmap('tab20b')
    tab20c = matplotlib.colormaps.get_cmap('tab20c')
    new_colors = np.vstack((tab20b.colors, tab20c.colors[:10]))
    custom_tab30 = ListedColormap(new_colors)
    superclass_colors = np.array(
        [
            custom_tab30(
                sup_clss.index(sc),
            ) for sc in sup_clss for _ in superclass_map[sc]
        ],
    )

    fontsize = 10
    rsm_cmap = cmap
    gx, gy, fx, fy = 12, 12, 3, 3
    gs = GridSpec(gx, gy, wspace=1, hspace=2)
    fig = plt.figure(figsize=(fx, fy), dpi=600)
    ax0 = fig.add_subplot(gs[10:11, 1:11])  # superclass x轴
    ax1 = fig.add_subplot(gs[0:10, 0:1])  # superclass y轴
    ax2 = fig.add_subplot(gs[0:10, 1:11])  # rsm轴
    if show_colorbar:
        ax3 = fig.add_subplot(gs[0:10, 11:12])  # colorbar轴

    # superclass y轴
    ax1.imshow(
        superclass_colors[:, np.newaxis],
        aspect='auto', cmap=ListedColormap(superclass_colors),
    )
    ax1.axis('off')

    # superclass x轴
    ax0.imshow(
        superclass_colors[np.newaxis, :],
        aspect='auto', cmap=ListedColormap(superclass_colors),
    )
    ax0.axis('off')

    # rsm轴
    cax = ax2.imshow(rdm, cmap=rsm_cmap, vmin=vmin, vmax=vmax)
    ax2.axis('off')

    # Colorbar轴
    if show_colorbar:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(rsm_cmap)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, cax=ax3, orientation='vertical')
        tick_values = np.linspace(vmin, vmax, num=5)  # 生成线性刻度
        cbar.set_ticks(tick_values)
        tick_labels = [f'{x:.1f}' for x in tick_values]
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=fontsize-3)

        ax3.text(
            x=5.5, y=0.5, s='Dissimilarity', ha='center', va='center',
            rotation=90, transform=ax3.transAxes, fontsize=fontsize,
        )

        ax3.set_position([0.93, 0.1, 0.03, 0.8])  # x, y, width, height

    ax0.set_position([0.1, 0.08, 0.8, 0.02])  # x, y, width, height
    ax1.set_position([0.08, 0.1, 0.02, 0.8])  # x, y, width, height
    ax2.set_position([0.10, 0.1, 0.8, 0.8])  # x, y, width, height
    # plt.tight_layout()

    return fig
