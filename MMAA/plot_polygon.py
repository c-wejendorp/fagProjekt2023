
from itertools import groupby
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import *

"""
Creating a simplex graph and saving it. Supplemental dataframe can provide coloring based on the first column.
This was taken from Benyamin Motevalli code in the original archetypal analysis package code.

Parameters:
-----------
alfa:
    pandas dataframe, read in archetypal analysis results
supplement = []:
    pandas dataframe, read in supplemental information. 
plot_args

grid_on = true

dataTitle = "":
    str, title for plot
"""


def plot_simplex(alfa,  supplement=None, plot_args={}, grid_on=True, dataTitle="", save_path=None):
    """
    group_color = None, color = None, marker = None, size = None
    group_color:    

        Dimension:      n_data x 1

        Description:    Contains the category of data point.
    """
    alfa = alfa.T
    archetypeNum = alfa.shape[0]

    labels = ('A'+str(i + 1) for i in range(archetypeNum))
    rotate_labels = True
    label_offset = 0.10
    data = alfa.T
    scaling = False
    sides = archetypeNum

    basis = np.array(
        [
            [
                np.cos(2*_*pi/sides + 90*pi/180),
                np.sin(2*_*pi/sides + 90*pi/180)
            ]
            for _ in range(sides)
        ]
    )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T, basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data, basis)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for i, l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i, 0]
        y = basis[i, 1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = (angle + 180) % 360  # mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
            x*(1 + label_offset),
            y*(1 + label_offset),
            l,
            horizontalalignment='center',
            verticalalignment='center',
            rotation=angle
        )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)

    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False
            if not (ignore):
                lst_ax_0.append(basis[i, 0] + [0, ])
                lst_ax_1.append(basis[i, 1] + [0, ])
                lst_ax_0.append(basis[j, 0] + [0, ])
                lst_ax_1.append(basis[j, 1] + [0, ])

    ax.plot(lst_ax_0, lst_ax_1, color='#FFFFFF',
            linewidth=1, alpha=0.5, zorder=1)

    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    for _ in range(sides):
        lst_ax_0.append(basis[_, 0] + [0, ])
        lst_ax_1.append(basis[_, 1] + [0, ])

    lst_ax_0.append(basis[0, 0] + [0, ])
    lst_ax_1.append(basis[0, 1] + [0, ])

    ax.plot(lst_ax_0, lst_ax_1, linewidth=1, zorder=2)  # , **edge_args )

    if supplement is not None:
        sc = ax.scatter(newdata[:, 0], newdata[:, 1],  zorder=3, alpha=0.5,
                        c=supplement.iloc[:, 0].factorize()[0], cmap='Spectral', label=supplement.iloc[:, 0].unique
                        )

        plt.legend(sc.legend_elements(num=len(supplement.iloc[:, 0].unique()))[0], list(supplement.iloc[:, 0].unique()), loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, ncol=10)

    elif supplement is None:
        ax.scatter(newdata[:, 0], newdata[:, 1],
                   color='red', zorder=3, alpha=0.5)

    else:
        if ('marker' in plot_args):
            marker_vals = plot_args['marker'].values
            marker_unq = np.unique(marker_vals)

            for marker in marker_unq:
                row_idx = np.where(marker_vals == marker)
                tmp_arg = {}
                for keys in plot_args:
                    if (keys != 'marker'):
                        tmp_arg[keys] = plot_args[keys].values[row_idx]

                ax.scatter(newdata[row_idx, 0], newdata[row_idx, 1],
                           **tmp_arg, marker=marker, alpha=0.5, zorder=3)
        else:
            ax.scatter(newdata[:, 0], newdata[:, 1], **plot_args,
                       color='pink', marker='s', zorder=3, alpha=0.5)

    plt.title(f'{dataTitle}, K = {archetypeNum}', fontsize=15)

    plt.savefig(save_path + f"aa.simplex_k_{archetypeNum}_{dataTitle}.jpg",
                dpi=500, bbox_inches='tight')
    plt.show()


def load_S(seed, path, split, mean, k):
    #load data
    split_path = path + f"/split_{split}"
    matrix_path = split_path + f"/Sms"

    #load all seed matrices
    data = []

    for s in seed:
        data.append(np.load(matrix_path + f"/Sms_split-{split}_k-{k}_seed-{s}.npy"))

    #average s matrix across all seeds
    data = np.mean(np.asarray(data), axis = 0)
    if mean:
        #average across 
        data = np.mean(data, axis = 1)

    return data





if __name__ == "__main__":
    path = 'data/MMAA_results/eeg-meg-fmri'
    seed = range(0,91,10)
    S = load_S(seed=seed, path=path, split=0, mean=True, k=4)
    modalities = ['EEG', 'MEG', 'fMRI']
    conditions = ['Famous', 'Nonfamous', 'Scrambled']
    for i, condition in enumerate(conditions):
        for j, modality in enumerate(modalities):
            plot_simplex(S[j,:, i * S.shape[2] // 3: i+1 * S.shape[2] // 3 ],dataTitle=f'cond-{condition}_mod-{modality}', save_path = 'MMAA/polygons/')
