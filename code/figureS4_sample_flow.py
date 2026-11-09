# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S4: participant sample flow chart.

Output: results/figures/figureS4_sample_flow.png (300 dpi)

Enrollment numbers below are the cohort counts reported in the manuscript
Methods 2.1 (measured on the analysis server):
    2,725 unique participants across the 10 included HBN releases
    2,480 resting-state EEG marked available            (-245 no resting data)
    2,446 complete CBCL four-factor data                (-34 incomplete CBCL)
    2,439 age 5-21 years, entered quality control       (-7 out of range)
    1,644 passed quality control                        (-795: 773 ICA / 15 EC /
                                                         5 bad channels / 2 all)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from config import FIGURE_DIR

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif',
                   'DejaVu Serif', 'serif'],
    'axes.unicode_minus': False,
})

OUT_DIR = str(FIGURE_DIR)
NAME = 'figureS4_sample_flow'

MAIN_FILL = '#E8F1FB'
FINAL_FILL = '#E3F2FD'
SIDE_FILL = '#FFEBEE'
SIDE_EDGE = '#C62828'
MAIN_EDGE = '#37474F'

# (main box text, side box text or None)
STEPS = [
    ('Participants across the 10\nincluded HBN releases\n(n = 2,725)',
     None),
    ('Resting-state EEG marked\navailable\n(n = 2,480)',
     'Excluded (n = 245):\nresting-state data not available'),
    ('Complete CBCL four-factor\ndata (n = 2,446)',
     'Excluded (n = 34): incomplete CBCL data\n'
     'age 17.9 \u00b1 3.6 vs. 10.2 \u00b1 3.3 years,\n'
     'Welch p < 0.001; sex \u03c7\u00b2 p = 0.770'),
    ('Age 5\u201321 years; entered\nquality control (n = 2,439)',
     'Excluded (n = 7):\nage outside 5\u201321 years'),
    ('Passed quality control:\nfinal analysis sample\n(n = 1,644)',
     'Excluded (n = 795):\n'
     '773 ICA variance loss > 40%\n'
     '15 insufficient eyes-closed data\n'
     '5 with > 30% bad channels\n'
     '2 with all channels removed'),
]


def main():
    fig = plt.figure(figsize=(7.0, 8.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    n = len(STEPS)
    top, bottom = 0.97, 0.03
    slot = (top - bottom) / n
    bh = slot * 0.62                     # box height
    main_x, main_w = 0.04, 0.46
    side_x, side_w = 0.56, 0.42

    centers = []
    for i, (main_txt, side_txt) in enumerate(STEPS):
        cy = top - slot * (i + 0.5)
        centers.append(cy)
        final = (i == n - 1)
        ax.add_patch(Rectangle((main_x, cy - bh / 2), main_w, bh,
                               facecolor=FINAL_FILL if final else MAIN_FILL,
                               edgecolor=MAIN_EDGE, linewidth=1.2, zorder=2))
        ax.text(main_x + main_w / 2, cy, main_txt, ha='center', va='center',
                fontsize=10.5, fontweight='bold' if final else 'normal',
                linespacing=1.35, zorder=3)
        if side_txt:
            ax.add_patch(Rectangle((side_x, cy - bh / 2), side_w, bh,
                                   facecolor=SIDE_FILL, edgecolor=SIDE_EDGE,
                                   linewidth=0.9, linestyle='--', zorder=2))
            ax.text(side_x + side_w / 2, cy, side_txt, ha='center',
                    va='center', fontsize=8.6, color=SIDE_EDGE,
                    linespacing=1.3, zorder=3)
            ax.add_patch(FancyArrowPatch(
                (main_x + main_w, cy), (side_x, cy),
                arrowstyle='-|>', mutation_scale=11, linewidth=1.0,
                color=SIDE_EDGE, shrinkA=0, shrinkB=0, zorder=4))

    # vertical flow arrows between consecutive main boxes
    for i in range(n - 1):
        y0 = centers[i] - bh / 2
        y1 = centers[i + 1] + bh / 2
        ax.add_patch(FancyArrowPatch(
            (main_x + main_w / 2, y0), (main_x + main_w / 2, y1),
            arrowstyle='-|>', mutation_scale=14, linewidth=1.3,
            color=MAIN_EDGE, shrinkA=0, shrinkB=0, zorder=4))

    path = os.path.join(OUT_DIR, '%s.png' % NAME)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('saved:', path)
    plt.close(fig)


if __name__ == '__main__':
    main()
