import os

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

sns.set_theme(style='ticks', font="Fira Sans Condensed", font_scale=2) 


# Define paths with ROCs
DIR_PATH = Path().resolve().parent
ROC_PATH = DIR_PATH / 'ROCs'

SIZE_W, SIZE_H = 9, 5

plt.style.use(DIR_PATH / 'plot_style.txt')

blue   = sns.color_palette("bright").as_hex()[0]
yellow = sns.color_palette("bright").as_hex()[1]
red    = sns.color_palette("bright").as_hex()[3]
purple = sns.color_palette("bright").as_hex()[4]

empty_dashes = [()]
dashes1 = [(2.0, 2.0), ()]
dashes2 = [(5.0, 5.0), (2.0, 2.0), ()]
dashes22 = [(2.0, 2.0), (5.0, 5.0)]
dashes_all = [(1.0, 1.0), (2.0, 1.0), (6.0, 2.0), (6.0, 1.0, 1.0, 1.0), (6.0, 1.0, 1.0, 1.0, 1.0, 1.0), ()]

random_line = Line2D([0],[0], color="green")

def plot_random_classifier(g):
    # line for random classifier
    diag = np.arange(0, 1, 0.001)
    sns.lineplot(x=diag, y=diag, color='green', linestyle='-')

    plt.setp(g.lines, alpha=.5)
    g.set(xlabel='FPR\n',
          ylabel='TPR',
          xlim=(0.000001, 0.25),
          ylim=(0.,1.01));
    
    
def set_log_scale(g):
    # force minor log ticks
    g.set_xscale('log')
    locmin = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    g.xaxis.set_minor_locator(locmin)
    g.xaxis.set_minor_formatter(NullFormatter())

    
def save_fig(name):
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f'{name}.pdf', dpi=300, bbox_inches='tight')
    os.system(f"pdfcrop --margin 0 {name}.pdf")
    os.system(f"mv {name}-crop.pdf {name}.pdf")