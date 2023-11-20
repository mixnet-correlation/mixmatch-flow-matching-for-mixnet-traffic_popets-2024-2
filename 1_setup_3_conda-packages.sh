#!/usr/bin/env bash

# Safety settings.
set -exuo pipefail
shopt -s failglob


# Create a new Conda environment named 'mixmatch' based on Python 3.10.
conda create -n mixmatch -c conda-forge -y python=3.10

# Install into the 'mixmatch' environment packets that we need for our experiments.
conda install -n mixmatch -c conda-forge -y tensorflow==2.10
conda install -n mixmatch -c conda-forge -y hyperopt scikit-learn pydot tqdm pandas seaborn jupyterlab nbconvert ipython
