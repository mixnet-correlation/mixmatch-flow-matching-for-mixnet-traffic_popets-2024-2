#!/usr/bin/env bash

# Safety settings.
set -exuo pipefail
shopt -s failglob


# Create a new Conda environment named 'mixmatch' based on Python 3.10.
conda create -n mixmatch -c conda-forge -y python=3.10

# Install into the 'mixmatch' environment packets that we need for our experiments.
conda install -n mixmatch -c conda-forge -y tensorflow==2.10
conda install -n mixmatch -c conda-forge -y hyperopt==0.2.7 scikit-learn==1.3.2 pydot==2.0.0 tqdm==4.66.1 pandas==2.1.4 seaborn==0.13.1 jupyterlab==4.0.10 nbconvert==7.14.0 ipython==8.20.0
