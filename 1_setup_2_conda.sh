#!/usr/bin/env bash

# Safety settings.
set -exuo pipefail
shopt -s failglob


# Create the ~/miniconda3 folder.
mkdir -p ~/miniconda3

# Download Miniconda install script in correct version from official Anaconda server.
wget 'https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh' -O ~/miniconda3/miniconda.sh

# Execute downloaded script to install Conda into folder ~/miniconda3.
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove Miniconda install script again.
rm -rf ~/miniconda3/miniconda.sh
