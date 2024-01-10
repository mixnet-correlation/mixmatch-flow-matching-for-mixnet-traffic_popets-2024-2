#!/usr/bin/env bash

# Safety settings.
set -exuo pipefail
shopt -s failglob


# Create folders, should they not already exist.
mkdir -p ~/mixmatch/deeplearning/delay_matrices

# Download zip file containing the delay matrices for `baseline`, `no-cover`, `low-delay`, `high-delay`, and `live-nym`.
# Mind: The downloaded zip file takes up 4.8 GB of disk space.
wget 'https://files.de-1.osf.io/v1/resources/m9gbz/providers/osfstorage/65155d20333aef0363d7034f/?zip=' -O ~/mixmatch/deeplearning/delay_matrices/osf_delay_matrices.zip

# Unzip the archive.
cd ~/mixmatch/deeplearning/delay_matrices
unzip osf_delay_matrices.zip
sync

# Remove the archive.
rm -rf ~/mixmatch/deeplearning/delay_matrices/osf_delay_matrices.zip

# Rename folders according to experiment names.
mv ~/mixmatch/deeplearning/delay_matrices/exp01 ~/mixmatch/deeplearning/delay_matrices/baseline
mv ~/mixmatch/deeplearning/delay_matrices/exp02 ~/mixmatch/deeplearning/delay_matrices/no-cover
mv ~/mixmatch/deeplearning/delay_matrices/exp05 ~/mixmatch/deeplearning/delay_matrices/low-delay
mv ~/mixmatch/deeplearning/delay_matrices/exp06 ~/mixmatch/deeplearning/delay_matrices/high-delay
mv ~/mixmatch/deeplearning/delay_matrices/exp08 ~/mixmatch/deeplearning/delay_matrices/live-nym
