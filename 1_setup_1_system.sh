#!/usr/bin/env bash

# Safety settings.
set -exuo pipefail
shopt -s failglob


# Update and install required Ubuntu packages.
DEBIAN_FRONTEND=noninteractive apt-get update --yes
DEBIAN_FRONTEND=noninteractive apt-get install --yes octave octave-statistics texlive-extra-utils tcsh tree tmux
DEBIAN_FRONTEND=noninteractive apt-get autoremove --yes --purge
DEBIAN_FRONTEND=noninteractive apt-get clean --yes


# Download "Fira Sans Condensed" font.
mkdir -p /usr/share/fonts/truetype/firasanscondensed
wget https://github.com/google/fonts/raw/main/ofl/firasanscondensed/FiraSansCondensed-Regular.ttf -O /usr/share/fonts/truetype/firasanscondensed/FiraSansCondensed-Regular.ttf
wget https://github.com/google/fonts/raw/main/ofl/firasanscondensed/FiraSansCondensed-Italic.ttf -O /usr/share/fonts/truetype/firasanscondensed/FiraSansCondensed-Italic.ttf
rm -rf ~/.cache/matplotlib/fontlist-v330.json
fc-cache -rfv
