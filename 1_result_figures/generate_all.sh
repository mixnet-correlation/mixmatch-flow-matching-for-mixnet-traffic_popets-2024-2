#!/bin/bash

# Safety settings.
set -euo pipefail
shopt -s failglob

# source ~/miniforge3/bin/activate

# Move to project root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Moving to ${SCRIPT_DIR}"
cd ${SCRIPT_DIR}

export PYTHONPATH=${SCRIPT_DIR}:${SCRIPT_DIR}/notebooks
echo $PYTHONPATH

# Reset pdf folder
rm -rf pdf
mkdir -p pdf

# Run ipynb notebooks and move result pdfs
for f in `find ~+ -not -path '*/.*' -name "*.ipynb"`; do
    echo "Processing ${f}..."

    # Get parts of path
    dname=$(dirname -- "$f")
    fname=$(basename -- "$f")
    name="${fname%.*}";

    pushd $dname

    # Run ipynb
    jupyter-nbconvert --to script ${name}.ipynb;
    mv ${name}.py ${name}-temp.py
    ipython ${name}-temp.py;

    popd

done

# Move results
rm ${SCRIPT_DIR}/notebooks/*-temp.py;
mv ${SCRIPT_DIR}/notebooks/*.pdf ${SCRIPT_DIR}/pdf/

# source deactivate
