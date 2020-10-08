#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TMP_WORKING_DIR=$SCRIPT_DIR/../doxygen

pushd $TMP_WORKING_DIR
doxygen Doxyfile
pushd latex
make
cp refman.pdf ../cviai.pdf
popd
rm -r latex
popd