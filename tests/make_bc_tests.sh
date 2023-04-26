#!/bin/bash

# Exit on error, unset variables and pipeline errors
set -euo pipefail

./build --debug

rm tests/*/*.{bc,bc.txt,json}

for fmlpath in tests/*/*.fml
do
    FML_COMPILED=1 ./tests/make_bc_test.sh "$fmlpath"
done
