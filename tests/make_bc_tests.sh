#!/bin/bash

./build

rm tests/*/*.{bc,bc.txt,json}

for fmlpath in tests/*/*.fml
do
    FML_COMPILED=1 ./tests/make_bc_test.sh "$fmlpath"
done
