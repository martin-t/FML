#!/bin/bash

fmlpath="$1"

# Optionally skip calling cargo since it's slow - e.g. when called from make_bc_tests.sh
if [[ ! "$FML_COMPILED" ]]; then
    ./build
fi

echo "Generating BC test for $fmlpath..."
name=$(dirname "$fmlpath")/$(basename "$fmlpath" .fml)

./fml parse "$name.fml" -o "$name.json"
./fml compile "$name.json" -o "$name.bc"
./fml disassemble "$name.bc" | tee "$name.bc.txt"
# --delimiter="\n" makes sure quotes and backslashes are not removed
./fml execute "$name.bc" | xargs --delimiter="\n" -I{} echo "// >" {} | tee --append "$name.bc.txt"

# LATER(martin-t) Check expected is the same
