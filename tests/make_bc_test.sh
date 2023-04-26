#!/bin/bash

# Exit on error, unset variables and pipeline errors
set -euo pipefail

# Optionally skip calling cargo since it's slow - e.g. when called from make_bc_tests.sh.
# If undefined - rebuild.
if [[ ! -v FML_COMPILED ]]; then
    ./build --debug
fi

for fmlpath in "$@"; do
    echo "Generating BC test for $fmlpath..."
    name=$(dirname "$fmlpath")/$(basename "$fmlpath" .fml)

    ./fml --debug parse "$name.fml" -o "$name.json"
    ./fml --debug compile "$name.json" -o "$name.bc"
    ./fml --debug disassemble "$name.bc" | tee "$name.bc.txt"
    # --delimiter="\n" makes sure quotes and backslashes are not removed
    ./fml --debug execute "$name.bc" | xargs --delimiter="\n" -I{} echo "// >" {} | tee --append "$name.bc.txt"
done

# LATER(martin-t) Check expected in .fml and .bc.txt is the same
# LATER(martin-t) Check on CI current expected is the same as main branch
