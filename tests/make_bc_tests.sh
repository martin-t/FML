#!/bin/bash

./build

rm tests/*/*.{bc,bc.txt,json}

for fmlpath in tests/*/*.fml
do
    echo "Generating BC test for $fmlpath..."
    
    name=$(dirname "$fmlpath")/$(basename "$fmlpath" .fml)

    ./fml parse "$name.fml" -o "$name.json"
    ./fml compile "$name.json" -o "$name.bc"
    ./fml disassemble "$name.bc" | tee "$name.bc.txt"
    ./fml execute "$name.bc" | xargs -I{} echo "// >" {} | tee --append "$name.bc.txt"

    # TODO check expected is the same
done
