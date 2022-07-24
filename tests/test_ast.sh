#!/bin/bash

# Build as a separate step so it prints the output and it doesn't look like the tests are stuck
./build

i=0
n=$(find tests -name '*.fml' | wc -l)
ret=0

for test in tests/*/*.fml
do
  i=$((i + 1))
  filename="$(dirname "$test")/$(basename "$test" .fml)"
  outfile="$filename.out"
  difffile="$filename.diff"


  message=$(echo -n "Executing test [$i/$n]: \"$test\"... ")
  echo -n "$message"

  message_length=$(echo -n "$message" | wc -c)
  for _ in $(seq 1 $((74 - $message_length)))
  do
    echo -n " "
  done

  ./fml run     "$test" 1> "$outfile" 2> "$outfile"
  echo "$?"

  diff <(grep -e '// > ' < "$test"    | sed 's| *\/\/ > \?||') "$outfile" > "$difffile"
  if test "$?" -eq 0
  then
    echo -e "\e[32mpassed\e[0m"
  else
    ret=1
    echo -e "\e[31mfailed\e[0m [details \"$difffile\"]"
    cat "$outfile"
    cat "$difffile"
  fi
done

exit $ret
