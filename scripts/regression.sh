#!/bin/bash
run() {
    result="PASSED"
    ./$2 || result="FAILED"
    echo "TEST $1 $2: $result"
}
# For normal CI check
CI_PROGRAMS=(init)

for ((i=0; i < ${#CI_PROGRAMS[@]}; i++))
do
    run $i ${CI_PROGRAMS[$i]}
done

# For daily build tests
if [[ "$1" != "daily" ]]; then
  exit 0
fi