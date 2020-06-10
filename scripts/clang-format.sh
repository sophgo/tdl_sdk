#!/bin/bash -ex
RED="\E[1;31m"
YELLOW="\E[1;33m"
ORIGINAL="\E[0m"

function help()
{
    echo "./clang-format.sh  <board type>"
    echo "                   board type: bm1880"
}

function check_clang_rootdir_is_set()
{
    TMPBASEDIR=$(basename $PWD)

    # This funcion checks if the BASEDIR is set correctly
    if [ -z "$1" ]; then
        if [ "$TMPBASEDIR" = "scripts" ]; then
            RETURNVAR=$(dirname $PWD)
        else
            RETURNVAR=$PWD
        fi;
    else
        RETURNVAR=$1
    fi;

    if [ -z "$RETURNVAR/.git" ]; then
        RETURNVAR=""
    fi;
    echo $RETURNVAR
}

function run_tidy {
    if [ -z ${3+x} ]; then
        run-clang-tidy $1 -p $2 -fix -quiet -j$(nproc)
    else
        run-clang-tidy $1 -p $2 -header-filter="$3/.*" -fix -quiet -j$(nproc)
    fi
}

# format source_path header_path
function format {
    if [ -z ${2+x} ]; then
        find $1 -regex '.*\.\(cpp\|h\|hpp\|cc\|cxx\|inc\)' -exec clang-format -i {} \;
    else
        find $1 -regex '.*\.\(cpp\|h\|hpp\|cc\|cxx\|inc\)' -exec clang-format -i {} \;
        find $2 -regex '.*\.\(cpp\|h\|hpp\|cc\|cxx\|inc\)' -exec clang-format -i {} \;
    fi
}


function check_diff()
{
    git diff --exit-code
}

main()
{
    BASEDIR=$(check_clang_rootdir_is_set $BASEDIR)
    set -x
    check_diff #you should commit before run this script
    run_tidy $BASEDIR/networks $BASEDIR/build/cmodel_$1/Release
    format $BASEDIR/networks
    check_diff #return error if file changed
}

if [ "${1}" != "--source-only" ]; then
    if [ "$1" = "bm1880" ]; then
        main "${@}"
    else
        echo -e "${YELLOW}Currently only bm1880 is supported.${ORIGINAL}"
        help
        exit
    fi
fi

# how to ignore checking with clang-format
# // clang-format off
# your code
# // clang-format on

# how to ignore checking with clang-tidy
# your code // NOLINT