#!/bin/bash

# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

#!/bin/bash -ex
BASEDIR=$PWD

function check_root_dir()
{
    if [ ! -d "$BASEDIR/.git" ] && [ ! -d "$BASEDIR/networks" ]; then
        echo "Error! Consider to call this script in the root folder of networks."
        exit
    fi;
}

function help()
{
    echo "./run_check.sh  <board type>/--clang-only"
    echo "                board type: bm1880"
}

function run_check()
{
    set -e
    . ./scripts/clang-format.sh --source-only

    echo "[Stage 1] Clang Format test"
    check_diff #you should commit before run this script
    format $PWD/networks
    check_diff #return error if file changed. If you see this line, amend you commit to fix the issue.
    echo "Done."

    for check_exit in "$@"
    do
        if [ "$check_exit" = "--clang-only" ]; then
            exit
        fi
    done

    board_type=$1

    echo "[Stage 2] compiling test and Clang Tidy test"
    echo "Compiling networks with legacy bmtap2"
    if [ -d "$BASEDIR/build/cmodel_$board_type/Release" ]; then
        rm -r $BASEDIR/build/cmodel_$board_type/Release
    fi;
    ./build.sh $board_type cmodel r || return $?
    echo "Running clang-tidy"
    run_tidy $BASEDIR/networks $BASEDIR/build/cmodel_$board_type/Release

    echo "Compiling networks with bmtap2 v2"
    if [ -d "$BASEDIR/build/cmodel_$board_type/Release" ]; then
        rm -r $BASEDIR/build/cmodel_$board_type/Release
    fi;
    ./build_v2.sh $board_type cmodel r || return $?
    echo "Running clang-tidy"
    run_tidy $BASEDIR/networks $BASEDIR/build/cmodel_$board_type/Release
    check_diff #return error if file changed
}

main()
{
    check_root_dir
    source $BASEDIR/scripts/utils.sh
    run_check $1 $2
}

main "${@}"