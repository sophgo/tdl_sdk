#!/bin/bash

# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>
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
    echo "./build_v2.sh  <board type> <all/soc/cmodel> <build_type> <path to toolchain folder> <path to BSP SDK folder>"
    echo "               <board type> doc"
    echo "               board type: bm1880"
}

main()
{
    check_root_dir
    source $BASEDIR/scripts/utils.sh
    build_selector help $1 OFF $2 $3 0 $4 $5
}

main "${@}"