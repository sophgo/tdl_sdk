#!/bin/bash

# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

function check_rootdir_is_set()
{
    TMPBASEDIR=$(basename $PWD)

    # This funcion checks if the BASEDIR is set correctly
    # check if $1 is empty
    if [ -z "$1" ]; then
        if [ "$TMPBASEDIR" = "scripts" ]; then
            RETURNVAR=$(dirname $PWD)
        else
            RETURNVAR=$PWD
        fi;
    else
        RETURNVAR=$1
    fi;

    # Check if its root folder
    if [ ! -d "$RETURNVAR/.git" ]; then
        RETURNVAR=""
    fi;
    echo $RETURNVAR
}

function help_bm1880()
{
    echo "./build_bm1880.sh  <all/soc/cmodel> <build_type> <use legacy bmtap2=1/0> <path to toolchain folder> <path to BSP SDK folder>"
}

main()
{
    BASEDIR=$(check_rootdir_is_set $BASEDIR)
    source $BASEDIR/scripts/utils.sh
    build_selector help_bm1880 bm1880 OFF $1 $2 $3 $4 $5
}

if [ "${1}" != "--source-only" ]; then
    main "${@}"
else
    source $BASEDIR/scripts/utils.sh
fi