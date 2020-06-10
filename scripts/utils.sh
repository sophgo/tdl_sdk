# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

# Do not source or run this shell script directly

regex_release=[Rr]
regex_debug=[Dd]
regex_asan=[Aa]
RED="\E[1;31m"
YELLOW="\E[1;33m"
ORIGINAL="\E[0m"

SUPPORTED_BOARDTYPE=(bm1880)
SUPPORTED_PLATFORMNAME=("cmodel" "soc" "usb")
SUPPORTED_BUILDTYPE=("Release" "Debug" "Asan")

function print_error()
{
  echo ${FUNCNAME[1]}: $* >&2
  return 1
}

# Get platform name string with given board type
function get_platform_name()
{
    if [ "$1" = "${SUPPORTED_PLATFORMNAME[0]}" ]; then
        RETURNVAR=cmodel_$2
    elif [ "$1" = "${SUPPORTED_PLATFORMNAME[1]}" ]; then
        RETURNVAR=soc_$2_asic
    elif [ "$1" = "${SUPPORTED_PLATFORMNAME[2]}" ]; then
        RETURNVAR=usb_$2
    else
        echo -e "${RED}Invalid platform type: $1. Supported types are "\
                "${SUPPORTED_PLATFORMNAME[*]}.${ORIGINAL}"
        return 1
    fi
    echo $RETURNVAR
    return 0
}

function get_build_type()
{
    build_type="Notype"
    if [[ $1 =~ $regex_release ]]; then
        build_type="${SUPPORTED_BUILDTYPE[0]}"
    elif [[ $1 =~ $regex_debug ]]; then
        build_type="${SUPPORTED_BUILDTYPE[1]}"
    elif [[ $1 =~ $regex_asan ]]; then
        build_type="${SUPPORTED_BUILDTYPE[2]}"
    else
        echo -e "${RED}Invalid build type: $1. Supported types are "\
                "${SUPPORTED_BUILDTYPE[*]}.${ORIGINAL}"
        return 1
    fi
    echo $build_type
    return 0
}

function build()
{
    # Assign variables
    board_type=$1            # $1: board type <bm1880>
    submodule_status=$2      # $2: Updates submodule or not
    platform_name=$3         # $3: <cmodel/soc/usb>
    build_type=$4            # $4: build type <release/debug/asan>
    use_bmtap2_legacy=$5     # $5: use legacy bmtap2 <1/0>
    toolchain_root=$6        # $6: toolchain root dir
    toolchain_file=$7        # $7: toolchain file
    bsp_sdk_root=$8          # $8: BSP_SDK

    # Variable gaurd
    # Check if use_bmtap2_legacy param is set properly.
    if [ -z "$use_bmtap2_legacy" ]; then
        echo -e "${RED}Invalid bmtap2 param: $use_bmtap2_legacy.${ORIGINAL}"
        return 1
    elif [ $use_bmtap2_legacy -ne 1 ] && [ $use_bmtap2_legacy -ne 0 ]; then
        echo -e "${RED}Wrong bmtap2 param: $use_bmtap2_legacy.${ORIGINAL}"
        return 1
    fi
    # Get platform name
    folder_name=$(get_platform_name $platform_name $board_type) || print_error "$folder_name" ||\
                return $?
    # Get build type
    build_type_regex=$(get_build_type $build_type) || print_error "$build_type_regex" || return $?

    # Create folder
    cd $BASEDIR
    mkdir -p build/$folder_name/$build_type_regex
    cd build/$folder_name/$build_type_regex

    # Build networks
    cmake -DCMAKE_MAKE_PROGRAM=$BASEDIR/ninja -GNinja ../../.. \
        -DGIT_SUBMODULE=$submodule_status \
        -DPLATFORM=$platform_name \
        -DCMAKE_BOARD_TYPE=$board_type \
        -DCMAKE_BUILD_TYPE=$build_type_regex \
        -DBUILD_DOC=OFF \
        -DUSE_LEGACY_BMTAP2=$use_bmtap2_legacy \
        -DTOOLCHAIN_ROOT_DIR=$toolchain_root \
        -DCMAKE_TOOLCHAIN_FILE=$toolchain_file \
        -DBSPSDK_ROOT_DIR=$bsp_sdk_root || print_error "configurate $1 failed" $? || return $?
    $BASEDIR/ninja -j8 && $BASEDIR/ninja install || print_error "configurate $1 failed" $? ||\
        return $?
    cd $BASEDIR
}

function build_doc()
{
    cd $BASEDIR
    cd doc
    mkdir -p doc_doxygen
    cd doc_doxygen
    cmake ../ || print_error "Make doc failed." $? || return $?
    make -j8 || print_error "Build doc failed" $? || return $?
    cd $BASEDIR
}

function make_test()
{
    platform_name=$1
    build_type=$2

    cd $BASEDIR
    cd build
    cd $(get_platform_name $platform_name $board_type)
    cd $(get_build_type $build_type)
    python3 $BASEDIR/test/python/run_test.py -t $platform_name -f $BASEDIR
}

# Build selector
function check_is_folder()
{
    if [ -d "$2" ]; then
        echo "Folder exists."
    else
        echo -e "${RED}Toolchain root not set properly.${ORIGINAL}"
        $1
        return 1
    fi
    return 0
}

function build_selector()
{
    # Assign variables
    help_func=$1
    board_type=$2            # $2: board type <bm1880>
    submodule=$3             # $3: Updates submodule or not
    mode=$4                  # $4: <cmodel/soc/usb>
    build_type=$5            # $5: build type <release/debug/asan>
    use_bmtap2_legacy=$6     # $6: use legacy bmtap2 <1/0>
    toolchain_root=$7        # $7: toolchain root dir
    bsp_sdk_root=$8          # $8: BSP_SDK

    # Check board type
    if [ "$board_type" != "bm1880" ]; then
        echo -e "${YELLOW}Unsupported board type: $board_type. Supported boardtypes are "\
                "${SUPPORTED_BOARDTYPE[*]}.${ORIGINAL}"
        $help_func
        return 1
    fi

    ## Build selector
    if [ "$mode" = "soc" ]; then
        check_is_folder $help_func $toolchain_root || return $?
        build $board_type $submodule $mode $build_type $use_bmtap2_legacy $toolchain_root \
            "toolchain/toolchain-aarch64-linux.cmake" $bsp_sdk_root || return $?
    elif [ "$mode" = "cmodel" ]; then
        build $board_type $submodule $mode $build_type $use_bmtap2_legacy || return $?
    elif [ "$mode" = "usb" ]; then
        build $board_type $submodule $mode $build_type $use_bmtap2_legacy || return $?
    elif [ "$mode" = "doc" ]; then
        build_doc || return $?
    elif [ "$mode" = "all" ]; then
        check_is_folder $help_func $toolchain_root || return $?
        # Make sure submodule is updated correctly.
        build $board_type ON soc $build_type $use_bmtap2_legacy $toolchain_root \
            "toolchain/toolchain-aarch64-linux.cmake" $bsp_sdk_root || return $?
        build $board_type OFF cmodel $build_type $use_bmtap2_legacy || return $?
        build $board_type OFF usb $build_type $use_bmtap2_legacy || return $?
        build_doc || return $?
        make_test cmodel $build_type || return $?
    else
        $help_func
        return 1
    fi
    return 0
}