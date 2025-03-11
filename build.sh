# print usage
print_usage() {
    echo "Usage: ${BASH_SOURCE[0]} [options]"
    echo "Options:"
    echo "  cv181x         Build 181x"
    echo "  cv186x         Build 186x"
    echo "  bm168x         Build BM168X edge"
    echo "  clean          Clean build"
}

# Check parameter
if [ "$#" -gt 2 ]; then
    echo "Error: Too many arguments"
    print_usage
    exit 1
fi

if [[ "$1" == "bm168x" ]]; then

    CHIP_ARCH=BM1688
    ./build_tdl_sdk.sh

elif [ "$1" = "cv181x" ]; then

    source ../build/envsetup_soc.sh
    defconfig sg2002_wevb_riscv64_sd
    export TPU_REL=1
    clean_all
    build_all

elif [ "$1" = "cv186x" ]; then

    source build/envsetup_soc.sh
    defconfig device_wevb_emmc
    export TPU_REL=1
    clean_all
    build_all

elif [ "$1" = "clean" ]; then

    ./build_tdl_sdk.sh $1

else
    echo "Error: Invalid option"
    print_usage
    exit 1

fi

