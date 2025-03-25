#!/bin/sh

# 获取脚本所在目录作为相对路径基准
SCRIPT_DIR=$(cd "scripts" && pwd)
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
SDK_ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "SDK_ROOT_DIR: ${SDK_ROOT_DIR}"
# 打印帮助信息
print_usage() {
    echo "TDL SDK 环境配置工具"
    echo "用法: source ${SCRIPT_DIR}/envsetup.sh [平台]"
    echo "平台选项:"
    echo "  CV181X    - 配置 CV181X 平台环境"
    echo "  CV186X    - 配置 CV186X 平台环境"
    echo "  BM1688    - 配置 BM1688 平台环境"
    echo "  BM1684X   - 配置 BM1684X 平台环境"
    echo "  CMODEL_CVITEK   - 配置 CMODEL_CVITEK 平台环境"
    echo ""
    echo "示例: source ${SCRIPT_DIR}/envsetup.sh bm1688"
}

# 检查是否使用了 source 命令运行脚本
# 使用更兼容的方式检测
case $0 in
    -sh|-bash|-ksh|-zsh|*/sh|*/bash|*/ksh|*/zsh)
        # 可能是使用source运行的
        ;;
    *)
        # 直接执行脚本
        echo "错误: 此脚本必须使用 'source' 命令运行"
        echo "请使用: source ${0} [平台]"
        exit 1
        ;;
esac

# 处理平台参数
if [ $# -gt 0 ]; then
    # 转换为大写以简化处理
    platform=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    case "$platform" in
        CV181X)
            export CHIP_ARCH="CV181X"
            ;;
        CV186X)
            export CHIP_ARCH="CV186X"
            ;;
        BM1688)
            export CHIP_ARCH="BM1688"
            ;;
        BM1684X)
            export CHIP_ARCH="BM1684X"
            ;;
        CMODEL_CVITEK)
            export CHIP_ARCH="CMODEL_CVITEK"
            ;;
        -H|--HELP)
            print_usage
            return 0
            ;;
        *)
            echo "错误: 未知平台 '$1'"
            print_usage
            return 1
            ;;
    esac
elif [ -z "${CHIP_ARCH}" ]; then
    echo "错误: 未指定平台且 CHIP_ARCH 环境变量未设置"
    print_usage
    return 1
fi

echo "配置 ${CHIP_ARCH} 平台环境..."

# 检查安装目录是否存在
TDL_INSTALL_DIR="${SDK_ROOT_DIR}/install/${CHIP_ARCH}"
if [ ! -d "${TDL_INSTALL_DIR}" ]; then
    echo "警告: 安装目录不存在: ${TDL_INSTALL_DIR}"
    echo "请先构建 TDL SDK: ./build_tdl_sdk.sh ${CHIP_ARCH}"
    return 1
fi

# 设置基本库路径和Python模块路径
LIB_PATHS="${TDL_INSTALL_DIR}/lib"
PYTHON_PATHS="${TDL_INSTALL_DIR}/lib"

# 平台依赖库路径
DEPENDENCY_BASE="${SDK_ROOT_DIR}/dependency"

# 根据平台配置特定环境
configure_platform_env() {
    case "${CHIP_ARCH}" in
        BM1688|BM1684X)
            # 设置通用的Sophon依赖
            TPU_SDK_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/libsophon"
            OPENCV_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/sophon-opencv"
            FFMPEG_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/sophon-ffmpeg"
            
            # 添加通用库路径
            add_lib_paths "${TPU_SDK_PATH}/lib" "${OPENCV_PATH}/lib" "${FFMPEG_PATH}/lib"
            
            # BM1688特有的ISP库
            if [ "${CHIP_ARCH}" = "BM1688" ]; then
                ISP_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/sophon-soc-libisp"
                add_lib_paths "${ISP_PATH}/lib"
            fi
            ;;
            
        CV181X|CV186X)
            # CVITEK 系列平台特定配置
            SDK_VER="${SDK_VER:-v2.6.0}"
            TPU_SDK_PATH="${SDK_ROOT_DIR}/../tpu_${SDK_VER}/cvitek_tpu_sdk"
            IVE_SDK_PATH="${SDK_ROOT_DIR}/../tpu_${SDK_VER}/cvitek_ive_sdk"
            MPI_PATH="${SDK_ROOT_DIR}/../cvi_mpi"
            
            # 添加CVITEK平台需要的环境变量
            export SDK_VER

            
            # 加载SOC环境
            load_soc_env
            
            # 添加依赖库路径
            add_lib_paths "${TPU_SDK_PATH}/lib" "${IVE_SDK_PATH}/lib" "${MPI_PATH}/lib"
            
            # 添加样例库路径
            add_lib_paths "${TDL_INSTALL_DIR}/sample/3rd/opencv/lib" \
                          "${TDL_INSTALL_DIR}/sample/3rd/tpu/lib" \
                          "${TDL_INSTALL_DIR}/sample/3rd/middleware/v2/lib" \
                          "${TDL_INSTALL_DIR}/sample/3rd/rtsp/lib"
            ;;
        CMODEL_CVITEK)
            TPU_SDK_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}"
            OPENCV_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/opencv"

            add_lib_paths "${TPU_SDK_PATH}/lib" "${OPENCV_PATH}/lib"

            ;;
    esac
}

# 添加库路径的辅助函数
add_lib_paths() {
    for path in "$@"; do
        if [ -d "${path}" ]; then
            if [ -z "${LIB_PATHS}" ]; then
                LIB_PATHS="${path}"
            else
                LIB_PATHS="${LIB_PATHS}:${path}"
            fi
        else
            echo "警告: 目录不存在: ${path}"
        fi
    done
}

# 加载SOC环境的辅助函数
load_soc_env() {
    SOC_ENV_SCRIPT="${SDK_ROOT_DIR}/../build/envsetup_soc.sh"
    if [ -f "${SOC_ENV_SCRIPT}" ]; then
        echo "检测到SOC环境脚本，将自动加载基础环境变量"
        CURRENT_DIR=$(pwd)
        cd "${SDK_ROOT_DIR}/.." 
        source build/envsetup_soc.sh > /dev/null 2>&1
        cd "${CURRENT_DIR}"
    fi
}

# 调用平台配置函数
configure_platform_env

# 构建路径字符串，并去除重复路径
LD_LIBRARY_PATH_NEW="${LIB_PATHS}"
PYTHONPATH_NEW="${PYTHON_PATHS}"

# 添加到现有路径（避免重复）
if [ -n "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NEW}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NEW}"
fi

if [ -n "${PYTHONPATH}" ]; then
    export PYTHONPATH="${PYTHONPATH_NEW}:${PYTHONPATH}"
else
    export PYTHONPATH="${PYTHONPATH_NEW}"
fi

# 打印配置信息
echo "环境配置完成!"
echo "CHIP_ARCH       = ${CHIP_ARCH}"
echo "SDK根目录       = ${SDK_ROOT_DIR}"
echo "安装目录        = ${TDL_INSTALL_DIR}"
echo ""
echo "配置的环境变量:"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
echo "PYTHONPATH      = ${PYTHONPATH}"

[ -n "${SDK_VER}" ] && echo "SDK_VER         = ${SDK_VER}"

echo ""

# 验证配置
if [ "${CHIP_ARCH}" = "BM1688" ] || [ "${CHIP_ARCH}" = "BM1684X" ]; then
    echo "验证 libbmrt.so 是否可访问..."
    if ldconfig -p 2>/dev/null | grep -q libbmrt.so; then
        echo "✓ libbmrt.so 可用"
    else
        echo "! 警告: libbmrt.so 不在系统库路径中"
        if find $(echo ${LD_LIBRARY_PATH} | tr ':' ' ') -name "libbmrt.so" 2>/dev/null | grep -q .; then
            echo "  但在 LD_LIBRARY_PATH 中找到它"
        else 
            echo "  且在 LD_LIBRARY_PATH 中也未找到"
            echo "  这可能导致运行时错误"
        fi
    fi
fi

echo ""
echo "测试 Python 模块导入:"
if python3 -c 'import sys; print("Python路径:", sys.path[:3])' 2>/dev/null; then
    echo "尝试导入 tdl 模块..."
    if python3 -c 'try: 
        import tdl
        print("✓ tdl 模块导入成功!")
    except ImportError as e: 
        print("! 错误:", e)' 2>/dev/null; then
        echo ""
        echo "可以运行示例:"
        echo "  cd ${TDL_INSTALL_DIR}/sample/python"
        echo "  python3 test_fd.py"
    fi
else
    echo "! 警告: Python3 不可用或发生错误"
fi