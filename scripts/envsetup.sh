#!/bin/sh

# 获取脚本所在目录作为相对路径基准
SCRIPT_DIR=$(cd "scripts" && pwd)
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
SDK_ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "SDK_ROOT_DIR: ${SDK_ROOT_DIR}"
# 打印帮助信息
print_usage() {
    echo "TDL SDK 环境配置工具"
    echo "用法: source ${SCRIPT_DIR}/envsetup.sh <平台> [DEPLOY]"
    echo "平台选项:"
    echo "  CV181X          - 配置 CV181X 平台环境"
    echo "  CV186X          - 配置 CV186X 平台环境"
    echo "  CV184X          - 配置 CV184X 平台环境"    
    echo "  BM1688          - 配置 BM1688 平台环境"
    echo "  BM1684          - 配置 BM1684 平台环境"   
    echo "  BM1684X         - 配置 BM1684X 平台环境"
    echo "  CMODEL_CV181X   - 配置 CMODEL_CV181X 平台环境"
    echo "  CMODEL_CV184X   - 配置 CMODEL_CV184X 平台环境"
    echo ""
    echo "部署选项:"
    echo "  DEPLOY          - 配置部署环境（可选）"
    echo ""
    echo "示例: source ${SCRIPT_DIR}/envsetup.sh BM1688"
    echo "      source ${SCRIPT_DIR}/envsetup.sh BM1688 DEPLOY"
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
        echo "请使用: source ${0} <平台> [DEPLOY]"
        exit 1
        ;;
esac

# 初始化部署模式变量
DEPLOY_MODE=0

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
        CV184X)
            export CHIP_ARCH="CV184X"
            ;;            
        BM1688)
            export CHIP_ARCH="BM1688"
            ;;
        BM1684)
            export CHIP_ARCH="BM1684"
            ;;            
        BM1684X)
            export CHIP_ARCH="BM1684X"
            ;;
        CMODEL_CV181X)
            export CHIP_ARCH="CMODEL_CV181X"
            ;;
        CMODEL_CV184X)
            export CHIP_ARCH="CMODEL_CV184X"
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
    
    # 检查是否有第二个参数（DEPLOY）
    if [ $# -gt 1 ]; then
        deploy_arg=$(echo "$2" | tr '[:lower:]' '[:upper:]')
        if [ "$deploy_arg" = "DEPLOY" ]; then
            DEPLOY_MODE=1
            echo "启用部署模式..."
        else
            echo "错误: 未知的第二个参数 '$2'"
            print_usage
            return 1
        fi
    fi
elif [ -z "${CHIP_ARCH}" ]; then
    echo "错误: 未指定平台且 CHIP_ARCH 环境变量未设置"
    print_usage
    return 1
fi

echo "配置 ${CHIP_ARCH} 平台环境..."
if [ $DEPLOY_MODE -eq 1 ]; then
    echo "部署模式: 启用"
fi

# 检查安装目录是否存在
if [ $DEPLOY_MODE -eq 1 ]; then
    TDL_INSTALL_DIR="/data/tdl_sdk/${CHIP_ARCH}"
else
    TDL_INSTALL_DIR="${SDK_ROOT_DIR}/install/${CHIP_ARCH}"
fi
if [ ! -d "${TDL_INSTALL_DIR}" ]; then
    echo "警告: 安装目录不存在: ${TDL_INSTALL_DIR}"
    echo "请先构建 TDL SDK: ./build_tdl_sdk.sh ${CHIP_ARCH}"
    return 1
fi

# 设置基本库路径和Python模块路径
LIB_PATHS="${TDL_INSTALL_DIR}/lib"
PYTHON_PATHS="${TDL_INSTALL_DIR}/lib"

# 平台依赖库路径
if [ $DEPLOY_MODE -eq 1 ]; then
    DEPENDENCY_BASE="/opt/sophon"
else
    DEPENDENCY_BASE="${SDK_ROOT_DIR}/dependency"
fi

# 根据平台配置特定环境
configure_platform_env() {
    case "${CHIP_ARCH}" in
        BM1688|BM1684X|BM1684)
            # 设置通用的Sophon依赖
            if [ $DEPLOY_MODE -eq 1 ]; then
                TPU_SDK_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "libsophon_*" 2>/dev/null | head -n 1)
                if [ -z "${TPU_SDK_PATH}" ]; then 
                    TPU_SDK_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "libsophon-*" 2>/dev/null | head -n 1)
                fi
                if [ -z "${TPU_SDK_PATH}" ]; then 
                    echo "错误: 未找到libsophon"
                    return 1
                fi

                OPENCV_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "sophon-opencv_*" 2>/dev/null | head -n 1)
                if [ -z "${OPENCV_PATH}" ]; then 
                    OPENCV_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "sophon-opencv-*" 2>/dev/null | head -n 1)
                fi
                if [ -z "${OPENCV_PATH}" ]; then 
                    echo "错误: 未找到sophon-opencv"
                    return 1
                fi

                FFMPEG_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "sophon-ffmpeg_*" 2>/dev/null | head -n 1)
                if [ -z "${FFMPEG_PATH}" ]; then 
                    FFMPEG_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "sophon-ffmpeg-*" 2>/dev/null | head -n 1)
                fi
                if [ -z "${FFMPEG_PATH}" ]; then 
                    echo "错误: 未找到sophon-ffmpeg"
                    return 1
                fi
                
            else
                TPU_SDK_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/libsophon"
                OPENCV_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/sophon-opencv"
                FFMPEG_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/sophon-ffmpeg"
            fi
            
            # 添加通用库路径
            add_lib_paths "${TPU_SDK_PATH}/lib" "${OPENCV_PATH}/lib" "${FFMPEG_PATH}/lib"
            
            # BM1688特有的ISP库
            if [ "${CHIP_ARCH}" = "BM1688" ]; then
                if [ $DEPLOY_MODE -eq 1 ]; then
                    ISP_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "sophon-soc-libisp_*" 2>/dev/null | head -n 1)
                    if [ -z "${ISP_PATH}" ]; then 
                        ISP_PATH=$(find "${DEPENDENCY_BASE}" -maxdepth 1 -type d -name "sophon-soc-libisp-*" 2>/dev/null | head -n 1)
                    fi
                    if [ -z "${ISP_PATH}" ]; then 
                        echo "错误: 未找到Sophon-ISP库"
                        return 1
                    fi
                else
                    ISP_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/sophon-soc-libisp"
                fi
                add_lib_paths "${ISP_PATH}/lib"
            fi
            ;;
            
        CV181X|CV186X|CV184X)
            # CVITEK 系列平台特定配置
            SDK_VER="${SDK_VER:-v2.6.0}"
            TPU_SDK_PATH="${SDK_ROOT_DIR}/../tpu_${SDK_VER}/cvitek_tpu_sdk"
            IVE_SDK_PATH="${SDK_ROOT_DIR}/../tpu_${SDK_VER}/cvitek_ive_sdk"
            MPI_PATH="${SDK_ROOT_DIR}/../cvi_mpi"
            
            # 添加CVITEK平台需要的环境变量
            export SDK_VER

            
            # 加载SOC环境
            if [ "${CHIP_ARCH}" != "CV181X" ]; then
                load_soc_env
            else
                CV181X_INSTALL_PATH=$(ls -d ${SDK_ROOT_DIR}/../install/soc_*/tpu_*/cvitek_tpu_sdk | head -n 1)
                add_lib_paths "${CV181X_INSTALL_PATH}/lib"
            fi

            
            # 添加依赖库路径
            add_lib_paths "${TPU_SDK_PATH}/lib" "${IVE_SDK_PATH}/lib" "${MPI_PATH}/lib" "${MPI_PATH}/lib/3rd"
            
            # 添加样例库路径
            add_lib_paths "${TDL_INSTALL_DIR}/sample/3rd/opencv/lib" \
                          "${TDL_INSTALL_DIR}/sample/3rd/tpu/lib" \
                          "${TDL_INSTALL_DIR}/sample/3rd/middleware/v2/lib" \
                          "${TDL_INSTALL_DIR}/sample/3rd/rtsp/lib"
            ;;
        CMODEL_CV181X)
            TPU_SDK_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}"
            OPENCV_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/opencv"

            add_lib_paths "${TPU_SDK_PATH}/lib" "${OPENCV_PATH}/lib"

            ;;
        CMODEL_CV184X)
            TPU_SDK_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}"
            OPENCV_PATH="${DEPENDENCY_BASE}/${CHIP_ARCH}/opencv"

            add_lib_paths "${TPU_SDK_PATH}/lib" "${OPENCV_PATH}/lib"
            export TPUKERNEL_FIRMWARE_PATH=${TPU_SDK_PATH}/lib/libcmodel_firmware.so
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
if [ $DEPLOY_MODE -eq 1 ]; then
    echo "部署模式        = 启用"
fi
echo ""
echo "配置的环境变量:"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
echo "PYTHONPATH      = ${PYTHONPATH}"

[ -n "${SDK_VER}" ] && echo "SDK_VER         = ${SDK_VER}"

echo ""

# 验证配置
if [ "${CHIP_ARCH}" = "BM1688" ] || [ "${CHIP_ARCH}" = "BM1684X" || [ "${CHIP_ARCH}" = "BM1684"]; then
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