#!/bin/bash

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "Usage: \$0 <chip_arch>"
    echo "Example: \$0 BM1684X or \$0 BM1688"
    exit 1
fi

CHIP_ARCH="$1"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 获取当前Python版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: ${PYTHON_VERSION}"

# 检查Python版本是否>=3.8
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python version must be >= 3.8, detected ${PYTHON_VERSION}"
    exit 1
fi

# 设置目标导出路径
TARGET_DIR="$SCRIPT_DIR/dependency/$CHIP_ARCH/python_depends"

# 创建目标目录
mkdir -p "$TARGET_DIR"

echo "Packing Python dependencies to: $TARGET_DIR"

# 创建目录结构
mkdir -p ${TARGET_DIR}/include
mkdir -p ${TARGET_DIR}/include/python
mkdir -p ${TARGET_DIR}/include/python/aarch64-linux-gnu/python${PYTHON_VERSION}
mkdir -p ${TARGET_DIR}/include/pybind11
mkdir -p ${TARGET_DIR}/lib

# 检查pybind11是否安装
if ! python3 -c "import pybind11" &>/dev/null; then
    echo "Error: pybind11 module not found. Please install it with:"
    echo "pip install pybind11"
    exit 1
fi

# 获取Python头文件路径
PYTHON_INCLUDE=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
if [ -z "$PYTHON_INCLUDE" ]; then
    echo "Error: Failed to get Python include path"
    exit 1
fi

# 获取Python库路径 - 改进错误处理
PYTHON_LIB_PATH=$(python3 -c "
import sysconfig, os, sys
try:
    libname = sysconfig.get_config_var('LDLIBRARY')
    if not libname:
        # 尝试常见的库名称模式
        libname = f'libpython{sys.version_info.major}.{sys.version_info.minor}.so'
        
    libdirs = [
        sysconfig.get_config_var('LIBDIR'),
        sysconfig.get_config_var('LIBPL'),
        '/usr/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/lib'
    ]
    
    # 过滤掉None值
    libdirs = [d for d in libdirs if d]
    
    for dir in libdirs:
        fullpath = os.path.join(dir, libname)
        if os.path.exists(fullpath):
            print(fullpath)
            break
    else:
        # 尝试查找任何匹配的Python库
        import glob
        for dir in libdirs:
            pattern = os.path.join(dir, f'libpython{sys.version_info.major}.{sys.version_info.minor}*.so*')
            matches = glob.glob(pattern)
            if matches:
                print(matches[0])
                break
        else:
            print('')
except Exception as e:
    print(f'# Error: {str(e)}', file=sys.stderr)
    print('')
")

if [ -z "$PYTHON_LIB_PATH" ] || [[ "$PYTHON_LIB_PATH" == \#* ]]; then
    echo "Warning: Python library not found. Continuing without library files."
    SKIP_LIB=1
else
    SKIP_LIB=0
fi

# 获取pybind11头文件路径
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")
if [ -z "$PYBIND11_INCLUDE" ]; then
    echo "Error: Failed to get pybind11 include path"
    exit 1
fi
echo "PYBIND11_INCLUDE: ${PYBIND11_INCLUDE}"

# 获取系统架构特定的pyconfig.h路径
PYCONFIG_ARCH_PATH=$(python3 -c "
import sysconfig, os, sys
try:
    # 获取Python版本
    py_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    
    # 尝试获取架构特定的配置路径
    arch_path = sysconfig.get_config_var('INCLUDEPY')
    if arch_path:
        # 尝试多种可能的路径模式
        possible_paths = [
            os.path.join(arch_path, 'aarch64-linux-gnu', f'python{py_version}'),
            os.path.join('/usr/include/aarch64-linux-gnu', f'python{py_version}'),
            os.path.join('/usr/include', f'python{py_version}')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(path)
                break
        else:
            # 如果不存在，尝试查找系统中的路径
            import glob
            paths = glob.glob(f'/usr/include/*/python{py_version}')
            if paths:
                print(paths[0])
            else:
                print('')
    else:
        print('')
except Exception as e:
    print(f'# Error: {str(e)}', file=sys.stderr)
    print('')
")

# 复制Python头文件
if [ -d "$PYTHON_INCLUDE" ]; then
    echo "Copying Python headers: ${PYTHON_INCLUDE}"
    cp -r ${PYTHON_INCLUDE}/* ${TARGET_DIR}/include/python/
else
    echo "Error: Python include directory not found: ${PYTHON_INCLUDE}"
    exit 1
fi

# 创建自定义的pyconfig.h文件
echo "Creating custom pyconfig.h for cross-compilation"
cat > ${TARGET_DIR}/include/python/pyconfig.h << EOF
#if defined(__aarch64__) && defined(__AARCH64EL__)
/* Include the local pyconfig.h directly */
#include "python${PYTHON_VERSION}/pyconfig.h"
#else
/* For other architectures, include the appropriate system-specific file */
#error "Unsupported architecture for cross-compilation"
#endif
EOF

# 复制Python库文件 - 使用-L选项跟随软链接，复制实际文件
if [ "$SKIP_LIB" -eq 0 ]; then
    echo "Copying Python libraries: ${PYTHON_LIB_PATH}"
    cp -L ${PYTHON_LIB_PATH} ${TARGET_DIR}/lib/
else
    echo "Skipping Python library copy as library was not found"
fi

# 复制pybind11头文件
if [ -d "$PYBIND11_INCLUDE" ]; then
    echo "Copying pybind11 headers: ${PYBIND11_INCLUDE}"
    cp -r ${PYBIND11_INCLUDE}/* ${TARGET_DIR}/include/pybind11/
else
    echo "Error: pybind11 include directory not found: ${PYBIND11_INCLUDE}"
    exit 1
fi

# 如果找到了架构特定的pyconfig.h，复制它
if [ -n "${PYCONFIG_ARCH_PATH}" ] && [[ "$PYCONFIG_ARCH_PATH" != \#* ]]; then
    echo "Copying architecture-specific pyconfig.h from: ${PYCONFIG_ARCH_PATH}"
    mkdir -p ${TARGET_DIR}/include/python/python${PYTHON_VERSION}
    cp -r ${PYCONFIG_ARCH_PATH}/* ${TARGET_DIR}/include/python/python${PYTHON_VERSION}/
else
    # 如果没有找到，创建一个基本的pyconfig.h
    echo "Creating basic pyconfig.h for aarch64"
    mkdir -p ${TARGET_DIR}/include/python/python${PYTHON_VERSION}
    cat > ${TARGET_DIR}/include/python/python${PYTHON_VERSION}/pyconfig.h << 'EOF'
/* pyconfig.h for aarch64-linux-gnu */
#define HAVE_LONG_LONG 1
#define PY_LONG_LONG long long
#define PY_LLONG_MIN (-PY_LLONG_MAX - 1)
#define PY_LLONG_MAX 0x7fffffffffffffffL
#define SIZEOF_LONG_LONG 8
#define SIZEOF_VOID_P 8
#define SIZEOF_SIZE_T 8
#define SIZEOF_TIME_T 8
#define SIZEOF_LONG 8
#define SIZEOF_INT 4
#define SIZEOF_SHORT 2
#define SIZEOF_FLOAT 4
#define SIZEOF_DOUBLE 8
#define SIZEOF_FPOS_T 16
#define SIZEOF_PTHREAD_T 8
#define SIZEOF_UINTPTR_T 8
#define SIZEOF_PID_T 4
#define ALIGNOF_LONG 8
#define ALIGNOF_SIZE_T 8
#define ALIGNOF_TIME_T 8
#define ALIGNOF_LONG_LONG 8
#define ALIGNOF_DOUBLE 8
#define ALIGNOF_LONG_DOUBLE 16
#define HAVE_GCC_ASM_FOR_X87 0
#define HAVE_GCC_ASM_FOR_X64 0
#define HAVE_GCC_ASM_FOR_MC68881 0
#define HAVE_GCC_UINT128_T 1
#define PY_FORMAT_LONG_LONG "ll"
#define PY_FORMAT_SIZE_T "z"
#define PY_FORMAT_TIME_T "l"
#define _GNU_SOURCE 1
#define _LARGEFILE_SOURCE 1
#define _FILE_OFFSET_BITS 64
EOF
fi

echo "ARM64 Python environment exported successfully to ${TARGET_DIR}"
