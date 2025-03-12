#!/bin/bash

# 设置目标导出路径
TARGET_DIR="./python_depends"

# 创建目录结构
mkdir -p ${TARGET_DIR}/include
mkdir -p ${TARGET_DIR}/include/python
mkdir -p ${TARGET_DIR}/include/python/aarch64-linux-gnu/python3.8
mkdir -p ${TARGET_DIR}/include/pybind11
mkdir -p ${TARGET_DIR}/lib

# 获取Python头文件路径
PYTHON_INCLUDE=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

# 获取Python库路径
PYTHON_LIB_PATH=$(python3 -c "
import sysconfig, os
libname = sysconfig.get_config_var('LDLIBRARY')
libdirs = [sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LIBPL')]
for dir in libdirs:
    fullpath = os.path.join(dir, libname)
    if os.path.exists(fullpath):
        print(fullpath)
        break
else:
    import sys
    sys.exit('Python library not found!')
")

# 获取pybind11头文件路径
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")

# 获取系统架构特定的pyconfig.h路径
PYCONFIG_ARCH_PATH=$(python3 -c "
import sysconfig, os
try:
    # 尝试获取架构特定的配置路径
    arch_path = sysconfig.get_config_var('INCLUDEPY')
    if arch_path:
        arch_specific = os.path.join(arch_path, 'aarch64-linux-gnu', 'python3.8')
        if os.path.exists(arch_specific):
            print(arch_specific)
        else:
            # 如果不存在，尝试查找系统中的路径
            import glob
            paths = glob.glob('/usr/include/*/python3.8')
            if paths:
                print(paths[0])
            else:
                print('')
    else:
        print('')
except:
    print('')
")

# 复制Python头文件
echo "Copying Python headers: ${PYTHON_INCLUDE}"
cp -r ${PYTHON_INCLUDE}/* ${TARGET_DIR}/include/python/

# 创建自定义的pyconfig.h文件
echo "Creating custom pyconfig.h for cross-compilation"
cat > ${TARGET_DIR}/include/python/pyconfig.h << 'EOF'
#if defined(__aarch64__) && defined(__AARCH64EL__)
/* Include the local pyconfig.h directly */
#include "python3.8/pyconfig.h"
#else
/* For other architectures, include the appropriate system-specific file */
#error "Unsupported architecture for cross-compilation"
#endif
EOF

# 复制Python库文件 - 使用-L选项跟随软链接，复制实际文件
echo "Copying Python libraries: ${PYTHON_LIB_PATH}"
cp -L ${PYTHON_LIB_PATH} ${TARGET_DIR}/lib/

# 复制pybind11头文件
echo "Copying pybind11 headers: ${PYBIND11_INCLUDE}"
cp -r ${PYBIND11_INCLUDE}/* ${TARGET_DIR}/include/pybind11/

# 如果找到了架构特定的pyconfig.h，复制它
if [ -n "${PYCONFIG_ARCH_PATH}" ]; then
    echo "Copying architecture-specific pyconfig.h from: ${PYCONFIG_ARCH_PATH}"
    mkdir -p ${TARGET_DIR}/include/python/python3.8
    cp -r ${PYCONFIG_ARCH_PATH}/* ${TARGET_DIR}/include/python/python3.8/
else
    # 如果没有找到，创建一个基本的pyconfig.h
    echo "Creating basic pyconfig.h for aarch64"
    mkdir -p ${TARGET_DIR}/include/python/python3.8
    cat > ${TARGET_DIR}/include/python/python3.8/pyconfig.h << 'EOF'
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