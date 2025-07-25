include(CMakeForceCompiler)

# usage
# cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-arm-linux.cmake ../
# The Generic system name is used for embedded targets (targets without OS) in
# CMake
set( CMAKE_SYSTEM_NAME          Linux )
set( CMAKE_SYSTEM_PROCESSOR     riscv64 )

# Set a toolchain path. You only need to set this if the toolchain isn't in
# your system path. Don't forget a trailing path separator!
set(TOOLCHAIN_TOPDIR "${TOOLCHAIN_ROOT_DIR}")
set( TC_PATH "${TOOLCHAIN_ROOT_DIR}/bin/" )

# The toolchain prefix for all toolchain executables
set( CROSS_COMPILE riscv64-unknown-linux-gnu- )
set( ARCH riscv )

# specify the cross compiler. We force the compiler so that CMake doesn't
# attempt to build a simple test program as this will fail without us using
# the -nostartfiles option on the command line
set(CMAKE_C_COMPILER ${TC_PATH}${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER ${TC_PATH}${CROSS_COMPILE}g++)

# To build the tests, we need to set where the target environment containing
# the required library is. On Debian-like systems, this is
# /usr/aarch64-linux-gnu.
SET(CMAKE_FIND_ROOT_PATH $ENV{TOOLCHAIN_TOPDIR})
# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# We must set the OBJCOPY setting into cache so that it's available to the
# whole project. Otherwise, this does not get set into the CACHE and therefore
# the build doesn't know what the OBJCOPY filepath is
set( CMAKE_OBJCOPY      ${TC_PATH}${CROSS_COMPILE}objcopy
	    CACHE FILEPATH "The toolchain objcopy command " FORCE )

# Set the CMAKE C flags (which should also be used by the assembler!
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -std=gnu11" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=c906fdv" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64imafdcv0p7xthead" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mabi=lp64d" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdata-sections" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-pointer-to-int-cast" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsigned-char -Wl,-gc-sections -lstdc++ -lm -lpthread" )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=c906fdv" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdcv0p7xthead" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mabi=lp64d" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-gc-sections -lm -lpthread" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes" )

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )
set( CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )
