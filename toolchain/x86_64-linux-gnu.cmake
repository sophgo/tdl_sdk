# x86_64-linux.cmake
# CMake toolchain file for native compilation on Linux x86_64 platform

# Set system name and processor
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Use system default compilers
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# Set compilation flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -std=gnu11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdata-sections")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-pointer-to-int-cast")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsigned-char -Wl,-gc-sections -lstdc++ -lm -lpthread")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fsigned-char")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-gc-sections -lm -lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

# Cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "")
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "")


message(STATUS "Using x86_64-linux toolchain configuration")