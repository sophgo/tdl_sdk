# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

# Set options
option(GIT_SUBMODULE "Check submodules during build" OFF)

option(CONFIG_FUNCTION_TRACE "Enable function trace" OFF)
option(CONFIG_FUNCTION_SYSTRACE "Enable function systrace" ON)
option(CONFIG_BUILD_SOC "Build soc" ON)
option(CONFIG_SYSTRACE "Use systrace" ON)

option(BUILD_TEST "Build test" ON)
option(BUILD_DOC "Build doc" OFF)

option(DEBUG_NETWORK "Enable debug logging in non Release mode." OFF)
option(DEBUG_NETWORK_IMG "Save image in non Release mode." OFF)
option(DEBUG_NETWORK_IMAGENET "Save image_net infos in non Release mode." OFF)
option(DEBUG_SAVE_IMG "Save image in non Release mode." OFF)

option(USE_NEON_F2S_NEAREST "Use float 2 short nearest methods in NEON." ON)

option(USE_VPP "Use VPP hw do resize, makeboard, split, CSC." ON)

if(CONFIG_FUNCTION_TRACE)
    set(ENABLE_FUNCTION_TRACE 1 CACHE BOOL "" FORCE)
else()
    set(ENABLE_FUNCTION_TRACE 0 CACHE BOOL "" FORCE)
endif()

if(CONFIG_FUNCTION_SYSTRACE)
    set(ENABLE_FUNCTION_SYSTRACE 1 CACHE BOOL "" FORCE)
else()
    set(ENABLE_FUNCTION_SYSTRACE 0 CACHE BOOL "" FORCE)
endif()

if(CONFIG_SYSTRACE)
    set(ENABLE_TRACE 1 CACHE BOOL "" FORCE)
else()
    set(ENABLE_TRACE 0 CACHE BOOL "" FORCE)
endif()

if(NOT DEBUG_NETWORK)
    set(DEBUG_NETWORK_IMG ${DEBUG_NETWORK} CACHE BOOL "" FORCE)
endif()

if(NOT DEBUG_NETWORK)
    set(DEBUG_NETWORK_IMAGENET ${DEBUG_NETWORK} CACHE BOOL "" FORCE)
endif()