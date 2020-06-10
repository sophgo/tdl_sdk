# Copyright 2018 Bitmain Inc.
# License
# Author Tim Ho <tim.ho@bitmain.com>

if("${LIBDEP_GTEST_DIR}" STREQUAL "")
    if(CMAKE_TOOLCHAIN_FILE)
        set(LIBDEP_GTEST_DIR ${PREBUILT_DIR}/gtest)
    else()
        set(LIBDEP_GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/host_${CMAKE_BOARD_TYPE}/gtest)
    endif()
endif()
set(GTEST_LIBRARIES "${LIBDEP_GTEST_DIR}/lib/libgtest.a")

include_directories(
    "${LIBDEP_GTEST_DIR}/include"
)

install(DIRECTORY ${LIBDEP_GTEST_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include/)
install(DIRECTORY ${LIBDEP_GTEST_DIR}/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)
