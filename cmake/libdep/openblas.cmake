# Copyright 2018 Bitmain Inc.
# License
# Author

if("${LIBDEP_OPENBLAS_DIR}" STREQUAL "")
    if(CMAKE_TOOLCHAIN_FILE)
        set(LIBDEP_OPENBLAS_DIR ${PREBUILT_DIR}/libopenblas)
    else()
        set(LIBDEP_OPENBLAS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/host_${CMAKE_BOARD_TYPE}/libopenblas)
    endif()
endif()

include_directories(
    "${LIBDEP_OPENBLAS_DIR}/include"
)

if(CMAKE_TOOLCHAIN_FILE)
    set(OPENBLAS_LIBRARIES "")
else()
    set(OPENBLAS_LIBRARIES "${LIBDEP_OPENBLAS_DIR}/lib/libopenblas.so")
    install(DIRECTORY ${LIBDEP_OPENBLAS_DIR}/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)
endif()
