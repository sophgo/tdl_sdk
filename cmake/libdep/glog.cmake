# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang yangwen.huang@bitmain.com

if(CMAKE_TOOLCHAIN_FILE)
    if("${LIBDEP_GLOG_DIR}" STREQUAL "")
        set(TAR_NAME "${PREBUILT_DIR}/libglog_${BM_NETWORKS_TARGET_BASENAME}.tar.gz")
        extract_package("${TAR_NAME}" "${PREBUILT_DIR}" "libglog" "")
        set(LIBDEP_GLOG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libglog)
    endif()
    include_directories(
        "${LIBDEP_GLOG_DIR}/include"
        "${LIBDEP_GLOG_DIR}/include/glog"
    )

    set(GLOG_LIBRARIES
        "${LIBDEP_GLOG_DIR}/lib/libglog.so"
    )

    install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libglog/include/glog DESTINATION ${BM_INSTALL_PREFIX}/include/glog)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libglog/include/gflags DESTINATION ${BM_INSTALL_PREFIX}/include/gflags)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libglog/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)
else()
endif()