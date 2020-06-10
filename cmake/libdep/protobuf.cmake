# Copyright 2018 Bitmain Inc.
# License
# Author Tim Ho <tim.ho@bitmain.com>

if("${LIBDEP_PROTOBUF_DIR}" STREQUAL "")
    set(TAR_NAME "${PREBUILT_DIR}/libprotobuf_${BM_NETWORKS_TARGET_BASENAME}.tar.gz")
    extract_package("${TAR_NAME}" "${PREBUILT_DIR}" "libprotobuf" "")
    set(LIBDEP_PROTOBUF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libprotobuf)
endif()
include_directories(
    "${LIBDEP_PROTOBUF_DIR}/include"
    "${LIBDEP_PROTOBUF_DIR}/include/google"
)

set(PROTOBUF_LIBRARIES
    "${LIBDEP_PROTOBUF_DIR}/lib/libprotobuf.so"
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libprotobuf/include/google DESTINATION ${BM_INSTALL_PREFIX}/include/google)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libprotobuf/include/ DESTINATION ${BM_INSTALL_PREFIX}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/libprotobuf/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)