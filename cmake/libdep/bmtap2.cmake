# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_BMTAP2_DIR}" STREQUAL "")
    if(USE_LEGACY_BMTAP2)
        set(BMTAP2_FOLDER_NAME "bmtap2")
        set(BMTAP2_SHA1 "88664c0c35ea4ef258ab0ac6da732de52dfbf684")
    else()
        set(BMTAP2_FOLDER_NAME "bmtap2_v2")
        set(BMTAP2_SHA1 "dada306d9d67ff381d73bccb1a08a93726946181")
    endif()

    if("${BM_NETWORKS_TARGET_BASENAME}" STREQUAL "soc_${CMAKE_BOARD_TYPE}_asic")
        set(TAR_NAME "${PREBUILT_DIR}/${BMTAP2_FOLDER_NAME}_${BM_NETWORKS_TARGET_BASENAME}.tar.gz")
    elseif("${BM_NETWORKS_TARGET_BASENAME}" STREQUAL "cmodel_${CMAKE_BOARD_TYPE}")
        set(TAR_NAME "${PREBUILT_DIR}/${BMTAP2_FOLDER_NAME}_cmodel.tar.gz")
    elseif("${BM_NETWORKS_TARGET_BASENAME}" STREQUAL "usb_${CMAKE_BOARD_TYPE}")
        set(TAR_NAME "${PREBUILT_DIR}/${BMTAP2_FOLDER_NAME}_usb.tar.gz")
    endif()
    extract_package("${TAR_NAME}" "${PREBUILT_DIR}" "${BMTAP2_FOLDER_NAME}" "${BMTAP2_SHA1}")

    if("${BM_NETWORKS_TARGET_BASENAME}" STREQUAL "usb_${CMAKE_BOARD_TYPE}")
        set(LIBDEP_BMTAP2_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/${BMTAP2_FOLDER_NAME}/${HOST_OS_VERSION})
    else()
        set(LIBDEP_BMTAP2_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BM_NETWORKS_TARGET_BASENAME}/${BMTAP2_FOLDER_NAME})
    endif()
endif()

include_directories(
    ${LIBDEP_BMTAP2_DIR}/include/
)

set(BM_LIBS
    ${LIBDEP_BMTAP2_DIR}/lib/libbmodel.so
    ${LIBDEP_BMTAP2_DIR}/lib/libbmkernel.so
    ${LIBDEP_BMTAP2_DIR}/lib/libbmruntime.so
    ${LIBDEP_BMTAP2_DIR}/lib/libbmutils.so
)

install(DIRECTORY ${LIBDEP_BMTAP2_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include/)
install(DIRECTORY ${LIBDEP_BMTAP2_DIR}/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)