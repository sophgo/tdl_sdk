# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if(CMAKE_TOOLCHAIN_FILE)
    if("${LIBDEP_BMVID_DIR}" STREQUAL "")
        extract_package("${LIBDEP_MIDDLEWARE_DIR}/decode_${BM_NETWORKS_TARGET_BASENAME}.tar.gz" "${LIBDEP_MIDDLEWARE_DIR}" "decode" "")
        set(LIBDEP_BMVID_DIR ${LIBDEP_MIDDLEWARE_DIR}/decode)
    endif()
    set(Bmvid_LIBS
        ${LIBDEP_BMVID_DIR}/lib/libvideo_bm.so
    )
    if(USE_BSPSDK STREQUAL "false")
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/prebuilt/soc_bm1880_asic/middleware-soc/decode/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)
    endif()
endif()