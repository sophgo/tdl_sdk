# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang yangwen.huang@bitmain.com

if(CMAKE_TOOLCHAIN_FILE)
    if("${LIBDEP_FFMPEG_DIR}" STREQUAL "")
        extract_package("${LIBDEP_MIDDLEWARE_DIR}/ffmpeg_${BM_NETWORKS_TARGET_BASENAME}.tar.gz" "${LIBDEP_MIDDLEWARE_DIR}" "ffmpeg" "")
        set(LIBDEP_FFMPEG_DIR ${LIBDEP_MIDDLEWARE_DIR}/ffmpeg/usr/local)
    endif()
    set(FFMPEG_INCLUDE_DIRS
        ${LIBDEP_FFMPEG_DIR}/include
        ${LIBDEP_FFMPEG_DIR}/libpostproc/include
    )
    set(FFMPEG_LIBS
        ${LIBDEP_FFMPEG_DIR}/lib/libpostproc.so
        ${LIBDEP_FFMPEG_DIR}/lib/libavcodec.so
        ${LIBDEP_FFMPEG_DIR}/lib/libavformat.so
        ${LIBDEP_FFMPEG_DIR}/lib/libavdevice.so
        ${LIBDEP_FFMPEG_DIR}/lib/libavfilter.so
        ${LIBDEP_FFMPEG_DIR}/lib/libavutil.so
        ${LIBDEP_FFMPEG_DIR}/lib/libswscale.so
        ${LIBDEP_FFMPEG_DIR}/lib/libswresample.so
    )

    install(DIRECTORY ${LIBDEP_FFMPEG_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include)
    install(DIRECTORY ${LIBDEP_FFMPEG_DIR}/lib/ DESTINATION ${BM_INSTALL_PREFIX}/lib)
endif()