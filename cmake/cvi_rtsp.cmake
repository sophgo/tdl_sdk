get_filename_component(FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME_WE)

set(SOURCE_DIR $ENV{TOP_DIR}/cvi_rtsp/install)

set(CVI_RTSP_LIBPATH ${SOURCE_DIR}/lib/libcvi_rtsp.so)
set(CVI_RTSP_INCLUDE ${SOURCE_DIR}/include/cvi_rtsp)

set(RTSP_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/rtsp)
install(FILES ${CVI_RTSP_LIBPATH} DESTINATION ${RTSP_PATH}/lib)
install(DIRECTORY ${CVI_RTSP_INCLUDE} DESTINATION ${RTSP_PATH}/include)
