set(RTSP_SOURCE_DIR ${TOP_DIR}/cvi_rtsp)

set(CVI_RTSP_LIB_SHARED ${RTSP_SOURCE_DIR}/install/lib/libcvi_rtsp.so)
set(CVI_RTSP_LIB_STATIC ${RTSP_SOURCE_DIR}/install/lib/libcvi_rtsp.a
                        ${RTSP_SOURCE_DIR}/prebuilt/lib/libBasicUsageEnvironment.a
                        ${RTSP_SOURCE_DIR}/prebuilt/lib/libgroupsock.a
                        ${RTSP_SOURCE_DIR}/prebuilt/lib/libUsageEnvironment.a
                        ${RTSP_SOURCE_DIR}/prebuilt/lib/libliveMedia.a)
set(CVI_RTSP_INCLUDE ${RTSP_SOURCE_DIR}/install/include/cvi_rtsp)

set(RTSP_PATH ${CMAKE_INSTALL_PREFIX}/sample/cvi_rtsp)
install(FILES ${CVI_RTSP_LIB_SHARED} DESTINATION ${RTSP_PATH}/lib)
install(FILES ${CVI_RTSP_LIB_STATIC} DESTINATION ${RTSP_PATH}/lib)
install(DIRECTORY ${CVI_RTSP_INCLUDE} DESTINATION ${RTSP_PATH}/include)
