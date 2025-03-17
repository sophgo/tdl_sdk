if("${MIDDLEWARE_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set MIDDLEWARE_SDK_ROOT before building IVE library.")
elseif(EXISTS "${MIDDLEWARE_SDK_ROOT}")
  message("-- Found MIDDLEWARE_SDK_ROOT (directory: ${MIDDLEWARE_SDK_ROOT})")
else()
  message(FATAL_ERROR "${MIDDLEWARE_SDK_ROOT} is not a valid folder.")
endif()


if(${CVI_PLATFORM} STREQUAL "BM1684X")
  #ffmpeg
  message("use ffmpeg on 1684x")
  # set(FFMPEG_ROOT_DIR /opt/sophon/sophon-ffmpeg_1.8.0)
  set(MIDDLEWARE_INCLUDES ${MIDDLEWARE_SDK_ROOT}/include/)
 
  set(FFMPEG_LIB_DIR ${MIDDLEWARE_SDK_ROOT}/lib/)
  file(GLOB FFMPEG_LIBS "${FFMPEG_LIB_DIR}/lib*.so")
  foreach (fname ${FFMPEG_LIBS})
    set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${fname})
  endforeach ()
  # set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} 
  # ${MLIR_SDK_ROOT}/lib/libbmcv.so
  # ${MLIR_SDK_ROOT}/lib/libbmjpuapi.so
  # ${MLIR_SDK_ROOT}/lib/libbmipulite.so  
  # )

  # file(GLOB ISP_LIBS "${ISP_ROOT_DIR}/lib/*.so")
  # foreach (fname ${ISP_LIBS})
  #   set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${fname})
  # endforeach ()
  return()
elseif(${CVI_PLATFORM} STREQUAL "BM1688")
  #ffmpeg
  message("use ffmpeg on 1688")
  # set(FFMPEG_ROOT_DIR /opt/sophon/sophon-ffmpeg_1.8.0)
  set(MIDDLEWARE_INCLUDES ${MIDDLEWARE_SDK_ROOT}/include/)
 
  set(FFMPEG_LIB_DIR ${MIDDLEWARE_SDK_ROOT}/lib/)
  file(GLOB FFMPEG_LIBS "${FFMPEG_LIB_DIR}/lib*.so")
  foreach (fname ${FFMPEG_LIBS})
    set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${fname})
  endforeach ()
  set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} 
  ${MLIR_SDK_ROOT}/lib/libbmcv.so
  ${MLIR_SDK_ROOT}/lib/libbmjpeg.so
  ${MLIR_SDK_ROOT}/lib/libbmvenc.so  
  ${MLIR_SDK_ROOT}/lib/libyuv.so
  ${MLIR_SDK_ROOT}/lib/libbmvd.so
  )

  file(GLOB ISP_LIBS "${ISP_ROOT_DIR}/lib/*.so")
  foreach (fname ${ISP_LIBS})
    set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${fname})
  endforeach ()
  return()
endif()

if("${MW_VER}" STREQUAL "v1")
  add_definitions(-D_MIDDLEWARE_V1_)
elseif("${MW_VER}" STREQUAL "v2")
  add_definitions(-D_MIDDLEWARE_V2_)
endif()

string(TOLOWER ${CVI_PLATFORM} CVI_PLATFORM_LOWER)
if("${CVI_PLATFORM}" STREQUAL "SOPHON")
  set(ISP_HEADER_PATH ${MIDDLEWARE_SDK_ROOT}/modules/isp/include/cv186x)
else()
  set(ISP_HEADER_PATH ${MIDDLEWARE_SDK_ROOT}/include/isp/${CVI_PLATFORM_LOWER}/
                      ${MIDDLEWARE_SDK_ROOT}/component/isp/common
    )
endif()

set(MIDDLEWARE_INCLUDES ${ISP_HEADER_PATH}
                        ${MIDDLEWARE_SDK_ROOT}/include/
                        ${MIDDLEWARE_SDK_ROOT}/include/linux/
)

# Set SAMPLE_LIBS based on platform
if(${CVI_PLATFORM} STREQUAL "SOPHON")
    set(MIDDLEWARE_LIBS 
    ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
    ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
    add_definitions(-DSENSOR_GCORE_GC4653)
else()
    # Default libraries for other platforms
    set(MIDDLEWARE_LIBS
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpu.so
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
        ${MLIR_SDK_ROOT}/lib/libz.so
    )

    # Additional libraries for v2
    if(${MW_VER} STREQUAL "v2")
        if(${CVI_PLATFORM} STRLESS "SOPHON")
            set(MIDDLEWARE_LIBS
                ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
                # ${MIDDLEWARE_SDK_ROOT}/lib/libldc.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
                ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
                ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
                ${MLIR_SDK_ROOT}/lib/libz.so
            )
        endif()
        set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so)
    else()
        set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so)
    endif()

    # Add isp_algo library for non-CV183X platforms
    if(NOT ${CVI_PLATFORM} STREQUAL "CV183X")
        set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
    endif()
endif()

message("MIDDLEWARE_INCLUDES: ${MIDDLEWARE_INCLUDES}")
message("KERNEL_ROOT: ${KERNEL_ROOT}")
message("MIDDLEWARE_LIBS: ${MIDDLEWARE_LIBS}")

set(MIDDLEWARE_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/middleware/${MW_VER})
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/include/ DESTINATION ${MIDDLEWARE_PATH}/include)
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/sample/common/ DESTINATION ${MIDDLEWARE_PATH}/include)
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/lib/ DESTINATION ${MIDDLEWARE_PATH}/lib)
endif()
