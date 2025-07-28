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
elseif(${CVI_PLATFORM} STREQUAL "BM1684")
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
elseif(${CVI_PLATFORM} STREQUAL "CMODEL_CV184X")
  message("use libcmodel on CMODEL_CV184X")
  # set(MIDDLEWARE_INCLUDES ${MIDDLEWARE_SDK_ROOT}/include/)
  set(MIDDLEWARE_LIBS ${MLIR_SDK_ROOT}/lib/libbmlib.so
                      ${MLIR_SDK_ROOT}/lib/libbmlib.so.0
                      ${MLIR_SDK_ROOT}/lib/libcmodel.so
  )
  # message(FATAL_ERROR "MIDDLEWARE_LIBS: ${MIDDLEWARE_LIBS}")
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

string(TOLOWER "${CVI_PLATFORM}" CHIP_ARCH_LOWER)
set(MIDDLEWARE_INCLUDES ${ISP_HEADER_PATH}
                        ${MIDDLEWARE_SDK_ROOT}/include/
                        ${MIDDLEWARE_SDK_ROOT}/include/isp/
                        ${MIDDLEWARE_SDK_ROOT}/include/linux/
                        ${MIDDLEWARE_SDK_ROOT}/3rdparty/inih/
                        ${MIDDLEWARE_SDK_ROOT}/sample/common/
                        ${MIDDLEWARE_SDK_ROOT}/sample_app/common/
                        ${MIDDLEWARE_SDK_ROOT}/component/panel/${CHIP_ARCH_LOWER}
)

# Set SAMPLE_OBJ based on platform
if(${CVI_PLATFORM} STREQUAL "CV180X" OR
   ${CVI_PLATFORM} STREQUAL "CV181X" OR
   ${CVI_PLATFORM} STREQUAL "CV182X" OR
   ${CVI_PLATFORM} STREQUAL "CV183X")
  set(MIDDLEWARE_OBJ
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_platform.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_sys.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_sensor.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_vi.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_isp.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_vpss.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_venc.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_vo.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_bin.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_peripheral.o
    )
elseif(${CVI_PLATFORM} STREQUAL "CV184X")
  set(MIDDLEWARE_OBJ
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_platform.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_sys.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_vi.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_isp.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_vpss.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_venc.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_vo.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_bin.o
    ${MIDDLEWARE_SDK_ROOT}/sample_app/common/sample_common_peripheral.o
    )
elseif(${CVI_PLATFORM} STREQUAL "SOPHON")
  set(MIDDLEWARE_OBJ
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_platform.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_sys.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_vi.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_isp.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_vpss.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_venc.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_vo.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_bin.o
    ${MIDDLEWARE_SDK_ROOT}/sample/common/sample_common_peripheral.o
    )
else()
  set(MIDDLEWARE_OBJ "")
endif()

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
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libhdmitx.so)
    set(MIDDLEWARE_LIBS_STATIC
        "-Wl,--whole-archive"
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvi.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvo.a
        ${MIDDLEWARE_SDK_ROOT}/lib/librgn.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.a
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libhdmitx.a
        "-Wl,--no-whole-archive"
        ${MLIR_SDK_ROOT}/lib/libz.a)
    add_definitions(-DSENSOR_GCORE_GC4653)
elseif(${CVI_PLATFORM} STREQUAL "CV184X")
    set(COMMON_ZLIB_URL_PREFIX "ftp://${FTP_SERVER_NAME}:${FTP_SERVER_PWD}@${FTP_SERVER_IP}/sw_rls/third_party/latest/")
    if(EXISTS "${OSS_TARBALL_PATH}/zlib.tar.gz")
      set(ZLIB_URL ${OSS_TARBALL_PATH}/zlib.tar.gz)
    elseif(EXISTS "${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/zlib.tar.gz")
      set(ZLIB_URL ${TOP_DIR}/oss/oss_release_tarball/${ARCHITECTURE}/zlib.tar.gz)
    elseif(IS_LOCAL)
      set(ZLIB_URL ${COMMON_ZLIB_URL_PREFIX}${ARCHITECTURE}/zlib.tar.gz)
    else()
      message(FATAL_ERROR "Failed to find zlib.tar.gz")
    endif()

    if(NOT IS_DIRECTORY "${BUILD_DOWNLOAD_DIR}/zlib-src/lib")
      FetchContent_Declare(
        zlib
        URL ${ZLIB_URL}
      )
      FetchContent_MakeAvailable(zlib)
      message("Zlib downloaded from ${ZLIB_URL} to ${zlib_SOURCE_DIR}")
    endif()
    set(ZLIB_ROOT ${BUILD_DOWNLOAD_DIR}/zlib-src)
    include_directories(${BUILD_DOWNLOAD_DIR}/zlib-src/include)

    set(MIDDLEWARE_LIBS
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsensor.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsensor_cfg.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libmipi.so
        ${ZLIB_ROOT}/lib/libz.so)
    set(MIDDLEWARE_LIBS_STATIC
        "-Wl,--whole-archive"
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.a
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvi.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvo.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libsensor.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libsensor_cfg.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libmipi.a
        "-Wl,--no-whole-archive"
        ${ZLIB_ROOT}/lib/libz.a)
    if(${CONFIG_DUAL_OS} STREQUAL "OFF")
        set(MIDDLEWARE_LIBS
          ${MIDDLEWARE_LIBS}
          ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so
          ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
          ${MIDDLEWARE_SDK_ROOT}/lib/libsensor_i2c.so)
        set(MIDDLEWARE_LIBS_STATIC
          ${MIDDLEWARE_LIBS_STATIC}
          ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.a
          ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.a
          ${MIDDLEWARE_SDK_ROOT}/lib/libsensor_i2c.a)
    endif()
else()
    # Default libraries for other platforms
    set(MIDDLEWARE_LIBS
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpu.so
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
        # ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
        ${MLIR_SDK_ROOT}/lib/libz.so)
    set(MIDDLEWARE_LIBS_STATIC
        "-Wl,--whole-archive"
        ${MIDDLEWARE_SDK_ROOT}/lib/libsys.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvpu.a
        ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.a
        # ${MIDDLEWARE_SDK_ROOT}/lib/libsample.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libisp.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libawb.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libae.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libaf.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.a
        ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.a
        "-Wl,--no-whole-archive"
        ${MLIR_SDK_ROOT}/lib/libz.a)

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
                # ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
                ${MLIR_SDK_ROOT}/lib/libz.so)
            set(MIDDLEWARE_LIBS_STATIC
                "-Wl,--whole-archive"
                ${MIDDLEWARE_SDK_ROOT}/lib/libsys.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libvi.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libvo.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.a
                # ${MIDDLEWARE_SDK_ROOT}/lib/libldc.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.a
                ${MIDDLEWARE_SDK_ROOT}/lib/librgn.a
                ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.a
                # ${MIDDLEWARE_SDK_ROOT}/lib/libsample.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libisp.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libawb.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libae.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libaf.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.a
                ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.a
                "-Wl,--no-whole-archive"
                ${MLIR_SDK_ROOT}/lib/libz.a)
        endif()
        set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so)
        set(MIDDLEWARE_LIBS_STATIC ${MIDDLEWARE_LIBS_STATIC} ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.a)
    else()
        set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so)
        set(MIDDLEWARE_LIBS_STATIC ${MIDDLEWARE_LIBS_STATIC} ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.a)
    endif()

    # Add isp_algo library for non-CV183X platforms
    if(NOT ${CVI_PLATFORM} STREQUAL "CV183X")
        set(MIDDLEWARE_LIBS ${MIDDLEWARE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
        set(MIDDLEWARE_LIBS_STATIC ${MIDDLEWARE_LIBS_STATIC} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.a)
    endif()
endif()

message("MIDDLEWARE_INCLUDES: ${MIDDLEWARE_INCLUDES}")
message("KERNEL_ROOT: ${KERNEL_ROOT}")
message("MIDDLEWARE_LIBS: ${MIDDLEWARE_LIBS}")
message("MIDDLEWARE_LIBS_STATIC: ${MIDDLEWARE_LIBS_STATIC}")

set(MIDDLEWARE_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/middleware/${MW_VER})
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/include/ DESTINATION ${MIDDLEWARE_PATH}/include)
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/sample/common/ DESTINATION ${MIDDLEWARE_PATH}/include)
  install(DIRECTORY ${MIDDLEWARE_SDK_ROOT}/lib/ DESTINATION ${MIDDLEWARE_PATH}/lib)
endif()
