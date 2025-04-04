if("${MLIR_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set MLIR_SDK_ROOT before building IVE library.")
elseif(EXISTS "${MLIR_SDK_ROOT}")
  message("-- Found MLIR_SDK_ROOT (directory: ${MLIR_SDK_ROOT})")
else()
  message(FATAL_ERROR "${MLIR_SDK_ROOT} is not a valid folder.")
endif()

if("${CVI_PLATFORM}" STREQUAL "SOPHON")
  set(MLIR_INCLUDES ${MLIR_SDK_ROOT}/libsophon-0.4.9/include/)
elseif("${CVI_PLATFORM}" STREQUAL "BM1688")
  set(MLIR_INCLUDES ${MLIR_SDK_ROOT}/include/)
elseif("${CVI_PLATFORM}" STREQUAL "BM1684X")
  set(MLIR_INCLUDES ${MLIR_SDK_ROOT}/include/)
else()
  set(MLIR_INCLUDES ${MLIR_SDK_ROOT}/include/)
endif()

if("${CVI_PLATFORM}" STREQUAL "SOPHON")
  set(MLIR_LIBS
      ${MLIR_SDK_ROOT}/libsophon-0.4.9/lib/libbmrt.so
      ${MLIR_SDK_ROOT}/libsophon-0.4.9/lib/libbmlib.so
  )
elseif("${CVI_PLATFORM}" STREQUAL "BM1688")
  set(MLIR_LIBS
      ${MLIR_SDK_ROOT}/lib/libbmrt.so
      ${MLIR_SDK_ROOT}/lib/libbmlib.so
  )
elseif("${CVI_PLATFORM}" STREQUAL "BM1684X")
  set(MLIR_LIBS
      ${MLIR_SDK_ROOT}/lib/libbmrt.so
      ${MLIR_SDK_ROOT}/lib/libbmlib.so
      ${MLIR_SDK_ROOT}/lib/libbmjpuapi.so
      ${MLIR_SDK_ROOT}/lib/libbmjpulite.so
      ${MLIR_SDK_ROOT}/lib/libbmcv.so
      ${MLIR_SDK_ROOT}/lib/libbmvpuapi.so
      ${MLIR_SDK_ROOT}/lib/libyuv.so
      ${MLIR_SDK_ROOT}/lib/libbmvpulite.so.0
      ${MLIR_SDK_ROOT}/lib/libbmvideo.so.0
      ${MLIR_SDK_ROOT}/lib/libbmion.so.0
      ${MLIR_SDK_ROOT}/lib/libbmlib.so
      ${MLIR_SDK_ROOT}/lib/libyuv.so
      ${MLIR_SDK_ROOT}/lib/libvpp_cmodel.so
  )
else()
  set(MLIR_LIBS
      ${MLIR_SDK_ROOT}/lib/libcnpy.so
      ${MLIR_SDK_ROOT}/lib/libcvikernel.so
      # ${MLIR_SDK_ROOT}/lib/libcvimath.so
      ${MLIR_SDK_ROOT}/lib/libcviruntime.so
  )
endif()

if("${CVI_PLATFORM}" STREQUAL "BM1688")
  return()
endif()

set(MLIR_PATH ${CMAKE_INSTALL_PREFIX}/sample/3rd/tpu)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  install(FILES ${MLIR_LIBS} DESTINATION ${MLIR_PATH}/lib)
  install(PROGRAMS ${MLIR_SDK_ROOT}/lib/libz.so.1.2.11 DESTINATION ${MLIR_PATH}/lib RENAME libz.so.1)
  install(PROGRAMS ${MLIR_SDK_ROOT}/lib/libz.so.1.2.11 DESTINATION ${MLIR_PATH}/lib RENAME libz.so)
  install(DIRECTORY ${MLIR_SDK_ROOT}/include/ DESTINATION ${MLIR_PATH}/include)
endif()