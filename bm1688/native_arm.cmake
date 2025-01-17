#######################################################################################################################
# native arm build settings
#######################################################################################################################
if (USE_CHIP_TYPE STREQUAL BM1688)
  message(STATUS "Build for bm1688")
  # c/cpp compilier settings

  if (USE_FFMPEG)
    if (CMAKE_COMPILER_IS_GNUCXX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -D__STDC_CONSTANT_MACROS")
    endif (CMAKE_COMPILER_IS_GNUCXX)
  endif ()
  add_definitions(-DUSE_ARM)
  add_definitions(-DUSE_ARMV8)
  add_definitions(-DUSE_BM1684)
  set(OS_NAME "linux")
  set(ARCH_NAME "armv8")

  # sdk settings

  set(LIB_SOPHON_ROOT /opt/sophon/libsophon-0.4.10)
  set(NNTOOLCHAIN_LIB
      ${LIB_SOPHON_ROOT}/lib/libbmrt.so
      ${LIB_SOPHON_ROOT}/lib/libcpuop.so
      ${LIB_SOPHON_ROOT}/lib/libusercpu.so
      ${LIB_SOPHON_ROOT}/lib/libbmlib.so
      ${LIB_SOPHON_ROOT}/lib/libbmcv.so
  )
  
  include_directories(${LIB_SOPHON_ROOT}/include)

  # opencv settings
  set(OpenCV_ROOT_DIR /opt/sophon/sophon-opencv_1.8.0)
  set(OpenCV_INCLUDE_DIRS ${OpenCV_ROOT_DIR}/include/opencv4)
  set(OpenCV_LIB_DIR ${OpenCV_ROOT_DIR}/lib)
  set(OpenCV_LIBRARIES ${OpenCV_LIB_DIR}/libopencv_core.so ${OpenCV_LIB_DIR}/libopencv_imgproc.so
    ${OpenCV_LIB_DIR}/libopencv_highgui.so ${OpenCV_LIB_DIR}/libopencv_imgcodecs.so
    ${OpenCV_LIB_DIR}/libopencv_videoio.so ${OpenCV_LIB_DIR}/libopencv_calib3d.so)
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  message(STATUS "OpenCV Include Dir: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV Libs: ${OpenCV_LIBRARIES}")

  #ffmpeg
  message("use ffmpeg on 1688")
  set(FFMPEG_ROOT_DIR /opt/sophon/sophon-ffmpeg_1.8.0)
  set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT_DIR}/include/)
  include_directories(SYSTEM ${FFMPEG_INCLUDE_DIR})
  set(FFMPEG_LIB_DIR ${FFMPEG_ROOT_DIR}/lib/)
  file(GLOB FFMPEG_LIBS "${FFMPEG_LIB_DIR}/lib*.so")
  foreach (fname ${FFMPEG_LIBS})
    set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${fname})
  endforeach ()
  ## boost
  include_directories(/usr/include/boost/include)
  set(Boost_LIBRARIES /usr/lib/aarch64-linux-gnu/libboost_system.so)

  #eigen
  include_directories(/usr/include/eigen3/)



else()
  message(FATAL_ERROR "Undefined USE_CHIP_TYPE!")
endif ()