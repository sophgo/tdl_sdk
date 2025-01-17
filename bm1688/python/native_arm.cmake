#######################################################################################################################
# native arm build settings
#######################################################################################################################
if(USE_CHIP_TYPE STREQUAL BM1688)
  message(STATUS "Build for bm1688")
  message(STATUS "Build for soc")
  # c/cpp compilier settings
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")

  # sdk settings
  set(LIB_SOPHON_ROOT /opt/sophon/libsophon-0.4.10)
  
  include_directories(${LIB_SOPHON_ROOT}/include)

  # opencv settings
  set(OpenCV_ROOT_DIR /opt/sophon/sophon-opencv_1.8.0)
  set(FFMPEG_ROOT_DIR /opt/sophon/sophon-ffmpeg_1.8.0)
  set(OpenCV_INCLUDE_DIRS
      ${OpenCV_ROOT_DIR}/include/opencv4
      ${FFMPEG_ROOT_DIR}/include/
      )


  set(FFMPEG_LIB_DIR ${FFMPEG_ROOT_DIR}/lib/)
  file(GLOB FFMPEG_LIBS "${FFMPEG_LIB_DIR}/lib*.so")
  foreach (fname ${FFMPEG_LIBS})
    set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} ${fname})
  endforeach ()

  set(OpenCV_LIBRARIES_DIR ${OpenCV_ROOT_DIR}/lib)
  set(OpenCV_LIBRARIES 
      ${OpenCV_LIBRARIES_DIR}/libopencv_core.so 
      ${OpenCV_LIBRARIES_DIR}/libopencv_highgui.so
      ${OpenCV_LIBRARIES_DIR}/libopencv_videoio.so 
      ${OpenCV_LIBRARIES_DIR}/libopencv_imgproc.so
      ${OpenCV_LIBRARIES_DIR}/libopencv_imgcodecs.so
      ${FFMPEG_LIBRARIES}
      )

  # misc settings
  set(ALGONN_ROOT ${CMAKE_SOURCE_DIR}/../)
  set(ALGONN_LIB ${ALGONN_ROOT}/libsdk_common_linux_armv8.so)
else()
  message(FATAL_ERROR "Undefined USE_CHIP_TYPE!")
endif ()

#config python
exec_program(python3-config ARGS --extension-suffix OUTPUT_VARIABLE EXTSUFFIX)
exec_program(python3-config ARGS --includes OUTPUT_VARIABLE PYTHON_INCLUDE_STR)
string(REPLACE "-I" "" PYTHON_INCLUDE_STR ${PYTHON_INCLUDE_STR})

#config pybind11
exec_program(python3 ARGS "-c 'import pybind11;print(pybind11.get_include(user=False)); '" OUTPUT_VARIABLE PYBIND11_INCLUDE_DIRS)
set(PYTHON_INCLUDE_STR "${PYTHON_INCLUDE_STR} ${PYBIND11_INCLUDE_DIRS}")
string(REPLACE " " ";" PYTHON_INCLUDE_STR ${PYTHON_INCLUDE_STR})
list(REMOVE_DUPLICATES PYTHON_INCLUDE_STR)
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_STR})

exec_program(python3-config ARGS --libs OUTPUT_VARIABLE PYTHON_LIB_STR)
string(FIND ${PYTHON_LIB_STR} " -l" SEG_POS)
string(SUBSTRING ${PYTHON_LIB_STR} 0 ${SEG_POS}-1 PYTHON_LIB_STR)
string(REPLACE "-l" "" PYTHON_LIB_NAME ${PYTHON_LIB_STR})
exec_program(python3-config ARGS --ldflags OUTPUT_VARIABLE PYTHON_LIB_STR)
string(FIND ${PYTHON_LIB_STR} " -l" SEG_POS)
string(SUBSTRING ${PYTHON_LIB_STR} 0 ${SEG_POS}-1 PYTHON_LIB_DIR_STR)
string(REPLACE "-L" "" PYTHON_LIB_DIR_STR ${PYTHON_LIB_DIR_STR})
string(REPLACE " " ";" PYTHON_LIB_DIR_STR ${PYTHON_LIB_DIR_STR})
foreach (LIB_DIR ${PYTHON_LIB_DIR_STR})
  set(check_lib "${LIB_DIR}/lib${PYTHON_LIB_NAME}.so")
  if (EXISTS ${check_lib})
    set(PYTHON_LIBRARIES ${check_lib})
    break()
  endif ()
endforeach ()

message(STATUS "python include dirs:" ${PYTHON_INCLUDE_DIRS})
message(STATUS "python libraries:" ${PYTHON_LIBRARIES})






