

project(sample)

# include_directories(
                    # ${MIDDLEWARE_INCLUDES}
                    # ${CMAKE_CURRENT_SOURCE_DIR}
                    # ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    # ${CMAKE_CURRENT_SOURCE_DIR}/../../include/core/utils
                    # ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/core/core
                    # )


function(buildninstallcpp)
  # From https://stackoverflow.com/questions/23327687/how-to-write-a-cmake-function-with-more-than-one-parameter-groups
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of the boolean arguments (only defined ones will be true)
    "NAME" # list of names of mono-valued arguments
    "INC;SRCS;DEPS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN} # arguments of the function to parse, here we take the all original ones
  )
  # note: if it remains unparsed arguments, here, they can be found in variable PARSED_ARGS_UNPARSED_ARGUMENTS
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "You must provide a name")
  endif(NOT PARSED_ARGS_NAME)
  project(${PARSED_ARGS_NAME})
  # For debug purpose.
  # message("Provided additional sources are:")
  # foreach(src ${PARSED_ARGS_SRCS})
  #   message("- ${src}")
  # endforeach(src)
  include_directories(${PARSED_ARGS_INC})
  add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS})
  install(TARGETS ${PROJECT_NAME} DESTINATION bin)
endfunction(buildninstallcpp)


set(SAMPLE_INC
${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${MIDDLEWARE_SDK_ROOT}/include
    ${IVE_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/../md
    ${CMAKE_CURRENT_SOURCE_DIR}/../draw_rect
)

set(SAMPLE_LIBS
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
set(OPENCV_LIB
    ${MLIR_SDK_ROOT}/opencv/lib/libopencv_core.so
    ${MLIR_SDK_ROOT}/opencv/lib/libopencv_imgproc.so
    ${MLIR_SDK_ROOT}/opencv/lib/libopencv_imgcodecs.so
)

if("${CVI_PLATFORM}" STREQUAL "CV181X"  OR "${CVI_PLATFORM}" STREQUAL "CV180X")
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MLIR_SDK_ROOT}/lib/libz.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so
)
else()
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
)
endif()

if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV180X")
  set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

if(NOT "${CVI_PLATFORM}" STREQUAL "CV180X")
buildninstallcpp(NAME sample_lib_draw_rect INC ${SAMPLE_INC} DEPS cviai cviai_evaluation cvi_draw_rect  ${SAMPLE_LIBS} ${IVE_LIBS} ${MIDDLEWARE_LIBS} ${OPENCV_LIB})
endif()
buildninstallcpp(NAME sample_lib_cvimd INC  ${SAMPLE_INC} DEPS cvi_md ${MLIR_LIBS} ${SAMPLE_LIBS} ${IVE_LIBS} ${MIDDLEWARE_LIBS})
buildninstallcpp(NAME sample_lib_preprocess INC ${SAMPLE_INC} DEPS cvi_preprocess ${MLIR_LIBS} ${SAMPLE_LIBS} ${IVE_LIBS} ${MIDDLEWARE_LIBS})