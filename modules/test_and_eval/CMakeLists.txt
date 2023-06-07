

project(test)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils)

set(REG_INCLUDES
    ${MIDDLEWARE_SDK_ROOT}/include
    ${MIDDLEWARE_SDK_ROOT}/sample/common
)

set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
    ${MLIR_SDK_ROOT}/lib/libz.so
)

if("${CVI_PLATFORM}" STREQUAL "CV182X" OR "${CVI_PLATFORM}" STREQUAL "CV181X")
  set(SAMPLE_LIBS ${SAMPLE_LIBS} ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
endif()

if("${CVI_PLATFORM}" STREQUAL "CV181X" OR "${CVI_PLATFORM}" STREQUAL "CV180X")
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libmisc.so
)
else()
set(SAMPLE_LIBS ${SAMPLE_LIBS}
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_vcodec.so
)
endif()
#test_model
buildninstallcpp(NAME test_img_od INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_fd INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_hand_detection INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_hand_cls INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_pet_detection INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_vehicle_detection INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_handfaceperson_det INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_md INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_fdfr INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_simcc_pose INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_hand_keypoint INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_vpss_pd INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_audio_cls INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_face_cap INC ${REG_INCLUDES} DEPS cviai cviai_evaluation cviai_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_capture INC ${REG_INCLUDES} DEPS cviai cviai_evaluation cviai_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
#eval_model
buildninstallcpp(NAME eval_hand_dataset INC ${REG_INCLUDES} DEPS cviai cviai_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_hand_keypoint INC ${REG_INCLUDES} DEPS cviai  cviai_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)