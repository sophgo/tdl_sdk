

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

if(${CVI_PLATFORM} STREQUAL "ATHENA2")
set(SAMPLE_LIBS
    ${MIDDLEWARE_SDK_ROOT}/lib/libsys.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvi.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvpss.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvenc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvdec.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libvo.so
    ${MIDDLEWARE_SDK_ROOT}/lib/librgn.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libgdc.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsample.so
    ${MIDDLEWARE_SDK_ROOT}/lib/3rd/libini.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libsns_full.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libawb.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libae.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libaf.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin_isp.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libcvi_bin.so
    ${MIDDLEWARE_SDK_ROOT}/lib/libisp_algo.so)
else()
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
endif()

#test_model
buildninstallcpp(NAME test_img_lpd_lpr INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_od INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_fd INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_hand_detection INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_hand_cls INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_pet_detection INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_vehicle_detection INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_handfaceperson_det INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_md INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_fdfr INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_pose INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_fall_monitor INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_img_hand_keypoint INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_vpss_pd INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_vpss_fuse_vehicle INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_audio_cls INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_face_cap INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_capture INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolov5 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolov5_roi INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_face_landmarker_det2 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolov8 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolov6 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolov7 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolo INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_image_classification INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME sample_yolox INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_ppyoloe INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_consumer_counting INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_people_vehicle INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME test_img_hrnet INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS})
buildninstallcpp(NAME test_dms_landmarks INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation cvi_tdl_app pthread ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)


#eval_model
buildninstallcpp(NAME eval_hand_dataset INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_hand_keypoint INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_head_person INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_face_landmarker_det2 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_yolov5 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_yolov7 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_yolov6 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_yolov8 INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_yolox INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_image_classification INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_ppyoloe INC ${REG_INCLUDES} DEPS cvi_tdl cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_pose_dataset INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)
buildninstallcpp(NAME eval_dms_landmarks_imgs INC ${REG_INCLUDES} DEPS cvi_tdl  cvi_tdl_evaluation ${SAMPLE_LIBS} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sys_utils.cpp)

install(FILES sample_yolo.cpp sample_yolov5.cpp sample_yolov5_roi.cpp sample_yolov6.cpp sample_yolov7.cpp sample_yolov8.cpp sample_yolox.cpp DESTINATION sample/cvi_tdl/)
