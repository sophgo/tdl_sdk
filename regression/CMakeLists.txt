# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(ai_sdk_regression)

function(buildninstall)
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
  add_executable(${PROJECT_NAME} ${PROJECT_NAME}.c ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS})
  #install(TARGETS ${PROJECT_NAME} DESTINATION regression)
endfunction(buildninstall)

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
  if (PROJECT_NAME STREQUAL "test_main")
      install(TARGETS ${PROJECT_NAME} DESTINATION regression)
  endif ()
endfunction(buildninstallcpp)

set(REG_INCLUDES

    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/framework
)


if(${CVI_PLATFORM} STREQUAL "BM1688")
  set(REG_LIBS
  ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES}
  )
else()
  set(REG_LIBS
  ${MIDDLEWARE_LIBS}
  )
endif()
if((${CVI_PLATFORM} STREQUAL "SOPHON") OR (${CVI_PLATFORM} STREQUAL "BM1688"))
  set(UnitTests
  # reg_daily_petdet.cpp
  # reg_daily_handdet.cpp
  # reg_daily_meet.cpp
  reg_daily_person_vehicle.cpp
  # reg_daily_hardhat_det.cpp
  # reg_daily_scrfd_det.cpp
  # reg_daily_lpd_v2.cpp
  # reg_daily_fr.cpp
  # reg_daily_mobiledetion.cpp
  # reg_daily_handcls.cpp
  # reg_daily_vpss_image.cpp
  reg_daily_preprocess.cpp
  reg_daily_scrfd_det_bm.cpp
  # reg_daily_handkeypoint.cpp
  )
else()
set(UnitTests
  reg_daily_fq.cpp
  reg_daily_mobiledet.cpp
  reg_daily_md.cpp
  reg_daily_fr.cpp
  reg_daily_mask_classification.cpp
  reg_daily_lpd.cpp
  reg_daily_lpr.cpp
  reg_daily_mot.cpp
  reg_daily_reid.cpp
  reg_daily_thermal_fd.cpp
  reg_daily_thermal_person_detection.cpp
  reg_daily_core.cpp
  reg_daily_liveness.cpp
  reg_daily_fall.cpp
  reg_daily_retinaface.cpp
  reg_daily_retinafaceIR.cpp
  reg_daily_retinaface_hardhat.cpp
  reg_daily_fdmask.cpp
  reg_daily_eye_classification.cpp
  reg_daily_yawn_classification.cpp
  reg_daily_fl.cpp
  reg_daily_incarod.cpp
  reg_daily_face_cap.cpp
  reg_daily_feature_matching.cpp
  reg_daily_mobiledetion.cpp
)
endif()

set(UnitTest_SRCS
  cvi_tdl_test.cpp
  # raii.cpp
  regression_utils.cpp
  ${UnitTests}
)

set(UnitTest_SRCS
  ${UnitTest_SRCS}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../modules/core
                    
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

buildninstallcpp(NAME test_main INC ${REG_INCLUDES} SRCS ${UnitTest_SRCS} DEPS cvi_tdl ${REG_LIBS} gtest_main stdc++fs )

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tmp/_deps/googletest-src/googletest/include)
# buildninstallcpp(NAME reg_daily_vpss_image INC ${REG_INCLUDES} SRCS reg_daily_vpss_image.cpp DEPS cvi_tdl cvi_tdl_app  ${SAMPLE_LIBS})


if((${CVI_PLATFORM} STREQUAL "SOPHON") OR (${CVI_PLATFORM} STREQUAL "BM1688"))
  set(DAILY_REGRESSION_ASSETS "assets_186x")
elseif("${CVI_PLATFORM}" STREQUAL "CV183X")
  set(DAILY_REGRESSION_ASSETS "assets_183x")
elseif("${CVI_PLATFORM}" STREQUAL "CV182X" )
  set(DAILY_REGRESSION_ASSETS "assets_182x")
elseif("${CVI_PLATFORM}" STREQUAL "CV181X" )
  set(DAILY_REGRESSION_ASSETS "assets_181x")
elseif("${CVI_PLATFORM}" STREQUAL "CV180X" )
  set(DAILY_REGRESSION_ASSETS "assets_180x")
endif()

file(GLOB asset_files "${DAILY_REGRESSION_ASSETS}/*.json")
install(DIRECTORY DESTINATION ${assets})

foreach(file ${asset_files})
  get_filename_component(asset_realpath "${file}" REALPATH)
  install(FILES ${asset_realpath} DESTINATION regression/assets)
endforeach()

install(FILES daily_regression.sh PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE DESTINATION regression)
