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
  if (PROJECT_NAME STREQUAL "test_main" OR PROJECT_NAME STREQUAL "generate_regression_info")
      install(TARGETS ${PROJECT_NAME} DESTINATION regression)
  endif ()
endfunction(buildninstallcpp)

set(REG_INCLUDES

    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/components
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
if((${CVI_PLATFORM} STREQUAL "SOPHON") OR (${CVI_PLATFORM} STREQUAL "BM1688") OR (${CVI_PLATFORM} STREQUAL "CV181X") OR (${CVI_PLATFORM} STREQUAL "CV184X")) 
  set(UnitTests

  # reg_daily_face_attribute_cls.cpp
  # reg_daily_mobiledetion_bm.cpp
  # reg_daily_preprocess.cpp
  # reg_daily_scrfd_det_bm.cpp
  # reg_daily_landmarkdet2_bm.cpp
  # reg_daily_face_feature_bm.cpp
  # reg_daily_handkeypoint.cpp
  reg_daily_det.cpp
  reg_daily_cls.cpp
  reg_daily_face_attribute_cls
  reg_daily_keypoint.cpp
  reg_daily_matcher.cpp
  reg_daily_feature.cpp
  reg_daily_segmentation.cpp
  reg_daily_ocr.cpp
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

include_directories(${REPO_DIR}/include/framework/utils
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
)

buildninstallcpp(NAME test_main INC ${REG_INCLUDES} SRCS ${UnitTest_SRCS} DEPS tdl_core tdl_ex ${REG_LIBS} gtest_main stdc++fs )
buildninstallcpp(NAME generate_regression_info INC ${REG_INCLUDES} SRCS generate_regression_info.cpp DEPS tdl_core tdl_ex ${REG_LIBS} gtest_main stdc++fs )


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
