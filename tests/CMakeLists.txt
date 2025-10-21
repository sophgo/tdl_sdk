# Copyright 2020 cvitek Inc.

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(tdl_test)

message("GTEST_INCLUDES:${GTEST_INCLUDES}")

include_directories(${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/nn
                    ${REPO_DIR}/include/components
                    ${GTEST_INCLUDES}
                    ${FBANK_INCLUDES}
                    ${CMAKE_CURRENT_SOURCE_DIR}/common
                    ${REPO_DIR}/src/components/nn
)
if(${CVI_PLATFORM} STREQUAL "BM1688" OR ${CVI_PLATFORM} STREQUAL "BM1684X" OR ${CVI_PLATFORM} STREQUAL "BM1684")
  set(REG_LIBS
  ${MIDDLEWARE_LIBS} ${OPENCV_LIBRARIES}
  )
else()
  set(REG_LIBS  ${MIDDLEWARE_LIBS})
endif()
add_subdirectory(common)
add_subdirectory(regression) 
add_subdirectory(unit_test)

set(CORES_SRCS
               $<TARGET_OBJECTS:common>
               $<TARGET_OBJECTS:regression>
               $<TARGET_OBJECTS:unit_test>
               )

add_executable(test_main ${CMAKE_CURRENT_SOURCE_DIR}/common/test_main.cpp ${CORES_SRCS})
target_link_libraries(test_main tdl_core gtest_main stdc++fs ${REG_LIBS})

add_executable(test_runner ${CMAKE_CURRENT_SOURCE_DIR}/common/test_runner.cpp ${CORES_SRCS})
target_link_libraries(test_runner tdl_core gtest_main stdc++fs ${REG_LIBS})


install(DIRECTORY config DESTINATION tests)
install(FILES scripts/daily_regression.sh PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE DESTINATION tests)
install(TARGETS test_main DESTINATION tests)
install(TARGETS test_runner DESTINATION tests)