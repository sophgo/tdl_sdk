project(consumer_counting_app)

include_directories(
                    ${REPO_DIR}/include
                    ${REPO_DIR}/include/framework
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    )

set(PROJ_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/consumer_counting_app.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/consumer_counting.cpp)

message(STATUS "build ${PROJECT_NAME}")
add_library(consumer_counting_app OBJECT ${PROJ_SRCS})
if(APP_VIDEO_ENABLE)
    target_compile_definitions(consumer_counting_app PRIVATE VIDEO_ENABLE)
endif()