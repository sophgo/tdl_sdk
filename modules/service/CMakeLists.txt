add_subdirectory(digital_tracking)
add_subdirectory(draw_rect)
add_subdirectory(tracker)
add_subdirectory(area_detect)
add_subdirectory(feature_matching)
add_subdirectory(face_angle)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../service
                    ${IVE_INCLUDES})

set(FRS_SRCS cviai_service.cpp)

project(cviai_service)
add_library(${PROJECT_NAME} OBJECT ${FRS_SRCS})