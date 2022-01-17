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

set(FRS_SRCS $<TARGET_OBJECTS:digital_tracking>
             $<TARGET_OBJECTS:draw_rect>
             $<TARGET_OBJECTS:tracker>
             $<TARGET_OBJECTS:area_detect>
             $<TARGET_OBJECTS:feature_matching>
	     $<TARGET_OBJECTS:face_angle>
             cviai_service.cpp)

set(FRS_DEPS cviai_core)
set(FRS_DEPS ${FRS_DEPS} ${OPENCV_LIBS_IMGPROC} ${OPENCV_LIBS_MIN})

project(cviai_service)
add_library(${PROJECT_NAME} SHARED ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} ${FRS_DEPS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_service-static)
add_library(${PROJECT_NAME} STATIC ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
