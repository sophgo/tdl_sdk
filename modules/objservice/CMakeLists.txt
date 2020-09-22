include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../service
                    ${IVE_INCLUDES})

set(FRS_SRCS $<TARGET_OBJECTS:digital_tracking>
             $<TARGET_OBJECTS:draw_rect>
             $<TARGET_OBJECTS:tracker>
             $<TARGET_OBJECTS:area_detect>
             cviai_objservice.cpp)

project(cviai_objservice)
add_library(${PROJECT_NAME} SHARED ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_objservice-static)
add_library(${PROJECT_NAME} STATIC ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
