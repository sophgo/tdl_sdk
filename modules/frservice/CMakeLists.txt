include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../service)

set(FRS_SRCS $<TARGET_OBJECTS:digital_tracking>
             cviai_frservice.cpp)

project(cviai_frservice)
add_library(${PROJECT_NAME} SHARED ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_frservice-static)
add_library(${PROJECT_NAME} STATIC ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
