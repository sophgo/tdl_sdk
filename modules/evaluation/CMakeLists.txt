add_subdirectory(coco)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils)

set(EVA_SRCS $<TARGET_OBJECTS:coco>
             cviai_evaluation.cpp)

project(cviai_evaluation)
add_library(${PROJECT_NAME} SHARED ${EVA_SRCS})
target_link_libraries(${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_evaluation-static)
add_library(${PROJECT_NAME} STATIC ${EVA_SRCS})
target_link_libraries(${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
