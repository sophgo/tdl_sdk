add_subdirectory(face_capture)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils)

set(FRS_SRCS $<TARGET_OBJECTS:face_capture>
             cviai_app.c)

project(cviai_app)
add_library(${PROJECT_NAME} SHARED ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_app-static)
add_library(${PROJECT_NAME} STATIC ${FRS_SRCS})
target_link_libraries(${PROJECT_NAME} cviai_core)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)