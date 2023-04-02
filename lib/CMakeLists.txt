

include_directories(${IVE_INCLUDES}
                    ${MIDDLEWARE_INCLUDES}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/ive
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/motion_detection
                    ${CMAKE_CURRENT_SOURCE_DIR})

set(PROJ_SRCS $<TARGET_OBJECTS:ivewrapper> 
              $<TARGET_OBJECTS:motion_detection>
              ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/utils/ccl.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/../modules/core/utils/profiler.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/cvi_md.cpp )
project(cvi_md)

set(MD_CORES_DEPS ${MLIR_LIBS} ${MIDDLEWARE_LIBS} ${IVE_LIBS})
add_library(${PROJECT_NAME} SHARED ${PROJ_SRCS})
target_link_libraries(${PROJECT_NAME} ${MD_CORES_DEPS})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cvi_md-static)
add_library(${PROJECT_NAME} STATIC ${PROJ_SRCS})
target_link_libraries(${PROJECT_NAME} ${MD_CORES_DEPS})
# target_link_libraries(${PROJECT_NAME} -lstdc++fs)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cvi_md.h DESTINATION include/md)

