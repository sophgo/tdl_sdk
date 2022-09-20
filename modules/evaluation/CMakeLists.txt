add_subdirectory(coco)
add_subdirectory(lfw)
add_subdirectory(wider_face)
add_subdirectory(market1501)
add_subdirectory(wflw)
add_subdirectory(cityscapes)
add_subdirectory(cvi_lpdr)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils)

set(EVA_SRCS $<TARGET_OBJECTS:coco>
             $<TARGET_OBJECTS:lfw>
             $<TARGET_OBJECTS:wider_face>
             $<TARGET_OBJECTS:market1501>
             $<TARGET_OBJECTS:wflw>
             $<TARGET_OBJECTS:cityscapes>
             $<TARGET_OBJECTS:cvi_lpdr>
             cviai_evaluation.cpp
             cviai_media.cpp)

project(cviai_evaluation)
add_library(${PROJECT_NAME} SHARED ${EVA_SRCS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs  ${OPENCV_LIBS_IMCODEC} ${OPENCV_LIBS_IMGPROC} ${OPENCV_LIBS_MIN} )
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(cviai_evaluation-static)
add_library(${PROJECT_NAME} STATIC ${EVA_SRCS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

