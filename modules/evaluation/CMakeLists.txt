include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils)

if(DEFINED NO_OPENCV)
set(EVA_SRCS img_process.cpp)
else()
add_subdirectory(coco)
add_subdirectory(lfw)
add_subdirectory(wider_face)
add_subdirectory(market1501)
add_subdirectory(wflw)
add_subdirectory(cityscapes)
add_subdirectory(cvi_lpdr)

set(EVA_SRCS $<TARGET_OBJECTS:coco>
             $<TARGET_OBJECTS:lfw>
             $<TARGET_OBJECTS:wider_face>
             $<TARGET_OBJECTS:market1501>
             $<TARGET_OBJECTS:wflw>
             $<TARGET_OBJECTS:cityscapes>
             $<TARGET_OBJECTS:cvi_lpdr>
             evaluation.cpp
             img_process.cpp)
set(LINK_LIB ${OPENCV_LIBS_IMGPROC})
set(LINK_LIB_STATIC ${OPENCV_LIBS_IMCODEC_STATIC})
endif()

project(cvi_tdl_evaluation)
add_library(${PROJECT_NAME} SHARED ${EVA_SRCS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs ${LINK_LIB})
install(TARGETS ${PROJECT_NAME} DESTINATION sample/3rd/lib)

project(cvi_tdl_evaluation-static)
add_library(${PROJECT_NAME} STATIC ${EVA_SRCS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs ${LINK_LIB_STATIC})
SET_TARGET_PROPERTIES(cvi_tdl_evaluation-static PROPERTIES OUTPUT_NAME "cvi_tdl_evaluation")
install(TARGETS ${PROJECT_NAME} DESTINATION sample/3rd/lib)
