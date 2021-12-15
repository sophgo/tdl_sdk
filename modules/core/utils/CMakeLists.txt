project(utils)
add_library(${PROJECT_NAME} OBJECT core_utils.cpp
                                   neon_utils.cpp
                                   face_utils.cpp
                                   rescale_utils.cpp
                                   demangle.cpp
                                   image_utils.cpp)

if ("${ENABLE_CVIAI_CV_UTILS}" STREQUAL "ON")
    add_subdirectory(cv)
endif()
