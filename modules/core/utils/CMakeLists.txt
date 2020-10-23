project(utils)
add_library(${PROJECT_NAME} OBJECT core_utils.cpp
                                   neon_utils.cpp
                                   face_utils.cpp
                                   rescale_utils.cpp
                                   coco_utils.cpp)