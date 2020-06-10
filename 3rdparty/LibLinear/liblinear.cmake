set(LIBLINEAR_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear
    ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear/blas
)

set(LIBLINEAR_SRC
    ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear/linear.cpp
    ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear/tron.cpp
    ${CMAKE_SOURCE_DIR}/3rdparty/LibLinear/blas/combined.c
)