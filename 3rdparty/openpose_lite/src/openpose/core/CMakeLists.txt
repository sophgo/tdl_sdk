set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;mm;CPP;cl)
set(SOURCES_OP_CORE
    array.cpp
    datum.cpp
    point.cpp
    rectangle.cpp)

#add_library(openpose_core SHARED ${SOURCES_OP_CORE})
#target_link_libraries(openpose_core openpose_utils ${OpenCV_LIBS})