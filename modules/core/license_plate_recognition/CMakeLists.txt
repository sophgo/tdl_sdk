project(license_plate_recognition)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
add_library(${PROJECT_NAME} OBJECT license_plate_recognition.cpp
				                   decode_tool.cpp)
