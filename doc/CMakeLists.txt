# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>
cmake_minimum_required(VERSION 3.12)
if(NOT CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/")
endif()
find_package(Doxygen REQUIRED)
if (NOT DOXYGEN_DOT_FOUND)
    message(FATAL_ERROR "Graphviz not installed, sudo apt install first.")
endif()
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    message(${DOXYGEN_OUT})
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")
    add_custom_target(
    doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc_doxygen
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
    )
else()
    message(FATAL_ERROR "Doxygen needs to be installed to generate doc.")
endif()
