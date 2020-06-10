# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

function(extract_package TAR_NAME EXTRACT_DIR BASEFOLDER VERSION)
    if (NOT EXISTS "${TAR_NAME}")
        message(FATAL_ERROR "${TAR_NAME} is missing!")
    endif()
    if(EXISTS "${EXTRACT_DIR}/${BASEFOLDER}/do_not_edit_this_file")
        execute_process(COMMAND cat ${EXTRACT_DIR}/${BASEFOLDER}/do_not_edit_this_file
                        OUTPUT_VARIABLE LIB_SHA
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT "${LIB_SHA}" STREQUAL "${VERSION}")
            message(FATAL_ERROR "${EXTRACT_DIR}/${BASEFOLDER} is an older version, please remove it manually.")
        endif()
    endif()
    if (NOT EXISTS "${EXTRACT_DIR}/${BASEFOLDER}")
        execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${TAR_NAME}
        WORKING_DIRECTORY ${EXTRACT_DIR}
        )
        if (NOT EXISTS "${EXTRACT_DIR}/${BASEFOLDER}")
            message(FATAL_ERROR "${TAR_NAME} unzip failed!")
        endif()
    endif()
endfunction(extract_package)

function(set_library_ext)
    cmake_parse_arguments(
        FEXT # prefix of output variables
        "OPTION" # list of names of the boolean arguments (only defined ones will be true)
        "NAME;TYPE;INSTALL" # list of names of mono-valued arguments
        "SRC;DEP;DEF" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )
    if(FEXT_OPTION)
        add_library(${FEXT_NAME} ${FEXT_TYPE} ${FEXT_SRC})
        target_link_libraries(${FEXT_NAME} ${FEXT_DEP})
        set_target_properties(${FEXT_NAME} PROPERTIES
                              VERSION ${QNN_VERSION_STRING}
                              SOVERSION ${QNN_VERSION_MAJOR})
        target_compile_definitions(${FEXT_NAME} PRIVATE ${FEXT_DEF})
        if(FEXT_INSTALL)
            install(TARGETS ${FEXT_NAME} DESTINATION ${BM_INSTALL_PREFIX}/lib)
            install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${BM_INSTALL_PREFIX}/include/${BM_INSTALL_INCLUDENAME})
        endif()
    endif()
endfunction(set_library_ext)

# FIXME: This function is not working properly
function(set_library_src)
cmake_parse_arguments(
        FEXT # prefix of output variables
        "OPTION" # list of names of the boolean arguments (only defined ones will be true)
        "" # list of names of mono-valued arguments
        "SRC;DEF;VAR;DEFVAR" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )
if(FEXT_OPTION)
    set(${FEXT_VAR} "${${FEXT_VAR}} ${${FEXT_SRC}}" PARENT_SCOPE)
    if(FEXT_DEF)
       set(${FEXT_DEFVAR} "${${FEXT_DEFVAR}} ${${FEXT_DEF}}" PARENT_SCOPE)
    endif()
endif()
endfunction(set_library_src)

function(add_test_ext)
cmake_parse_arguments(
        FEXT # prefix of output variables
        "WILLPASS" # list of names of the boolean arguments (only defined ones will be true)
        "NAME" # list of names of mono-valued arguments
        "CMD;REGP;REGF;LD" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )
add_test(${FEXT_NAME} ${FEXT_CMD})
if(${FEXT_WILLPASS})
    set_tests_properties (${FEXT_NAME} PROPERTIES PASS_REGULAR_EXPRESSION ${FEXT_REGP} FAIL_REGULAR_EXPRESSION ${FEXT_REGF})
else()
    set_tests_properties (${FEXT_NAME} PROPERTIES WILL_FAIL true PASS_REGULAR_EXPRESSION ${FEXT_REGP} FAIL_REGULAR_EXPRESSION ${FEXT_REGF})
endif()
set_property(TEST ${FEXT_NAME} PROPERTY ENVIRONMENT "${FEXT_LD}")
endfunction(add_test_ext)