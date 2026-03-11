project(nn)

include_directories(
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/components
                    ${REPO_DIR}/include/nn
                    ${CMAKE_CURRENT_SOURCE_DIR}/
                    ${FBANK_INCLUDES}
                    )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


foreach(src ${SRC_FRAMWORK_FILES_CUR})
    get_filename_component(fname "${src}" NAME)
    get_filename_component(basename "${src}" NAME_WE)

    string(REPLACE "/" ";" _parts "${src}")
    list(LENGTH _parts parts_count)    
    math(EXPR second_last_index "${parts_count} - 2")
    list(GET _parts ${second_last_index} second_last_part)
    string(TOUPPER "${second_last_part}" dir_var)
    set(dir_var "ENABLE_${dir_var}")

    if(DEFINED ${dir_var} AND NOT ${dir_var})
        list(REMOVE_ITEM SRC_FRAMWORK_FILES_CUR ${src})
    endif()
endforeach()

message(STATUS "SRC_FRAMWORK_FILES_CUR: ${SRC_FRAMWORK_FILES_CUR}")
add_library(${PROJECT_NAME} OBJECT ${SRC_FRAMWORK_FILES_CUR})
