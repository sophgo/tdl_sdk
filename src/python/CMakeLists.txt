project(tdl_python)


include_directories(    
                    ${REPO_DIR}/include/framework
                    ${REPO_DIR}/include/nn
                    ${REPO_DIR}/include/components/matcher
                    ${REPO_DIR}/include/components/tracker
                    ${REPO_DIR}/src/components/llm/qwen
                    ${REPO_DIR}/src/components/llm/qwen2VL
                    ${REPO_DIR}/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends/include/python
                    ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends/include/pybind11
                    )

file(GLOB_RECURSE SRC_FRAMWORK_FILES_CUR ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# 添加tdl_image模块的源码
add_library(${PROJECT_NAME} MODULE
    ${SRC_FRAMWORK_FILES_CUR}
)


file(GLOB_RECURSE LIB_PYTHON_FILES ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends/lib/*.so)
foreach(PYTHON_LIB ${LIB_PYTHON_FILES})
    if(PYTHON_LIB MATCHES "libpython([0-9]+\\.[0-9]+)\\.so")
        set(PYTHON_VERSION_SUFFIX "python${CMAKE_MATCH_1}")
        break()
    endif()
endforeach()
message(STATUS "LIB_PYTHON_FILES: ${LIB_PYTHON_FILES}")
target_link_libraries(${PROJECT_NAME} PRIVATE
    # pybind11::module
    ${LIB_PYTHON_FILES}
    tdl_core
    tdl_ex
)

# 设置目标文件名
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "tdl"
)

# 设置编译优化参数（可选）
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -fPIC)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)