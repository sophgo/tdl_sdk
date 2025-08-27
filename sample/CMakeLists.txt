add_subdirectory(c)
add_subdirectory(cpp)

# 处理Python示例文件
if(EXISTS ${REPO_DIR}/dependency/${CVI_PLATFORM}/python_depends)
    # 在当前目录找到所有Python文件
    file(GLOB PY_SAMPLE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/*.py")

    # 安装所有Python示例文件
    install(FILES ${PY_SAMPLE_FILES} DESTINATION python)

    message(STATUS "Python samples will be installed to ${CMAKE_INSTALL_PREFIX}/python")
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config DESTINATION .)