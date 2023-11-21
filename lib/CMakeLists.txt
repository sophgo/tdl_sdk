if(NOT "${CVI_PLATFORM}" STREQUAL "CV180X")
add_subdirectory(cvi_draw_rect)
endif()
add_subdirectory(cvi_md)
add_subdirectory(cvi_preprocess)
add_subdirectory(sample)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample/sample_lib_cvimd.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sample/Makefile DESTINATION sample/cvi_md/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample/sample_lib_draw_rect.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sample/Makefile DESTINATION sample/cvi_draw_rect/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample/sample_lib_preprocess.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sample/Makefile DESTINATION sample/cvi_preprocess/)

