# Copyright 2020 cvitek Inc.
cmake_minimum_required(VERSION 3.18.4)

project(sample)

function(buildninstall)
  # From https://stackoverflow.com/questions/23327687/how-to-write-a-cmake-function-with-more-than-one-parameter-groups
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of the boolean arguments (only defined ones will be true)
    "NAME" # list of names of mono-valued arguments
    "INC;SRCS;DEPS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN} # arguments of the function to parse, here we take the all original ones
  )
  # note: if it remains unparsed arguments, here, they can be found in variable PARSED_ARGS_UNPARSED_ARGUMENTS
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "You must provide a name")
  endif(NOT PARSED_ARGS_NAME)
  project(${PARSED_ARGS_NAME})
  # For debug purpose.
  # message("Provided additional sources are:")
  # foreach(src ${PARSED_ARGS_SRCS})
  #   message("- ${src}")
  # endforeach(src)
  include_directories(${PARSED_ARGS_INC})
  add_executable(${PROJECT_NAME} ${PROJECT_NAME}.c ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS} atomic)
  install(TARGETS ${PROJECT_NAME} DESTINATION bin)
  string(SUBSTRING "${PROJECT_NAME}" 0 10 substring1)
  if (substring1 STREQUAL "sample_app")
      install(FILES ${PROJECT_NAME}.c DESTINATION sample/cvi_tdl_app/)
  else ()
      install(FILES ${PROJECT_NAME}.c DESTINATION sample/cvi_tdl/)
  endif()
endfunction(buildninstall)

# install example
# buildninstall(NAME sample_vi_od INC ${SAMPLE_INC} SRCS middleware_utils.c sample_utils.c ${SAMPLE_LIBS_FILE} DEPS cvi_tdl ${SAMPLE_LIBS} )
