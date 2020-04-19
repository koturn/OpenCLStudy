include(CMakeParseArguments)

set(CL_HPP_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(generate_clhpp_wrapper_header header_file_path)
  set(options)
  set(oneValueArgs
    HEADER_VERSION
    MINIMUM_OPENCL_VERSION
    TARGET_OPENCL_VERSION
    ENABLE_EXCEPTIONS
    NO_STD_STRING
    NO_STD_VECTOR
    NO_STD_ARRAY
    NO_STD_UNIQUE_PTR
    ENABLE_SIZE_T_COMPATIBILITY
    ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
    CL_1_2_DEFAULT_BUILD
    USE_CL_DEVICE_FISSION
    USE_CL_SUB_GROUPS_KHR
    USE_CL_IMAGE2D_FROM_BUFFER_KHR
    USE_DX_INTEROP
    USE_IL_KHR
    USER_OVERRIDE_ERROR_STRINGS
    SILENCE_DEPRECATION)
  cmake_parse_arguments(CL_HPP "${options}" "${oneValueArgs}" "" ${ARGN})

  if(NOT DEFINED CL_HPP_HEADER_VERSION)
    set(CL_HPP_HEADER_VERSION "2")
  endif()

  if(CL_HPP_SILENCE_DEPRECATION)
    set(CL_SILENCE_DEPRECATION ON)
  endif()

  get_filename_component(CL_HPP_INCLUDE_GUARD_MACRO "${header_file_path}" NAME)
  string(REGEX REPLACE "[^0-9A-Za-z_]" "_" CL_HPP_INCLUDE_GUARD_MACRO "${CL_HPP_INCLUDE_GUARD_MACRO}")
  string(TOUPPER ${CL_HPP_INCLUDE_GUARD_MACRO} CL_HPP_INCLUDE_GUARD_MACRO)

  if(CL_HPP_HEADER_VERSION EQUAL 1)
    if(CL_HPP_ENABLE_EXCEPTIONS)
      set(__CL_ENABLE_EXCEPTIONS ON)
    endif()
    if(CL_HPP_MINIMUM_OPENCL_VERSION LESS_EQUAL 100)
      set(CL_USE_DEPRECATED_OPENCL_1_0_APIS ON)
    endif()
    if(CL_HPP_MINIMUM_OPENCL_VERSION LESS_EQUAL 110)
      set(CL_USE_DEPRECATED_OPENCL_1_1_APIS ON)
    endif()
    if(CL_HPP_MINIMUM_OPENCL_VERSION LESS_EQUAL 120)
      set(CL_USE_DEPRECATED_OPENCL_1_2_APIS ON)
    endif()
    if(CL_HPP_MINIMUM_OPENCL_VERSION LESS_EQUAL 200)
      set(CL_USE_DEPRECATED_OPENCL_2_0_APIS ON)
    endif()
    if(CL_HPP_MINIMUM_OPENCL_VERSION LESS_EQUAL 210)
      set(CL_USE_DEPRECATED_OPENCL_2_1_APIS ON)
    endif()
    if(CL_HPP_MINIMUM_OPENCL_VERSION LESS_EQUAL 220)
      set(CL_USE_DEPRECATED_OPENCL_2_2_APIS ON)
    endif()
    if(CL_HPP_NO_STD_STRING)
      set(__NO_STD_STRING ON)
    endif()
    if(CL_HPP_NO_STD_VECTOR)
      set(__NO_STD_VECTOR ON)
    endif()
    if(CL_HPP_USER_OVERRIDE_ERROR_STRINGS)
      set(__CL_USER_OVERRIDE_ERROR_STRINGS ON)
    endif()
    if(CL_HPP_USE_CL_DEVICE_FISSION)
      set(USE_CL_DEVICE_FISSION ON)
    endif()
    if(CL_HPP_USE_DX_INTEROP)
      set(USE_DX_INTEROP ON)
    endif()

    configure_file(
      ${CL_HPP_CURRENT_LIST_DIR}/templates/clWrapper.hpp.in
      ${header_file_path}
      @ONLY)
  elseif(CL_HPP_HEADER_VERSION EQUAL 2)
    configure_file(
      ${CL_HPP_CURRENT_LIST_DIR}/templates/cl2Wrapper.hpp.in
      ${header_file_path}
      @ONLY)
  else()
    message(FATAL_ERROR "Specified HEADER_VERSION is ${CL_HPP_HEADER_VERSION}, but it must be 1 or 2")
  endif()

  message(STATUS "Configure done. Output file: ${header_file_path}")
endfunction()
