cmake_minimum_required(VERSION 3.3)
project(CxxEnumeratePlatformsDevices
  VERSION "1.0.0.0"
  LANGUAGES CXX)

set(BUILD_TARGET ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD ${LATEST_CXX_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_INCLUDE_CURRENT_DIR ON)


find_package(OpenCL REQUIRED)

include(../cmake/GenerateCLHppWrapperHeader.cmake)
set(CLHPP_WRAPPER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/config/opencl.hpp")
generate_clhpp_wrapper_header(
  ${CLHPP_WRAPPER_HEADER}
  ENABLE_EXCEPTIONS ON
  MINIMUM_OPENCL_VERSION 200
  TARGET_OPENCL_VERSION 200)

file(GLOB SRCS *.c *.cpp *.cxx *.cc *.h *.hpp *.hxx *.hh *.inl)
add_executable(
  ${BUILD_TARGET}
  ${SRCS}
  ${CLHPP_WRAPPER_HEADER})

target_include_directories(${BUILD_TARGET} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${BUILD_TARGET} PRIVATE ${OpenCL_LIBRARIES})

ExternalProject_Get_Property(OpenCL-CLHPP SOURCE_DIR)
target_include_directories(${BUILD_TARGET} PRIVATE "${SOURCE_DIR}/include")
add_dependencies(${BUILD_TARGET} OpenCL-CLHPP)


target_compile_definitions(
  ${BUILD_TARGET} PRIVATE
  ${DEFINES}
  $<$<CONFIG:Release>:${DEFINES_RELEASE}>
  $<$<CONFIG:Debug>:${DEFINES_DEBUG}>
  $<$<CONFIG:RelWithDebInfo>:${DEFINES_RELWITHDEBINFO}>
  $<$<CONFIG:MinSizeRel>:${DEFINES_MINSIZEREL}>)


get_property(PROJECT_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

target_compile_options(
  ${BUILD_TARGET} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    ${CXX_FLAGS}
    $<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>
    $<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>
    $<$<CONFIG:RelWithDebInfo>:${CXX_FLAGS_RELWITHDEBINFO}>
    $<$<CONFIG:MinSizeRel>:${CXX_FLAGS_MINSIZEREL}>
  >)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
  target_link_options(
    ${BUILD_TARGET} PRIVATE
    ${EXE_LINKER_FLAGS}
    $<$<CONFIG:Release>:${EXE_LINKER_FLAGS_RELEASE}>
    $<$<CONFIG:Debug>:${EXE_LINKER_FLAGS_DEBUG}>
    $<$<CONFIG:RelWithDebInfo>:${EXE_LINKER_FLAGS_RELWITHDEBINFO}>
    $<$<CONFIG:MinSizeRel>:${EXE_LINKER_FLAGS_MINSIZEREL}>)
else()
  foreach(TARGET_FLAG
      EXE_LINKER_FLAGS
      EXE_LINKER_FLAGS_DEBUG
      EXE_LINKER_FLAGS_RELEASE
      EXE_LINKER_FLAGS_RELWITHDEBINFO
      EXE_LINKER_FLAGS_MINSIZEREL)
    string(REPLACE ";" " " ${TARGET_FLAG} "${${TARGET_FLAG}}")
    string(REGEX REPLACE "  +" " " "CMAKE_${TARGET_FLAG}" "${${TARGET_FLAG}}")
  endforeach(TARGET_FLAG)
endif()
