# Copyright © 2023-2024 Apple Inc.

cmake_minimum_required(VERSION 3.16)

# ----------------------------- Setup -----------------------------
project(mlx.data LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

# ----------------------------- Configuration -----------------------------
if(NOT MLX_C_VERSION)
  set(MLX_C_VERSION 0.6.0)
endif()

option(BUILD_SHARED_LIBS "Build mlx C as a shared library" OFF)
option(MLX_C_BUILD_EXAMPLES "Build examples for mlx C" ON)
option(MLX_C_USE_SYSTEM_MLX "Use system MLX" OFF)

# ----------------------------- mlx -----------------------------

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# mlx
set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)

if(MLX_C_USE_SYSTEM_MLX)
  find_package(MLX REQUIRED)
else()
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG v0.32.2)
  FetchContent_MakeAvailable(mlx)
endif()

# lfm3-engine: MLX's CUDA backend compiles kernels at run time (NVRTC) and finds the headers they include (CCCL, and
# CUTLASS's cute for the 8-bit products) next to a Python wheel's library only; a statically linked program needs
# their place built in. MLX takes one such place (MLX_CCCL_DIR), so it is a directory of links to both: the CUDA
# toolkit's CCCL (else the one MLX downloaded) and the CUTLASS MLX downloaded.
if(MLX_BUILD_CUDA AND TARGET mlx_dirs)
  find_package(CUDAToolkit REQUIRED)
  set(MLX_C_CCCL "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  if(NOT EXISTS "${MLX_C_CCCL}/cuda/std/tuple")
    FetchContent_GetProperties(cccl)
    set(MLX_C_CCCL "${cccl_SOURCE_DIR}/include")
  endif()
  FetchContent_GetProperties(cutlass)
  set(MLX_C_JIT_INCLUDE "${CMAKE_BINARY_DIR}/jit-include")
  file(MAKE_DIRECTORY "${MLX_C_JIT_INCLUDE}")
  file(GLOB MLX_C_CCCL_DIRS "${MLX_C_CCCL}/*")
  foreach(dir ${MLX_C_CCCL_DIRS} "${cutlass_SOURCE_DIR}/include/cute" "${cutlass_SOURCE_DIR}/include/cutlass")
    get_filename_component(name "${dir}" NAME)
    file(CREATE_LINK "${dir}" "${MLX_C_JIT_INCLUDE}/${name}" SYMBOLIC)
  endforeach()
  target_compile_definitions(mlx_dirs PRIVATE MLX_CCCL_DIR="${MLX_C_JIT_INCLUDE}")
endif()

# ----------------------------- lib -----------------------------

set(mlxc-src
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/array.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/closure.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/compile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/cuda.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/device.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed_group.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/error.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/fast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/fft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/graph_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/io_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/linalg.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/map.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/metal.cpp
    # ${CMAKE_CURRENT_LIST_DIR}/mlx/c/object.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/random.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/string.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/transforms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/transforms_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/vector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/version.cpp)

add_library(mlxc ${mlxc-src})

target_link_libraries(mlxc PRIVATE mlx)
target_include_directories(mlxc
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
set_property(TARGET mlxc PROPERTY POSITION_INDEPENDENT_CODE ON)

# Windows DLL symbol exports
if(WIN32)
  set_target_properties(mlxc PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

if(MLX_C_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples)
endif()

# ----------------------------- Installation -----------------------------
include(GNUInstallDirs)

# Install library
install(
  TARGETS mlxc
  EXPORT MLXCTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install headers
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/mlx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "private" EXCLUDE)

# Install cmake config
set(MLX_C_CMAKE_BUILD_CONFIG ${CMAKE_BINARY_DIR}/MLXCConfig.cmake)
set(MLX_C_CMAKE_BUILD_VERSION_CONFIG
    ${CMAKE_BINARY_DIR}/MLXCConfigVersion.cmake)
set(MLX_C_CMAKE_INSTALL_MODULE_DIR share/cmake/MLXC)

install(
  EXPORT MLXCTargets
  FILE MLXCTargets.cmake
  DESTINATION ${MLX_C_CMAKE_INSTALL_MODULE_DIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${MLX_C_CMAKE_BUILD_VERSION_CONFIG}
  COMPATIBILITY SameMajorVersion
  VERSION ${MLX_C_VERSION})

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/mlx-c.pc.in ${MLX_C_CMAKE_BUILD_CONFIG}
  INSTALL_DESTINATION ${MLX_C_CMAKE_INSTALL_MODULE_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
            MLX_C_CMAKE_INSTALL_MODULE_DIR)

install(FILES ${MLX_C_CMAKE_BUILD_CONFIG} ${MLX_C_CMAKE_BUILD_VERSION_CONFIG}
        DESTINATION ${MLX_C_CMAKE_INSTALL_MODULE_DIR})
