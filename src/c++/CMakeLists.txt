# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

# project(cc-clients LANGUAGES C CXX)

# #
# # Options
# #
# option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" OFF)
# option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
# option(TRITON_ENABLE_PERF_ANALYZER "Enable Performance Analyzer" OFF)
# option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
# option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
# option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)
# option(TRITON_USE_THIRD_PARTY "Use local version of third party libraries" ON)
# option(TRITON_KEEP_TYPEINFO "Keep typeinfo symbols by disabling ldscript" OFF)

# set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
# set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# #
# # Dependencies
# #
# include(FetchContent)

# FetchContent_Declare(
#   repo-common
#   GIT_REPOSITORY https://github.com/triton-inference-server/common.git
#   GIT_TAG ${TRITON_COMMON_REPO_TAG}
#   GIT_SHALLOW ON
# )

# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/9406a60c7839052e4944ea4dbc8344762a89f9bd.zip
# )

# if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
#   set(TRITON_COMMON_ENABLE_PROTOBUF ON)
#   set(TRITON_COMMON_ENABLE_GRPC ON)

#   if(TRITON_ENABLE_PERF_ANALYZER)
#     FetchContent_Declare(
#       repo-core
#       GIT_REPOSITORY https://github.com/triton-inference-server/core.git
#       GIT_TAG ${TRITON_CORE_REPO_TAG}
#       GIT_SHALLOW ON 
#     )
#     FetchContent_MakeAvailable(repo-core)
#   endif() # TRITON_ENABLE_PERF_ANALYZER
# endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

# if(NOT TRITON_ENABLE_PERF_ANALYZER AND NOT TRITON_ENABLE_CC_HTTP AND NOT TRITON_ENABLE_EXAMPLES)
#   set(TRITON_COMMON_ENABLE_JSON OFF)
# endif()

# if(TRITON_ENABLE_TESTS OR TRITON_ENABLE_PERF_ANALYZER)
#   FetchContent_MakeAvailable(googletest)
# endif()
# FetchContent_MakeAvailable(repo-common)

# #
# # CUDA
# #
# if(TRITON_ENABLE_GPU)
#   find_package(CUDAToolkit REQUIRED)
# endif() # TRITON_ENABLE_GPU

# #
# # libcurl
# #
# if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER)
#   find_package(CURL REQUIRED)
#   message(STATUS "Using curl ${CURL_VERSION}")
# endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER

# #
# # Protobuf
# #
# if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
#   set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
#   find_package(Protobuf CONFIG REQUIRED)
#   message(STATUS "Using protobuf ${Protobuf_VERSION}")
#   include_directories(${Protobuf_INCLUDE_DIRS})
# endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

# #
# # GRPC
# #
# if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
#   find_package(gRPC CONFIG REQUIRED)
#   message(STATUS "Using gRPC ${gRPC_VERSION}")
#   include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)
# endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

# if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
#   add_subdirectory(library)
# endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

# if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC)
#   if(TRITON_ENABLE_EXAMPLES)
#     add_subdirectory(examples)
#   endif() # TRITON_ENABLE_EXAMPLES

#   if(TRITON_ENABLE_TESTS)
#     add_subdirectory(tests)
#   endif() # TRITON_ENABLE_TESTS
# endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC

# if(TRITON_ENABLE_PERF_ANALYZER)
  # add_subdirectory(perf_analyzer)
# endif() # TRITON_ENABLE_PERF_ANALYZER

set(TRITON_THIRD_PARTY_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third-party)

include(ExternalProject)
ExternalProject_Add(curl
  PREFIX curl
  GIT_REPOSITORY "https://github.com/curl/curl.git"
  GIT_TAG "curl-7_86_0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/curl/src/curl"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DBUILD_TESTING:BOOL=OFF
    -DBUILD_CURL_EXE:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DCURL_STATICLIB:BOOL=ON
    -DHTTP_ONLY:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl
    ${LIBCURL_EXTRA_BUILD_ARGS}
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)
find_package(CURL REQUIRED)

ExternalProject_Add(grpc
  PREFIX grpc
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.48.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  EXCLUDE_FROM_ALL ON
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG} --dest-basename=grpc_1.48.0
  CMAKE_CACHE_ARGS
  -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc
)
message(STATUS "Using gRPC ${gRPC_VERSION}")
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)



include(FetchContent)
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Repos branch")
FetchContent_Declare(common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON 
)
FetchContent_Populate(common)



set(TRITON_CLIENT_REPO_TAG "main" CACHE STRING "Repos branch")
FetchContent_Declare(client
  GIT_REPOSITORY https://github.com/triton-inference-server/client.git
  GIT_TAG ${TRITON_CLIENT_REPO_TAG}
  GIT_SHALLOW ON 
)
FetchContent_Populate(client)


find_package(RapidJSON CONFIG REQUIRED)

add_library(json-utils-library 
  EXCLUDE_FROM_ALL OBJECT
  ${client_SOURCE_DIR}/src/c++/library/json_utils.h 
  ${client_SOURCE_DIR}/src/c++/library/json_utils.cc
)
target_include_directories(json-utils-library
  PUBLIC 
  ${RapidJSON_INCLUDE_DIRS}
)
add_library(shm-utils-library 
  EXCLUDE_FROM_ALL OBJECT
  ${client_SOURCE_DIR}/src/c++/library/shm_utils.h 
  ${client_SOURCE_DIR}/src/c++/library/shm_utils.cc
)

include_directories(
  ${client_SOURCE_DIR}/src/c++/library
)




add_subdirectory(perf_analyzer)