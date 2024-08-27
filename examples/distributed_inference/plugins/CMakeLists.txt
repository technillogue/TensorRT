#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#



set(PLUGIN_TARGET_NAME nvinfer_plugin_tensorrt_llm)
set(PLUGIN_SHARED_TARGET ${PLUGIN_TARGET_NAME})

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map) # Linux
set(PLUGIN_EXPORT_DEF ${TARGET_DIR}/exports.def) # Windows


set(ENABLE_MULTI_DEVICE 1)

add_compile_definitions(ENABLE_MULTI_DEVICE=1)
add_compile_definitions(ENABLE_BF16=1)
add_compile_definitions(ENABLE_FP8=1)

find_package(CUDA ${CUDA_VERSION} REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)

find_package(MPI REQUIRED)
message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
message(STATUS "Using MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")

if(ENABLE_MULTI_DEVICE EQUAL 1)
  # NCCL dependencies
  set(NCCL_LIB_DIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/)
  set(NCCL_INCLUDE_DIR /usr/include/)
  find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})
endif()

message(STATUS "Using NCCL_LIB: ${NCCL_LIB}")

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

link_directories(${TENSORRT_LIB_DIR})

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
  include_directories(${PLUGIN_ITER})
  add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)


include_directories(ncclPlugin)
add_subdirectory(ncclPlugin)

include_directories(common)
add_subdirectory(common)

include_directories(plugins_common)
add_subdirectory(plugins_common)

include_directories(${CMAKE_SOURCE_DIR})

include_directories(${CUDA_INSTALL_DIR}/include)

if (NOT DEFINED TENSORRT_INCLUDE_DIR)
    set(TENSORRT_INCLUDE_DIR ${TRT_INCLUDE_DIR})
endif()

include_directories(${TENSORRT_INCLUDE_DIR})

# Set gencodes
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/api/tllmPlugin.cpp")

# ################################# SHARED LIBRARY
# ##############################################################################

if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif()

add_library(${PLUGIN_SHARED_TARGET} SHARED ${PLUGIN_SOURCES})

message(STATUS "Using TRT_INCLUDE_DIR: ${TRT_INCLUDE_DIR}")

target_include_directories(
  ${PLUGIN_SHARED_TARGET}
  PUBLIC ${CUDA_INSTALL_DIR}/include ${MPI_C_INCLUDE_DIRS} ${TRT_INCLUDE_DIR}
  PRIVATE ${TARGET_DIR})

if(CUDA_VERSION VERSION_LESS 11.0)
  target_include_directories(${PLUGIN_SHARED_TARGET} PUBLIC ${CUB_ROOT_DIR})
endif()

set_target_properties(
  ${PLUGIN_SHARED_TARGET}
  PROPERTIES CXX_STANDARD "17"
             CXX_STANDARD_REQUIRED "YES"
             CXX_EXTENSIONS "NO"
             ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

if(WIN32)
  set_target_properties(
    ${PLUGIN_SHARED_TARGET}
    PROPERTIES LINK_FLAGS "/DEF:${PLUGIN_EXPORT_DEF} ${UNDEFINED_FLAG}")
else()
  set_target_properties(
    ${PLUGIN_SHARED_TARGET}
    PROPERTIES
      LINK_FLAGS
      "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,-rpath,'$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()

set_target_properties(
  ${PLUGIN_SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION
                                                            ${TRT_SOVERSION})

set_property(TARGET ${PLUGIN_SHARED_TARGET} PROPERTY CUDA_STANDARD 17)

target_link_libraries(
  ${PLUGIN_SHARED_TARGET}
  ${CUBLAS_LIB}
  ${CUBLASLT_LIB}
  nvinfer
  ${CUDA_LIBRARIES}
  ${CUDA_DRV_LIB}
  ${CUDA_RT_LIB}
  ${CMAKE_DL_LIBS}
  ${MPI_C_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  ${NCCL_LIB}
  cublas
  cublasLt
  ${SHARED_TARGET})
