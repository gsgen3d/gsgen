cmake_minimum_required(VERSION 3.3)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif(POLICY CMP0069)

if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif(POLICY CMP0072)

project(gs)

set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)
message(STATUS "CUDA enabled")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -g -Xcudafe \"--display_error_number --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=3060\" -lineinfo -arch=sm_75 ")

# -Xptxas=\"-v\"
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /GLT /Ox")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=\"/MT\"")
endif()

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(gs-test SHARED ${SOURCES})
target_link_libraries(gs-test PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(gs-test PRIVATE "${INCLUDE_DIR}")

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET gs-test
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:example-app>)
endif(MSVC)
