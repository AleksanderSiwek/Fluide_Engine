cmake_minimum_required(VERSION 3.7)

set(PROJECT_NAME Fluid_Engine)
set(PROJECT_LIB Fluid_Engine_lib)
set(PROJECT_BINARY Fluid_Engine_binary)
set(PROJECT_TESTS Fluid_Engine_tests)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION ASC)

project(${PROJECT_NAME} LANGUAGES C CXX CUDA)

# CUDA configuration
enable_language(CUDA)
#find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_22,code=sm_22 -rdc=true -lcudart -lcudadevrt -lcuda)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON )

# python
find_package(PythonInterp 3.7.7 EXACT REQUIRED)
find_package(PythonLibs 3.7.7 EXACT REQUIRED)

include_directories(googletest/googletest/include/gtest/)
include_directories(external/pybind11/include)

message(STATUS "CMake version   : " ${CMAKE_SYSTEM_VERSION})
message(STATUS "Compiler        : " ${CMAKE_CXX_COMPILER})
message(STATUS "Operating System: " ${CMAKE_SYSTEM})

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(libs/pybind11/)
if (BUILD_FROM_PIP)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

add_subdirectory(src)
add_subdirectory(test)




