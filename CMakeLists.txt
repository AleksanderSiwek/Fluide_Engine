cmake_minimum_required(VERSION 3.2)

set(PROJECT_NAME Fluid_Engine)
set(PROJECT_LIB Fluid_Engine_lib)
set(PROJECT_BINARY Fluid_Engine_binary)
set(PROJECT_TESTS Fluid_Engine_tests)

project(${PROJECT_NAME} C CXX)

include_directories(googletest/googletest/include/gtest/)

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

add_subdirectory(src)
add_subdirectory(test)


