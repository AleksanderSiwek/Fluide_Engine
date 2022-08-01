
enable_testing()

file(GLOB TEST_SRCS *.cpp *.cu)
add_executable(${PROJECT_TESTS} ${TEST_SRCS})
set_target_properties(${PROJECT_TESTS}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(${PROJECT_TESTS} PROPERTIES LINK_OPTIONS -nodefaultlibs)
target_link_libraries(${PROJECT_TESTS} gtest_main PROJECT_LIB)

include(GoogleTest)
gtest_discover_tests(${PROJECT_TESTS})