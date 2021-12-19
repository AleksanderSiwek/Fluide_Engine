
enable_testing()

file(GLOB TEST_SRCS *.cpp)
add_executable(${PROJECT_TESTS} ${TEST_SRCS})
target_link_libraries(${PROJECT_TESTS} gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_TESTS})