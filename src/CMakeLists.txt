
file(GLOB HEADERS *.hpp)
file(GLOB SOURCES *.cpp)
add_library(PROJECT_LIB ${SOURCES})
add_executable(${PROJECT_BINARY} ${HEADERS} ${SOURCES})