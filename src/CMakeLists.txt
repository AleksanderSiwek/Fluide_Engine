
file(GLOB HEADERS *.hpp)
file(GLOB SOURCES *.cpp)
add_executable(${PROJECT_BINARY} ${HEADERS} ${SOURCES})