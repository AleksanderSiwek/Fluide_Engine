
file(GLOB HEADERS *.hpp ./common/*.hpp ./forces/*.hpp ./grid_systems/*.hpp ./particle_systems/*.hpp ./solvers/*.hpp)
file(GLOB SOURCES *.cpp ./common/*.cpp ./forces/*.cpp ./grid_systems/*.cpp ./particle_systems/*.cpp ./solvers/*.cpp)
add_library(PROJECT_LIB ${SOURCES})
add_executable(${PROJECT_BINARY} ${HEADERS} ${SOURCES})