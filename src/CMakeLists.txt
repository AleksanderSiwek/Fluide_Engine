
file(GLOB HEADERS *.hpp ./common/*.hpp ./forces/*.hpp ./grid_systems/*.hpp ./particle_systems/*.hpp ./fluid_solvers/*.hpp ./linear_system/*.hpp)
file(GLOB SOURCES *.cpp ./common/*.cpp ./forces/*.cpp ./grid_systems/*.cpp ./particle_systems/*.cpp ./fluid_solvers/*.cpp ./linear_system/*.cpp)
add_library(PROJECT_LIB ${SOURCES})
add_executable(${PROJECT_BINARY} ${HEADERS} ${SOURCES})