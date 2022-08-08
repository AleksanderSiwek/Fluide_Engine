
file(GLOB HEADERS *.hpp 
        ./common/*.hpp 
        ./forces/*.hpp 
        ./grid_systems/*.hpp 
        ./particle_systems/*.hpp 
        ./fluid_solvers/*.hpp 
        ./linear_system/*.hpp 
        ./file_systems/*.hpp
        ./3d/*hpp)

file(GLOB SOURCES *.cpp 
        ./common/*.cpp 
        ./forces/*.cpp 
        ./grid_systems/*.cpp 
        ./particle_systems/*.cpp 
        ./fluid_solvers/*.cpp 
        ./linear_system/*.cpp 
        ./file_systems/*.cpp
        ./3d/*cpp)

file(GLOB CUDA_SOURCES *.cu 
        ./common/*.cu 
        ./forces/*.cu 
        ./grid_systems/*.cu 
        ./particle_systems/*.cu 
        ./fluid_solvers/*.cu 
        ./linear_system/*.cu 
        ./file_systems/*.cu
        ./3d/*cu)

add_library(PROJECT_LIB STATIC ${SOURCES} ${CUDA_SOURCES})
# target_compile_features(PROJECT_LIB PUBLIC cxx_std_11)
set_target_properties(PROJECT_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PROJECT_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_executable(${PROJECT_BINARY} ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
set_target_properties(${PROJECT_BINARY}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(${PROJECT_BINARY} PROPERTIES LINK_OPTIONS -nodefaultlibs)

add_subdirectory(python_wrapper)
# set_target_properties(${PROJECT_BINARY} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
