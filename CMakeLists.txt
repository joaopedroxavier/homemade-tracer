cmake_minimum_required(VERSION 3.13)

project(RayTracing)

add_subdirectory(src/Geometry)
add_subdirectory(src/Material)
add_subdirectory(src/Common)

# add the executable
add_executable(RayTracing src/RayTracing.cpp)

target_link_libraries(RayTracing PRIVATE 
    Geometry
    Material
    Common
)