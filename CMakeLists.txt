cmake_minimum_required(VERSION 3.13)

project(RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set ( HEADER_FILES
    "inc/Geometry/Hitable.cuh"
    "inc/Geometry/HitableList.cuh"
    "inc/Geometry/HitRecord.cuh"
    "inc/Geometry/Ray.cuh"
    "inc/Geometry/Sphere.cuh"
    "inc/Geometry/Triangle.cuh"
    "inc/Geometry/Vector3.cuh"
    "inc/Material/Camera.cuh"
    "inc/Material/Diffuse.cuh"
    "inc/Material/Glass.cuh"
    "inc/Material/Metallic.cuh"
    "inc/Material/Scatterable.cuh"
)

set ( SRC_FILES
    "src/Geometry/HitableList.cu"
    "src/Geometry/Sphere.cu"
    "src/Geometry/Triangle.cu"
    "src/Geometry/Vector3.cu"
    "src/Material/Diffuse.cu"
    "src/Material/Glass.cu"
    "src/Material/Metallic.cu"
    "src/RayTracing.cu"
)

# add the executable
add_executable(RayTracing src/RayTracing.cu
    ${SRC_FILES}
    ${HEADER_FILES}
)

set_target_properties( RayTracing 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(RayTracing PROPERTIES CUDA_ARCHITECTURES 61)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g; -G; -gencode arch=compute_61,code=sm_61;)

target_include_directories(RayTracing PUBLIC
    "${CMAKE_SOURCE_DIR}/inc/"
    "${CMAKE_SOURCE_DIR}/inc/Geometry"
    "${CMAKE_SOURCE_DIR}/inc/Material"
)

# TODO: Add tests and install targets if needed.
