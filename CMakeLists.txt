cmake_minimum_required(VERSION 3.13)

project(RayTracing LANGUAGES CXX)

set ( HEADER_FILES
    "inc/Common/Common.h"
    "inc/Common/Helpers.h"
    "inc/DX/d3dx12.h"
    "inc/Geometry/Hitable.h"
    "inc/Geometry/HitableList.h"
    "inc/Geometry/HitRecord.h"
    "inc/Geometry/Ray.h"
    "inc/Geometry/Sphere.h"
    "inc/Geometry/Vector3.h"
    "inc/Material/Camera.h"
    "inc/Material/Diffuse.h"
    "inc/Material/Glass.h"
    "inc/Material/Metallic.h"
    "inc/Material/Scatterable.h"
)

set ( SRC_FILES
    "src/Common/Common.cpp"
    "src/Geometry/HitableList.cpp"
    "src/Geometry/Sphere.cpp"
    "src/Geometry/Vector3.cpp"
    "src/Material/Diffuse.cpp"
    "src/Material/Glass.cpp"
    "src/Material/Metallic.cpp"
)

set ( PCH_HEADER_FILE
    "inc/Common/pch.h"
)

# add the executable
add_executable(RayTracing src/RayTracing.cpp
    ${HEADER_FILES}
    ${SRC_FILES}
    ${PCH_HEADER_FILE}
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET RayTracing PROPERTY CXX_STANDARD 20)
endif()

target_include_directories(RayTracing PUBLIC
    "${CMAKE_SOURCE_DIR}/inc/Common"
    "${CMAKE_SOURCE_DIR}/inc/DX"
    "${CMAKE_SOURCE_DIR}/inc/Material"
    "${CMAKE_SOURCE_DIR}/inc/Geometry"
)

# TODO: Add tests and install targets if needed.

target_link_libraries( RayTracing 
    d3d12.lib
    dxgi.lib
    dxguid.lib
)
