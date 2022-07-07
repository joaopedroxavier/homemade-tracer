set(SOURCES src/Diffuse.cpp)

set(HEADERS include/Diffuse.h 
            include/Scatterable.h 
            include/Camera.h
)

add_library(Material STATIC ${SOURCES} ${HEADERS})

target_link_libraries(Material PRIVATE Geometry Common)

target_include_directories(Material PUBLIC
  "${PROJECT_SOURCE_DIR}/src/Material/include"
)