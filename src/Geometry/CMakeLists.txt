set(SOURCES src/HitableList.cpp 
            src/Sphere.cpp
)

set(HEADERS include/Hitable.h 
            include/HitableList.h 
	    include/HitRecord.h
            include/Ray.h 
            include/Sphere.h
	    include/Vector3.h
)

add_library(Geometry STATIC ${SOURCES} ${HEADERS})

target_include_directories(Geometry PUBLIC
  "${PROJECT_SOURCE_DIR}/src/Geometry/include"
)