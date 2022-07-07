set(HEADERS include/Common.h
)

add_library(Geometry STATIC ${HEADERS})

target_include_directories(Common PUBLIC
  "${PROJECT_SOURCE_DIR}/src/Common/include"
)