set(SOURCES src/Common.cpp
)

set(HEADERS include/Common.h
)

add_library(Common STATIC ${HEADERS} ${SOURCES})

target_include_directories(Common PUBLIC
  "${PROJECT_SOURCE_DIR}/src/Common/include"
)