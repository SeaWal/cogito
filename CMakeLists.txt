cmake_minimum_required(VERSION 3.20)

project(cogito VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Add source files
set(LINALG_SOURCES
    src/linalg/Matrix.cpp
    src/linalg/Operations.cpp
)

# Combine all source files
set(SOURCES
    ${LINALG_SOURCES}
)

set(MAIN_SOURCES
    src/main.cpp
)

# Create the static library
add_library(cogito STATIC ${SOURCES})
# Include directories
target_include_directories(cogito PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Create test executables
#add_executable(test_matrix tests/test_matrix.cpp)
#target_link_libraries(test_matrix cogito)

# Create the main executable
add_executable(main_executable ${MAIN_SOURCES})
target_link_libraries(main_executable cogito)
