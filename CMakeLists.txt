cmake_minimum_required(VERSION 3.20)

project(cogito VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Enable debugging symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Treat warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

# Add source files
set(LINALG_SOURCES
    src/linalg/Matrix.cpp
    src/linalg/Operations.cpp
)

set(NN_SOURCES
    src/nn/Dense.cpp
)

# Combine all source files
set(SOURCES
    ${LINALG_SOURCES}
    ${NN_SOURCES}
)

set(MAIN_SOURCES
    src/main.cpp
)

# Create the static library
add_library(cogito STATIC ${SOURCES})
# Include directories
target_include_directories(cogito PUBLIC ${PROJECT_SOURCE_DIR}/include)


# Create the main executable
add_executable(main_executable ${MAIN_SOURCES})
target_link_libraries(main_executable cogito)

# Include tests
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

enable_testing()
file(GLOB TEST_SOURCES tests/*.cpp)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests cogito GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)