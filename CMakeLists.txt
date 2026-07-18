cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_COMPILER "clang++")
project(autograd LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_library(autograd STATIC
        src/tensor_one.cpp
        src/node.cpp
        src/tape/tape_one.cpp
        src/tape/tape_string.cpp
        src/ops/addition.cpp
        src/ops/subtraction.cpp
        src/ops/multiplication.cpp
)

target_include_directories(autograd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(autograd PRIVATE
        src/consts.h
        include/tensor_base.h
        include/tensor_one.h
        include/tape.h
        include/operations.h
)

# ---- gtest ----
include(FetchContent)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(autograd_tests
        tests/1d/basic.cpp
        tests/1d/basic_tape.cpp
        tests/1d/advanced_tape.cpp
        tests/1d/stress_tape.cpp
)

# Belt-and-braces: force C++ link language for tests
set_target_properties(autograd_tests PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(autograd_tests
        PRIVATE
        autograd
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(autograd_tests)