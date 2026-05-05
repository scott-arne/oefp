cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(oefp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(OEFP_BUILD_TESTS "Build tests" ON)
option(OEFP_BUILD_PYTHON "Build Python bindings" ON)
option(OEFP_UNIVERSAL2 "Build universal2 binary for macOS" OFF)
option(OEFP_USE_STABLE_ABI "Use Python stable ABI" ON)

if(APPLE AND OEFP_UNIVERSAL2)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures" FORCE)
elseif(APPLE)
    unset(CMAKE_OSX_ARCHITECTURES CACHE)
endif()

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

# Dependencies
include(FetchContent)

FetchContent_Declare(
    cmake_openeye
    GIT_REPOSITORY https://github.com/scott-arne/cmake-openeye.git
    GIT_TAG v1.2.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cmake_openeye)
list(APPEND CMAKE_MODULE_PATH "${cmake_openeye_SOURCE_DIR}")

if(OEFP_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(SWIG 4.0)
endif()

if(OPENEYE_USE_SHARED)
    find_package(OpenEyePython REQUIRED)
endif()

find_package(OpenEye REQUIRED)

# Library
set(OEFP_SOURCES
    src/batch.cpp
    src/compare.cpp
    src/fingerprint.cpp
    src/metric.cpp
    src/oefp.cpp
    src/openeye.cpp
)

add_library(oefp ${OEFP_SOURCES})

set_target_properties(oefp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(oefp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(oefp
    PUBLIC
        OpenEye::OEChem
        OpenEye::OEGraphSim
)

# Tests
if(OEFP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Python bindings
if(OEFP_BUILD_PYTHON AND SWIG_FOUND)
    add_subdirectory(swig)
endif()

# Install
install(TARGETS oefp
    EXPORT oefp-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/oefp DESTINATION include)

install(EXPORT oefp-targets
    FILE OEFPTargets.cmake
    NAMESPACE OEFP::
    DESTINATION lib/cmake/oefp
)
