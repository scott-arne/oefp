cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(oefp VERSION 0.2.8 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(OEFP_BUILD_TESTS "Build tests" ON)
option(OEFP_BUILD_PYTHON "Build Python bindings" ON)
option(OEFP_BUILD_BENCHMARKS "Build OEFP benchmarks" OFF)
option(OEFP_UNIVERSAL2 "Build universal2 binary for macOS" OFF)
option(OEFP_USE_STABLE_ABI "Use Python stable ABI" ON)

set(OEFP_OECLUSTER_SOURCE_DIR "" CACHE PATH "Path to local oecluster source tree")

if(APPLE AND OEFP_UNIVERSAL2)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures" FORCE)
elseif(APPLE)
    unset(CMAKE_OSX_ARCHITECTURES CACHE)
endif()

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)

FetchContent_Declare(
    cmake_openeye
    GIT_REPOSITORY https://github.com/scott-arne/cmake-openeye.git
    GIT_TAG v1.2.1
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

set(_OEFP_OPENEYE_SEARCH_PATHS
    ${OPENEYE_LIB_DIR}
    ${OPENEYE_RUNTIME_LIB_DIR}
    ${OPENEYE_ROOT}/lib
    ${OE_DIR}/lib
    $ENV{OPENEYE_ROOT}/lib
    $ENV{OE_DIR}/lib
)
if(OPENEYE_INCLUDE_DIR)
    get_filename_component(_OEFP_OPENEYE_DEFAULT_LIB_DIR "${OPENEYE_INCLUDE_DIR}/../lib" ABSOLUTE)
    list(APPEND _OEFP_OPENEYE_SEARCH_PATHS "${_OEFP_OPENEYE_DEFAULT_LIB_DIR}")
endif()

function(_oefp_find_openeye_library out_var library_name)
    set(_OEFP_LIBRARY "")
    if(OPENEYE_USE_SHARED)
        foreach(_OEFP_SEARCH_PATH IN LISTS _OEFP_OPENEYE_SEARCH_PATHS)
            if(_OEFP_SEARCH_PATH)
                file(GLOB _OEFP_VERSIONED_LIBRARY
                    "${_OEFP_SEARCH_PATH}/lib${library_name}-*.dylib"
                    "${_OEFP_SEARCH_PATH}/lib${library_name}-*.so"
                )
                if(_OEFP_VERSIONED_LIBRARY)
                    list(GET _OEFP_VERSIONED_LIBRARY 0 _OEFP_LIBRARY)
                    break()
                endif()
            endif()
        endforeach()
    endif()
    if(NOT _OEFP_LIBRARY)
        unset(_OEFP_FOUND_LIBRARY CACHE)
        find_library(_OEFP_FOUND_LIBRARY
            NAMES ${library_name}
            PATHS ${_OEFP_OPENEYE_SEARCH_PATHS}
            NO_DEFAULT_PATH
        )
        set(_OEFP_LIBRARY "${_OEFP_FOUND_LIBRARY}")
    endif()
    set(${out_var} "${_OEFP_LIBRARY}" PARENT_SCOPE)
endfunction()

if(NOT TARGET OpenEye::OEMolProp)
    if(OPENEYE_USE_SHARED)
        foreach(_OEFP_OEMOLPROP_SEARCH_PATH IN LISTS _OEFP_OPENEYE_SEARCH_PATHS)
            if(_OEFP_OEMOLPROP_SEARCH_PATH)
                file(GLOB _OEFP_VERSIONED_OEMOLPROP
                    "${_OEFP_OEMOLPROP_SEARCH_PATH}/liboemolprop-*.dylib"
                    "${_OEFP_OEMOLPROP_SEARCH_PATH}/liboemolprop-*.so"
                )
                if(_OEFP_VERSIONED_OEMOLPROP)
                    list(GET _OEFP_VERSIONED_OEMOLPROP 0 OEMOLPROP_LIBRARY)
                    break()
                endif()
            endif()
        endforeach()
    endif()
    if(NOT OEMOLPROP_LIBRARY)
        find_library(OEMOLPROP_LIBRARY
            NAMES oemolprop
            PATHS ${_OEFP_OPENEYE_SEARCH_PATHS}
            NO_DEFAULT_PATH
        )
    endif()
    if(NOT OEMOLPROP_LIBRARY)
        message(FATAL_ERROR "OpenEye OEMolProp library is required for Mordred-compatible descriptors.")
    endif()
    add_library(OpenEye::OEMolProp UNKNOWN IMPORTED)
    set_target_properties(OpenEye::OEMolProp PROPERTIES
        IMPORTED_LOCATION "${OEMOLPROP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENEYE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "OpenEye::OEChem"
    )
endif()

find_package(Arrow CONFIG QUIET)
find_package(Parquet CONFIG QUIET)

if(TARGET Arrow::arrow_shared AND TARGET Parquet::parquet_shared)
    set(OEFP_ARROW_TARGET Arrow::arrow_shared)
    set(OEFP_PARQUET_TARGET Parquet::parquet_shared)
else()
    find_package(ArrowPython REQUIRED)
    set(OEFP_ARROW_TARGET ArrowPython::Arrow)
    set(OEFP_PARQUET_TARGET ArrowPython::Parquet)
endif()

# Library
set(OEFP_SOURCES
    src/annotation.cpp
    src/atom_descriptor.cpp
    src/atom_pair.cpp
    src/batch.cpp
    src/column_request.cpp
    src/compare.cpp
    src/compute_context.cpp
    src/count.cpp
    src/count_batch.cpp
    src/descriptor.cpp
    src/descriptor_arrow.cpp
    src/descriptor_batch.cpp
    src/descriptor_calculator.cpp
    src/descriptor_schema.cpp
    src/descriptor_selection.cpp
    src/descriptor_source.cpp
    src/descriptor_value.cpp
    src/fingerprint.cpp
    src/input_normalization.cpp
    src/kallisto_data.cpp
    src/kallisto_descriptors.cpp
    src/kallisto_geometry.cpp
    src/metric.cpp
    src/molecular_properties.cpp
    src/mordred.cpp
    src/mordred_schema.cpp
    src/morgan.cpp
    src/openeye.cpp
    src/rdkit_descriptors.cpp
    src/rdkit_schema.cpp
    src/rdkit_stereogenicity.cpp
    src/sparse.cpp
    src/sparse_batch.cpp
    src/stereo.cpp
    src/topological_torsions.cpp
)

add_library(oefp ${OEFP_SOURCES})

set_target_properties(oefp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(oefp PRIVATE cxx_std_20)

target_include_directories(oefp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(oefp
    PUBLIC
        OpenEye::OEChem
        OpenEye::OEGraphSim
        OpenEye::OEMolProp
        ${OEFP_ARROW_TARGET}
        ${OEFP_PARQUET_TARGET}
)

if(TARGET OpenEye::OEGrid)
    target_link_libraries(oefp PUBLIC OpenEye::OEGrid)
endif()

# Tests
if(OEFP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Python bindings
if(OEFP_BUILD_PYTHON AND SWIG_FOUND)
    add_subdirectory(swig)
endif()

if(OEFP_BUILD_BENCHMARKS)
    if(NOT OEFP_OECLUSTER_SOURCE_DIR)
        message(FATAL_ERROR "Set OEFP_OECLUSTER_SOURCE_DIR to build oecluster parity benchmarks")
    endif()
    set(OECLUSTER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(OECLUSTER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(OECLUSTER_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${OEFP_OECLUSTER_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/oecluster")
    add_subdirectory(benchmarks)
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
