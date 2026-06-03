include(FindPackageHandleStandardArgs)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pyarrow as pa; print(pa.get_include()); print('\\n'.join(pa.get_library_dirs()))"
    OUTPUT_VARIABLE _arrow_python_probe
    ERROR_VARIABLE _arrow_python_error
    RESULT_VARIABLE _arrow_python_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT _arrow_python_result EQUAL 0)
    message(FATAL_ERROR
        "Could not import pyarrow with ${Python3_EXECUTABLE}. "
        "Install pyarrow>=24.\n${_arrow_python_error}"
    )
endif()

string(REPLACE "\n" ";" _arrow_python_lines "${_arrow_python_probe}")
list(LENGTH _arrow_python_lines _arrow_python_line_count)
if(_arrow_python_line_count LESS 2)
    message(FATAL_ERROR
        "pyarrow did not report both an include directory and a library directory."
    )
endif()

list(GET _arrow_python_lines 0 ARROW_PYTHON_INCLUDE_DIR)
set(ARROW_PYTHON_LIBRARY_DIRS ${_arrow_python_lines})
list(REMOVE_AT ARROW_PYTHON_LIBRARY_DIRS 0)

# pyarrow reports native paths; on Windows these use backslashes, which are
# escape characters inside file(GLOB) patterns (a path like
# C:\hostedtoolcache\... fails to parse as "Invalid character escape"). Convert
# every path to a forward-slash CMake path before it is used in a glob or as an
# include directory.
file(TO_CMAKE_PATH "${ARROW_PYTHON_INCLUDE_DIR}" ARROW_PYTHON_INCLUDE_DIR)
set(_arrow_python_cmake_dirs "")
foreach(_arrow_python_native_dir IN LISTS ARROW_PYTHON_LIBRARY_DIRS)
    file(TO_CMAKE_PATH "${_arrow_python_native_dir}" _arrow_python_native_dir)
    list(APPEND _arrow_python_cmake_dirs "${_arrow_python_native_dir}")
endforeach()
set(ARROW_PYTHON_LIBRARY_DIRS ${_arrow_python_cmake_dirs})

foreach(_arrow_python_dir IN LISTS ARROW_PYTHON_LIBRARY_DIRS)
    file(GLOB _arrow_python_arrow_candidates CONFIGURE_DEPENDS
        "${_arrow_python_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}arrow${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_arrow_python_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}arrow.*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_arrow_python_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}arrow${CMAKE_SHARED_LIBRARY_SUFFIX}.*"
    )
    file(GLOB _arrow_python_parquet_candidates CONFIGURE_DEPENDS
        "${_arrow_python_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}parquet${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_arrow_python_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}parquet.*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_arrow_python_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}parquet${CMAKE_SHARED_LIBRARY_SUFFIX}.*"
    )
    list(APPEND _arrow_python_all_arrow_candidates ${_arrow_python_arrow_candidates})
    list(APPEND _arrow_python_all_parquet_candidates ${_arrow_python_parquet_candidates})
endforeach()

list(SORT _arrow_python_all_arrow_candidates)
list(SORT _arrow_python_all_parquet_candidates)
list(REVERSE _arrow_python_all_arrow_candidates)
list(REVERSE _arrow_python_all_parquet_candidates)

if(_arrow_python_all_arrow_candidates)
    list(GET _arrow_python_all_arrow_candidates 0 ARROW_PYTHON_ARROW_LIBRARY)
endif()
if(_arrow_python_all_parquet_candidates)
    list(GET _arrow_python_all_parquet_candidates 0 ARROW_PYTHON_PARQUET_LIBRARY)
endif()

if(WIN32)
    find_library(ARROW_PYTHON_ARROW_IMPLIB
        NAMES arrow libarrow
        PATHS ${ARROW_PYTHON_LIBRARY_DIRS}
        NO_DEFAULT_PATH
    )
    find_library(ARROW_PYTHON_PARQUET_IMPLIB
        NAMES parquet libparquet
        PATHS ${ARROW_PYTHON_LIBRARY_DIRS}
        NO_DEFAULT_PATH
    )
endif()

set(_arrow_python_required_vars
    ARROW_PYTHON_INCLUDE_DIR
    ARROW_PYTHON_ARROW_LIBRARY
    ARROW_PYTHON_PARQUET_LIBRARY
)

if(WIN32)
    list(APPEND _arrow_python_required_vars
        ARROW_PYTHON_ARROW_IMPLIB
        ARROW_PYTHON_PARQUET_IMPLIB
    )
endif()

find_package_handle_standard_args(ArrowPython
    REQUIRED_VARS ${_arrow_python_required_vars}
)

if(ArrowPython_FOUND)
    add_library(ArrowPython::Arrow SHARED IMPORTED)
    set_target_properties(ArrowPython::Arrow PROPERTIES
        IMPORTED_LOCATION "${ARROW_PYTHON_ARROW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARROW_PYTHON_INCLUDE_DIR}"
    )
    if(WIN32)
        set_target_properties(ArrowPython::Arrow PROPERTIES
            IMPORTED_IMPLIB "${ARROW_PYTHON_ARROW_IMPLIB}"
        )
    endif()

    add_library(ArrowPython::Parquet SHARED IMPORTED)
    set_target_properties(ArrowPython::Parquet PROPERTIES
        IMPORTED_LOCATION "${ARROW_PYTHON_PARQUET_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARROW_PYTHON_INCLUDE_DIR}"
    )
    if(WIN32)
        set_target_properties(ArrowPython::Parquet PROPERTIES
            IMPORTED_IMPLIB "${ARROW_PYTHON_PARQUET_IMPLIB}"
        )
    endif()
endif()
