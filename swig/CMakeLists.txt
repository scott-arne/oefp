include(OpenEyeSWIG)

set(_oefp_swig_flags "-doxygen" "-I${CMAKE_SOURCE_DIR}/include")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _oefp_swig_flags "-DSWIGWORDSIZE64")
endif()

openeye_add_swig_module(
    NAME oefp
    SWIG_FILE oefp.i
    LINK_LIBS oefp
    PYTHON_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/python/oefp
    STABLE_ABI ${OEFP_USE_STABLE_ABI}
    SWIG_FLAGS ${_oefp_swig_flags}
    EXPECTED_LIB_VARS
        OECHEM_LIBRARY OESYSTEM_LIBRARY OEPLATFORM_LIBRARY
        OEMATH_LIBRARY OEMOLPROP_LIBRARY OEZSTD_LIBRARY
    INIT_PY ${CMAKE_SOURCE_DIR}/python/oefp/__init__.py
)

add_custom_command(TARGET oefp_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/_oefp.py
        ${CMAKE_SOURCE_DIR}/python/oefp/_native.py
    COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_SOURCE_DIR}/python/oefp/oefp.py
    COMMENT "Copying private OEFP SWIG proxy to python/oefp/_native.py"
)

# -----------------------------------------------------------------------------
# Editable dev RPATH fix for pyarrow
# -----------------------------------------------------------------------------
# The extension links libarrow/libparquet out of the active interpreter's
# pyarrow wheel, but those live in site-packages rather than anywhere on the
# default loader search path. openeye_add_swig_module() sets an RPATH covering
# @loader_path and the OpenEye runtime only, so the copy it drops in
# PYTHON_OUTPUT_DIR cannot resolve Arrow and the editable install fails to
# import with "Library not loaded: @rpath/libarrow.<soversion>.dylib".
#
# Patching the copy here keeps the fix tied to the build that produced it:
# pyarrow bumps its soversion on every major release, so a hand-applied
# install_name_tool goes stale silently the next time the environment moves.
#
# Wheels are excluded deliberately. There the extension reaches pyarrow through
# a relative @loader_path/../pyarrow entry, which is correct for any install
# location; baking a build machine's absolute site-packages path into a
# redistributable artifact would not be.
if(NOT SKBUILD)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import os, pyarrow; print(os.path.dirname(pyarrow.__file__))"
        OUTPUT_VARIABLE _OEFP_DEV_PYARROW_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(_OEFP_DEV_PYARROW_DIR AND IS_DIRECTORY "${_OEFP_DEV_PYARROW_DIR}")
        set(_OEFP_DEV_SO
            ${CMAKE_SOURCE_DIR}/python/oefp/$<TARGET_FILE_NAME:oefp_python>)

        if(APPLE)
            find_program(INSTALL_NAME_TOOL install_name_tool)
            if(INSTALL_NAME_TOOL)
                # Adding an RPATH invalidates the signature, so re-sign ad hoc.
                add_custom_command(TARGET oefp_python POST_BUILD
                    COMMAND ${INSTALL_NAME_TOOL} -add_rpath
                        "${_OEFP_DEV_PYARROW_DIR}" ${_OEFP_DEV_SO}
                    COMMAND codesign -f -s - ${_OEFP_DEV_SO}
                    COMMENT "Adding pyarrow RPATH for editable dev install (oefp)"
                )
            endif()
        elseif(UNIX)
            find_program(PATCHELF patchelf)
            if(PATCHELF)
                add_custom_command(TARGET oefp_python POST_BUILD
                    COMMAND ${PATCHELF} --add-rpath
                        "${_OEFP_DEV_PYARROW_DIR}" ${_OEFP_DEV_SO}
                    COMMENT "Adding pyarrow RPATH for editable dev install (oefp)"
                )
            endif()
        endif()
    else()
        message(WARNING
            "pyarrow not importable from ${Python3_EXECUTABLE}; the editable "
            "install will not be able to resolve libarrow at import time.")
    endif()
endif()

install(CODE [[
    set(_proxy "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/oefp/oefp.py")
    set(_private_proxy "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/oefp/_native.py")
    if(EXISTS "${_proxy}")
        file(RENAME "${_proxy}" "${_private_proxy}")
        list(REMOVE_ITEM CMAKE_INSTALL_MANIFEST_FILES "${_proxy}")
        list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${_private_proxy}")
    endif()
]])

install(FILES
    ${CMAKE_SOURCE_DIR}/python/oefp/py.typed
    ${CMAKE_SOURCE_DIR}/python/oefp/mordred_references.json
    ${CMAKE_SOURCE_DIR}/python/oefp/rdkit_references.json
    ${CMAKE_SOURCE_DIR}/python/oefp/kallisto_references.json
    ${CMAKE_SOURCE_DIR}/python/oefp/THIRD_PARTY_NOTICES
    DESTINATION oefp
)
