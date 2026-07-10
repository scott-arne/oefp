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
    DESTINATION oefp
)
