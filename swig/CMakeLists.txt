include(OpenEyeSWIG)

openeye_add_swig_module(
    NAME oefp
    SWIG_FILE oefp.i
    LINK_LIBS oefp
    PYTHON_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/python/oefp
    STABLE_ABI ${OEFP_USE_STABLE_ABI}
    SWIG_FLAGS "-doxygen"
    EXPECTED_LIB_VARS
        OECHEM_LIBRARY OESYSTEM_LIBRARY OEPLATFORM_LIBRARY
        OEMATH_LIBRARY OEZSTD_LIBRARY
    INIT_PY ${CMAKE_SOURCE_DIR}/python/oefp/__init__.py
)
