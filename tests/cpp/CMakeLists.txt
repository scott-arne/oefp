include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(oefp_tests
    test_annotation.cpp
    test_atom_pair.cpp
    test_batch.cpp
    test_column_request.cpp
    test_compare.cpp
    test_compute_context.cpp
    test_count.cpp
    test_count_batch.cpp
    test_descriptor.cpp
    test_descriptor_arrow.cpp
    test_descriptor_batch.cpp
    test_descriptor_calculator.cpp
    test_descriptor_compare.cpp
    test_descriptor_schema.cpp
    test_descriptor_selection.cpp
    test_descriptor_set.cpp
    test_descriptor_source.cpp
    test_descriptor_value.cpp
    test_fingerprint.cpp
    test_metric.cpp
    test_molecular_properties.cpp
    test_mordred.cpp
    test_mordred_pruning_equivalence.cpp
    test_mordred_schema.cpp
    test_morgan.cpp
    test_openeye_interop.cpp
    test_rdkit_schema.cpp
    test_sparse.cpp
    test_sparse_batch.cpp
    test_topological_torsions.cpp
)

target_link_libraries(oefp_tests
    PRIVATE
        oefp
        GTest::gtest_main
)
target_compile_features(oefp_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(oefp_tests)
