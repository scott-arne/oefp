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
    test_batch.cpp
    test_compare.cpp
    test_count.cpp
    test_fingerprint.cpp
    test_metric.cpp
    test_morgan.cpp
    test_oefp.cpp
    test_openeye_interop.cpp
)

target_link_libraries(oefp_tests
    PRIVATE
        oefp
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(oefp_tests)
