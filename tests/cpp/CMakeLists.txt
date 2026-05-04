include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(oefp_tests
    test_oefp.cpp
)

target_link_libraries(oefp_tests
    PRIVATE
        oefp
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(oefp_tests)
