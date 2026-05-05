add_executable(oefp_oecluster_fingerprint_benchmark
    benchmark_oecluster_fingerprint.cpp
)

target_link_libraries(oefp_oecluster_fingerprint_benchmark
    PRIVATE
        oefp
        oecluster
)
