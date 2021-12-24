cmake_minimum_required(VERSION 3.10)
project(ASL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -O3 -ffast-math -mavx2 -mfma -march=native")

SET_SOURCE_FILES_PROPERTIES(helper_utilities.cpp PROPERTIES COMPILE_FLAGS "-fno-fast-math -O0")
SET_SOURCE_FILES_PROPERTIES(verifications.cpp PROPERTIES COMPILE_FLAGS "-fno-fast-math -O0")
SET_SOURCE_FILES_PROPERTIES(benchmarks.cpp PROPERTIES COMPILE_FLAGS "-fno-fast-math -O0")

add_executable(verification
    common.cpp
    helper_utilities.cpp
    verifications.cpp
    implementations/baseline.cpp
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/unrolled_optimized.cpp
    implementations/vector_optimized.cpp
    implementations/combined_optimized.cpp
    implementations/scalar_optimized_playground.cpp
    implementations/unrolled_emit_prob.cpp
#    implementations/baseline_transposed_emit_prob.cpp # Test transpose of emit_prob
)

add_executable(benchmarks
    common.cpp
    helper_utilities.cpp
    benchmarks.cpp
    implementations/baseline.cpp
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/unrolled_optimized.cpp
    implementations/vector_optimized.cpp
    implementations/combined_optimized.cpp
    #implementations/scalar_optimized_playground.cpp
    #implementations/unrolled_emit_prob.cpp
#    implementations/baseline_transposed_emit_prob.cpp # Test transpose of emit_prob
)

add_executable(benchmarks-no-vector
    common.cpp
    helper_utilities.cpp
    benchmarks.cpp
    implementations/baseline.cpp
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/unrolled_optimized.cpp
    implementations/vector_optimized.cpp
    implementations/combined_optimized.cpp
    #implementations/scalar_optimized_playground.cpp
    #implementations/unrolled_emit_prob.cpp
#    implementations/baseline_transposed_emit_prob.cpp # Test transpose of emit_prob
)

add_executable(benchmarks-unroll
    common.cpp
    helper_utilities.cpp
    benchmarks.cpp
    implementations/baseline.cpp
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/unrolled_optimized.cpp
    implementations/vector_optimized.cpp
    implementations/combined_optimized.cpp
    #implementations/scalar_optimized_playground.cpp
    #implementations/unrolled_emit_prob.cpp
#    implementations/baseline_transposed_emit_prob.cpp # Test transpose of emit_prob
)

SET_TARGET_PROPERTIES(benchmarks-no-vector PROPERTIES COMPILE_FLAGS "-fno-tree-vectorize")
SET_TARGET_PROPERTIES(benchmarks-unroll PROPERTIES COMPILE_FLAGS "-funroll-loops")
