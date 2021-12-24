/*
    Helper Utilities
    Throw all useful functions that may or may not be used more than once in here

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#if !defined(__BW_HELPER_UTILITIES_H)
#define __BW_HELPER_UTILITIES_H

#include <cstdlib>
#include <cstdio>
#include <random>

#include "common.h"

#define PRINT_PASSED(msg, ...) printf("\x1b[1;32mPASSED:\x1b[0m " msg "\n",  ##__VA_ARGS__)
#define PRINT_FAIL(msg, ...) printf("\x1b[1;31mFAIL:\x1b[0m " msg "\n",  ##__VA_ARGS__)
#define PRINT_VIOLATION(msg, num, ...) printf("\x1b[1;35m%zu VIOLATIONS:\x1b[0m " msg "\n", num,  ##__VA_ARGS__)

/**
 * Initializes the given BWdata with uniform at random data.
 * Initialized Data: init_prob, trans_prob, emit_prob, observations
 */
void initialize_uar(const BWdata& bw);

/**
 * Initializes the given BWdata with random data.
 * Initialized Data: init_prob, trans_prob, emit_prob, observations
 */
void initialize_random(const BWdata& bw);

/**
 * Checks and verifies that BWdata has the following properties:
 * - Initial distribution sums to 1.0
 * - Rows of the transition distribution sums to 1.0
 * - Rows of the emission distribution sums to 1.0
 * - Negative log likelihood sequence must be monotonically increasing
 *
 * Returns: True if there was no error, false otherwise
 */
bool check_and_verify(const BWdata& bw);

/**
 * Prints the following states of the given BWdata:
 * - Initialization probabilities
 * - Transition probabilities
 * - Emission probabilities
 */
void print_states(const BWdata& bw);

/**
 * Prints all the data from the BWdata struct
 * Used for debugging with small values for K, N, M, T
 */
void print_BWdata(const BWdata& bw);

/**
 * Prints additional information in addition to print_BWdata
 * Additional information: iteration_variable, message
 */
inline void print_BWdata_debug_helper(const BWdata& bw, const size_t iteration_variable, const char* message) {
    printf("\n\x1b[1;33m[i = %zu] %s\x1b[0m", iteration_variable, message);
    print_BWdata(bw);
}

/**
 * Transposes a matrix from input to matrix output.
 */
void transpose_matrix(double* output, const double* input, const size_t N, const size_t M);

/**
 * Compares only the following fields of the two given BWdata structs (considering EPSILON)
 * - Initialization Probabilities
 * - Transition Probabilities
 * - Emission Probabilities
 *
 * Returns: true if the checked fields of both structs contain the same data up to EPSILON
 * */
bool is_BWdata_equal_only_probabilities(const BWdata& bw1, const BWdata& bw2);

/**
 * Compares all fields of the two given BWdata structs (considering EPSILON)
 *
 * Returns: true if both structs contain the same data up to EPSILON
 * */
bool is_BWdata_equal(const BWdata& bw1, const BWdata& bw2);

#endif /* __BW_HELPER_UTILITIES_H */