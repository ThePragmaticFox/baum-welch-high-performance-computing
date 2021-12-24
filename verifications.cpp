/*
    Verifications for the various Baum Welch algorithm implementations
    If you find other test cases, add them!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#include <cstdlib>
#include <cstdio>
#include <tuple>
#include <random>
#include <ctime>
#include <unistd.h>
// custom files for the project
#include "helper_utilities.h"
#include "common.h"

void check_baseline(void);
void check_user_functions(const size_t& nb_random_tests);
bool test_case_ghmm_0(compute_bw_func func);
bool test_case_ghmm_1(compute_bw_func func);
bool test_case_ghmm_2(compute_bw_func func);
bool test_case_ghmm_3(compute_bw_func func);
bool test_case_randomized(compute_bw_func func);

int main() {
    // maybe add commandline arguments, dunno
    if (true) check_baseline();
    const size_t nb_random_tests = 5;
    if (true) check_user_functions(nb_random_tests);
}

/**
 * Verifies the baseline implementation using an existing and verified example created in ghmm
 * For reproducibility purposes, the code can be found in misc/ghmm_experiments.ipynb
 * Requires Jupyter Notebook, Python 2.7 and Linux
 * ghmm documentation: http://ghmm.sourceforge.net/index.html
 * Download link (working 09.05.2020): https://sourceforge.net/projects/ghmm/
 */
inline void check_baseline(void) {
    const size_t baseline_random_seed = time(NULL);
    srand(baseline_random_seed);
    const size_t baseline_random_number = rand();

    // hardcoded test cases to check correctness of the baseline implementation
    printf("\x1b[1m\n-------------------------------------------------------------------------------\x1b[0m\n");
    printf("\x1b[1mBaseline Verifications with Baseline Random Number [%zu]\x1b[0m\n", baseline_random_number);
    printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");

    // TEST 0
    const bool success_test_case_ghmm_0 = test_case_ghmm_0(FuncRegister::baseline_func);
    if (success_test_case_ghmm_0) {
        printf("\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 0\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 0\n");
    }

    // TEST 1
    const bool success_test_case_ghmm_1 = test_case_ghmm_1(FuncRegister::baseline_func);
    if (success_test_case_ghmm_1) {
        printf("\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 1\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 1\n");
    }

    // TEST 2
    const bool success_test_case_ghmm_2 = test_case_ghmm_2(FuncRegister::baseline_func);
    if (success_test_case_ghmm_2) {
        printf("\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 2\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 2\n");
    }

    // TEST 3
    const bool success_test_case_ghmm_3 = test_case_ghmm_3(FuncRegister::baseline_func);
    if (success_test_case_ghmm_3) {
        printf("\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 3\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 3\n");
    }
    printf("-------------------------------------------------------------------------------\n");
}

/**
 * Verifies the correctness of the optimized w.r.t. the baseline using randomized tests with the following assumptions:
 * - sufficiently high (>= 16) values
 * - divisibility of (16)
 */
inline void check_user_functions(const size_t& nb_random_tests) {

    const size_t nb_user_functions = FuncRegister::size();
    bool test_results[nb_user_functions][nb_random_tests];

    // check optimizations w.r.t. the baseline using randomized tests
    for (size_t i = 0; i < nb_random_tests; i++) {

        // randomize seed (new for each random test case)
        const size_t baseline_random_seed = time(NULL)*i;
        srand(baseline_random_seed);
        size_t baseline_random_number = rand();

        // initialize data for random test case i
        // assumption: sufficiently high (>= 16) values && divisibility of (16)
        // T = 2 (mod 16), due to "1 to T-2" loops in compute_gamma
        const size_t K = (rand() % 2)*16 + 16; // don't touch
        const size_t N = (rand() % 2)*16 + 16; // don't touch
        const size_t M = (rand() % 2)*16 + 16; // don't touch
        const size_t T = (rand() % 2)*16 + 32; // don't touch
        const size_t max_iterations = 500;

        // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
        const BWdata& bw_baseline_initialized = *new BWdata(K, N, M, T, max_iterations);
        // initialize_uar(bw_baseline_initialized); // converges fast, but works now.
        initialize_random(bw_baseline_initialized);
        const BWdata& bw_baseline = bw_baseline_initialized.deep_copy();

        // run baseline and don't touch the bw_baseline data:
        // each of the users function bw_user_function data in
        // the loop below will be checked against bw_baseline data!
        printf("\x1b[1m\n-------------------------------------------------------------------------------\x1b[0m\n");
        printf("\x1b[1mTest Case Randomized [%zu] with Baseline Random Number [%zu]\x1b[0m\n", i, baseline_random_number);
        printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
        printf("Initialized: K = %zu, N = %zu, M = %zu, T = %zu and max_iterations = %zu\n", K, N, M, T, max_iterations);
        printf("-------------------------------------------------------------------------------\n");
        printf("Running \x1b[1m'Baseline'\x1b[0m\n");
        printf("-------------------------------------------------------------------------------\n");
        const size_t baseline_convergence = FuncRegister::baseline_func(bw_baseline);
        printf("It took \x1b[1m[%zu] iterations\x1b[0m until convergence\n", baseline_convergence);
        printf("-------------------------------------------------------------------------------\n");
        const bool baseline_success = check_and_verify(bw_baseline);
        printf("-------------------------------------------------------------------------------\n");

        // run all user functions and compare against the data
        for(size_t f = 0; f < nb_user_functions; f++) {
            printf("Running User Function \x1b[1m'%s'\x1b[0m\n", FuncRegister::funcs->at(f).name.c_str());
            printf("-------------------------------------------------------------------------------\n");
            const BWdata& bw_user_function = bw_baseline_initialized.deep_copy();

            // Hacky but it works: Transpose emit_prob
            if(FuncRegister::funcs->at(f).transpose_emit_prob){
                double *new_emit_prob = (double *)malloc(bw_user_function.N*bw_user_function.M * sizeof(double));
                transpose_matrix(new_emit_prob, bw_user_function.emit_prob, bw_user_function.N, bw_user_function.M);
                memcpy(bw_user_function.emit_prob, new_emit_prob, bw_user_function.N*bw_user_function.M * sizeof(double));
                free(new_emit_prob);
            }

            const size_t user_function_convergence = FuncRegister::funcs->at(f).func(bw_user_function);

            // Transpose back to do verification
            if(FuncRegister::funcs->at(f).transpose_emit_prob){
                double *new_emit_prob = (double *)malloc(bw_user_function.N*bw_user_function.M * sizeof(double));
                transpose_matrix(new_emit_prob, bw_user_function.emit_prob, bw_user_function.M, bw_user_function.N);
                memcpy(bw_user_function.emit_prob, new_emit_prob, bw_user_function.N*bw_user_function.M * sizeof(double));
                free(new_emit_prob);
            }

            printf("It took \x1b[1m[%zu] iterations\x1b[0m to converge\n", user_function_convergence);
            printf("-------------------------------------------------------------------------------\n");
            const bool user_function_success = check_and_verify(bw_user_function);
            printf("-------------------------------------------------------------------------------\n");
            const bool is_bw_baseline_equal_bw_user_function = is_BWdata_equal(bw_baseline, bw_user_function);
            //const bool is_bw_baseline_equal_bw_user_function = is_BWdata_equal_only_probabilities(bw_baseline, bw_user_function);
            printf("-------------------------------------------------------------------------------\n");
            //const bool is_bw_baseline_equal_bw_user_function = is_BWdata_equal_only_probabilities(bw_baseline, bw_user_function);

            // Okay, hear me out!
            // If baseline is correct, then that's dope and we wanna have user function also correct, right?
            // Though, if baseline is wrong, then user function being true might be some potential bug-problem!
            // NOTE 1
            // This only concerns the "check_and_verify" function; which checks e.g. whether probabilities sum to 1.
            // However, the functional correctness may still be wrong, but that wouldn't be the fault of the user_function.
            // The only job of the user_function is to match the baseline; the baseline itself has to be correct!
            // NOTE 2
            // Checking the convergence rate of the Baseline with the User Function may or may not be a good idea
            // U may change (false -> true), but no big h8sies pls uwu
            test_results[f][i] = (
                   ( false || is_bw_baseline_equal_bw_user_function )
                && ( false || user_function_success )
                && ( true  || (user_function_convergence == baseline_convergence) )
                && ( false || (user_function_success == baseline_success) )
            );

            delete &bw_user_function;
        }

        delete &bw_baseline;
        delete &bw_baseline_initialized;
    }

    printf("\nAll Tests Done!\n\n");
    printf("Results:\n");
    printf("-------------------------------------------------------------------------------\n");
    for (size_t f = 0; f < nb_user_functions; f++) {

        size_t nb_fails = 0;
        for (size_t i = 0; i < nb_random_tests; i++) {
            if (!test_results[f][i]) nb_fails++;
        }

        printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
        if(nb_fails == 0){
            printf("\x1b[1;32mALL CASES PASSED:\x1b[0m '%s': %s\n", FuncRegister::funcs->at(f).name.c_str(), FuncRegister::funcs->at(f).description.c_str());
        } else {
            printf("\x1b[1;31m[%zu/%zu] CASES FAILED:\x1b[0m '%s': %s \n", nb_fails, nb_random_tests, FuncRegister::funcs->at(f).name.c_str(), FuncRegister::funcs->at(f).description.c_str());
        }
        printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
        for (size_t i = 0; i < nb_random_tests; i++) {
            if(test_results[f][i]){
                printf("\x1b[1;32mPASSED\x1b[0m Test Case Randomized [%zu]\n", i);
            } else {
                printf("\x1b[1;31mFAILED:\x1b[0m Test Case Randomized [%zu]\n", i);
            }
        }
    }
    printf("-------------------------------------------------------------------------------\n");
}

/**
 * The following test cases check against examples created in ghmm
 * For reproducibility purposes, the code can be found in misc/ghmm_experiments.ipynb
 * Requires Jupyter Notebook, Python 2.7 and Linux
 * ghmm documentation: http://ghmm.sourceforge.net/index.html
 * Download link (working 09.05.2020): https://sourceforge.net/projects/ghmm/
 */
bool test_case_ghmm_0(compute_bw_func func) {
    const size_t K = 1;
    const size_t N = 2;
    const size_t M = 2;
    const size_t T = 10;
    const size_t max_iterations = 1; 

    const BWdata& bw = *new BWdata(K, N, M, T, max_iterations);
    const BWdata& bw_check = *new BWdata(K, N, M, T, max_iterations);

    bw.observations[0*T + 0] = 0;
    bw.observations[0*T + 1] = 0;
    bw.observations[0*T + 2] = 0;
    bw.observations[0*T + 3] = 0;
    bw.observations[0*T + 4] = 0;
    bw.observations[0*T + 5] = 1;
    bw.observations[0*T + 6] = 1;
    bw.observations[0*T + 7] = 0;
    bw.observations[0*T + 8] = 0;
    bw.observations[0*T + 9] = 0;

    bw.init_prob[0] = 0.20000000;
    bw.init_prob[1] = 0.80000000;

    bw.trans_prob[0*N + 0] = 0.50000000;
    bw.trans_prob[0*N + 1] = 0.50000000;
    bw.trans_prob[1*N + 0] = 0.30000000;
    bw.trans_prob[1*N + 1] = 0.70000000;

    bw.emit_prob[0*M + 0] = 0.30000000;
    bw.emit_prob[0*M + 1] = 0.70000000;
    bw.emit_prob[1*M + 0] = 0.80000000;
    bw.emit_prob[1*M + 1] = 0.20000000;

    bw_check.init_prob[0] = 0.07187023;
    bw_check.init_prob[1] = 0.92812977;

    bw_check.trans_prob[0*N + 0] = 0.43921478;
    bw_check.trans_prob[0*N + 1] = 0.56078522;
    bw_check.trans_prob[1*N + 0] = 0.21445682;
    bw_check.trans_prob[1*N + 1] = 0.78554318;

    bw_check.emit_prob[0*M + 0] = 0.46160107;
    bw_check.emit_prob[0*M + 1] = 0.53839893;
    bw_check.emit_prob[1*M + 0] = 0.91501557;
    bw_check.emit_prob[1*M + 1] = 0.08498443;

    // run learning algorithm baum-welch
    func(bw);

    //print_BWdata(bw); // debugging helper

    // checks only conceptual stuff;
    // e.g. whether probabilities work out (sum to 1)
    bool success = check_and_verify(bw);
    success = success && is_BWdata_equal_only_probabilities(bw, bw_check);

    //print_states(bw);

    delete &bw;
    delete &bw_check;

    return success;
}

bool test_case_ghmm_1(compute_bw_func func) {
    const size_t K = 1;
    const size_t N = 2;
    const size_t M = 2;
    const size_t T = 10;
    const size_t max_iterations = 1000;

    const BWdata& bw = *new BWdata(K, N, M, T, max_iterations);
    const BWdata& bw_check = *new BWdata(K, N, M, T, max_iterations);

    bw.observations[0*T + 0] = 0;
    bw.observations[0*T + 1] = 0;
    bw.observations[0*T + 2] = 0;
    bw.observations[0*T + 3] = 0;
    bw.observations[0*T + 4] = 0;
    bw.observations[0*T + 5] = 1;
    bw.observations[0*T + 6] = 1;
    bw.observations[0*T + 7] = 0;
    bw.observations[0*T + 8] = 0;
    bw.observations[0*T + 9] = 0;

    bw.init_prob[0] = 0.20000000;
    bw.init_prob[1] = 0.80000000;

    bw.trans_prob[0*N + 0] = 0.50000000;
    bw.trans_prob[0*N + 1] = 0.50000000;
    bw.trans_prob[1*N + 0] = 0.30000000;
    bw.trans_prob[1*N + 1] = 0.70000000;

    bw.emit_prob[0*M + 0] = 0.30000000;
    bw.emit_prob[0*M + 1] = 0.70000000;
    bw.emit_prob[1*M + 0] = 0.80000000;
    bw.emit_prob[1*M + 1] = 0.20000000;

    bw_check.init_prob[0] = 0.00000000;
    bw_check.init_prob[1] = 1.00000000;

    bw_check.trans_prob[0*N + 0] = 0.50000000;
    bw_check.trans_prob[0*N + 1] = 0.50000000;
    bw_check.trans_prob[1*N + 0] = 0.14285714;
    bw_check.trans_prob[1*N + 1] = 0.85714286;

    bw_check.emit_prob[0*M + 0] = 0.00000000;
    bw_check.emit_prob[0*M + 1] = 1.00000000;
    bw_check.emit_prob[1*M + 0] = 1.00000000;
    bw_check.emit_prob[1*M + 1] = 0.00000000;

    // run learning algorithm baum-welch
    func(bw);

    // checks only conceptual stuff;
    // e.g. whether probabilities work out (sum to 1)
    bool success = check_and_verify(bw);
    success = success && is_BWdata_equal_only_probabilities(bw, bw_check);

    //print_states(bw);

    delete &bw;
    delete &bw_check;

    return success;
}

bool test_case_ghmm_2(compute_bw_func func) {
    const size_t K = 1;
    const size_t N = 3;
    const size_t M = 3;
    const size_t T = 16;
    const size_t max_iterations = 1;

    const BWdata& bw = *new BWdata(K, N, M, T, max_iterations);
    const BWdata& bw_check = *new BWdata(K, N, M, T, max_iterations);

    bw.observations[0*T + 0] = 2;
    bw.observations[0*T + 1] = 1;
    bw.observations[0*T + 2] = 0;
    bw.observations[0*T + 3] = 1;
    bw.observations[0*T + 4] = 0;
    bw.observations[0*T + 5] = 2;
    bw.observations[0*T + 6] = 2;
    bw.observations[0*T + 7] = 2;
    bw.observations[0*T + 8] = 1;
    bw.observations[0*T + 9] = 1;
    bw.observations[0*T + 10] = 2;
    bw.observations[0*T + 11] = 2;
    bw.observations[0*T + 12] = 0;
    bw.observations[0*T + 13] = 1;
    bw.observations[0*T + 14] = 1;
    bw.observations[0*T + 15] = 0;

    bw.init_prob[0] = 0.10000000;
    bw.init_prob[1] = 0.50000000;
    bw.init_prob[2] = 0.40000000;

    bw.trans_prob[0*N + 0] = 0.50000000;
    bw.trans_prob[0*N + 1] = 0.40000000;
    bw.trans_prob[0*N + 2] = 0.10000000;
    bw.trans_prob[1*N + 0] = 0.30000000;
    bw.trans_prob[1*N + 1] = 0.30000000;
    bw.trans_prob[1*N + 2] = 0.40000000;
    bw.trans_prob[2*N + 0] = 0.10000000;
    bw.trans_prob[2*N + 1] = 0.10000000;
    bw.trans_prob[2*N + 2] = 0.80000000;

    bw.emit_prob[0*M + 0] = 0.10000000;
    bw.emit_prob[0*M + 1] = 0.00000000;
    bw.emit_prob[0*M + 2] = 0.90000000;
    bw.emit_prob[1*M + 0] = 0.05000000;
    bw.emit_prob[1*M + 1] = 0.95000000;
    bw.emit_prob[1*M + 2] = 0.00000000;
    bw.emit_prob[2*M + 0] = 0.30000000;
    bw.emit_prob[2*M + 1] = 0.52000000;
    bw.emit_prob[2*M + 2] = 0.18000000;

    bw_check.init_prob[0] = 0.43136131;
    bw_check.init_prob[1] = 0.00000000;
    bw_check.init_prob[2] = 0.56863869;

    bw_check.trans_prob[0*N + 0] = 0.54129839;
    bw_check.trans_prob[0*N + 1] = 0.33806567;
    bw_check.trans_prob[0*N + 2] = 0.12063595;
    bw_check.trans_prob[1*N + 0] = 0.24124142;
    bw_check.trans_prob[1*N + 1] = 0.30826810;
    bw_check.trans_prob[1*N + 2] = 0.45049048;
    bw_check.trans_prob[2*N + 0] = 0.14334083;
    bw_check.trans_prob[2*N + 1] = 0.07031231;
    bw_check.trans_prob[2*N + 2] = 0.78634686;

    bw_check.emit_prob[0*M + 0] = 0.13073710;
    bw_check.emit_prob[0*M + 1] = 0.00000000;
    bw_check.emit_prob[0*M + 2] = 0.86926290;
    bw_check.emit_prob[1*M + 0] = 0.09380801;
    bw_check.emit_prob[1*M + 1] = 0.90619199;
    bw_check.emit_prob[1*M + 2] = 0.00000000;
    bw_check.emit_prob[2*M + 0] = 0.37463835;
    bw_check.emit_prob[2*M + 1] = 0.39607634;
    bw_check.emit_prob[2*M + 2] = 0.22928531;

    // run learning algorithm baum-welch
    func(bw);

    // checks only conceptual stuff;
    // e.g. whether probabilities work out (sum to 1)
    bool success = check_and_verify(bw);
    success = success && is_BWdata_equal_only_probabilities(bw, bw_check);

    //print_states(bw);

    delete &bw;
    delete &bw_check;

    return success;
}


bool test_case_ghmm_3(compute_bw_func func) {
    const size_t K = 1;
    const size_t N = 3;
    const size_t M = 3;
    const size_t T = 16;
    // ghmm stops after 93 iterations
    // need to set the same, otherwise our implementation will converge to better values
    const size_t max_iterations = 93;

    const BWdata& bw = *new BWdata(K, N, M, T, max_iterations);
    const BWdata& bw_check = *new BWdata(K, N, M, T, max_iterations);

    bw.observations[0*T + 0] = 2;
    bw.observations[0*T + 1] = 1;
    bw.observations[0*T + 2] = 0;
    bw.observations[0*T + 3] = 1;
    bw.observations[0*T + 4] = 0;
    bw.observations[0*T + 5] = 2;
    bw.observations[0*T + 6] = 2;
    bw.observations[0*T + 7] = 2;
    bw.observations[0*T + 8] = 1;
    bw.observations[0*T + 9] = 1;
    bw.observations[0*T + 10] = 2;
    bw.observations[0*T + 11] = 2;
    bw.observations[0*T + 12] = 0;
    bw.observations[0*T + 13] = 1;
    bw.observations[0*T + 14] = 1;
    bw.observations[0*T + 15] = 0;

    bw.init_prob[0] = 0.10000000;
    bw.init_prob[1] = 0.50000000;
    bw.init_prob[2] = 0.40000000;

    bw.trans_prob[0*N + 0] = 0.50000000;
    bw.trans_prob[0*N + 1] = 0.40000000;
    bw.trans_prob[0*N + 2] = 0.10000000;
    bw.trans_prob[1*N + 0] = 0.30000000;
    bw.trans_prob[1*N + 1] = 0.30000000;
    bw.trans_prob[1*N + 2] = 0.40000000;
    bw.trans_prob[2*N + 0] = 0.10000000;
    bw.trans_prob[2*N + 1] = 0.10000000;
    bw.trans_prob[2*N + 2] = 0.80000000;

    bw.emit_prob[0*M + 0] = 0.10000000;
    bw.emit_prob[0*M + 1] = 0.00000000;
    bw.emit_prob[0*M + 2] = 0.90000000;
    bw.emit_prob[1*M + 0] = 0.05000000;
    bw.emit_prob[1*M + 1] = 0.95000000;
    bw.emit_prob[1*M + 2] = 0.00000000;
    bw.emit_prob[2*M + 0] = 0.30000000;
    bw.emit_prob[2*M + 1] = 0.52000000;
    bw.emit_prob[2*M + 2] = 0.18000000;

    bw_check.init_prob[0] = 1.00000000;
    bw_check.init_prob[1] = 0.00000000;
    bw_check.init_prob[2] = 0.00000000;

    bw_check.trans_prob[0*N + 0] = 0.50000000;
    bw_check.trans_prob[0*N + 1] = 0.50000000;
    bw_check.trans_prob[0*N + 2] = 0.00000000;
    bw_check.trans_prob[1*N + 0] = 0.00000000;
    bw_check.trans_prob[1*N + 1] = 0.52712619;
    bw_check.trans_prob[1*N + 2] = 0.47287381;
    bw_check.trans_prob[2*N + 0] = 0.61350576;
    bw_check.trans_prob[2*N + 1] = 0.00000000;
    bw_check.trans_prob[2*N + 2] = 0.38649424;

    bw_check.emit_prob[0*M + 0] = 0.00000000;
    bw_check.emit_prob[0*M + 1] = 0.00000000;
    bw_check.emit_prob[0*M + 2] = 1.00000000;
    bw_check.emit_prob[1*M + 0] = 0.33263613;
    bw_check.emit_prob[1*M + 1] = 0.66736387;
    bw_check.emit_prob[1*M + 2] = 0.00000000;
    bw_check.emit_prob[2*M + 0] = 0.50213607;
    bw_check.emit_prob[2*M + 1] = 0.49786393;
    bw_check.emit_prob[2*M + 2] = 0.00000000;

    // run learning algorithm baum-welch
    func(bw);

    // checks only conceptual stuff;
    // e.g. whether probabilities work out (sum to 1)
    bool success = check_and_verify(bw);
    success = success && is_BWdata_equal_only_probabilities(bw, bw_check);

    //print_states(bw);

    delete &bw;
    delete &bw_check;

    return success;
}
