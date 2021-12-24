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

#include "helper_utilities.h"

#define PRINT_BWDATA_MISSMATCH(msg, ...) printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m " msg, ##__VA_ARGS__)

void initialize_uar(const BWdata& bw) {
    const size_t K = bw.K;
    const size_t N = bw.N;
    const size_t M = bw.M;
    const size_t T = bw.T;

    // uniform at random set init_prob and trans_prob
    for (size_t n0 = 0; n0 < N; n0++) {
        bw.init_prob[n0] = 1.0/N;
        for (size_t n1 = 0; n1 < N; n1++) {
            bw.trans_prob[n0*N + n1] = 1.0/N;
        }
    }

    // uniform at random set emit_prob
    for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
            bw.emit_prob[n*M + m] = 1.0/M;
        }
    }

    // uniform at random set observations
    // (well, not really u.a.r. but let's pretend)
    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            // % T would be wrong, because the observation sequence over time 0 <= t < T
            // represents observations that access the emission states
            // emit_prob[n][observations[k][t]] in 0 <= m < M,
            // which is a categorical random variable
            bw.observations[k*T + t] = t % M;
        }
    }
}

void initialize_random(const BWdata& bw) {
    const size_t K = bw.K;
    const size_t N = bw.N;
    const size_t M = bw.M;
    const size_t T = bw.T;

    double init_sum;
    double trans_sum;
    double emit_sum;

    // randomly initialized init_prob
    init_sum = 0.0;
    while(init_sum == 0.0){
        for (size_t n = 0; n < N; n++) {
            bw.init_prob[n] = rand();
            init_sum += bw.init_prob[n];
        }
    }

    // the array init_prob must sum to 1.0
    for (size_t n = 0; n < N; n++) {
        bw.init_prob[n] /= init_sum;
    }

    // randomly initialized trans_prob rows
    for (size_t n0 = 0; n0 < N; n0++) {
        trans_sum = 0.0;
        while(trans_sum == 0.0){
            for (size_t n1 = 0; n1 < N; n1++) {
                bw.trans_prob[n0*N + n1] = rand();
                trans_sum += bw.trans_prob[n0*N + n1];
            }
        }

        // the row trans_prob[n0*N] must sum to 1.0
        for (size_t n1 = 0; n1 < N; n1++) {
            bw.trans_prob[n0*N + n1] /= trans_sum;
        }
    }

    // randomly initialized emit_prob rows
    for (size_t n = 0; n < N; n++) {
        emit_sum = 0.0;
        while (emit_sum == 0.0) {
            for (size_t m = 0; m < M; m++) {
                bw.emit_prob[n * M + m] = rand();
                emit_sum += bw.emit_prob[n * M + m];
            }
        }

        // the row emit_prob[n*M] must sum to 1.0
        for (size_t m = 0; m < M; m++) {
            bw.emit_prob[n*M + m] /= emit_sum;
        }
    }

    // fixed observation (can be changed to e.g. all 1 for verification)
    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            // % T would be wrong, because the observation sequence over time 0 <= t < T
            // represents observations that access the emission states
            // emit_prob[n][observations[k][t]] in 0 <= m < M,
            // which is a categorical random variable
            bw.observations[k*T + t] = rand() % M;
        }
    }
}

bool check_and_verify(const BWdata& bw) {
    const size_t N = bw.N;
    const size_t M = bw.M;

    size_t errors;
    double init_sum;
    double trans_sum;
    double emit_sum;
    bool success = true;

    // check if the initial distribution sums to 1.0
    init_sum = 0.0;
    printf("\n");
    for (size_t n = 0; n < N; n++) init_sum += bw.init_prob[n];
    if ( fabs(init_sum - 1.0) < EPSILON ) PRINT_PASSED("init_prob sums to 1.0");
    else PRINT_VIOLATION("init_prob sums to %lf", 1L, init_sum);

    // check if the rows of the transition distribution sums to 1.0
    errors = 0;
    for (size_t n0 = 0; n0 < N; n0++) {
        trans_sum = 0.0;
        for (size_t n1 = 0; n1 < N; n1++) {
            trans_sum += bw.trans_prob[n0*N + n1];
        }
        if (!(fabs(trans_sum - 1.0) < EPSILON)) {
            errors++;
            PRINT_FAIL("trans_prob[%zu] sums to %lf", n0, trans_sum);
        }
    }
    if (errors > 0) {
        PRINT_VIOLATION("of rows in trans_prob that do not sum to 1.0", errors);
        success = false;
    } else {
        PRINT_PASSED("trans_prob rows sum to 1.0");
    }

    // check if the rows of the emission distribution sums to 1.0
    errors = 0;
    for (size_t n = 0; n < N; n++) {
        emit_sum = 0.0;
        for (size_t m = 0; m < M; m++) {
            emit_sum += bw.emit_prob[n*M + m];
        }
        if (!(fabs(emit_sum - 1.0) < EPSILON)) {
            errors++;
            PRINT_FAIL("emit_prob[%zu] sums to %lf", n, emit_sum);
        }
    }
    if (errors > 0) {
        PRINT_VIOLATION("of rows in emit_prob that do not sum to 1.0", errors);
        success = false;
    } else {
        PRINT_PASSED("emit_prob rows sum to 1.0");
    }

    // check the negative log likelihood sequence for monotonicity
    errors = 0;
    for (size_t iterations = 1; iterations < bw.max_iterations; iterations++) {
        double old_nll = bw.neg_log_likelihoods[iterations-1];
        double new_nll = bw.neg_log_likelihoods[iterations];
        // Note that we We want old_nll >= new_nll,
        // because we want to minimize the negative log likelihood.
        // Hence, we want to throw an error if and only if old_nll < new_nll.
        // Therefore, we need the EPSILON here to account for numerical errors of small numbers.
        // (we always operate on the scale where -infinity < log(x) <= 0, i.e. 0 < x <= 1, due to x being a probability)
        if (old_nll < new_nll - EPSILON) {
            errors++;
            printf("[%zu]\t%lf\t > \t%lf \t(old nll < new nll)\n", iterations, old_nll, new_nll);
        }
    }
    if (errors > 0){
        PRINT_VIOLATION("of the monotonicity of the negative log likelihood", errors);
        success = false;
    } else {
        PRINT_PASSED("monotonocity of the negative log likelihood");
    }

    return success;
}

void print_states(const BWdata& bw) {
    const size_t N = bw.N;
    const size_t M = bw.M;

    printf("\nInitialization probabilities:\n");
    for(size_t n = 0; n < N; n++) {
        printf("Pr[X_1 = %zu] = %f\n", n, bw.init_prob[n]);
    }

    printf("\nTransition probabilities:\n");
    for(size_t n0 = 0; n0 < N; n0++) {
        for(size_t n1 = 0; n1 < N; n1++) {
            printf("Pr[X_t = %zu | X_(t-1) = %zu ] = %f\n", n1, n0, bw.trans_prob[n0*N + n1]);
        }
    }

    printf("\nEmission probabilities:\n");
    for(size_t n = 0; n < N; n++) {
        for(size_t m = 0; m < M; m++) {
            printf("Pr[Y_t = %zu | X_t = %zu] = %f\n", m, n, bw.emit_prob[n*M + m]);
        }
    }
    printf("\n");
}

void print_BWdata(const BWdata& bw) {

    printf("\nK %zu, N %zu, M %zu, T %zu, max_iterations %zu\n", bw.K, bw.N, bw.M, bw.T, bw.max_iterations);

    printf("\nObservation Data (tip: shouldn't change after initialization):\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            printf("obs[k = %zu][t = %zu] = %zu\n", k, t, bw.observations[k*bw.T + t]);
        }
    }

    print_states(bw); // prints bw.init_prob, bw.trans_prob and bw.emit_prob

    printf("Negative Log Likelihoods (tip: should change once per iteration)\n");
    for (size_t it = 0; it < bw.max_iterations; it++) {
        printf("NLL[it = %zu] = %f\n", it, bw.neg_log_likelihoods[it]);
    }

    printf("\n(tip: Should only change during the forward_step):\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            printf("c_norm[k = %zu][t = %zu] = %f\n", k, t, bw.c_norm[k*bw.T + t]);
        }
    }

    printf("\n(tip: Should only change during the forward_step)\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("alpha[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.alpha[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    printf("\ntip: Can be NaNs, overflow, underflow or vanish to zero (that's why we use scaling)\n");
    for (size_t k = 0; k < bw.K; k++) {
        double C_t = 1.0;
        for (size_t t = 0; t < bw.T; t++) {
            C_t *= bw.c_norm[k*bw.T + t];
            for (size_t n = 0; n < bw.N; n++) {
                printf("DE-SCALEDalpha[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.alpha[(k*bw.T + t)*bw.N + n]/C_t);
            }
        }
    }

    printf("\n(tip: Should only change during the backward_step)\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("beta[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.beta[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    printf("\ntip: Can be NaNs, overflow, underflow or vanish to zero (that's why we use scaling)\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            double D_t = 1.0;
            for (size_t tt = t; tt < bw.T; tt++) {
                D_t *= bw.c_norm[k*bw.T + tt];
            }
            for (size_t n = 0; n < bw.N; n++) {
                printf("DE-SCALEDbeta[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.beta[(k*bw.T + t)*bw.N + n]/D_t);
            }
        }
    }

    //printf("\nggamma:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("ggamma[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.ggamma[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    //printf("\ngamma_sum:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            printf("gamma_sum[k = %zu][n = %zu] = %f\n", k, n, bw.gamma_sum[k*bw.N + n]);
        }
    }

    //printf("\nsigma:\n");
    for(size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                for(size_t n1 = 0; n1 < bw.N; n1++) {
                    printf("sigma[k = %zu][t = %zu][n0 = %zu][n1 = %zu] = %f\n", k, t, n0, n1, bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1]);
                }
            }
        }
    }

    //printf("\nsigma_sum:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                printf("sigma_sum[k = %zu][n0 = %zu][n1 = %zu] = %f\n", k, n0, n1, bw.sigma_sum[(k*bw.N + n0)*bw.N + n1]);
            }
        }
    }

}

void transpose_matrix(double* output, const double* input, const size_t N, const size_t M) {
    for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
            // this can't be really vectorized (maybe combined with 4x4 blocking)
            output[m*N + n] = input[n*M + m];
        }
    }
}

/**
 * Helper function that compares the probabilities of the BWdata struct. 
 *
 * Returns:
 * - 0 if the probability model matches
 * - (-1) if the constants (K, N, M, T, max_iterations) do not match (as comparing does not make any sense then)
 * - 'the number of errors that were found while comparing' in any other case
 */
size_t get_BWdata_probabilities_differences(const BWdata& bw1, const BWdata& bw2) {
    const size_t N = bw1.N;
    const size_t M = bw1.M;

    size_t errors_local = 0;
    size_t errors_total = 0;

    for(size_t n = 0; n < N; n++) {
        const size_t index = n;
        const double err_abs_diff = fabs(bw1.init_prob[index] - bw2.init_prob[index]);
        if (!(err_abs_diff < EPSILON)) {
            errors_local++;
        }
    }

    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in init_prob!\n", errors_local);
    }

    errors_total += errors_local;
    errors_local = 0;

    for(size_t n0 = 0; n0 < N; n0++) {
        for(size_t n1 = 0; n1 < N; n1++) {
            const size_t index = n0*N + n1;
            const double err_abs_diff = fabs(bw1.trans_prob[index] - bw2.trans_prob[index]);
            if (!(err_abs_diff < EPSILON)) {
                errors_local++;
            }
        }
    }

    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in trans_prob!\n", errors_local);
    }


    errors_total += errors_local;
    errors_local = 0;

    for(size_t n = 0; n < N; n++) {
        for(size_t m = 0; m < M; m++) {
            const size_t index = n*M + m;
            const double err_abs_diff = fabs(bw1.emit_prob[index] - bw2.emit_prob[index]);
            if (!(err_abs_diff < EPSILON)) {
                errors_local++;
            }
        }
    }

    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in emit_prob!\n", errors_local);
    }

    errors_total += errors_local;

    return errors_total;
}

bool is_BWdata_equal_only_probabilities(const BWdata& bw1, const BWdata& bw2) {
    if (bw1.K != bw2.K) {
        PRINT_BWDATA_MISSMATCH("K1 = %zu is not %zu = K2\n", bw1.K, bw2.K);
        return false;
    }

    if (bw1.N != bw2.N) {
        PRINT_BWDATA_MISSMATCH(" N1 = %zu is not %zu = N2\n", bw1.N, bw2.N);
        return false;
    }

    if (bw1.M != bw2.M) {
        PRINT_BWDATA_MISSMATCH("M1 = %zu is not %zu = M2\n", bw1.M, bw2.M);
        return false;
    }

    if (bw1.T != bw2.T) {
        PRINT_BWDATA_MISSMATCH("T1 = %zu is not %zu = T2\n", bw1.T, bw2.T);
        return false;
    }

    if (bw1.max_iterations != bw2.max_iterations) {
        PRINT_BWDATA_MISSMATCH("maxIterations1 = %zu is not %zu = maxIterations2\n",
            bw1.max_iterations, bw2.max_iterations
        );
        return false;
    }
    size_t errors_total = get_BWdata_probabilities_differences(bw1, bw2);
    if (errors_total > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in total!\n", errors_total);
    } else {
        printf("\x1b[1;32mBWdata IS equal:\x1b[0m All Probabilities Match!\n");
    }

    return !errors_total;
}

bool is_BWdata_equal(const BWdata& bw1, const BWdata& bw2) {
    if (bw1.K != bw2.K) {
        PRINT_BWDATA_MISSMATCH("K1 = %zu is not %zu = K2\n", bw1.K, bw2.K);
        return false;
    }

    if (bw1.N != bw2.N) {
        PRINT_BWDATA_MISSMATCH(" N1 = %zu is not %zu = N2\n", bw1.N, bw2.N);
        return false;
    }

    if (bw1.M != bw2.M) {
        PRINT_BWDATA_MISSMATCH("M1 = %zu is not %zu = M2\n", bw1.M, bw2.M);
        return false;
    }

    if (bw1.T != bw2.T) {
        PRINT_BWDATA_MISSMATCH("T1 = %zu is not %zu = T2\n", bw1.T, bw2.T);
        return false;
    }

    if (bw1.max_iterations != bw2.max_iterations) {
        PRINT_BWDATA_MISSMATCH("maxIterations1 = %zu is not %zu = maxIterations2\n",
            bw1.max_iterations, bw2.max_iterations
        );
        return false;
    }
    size_t errors_total = get_BWdata_probabilities_differences(bw1, bw2);
    size_t errors_local = 0;

    const size_t K = bw1.K;
    const size_t N = bw1.N;
    const size_t T = bw1.T;
    const size_t max_iterations = bw1.max_iterations;

    for (size_t it = 0; it < max_iterations; it++) {
        const size_t index = it;
        const double err_abs_diff = fabs(bw1.neg_log_likelihoods[index] - bw2.neg_log_likelihoods[index]);
        if (!(err_abs_diff < EPSILON)) {
            errors_local++;
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in neg_log_likelihoods!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            const size_t index = k*T + t;
            const double err_abs_diff = fabs(bw1.c_norm[index] - bw2.c_norm[index]);
            if (!(err_abs_diff < EPSILON)) {
                errors_local++;
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in c_norm!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n = 0; n < N; n++) {
                const size_t index = (k*T + t)*N + n;
                const double err_abs_diff = fabs(bw1.alpha[index] - bw2.alpha[index]);
                if (!(err_abs_diff < EPSILON)) {
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in alpha!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n = 0; n < N; n++) {
                const size_t index = (k*T + t)*N + n;
                const double err_abs_diff = fabs(bw1.beta[index] - bw2.beta[index]);
                if (!(err_abs_diff < EPSILON)) {
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in beta!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n = 0; n < N; n++) {
                const size_t index = (k*T + t)*N + n;
                const double err_abs_diff = fabs(bw1.ggamma[index] - bw2.ggamma[index]);
                if (!(err_abs_diff < EPSILON)) {
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in ggamma!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            const size_t index = k*N + n;
            const double err_abs_diff = fabs(bw1.gamma_sum[index] - bw2.gamma_sum[index]);
            if (!(err_abs_diff < EPSILON)) {
                errors_local++;
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in gamma_sum!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for(size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n0 = 0; n0 < N; n0++) {
                for(size_t n1 = 0; n1 < N; n1++) {
                    const size_t index =((k*T + t)*N + n0)*N + n1;
                    const double err_abs_diff = fabs(bw1.sigma[index] - bw2.sigma[index]);
                    if (!(err_abs_diff < EPSILON)) {
                        errors_local++;
                    }
                }
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in sigma!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t n0 = 0; n0 < N; n0++) {
            for (size_t n1 = 0; n1 < N; n1++) {
                const size_t index = (k*N + n0)*N + n1;
                const double err_abs_diff = fabs(bw1.sigma_sum[index] - bw2.sigma_sum[index]);
                if (!(err_abs_diff < EPSILON)) {
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in sigma_sum!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    if (errors_total > 0) {
        PRINT_BWDATA_MISSMATCH("[%zu] errors in total!\n", errors_total);
    } else {
        printf("\x1b[1;32mBWdata IS equal:\x1b[0m Everything Matches!\n");
    }

    return (!errors_total);
}
