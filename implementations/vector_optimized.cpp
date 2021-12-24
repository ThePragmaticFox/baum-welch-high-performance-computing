/*
    Vectorized implementation
    Using AVX2 and FMA intrinsics, optimize as best as possible!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

// nice printf template
//printf("\nggamma[(k*bw.T + 0)*bw.N + n] = %f = %f [Address: %p]\n",
//    *(bw.ggamma + index + 0), bw.ggamma[index + 0], (bw.ggamma + index + 0)
//); fflush(0);

#include <cmath>
#include <cstring>
#include "../common.h"

static void forward_step(const BWdata& bw);
static void backward_step(const BWdata& bw);
static void compute_gamma(const BWdata& bw);
static void compute_sigma(const BWdata& bw);
static void update_init_prob(const BWdata& bw);
static void update_trans_prob(const BWdata& bw);
static void update_emit_prob(const BWdata& bw);
static size_t comp_bw_vector_optimized(const BWdata& bw);

REGISTER_FUNCTION(comp_bw_vector_optimized, "vector_optimized", "Vector Optimized: AVX2 & FMA");


/* BEGIN DECLARE HELPER STUFF */

static void transpose_matrix(double* output, const double* input, const size_t N, const size_t M);
static void rotate_indices_left(double* output, const double* input, const size_t N, const size_t K, const size_t T);
//static void rotate_somehow_last_t_index_gets_value_0_important(double* output, const double* input, const size_t K, const size_t N, const size_t M, const size_t T);

static __m256d _mm256_stuff_pd(const __m256d incr_vec, const double* input_array, const size_t index_0, const size_t index_1, const size_t index_2, const size_t index_3);
static __m256d _mm256_sumFourRowsIntoOneCol_pd(const __m256d row_0, const __m256d row_1, const __m256d row_2, const __m256d row_3);

// forward and backward pass are recursively dependent on T
#define STRIDE_LAYER_T_RECURSIVE 1
// note, though, the computation of trans_prob requires
// shifting and the computation of emit_prob requires masking
#define STRIDE_LAYER_T_NON_RECURSIVE 4
// all other loops are fully independent
#define STRIDE_LAYER_N 4
#define STRIDE_LAYER_M 4
#define STRIDE_LAYER_K 4

// local globals (heh)
static double* ggamma_N_K_T;
static double* helper_4_doubles;
static double* emit_prob_transpose;
static double* trans_prob_transpose;
static double* observations_double_array;
static const __m256d ones = _mm256_set1_pd(1.0);
static const __m256d zeros = _mm256_setzero_pd();

/* END DECLARE HELPER STUFF */


size_t comp_bw_vector_optimized(const BWdata& bw){

    /* BEGIN INIT HELPER STUFF */

    helper_4_doubles = (double *)aligned_alloc(32, 4*sizeof(double));

    emit_prob_transpose = (double *)aligned_alloc(32, bw.N*bw.M*sizeof(double));
    trans_prob_transpose = (double *)aligned_alloc(32, bw.N*bw.N*sizeof(double));

    ggamma_N_K_T = (double *)aligned_alloc(32, bw.N*bw.K*bw.T*sizeof(double));

    // AVX doesn't play nice with "size_t"
    // currently affects only update_emit_prob
    // quadratic overhead: relatively negligible, but fixing would be dope
    observations_double_array = (double *)aligned_alloc(32, bw.K*bw.T*sizeof(double));
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            observations_double_array[k*bw.T + t] = (double) bw.observations[k*bw.T + t];
        }
    }

    /* END INIT HELPER STUFF */

    size_t itera = 0;
    size_t res = 0;
    double neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;

    // run for all iterations
    for (size_t iter = 0; iter < bw.max_iterations; iter++) {
        itera++;

        // this must be here (before the backward_step)
        transpose_matrix(emit_prob_transpose, bw.emit_prob, bw.N, bw.M); // actually worth

        // this must be here (before the forward_step)
        transpose_matrix(trans_prob_transpose, bw.trans_prob, bw.N, bw.N); // actually worth

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        compute_sigma(bw);
        update_init_prob(bw);
        update_trans_prob(bw);

        // used for update_emit_prob
        // needs to be done after compute_gamma
        rotate_indices_left(ggamma_N_K_T, bw.ggamma, bw.N, bw.K, bw.T);

        update_emit_prob(bw);

        double neg_log_likelihood_sum = 0.0;
        for (size_t k = 0; k < bw.K; k++) {
            for (size_t t = 0; t < bw.T; t++) {
                // there's no AVX instruction for the logarithm ._.
                neg_log_likelihood_sum = neg_log_likelihood_sum + log(bw.c_norm[k*bw.T + t]);
            }
        }
        bw.neg_log_likelihoods[iter] = neg_log_likelihood_sum;

        if (first && iter > 0 && fabs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < EPSILON){
            first = false;
            res = itera;
        }

        neg_log_likelihood_sum_old = neg_log_likelihood_sum;

    }

    /* BEGIN FREEING HELPER STUFF */

    free(helper_4_doubles);

    free(emit_prob_transpose);
    free(trans_prob_transpose);

    free(ggamma_N_K_T);

    free(observations_double_array);

    /* END FREEING HELPER STUFF */

    return res;
}


/* BEGIN IMPLEMENTING HELPER STUFF */


inline __m256d _mm256_stuff_pd(const __m256d incr_vec, const double* input_array, const size_t index_0, const size_t index_1, const size_t index_2, const size_t index_3) {
    return _mm256_add_pd(incr_vec, _mm256_set_pd(input_array[index_0], input_array[index_1], input_array[index_2], input_array[index_3]));
}


inline __m256d _mm256_sumFourRowsIntoOneCol_pd(const __m256d row_0, const __m256d row_1, const __m256d row_2, const __m256d row_3) {
    const __m256d tmp0 = _mm256_hadd_pd(row_0, row_1);
    const __m256d tmp1 = _mm256_hadd_pd(row_2, row_3);
    const __m256d tmp2 = _mm256_blend_pd(tmp0, tmp1, 0b1100);
    const __m256d tmp3 = _mm256_permute2f128_pd(tmp0, tmp1, 0b00100001);
    const __m256d tmp4 = _mm256_add_pd(tmp2, tmp3);
    return tmp4;
}


inline void transpose_matrix(double* output, const double* input, const size_t N, const size_t M) {
    for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
            // this can't be really vectorized (maybe combined with 4x4 blocking)
            output[m*N + n] = input[n*M + m];
        }
    }
}


inline void rotate_indices_left(double* output, const double* input, const size_t N, const size_t K, const size_t T) {
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t t = 0; t < T; t++) {
                // not really vectorizable afaik
                output[(n*K + k)*T + t] = input[(k*T + t)*N + n];
            }
        }
    }
}


/* END IMPLEMENTING HELPER STUFF */


inline void forward_step(const BWdata& bw) {

    // very tedious to vectorize; T is recursively dependent
    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        __m256d vec_c_norm = zeros;

        // t = 0, base case
        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {

            const __m256d vec_init_prob = _mm256_load_pd(bw.init_prob + n);

            const __m256d vec_emit_prob_kp0 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 0)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 0)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 0)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 0)*bw.T + 0]]
            );

            const __m256d vec_emit_prob_kp1 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 1)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 1)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 1)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 1)*bw.T + 0]]
            );

            const __m256d vec_emit_prob_kp2 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 2)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 2)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 2)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 2)*bw.T + 0]]
            );

            const __m256d vec_emit_prob_kp3 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 3)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 3)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 3)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 3)*bw.T + 0]]
            );

            const __m256d vec_kp0 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp0);
            const __m256d vec_kp1 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp1);
            const __m256d vec_kp2 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp2);
            const __m256d vec_kp3 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp3);

            const __m256d sum_0_0 = _mm256_hadd_pd(vec_kp0, vec_kp1);
            const __m256d sum_0_1 = _mm256_hadd_pd(vec_kp2, vec_kp3);
            const __m256d sum_0_2 = _mm256_blend_pd(sum_0_0, sum_0_1, 0b1100);
            const __m256d sum_0_3 = _mm256_permute2f128_pd(sum_0_0, sum_0_1, 0b00100001);
            const __m256d sum_0_4 = _mm256_add_pd(sum_0_2, sum_0_3);
            vec_c_norm = _mm256_add_pd(vec_c_norm, sum_0_4);

            _mm256_store_pd((bw.alpha + ((k + 0)*bw.T + 0)*bw.N + n), vec_kp0);
            _mm256_store_pd((bw.alpha + ((k + 1)*bw.T + 0)*bw.N + n), vec_kp1);
            _mm256_store_pd((bw.alpha + ((k + 2)*bw.T + 0)*bw.N + n), vec_kp2);
            _mm256_store_pd((bw.alpha + ((k + 3)*bw.T + 0)*bw.N + n), vec_kp3);
        }

        vec_c_norm = _mm256_div_pd(ones, vec_c_norm);
        _mm256_store_pd(helper_4_doubles, vec_c_norm);

        const __m256d vec_c_norm_kp0 = _mm256_set1_pd(helper_4_doubles[0]);
        const __m256d vec_c_norm_kp1 = _mm256_set1_pd(helper_4_doubles[1]);
        const __m256d vec_c_norm_kp2 = _mm256_set1_pd(helper_4_doubles[2]);
        const __m256d vec_c_norm_kp3 = _mm256_set1_pd(helper_4_doubles[3]);

        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N){

            double* index_kp0 = bw.alpha + (((k + 0)*bw.T + 0)*bw.N + n);
            double* index_kp1 = bw.alpha + (((k + 1)*bw.T + 0)*bw.N + n);
            double* index_kp2 = bw.alpha + (((k + 2)*bw.T + 0)*bw.N + n);
            double* index_kp3 = bw.alpha + (((k + 3)*bw.T + 0)*bw.N + n);

            const __m256d vec_alpha_kp0 = _mm256_load_pd(index_kp0);
            const __m256d vec_alpha_kp1 = _mm256_load_pd(index_kp1);
            const __m256d vec_alpha_kp2 = _mm256_load_pd(index_kp2);
            const __m256d vec_alpha_kp3 = _mm256_load_pd(index_kp3);

            _mm256_store_pd(index_kp0, _mm256_mul_pd(vec_alpha_kp0, vec_c_norm_kp0));
            _mm256_store_pd(index_kp1, _mm256_mul_pd(vec_alpha_kp1, vec_c_norm_kp1));
            _mm256_store_pd(index_kp2, _mm256_mul_pd(vec_alpha_kp2, vec_c_norm_kp2));
            _mm256_store_pd(index_kp3, _mm256_mul_pd(vec_alpha_kp3, vec_c_norm_kp3));
        }

        bw.c_norm[(k + 0)*bw.T + 0] = helper_4_doubles[0];
        bw.c_norm[(k + 1)*bw.T + 0] = helper_4_doubles[1];
        bw.c_norm[(k + 2)*bw.T + 0] = helper_4_doubles[2];
        bw.c_norm[(k + 3)*bw.T + 0] = helper_4_doubles[3];

        // recursion step
        for (size_t t = 1; t < bw.T; t += STRIDE_LAYER_T_RECURSIVE) {

            vec_c_norm = zeros;

            for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

                const __m256d vec_emit_prob_kp0 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 0)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 0)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 0)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 0)*bw.T + t]]
                );

                const __m256d vec_emit_prob_kp1 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 1)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 1)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 1)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 1)*bw.T + t]]
                );

                const __m256d vec_emit_prob_kp2 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 2)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 2)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 2)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 2)*bw.T + t]]
                );

                const __m256d vec_emit_prob_kp3 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 3)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 3)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 3)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 3)*bw.T + t]]
                );

                __m256d vec_trans_prob_sum_np0_kp0 = zeros;
                __m256d vec_trans_prob_sum_np1_kp0 = zeros;
                __m256d vec_trans_prob_sum_np2_kp0 = zeros;
                __m256d vec_trans_prob_sum_np3_kp0 = zeros;

                __m256d vec_trans_prob_sum_np0_kp1 = zeros;
                __m256d vec_trans_prob_sum_np1_kp1 = zeros;
                __m256d vec_trans_prob_sum_np2_kp1 = zeros;
                __m256d vec_trans_prob_sum_np3_kp1 = zeros;

                __m256d vec_trans_prob_sum_np0_kp2 = zeros;
                __m256d vec_trans_prob_sum_np1_kp2 = zeros;
                __m256d vec_trans_prob_sum_np2_kp2 = zeros;
                __m256d vec_trans_prob_sum_np3_kp2 = zeros;

                __m256d vec_trans_prob_sum_np0_kp3 = zeros;
                __m256d vec_trans_prob_sum_np1_kp3 = zeros;
                __m256d vec_trans_prob_sum_np2_kp3 = zeros;
                __m256d vec_trans_prob_sum_np3_kp3 = zeros;

                for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                    const double* index_alpha_kp0 = bw.alpha + ((k + 0)*bw.T + (t-1))*bw.N + n1;
                    const double* index_alpha_kp1 = bw.alpha + ((k + 1)*bw.T + (t-1))*bw.N + n1;
                    const double* index_alpha_kp2 = bw.alpha + ((k + 2)*bw.T + (t-1))*bw.N + n1;
                    const double* index_alpha_kp3 = bw.alpha + ((k + 3)*bw.T + (t-1))*bw.N + n1;

                    const __m256d vec_trans_np0 = _mm256_load_pd(trans_prob_transpose + ((n0 + 0)*bw.N + n1));
                    const __m256d vec_trans_np1 = _mm256_load_pd(trans_prob_transpose + ((n0 + 1)*bw.N + n1));
                    const __m256d vec_trans_np2 = _mm256_load_pd(trans_prob_transpose + ((n0 + 2)*bw.N + n1));
                    const __m256d vec_trans_np3 = _mm256_load_pd(trans_prob_transpose + ((n0 + 3)*bw.N + n1));

                    const __m256d vec_alpha_kp0 = _mm256_load_pd(index_alpha_kp0);
                    const __m256d vec_alpha_kp1 = _mm256_load_pd(index_alpha_kp1);
                    const __m256d vec_alpha_kp2 = _mm256_load_pd(index_alpha_kp2);
                    const __m256d vec_alpha_kp3 = _mm256_load_pd(index_alpha_kp3);

                    vec_trans_prob_sum_np0_kp0 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp0, vec_trans_prob_sum_np0_kp0);
                    vec_trans_prob_sum_np1_kp0 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp0, vec_trans_prob_sum_np1_kp0);
                    vec_trans_prob_sum_np2_kp0 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp0, vec_trans_prob_sum_np2_kp0);
                    vec_trans_prob_sum_np3_kp0 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp0, vec_trans_prob_sum_np3_kp0);

                    vec_trans_prob_sum_np0_kp1 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp1, vec_trans_prob_sum_np0_kp1);
                    vec_trans_prob_sum_np1_kp1 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp1, vec_trans_prob_sum_np1_kp1);
                    vec_trans_prob_sum_np2_kp1 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp1, vec_trans_prob_sum_np2_kp1);
                    vec_trans_prob_sum_np3_kp1 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp1, vec_trans_prob_sum_np3_kp1);

                    vec_trans_prob_sum_np0_kp2 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp2, vec_trans_prob_sum_np0_kp2);
                    vec_trans_prob_sum_np1_kp2 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp2, vec_trans_prob_sum_np1_kp2);
                    vec_trans_prob_sum_np2_kp2 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp2, vec_trans_prob_sum_np2_kp2);
                    vec_trans_prob_sum_np3_kp2 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp2, vec_trans_prob_sum_np3_kp2);

                    vec_trans_prob_sum_np0_kp3 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp3, vec_trans_prob_sum_np0_kp3);
                    vec_trans_prob_sum_np1_kp3 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp3, vec_trans_prob_sum_np1_kp3);
                    vec_trans_prob_sum_np2_kp3 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp3, vec_trans_prob_sum_np2_kp3);
                    vec_trans_prob_sum_np3_kp3 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp3, vec_trans_prob_sum_np3_kp3);

                }

                const __m256d a0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp0, vec_trans_prob_sum_np1_kp0);
                const __m256d a1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp0, vec_trans_prob_sum_np3_kp0);
                const __m256d a2 = _mm256_blend_pd(a0, a1, 0b1100);
                const __m256d a3 = _mm256_permute2f128_pd(a0, a1, 0b00100001);
                const __m256d a4 = _mm256_add_pd(a2, a3);

                const __m256d b0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp1, vec_trans_prob_sum_np1_kp1);
                const __m256d b1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp1, vec_trans_prob_sum_np3_kp1);
                const __m256d b2 = _mm256_blend_pd(b0, b1, 0b1100);
                const __m256d b3 = _mm256_permute2f128_pd(b0, b1, 0b00100001);
                const __m256d b4 = _mm256_add_pd(b2, b3);

                const __m256d c0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp2, vec_trans_prob_sum_np1_kp2);
                const __m256d c1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp2, vec_trans_prob_sum_np3_kp2);
                const __m256d c2 = _mm256_blend_pd(c0, c1, 0b1100);
                const __m256d c3 = _mm256_permute2f128_pd(c0, c1, 0b00100001);
                const __m256d c4 = _mm256_add_pd(c2, c3);

                const __m256d d0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp3, vec_trans_prob_sum_np1_kp3);
                const __m256d d1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp3, vec_trans_prob_sum_np3_kp3);
                const __m256d d2 = _mm256_blend_pd(d0, d1, 0b1100);
                const __m256d d3 = _mm256_permute2f128_pd(d0, d1, 0b00100001);
                const __m256d d4 = _mm256_add_pd(d2, d3);

                const __m256d vec_kp0 = _mm256_mul_pd(a4, vec_emit_prob_kp0);
                const __m256d vec_kp1 = _mm256_mul_pd(b4, vec_emit_prob_kp1);
                const __m256d vec_kp2 = _mm256_mul_pd(c4, vec_emit_prob_kp2);
                const __m256d vec_kp3 = _mm256_mul_pd(d4, vec_emit_prob_kp3);

                const __m256d sum_0_0 = _mm256_hadd_pd(vec_kp0, vec_kp1);
                const __m256d sum_0_1 = _mm256_hadd_pd(vec_kp2, vec_kp3);
                const __m256d sum_0_2 = _mm256_blend_pd(sum_0_0, sum_0_1, 0b1100);
                const __m256d sum_0_3 = _mm256_permute2f128_pd(sum_0_0, sum_0_1, 0b00100001);
                const __m256d sum_0_4 = _mm256_add_pd(sum_0_2, sum_0_3);
                vec_c_norm = _mm256_add_pd(vec_c_norm, sum_0_4);

                _mm256_store_pd((bw.alpha + ((k + 0)*bw.T + t)*bw.N + n0), vec_kp0);
                _mm256_store_pd((bw.alpha + ((k + 1)*bw.T + t)*bw.N + n0), vec_kp1);
                _mm256_store_pd((bw.alpha + ((k + 2)*bw.T + t)*bw.N + n0), vec_kp2);
                _mm256_store_pd((bw.alpha + ((k + 3)*bw.T + t)*bw.N + n0), vec_kp3);
            }

            vec_c_norm = _mm256_div_pd(ones, vec_c_norm);
            _mm256_store_pd(helper_4_doubles, vec_c_norm);

            const __m256d vec_c_norm_kp0 = _mm256_set1_pd(helper_4_doubles[0]);
            const __m256d vec_c_norm_kp1 = _mm256_set1_pd(helper_4_doubles[1]);
            const __m256d vec_c_norm_kp2 = _mm256_set1_pd(helper_4_doubles[2]);
            const __m256d vec_c_norm_kp3 = _mm256_set1_pd(helper_4_doubles[3]);

            for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N){

                double* index_kp0 = bw.alpha + (((k + 0)*bw.T + t)*bw.N + n);
                double* index_kp1 = bw.alpha + (((k + 1)*bw.T + t)*bw.N + n);
                double* index_kp2 = bw.alpha + (((k + 2)*bw.T + t)*bw.N + n);
                double* index_kp3 = bw.alpha + (((k + 3)*bw.T + t)*bw.N + n);

                const __m256d vec_alpha_kp0 = _mm256_load_pd(index_kp0);
                const __m256d vec_alpha_kp1 = _mm256_load_pd(index_kp1);
                const __m256d vec_alpha_kp2 = _mm256_load_pd(index_kp2);
                const __m256d vec_alpha_kp3 = _mm256_load_pd(index_kp3);

                _mm256_store_pd(index_kp0, _mm256_mul_pd(vec_alpha_kp0, vec_c_norm_kp0));
                _mm256_store_pd(index_kp1, _mm256_mul_pd(vec_alpha_kp1, vec_c_norm_kp1));
                _mm256_store_pd(index_kp2, _mm256_mul_pd(vec_alpha_kp2, vec_c_norm_kp2));
                _mm256_store_pd(index_kp3, _mm256_mul_pd(vec_alpha_kp3, vec_c_norm_kp3));
            }

            bw.c_norm[(k + 0)*bw.T + t] = helper_4_doubles[0];
            bw.c_norm[(k + 1)*bw.T + t] = helper_4_doubles[1];
            bw.c_norm[(k + 2)*bw.T + t] = helper_4_doubles[2];
            bw.c_norm[(k + 3)*bw.T + t] = helper_4_doubles[3];

        }

    }
}


inline void backward_step(const BWdata& bw) {

    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        // t = bw.T, base case
        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {
            _mm256_store_pd((bw.beta + (((k + 0)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + (bw.T-1)]));
            _mm256_store_pd((bw.beta + (((k + 1)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + (bw.T-1)]));
            _mm256_store_pd((bw.beta + (((k + 2)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + (bw.T-1)]));
            _mm256_store_pd((bw.beta + (((k + 3)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + (bw.T-1)]));
        }

        // recursion step
        for (int t = bw.T-2; t >= 0; t -= STRIDE_LAYER_T_RECURSIVE) {

            const size_t index_emitobs_kp0 = bw.observations[(k + 0)*bw.T + (t+1)];
            const size_t index_emitobs_kp1 = bw.observations[(k + 1)*bw.T + (t+1)];
            const size_t index_emitobs_kp2 = bw.observations[(k + 2)*bw.T + (t+1)];
            const size_t index_emitobs_kp3 = bw.observations[(k + 3)*bw.T + (t+1)];

            for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

                __m256d vec_beta_tmp_np0_kp0 = zeros;
                __m256d vec_beta_tmp_np0_kp1 = zeros;
                __m256d vec_beta_tmp_np0_kp2 = zeros;
                __m256d vec_beta_tmp_np0_kp3 = zeros;

                __m256d vec_beta_tmp_np1_kp0 = zeros;
                __m256d vec_beta_tmp_np1_kp1 = zeros;
                __m256d vec_beta_tmp_np1_kp2 = zeros;
                __m256d vec_beta_tmp_np1_kp3 = zeros;

                __m256d vec_beta_tmp_np2_kp0 = zeros;
                __m256d vec_beta_tmp_np2_kp1 = zeros;
                __m256d vec_beta_tmp_np2_kp2 = zeros;
                __m256d vec_beta_tmp_np2_kp3 = zeros;

                __m256d vec_beta_tmp_np3_kp0 = zeros;
                __m256d vec_beta_tmp_np3_kp1 = zeros;
                __m256d vec_beta_tmp_np3_kp2 = zeros;
                __m256d vec_beta_tmp_np3_kp3 = zeros;

                for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                    const __m256d vec_beta_kp0 = _mm256_load_pd(bw.beta + (((k + 0)*bw.T + (t+1))*bw.N + n1));
                    const __m256d vec_beta_kp1 = _mm256_load_pd(bw.beta + (((k + 1)*bw.T + (t+1))*bw.N + n1));
                    const __m256d vec_beta_kp2 = _mm256_load_pd(bw.beta + (((k + 2)*bw.T + (t+1))*bw.N + n1));
                    const __m256d vec_beta_kp3 = _mm256_load_pd(bw.beta + (((k + 3)*bw.T + (t+1))*bw.N + n1));

                    const __m256d vec_trans_prob_np0 = _mm256_load_pd(bw.trans_prob + (n0 + 0)*bw.N + n1);
                    const __m256d vec_trans_prob_np1 = _mm256_load_pd(bw.trans_prob + (n0 + 1)*bw.N + n1);
                    const __m256d vec_trans_prob_np2 = _mm256_load_pd(bw.trans_prob + (n0 + 2)*bw.N + n1);
                    const __m256d vec_trans_prob_np3 = _mm256_load_pd(bw.trans_prob + (n0 + 3)*bw.N + n1);

                    const __m256d vec_emit_prob_kp0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0*bw.N + n1));
                    const __m256d vec_emit_prob_kp1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1*bw.N + n1));
                    const __m256d vec_emit_prob_kp2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2*bw.N + n1));
                    const __m256d vec_emit_prob_kp3 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3*bw.N + n1));

                    vec_beta_tmp_np0_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp0), vec_beta_tmp_np0_kp0);
                    vec_beta_tmp_np0_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp1), vec_beta_tmp_np0_kp1);
                    vec_beta_tmp_np0_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp2), vec_beta_tmp_np0_kp2);
                    vec_beta_tmp_np0_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp3), vec_beta_tmp_np0_kp3);

                    vec_beta_tmp_np1_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp0), vec_beta_tmp_np1_kp0);
                    vec_beta_tmp_np1_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp1), vec_beta_tmp_np1_kp1);
                    vec_beta_tmp_np1_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp2), vec_beta_tmp_np1_kp2);
                    vec_beta_tmp_np1_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp3), vec_beta_tmp_np1_kp3);

                    vec_beta_tmp_np2_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp0), vec_beta_tmp_np2_kp0);
                    vec_beta_tmp_np2_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp1), vec_beta_tmp_np2_kp1);
                    vec_beta_tmp_np2_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp2), vec_beta_tmp_np2_kp2);
                    vec_beta_tmp_np2_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp3), vec_beta_tmp_np2_kp3);

                    vec_beta_tmp_np3_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp0), vec_beta_tmp_np3_kp0);
                    vec_beta_tmp_np3_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp1), vec_beta_tmp_np3_kp1);
                    vec_beta_tmp_np3_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp2), vec_beta_tmp_np3_kp2);
                    vec_beta_tmp_np3_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp3), vec_beta_tmp_np3_kp3);

                }

                const __m256d a0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp0, vec_beta_tmp_np1_kp0);
                const __m256d a1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp0, vec_beta_tmp_np3_kp0);
                const __m256d a2 = _mm256_blend_pd(a0, a1, 0b1100);
                const __m256d a3 = _mm256_permute2f128_pd(a0, a1, 0b00100001);
                const __m256d a4 = _mm256_add_pd(a2, a3);

                const __m256d b0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp1, vec_beta_tmp_np1_kp1);
                const __m256d b1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp1, vec_beta_tmp_np3_kp1);
                const __m256d b2 = _mm256_blend_pd(b0, b1, 0b1100);
                const __m256d b3 = _mm256_permute2f128_pd(b0, b1, 0b00100001);
                const __m256d b4 = _mm256_add_pd(b2, b3);

                const __m256d c0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp2, vec_beta_tmp_np1_kp2);
                const __m256d c1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp2, vec_beta_tmp_np3_kp2);
                const __m256d c2 = _mm256_blend_pd(c0, c1, 0b1100);
                const __m256d c3 = _mm256_permute2f128_pd(c0, c1, 0b00100001);
                const __m256d c4 = _mm256_add_pd(c2, c3);

                const __m256d d0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp3, vec_beta_tmp_np1_kp3);
                const __m256d d1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp3, vec_beta_tmp_np3_kp3);
                const __m256d d2 = _mm256_blend_pd(d0, d1, 0b1100);
                const __m256d d3 = _mm256_permute2f128_pd(d0, d1, 0b00100001);
                const __m256d d4 = _mm256_add_pd(d2, d3);

                _mm256_store_pd((bw.beta + (((k + 0)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + t]), a4));
                _mm256_store_pd((bw.beta + (((k + 1)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + t]), b4));
                _mm256_store_pd((bw.beta + (((k + 2)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + t]), c4));
                _mm256_store_pd((bw.beta + (((k + 3)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + t]), d4));

            }

        }

    }

}


inline void compute_gamma(const BWdata& bw) {


    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        for (size_t t = 0; t < bw.T; t += STRIDE_LAYER_T_NON_RECURSIVE) {

            const __m256d vec_c_norm_inv_kp0_tp0 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + (t + 0)]));
            const __m256d vec_c_norm_inv_kp0_tp1 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + (t + 1)]));
            const __m256d vec_c_norm_inv_kp0_tp2 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + (t + 2)]));
            const __m256d vec_c_norm_inv_kp0_tp3 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + (t + 3)]));

            const __m256d vec_c_norm_inv_kp1_tp0 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + (t + 0)]));
            const __m256d vec_c_norm_inv_kp1_tp1 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + (t + 1)]));
            const __m256d vec_c_norm_inv_kp1_tp2 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + (t + 2)]));
            const __m256d vec_c_norm_inv_kp1_tp3 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + (t + 3)]));

            const __m256d vec_c_norm_inv_kp2_tp0 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + (t + 0)]));
            const __m256d vec_c_norm_inv_kp2_tp1 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + (t + 1)]));
            const __m256d vec_c_norm_inv_kp2_tp2 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + (t + 2)]));
            const __m256d vec_c_norm_inv_kp2_tp3 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + (t + 3)]));

            const __m256d vec_c_norm_inv_kp3_tp0 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + (t + 0)]));
            const __m256d vec_c_norm_inv_kp3_tp1 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + (t + 1)]));
            const __m256d vec_c_norm_inv_kp3_tp2 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + (t + 2)]));
            const __m256d vec_c_norm_inv_kp3_tp3 = _mm256_div_pd(ones, _mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + (t + 3)]));

            for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {
                
                const __m256d vec_alpha_kp0_tp0 = _mm256_load_pd(bw.alpha + (((k + 0)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_alpha_kp0_tp1 = _mm256_load_pd(bw.alpha + (((k + 0)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_alpha_kp0_tp2 = _mm256_load_pd(bw.alpha + (((k + 0)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_alpha_kp0_tp3 = _mm256_load_pd(bw.alpha + (((k + 0)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_alpha_kp1_tp0 = _mm256_load_pd(bw.alpha + (((k + 1)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_alpha_kp1_tp1 = _mm256_load_pd(bw.alpha + (((k + 1)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_alpha_kp1_tp2 = _mm256_load_pd(bw.alpha + (((k + 1)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_alpha_kp1_tp3 = _mm256_load_pd(bw.alpha + (((k + 1)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_alpha_kp2_tp0 = _mm256_load_pd(bw.alpha + (((k + 2)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_alpha_kp2_tp1 = _mm256_load_pd(bw.alpha + (((k + 2)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_alpha_kp2_tp2 = _mm256_load_pd(bw.alpha + (((k + 2)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_alpha_kp2_tp3 = _mm256_load_pd(bw.alpha + (((k + 2)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_alpha_kp3_tp0 = _mm256_load_pd(bw.alpha + (((k + 3)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_alpha_kp3_tp1 = _mm256_load_pd(bw.alpha + (((k + 3)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_alpha_kp3_tp2 = _mm256_load_pd(bw.alpha + (((k + 3)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_alpha_kp3_tp3 = _mm256_load_pd(bw.alpha + (((k + 3)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_beta_kp0_tp0 = _mm256_load_pd(bw.beta + (((k + 0)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_beta_kp0_tp1 = _mm256_load_pd(bw.beta + (((k + 0)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_beta_kp0_tp2 = _mm256_load_pd(bw.beta + (((k + 0)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_beta_kp0_tp3 = _mm256_load_pd(bw.beta + (((k + 0)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_beta_kp1_tp0 = _mm256_load_pd(bw.beta + (((k + 1)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_beta_kp1_tp1 = _mm256_load_pd(bw.beta + (((k + 1)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_beta_kp1_tp2 = _mm256_load_pd(bw.beta + (((k + 1)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_beta_kp1_tp3 = _mm256_load_pd(bw.beta + (((k + 1)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_beta_kp2_tp0 = _mm256_load_pd(bw.beta + (((k + 2)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_beta_kp2_tp1 = _mm256_load_pd(bw.beta + (((k + 2)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_beta_kp2_tp2 = _mm256_load_pd(bw.beta + (((k + 2)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_beta_kp2_tp3 = _mm256_load_pd(bw.beta + (((k + 2)*bw.T + (t + 3))*bw.N + n));

                const __m256d vec_beta_kp3_tp0 = _mm256_load_pd(bw.beta + (((k + 3)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_beta_kp3_tp1 = _mm256_load_pd(bw.beta + (((k + 3)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_beta_kp3_tp2 = _mm256_load_pd(bw.beta + (((k + 3)*bw.T + (t + 2))*bw.N + n));
                const __m256d vec_beta_kp3_tp3 = _mm256_load_pd(bw.beta + (((k + 3)*bw.T + (t + 3))*bw.N + n));

                const __m256d result_kp0_tp0 = _mm256_mul_pd(vec_c_norm_inv_kp0_tp0, _mm256_mul_pd(vec_alpha_kp0_tp0, vec_beta_kp0_tp0));
                const __m256d result_kp0_tp1 = _mm256_mul_pd(vec_c_norm_inv_kp0_tp1, _mm256_mul_pd(vec_alpha_kp0_tp1, vec_beta_kp0_tp1));
                const __m256d result_kp0_tp2 = _mm256_mul_pd(vec_c_norm_inv_kp0_tp2, _mm256_mul_pd(vec_alpha_kp0_tp2, vec_beta_kp0_tp2));
                const __m256d result_kp0_tp3 = _mm256_mul_pd(vec_c_norm_inv_kp0_tp3, _mm256_mul_pd(vec_alpha_kp0_tp3, vec_beta_kp0_tp3));

                const __m256d result_kp1_tp0 = _mm256_mul_pd(vec_c_norm_inv_kp1_tp0, _mm256_mul_pd(vec_alpha_kp1_tp0, vec_beta_kp1_tp0));
                const __m256d result_kp1_tp1 = _mm256_mul_pd(vec_c_norm_inv_kp1_tp1, _mm256_mul_pd(vec_alpha_kp1_tp1, vec_beta_kp1_tp1));
                const __m256d result_kp1_tp2 = _mm256_mul_pd(vec_c_norm_inv_kp1_tp2, _mm256_mul_pd(vec_alpha_kp1_tp2, vec_beta_kp1_tp2));
                const __m256d result_kp1_tp3 = _mm256_mul_pd(vec_c_norm_inv_kp1_tp3, _mm256_mul_pd(vec_alpha_kp1_tp3, vec_beta_kp1_tp3));

                const __m256d result_kp2_tp0 = _mm256_mul_pd(vec_c_norm_inv_kp2_tp0, _mm256_mul_pd(vec_alpha_kp2_tp0, vec_beta_kp2_tp0));
                const __m256d result_kp2_tp1 = _mm256_mul_pd(vec_c_norm_inv_kp2_tp1, _mm256_mul_pd(vec_alpha_kp2_tp1, vec_beta_kp2_tp1));
                const __m256d result_kp2_tp2 = _mm256_mul_pd(vec_c_norm_inv_kp2_tp2, _mm256_mul_pd(vec_alpha_kp2_tp2, vec_beta_kp2_tp2));
                const __m256d result_kp2_tp3 = _mm256_mul_pd(vec_c_norm_inv_kp2_tp3, _mm256_mul_pd(vec_alpha_kp2_tp3, vec_beta_kp2_tp3));

                const __m256d result_kp3_tp0 = _mm256_mul_pd(vec_c_norm_inv_kp3_tp0, _mm256_mul_pd(vec_alpha_kp3_tp0, vec_beta_kp3_tp0));
                const __m256d result_kp3_tp1 = _mm256_mul_pd(vec_c_norm_inv_kp3_tp1, _mm256_mul_pd(vec_alpha_kp3_tp1, vec_beta_kp3_tp1));
                const __m256d result_kp3_tp2 = _mm256_mul_pd(vec_c_norm_inv_kp3_tp2, _mm256_mul_pd(vec_alpha_kp3_tp2, vec_beta_kp3_tp2));
                const __m256d result_kp3_tp3 = _mm256_mul_pd(vec_c_norm_inv_kp3_tp3, _mm256_mul_pd(vec_alpha_kp3_tp3, vec_beta_kp3_tp3));

                _mm256_store_pd(bw.ggamma + (((k + 0)*bw.T + (t + 0))*bw.N + n), result_kp0_tp0);
                _mm256_store_pd(bw.ggamma + (((k + 0)*bw.T + (t + 1))*bw.N + n), result_kp0_tp1);
                _mm256_store_pd(bw.ggamma + (((k + 0)*bw.T + (t + 2))*bw.N + n), result_kp0_tp2);
                _mm256_store_pd(bw.ggamma + (((k + 0)*bw.T + (t + 3))*bw.N + n), result_kp0_tp3);

                _mm256_store_pd(bw.ggamma + (((k + 1)*bw.T + (t + 0))*bw.N + n), result_kp1_tp0);
                _mm256_store_pd(bw.ggamma + (((k + 1)*bw.T + (t + 1))*bw.N + n), result_kp1_tp1);
                _mm256_store_pd(bw.ggamma + (((k + 1)*bw.T + (t + 2))*bw.N + n), result_kp1_tp2);
                _mm256_store_pd(bw.ggamma + (((k + 1)*bw.T + (t + 3))*bw.N + n), result_kp1_tp3);

                _mm256_store_pd(bw.ggamma + (((k + 2)*bw.T + (t + 0))*bw.N + n), result_kp2_tp0);
                _mm256_store_pd(bw.ggamma + (((k + 2)*bw.T + (t + 1))*bw.N + n), result_kp2_tp1);
                _mm256_store_pd(bw.ggamma + (((k + 2)*bw.T + (t + 2))*bw.N + n), result_kp2_tp2);
                _mm256_store_pd(bw.ggamma + (((k + 2)*bw.T + (t + 3))*bw.N + n), result_kp2_tp3);

                _mm256_store_pd(bw.ggamma + (((k + 3)*bw.T + (t + 0))*bw.N + n), result_kp3_tp0);
                _mm256_store_pd(bw.ggamma + (((k + 3)*bw.T + (t + 1))*bw.N + n), result_kp3_tp1);
                _mm256_store_pd(bw.ggamma + (((k + 3)*bw.T + (t + 2))*bw.N + n), result_kp3_tp2);
                _mm256_store_pd(bw.ggamma + (((k + 3)*bw.T + (t + 3))*bw.N + n), result_kp3_tp3);

            }

        }

    }

    // sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob)

    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {

            __m256d vec_g_sum_kp0 = zeros;
            __m256d vec_g_sum_kp1 = zeros;
            __m256d vec_g_sum_kp2 = zeros;
            __m256d vec_g_sum_kp3 = zeros;

            for (size_t t = 0; t < bw.T-1; t += STRIDE_LAYER_T_NON_RECURSIVE) {

                const __m256d vec_ggamma_kp0_tp0 = _mm256_load_pd(bw.ggamma + (((k + 0)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_ggamma_kp0_tp1 = _mm256_load_pd(bw.ggamma + (((k + 0)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_ggamma_kp0_tp2 = _mm256_load_pd(bw.ggamma + (((k + 0)*bw.T + (t + 2))*bw.N + n));

                const __m256d vec_ggamma_kp1_tp0 = _mm256_load_pd(bw.ggamma + (((k + 1)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_ggamma_kp1_tp1 = _mm256_load_pd(bw.ggamma + (((k + 1)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_ggamma_kp1_tp2 = _mm256_load_pd(bw.ggamma + (((k + 1)*bw.T + (t + 2))*bw.N + n));

                const __m256d vec_ggamma_kp2_tp0 = _mm256_load_pd(bw.ggamma + (((k + 2)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_ggamma_kp2_tp1 = _mm256_load_pd(bw.ggamma + (((k + 2)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_ggamma_kp2_tp2 = _mm256_load_pd(bw.ggamma + (((k + 2)*bw.T + (t + 2))*bw.N + n));

                const __m256d vec_ggamma_kp3_tp0 = _mm256_load_pd(bw.ggamma + (((k + 3)*bw.T + (t + 0))*bw.N + n));
                const __m256d vec_ggamma_kp3_tp1 = _mm256_load_pd(bw.ggamma + (((k + 3)*bw.T + (t + 1))*bw.N + n));
                const __m256d vec_ggamma_kp3_tp2 = _mm256_load_pd(bw.ggamma + (((k + 3)*bw.T + (t + 2))*bw.N + n));

                __m256d vec_tmp_kp0 = _mm256_add_pd(vec_ggamma_kp0_tp0, _mm256_add_pd(vec_ggamma_kp0_tp1, vec_ggamma_kp0_tp2));
                __m256d vec_tmp_kp1 = _mm256_add_pd(vec_ggamma_kp1_tp0, _mm256_add_pd(vec_ggamma_kp1_tp1, vec_ggamma_kp1_tp2));
                __m256d vec_tmp_kp2 = _mm256_add_pd(vec_ggamma_kp2_tp0, _mm256_add_pd(vec_ggamma_kp2_tp1, vec_ggamma_kp2_tp2));
                __m256d vec_tmp_kp3 = _mm256_add_pd(vec_ggamma_kp3_tp0, _mm256_add_pd(vec_ggamma_kp3_tp1, vec_ggamma_kp3_tp2));

                if (t < bw.T - 4) {

                    const __m256d vec_ggamma_kp0_tp3 = _mm256_load_pd(bw.ggamma + (((k + 0)*bw.T + (t + 3))*bw.N + n));
                    const __m256d vec_ggamma_kp1_tp3 = _mm256_load_pd(bw.ggamma + (((k + 1)*bw.T + (t + 3))*bw.N + n));
                    const __m256d vec_ggamma_kp2_tp3 = _mm256_load_pd(bw.ggamma + (((k + 2)*bw.T + (t + 3))*bw.N + n));
                    const __m256d vec_ggamma_kp3_tp3 = _mm256_load_pd(bw.ggamma + (((k + 3)*bw.T + (t + 3))*bw.N + n));

                    vec_tmp_kp0 = _mm256_add_pd(vec_ggamma_kp0_tp3, vec_tmp_kp0);
                    vec_tmp_kp1 = _mm256_add_pd(vec_ggamma_kp1_tp3, vec_tmp_kp1);
                    vec_tmp_kp2 = _mm256_add_pd(vec_ggamma_kp2_tp3, vec_tmp_kp2);
                    vec_tmp_kp3 = _mm256_add_pd(vec_ggamma_kp3_tp3, vec_tmp_kp3);

                }

                vec_g_sum_kp0 = _mm256_add_pd(vec_tmp_kp0, vec_g_sum_kp0);
                vec_g_sum_kp1 = _mm256_add_pd(vec_tmp_kp1, vec_g_sum_kp1);
                vec_g_sum_kp2 = _mm256_add_pd(vec_tmp_kp2, vec_g_sum_kp2);
                vec_g_sum_kp3 = _mm256_add_pd(vec_tmp_kp3, vec_g_sum_kp3);
            }

            _mm256_store_pd((bw.gamma_sum + ((k + 0)*bw.N + n)), vec_g_sum_kp0);
            _mm256_store_pd((bw.gamma_sum + ((k + 1)*bw.N + n)), vec_g_sum_kp1);
            _mm256_store_pd((bw.gamma_sum + ((k + 2)*bw.N + n)), vec_g_sum_kp2);
            _mm256_store_pd((bw.gamma_sum + ((k + 3)*bw.N + n)), vec_g_sum_kp3);

        }

    }

}


inline void compute_sigma(const BWdata& bw) {

    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        const size_t kp0 = k + 0;
        const size_t kp1 = k + 1;
        const size_t kp2 = k + 2;
        const size_t kp3 = k + 3;

        for (size_t t = 0; t < bw.T-1; t += STRIDE_LAYER_T_NON_RECURSIVE) {

            const size_t tp0 = t + 0;
            const size_t tp1 = t + 1;
            const size_t tp2 = t + 2;
            const size_t tp3 = t + 3;

            const size_t tp1p0 = (t+1) + 0;
            const size_t tp1p1 = (t+1) + 1;
            const size_t tp1p2 = (t+1) + 2;
            const size_t tp1p3 = (t+1) + 3;

            const size_t index_emitobs_kp0_tp1p0 = bw.observations[kp0*bw.T + tp1p0];
            const size_t index_emitobs_kp0_tp1p1 = bw.observations[kp0*bw.T + tp1p1];
            const size_t index_emitobs_kp0_tp1p2 = bw.observations[kp0*bw.T + tp1p2];
            const size_t index_emitobs_kp0_tp1p3 = bw.observations[kp0*bw.T + tp1p3];

            const size_t index_emitobs_kp1_tp1p0 = bw.observations[kp1*bw.T + tp1p0];
            const size_t index_emitobs_kp1_tp1p1 = bw.observations[kp1*bw.T + tp1p1];
            const size_t index_emitobs_kp1_tp1p2 = bw.observations[kp1*bw.T + tp1p2];
            const size_t index_emitobs_kp1_tp1p3 = bw.observations[kp1*bw.T + tp1p3];

            const size_t index_emitobs_kp2_tp1p0 = bw.observations[kp2*bw.T + tp1p0];
            const size_t index_emitobs_kp2_tp1p1 = bw.observations[kp2*bw.T + tp1p1];
            const size_t index_emitobs_kp2_tp1p2 = bw.observations[kp2*bw.T + tp1p2];
            const size_t index_emitobs_kp2_tp1p3 = bw.observations[kp2*bw.T + tp1p3];

            const size_t index_emitobs_kp3_tp1p0 = bw.observations[kp3*bw.T + tp1p0];
            const size_t index_emitobs_kp3_tp1p1 = bw.observations[kp3*bw.T + tp1p1];
            const size_t index_emitobs_kp3_tp1p2 = bw.observations[kp3*bw.T + tp1p2];
            const size_t index_emitobs_kp3_tp1p3 = bw.observations[kp3*bw.T + tp1p3];

            if (t < bw.T - 4) {

                for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

                    const size_t n0p0 = n0 + 0;
                    const size_t n0p1 = n0 + 1;
                    const size_t n0p2 = n0 + 2;
                    const size_t n0p3 = n0 + 3;

                    const __m256d alpha_kp0_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp0_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp0_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp0_tp3_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp3)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp3_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp3)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp3_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp3)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp3_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp3)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp3_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp3)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp3_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp3)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp3_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp3)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp3_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp3)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp3_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp3)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp3_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp3)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp3_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp3)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp3_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp3)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp3_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp3)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp3_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp3)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp3_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp3)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp3_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp3)*bw.N + n0p3]);

                    for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                        const __m256d trans_n0p0 = _mm256_load_pd(bw.trans_prob + (n0p0*bw.N + n1));
                        const __m256d trans_n0p1 = _mm256_load_pd(bw.trans_prob + (n0p1*bw.N + n1));
                        const __m256d trans_n0p2 = _mm256_load_pd(bw.trans_prob + (n0p2*bw.N + n1));
                        const __m256d trans_n0p3 = _mm256_load_pd(bw.trans_prob + (n0p3*bw.N + n1));

                        const __m256d beta_kp0_tp1p0 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp0_tp1p1 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp0_tp1p2 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p2)*bw.N + n1));
                        const __m256d beta_kp0_tp1p3 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p3)*bw.N + n1));

                        const __m256d beta_kp1_tp1p0 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp1_tp1p1 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp1_tp1p2 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p2)*bw.N + n1));
                        const __m256d beta_kp1_tp1p3 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p3)*bw.N + n1));

                        const __m256d beta_kp2_tp1p0 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp2_tp1p1 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp2_tp1p2 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p2)*bw.N + n1));
                        const __m256d beta_kp2_tp1p3 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p3)*bw.N + n1));

                        const __m256d beta_kp3_tp1p0 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp3_tp1p1 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp3_tp1p2 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p2)*bw.N + n1));
                        const __m256d beta_kp3_tp1p3 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p3)*bw.N + n1));

                        const __m256d emit_kp0_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p0*bw.N + n1));
                        const __m256d emit_kp0_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p1*bw.N + n1));
                        const __m256d emit_kp0_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p2*bw.N + n1));
                        const __m256d emit_kp0_tp1p3 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p3*bw.N + n1));

                        const __m256d emit_kp1_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p0*bw.N + n1));
                        const __m256d emit_kp1_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p1*bw.N + n1));
                        const __m256d emit_kp1_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p2*bw.N + n1));
                        const __m256d emit_kp1_tp1p3 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p3*bw.N + n1));

                        const __m256d emit_kp2_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p0*bw.N + n1));
                        const __m256d emit_kp2_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p1*bw.N + n1));
                        const __m256d emit_kp2_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p2*bw.N + n1));
                        const __m256d emit_kp2_tp1p3 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p3*bw.N + n1));

                        const __m256d emit_kp3_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p0*bw.N + n1));
                        const __m256d emit_kp3_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p1*bw.N + n1));
                        const __m256d emit_kp3_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p2*bw.N + n1));
                        const __m256d emit_kp3_tp1p3 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p3*bw.N + n1));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp3)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp3_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p3, emit_kp0_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp3)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp3_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p3, emit_kp0_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp3)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp3_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p3, emit_kp0_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp3)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp3_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p3, emit_kp0_tp1p3))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp3)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp3_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p3, emit_kp1_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp3)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp3_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p3, emit_kp1_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp3)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp3_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p3, emit_kp1_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp3)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp3_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p3, emit_kp1_tp1p3))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp3)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp3_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p3, emit_kp2_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp3)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp3_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p3, emit_kp2_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp3)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp3_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p3, emit_kp2_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp3)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp3_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p3, emit_kp2_tp1p3))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp3)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp3_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p3, emit_kp3_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp3)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp3_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p3, emit_kp3_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp3)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp3_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p3, emit_kp3_tp1p3))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp3)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp3_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p3, emit_kp3_tp1p3))));

                    }

                }

            } else {

                for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

                    const size_t n0p0 = n0 + 0;
                    const size_t n0p1 = n0 + 1;
                    const size_t n0p2 = n0 + 2;
                    const size_t n0p3 = n0 + 3;

                    const __m256d alpha_kp0_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp0_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp0_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp0_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp0_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp0_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp0*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp1_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp1_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp1_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp1_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp1*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp2_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp2_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp2_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp2_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp2*bw.T + tp2)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp0_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp0_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp0_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp0_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp0)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp1_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp1_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp1_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp1_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp1)*bw.N + n0p3]);

                    const __m256d alpha_kp3_tp2_n0p0 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p0]);
                    const __m256d alpha_kp3_tp2_n0p1 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p1]);
                    const __m256d alpha_kp3_tp2_n0p2 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p2]);
                    const __m256d alpha_kp3_tp2_n0p3 = _mm256_set1_pd(bw.alpha[(kp3*bw.T + tp2)*bw.N + n0p3]);

                    for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                        const __m256d trans_n0p0 = _mm256_load_pd(bw.trans_prob + (n0p0*bw.N + n1));
                        const __m256d trans_n0p1 = _mm256_load_pd(bw.trans_prob + (n0p1*bw.N + n1));
                        const __m256d trans_n0p2 = _mm256_load_pd(bw.trans_prob + (n0p2*bw.N + n1));
                        const __m256d trans_n0p3 = _mm256_load_pd(bw.trans_prob + (n0p3*bw.N + n1));

                        const __m256d beta_kp0_tp1p0 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp0_tp1p1 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp0_tp1p2 = _mm256_load_pd(bw.beta + ((kp0*bw.T + tp1p2)*bw.N + n1));

                        const __m256d beta_kp1_tp1p0 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp1_tp1p1 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp1_tp1p2 = _mm256_load_pd(bw.beta + ((kp1*bw.T + tp1p2)*bw.N + n1));

                        const __m256d beta_kp2_tp1p0 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp2_tp1p1 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp2_tp1p2 = _mm256_load_pd(bw.beta + ((kp2*bw.T + tp1p2)*bw.N + n1));

                        const __m256d beta_kp3_tp1p0 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p0)*bw.N + n1));
                        const __m256d beta_kp3_tp1p1 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p1)*bw.N + n1));
                        const __m256d beta_kp3_tp1p2 = _mm256_load_pd(bw.beta + ((kp3*bw.T + tp1p2)*bw.N + n1));

                        const __m256d emit_kp0_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p0*bw.N + n1));
                        const __m256d emit_kp0_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p1*bw.N + n1));
                        const __m256d emit_kp0_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0_tp1p2*bw.N + n1));

                        const __m256d emit_kp1_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p0*bw.N + n1));
                        const __m256d emit_kp1_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p1*bw.N + n1));
                        const __m256d emit_kp1_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1_tp1p2*bw.N + n1));

                        const __m256d emit_kp2_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p0*bw.N + n1));
                        const __m256d emit_kp2_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p1*bw.N + n1));
                        const __m256d emit_kp2_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2_tp1p2*bw.N + n1));

                        const __m256d emit_kp3_tp1p0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p0*bw.N + n1));
                        const __m256d emit_kp3_tp1p1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p1*bw.N + n1));
                        const __m256d emit_kp3_tp1p2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3_tp1p2*bw.N + n1));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p0, emit_kp0_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p1, emit_kp0_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp0*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp0_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp0_tp1p2, emit_kp0_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p0, emit_kp1_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p1, emit_kp1_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp1*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp1_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp1_tp1p2, emit_kp1_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p0, emit_kp2_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p1, emit_kp2_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp2*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp2_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp2_tp1p2, emit_kp2_tp1p2))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp0_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p0, emit_kp3_tp1p0))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp1)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp1_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p1, emit_kp3_tp1p1))));

                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p0)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p0, _mm256_mul_pd(trans_n0p0, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p1)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p1, _mm256_mul_pd(trans_n0p1, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p2)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p2, _mm256_mul_pd(trans_n0p2, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));
                        _mm256_store_pd((bw.sigma + (((kp3*bw.T + tp2)*bw.N + n0p3)*bw.N + n1)), _mm256_mul_pd(alpha_kp3_tp2_n0p3, _mm256_mul_pd(trans_n0p3, _mm256_mul_pd(beta_kp3_tp1p2, emit_kp3_tp1p2))));

                    }

                }

            }

        }

        // sum up bw.sigma (from t = 0 to bw.T-1)
        for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

            const size_t n0p0 = n0 + 0;
            const size_t n0p1 = n0 + 1;
            const size_t n0p2 = n0 + 2;
            const size_t n0p3 = n0 + 3;

            for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                const size_t n1p0 = n1 + 0;

                __m256d sum_n0p0_kp0 = zeros;
                __m256d sum_n0p0_kp1 = zeros;
                __m256d sum_n0p0_kp2 = zeros;
                __m256d sum_n0p0_kp3 = zeros;

                __m256d sum_n0p1_kp0 = zeros;
                __m256d sum_n0p1_kp1 = zeros;
                __m256d sum_n0p1_kp2 = zeros;
                __m256d sum_n0p1_kp3 = zeros;

                __m256d sum_n0p2_kp0 = zeros;
                __m256d sum_n0p2_kp1 = zeros;
                __m256d sum_n0p2_kp2 = zeros;
                __m256d sum_n0p2_kp3 = zeros;

                __m256d sum_n0p3_kp0 = zeros;
                __m256d sum_n0p3_kp1 = zeros;
                __m256d sum_n0p3_kp2 = zeros;
                __m256d sum_n0p3_kp3 = zeros;

                for (size_t t = 0; t < bw.T-1; t++) {

                    const size_t tp0 = t + 0;

                    sum_n0p0_kp0 = _mm256_add_pd(sum_n0p0_kp0, _mm256_load_pd(bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p0)*bw.N + n1p0)));
                    sum_n0p0_kp1 = _mm256_add_pd(sum_n0p0_kp1, _mm256_load_pd(bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p0)*bw.N + n1p0)));
                    sum_n0p0_kp2 = _mm256_add_pd(sum_n0p0_kp2, _mm256_load_pd(bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p0)*bw.N + n1p0)));
                    sum_n0p0_kp3 = _mm256_add_pd(sum_n0p0_kp3, _mm256_load_pd(bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p0)*bw.N + n1p0)));

                    sum_n0p1_kp0 = _mm256_add_pd(sum_n0p1_kp0, _mm256_load_pd(bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p1)*bw.N + n1p0)));
                    sum_n0p1_kp1 = _mm256_add_pd(sum_n0p1_kp1, _mm256_load_pd(bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p1)*bw.N + n1p0)));
                    sum_n0p1_kp2 = _mm256_add_pd(sum_n0p1_kp2, _mm256_load_pd(bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p1)*bw.N + n1p0)));
                    sum_n0p1_kp3 = _mm256_add_pd(sum_n0p1_kp3, _mm256_load_pd(bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p1)*bw.N + n1p0)));

                    sum_n0p2_kp0 = _mm256_add_pd(sum_n0p2_kp0, _mm256_load_pd(bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p2)*bw.N + n1p0)));
                    sum_n0p2_kp1 = _mm256_add_pd(sum_n0p2_kp1, _mm256_load_pd(bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p2)*bw.N + n1p0)));
                    sum_n0p2_kp2 = _mm256_add_pd(sum_n0p2_kp2, _mm256_load_pd(bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p2)*bw.N + n1p0)));
                    sum_n0p2_kp3 = _mm256_add_pd(sum_n0p2_kp3, _mm256_load_pd(bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p2)*bw.N + n1p0)));

                    sum_n0p3_kp0 = _mm256_add_pd(sum_n0p3_kp0, _mm256_load_pd(bw.sigma + (((kp0*bw.T + tp0)*bw.N + n0p3)*bw.N + n1p0)));
                    sum_n0p3_kp1 = _mm256_add_pd(sum_n0p3_kp1, _mm256_load_pd(bw.sigma + (((kp1*bw.T + tp0)*bw.N + n0p3)*bw.N + n1p0)));
                    sum_n0p3_kp2 = _mm256_add_pd(sum_n0p3_kp2, _mm256_load_pd(bw.sigma + (((kp2*bw.T + tp0)*bw.N + n0p3)*bw.N + n1p0)));
                    sum_n0p3_kp3 = _mm256_add_pd(sum_n0p3_kp3, _mm256_load_pd(bw.sigma + (((kp3*bw.T + tp0)*bw.N + n0p3)*bw.N + n1p0)));

                }

                _mm256_store_pd((bw.sigma_sum + ((kp0*bw.N + n0p0)*bw.N + n1p0)), sum_n0p0_kp0);
                _mm256_store_pd((bw.sigma_sum + ((kp1*bw.N + n0p0)*bw.N + n1p0)), sum_n0p0_kp1);
                _mm256_store_pd((bw.sigma_sum + ((kp2*bw.N + n0p0)*bw.N + n1p0)), sum_n0p0_kp2);
                _mm256_store_pd((bw.sigma_sum + ((kp3*bw.N + n0p0)*bw.N + n1p0)), sum_n0p0_kp3);

                _mm256_store_pd((bw.sigma_sum + ((kp0*bw.N + n0p1)*bw.N + n1p0)), sum_n0p1_kp0);
                _mm256_store_pd((bw.sigma_sum + ((kp1*bw.N + n0p1)*bw.N + n1p0)), sum_n0p1_kp1);
                _mm256_store_pd((bw.sigma_sum + ((kp2*bw.N + n0p1)*bw.N + n1p0)), sum_n0p1_kp2);
                _mm256_store_pd((bw.sigma_sum + ((kp3*bw.N + n0p1)*bw.N + n1p0)), sum_n0p1_kp3);

                _mm256_store_pd((bw.sigma_sum + ((kp0*bw.N + n0p2)*bw.N + n1p0)), sum_n0p2_kp0);
                _mm256_store_pd((bw.sigma_sum + ((kp1*bw.N + n0p2)*bw.N + n1p0)), sum_n0p2_kp1);
                _mm256_store_pd((bw.sigma_sum + ((kp2*bw.N + n0p2)*bw.N + n1p0)), sum_n0p2_kp2);
                _mm256_store_pd((bw.sigma_sum + ((kp3*bw.N + n0p2)*bw.N + n1p0)), sum_n0p2_kp3);

                _mm256_store_pd((bw.sigma_sum + ((kp0*bw.N + n0p3)*bw.N + n1p0)), sum_n0p3_kp0);
                _mm256_store_pd((bw.sigma_sum + ((kp1*bw.N + n0p3)*bw.N + n1p0)), sum_n0p3_kp1);
                _mm256_store_pd((bw.sigma_sum + ((kp2*bw.N + n0p3)*bw.N + n1p0)), sum_n0p3_kp2);
                _mm256_store_pd((bw.sigma_sum + ((kp3*bw.N + n0p3)*bw.N + n1p0)), sum_n0p3_kp3);

            }
        }
    }

}


inline void update_init_prob(const BWdata& bw) {

    const __m256d vec_K_inv = _mm256_set1_pd(1.0/bw.K);

    for (size_t n = 0; n < bw.N; n += 4) {

        __m256d vec_g0_sum = zeros;

        for (size_t k = 0; k < bw.K; k += 4) {

            const size_t index_0 = ((k + 0)*bw.T + 0)*bw.N + n;
            const size_t index_1 = ((k + 1)*bw.T + 0)*bw.N + n;
            const size_t index_2 = ((k + 2)*bw.T + 0)*bw.N + n;
            const size_t index_3 = ((k + 3)*bw.T + 0)*bw.N + n;

            const __m256d vec_gamma_k_0_n_0 = _mm256_load_pd(bw.ggamma + index_0);
            const __m256d vec_gamma_k_1_n_0 = _mm256_load_pd(bw.ggamma + index_1);
            const __m256d vec_gamma_k_2_n_0 = _mm256_load_pd(bw.ggamma + index_2);
            const __m256d vec_gamma_k_3_n_0 = _mm256_load_pd(bw.ggamma + index_3);

            const __m256d a = _mm256_add_pd(vec_gamma_k_0_n_0, vec_gamma_k_1_n_0);
            const __m256d b = _mm256_add_pd(vec_gamma_k_2_n_0, vec_gamma_k_3_n_0);

            vec_g0_sum = _mm256_add_pd(vec_g0_sum, _mm256_add_pd(a, b));
        }

        _mm256_store_pd(bw.init_prob + n, _mm256_mul_pd(vec_g0_sum, vec_K_inv));

    }

}


inline void update_trans_prob(const BWdata& bw) {

    for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

        for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

            __m256d vec_numerator_sum_np0 = zeros;
            __m256d vec_numerator_sum_np1 = zeros;
            __m256d vec_numerator_sum_np2 = zeros;
            __m256d vec_numerator_sum_np3 = zeros;

            __m256d vec_denominator_sum = zeros;

            for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

                const __m256d vec_sigma_sum_np0_kp0 = _mm256_load_pd(bw.sigma_sum + (((k + 0)*bw.N + (n0 + 0))*bw.N + n1));
                const __m256d vec_sigma_sum_np0_kp1 = _mm256_load_pd(bw.sigma_sum + (((k + 1)*bw.N + (n0 + 0))*bw.N + n1));
                const __m256d vec_sigma_sum_np0_kp2 = _mm256_load_pd(bw.sigma_sum + (((k + 2)*bw.N + (n0 + 0))*bw.N + n1));
                const __m256d vec_sigma_sum_np0_kp3 = _mm256_load_pd(bw.sigma_sum + (((k + 3)*bw.N + (n0 + 0))*bw.N + n1));

                const __m256d vec_sigma_sum_np1_kp0 = _mm256_load_pd(bw.sigma_sum + (((k + 0)*bw.N + (n0 + 1))*bw.N + n1));
                const __m256d vec_sigma_sum_np1_kp1 = _mm256_load_pd(bw.sigma_sum + (((k + 1)*bw.N + (n0 + 1))*bw.N + n1));
                const __m256d vec_sigma_sum_np1_kp2 = _mm256_load_pd(bw.sigma_sum + (((k + 2)*bw.N + (n0 + 1))*bw.N + n1));
                const __m256d vec_sigma_sum_np1_kp3 = _mm256_load_pd(bw.sigma_sum + (((k + 3)*bw.N + (n0 + 1))*bw.N + n1));

                const __m256d vec_sigma_sum_np2_kp0 = _mm256_load_pd(bw.sigma_sum + (((k + 0)*bw.N + (n0 + 2))*bw.N + n1));
                const __m256d vec_sigma_sum_np2_kp1 = _mm256_load_pd(bw.sigma_sum + (((k + 1)*bw.N + (n0 + 2))*bw.N + n1));
                const __m256d vec_sigma_sum_np2_kp2 = _mm256_load_pd(bw.sigma_sum + (((k + 2)*bw.N + (n0 + 2))*bw.N + n1));
                const __m256d vec_sigma_sum_np2_kp3 = _mm256_load_pd(bw.sigma_sum + (((k + 3)*bw.N + (n0 + 2))*bw.N + n1));

                const __m256d vec_sigma_sum_np3_kp0 = _mm256_load_pd(bw.sigma_sum + (((k + 0)*bw.N + (n0 + 3))*bw.N + n1));
                const __m256d vec_sigma_sum_np3_kp1 = _mm256_load_pd(bw.sigma_sum + (((k + 1)*bw.N + (n0 + 3))*bw.N + n1));
                const __m256d vec_sigma_sum_np3_kp2 = _mm256_load_pd(bw.sigma_sum + (((k + 2)*bw.N + (n0 + 3))*bw.N + n1));
                const __m256d vec_sigma_sum_np3_kp3 = _mm256_load_pd(bw.sigma_sum + (((k + 3)*bw.N + (n0 + 3))*bw.N + n1));

                vec_numerator_sum_np0 = _mm256_add_pd(_mm256_add_pd(
                    _mm256_add_pd(vec_sigma_sum_np0_kp0, vec_sigma_sum_np0_kp1),
                    _mm256_add_pd(vec_sigma_sum_np0_kp2, vec_sigma_sum_np0_kp3)
                ), vec_numerator_sum_np0);

                vec_numerator_sum_np1 = _mm256_add_pd(_mm256_add_pd(
                    _mm256_add_pd(vec_sigma_sum_np1_kp0, vec_sigma_sum_np1_kp1),
                    _mm256_add_pd(vec_sigma_sum_np1_kp2, vec_sigma_sum_np1_kp3)
                ), vec_numerator_sum_np1);

                vec_numerator_sum_np2 = _mm256_add_pd(_mm256_add_pd(
                    _mm256_add_pd(vec_sigma_sum_np2_kp0, vec_sigma_sum_np2_kp1),
                    _mm256_add_pd(vec_sigma_sum_np2_kp2, vec_sigma_sum_np2_kp3)
                ), vec_numerator_sum_np2);

                vec_numerator_sum_np3 = _mm256_add_pd(_mm256_add_pd(
                    _mm256_add_pd(vec_sigma_sum_np3_kp0, vec_sigma_sum_np3_kp1),
                    _mm256_add_pd(vec_sigma_sum_np3_kp2, vec_sigma_sum_np3_kp3)
                ), vec_numerator_sum_np3);

                const __m256d vec_gamma_sum_kp0 = _mm256_load_pd(bw.gamma_sum + ((k + 0)*bw.N + n0));
                const __m256d vec_gamma_sum_kp1 = _mm256_load_pd(bw.gamma_sum + ((k + 1)*bw.N + n0));
                const __m256d vec_gamma_sum_kp2 = _mm256_load_pd(bw.gamma_sum + ((k + 2)*bw.N + n0));
                const __m256d vec_gamma_sum_kp3 = _mm256_load_pd(bw.gamma_sum + ((k + 3)*bw.N + n0));

                vec_denominator_sum = _mm256_add_pd(_mm256_add_pd(
                    _mm256_add_pd(vec_gamma_sum_kp0, vec_gamma_sum_kp1),
                    _mm256_add_pd(vec_gamma_sum_kp2, vec_gamma_sum_kp3)
                ), vec_denominator_sum);

            }

            _mm256_store_pd(helper_4_doubles, vec_denominator_sum);

            _mm256_store_pd((bw.trans_prob + ((n0 + 0)*bw.N + n1)), _mm256_div_pd(vec_numerator_sum_np0, _mm256_set1_pd(helper_4_doubles[0])));
            _mm256_store_pd((bw.trans_prob + ((n0 + 1)*bw.N + n1)), _mm256_div_pd(vec_numerator_sum_np1, _mm256_set1_pd(helper_4_doubles[1])));
            _mm256_store_pd((bw.trans_prob + ((n0 + 2)*bw.N + n1)), _mm256_div_pd(vec_numerator_sum_np2, _mm256_set1_pd(helper_4_doubles[2])));
            _mm256_store_pd((bw.trans_prob + ((n0 + 3)*bw.N + n1)), _mm256_div_pd(vec_numerator_sum_np3, _mm256_set1_pd(helper_4_doubles[3])));

        }

    }

}


inline void update_emit_prob(const BWdata& bw) {

    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {

            double* index_gamma_sum_kp0 = bw.gamma_sum + ((k + 0)*bw.N + n);
            double* index_gamma_sum_kp1 = bw.gamma_sum + ((k + 1)*bw.N + n);
            double* index_gamma_sum_kp2 = bw.gamma_sum + ((k + 2)*bw.N + n);
            double* index_gamma_sum_kp3 = bw.gamma_sum + ((k + 3)*bw.N + n);

            const __m256d vec_ggamma_kp0 = _mm256_load_pd(bw.ggamma + (((k + 0)*bw.T + (bw.T-1))*bw.N + n));
            const __m256d vec_ggamma_kp1 = _mm256_load_pd(bw.ggamma + (((k + 1)*bw.T + (bw.T-1))*bw.N + n));
            const __m256d vec_ggamma_kp2 = _mm256_load_pd(bw.ggamma + (((k + 2)*bw.T + (bw.T-1))*bw.N + n));
            const __m256d vec_ggamma_kp3 = _mm256_load_pd(bw.ggamma + (((k + 3)*bw.T + (bw.T-1))*bw.N + n));

            const __m256d vec_ggamma_sum_kp0 = _mm256_load_pd(index_gamma_sum_kp0);
            const __m256d vec_ggamma_sum_kp1 = _mm256_load_pd(index_gamma_sum_kp1);
            const __m256d vec_ggamma_sum_kp2 = _mm256_load_pd(index_gamma_sum_kp2);
            const __m256d vec_ggamma_sum_kp3 = _mm256_load_pd(index_gamma_sum_kp3);

            const __m256d result_kp0 = _mm256_add_pd(vec_ggamma_kp0, vec_ggamma_sum_kp0);
            const __m256d result_kp1 = _mm256_add_pd(vec_ggamma_kp1, vec_ggamma_sum_kp1);
            const __m256d result_kp2 = _mm256_add_pd(vec_ggamma_kp2, vec_ggamma_sum_kp2);
            const __m256d result_kp3 = _mm256_add_pd(vec_ggamma_kp3, vec_ggamma_sum_kp3);

            _mm256_store_pd(index_gamma_sum_kp0, result_kp0);
            _mm256_store_pd(index_gamma_sum_kp1, result_kp1);
            _mm256_store_pd(index_gamma_sum_kp2, result_kp2);
            _mm256_store_pd(index_gamma_sum_kp3, result_kp3);

        }

    }

    // update bw.emit_prob
    for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {

        /* DANGER INSANITY ZONE */

        for (size_t m = 0; m < bw.M; m += STRIDE_LAYER_M) {

            const __m256d mask_mp0 = _mm256_set1_pd(m + 0);
            const __m256d mask_mp1 = _mm256_set1_pd(m + 1);
            const __m256d mask_mp2 = _mm256_set1_pd(m + 2);
            const __m256d mask_mp3 = _mm256_set1_pd(m + 3);

            __m256d vec_numerator_sum_np0 = zeros;
            __m256d vec_numerator_sum_np1 = zeros;
            __m256d vec_numerator_sum_np2 = zeros;
            __m256d vec_numerator_sum_np3 = zeros;

            __m256d vec_numerator_sum_np0_mp0 = zeros;
            __m256d vec_numerator_sum_np0_mp1 = zeros;
            __m256d vec_numerator_sum_np0_mp2 = zeros;
            __m256d vec_numerator_sum_np0_mp3 = zeros;

            __m256d vec_numerator_sum_np1_mp0 = zeros;
            __m256d vec_numerator_sum_np1_mp1 = zeros;
            __m256d vec_numerator_sum_np1_mp2 = zeros;
            __m256d vec_numerator_sum_np1_mp3 = zeros;

            __m256d vec_numerator_sum_np2_mp0 = zeros;
            __m256d vec_numerator_sum_np2_mp1 = zeros;
            __m256d vec_numerator_sum_np2_mp2 = zeros;
            __m256d vec_numerator_sum_np2_mp3 = zeros;

            __m256d vec_numerator_sum_np3_mp0 = zeros;
            __m256d vec_numerator_sum_np3_mp1 = zeros;
            __m256d vec_numerator_sum_np3_mp2 = zeros;
            __m256d vec_numerator_sum_np3_mp3 = zeros;

            __m256d vec_denominator_sum_np0 = zeros;
            __m256d vec_denominator_sum_np1 = zeros;
            __m256d vec_denominator_sum_np2 = zeros;
            __m256d vec_denominator_sum_np3 = zeros;

            // https://www.youtube.com/watch?v=dQw4w9WgXcQ

            for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

                size_t index_kp0 = (k + 0)*bw.N + n;
                size_t index_kp1 = (k + 1)*bw.N + n;
                size_t index_kp2 = (k + 2)*bw.N + n;
                size_t index_kp3 = (k + 3)*bw.N + n;

                // AVX memory layout 3, 2, 1, 0 ...
                vec_denominator_sum_np0 = _mm256_stuff_pd(vec_denominator_sum_np0, bw.gamma_sum, index_kp3+0, index_kp2+0, index_kp1+0, index_kp0+0);
                vec_denominator_sum_np1 = _mm256_stuff_pd(vec_denominator_sum_np1, bw.gamma_sum, index_kp3+1, index_kp2+1, index_kp1+1, index_kp0+1);
                vec_denominator_sum_np2 = _mm256_stuff_pd(vec_denominator_sum_np2, bw.gamma_sum, index_kp3+2, index_kp2+2, index_kp1+2, index_kp0+2);
                vec_denominator_sum_np3 = _mm256_stuff_pd(vec_denominator_sum_np3, bw.gamma_sum, index_kp3+3, index_kp2+3, index_kp1+3, index_kp0+3);

                __m256d csum_np0_mp0_kp0 = zeros;
                __m256d csum_np0_mp0_kp1 = zeros;
                __m256d csum_np0_mp0_kp2 = zeros;
                __m256d csum_np0_mp0_kp3 = zeros;

                __m256d csum_np0_mp1_kp0 = zeros;
                __m256d csum_np0_mp1_kp1 = zeros;
                __m256d csum_np0_mp1_kp2 = zeros;
                __m256d csum_np0_mp1_kp3 = zeros;

                __m256d csum_np0_mp2_kp0 = zeros;
                __m256d csum_np0_mp2_kp1 = zeros;
                __m256d csum_np0_mp2_kp2 = zeros;
                __m256d csum_np0_mp2_kp3 = zeros;

                __m256d csum_np0_mp3_kp0 = zeros;
                __m256d csum_np0_mp3_kp1 = zeros;
                __m256d csum_np0_mp3_kp2 = zeros;
                __m256d csum_np0_mp3_kp3 = zeros;

                __m256d csum_np1_mp0_kp0 = zeros;
                __m256d csum_np1_mp0_kp1 = zeros;
                __m256d csum_np1_mp0_kp2 = zeros;
                __m256d csum_np1_mp0_kp3 = zeros;

                __m256d csum_np1_mp1_kp0 = zeros;
                __m256d csum_np1_mp1_kp1 = zeros;
                __m256d csum_np1_mp1_kp2 = zeros;
                __m256d csum_np1_mp1_kp3 = zeros;

                __m256d csum_np1_mp2_kp0 = zeros;
                __m256d csum_np1_mp2_kp1 = zeros;
                __m256d csum_np1_mp2_kp2 = zeros;
                __m256d csum_np1_mp2_kp3 = zeros;

                __m256d csum_np1_mp3_kp0 = zeros;
                __m256d csum_np1_mp3_kp1 = zeros;
                __m256d csum_np1_mp3_kp2 = zeros;
                __m256d csum_np1_mp3_kp3 = zeros;

                __m256d csum_np2_mp0_kp0 = zeros;
                __m256d csum_np2_mp0_kp1 = zeros;
                __m256d csum_np2_mp0_kp2 = zeros;
                __m256d csum_np2_mp0_kp3 = zeros;

                __m256d csum_np2_mp1_kp0 = zeros;
                __m256d csum_np2_mp1_kp1 = zeros;
                __m256d csum_np2_mp1_kp2 = zeros;
                __m256d csum_np2_mp1_kp3 = zeros;

                __m256d csum_np2_mp2_kp0 = zeros;
                __m256d csum_np2_mp2_kp1 = zeros;
                __m256d csum_np2_mp2_kp2 = zeros;
                __m256d csum_np2_mp2_kp3 = zeros;

                __m256d csum_np2_mp3_kp0 = zeros;
                __m256d csum_np2_mp3_kp1 = zeros;
                __m256d csum_np2_mp3_kp2 = zeros;
                __m256d csum_np2_mp3_kp3 = zeros;

                __m256d csum_np3_mp0_kp0 = zeros;
                __m256d csum_np3_mp0_kp1 = zeros;
                __m256d csum_np3_mp0_kp2 = zeros;
                __m256d csum_np3_mp0_kp3 = zeros;

                __m256d csum_np3_mp1_kp0 = zeros;
                __m256d csum_np3_mp1_kp1 = zeros;
                __m256d csum_np3_mp1_kp2 = zeros;
                __m256d csum_np3_mp1_kp3 = zeros;

                __m256d csum_np3_mp2_kp0 = zeros;
                __m256d csum_np3_mp2_kp1 = zeros;
                __m256d csum_np3_mp2_kp2 = zeros;
                __m256d csum_np3_mp2_kp3 = zeros;

                __m256d csum_np3_mp3_kp0 = zeros;
                __m256d csum_np3_mp3_kp1 = zeros;
                __m256d csum_np3_mp3_kp2 = zeros;
                __m256d csum_np3_mp3_kp3 = zeros;

                for (size_t t = 0; t < bw.T; t += STRIDE_LAYER_T_NON_RECURSIVE) {

                    const __m256d vec_observations_kp0 = _mm256_load_pd(observations_double_array + ((k + 0)*bw.T + t));
                    const __m256d vec_observations_kp1 = _mm256_load_pd(observations_double_array + ((k + 1)*bw.T + t));
                    const __m256d vec_observations_kp2 = _mm256_load_pd(observations_double_array + ((k + 2)*bw.T + t));
                    const __m256d vec_observations_kp3 = _mm256_load_pd(observations_double_array + ((k + 3)*bw.T + t));

                    const __m256d mask_mp0_kp0 = _mm256_cmp_pd(vec_observations_kp0, mask_mp0, _CMP_EQ_OQ);
                    const __m256d mask_mp0_kp1 = _mm256_cmp_pd(vec_observations_kp1, mask_mp0, _CMP_EQ_OQ);
                    const __m256d mask_mp0_kp2 = _mm256_cmp_pd(vec_observations_kp2, mask_mp0, _CMP_EQ_OQ);
                    const __m256d mask_mp0_kp3 = _mm256_cmp_pd(vec_observations_kp3, mask_mp0, _CMP_EQ_OQ);

                    const __m256d mask_mp1_kp0 = _mm256_cmp_pd(vec_observations_kp0, mask_mp1, _CMP_EQ_OQ);
                    const __m256d mask_mp1_kp1 = _mm256_cmp_pd(vec_observations_kp1, mask_mp1, _CMP_EQ_OQ);
                    const __m256d mask_mp1_kp2 = _mm256_cmp_pd(vec_observations_kp2, mask_mp1, _CMP_EQ_OQ);
                    const __m256d mask_mp1_kp3 = _mm256_cmp_pd(vec_observations_kp3, mask_mp1, _CMP_EQ_OQ);

                    const __m256d mask_mp2_kp0 = _mm256_cmp_pd(vec_observations_kp0, mask_mp2, _CMP_EQ_OQ);
                    const __m256d mask_mp2_kp1 = _mm256_cmp_pd(vec_observations_kp1, mask_mp2, _CMP_EQ_OQ);
                    const __m256d mask_mp2_kp2 = _mm256_cmp_pd(vec_observations_kp2, mask_mp2, _CMP_EQ_OQ);
                    const __m256d mask_mp2_kp3 = _mm256_cmp_pd(vec_observations_kp3, mask_mp2, _CMP_EQ_OQ);

                    const __m256d mask_mp3_kp0 = _mm256_cmp_pd(vec_observations_kp0, mask_mp3, _CMP_EQ_OQ);
                    const __m256d mask_mp3_kp1 = _mm256_cmp_pd(vec_observations_kp1, mask_mp3, _CMP_EQ_OQ);
                    const __m256d mask_mp3_kp2 = _mm256_cmp_pd(vec_observations_kp2, mask_mp3, _CMP_EQ_OQ);
                    const __m256d mask_mp3_kp3 = _mm256_cmp_pd(vec_observations_kp3, mask_mp3, _CMP_EQ_OQ);

                    __m256d vec_ggamma_np0_kp0 = _mm256_load_pd(ggamma_N_K_T + (((n + 0)*bw.K + (k + 0))*bw.T + t));
                    __m256d vec_ggamma_np0_kp1 = _mm256_load_pd(ggamma_N_K_T + (((n + 0)*bw.K + (k + 1))*bw.T + t));
                    __m256d vec_ggamma_np0_kp2 = _mm256_load_pd(ggamma_N_K_T + (((n + 0)*bw.K + (k + 2))*bw.T + t));
                    __m256d vec_ggamma_np0_kp3 = _mm256_load_pd(ggamma_N_K_T + (((n + 0)*bw.K + (k + 3))*bw.T + t));

                    __m256d vec_ggamma_np1_kp0 = _mm256_load_pd(ggamma_N_K_T + (((n + 1)*bw.K + (k + 0))*bw.T + t));
                    __m256d vec_ggamma_np1_kp1 = _mm256_load_pd(ggamma_N_K_T + (((n + 1)*bw.K + (k + 1))*bw.T + t));
                    __m256d vec_ggamma_np1_kp2 = _mm256_load_pd(ggamma_N_K_T + (((n + 1)*bw.K + (k + 2))*bw.T + t));
                    __m256d vec_ggamma_np1_kp3 = _mm256_load_pd(ggamma_N_K_T + (((n + 1)*bw.K + (k + 3))*bw.T + t));

                    __m256d vec_ggamma_np2_kp0 = _mm256_load_pd(ggamma_N_K_T + (((n + 2)*bw.K + (k + 0))*bw.T + t));
                    __m256d vec_ggamma_np2_kp1 = _mm256_load_pd(ggamma_N_K_T + (((n + 2)*bw.K + (k + 1))*bw.T + t));
                    __m256d vec_ggamma_np2_kp2 = _mm256_load_pd(ggamma_N_K_T + (((n + 2)*bw.K + (k + 2))*bw.T + t));
                    __m256d vec_ggamma_np2_kp3 = _mm256_load_pd(ggamma_N_K_T + (((n + 2)*bw.K + (k + 3))*bw.T + t));

                    __m256d vec_ggamma_np3_kp0 = _mm256_load_pd(ggamma_N_K_T + (((n + 3)*bw.K + (k + 0))*bw.T + t));
                    __m256d vec_ggamma_np3_kp1 = _mm256_load_pd(ggamma_N_K_T + (((n + 3)*bw.K + (k + 1))*bw.T + t));
                    __m256d vec_ggamma_np3_kp2 = _mm256_load_pd(ggamma_N_K_T + (((n + 3)*bw.K + (k + 2))*bw.T + t));
                    __m256d vec_ggamma_np3_kp3 = _mm256_load_pd(ggamma_N_K_T + (((n + 3)*bw.K + (k + 3))*bw.T + t));

                    csum_np0_mp0_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp0, mask_mp0_kp0), csum_np0_mp0_kp0);
                    csum_np0_mp0_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp1, mask_mp0_kp1), csum_np0_mp0_kp1);
                    csum_np0_mp0_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp2, mask_mp0_kp2), csum_np0_mp0_kp2);
                    csum_np0_mp0_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp3, mask_mp0_kp3), csum_np0_mp0_kp3);

                    csum_np0_mp1_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp0, mask_mp1_kp0), csum_np0_mp1_kp0);
                    csum_np0_mp1_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp1, mask_mp1_kp1), csum_np0_mp1_kp1);
                    csum_np0_mp1_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp2, mask_mp1_kp2), csum_np0_mp1_kp2);
                    csum_np0_mp1_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp3, mask_mp1_kp3), csum_np0_mp1_kp3);

                    csum_np0_mp2_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp0, mask_mp2_kp0), csum_np0_mp2_kp0);
                    csum_np0_mp2_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp1, mask_mp2_kp1), csum_np0_mp2_kp1);
                    csum_np0_mp2_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp2, mask_mp2_kp2), csum_np0_mp2_kp2);
                    csum_np0_mp2_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp3, mask_mp2_kp3), csum_np0_mp2_kp3);

                    csum_np0_mp3_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp0, mask_mp3_kp0), csum_np0_mp3_kp0);
                    csum_np0_mp3_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp1, mask_mp3_kp1), csum_np0_mp3_kp1);
                    csum_np0_mp3_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp2, mask_mp3_kp2), csum_np0_mp3_kp2);
                    csum_np0_mp3_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np0_kp3, mask_mp3_kp3), csum_np0_mp3_kp3);

                    csum_np1_mp0_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp0, mask_mp0_kp0), csum_np1_mp0_kp0);
                    csum_np1_mp0_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp1, mask_mp0_kp1), csum_np1_mp0_kp1);
                    csum_np1_mp0_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp2, mask_mp0_kp2), csum_np1_mp0_kp2);
                    csum_np1_mp0_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp3, mask_mp0_kp3), csum_np1_mp0_kp3);

                    csum_np1_mp1_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp0, mask_mp1_kp0), csum_np1_mp1_kp0);
                    csum_np1_mp1_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp1, mask_mp1_kp1), csum_np1_mp1_kp1);
                    csum_np1_mp1_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp2, mask_mp1_kp2), csum_np1_mp1_kp2);
                    csum_np1_mp1_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp3, mask_mp1_kp3), csum_np1_mp1_kp3);

                    csum_np1_mp2_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp0, mask_mp2_kp0), csum_np1_mp2_kp0);
                    csum_np1_mp2_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp1, mask_mp2_kp1), csum_np1_mp2_kp1);
                    csum_np1_mp2_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp2, mask_mp2_kp2), csum_np1_mp2_kp2);
                    csum_np1_mp2_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp3, mask_mp2_kp3), csum_np1_mp2_kp3);

                    csum_np1_mp3_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp0, mask_mp3_kp0), csum_np1_mp3_kp0);
                    csum_np1_mp3_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp1, mask_mp3_kp1), csum_np1_mp3_kp1);
                    csum_np1_mp3_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp2, mask_mp3_kp2), csum_np1_mp3_kp2);
                    csum_np1_mp3_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np1_kp3, mask_mp3_kp3), csum_np1_mp3_kp3);

                    csum_np2_mp0_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp0, mask_mp0_kp0), csum_np2_mp0_kp0);
                    csum_np2_mp0_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp1, mask_mp0_kp1), csum_np2_mp0_kp1);
                    csum_np2_mp0_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp2, mask_mp0_kp2), csum_np2_mp0_kp2);
                    csum_np2_mp0_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp3, mask_mp0_kp3), csum_np2_mp0_kp3);

                    csum_np2_mp1_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp0, mask_mp1_kp0), csum_np2_mp1_kp0);
                    csum_np2_mp1_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp1, mask_mp1_kp1), csum_np2_mp1_kp1);
                    csum_np2_mp1_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp2, mask_mp1_kp2), csum_np2_mp1_kp2);
                    csum_np2_mp1_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp3, mask_mp1_kp3), csum_np2_mp1_kp3);

                    csum_np2_mp2_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp0, mask_mp2_kp0), csum_np2_mp2_kp0);
                    csum_np2_mp2_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp1, mask_mp2_kp1), csum_np2_mp2_kp1);
                    csum_np2_mp2_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp2, mask_mp2_kp2), csum_np2_mp2_kp2);
                    csum_np2_mp2_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp3, mask_mp2_kp3), csum_np2_mp2_kp3);

                    csum_np2_mp3_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp0, mask_mp3_kp0), csum_np2_mp3_kp0);
                    csum_np2_mp3_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp1, mask_mp3_kp1), csum_np2_mp3_kp1);
                    csum_np2_mp3_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp2, mask_mp3_kp2), csum_np2_mp3_kp2);
                    csum_np2_mp3_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np2_kp3, mask_mp3_kp3), csum_np2_mp3_kp3);

                    csum_np3_mp0_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp0, mask_mp0_kp0), csum_np3_mp0_kp0);
                    csum_np3_mp0_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp1, mask_mp0_kp1), csum_np3_mp0_kp1);
                    csum_np3_mp0_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp2, mask_mp0_kp2), csum_np3_mp0_kp2);
                    csum_np3_mp0_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp3, mask_mp0_kp3), csum_np3_mp0_kp3);

                    csum_np3_mp1_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp0, mask_mp1_kp0), csum_np3_mp1_kp0);
                    csum_np3_mp1_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp1, mask_mp1_kp1), csum_np3_mp1_kp1);
                    csum_np3_mp1_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp2, mask_mp1_kp2), csum_np3_mp1_kp2);
                    csum_np3_mp1_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp3, mask_mp1_kp3), csum_np3_mp1_kp3);

                    csum_np3_mp2_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp0, mask_mp2_kp0), csum_np3_mp2_kp0);
                    csum_np3_mp2_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp1, mask_mp2_kp1), csum_np3_mp2_kp1);
                    csum_np3_mp2_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp2, mask_mp2_kp2), csum_np3_mp2_kp2);
                    csum_np3_mp2_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp3, mask_mp2_kp3), csum_np3_mp2_kp3);

                    csum_np3_mp3_kp0 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp0, mask_mp3_kp0), csum_np3_mp3_kp0);
                    csum_np3_mp3_kp1 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp1, mask_mp3_kp1), csum_np3_mp3_kp1);
                    csum_np3_mp3_kp2 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp2, mask_mp3_kp2), csum_np3_mp3_kp2);
                    csum_np3_mp3_kp3 = _mm256_add_pd(_mm256_and_pd(vec_ggamma_np3_kp3, mask_mp3_kp3), csum_np3_mp3_kp3);

                }

                vec_numerator_sum_np0_mp0 = _mm256_add_pd(vec_numerator_sum_np0_mp0, _mm256_sumFourRowsIntoOneCol_pd(csum_np0_mp0_kp0, csum_np0_mp0_kp1, csum_np0_mp0_kp2, csum_np0_mp0_kp3));
                vec_numerator_sum_np0_mp1 = _mm256_add_pd(vec_numerator_sum_np0_mp1, _mm256_sumFourRowsIntoOneCol_pd(csum_np0_mp1_kp0, csum_np0_mp1_kp1, csum_np0_mp1_kp2, csum_np0_mp1_kp3));
                vec_numerator_sum_np0_mp2 = _mm256_add_pd(vec_numerator_sum_np0_mp2, _mm256_sumFourRowsIntoOneCol_pd(csum_np0_mp2_kp0, csum_np0_mp2_kp1, csum_np0_mp2_kp2, csum_np0_mp2_kp3));
                vec_numerator_sum_np0_mp3 = _mm256_add_pd(vec_numerator_sum_np0_mp3, _mm256_sumFourRowsIntoOneCol_pd(csum_np0_mp3_kp0, csum_np0_mp3_kp1, csum_np0_mp3_kp2, csum_np0_mp3_kp3));

                vec_numerator_sum_np1_mp0 = _mm256_add_pd(vec_numerator_sum_np1_mp0, _mm256_sumFourRowsIntoOneCol_pd(csum_np1_mp0_kp0, csum_np1_mp0_kp1, csum_np1_mp0_kp2, csum_np1_mp0_kp3));
                vec_numerator_sum_np1_mp1 = _mm256_add_pd(vec_numerator_sum_np1_mp1, _mm256_sumFourRowsIntoOneCol_pd(csum_np1_mp1_kp0, csum_np1_mp1_kp1, csum_np1_mp1_kp2, csum_np1_mp1_kp3));
                vec_numerator_sum_np1_mp2 = _mm256_add_pd(vec_numerator_sum_np1_mp2, _mm256_sumFourRowsIntoOneCol_pd(csum_np1_mp2_kp0, csum_np1_mp2_kp1, csum_np1_mp2_kp2, csum_np1_mp2_kp3));
                vec_numerator_sum_np1_mp3 = _mm256_add_pd(vec_numerator_sum_np1_mp3, _mm256_sumFourRowsIntoOneCol_pd(csum_np1_mp3_kp0, csum_np1_mp3_kp1, csum_np1_mp3_kp2, csum_np1_mp3_kp3));

                vec_numerator_sum_np2_mp0 = _mm256_add_pd(vec_numerator_sum_np2_mp0, _mm256_sumFourRowsIntoOneCol_pd(csum_np2_mp0_kp0, csum_np2_mp0_kp1, csum_np2_mp0_kp2, csum_np2_mp0_kp3));
                vec_numerator_sum_np2_mp1 = _mm256_add_pd(vec_numerator_sum_np2_mp1, _mm256_sumFourRowsIntoOneCol_pd(csum_np2_mp1_kp0, csum_np2_mp1_kp1, csum_np2_mp1_kp2, csum_np2_mp1_kp3));
                vec_numerator_sum_np2_mp2 = _mm256_add_pd(vec_numerator_sum_np2_mp2, _mm256_sumFourRowsIntoOneCol_pd(csum_np2_mp2_kp0, csum_np2_mp2_kp1, csum_np2_mp2_kp2, csum_np2_mp2_kp3));
                vec_numerator_sum_np2_mp3 = _mm256_add_pd(vec_numerator_sum_np2_mp3, _mm256_sumFourRowsIntoOneCol_pd(csum_np2_mp3_kp0, csum_np2_mp3_kp1, csum_np2_mp3_kp2, csum_np2_mp3_kp3));

                vec_numerator_sum_np3_mp0 = _mm256_add_pd(vec_numerator_sum_np3_mp0, _mm256_sumFourRowsIntoOneCol_pd(csum_np3_mp0_kp0, csum_np3_mp0_kp1, csum_np3_mp0_kp2, csum_np3_mp0_kp3));
                vec_numerator_sum_np3_mp1 = _mm256_add_pd(vec_numerator_sum_np3_mp1, _mm256_sumFourRowsIntoOneCol_pd(csum_np3_mp1_kp0, csum_np3_mp1_kp1, csum_np3_mp1_kp2, csum_np3_mp1_kp3));
                vec_numerator_sum_np3_mp2 = _mm256_add_pd(vec_numerator_sum_np3_mp2, _mm256_sumFourRowsIntoOneCol_pd(csum_np3_mp2_kp0, csum_np3_mp2_kp1, csum_np3_mp2_kp2, csum_np3_mp2_kp3));
                vec_numerator_sum_np3_mp3 = _mm256_add_pd(vec_numerator_sum_np3_mp3, _mm256_sumFourRowsIntoOneCol_pd(csum_np3_mp3_kp0, csum_np3_mp3_kp1, csum_np3_mp3_kp2, csum_np3_mp3_kp3));

            }

            vec_numerator_sum_np0 = _mm256_add_pd(vec_numerator_sum_np0, _mm256_sumFourRowsIntoOneCol_pd(vec_numerator_sum_np0_mp0, vec_numerator_sum_np0_mp1, vec_numerator_sum_np0_mp2, vec_numerator_sum_np0_mp3));
            vec_numerator_sum_np1 = _mm256_add_pd(vec_numerator_sum_np1, _mm256_sumFourRowsIntoOneCol_pd(vec_numerator_sum_np1_mp0, vec_numerator_sum_np1_mp1, vec_numerator_sum_np1_mp2, vec_numerator_sum_np1_mp3));
            vec_numerator_sum_np2 = _mm256_add_pd(vec_numerator_sum_np2, _mm256_sumFourRowsIntoOneCol_pd(vec_numerator_sum_np2_mp0, vec_numerator_sum_np2_mp1, vec_numerator_sum_np2_mp2, vec_numerator_sum_np2_mp3));
            vec_numerator_sum_np3 = _mm256_add_pd(vec_numerator_sum_np3, _mm256_sumFourRowsIntoOneCol_pd(vec_numerator_sum_np3_mp0, vec_numerator_sum_np3_mp1, vec_numerator_sum_np3_mp2, vec_numerator_sum_np3_mp3));

            _mm256_store_pd(helper_4_doubles, _mm256_sumFourRowsIntoOneCol_pd(vec_denominator_sum_np0, vec_denominator_sum_np1, vec_denominator_sum_np2, vec_denominator_sum_np3));

            vec_denominator_sum_np0 = _mm256_set1_pd(helper_4_doubles[0]);
            vec_denominator_sum_np1 = _mm256_set1_pd(helper_4_doubles[1]);
            vec_denominator_sum_np2 = _mm256_set1_pd(helper_4_doubles[2]);
            vec_denominator_sum_np3 = _mm256_set1_pd(helper_4_doubles[3]);

            const __m256d result_np0 = _mm256_div_pd(vec_numerator_sum_np0, vec_denominator_sum_np0);
            const __m256d result_np1 = _mm256_div_pd(vec_numerator_sum_np1, vec_denominator_sum_np1);
            const __m256d result_np2 = _mm256_div_pd(vec_numerator_sum_np2, vec_denominator_sum_np2);
            const __m256d result_np3 = _mm256_div_pd(vec_numerator_sum_np3, vec_denominator_sum_np3);

            _mm256_store_pd((bw.emit_prob + ((n + 0)*bw.M + m)), result_np0);
            _mm256_store_pd((bw.emit_prob + ((n + 1)*bw.M + m)), result_np1);
            _mm256_store_pd((bw.emit_prob + ((n + 2)*bw.M + m)), result_np2);
            _mm256_store_pd((bw.emit_prob + ((n + 3)*bw.M + m)), result_np3);

        }

    }

}

