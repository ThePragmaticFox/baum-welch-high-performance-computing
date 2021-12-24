/*
    Scalar optimized implementation
    No vectorization intrinsics here, otherwise optimize as you like!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#include <cmath>
#include <cstring>

#include "../common.h"


static void forward_step(const BWdata& bw, double& neg_log_likelihood_sum);
static void backward_step(const BWdata& bw, const size_t& k);
static void compute_gamma(const BWdata& bw, const size_t& k);
static void update_trans_prob(const BWdata& bw);
static void update_emit_prob(const BWdata& bw);
static size_t comp_bw_scalar_unroll(const BWdata& bw);


REGISTER_FUNCTION(comp_bw_scalar_unroll, "other-unroll", "Another approach to unrolling the code");

static double* denominator_sum = nullptr;
static double* numerator_sum = nullptr;

static size_t comp_bw_scalar_unroll(const BWdata& bw){
    size_t res = 0;
    double neg_log_likelihood_sum, neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;
    denominator_sum = (double *)aligned_alloc(32,bw.N * sizeof(double));
    numerator_sum = (double *)aligned_alloc(32,bw.N*bw.M * sizeof(double));

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {
        neg_log_likelihood_sum = 0.0;

        forward_step(bw, neg_log_likelihood_sum);
        for (size_t k = 0; k < bw.K; k++) {

            backward_step(bw, k);
            compute_gamma(bw, k);
        }


        bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;

        if (first && i > 0 && fabs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < EPSILON){
            first = false;
            res = i+1;
        }

        neg_log_likelihood_sum_old = neg_log_likelihood_sum;


        update_trans_prob(bw);
        update_emit_prob(bw);
    }
    free(denominator_sum);
    free(numerator_sum);
    return res;
}


static inline void forward_step(const BWdata& bw, double& neg_log_likelihood_sum) {
    //Init
    double c_norm, alpha, alpha_sum, init_prob, emit_prob, trans_prob;
    double c_norm0, alpha0, alpha_sum0, init_prob0, emit_prob0, trans_prob0;
    double c_norm1, alpha1, alpha_sum1, init_prob1, emit_prob1, trans_prob1;
    double c_norm2, alpha2, alpha_sum2, init_prob2, emit_prob2, trans_prob2;
    double c_norm3, alpha3, alpha_sum3, init_prob3, emit_prob3, trans_prob3;

    size_t kTN, kT;
    size_t kT0N, kT1N, kT2N, kT3N;
    // t = 0, base case

    // Init
    for(size_t k=0; k < bw.K; k++){
        c_norm0 = 0;
        c_norm1 = 0;
        c_norm2 = 0;
        c_norm3 = 0;
        kTN = k*bw.T*bw.N;
        kT = k*bw.T;

        size_t observations = bw.observations[k*bw.T];

        for (size_t n = 0; n < bw.N; n+=4){
            // Load
            init_prob0 = bw.init_prob[n + 0];
            init_prob1 = bw.init_prob[n + 1];
            init_prob2 = bw.init_prob[n + 2];
            init_prob3 = bw.init_prob[n + 3];
            emit_prob0 = bw.emit_prob[(n + 0)*bw.M + observations];
            emit_prob1 = bw.emit_prob[(n + 1)*bw.M + observations];
            emit_prob2 = bw.emit_prob[(n + 2)*bw.M + observations];
            emit_prob3 = bw.emit_prob[(n + 3)*bw.M + observations];

            // Calculate
            alpha0 = init_prob0 * emit_prob0;
            alpha1 = init_prob1 * emit_prob1;
            alpha2 = init_prob2 * emit_prob2;
            alpha3 = init_prob3 * emit_prob3;
            c_norm0 += alpha0;
            c_norm1 += alpha1;
            c_norm2 += alpha2;
            c_norm3 += alpha3;

            // Store
            bw.alpha[kTN + n + 0] = alpha0;
            bw.alpha[kTN + n + 1] = alpha1;
            bw.alpha[kTN + n + 2] = alpha2;
            bw.alpha[kTN + n + 3] = alpha3;
        }

        // Calculate
        c_norm = 1.0/(c_norm0 + c_norm1 + c_norm2 + c_norm3);


        for (size_t n = 0; n < bw.N; n+=4){
            // Load
            alpha0 = bw.alpha[kTN + n+0];
            alpha1 = bw.alpha[kTN + n+1];
            alpha2 = bw.alpha[kTN + n+2];
            alpha3 = bw.alpha[kTN + n+3];

            // Calculate
            alpha0 *= c_norm;
            alpha1 *= c_norm;
            alpha2 *= c_norm;
            alpha3 *= c_norm;

            // Store
            bw.alpha[kTN + n+0] = alpha0;
            bw.alpha[kTN + n+1] = alpha1;
            bw.alpha[kTN + n+2] = alpha2;
            bw.alpha[kTN + n+3] = alpha3;
        }

        // Store
        bw.c_norm[kT] = c_norm;
        neg_log_likelihood_sum += log(c_norm);

        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            c_norm0 = 0;
            c_norm1 = 0;
            c_norm2 = 0;
            c_norm3 = 0;
            observations = bw.observations[kT + t];
            kTN = (kT + t)*bw.N;

            for (size_t n0 = 0; n0 < bw.N; n0+=4) {

                // Load
                alpha_sum0 = 0.0;
                alpha_sum1 = 0.0;
                alpha_sum2 = 0.0;
                alpha_sum3 = 0.0;
                emit_prob0 = bw.emit_prob[(n0 + 0)*bw.M + observations];
                emit_prob1 = bw.emit_prob[(n0 + 1)*bw.M + observations];
                emit_prob2 = bw.emit_prob[(n0 + 2)*bw.M + observations];
                emit_prob3 = bw.emit_prob[(n0 + 3)*bw.M + observations];

                for (size_t n1 = 0; n1 < bw.N; n1++) {

                    // Load
                    alpha = bw.alpha[kTN - bw.N + n1];
                    trans_prob0 = bw.trans_prob[n1*bw.N + n0 + 0];
                    trans_prob1 = bw.trans_prob[n1*bw.N + n0 + 1];
                    trans_prob2 = bw.trans_prob[n1*bw.N + n0 + 2];
                    trans_prob3 = bw.trans_prob[n1*bw.N + n0 + 3];

                    // Calculate
                    alpha_sum0 += alpha * trans_prob0;
                    alpha_sum1 += alpha * trans_prob1;
                    alpha_sum2 += alpha * trans_prob2;
                    alpha_sum3 += alpha * trans_prob3;
                }

                // Calculate
                alpha0 = emit_prob0 * alpha_sum0;
                alpha1 = emit_prob1 * alpha_sum1;
                alpha2 = emit_prob2 * alpha_sum2;
                alpha3 = emit_prob3 * alpha_sum3;
                c_norm0 += alpha0;
                c_norm1 += alpha1;
                c_norm2 += alpha2;
                c_norm3 += alpha3;

                // Store
                bw.alpha[kTN + n0 + 0] = alpha0;
                bw.alpha[kTN + n0 + 1] = alpha1;
                bw.alpha[kTN + n0 + 2] = alpha2;
                bw.alpha[kTN + n0 + 3] = alpha3;
            }

            // Calculate
            c_norm = 1.0/(c_norm0 + c_norm1 + c_norm2 + c_norm3);

            for (size_t n0 = 0; n0 < bw.N; n0++) {

                // Load
                alpha = bw.alpha[kTN + n0];

                // Calculate
                alpha *= c_norm;

                // Store
                bw.alpha[kTN + n0] = alpha;
            }

            // Store
            bw.c_norm[kT + t] = c_norm;
            neg_log_likelihood_sum += log(c_norm);
        }
    }
}

static inline void backward_step(const BWdata& bw, const size_t& k) {
    // Init
    double alpha, c_norm, gamma, sigma;
    double beta_sum0, beta_temp0, beta0, emit_prob0, trans_prob0;
    double beta_sum1, beta_temp1, beta1, emit_prob1, trans_prob1;
    double beta_sum2, beta_temp2, beta2, emit_prob2, trans_prob2;
    double beta_sum3, beta_temp3, beta3, emit_prob3, trans_prob3;

    size_t observations, kTN, kTNN, kT, nN;
    // t = bw.T, base case
    kTN = (k*bw.T + (bw.T-1))*bw.N;

    // Load
    memcpy(bw.ggamma + kTN, bw.alpha + kTN, bw.N * sizeof(double));
    c_norm = bw.c_norm[k*bw.T + (bw.T-1)];
    for (size_t n = 0; n < bw.N; n++) {
        // Store
        bw.beta[kTN + n] = c_norm;
    }

    // Recursion step
    kT = k*bw.T;
    for (int t = bw.T-2; t >= 0; t--) {
        // Load
        observations = bw.observations[kT + (t+1)];
        c_norm = bw.c_norm[kT + t];
        kTN = (kT + t) * bw.N;

        for (size_t n0 = 0; n0 < bw.N; n0++) {

            // Load
            beta_sum0 = 0.0;
            beta_sum1 = 0.0;
            beta_sum2 = 0.0;
            beta_sum3 = 0.0;
            alpha = bw.alpha[(kT + t)*bw.N + n0];
            kTNN = (kTN + n0)*bw.N;

            for (size_t n1 = 0; n1 < bw.N; n1+=4) {
                // Load
                beta0 = bw.beta[kTN + bw.N + n1 + 0];
                beta1 = bw.beta[kTN + bw.N + n1 + 1];
                beta2 = bw.beta[kTN + bw.N + n1 + 2];
                beta3 = bw.beta[kTN + bw.N + n1 + 3];
                trans_prob0 = bw.trans_prob[n0 * bw.N + n1 + 0];
                trans_prob1 = bw.trans_prob[n0 * bw.N + n1 + 1];
                trans_prob2 = bw.trans_prob[n0 * bw.N + n1 + 2];
                trans_prob3 = bw.trans_prob[n0 * bw.N + n1 + 3];
                emit_prob0 = bw.emit_prob[(n1 + 0) * bw.M + observations];
                emit_prob1 = bw.emit_prob[(n1 + 1) * bw.M + observations];
                emit_prob2 = bw.emit_prob[(n1 + 2) * bw.M + observations];
                emit_prob3 = bw.emit_prob[(n1 + 3) * bw.M + observations];

                // Calculate & store
                beta_temp0 = beta0 * trans_prob0 * emit_prob0;
                beta_temp1 = beta1 * trans_prob1 * emit_prob1;
                beta_temp2 = beta2 * trans_prob2 * emit_prob2;
                beta_temp3 = beta3 * trans_prob3 * emit_prob3;
                beta_sum0 += beta_temp0;
                beta_sum1 += beta_temp1;
                beta_sum2 += beta_temp2;
                beta_sum3 += beta_temp3;
                bw.sigma[kTNN + n1 + 0] = alpha * beta_temp0;
                bw.sigma[kTNN + n1 + 1] = alpha * beta_temp1;
                bw.sigma[kTNN + n1 + 2] = alpha * beta_temp2;
                bw.sigma[kTNN + n1 + 3] = alpha * beta_temp3;
            }

            // Calculate & store
            bw.beta[kTN + n0] = (beta_sum0 + beta_sum1 + beta_sum2 + beta_sum3) * c_norm;
            bw.ggamma[kTN + n0] = alpha * (beta_sum0 + beta_sum1 + beta_sum2 + beta_sum3);
        }
    }
}

static inline void compute_gamma(const BWdata& bw, const size_t& k) {
    double g_sum0, g_sum1, g_sum2, g_sum3;
    double s_sum0, s_sum1, s_sum2, s_sum3, s_sum4, s_sum5, s_sum6, s_sum7, s_sum8, s_sum9, s_sum10, s_sum11, s_sum12, s_sum13, s_sum14, s_sum15, s_sum16;
    for (size_t n0 = 0; n0 < bw.N; n0+=4) {
        g_sum0 = 0.0;
        g_sum1 = 0.0;
        g_sum2 = 0.0;
        g_sum3 = 0.0;

        // blocking here if you want to include n1 in this loop instead of after this loop
        for (size_t t = 0; t < bw.T-1; t++) {
            g_sum0 += bw.ggamma[(k*bw.T + t)*bw.N + n0+0];
            g_sum1 += bw.ggamma[(k*bw.T + t)*bw.N + n0+1];
            g_sum2 += bw.ggamma[(k*bw.T + t)*bw.N + n0+2];
            g_sum3 += bw.ggamma[(k*bw.T + t)*bw.N + n0+3];
        }
        // Store
        bw.gamma_sum[k*bw.N + n0+0] = g_sum0;
        bw.gamma_sum[k*bw.N + n0+1] = g_sum1;
        bw.gamma_sum[k*bw.N + n0+2] = g_sum2;
        bw.gamma_sum[k*bw.N + n0+3] = g_sum3;

        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            s_sum0 = 0.0;
            s_sum1 = 0.0;
            s_sum2 = 0.0;
            s_sum3 = 0.0;
            s_sum4 = 0.0;
            s_sum5 = 0.0;
            s_sum6 = 0.0;
            s_sum7 = 0.0;
            s_sum8 = 0.0;
            s_sum9 = 0.0;
            s_sum10 = 0.0;
            s_sum11 = 0.0;
            s_sum12 = 0.0;
            s_sum13 = 0.0;
            s_sum14 = 0.0;
            s_sum15 = 0.0;

            for (size_t t = 0; t < bw.T-1; t++) {
                // Calculation
                s_sum0 += bw.sigma[((k*bw.T + t)*bw.N + n0+0) * bw.N + n1+0];
                s_sum1 += bw.sigma[((k*bw.T + t)*bw.N + n0+0) * bw.N + n1+1];
                s_sum2 += bw.sigma[((k*bw.T + t)*bw.N + n0+0) * bw.N + n1+2];
                s_sum3 += bw.sigma[((k*bw.T + t)*bw.N + n0+0) * bw.N + n1+3];
                s_sum4 += bw.sigma[((k*bw.T + t)*bw.N + n0+1) * bw.N + n1+0];
                s_sum5 += bw.sigma[((k*bw.T + t)*bw.N + n0+1) * bw.N + n1+1];
                s_sum6 += bw.sigma[((k*bw.T + t)*bw.N + n0+1) * bw.N + n1+2];
                s_sum7 += bw.sigma[((k*bw.T + t)*bw.N + n0+1) * bw.N + n1+3];
                s_sum8 += bw.sigma[((k*bw.T + t)*bw.N + n0+2) * bw.N + n1+0];
                s_sum9 += bw.sigma[((k*bw.T + t)*bw.N + n0+2) * bw.N + n1+1];
                s_sum10 += bw.sigma[((k*bw.T + t)*bw.N + n0+2) * bw.N + n1+2];
                s_sum11 += bw.sigma[((k*bw.T + t)*bw.N + n0+2) * bw.N + n1+3];
                s_sum12 += bw.sigma[((k*bw.T + t)*bw.N + n0+3) * bw.N + n1+0];
                s_sum13 += bw.sigma[((k*bw.T + t)*bw.N + n0+3) * bw.N + n1+1];
                s_sum14 += bw.sigma[((k*bw.T + t)*bw.N + n0+3) * bw.N + n1+2];
                s_sum15 += bw.sigma[((k*bw.T + t)*bw.N + n0+3) * bw.N + n1+3];
            }

            // Store
            bw.sigma_sum[(k*bw.N + n0+0) * bw.N + n1+0] = s_sum0;
            bw.sigma_sum[(k*bw.N + n0+0) * bw.N + n1+1] = s_sum1;
            bw.sigma_sum[(k*bw.N + n0+0) * bw.N + n1+2] = s_sum2;
            bw.sigma_sum[(k*bw.N + n0+0) * bw.N + n1+3] = s_sum3;
            bw.sigma_sum[(k*bw.N + n0+1) * bw.N + n1+0] = s_sum4;
            bw.sigma_sum[(k*bw.N + n0+1) * bw.N + n1+1] = s_sum5;
            bw.sigma_sum[(k*bw.N + n0+1) * bw.N + n1+2] = s_sum6;
            bw.sigma_sum[(k*bw.N + n0+1) * bw.N + n1+3] = s_sum7;
            bw.sigma_sum[(k*bw.N + n0+2) * bw.N + n1+0] = s_sum8;
            bw.sigma_sum[(k*bw.N + n0+2) * bw.N + n1+1] = s_sum9;
            bw.sigma_sum[(k*bw.N + n0+2) * bw.N + n1+2] = s_sum10;
            bw.sigma_sum[(k*bw.N + n0+2) * bw.N + n1+3] = s_sum11;
            bw.sigma_sum[(k*bw.N + n0+3) * bw.N + n1+0] = s_sum12;
            bw.sigma_sum[(k*bw.N + n0+3) * bw.N + n1+1] = s_sum13;
            bw.sigma_sum[(k*bw.N + n0+3) * bw.N + n1+2] = s_sum14;
            bw.sigma_sum[(k*bw.N + n0+3) * bw.N + n1+3] = s_sum15;
        }
    }
}


static inline void update_trans_prob(const BWdata& bw) {
    //Init (init_prob)
    double g0_sum, denominator_sum_n, denominator_sum_inv;
    double numerator_sum0, numerator_sum1, numerator_sum2, numerator_sum3;
    double K_inv = 1.0/bw.K;

    for (size_t n = 0; n < bw.N; n++) {
        denominator_sum_n = 0;
        g0_sum = 0;

        for (size_t k = 0; k < bw.K; k++) {
            denominator_sum_n += bw.gamma_sum[k*bw.N + n];
            g0_sum += bw.ggamma[(k*bw.T)*bw.N + n];
        }

        denominator_sum[n] = 1.0/denominator_sum_n;
        bw.init_prob[n] = g0_sum*K_inv;
    }

    for (size_t n0 = 0; n0 < bw.N; n0++) {
        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            // Init (trans_prob)
            numerator_sum0 = 0.0;
            numerator_sum1 = 0.0;
            numerator_sum2 = 0.0;
            numerator_sum3 = 0.0;

            for (size_t k = 0; k < bw.K; k++) {
                // Calculate (trans_prob)
                numerator_sum0 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+0];
                numerator_sum1 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+1];
                numerator_sum2 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+2];
                numerator_sum3 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+3];
            }

            // Store (trans_prob)
            bw.trans_prob[n0*bw.N + n1+0] = numerator_sum0*denominator_sum[n0];
            bw.trans_prob[n0*bw.N + n1+1] = numerator_sum1*denominator_sum[n0];
            bw.trans_prob[n0*bw.N + n1+2] = numerator_sum2*denominator_sum[n0];
            bw.trans_prob[n0*bw.N + n1+3] = numerator_sum3*denominator_sum[n0];
        }
    }
}


static inline void update_emit_prob(const BWdata& bw) {
    // Init
    double denominator_sum0, denominator_sum1, denominator_sum2, denominator_sum3, denominator_sum4, denominator_sum5, denominator_sum6, denominator_sum7;
    double ggamma_cond_sum_tot0, ggamma_cond_sum_tot1, ggamma_cond_sum_tot2, ggamma_cond_sum_tot3;
    double ggamma_cond_sum0, ggamma_cond_sum1, ggamma_cond_sum2, ggamma_cond_sum3;


    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n+=4) {
            bw.gamma_sum[k*bw.N + n+0] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+0];
            bw.gamma_sum[k*bw.N + n+1] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+1];
            bw.gamma_sum[k*bw.N + n+2] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+2];
            bw.gamma_sum[k*bw.N + n+3] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+3];
        }
    }

    // denominator_sum (top-down)
    for (size_t n = 0; n < bw.N; n += 8){
        denominator_sum0 = bw.gamma_sum[0*bw.N + n+0];
        denominator_sum1 = bw.gamma_sum[0*bw.N + n+1];
        denominator_sum2 = bw.gamma_sum[0*bw.N + n+2];
        denominator_sum3 = bw.gamma_sum[0*bw.N + n+3];
        denominator_sum4 = bw.gamma_sum[0*bw.N + n+4];
        denominator_sum5 = bw.gamma_sum[0*bw.N + n+5];
        denominator_sum6 = bw.gamma_sum[0*bw.N + n+6];
        denominator_sum7 = bw.gamma_sum[0*bw.N + n+7];

        for (size_t k = 1; k < bw.K; k++) {
            denominator_sum0 += bw.gamma_sum[k*bw.N + n+0];
            denominator_sum1 += bw.gamma_sum[k*bw.N + n+1];
            denominator_sum2 += bw.gamma_sum[k*bw.N + n+2];
            denominator_sum3 += bw.gamma_sum[k*bw.N + n+3];
            denominator_sum4 += bw.gamma_sum[k*bw.N + n+4];
            denominator_sum5 += bw.gamma_sum[k*bw.N + n+5];
            denominator_sum6 += bw.gamma_sum[k*bw.N + n+6];
            denominator_sum7 += bw.gamma_sum[k*bw.N + n+7];
        }

        denominator_sum[n+0] = denominator_sum0;
        denominator_sum[n+1] = denominator_sum1;
        denominator_sum[n+2] = denominator_sum2;
        denominator_sum[n+3] = denominator_sum3;
        denominator_sum[n+4] = denominator_sum4;
        denominator_sum[n+5] = denominator_sum5;
        denominator_sum[n+6] = denominator_sum6;
        denominator_sum[n+7] = denominator_sum7;
    }

    // numerator_sum
    for (size_t m = 0; m < bw.M; m++) {
        for (size_t n = 0; n < bw.N; n+=4) {
            ggamma_cond_sum_tot0 = 0.0;
            ggamma_cond_sum_tot1 = 0.0;
            ggamma_cond_sum_tot2 = 0.0;
            ggamma_cond_sum_tot3 = 0.0;

            for (size_t k = 0; k < bw.K; k++) {
                ggamma_cond_sum0 = 0.0;
                ggamma_cond_sum1 = 0.0;
                ggamma_cond_sum2 = 0.0;
                ggamma_cond_sum3 = 0.0;

                for (size_t t = 0; t < bw.T; t+=4) {
                    if (bw.observations[k*bw.T + t] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t)*bw.N + n+3];
                    }

                    if (bw.observations[k*bw.T + t+1] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+3];
                    }

                    if (bw.observations[k*bw.T + t+2] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+3];
                    }

                    if (bw.observations[k*bw.T + t+3] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+3];
                    }
                }
                ggamma_cond_sum_tot0 += ggamma_cond_sum0;
                ggamma_cond_sum_tot1 += ggamma_cond_sum1;
                ggamma_cond_sum_tot2 += ggamma_cond_sum2;
                ggamma_cond_sum_tot3 += ggamma_cond_sum3;
            }
            numerator_sum[(n+0)*bw.M + m] = ggamma_cond_sum_tot0;
            numerator_sum[(n+1)*bw.M + m] = ggamma_cond_sum_tot1;
            numerator_sum[(n+2)*bw.M + m] = ggamma_cond_sum_tot2;
            numerator_sum[(n+3)*bw.M + m] = ggamma_cond_sum_tot3;
        }
    }

    // emit_prob
    for (size_t n = 0; n < bw.N; n++) {
        double denominator_sum_inv = 1.0/denominator_sum[n];
        for (size_t m = 0; m < bw.M; m+=4) {
            bw.emit_prob[n*bw.M + m+0] = numerator_sum[n*bw.M + m+0] * denominator_sum_inv;
            bw.emit_prob[n*bw.M + m+1] = numerator_sum[n*bw.M + m+1] * denominator_sum_inv;
            bw.emit_prob[n*bw.M + m+2] = numerator_sum[n*bw.M + m+2] * denominator_sum_inv;
            bw.emit_prob[n*bw.M + m+3] = numerator_sum[n*bw.M + m+3] * denominator_sum_inv;
        }
    }
}
