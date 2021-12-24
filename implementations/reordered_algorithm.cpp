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


static void forward_step_jc(const BWdata& bw, const int& i, const size_t& iter, size_t& res, bool& first, double& neg_log_likelihood_sum_old);
static void update_trans_prob_jc(const BWdata& bw);
static size_t comp_bw_scalar_jc1(const BWdata& bw);


REGISTER_FUNCTION(comp_bw_scalar_jc1, "jc-reordered", "Reordering of the computation in the algorithm");


size_t comp_bw_scalar_jc1(const BWdata& bw){

    size_t iter = 0;
    size_t res = 0;
    double neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {

        iter++;

        forward_step_jc(bw, i, iter, res, first, neg_log_likelihood_sum_old);
        update_trans_prob_jc(bw);
    }

    return res;
}


inline void forward_step_jc(const BWdata& bw, const int& i, const size_t& iter, size_t& res, bool& first, double& neg_log_likelihood_sum_old) {
    double neg_log_likelihood_sum = 0.0;
    for (size_t k = 0; k < bw.K; k++) {
        // t = 0, base case
        double c_norm = 0;
        double alpha = 0;
        // can be blocked
        for (size_t n = 0; n < bw.N; n++) {
            alpha = bw.init_prob[n]*bw.emit_prob[n*bw.M + bw.observations[k*bw.T + 0]];
            c_norm += alpha;
            bw.alpha[(k*bw.T + 0)*bw.N + n] = alpha;
        }

        c_norm = 1.0/c_norm;
        for (size_t n = 0; n < bw.N; n++){
            bw.alpha[(k*bw.T + 0)*bw.N + n] *= c_norm;
        }
        bw.c_norm[k*bw.T + 0] = c_norm;
        neg_log_likelihood_sum += log(c_norm);

        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            c_norm = 0;
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                double alpha_temp = 0.0;
                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    alpha_temp += bw.alpha[(k*bw.T + (t-1))*bw.N + n1]*bw.trans_prob[n1*bw.N + n0];
                }
                alpha = bw.emit_prob[n0*bw.M + bw.observations[k*bw.T + t]] * alpha_temp;
                c_norm += alpha;
                bw.alpha[(k*bw.T + t)*bw.N + n0] = alpha;
            }
            c_norm = 1.0/c_norm;
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                bw.alpha[(k*bw.T + t)*bw.N + n0] *= c_norm;
            }
            bw.c_norm[k*bw.T + t] = c_norm;

            neg_log_likelihood_sum += log(c_norm);
        }
        c_norm = bw.c_norm[k*bw.T + (bw.T-1)];
        for (size_t n = 0; n < bw.N; n++) {
            bw.beta[(k*bw.T + (bw.T-1))*bw.N + n] = c_norm;
            bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n] = bw.alpha[(k*bw.T + (bw.T-1))*bw.N + n];
        }

        for (int t = bw.T-2; t >= 0; t--) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                double beta_temp = 0.0;
                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    beta_temp += bw.beta[(k*bw.T + (t+1))*bw.N + n1] * bw.trans_prob[n0*bw.N + n1] * bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                    bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1] = \
                        bw.alpha[(k*bw.T + t)*bw.N + n0]*bw.trans_prob[n0*bw.N + n1]*bw.beta[(k*bw.T + (t+1))*bw.N + n1]*bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                }
                bw.beta[(k*bw.T + t)*bw.N + n0] = beta_temp * bw.c_norm[k*bw.T + t];
                double gamma = bw.alpha[(k*bw.T + t)*bw.N + n0] * beta_temp;
                bw.ggamma[(k*bw.T + t)*bw.N + n0] = gamma;


            }
        }
        for (size_t n = 0; n < bw.N; n++) {
            double g_sum = 0.0;
            for (size_t t = 0; t < bw.T-1; t++) {
                g_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];
            }
            bw.gamma_sum[k*bw.N + n] = g_sum;

            for (size_t n1 = 0; n1 < bw.N; n1++) {
                double s_sum = 0.0;
                for (size_t t = 0; t < bw.T-1; t++) {
                    s_sum += bw.sigma[((k*bw.T + t)*bw.N + n)*bw.N + n1];
                }
                bw.sigma_sum[(k*bw.N + n)*bw.N + n1] = s_sum;

            }
        }
    }

    // Neg log likelihood sum check
    bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;

    if (first && i > 0 && fabs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < EPSILON){
        first = false;
        res = iter;
    }

    neg_log_likelihood_sum_old = neg_log_likelihood_sum;
}


inline void update_trans_prob_jc(const BWdata& bw) {
    for (size_t n0 = 0; n0 < bw.N; n0++) {
        for (size_t n1 = 0; n1 < bw.N; n1++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            double g0_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                numerator_sum += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1];
                denominator_sum += bw.gamma_sum[k*bw.N + n0];
                if(n0 == 0) {
                    g0_sum += bw.ggamma[(k * bw.T + 0) * bw.N + n1];
                }

            }
            bw.trans_prob[n0*bw.N + n1] = numerator_sum / denominator_sum;
            if(n0 == 0) {
                bw.init_prob[n1] = g0_sum / bw.K;
            }
        }

        // Update emit_prob
        for (size_t k = 0; k < bw.K; k++) {
            bw.gamma_sum[k*bw.N + n0] += bw.ggamma[(k * bw.T + (bw.T - 1)) * bw.N + n0];
        }

        for (size_t m = 0; m < bw.M; m++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                double ggamma_cond_sum = 0.0;
                for (size_t t = 0; t < bw.T; t++) {
                    if (bw.observations[k*bw.T + t] == m) {
                        ggamma_cond_sum += bw.ggamma[(k*bw.T + t)*bw.N + n0];
                    }
                }
                numerator_sum += ggamma_cond_sum;
                denominator_sum += bw.gamma_sum[k*bw.N + n0];
            }
            bw.emit_prob[n0 * bw.M + m] = numerator_sum / denominator_sum;
        }
    }
}

