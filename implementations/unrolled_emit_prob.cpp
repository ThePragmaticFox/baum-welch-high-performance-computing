/*
    Best optimized implementation
    Final and best possible optimization, combination of all previous approaches!

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


static void forward_step(const BWdata& bw);
static void backward_step(const BWdata& bw);
static void compute_gamma(const BWdata& bw);
static void compute_sigma(const BWdata& bw);
static void update_init_prob(const BWdata& bw);
static void update_trans_prob(const BWdata& bw);
static void update_emit_prob(const BWdata& bw);
static size_t comp_bw_emit_unrolled(const BWdata& bw);


REGISTER_FUNCTION(comp_bw_emit_unrolled, "unroll-emitprob", "Unrolled ");
static double* denominator_sum = nullptr;
static double* numerator_sum = nullptr;

size_t comp_bw_emit_unrolled(const BWdata& bw){

    size_t iter = 0;
    size_t res = 0;
    double neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;
    denominator_sum = (double *)aligned_alloc(32,bw.N * sizeof(double));
    numerator_sum = (double *)aligned_alloc(32,bw.N*bw.M * sizeof(double));

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {

        iter++;

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        compute_sigma(bw);
        update_init_prob(bw);
        update_trans_prob(bw);
        update_emit_prob(bw);

        double neg_log_likelihood_sum = 0.0;
        for (size_t k = 0; k < bw.K; k++) {
            for (size_t t = 0; t < bw.T; t++) {
                neg_log_likelihood_sum = neg_log_likelihood_sum + log(bw.c_norm[k*bw.T + t]);
            }
        }
        bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;

        if (first && i > 0 && fabs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < EPSILON){
            first = false;
            res = iter;
        }

        neg_log_likelihood_sum_old = neg_log_likelihood_sum;

    }
    free(denominator_sum);
    free(numerator_sum);

    return res;
}


inline void forward_step(const BWdata& bw) {
    for (size_t k = 0; k < bw.K; k++) {
        // t = 0, base case
        bw.c_norm[k*bw.T + 0] = 0;
        for (size_t n = 0; n < bw.N; n++) {
            bw.alpha[(k*bw.T + 0)*bw.N + n] = bw.init_prob[n]*bw.emit_prob[n*bw.M + bw.observations[k*bw.T + 0]];
            bw.c_norm[k*bw.T + 0] += bw.alpha[(k*bw.T + 0)*bw.N + n];
        }

        bw.c_norm[k*bw.T + 0] = 1.0/bw.c_norm[k*bw.T + 0];
        for (size_t n = 0; n < bw.N; n++){
	        bw.alpha[(k*bw.T + 0)*bw.N + n] *= bw.c_norm[k*bw.T + 0];
	    }

        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            bw.c_norm[k*bw.T + t] = 0;
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                double alpha_temp = 0.0;
                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    alpha_temp += bw.alpha[(k*bw.T + (t-1))*bw.N + n1]*bw.trans_prob[n1*bw.N + n0];
                }
                bw.alpha[(k*bw.T + t)*bw.N + n0] = bw.emit_prob[n0*bw.M + bw.observations[k*bw.T + t]] * alpha_temp;
                bw.c_norm[k*bw.T + t] += bw.alpha[(k*bw.T + t)*bw.N + n0];
            }
            bw.c_norm[k*bw.T + t] = 1.0/bw.c_norm[k*bw.T + t];
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                bw.alpha[(k*bw.T + t)*bw.N + n0] *= bw.c_norm[k*bw.T + t];
            }
        }
    }
}


inline void backward_step(const BWdata& bw) {
    for (size_t k = 0; k < bw.K; k++) {
        // t = bw.T, base case
        for (size_t n = 0; n < bw.N; n++) {
            bw.beta[(k*bw.T + (bw.T-1))*bw.N + n] = bw.c_norm[k*bw.T + (bw.T-1)];
            bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n] = bw.alpha[(k*bw.T + (bw.T-1))*bw.N + n];
        }

        // recursion step
        for (int t = bw.T-2; t >= 0; t--) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                double beta_temp = 0.0;
                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    beta_temp += bw.beta[(k*bw.T + (t+1))*bw.N + n1] * bw.trans_prob[n0*bw.N + n1] * bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                }
                bw.beta[(k*bw.T + t)*bw.N + n0] = beta_temp * bw.c_norm[k*bw.T + t];
                bw.ggamma[(k*bw.T + t)*bw.N + n0] = bw.alpha[(k*bw.T + t)*bw.N + n0] * beta_temp;
            }
        }
    }
}


inline void compute_gamma(const BWdata& bw) {
    //for (size_t k = 0; k < bw.K; k++) {
    //    for (size_t t = 0; t < bw.T; t++) {
    //        for (size_t n = 0; n < bw.N; n++) {
    //            
    //        }
    //    }
    //}


    // sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob)
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            double g_sum = 0.0;
            for (size_t t = 0; t < bw.T-1; t++) {
                g_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];
            }
            bw.gamma_sum[k*bw.N + n] = g_sum;
        }
    }
}


inline void compute_sigma(const BWdata& bw) {
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T-1; t++) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1] = \
                        bw.alpha[(k*bw.T + t)*bw.N + n0]*bw.trans_prob[n0*bw.N + n1]*bw.beta[(k*bw.T + (t+1))*bw.N + n1]*bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                }
            }
        }

        // sum up bw.sigma (from t = 0 to bw.T-1)
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                double s_sum = 0.0;
                for (size_t t = 0; t < bw.T-1; t++) {
                    s_sum += bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1];
                }
                bw.sigma_sum[(k*bw.N + n0)*bw.N + n1] = s_sum;
            }
        }
    }
}


inline void update_init_prob(const BWdata& bw) {
    for (size_t n = 0; n < bw.N; n++) {
        double g0_sum = 0.0;
        for (size_t k = 0; k < bw.K; k++) {
            g0_sum += bw.ggamma[(k*bw.T + 0)*bw.N + n];
        }
        bw.init_prob[n] = g0_sum/bw.K;
    }
}


inline void update_trans_prob(const BWdata& bw) {
    for (size_t n0 = 0; n0 < bw.N; n0++) {
        for (size_t n1 = 0; n1 < bw.N; n1++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                numerator_sum += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1];
                denominator_sum += bw.gamma_sum[k*bw.N + n0];
            }
            bw.trans_prob[n0*bw.N + n1] = numerator_sum / denominator_sum;
        }
    }
}


inline void update_emit_prob(const BWdata& bw) {
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
