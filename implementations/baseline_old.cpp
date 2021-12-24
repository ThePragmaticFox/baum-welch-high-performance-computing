/*
    Baseline implementation

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------

    Make sure you understand it! Refer to
    https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
    https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf
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
static size_t comp_bw(const BWdata& bw);


SET_BASELINE(comp_bw, "Baseline");


size_t comp_bw(const BWdata& bw){

    size_t iter = 0;

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
    }

    return iter;
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
        }

        // recursion step
        for (int t = bw.T-2; t >= 0; t--) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                double beta_temp = 0.0;
                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    beta_temp += bw.beta[(k*bw.T + (t+1))*bw.N + n1] * bw.trans_prob[n0*bw.N + n1] * bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                }
                bw.beta[(k*bw.T + t)*bw.N + n0] = beta_temp * bw.c_norm[k*bw.T + t];
            }
        }
    }
}


inline void compute_gamma(const BWdata& bw) {
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                bw.ggamma[(k*bw.T + t)*bw.N + n] = bw.alpha[(k*bw.T + t)*bw.N + n] * bw.beta[(k*bw.T + t)*bw.N + n] / bw.c_norm[k*bw.T + t];
            }
        }
    }

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
    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            bw.gamma_sum[k*bw.N + n] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n];
        }
    }
    // update bw.emit_prob
    for (size_t n = 0; n < bw.N; n++) {
        for (size_t m = 0; m < bw.M; m++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                double ggamma_cond_sum = 0.0;
                for (size_t t = 0; t < bw.T; t++) {
                    if (bw.observations[k*bw.T + t] == m) {
                        ggamma_cond_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];
                    }
                }
                numerator_sum += ggamma_cond_sum;
                denominator_sum += bw.gamma_sum[k*bw.N + n];
            }
            bw.emit_prob[n*bw.M + m] = numerator_sum / denominator_sum;
        }
    }
}
