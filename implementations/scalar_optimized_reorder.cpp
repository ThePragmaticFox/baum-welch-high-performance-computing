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


static void forward_step(const BWdata& bw);
static void backward_step(const BWdata& bw);
static void compute_gamma(const BWdata& bw);
static void compute_sigma(const BWdata& bw);
static void update_init_prob(const BWdata& bw);
static void update_trans_prob(const BWdata& bw);
static void update_emit_prob(const BWdata& bw);
static size_t comp_bw_scalar_reorder(const BWdata& bw);


REGISTER_FUNCTION(comp_bw_scalar_reorder, "scalar-reorder", "Scalar Optimized: reorder calculations");


size_t comp_bw_scalar_reorder(const BWdata& bw){

    size_t iter = 0;
    size_t res = 0;
    double neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {
        iter++;

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        //compute_sigma(bw);
        //update_init_prob(bw);
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

    return res;
}


inline void forward_step(const BWdata& bw) {
    //Init
    double c_norm, alpha, alpha_sum, init_prob, emit_prob, trans_prob;

    size_t kTN, kT;

    for (size_t k = 0; k < bw.K; k++) {
        // t = 0, base case

        // Init
        c_norm = 0;
        kT = k*bw.T;
        kTN = kT*bw.N;
        size_t observations = bw.observations[kT];


        for (size_t n = 0; n < bw.N; n++){
            // Load
            init_prob = bw.init_prob[n];
            emit_prob = bw.emit_prob[n*bw.M + observations];

            // Calculate
            alpha = init_prob * emit_prob;
            c_norm += alpha;

            // Store
            bw.alpha[kTN + n] = alpha;
        }

        // Calculate
        c_norm = 1.0/c_norm;


        for (size_t n = 0; n < bw.N; n++){
            // Load
            alpha = bw.alpha[kTN + n];

            // Calculate
            alpha *= c_norm;

            // Store
            bw.alpha[kTN + n] = alpha;
        }

        // Store
        bw.c_norm[kT] = c_norm;

        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            c_norm = 0;
            observations = bw.observations[kT + t];
            kTN = (kT + t)*bw.N;

            for (size_t n0 = 0; n0 < bw.N; n0++) {

                // Load
                alpha_sum = 0.0;
                emit_prob = bw.emit_prob[n0*bw.M + observations];

                for (size_t n1 = 0; n1 < bw.N; n1++) {

                    // Load
                    alpha = bw.alpha[kTN - bw.N + n1];
                    trans_prob = bw.trans_prob[n1*bw.N + n0];

                    // Calculate
                    alpha_sum += alpha * trans_prob;
                }

                // Calculate
                alpha = emit_prob * alpha_sum;
                c_norm += alpha;

                // Store
                bw.alpha[kTN + n0] = alpha;
            }

            // Calculate
            c_norm = 1.0/c_norm;

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
        }

    }
}


inline void backward_step(const BWdata& bw) {
    // Init
    double alpha, beta, beta_sum, c_norm, gamma, sigma, emit_prob, trans_prob;

    size_t observations, kTN, kTNN, kT, nN;

    for (size_t k = 0; k < bw.K; k++) {
        // t = bw.T, base case
        kTN = (k*bw.T + (bw.T-1))*bw.N;

        // Load
        c_norm = bw.c_norm[k*bw.T + (bw.T-1)];
        for (size_t n = 0; n < bw.N; n++) {
            // Load
            alpha = bw.alpha[kTN + n];

            // Store
            bw.beta[kTN + n] = c_norm;
            bw.ggamma[kTN + n] = alpha;

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
                beta_sum = 0.0;
                alpha = bw.alpha[(kT + t)*bw.N + n0];
                nN = n0 * bw.N;
                kTNN = (kTN + n0)*bw.N;

                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    // Load
                    beta = bw.beta[kTN + bw.N + n1];
                    trans_prob = bw.trans_prob[nN + n1];
                    emit_prob = bw.emit_prob[n1 * bw.M + observations];

                    // Calculate & store
                    beta_sum += beta * trans_prob * emit_prob;
                    bw.sigma[kTNN + n1] = alpha * trans_prob * beta * emit_prob;
                }

                // Calculate & store
                bw.beta[kTN + n0] = beta_sum * c_norm;
                bw.ggamma[kTN + n0] = alpha * beta_sum;
            }
        }
    }
}


// NOTE: This computes gamma and sigma
inline void compute_gamma(const BWdata& bw) {
    // ====== Sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob) =====
    for (size_t k = 0; k < bw.K; k++) {
        // Init
        double g_sum, s_sum;

        // ----- gamma -----
        // TODO: Loop switch+blocking if N << T or vice versa i don't know
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            // Init
            g_sum = bw.ggamma[(k*bw.T + 0)*bw.N + n0];

            for (size_t t = 1; t < bw.T-1; t++) {
                // Calculation
                g_sum += bw.ggamma[(k*bw.T + t)*bw.N + n0];
            }

            // Store
            bw.gamma_sum[k*bw.N + n0] = g_sum;

            // ----- sigma -----
            // TODO: Loop switch+blocking if N << T or vice versa i don't know
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                // Init
                s_sum = bw.sigma[((k*bw.T + 0)*bw.N + n0) * bw.N + n1];

                for (size_t t = 1; t < bw.T-1; t++) {
                    // Calculation
                    s_sum += bw.sigma[((k*bw.T + t)*bw.N + n0) * bw.N + n1];
                }

                // Store
                bw.sigma_sum[(k*bw.N + n0) * bw.N + n1] = s_sum;
            }
        }
    }
}

// unused
inline void compute_sigma(const BWdata& bw) {
    // Init
    double sigma, sigma_sum, alpha, beta, trans_prob, emit_prob;
    size_t observations, kTN, nN, kTNN, kNN, kT, kN;

    for (size_t k = 0; k < bw.K; k++) {

        kT = k*bw.T;
        kN = k*bw.N;
        /*
        for (size_t t = 0; t < bw.T-1; t++) {
            //Load
            observations = bw.observations[k*bw.T + (t+1)];

            for (size_t n0 = 0; n0 < bw.N; n0++) {
                // Load
                alpha = bw.alpha[(k*bw.T + t)*bw.N + n0];
                kTN = (kT + (t+1))*bw.N;
                nN = n0*bw.N ;
                kTNN = ((kT + t)*bw.N + n0)*bw.N;

                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    // Load
                    beta = bw.beta[kTN + n1];
                    emit_prob = bw.emit_prob[n1*bw.M + observations];
                    trans_prob = bw.trans_prob[nN + n1];

                    // Calculate
                    sigma = alpha*trans_prob*beta*emit_prob;

                    // Store
                    bw.sigma[kTNN + n1] = sigma;
                }
            }
        }
        */
        // sum up bw.sigma (from t = 0 to bw.T-2)
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            kNN = (kN + n0)*bw.N;
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                // Init
                sigma_sum = 0.0;

                for (size_t t = 0; t < bw.T-1; t++) {
                    // Calculate
                    sigma_sum += bw.sigma[((kT + t) * bw.N + n0) * bw.N + n1];
                }

                // Store
                bw.sigma_sum[kNN + n1] = sigma_sum;
            }
        }
    }
}

//unused
inline void update_init_prob(const BWdata& bw) {
    // Init
    double g0_sum;

    for (size_t n = 0; n < bw.N; n++) {

        // Init
        g0_sum = 0.0;

        for (size_t k = 0; k < bw.K; k++) {
            // Calculate
            g0_sum += bw.ggamma[k*bw.T*bw.N + n];
        }

        //Store
        bw.init_prob[n] = g0_sum/bw.K;
    }
}


// NOTE: This updates trans_prob and init_prob
inline void update_trans_prob(const BWdata& bw) {
    //Init (trans_prob)
    double numerator_sum, denominator_sum;
    size_t nN;

    //Init (init_prob)
    double g0_sum;

    for (size_t n0 = 0; n0 < bw.N; n0++) {
        nN = n0*bw.N;

        for (size_t n1 = 0; n1 < bw.N; n1++) {
            // Init (trans_prob)
            numerator_sum = 0.0;
            denominator_sum = 0.0;

            // Init (init_prob)
            g0_sum = 0.0;


            for (size_t k = 0; k < bw.K; k++) {
                // Calculate (trans_prob)
                numerator_sum += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1];
                denominator_sum += bw.gamma_sum[k*bw.N + n0];

                // Calculate (init_prob)
                if(n0 == 0){
                    g0_sum += bw.ggamma[(k*bw.T + 0)*bw.N + n1];
                }
            }

            // Store (trans_prob)
            bw.trans_prob[nN + n1] = numerator_sum / denominator_sum;

            // Store (init_prob)
            if(n0 == 0){
                bw.init_prob[n1] = g0_sum/bw.K;
            }
        }
    }
}


inline void update_emit_prob(const BWdata& bw) {
    // Init
    double numerator_sum, denominator_sum, ggamma_cond_sum;
    size_t kTN, kN;

    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k++) {

        kTN = (k*bw.T + (bw.T-1))*bw.N;

        kN = k*bw.N;

        for (size_t n = 0; n < bw.N; n++) {
            bw.gamma_sum[kN + n] += bw.ggamma[kTN + n];
        }
    }
    // update bw.emit_prob
    for (size_t m = 0; m < bw.M; m++) {
        for (size_t n = 0; n < bw.N; n++) {
            numerator_sum = 0.0;
            denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                ggamma_cond_sum = 0.0;
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
