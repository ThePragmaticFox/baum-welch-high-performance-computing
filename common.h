/*
    Declarations for different implementations and optimizations for the algorithm. Also
    provides functionality to register functions to benchmark and test the implementations

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#if !defined(__BW_COMMON_H)
#define __BW_COMMON_H

#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <immintrin.h>

#define EPSILON 1e-4

/**
 * Struct containing all data for the Baum-Welch algorithm
 */
struct BWdata {
    // (for each observation/training sequence 0 <= k < K)
    size_t* observations; //        [K][T]          [k][t]            :=  observation sequence k at time_step t
    double* init_prob; //           [N]             [n]               :=  P(X_1 = n)
    double* trans_prob; //          [N][N]          [n0][n1]          :=  P(X_t = n1 | X_(t-1) = n0)
    double* emit_prob; //           [N][M]          [n][m]            :=  P(Y_t = y_m | X_t = n)
                       // Some optimizations require this matrix to be [M][N]
    double* neg_log_likelihoods; // [max_iterations] [it]             :=  array to store the neg_log_likelihood for each iteration
    double* c_norm; //              [K][T]          [k][t]            :=  scaling/normalization factor for numerical stability
    // NOTE that
    // min_{λ={init_prob, trans_prob, emit_prob}} -Σ_{k}(Σ_{t} log(Pr[obs[k][t]|λ]))
    // = min_{λ} -Σ_{k}(-Σ_{t} log(c_norm[k][t]))
    // = min_{λ} +Σ_{k}Σ_{t}log(c_norm[k][t])
    // (see the tutorial linked in baseline.cpp)
    double* alpha; //               [K][T][N]       [k][t][n]         :=  P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
    double* beta; //                [K][T][N]       [k][t][n]         :=  P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
    double* ggamma; //              [K][T][N]       [k][t][n]         :=  P(X_t = n | Y, theta)
    double* sigma; //               [K][T][N][N]    [k][t][n0][n1]    :=  P(X_t = n0, X_(t+1) = n1 | Y, theta)
    // where theta = {init_prob, trans_prob, emit_prob} represent the model parameters we want to learn/refine/estimate iteratively.
    double* gamma_sum; //           [K][N]
    double* sigma_sum; //           [K][N][N]
    
    const size_t K;  // number of observation sequences / training datasets
    const size_t N;  // number of hidden state variables
    const size_t M;  // number of distinct observations
    const size_t T;  // number of time steps
    const size_t max_iterations; // Number of maximum iterations that should be performed
    
    const bool full_copy;
    
    /**
     * Creates a BWdata from given data (Constructor)
     */
    BWdata(const size_t K,
           const size_t N,
           const size_t M,
           const size_t T,
           const size_t max_iterations):
            K(K), N(N), M(M), T(T), max_iterations(max_iterations), full_copy(true){
        init_prob = (double *)aligned_alloc(32, N * sizeof(double));
        trans_prob = (double *)aligned_alloc(32, N*N * sizeof(double));
        emit_prob = (double *)aligned_alloc(32, N*M * sizeof(double));
        observations = (size_t *)aligned_alloc(32, K*T * sizeof(size_t));
        neg_log_likelihoods = (double *)aligned_alloc(32, max_iterations * sizeof(double));
        c_norm = (double *)aligned_alloc(32, K*T * sizeof(double));
        alpha = (double *)aligned_alloc(32, K*T*N * sizeof(double));
        beta = (double *)aligned_alloc(32, K*T*N * sizeof(double));
        ggamma = (double *)aligned_alloc(32, K*T*N * sizeof(double));
        sigma = (double *)aligned_alloc(32,K*T*N*N * sizeof(double));
        gamma_sum = (double *)aligned_alloc(32, K*N * sizeof(double));
        sigma_sum = (double *)aligned_alloc(32, K*N*N * sizeof(double));

        assert(observations != NULL && "Failed to allocate observations");
        assert(init_prob != NULL && "Failed to allocate init_prob");
        assert(trans_prob != NULL && "Failed to allocate trans_prob");
        assert(emit_prob != NULL && "Failed to allocate emit_prob");
        assert(neg_log_likelihoods != NULL && "Failed to allocate neg_log_likelihoods");
        assert(c_norm != NULL && "Failed to allocate c_norm");
        assert(alpha != NULL && "Failed to allocate alpha");
        assert(beta != NULL && "Failed to allocate beta");
        assert(ggamma != NULL && "Failed to allocate ggamma");
        assert(sigma != NULL && "Failed to allocate sigma");
        assert(gamma_sum != NULL && "Failed to allocate gamma_sum");
        assert(sigma_sum != NULL && "Failed to allocate sigma_sum");
    }
    
    /**
     * Creates a BWdata from a given BWdata (constructor).
     * This is no deep copy. As no parallelization is used, the reuse of constant memory data is permitted
     */
    BWdata(const BWdata& other): K(other.K), N(other.N), M(other.M), T(other.T), max_iterations(other.max_iterations), full_copy(false){
        init_prob = (double *)aligned_alloc(32, N *sizeof(double));
        trans_prob = (double *)aligned_alloc(32, N*N * sizeof(double));
        emit_prob = (double *)aligned_alloc(32, N*M * sizeof(double));
        observations = other.observations;
        neg_log_likelihoods = other.neg_log_likelihoods;
        c_norm = other.c_norm;
        alpha = other.alpha;
        beta = other.beta;
        ggamma = other.ggamma;
        sigma = other.sigma;
        gamma_sum = other.gamma_sum;
        sigma_sum = other.sigma_sum;

        assert(init_prob != NULL && "Failed to allocate init_prob");
        assert(trans_prob != NULL && "Failed to allocate trans_prob");
        assert(emit_prob != NULL && "Failed to allocate emit_prob");

        memcpy(init_prob, other.init_prob, N * sizeof(double));
        memcpy(trans_prob, other.trans_prob, N * N * sizeof(double));
        memcpy(emit_prob, other.emit_prob, N * M * sizeof(double));
    }

    /**
     * Copies the current BWdata into a new one (deep copy).
     */
    const BWdata& deep_copy() const{
        BWdata* other = new BWdata(K, N, M, T, max_iterations);
        memcpy(other->init_prob, init_prob, N * sizeof(double));
        memcpy(other->trans_prob, trans_prob, N * N * sizeof(double));
        memcpy(other->emit_prob, emit_prob, N * M * sizeof(double));
        memcpy(other->observations, observations, K*T*sizeof(double));
        memcpy(other->neg_log_likelihoods, neg_log_likelihoods, max_iterations*sizeof(double));
        memcpy(other->c_norm, c_norm, K*T*sizeof(double));
        memcpy(other->alpha, alpha,  K*T*N*sizeof(double));
        memcpy(other->beta, beta,  K*T*N*sizeof(double));
        memcpy(other->ggamma, ggamma,  K*T*N*sizeof(double));
        memcpy(other->sigma, sigma,  K*T*N*N*sizeof(double));
        memcpy(other->gamma_sum, gamma_sum,  K*N*sizeof(double));
        memcpy(other->sigma_sum, sigma_sum,  K*N*N*sizeof(double));
        
        return *other;
    }

    /**
     * Frees all allocated memory space for BWdata upon destruction.
     * Only in full_copy BWdata all fields are free'd. It can be assumed all non_full_copies are released beforehand.
     */
    ~BWdata(){
        if(full_copy){
            // WARNING: Possible Use after free when releasing a full_copy before all non-full_copies are released
            // We ignore this as we make sure we delete the copies first.
            free(c_norm);
            free(alpha);
            free(beta);
            free(ggamma);
            free(sigma);
            free(gamma_sum);
            free(sigma_sum);
            free(observations);
            free(neg_log_likelihoods);
        }
        free(init_prob);
        free(trans_prob);
        free(emit_prob);
    }
};


/**
 * Function interface for an implementation for the Baum-Welch algorithm
 */
typedef size_t(*compute_bw_func)(const BWdata& bw);

struct RegisteredFunction{
    compute_bw_func func;
    std::string name;
    std::string description;
    bool transpose_emit_prob;
};

/**
 * Static class that handles the function registration. 
 */
class FuncRegister
{
public:

    /**
     * Set the function that is considered the baseline and other
     * implementations are compared against. There can only be one baseline
     */
    static void set_baseline(compute_bw_func f, const std::string& name);

    static void add_function(compute_bw_func f, const std::string& name, const std::string& description, const bool transpose_emit_prob = false);
    
    static void printRegisteredFuncs();

    static size_t size()
    {
        return (*funcs).size();
    }
    
    static std::vector<struct RegisteredFunction> *funcs;
    static compute_bw_func baseline_func;
    static std::string baseline_name;
};

// Macro to register a function and a name that should be executed
#define REGISTER_FUNCTION(f, name, description)                   \
    static struct f##_                                            \
    {                                                             \
        f##_()                                                    \
        {                                                         \
            FuncRegister::add_function(f, name, description);     \
        }                                                         \
    } f##__BW_

//Macro to register a function that requires emit_prob to be column major
#define REGISTER_FUNCTION_TRANSPOSE_EMIT_PROB(f, name, description)                   \
    static struct f##_                                            \
    {                                                             \
        f##_()                                                    \
        {                                                         \
            FuncRegister::add_function(f, name, description, true);     \
        }                                                         \
    } f##__BW_

// Macro to register a function and a name that should be executed
#define SET_BASELINE(f, name)                                     \
    static struct f##_                                            \
    {                                                             \
        f##_()                                                    \
        {                                                         \
            FuncRegister::set_baseline(f, name);                  \
        }                                                         \
    } f##__BW_

#endif /* __BW_COMMON_H */