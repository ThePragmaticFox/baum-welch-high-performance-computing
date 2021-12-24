# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import ghmm
from ghmm import *
import numpy as np

f = lambda x: "%.8f" % (x,) # float rounding function

def get_model_parameters(hmm_model, A_star, B_star, pi_star):
        hmm = hmm_model.cmodel

        if hmm_model.hasFlags(kHigherOrderEmissions):
            order = ghmmwrapper.int_array2list(hmm_model.cmodel.order, hmm_model.N)
        else:
            order = [0]*hmm.N

        if hmm.N <= 4:
            iter_list = range(hmm_model.N)
        else:
            iter_list = [0,1,'X',hmm.N-2,hmm.N-1]

        for k in iter_list:
            if k == 'X':
                continue

            state = hmm.getState(k)

            pi_star[k] = state.pi

            for outp in range(hmm.M**(order[k]+1)):
                B_star[k][outp] = ghmmwrapper.double_array_getitem(state.b,outp)

            for i in range( state.out_states):
                A_star[k][state.getOutState(i)] = ghmmwrapper.double_array_getitem(state.out_a,i)


def print_pi(pi, bw_str):
    for i in range(pi.shape[0]):
        print("    {}.init_prob[{}] = ".format(bw_str, i) + f(pi[i]) + ";")


def print_A(A, bw_str):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print("    {}.trans_prob[{}*N + {}] = ".format(bw_str, i, j) + f(A[i][j]) + ";")


def print_B(B, bw_str):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            print("    {}.emit_prob[{}*M + {}] = ".format(bw_str, i, j) + f(B[i][j]) + ";")


def print_model(A, B, pi, bw_str):
    print_pi(pi, bw_str)
    print("")
    print_A(A, bw_str)
    print("")
    print_B(B, bw_str)
    print("")


def print_converged_model(hmm_model, A, B, pi, bw_str):
    A_star = np.copy(A)
    B_star = np.copy(B)
    pi_star = np.copy(pi)
    get_model_parameters(hmm_model, A_star, B_star, pi_star)
    print_model(A_star, B_star, pi_star, bw_str)

def print_observations(Y, bw_str):
    print("")
    for i in range(Y.shape[0]):
        print("    {}.observations[{}*T + {}] = ".format(bw_str, 0, i) + str(Y[i]) + ";")
    print("")





# %%
# test_case_ghmm_0

A = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
B = np.array([[0.3, 0.7], [0.8, 0.2]], dtype=np.float64)
pi = np.array([0.2, 0.8], dtype=np.float64)
Y = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], np.int)

nrSteps = 1
loglikelihoodCutoff = 0
sigma = IntegerRange(0,2)

print_observations(Y, "bw")
print_model(A, B, pi, "bw")
hmm_model = HMMFromMatrices(sigma, DiscreteDistribution(sigma), list(A), list(B), list(pi))
train_seq = EmissionSequence(sigma, list(Y))
hmm_model.baumWelch(train_seq, nrSteps, loglikelihoodCutoff)
print_converged_model(hmm_model, A, B, pi, "bw_check")



# %%
# test_case_ghmm_1
# the same as test_case_ghmm_0
# run until convergence, instead of just 1 iteration

A = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
B = np.array([[0.3, 0.7], [0.8, 0.2]], dtype=np.float64)
pi = np.array([0.2, 0.8], dtype=np.float64)
Y = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], np.int)

nrSteps = 1000
loglikelihoodCutoff = 0
sigma = IntegerRange(0,2)

print_observations(Y, "bw")
print_model(A, B, pi, "bw")
hmm_model = HMMFromMatrices(sigma, DiscreteDistribution(sigma), list(A), list(B), list(pi))
train_seq = EmissionSequence(sigma, list(Y))
hmm_model.baumWelch(train_seq, nrSteps, loglikelihoodCutoff)
print_converged_model(hmm_model, A, B, pi, "bw_check")



# %%
# test_case_ghmm_2

A = np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]], dtype=np.float64)
B = np.array([[0.1, 0.0, 0.9], [0.05, 0.95, 0.0], [0.3, 0.52, 0.18]], dtype=np.float64)
pi = np.array([0.1, 0.5, 0.4], dtype=np.float64)
Y = np.array([2, 1, 0, 1, 0, 2, 2, 2, 1, 1, 2, 2, 0, 1, 1, 0], np.int)

nrSteps = 1
loglikelihoodCutoff = 0
sigma = IntegerRange(0,3)

print_observations(Y, "bw")
print_model(A, B, pi, "bw")
hmm_model = HMMFromMatrices(sigma, DiscreteDistribution(sigma), list(A), list(B), list(pi))
train_seq = EmissionSequence(sigma, list(Y))
hmm_model.baumWelch(train_seq, nrSteps, loglikelihoodCutoff)
print_converged_model(hmm_model, A, B, pi, "bw_check")



# %%
# test_case_ghmm_3

A = np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]], dtype=np.float64)
B = np.array([[0.1, 0.0, 0.9], [0.05, 0.95, 0.0], [0.3, 0.52, 0.18]], dtype=np.float64)
pi = np.array([0.1, 0.5, 0.4], dtype=np.float64)
Y = np.array([2, 1, 0, 1, 0, 2, 2, 2, 1, 1, 2, 2, 0, 1, 1, 0], np.int)

nrSteps = 1000
loglikelihoodCutoff = 0
sigma = IntegerRange(0,3)

print_observations(Y, "bw")
print_model(A, B, pi, "bw")
hmm_model = HMMFromMatrices(sigma, DiscreteDistribution(sigma), list(A), list(B), list(pi))
train_seq = EmissionSequence(sigma, list(Y))
hmm_model.baumWelch(train_seq, nrSteps, loglikelihoodCutoff)
print_converged_model(hmm_model, A, B, pi, "bw_check")


