# Advanced Systems Lab (How to Write Fast Numerical Code) - Spring 2020
Semester Project: Baum-Welch algorithm

### Authors

Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi

ETH Computer Science MSc, Computer Science Department ETH Zurich

## Overleaf

https://www.overleaf.com/read/fsqqnxhfhnvx

View-only link for supervisors to observe progress on the report.

## Compiler

We use gcc 9.2.1.

## Building the project

The project uses CMake. 

Create a folder named `build` and change into it. Then run `cmake ..` to generate the Makefile and then `make` to build the project. 

The project generates two executables: `benchmarks` and `verifications`. 

## Running the project

`benchmarks` executes the performance benchmark test without verifications if the implementations are correct. 

`benchmarks` supports certain arguments:

```
Usage: ./benchmarks [OPTIONS]
Benchmarks the registered implementations against the registered baseline.

Options:
  -h, --help			Prints this message and exits
  -t, --test			Perform test that can be used for the report (Not yet implemented)
  -o, --only <name>		Only execute the implementation with the given name (case-sensitive). 
  				 Can occur multiple times and is compatible with --test. The baseline is
  				 always run.
      --list			Lists all available implementations and exits
      --max-iterations <value>	Sets the max-iteration to a value
```

`verification` checks if the implementations behave correctly and compares the implementations against the baseline that is verified differently.

## Goal

Starting with a baseline version, we implement various optimizations to significantly speed up the performance of the Baum-Welch algorithm.

## Assumptions

K >= 16 and divisible by 16

T >= 32 and divisible by 16

N >= 16 and divisible by 16

M >= 16 and divisible by 16

This is sufficiently large to take most to all optimization possibilities into account.

Furthermore, to check equality for doubles, we use EPSILON 1e-4, to not get caught up in numerical instabilities and other sources of randomness.

Lastly, we omitted the convergence criterion by the minimization of the monotonously decreasing negative log likelihood sequence, because it adds an unnecessary source of randomness.
Note that Expectation-Maximization is provably guaranteed to not change after convergence, so running more than fewer iterations causes no harm, except for overfitting (irrelevant for our purposes) and increased runtime (wanted for benchmarking).

## Analysis

### Cost Analysis

Our reordering removed `K*N*T divs` from the algorithm. In the file `implementations/baseline_old.cpp` we have the original implementation with the `K*N*T divs`. To match the flop count, we did that optimization that removed those flops also in the baseline.

Cost analysis (add, mul and div is one flop) for the original baseline for one iteration:
* forward: `(1 add + 1 mul)*K*N*N*T + (1 add + 2 mults)*K*N*T + (1 add + 2 mults)*K*N + (1 div)*K*T + (1 div)*K`
* backward: `(1 add + 2 muls)*K*N*N*T + (1 mult)*K*N*T`
* compute gamma: `(1 div + 1 mult)*K*N*T + (1 add)*K*N*T`
* compute sigma: `(1 add + 3 mults)*K*N*N*T`
* update init: `(1 add)*K*N + (1 div)*N`
* update trans: `(2 adds)*K*N*N + (1 div)*N*N`
* update emit: `(2 adds)*N*M*K + (1 add)*K*N*T + (1 add)*K*N + (1 div)*N*M`
* neg-log-likelyhood: `K*T*(add)`

total: `K*N*N*T(3 add + 6 mults) + K*N*T(3 add + 4 mult + div) + K*N(3 add + 2 mult) + K*T(div + add) + K*(1 div) + K*N*N(2 add) + N*N*(div) + K*N*M(2 add) + N*M(1 div)`

* adds:  3KNNT + 3KNT + 3KN + KT + 2KNN + 2KNM
* mults: 6KNNT + 4KNT + 2KN
* div:   KNT + KT + K + N + NN + NM

For the new baseline with the reordering optimization:
* forward: `(1 add + 1 mul)*K*N*N*T + (1 add + 2 mults)*K*N*T + (1 add + 2 mults)*K*N + (1 div)*K*T + (1 div)*K`
* backward: `(1 add + 2 muls)*K*N*N*T + (2 mult)*K*N*T`
* compute gamma: `(1 add)*K*N*T`
* compute sigma: `(1 add + 3 mults)*K*N*N*T`
* update init: `(1 add)*K*N + (1 div)*N`
* update trans: `(2 adds)*K*N*N + (1 div)*N*N`
* update emit: `(2 adds)*N*M*K + (1 add)*K*N*T + (1 add)*K*N + (1 div)*N*M`
* neg-log-likelyhood: `K*T*(add)`

total: `K*N*N*T(3 add + 6 mults) + K*N*T(3 add + 4 mult) + K*N(3 add + 2 mult) + K*T(div + add) + K*(1 div) + K*N*N(2 add) + N*N*(div) + K*N*M(2 add) + N*M(1 div)`

* adds:  3KNNT + 3KNT + 3KN + KT + 2KNN + 2KNM
* mults: 6KNNT + 4KNT + 2KN
* div:   KT + K + N + NN + NM

### Memory Usage

The BWdata struct itself is `144 bytes` in size. The hole struct is loaded at least once per execution of the algorithm as the adresses of the arrays need to be resolved.

This are the sizes of the arrays in the BWdata struct. Their sizes are given in doubles, thus to get bytes multiply with 8.

* init\_prob: `N`
* trans\_prob: `N*N`
* emit\_prob: `N*M`
* observation: `K*T`
* neg\_log\_likelihood: `max_iterations`
* c\_norm: `K*T`
* alpha: `K*T*N`
* beta: `K*T*N`
* gamma: `K*T*N`
* sigma: `K*T*N*N`
* gamma\_sum: `K*N`
* sigma\_sum: `K*N*N`

So the total is: `(N + N*N + N*M + 2*K*T + max_iterations + 3*K*T*N + K*T*N*N + K*N + K*N*N)*8 + 144` bytes.

## Verification

### Baseline

We have "test_case_ghmm_x" functions to check against hardcoded examples that were verified using the Python 2.x, Linux only ghmm library ( http://ghmm.sourceforge.net/index.html and download https://sourceforge.net/projects/ghmm/ ).

For reproducibility purposes, the code can be found in misc/asl_baum_welch_ghmm_experiments.ipynb (jupyter notebook) or, alternatively, misc/asl_baum_welch_ghmm_experiments.py.

Note that Python 3 does not work with ghmm and to install the library, there are quite some dependencies to take into account.

### Wikipedia Example

We talked about the Wikipedia example ( https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Example ) during the meeting.
The example uses a joint probability factorization into conditional probabilities and usage of the likelihood to compute one "iteration".
This has nothing to do with the actual Baum-Welch algorithm described on this very same page. It is confusing and misleading. And it cost me DAYS.

The example is still used as "test_case_ghmm_0" and "test_case_ghmm_1", though obviously compared against Baum-Welch implementations.
And as it can be seen, not only in the verifications, but also in misc/BaumWelchWikipediaExample.xlsx
and the Matlab implementation ( https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/ ) that corresponds to the Tutorial 
( https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf ) linked on the ASL website for Baum-Welch project:

Our approach is absolutely correct and thoroughly verified for the project's scope and purpose!

### Optimizations

1. Randomly initialize K, N, M, T, observations, init_prob, trans_prob and emit_prob, by generating random numbers and normalize where needed.
2. Run the baseline implementation for max_iterations.
3. Check whether the sequence of the negative log likelihood is monotonously decreasing in each iteration, which is guaranteed by the expectation-maximization algorithm and shows correctness of the (unscaled) Baum-Welch algorithm conceptually.
4. Check whether the rows of init_prob, trans_prob and emit_prob sum to 1.0 each, as they represent (learned) probability distributions, both before and after the run. 
5. For each optimization: Do the same as 2., 3. and 4. and additionally check the resulting probability tables of init_prob, trans_prob and emit_prob directly with the corresponding ones from the baseline implementation.

## Implementations

All implementations are found in the `implementations` folder. To create a new implementation follow those steps:

1. Create a new `.cpp` file in the folder `implementations`
2. Add the file to both executables in the `CMakeLists.txt`
3. Implement

Your implementation must have a function with the following signature to allow it to be called by the benchmark and verification system.

```C
size_t func_name(const BWdata& bw){...}
```

To register your function in the benchmark system add the following line to your file. You can register multiple functions in one file.

```C
REGISTER_FUNCTION(func_name, "name", "A description about the implementation");
```

The name string should be writable in the commandline, as it is used to limit which implemetations should be benchmarked.

**CAUTION**: Be aware that you cannot name your function the same as another implementation in a different file. The linker is not able to do that right now.

### "baseline.cpp" Implementation

Implementation of the Baum-Welch algorithm with scaling taken into account for numerical stability.

Main references used

https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf

https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf

### "scalar_optimized.cpp" Implementation

Partial to full vectorization with AVX and FMA Intel intrinsics.

### "vector_optimized.cpp" Implementation

TODO

### "combined_optimized.cpp" Implementation

TODO


