import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path

colors = ['black', 'dimgray', 'slateblue', "darkblue", "darkcyan","indigo","darkorange"]
markers = ['s', 'o', '^', 'D', '*', 'X', 'p']
color_counter = 0

implementations = [{"name": 'Baseline', "vector": False, "additional-memory": lambda N, M, K, T, max_iterations: 0},
                   {"name": 'jc-reordered', "vector": False, "additional-memory": lambda N, M, K, T, max_iterations: 0},
                   {"name": 'scalar-reorder', "vector": False,
                    "additional-memory": lambda N, M, K, T, max_iterations: 0},
                   {"name": 'scalar-blocking', "vector": False,
                    "additional-memory": lambda N, M, K, T, max_iterations: 0},
                   {"name": 'other-unroll', "vector": False,
                    "additional-memory": lambda N, M, K, T, max_iterations: N * 8 + N * M * 8 + N * 8},
                   {"name": 'vector_optimized', "vector": True, "additional-memory": lambda N, M, K, T,
                                                                                            max_iterations: 4 * 8 + N * M * 8 + N * M * 8 + N * K * T * 8 + K * T * 8},
                   {"name": 'combined', "vector": True,
                    "additional-memory": lambda N, M, K, T, max_iterations: N * 8 + N * M * 8}]
variables = {'K', 'M', 'N', 'T'}
filenames = {
    "log-normal-gcc.csv": {
        "compiler": "gcc (9.2.0)",
        "flags": "-O3 -ffast-math -mavx2 -mfma -march=native"
    },
    "log-vector-gcc.csv": {
        "compiler": "gcc (9.2.0)",
        "flags": "-O3 -fno-vectorize -ffast-math -mavx2 -mfma -march=native"
    },
    "log-unroll-gcc.csv": {
        "compiler": "gcc (9.2.0)",
        "flags": "-O3 -funroll-loops -ffast-math -mavx2 -mfma -march=native"
    },
    "log-normal-clang.csv": {
        "compiler": "clang (7.0.1)",
        "flags": "-O3 -ffast-math -mavx2 -mfma -march=native"
    },
    "log-vector-clang.csv": {
        "compiler": "clang (7.0.1)",
        "flags": "-O3 -fno-vectorize -ffast-math -mavx2 -mfma -march=native"
    },
    "log-unroll-clang.csv": {
        "compiler": "clang (7.0.1)",
        "flags": "-O3 -funroll-loops -ffast-math -mavx2 -mfma -march=native"
    }}

path_to_csv = "../misc"

def beta_i(x):
    return 16 * x


def beta_i_inv(x):
    return x / 16


def pi_(x):
    return 4 + x * 0


def pi4_(x):
    return 4 * 4 + x * 0


def get_csv(filename):
    return pd.read_csv(os.path.join(path_to_csv, filename), delimiter=';')


def graphs(filename, K, N, M, T, impls):
    df = get_csv(filename)

    max_iterations = 500

    # Properties
    linewidth = 2
    title = "Roofline Plot - (K={},N={},M={},T={})\n{} {}\n".format(K, N, M, T, filenames[filename]["compiler"],
                                                                    filenames[filename]["flags"])
    title_y = 1.03
    x_label = "Operational Intensity [flops/byte]"
    y_label = "Performance [flops/cycle]"
    y_labelpad = -6.5
    y_y = 1.03

    # x-axis
    x = np.logspace(-6, 8, base=2)

    # y-axis of different functions
    beta_l = beta_i(x)
    pi = pi_(x)
    pi4 = pi4_(x)
    pi_beta_I = beta_i_inv(4)
    pi4_beta_I = beta_i_inv(16)

    # Set axis properties
    mul = 1.5
    xf = mul * 5#(6.4 + 0.6)
    yf = mul * 5#(4.8 - 0.8625)
    plt.figure(figsize=[xf, yf], dpi=200, constrained_layout=True)
    ax = plt.axes()
    ax.set_facecolor("lightgray")
    plt.grid(b=True, which="major", ls="-", color="white")
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='white', linestyle='-', alpha=0.2)
    plt.xlim(2 ** -4, 2 ** 6)
    plt.ylim(2 ** -1, 2 ** 5)
    plt.xscale("log", basex=2)
    plt.yscale("log", basey=2)
    plt.title(title, y=title_y, x=-0.045, loc='left', weight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label, labelpad=y_labelpad, rotation=0, horizontalalignment="left", y=y_y)

    # Plots of (x,y_i)
    plt.plot(x, beta_l, 'firebrick', label="$\\beta*I(n)$", linewidth=linewidth)
    plt.plot(x, pi, 'darkgreen', label="$\\pi$", linewidth=linewidth)
    plt.plot(x, pi4, 'forestgreen', label="$4\\pi$ (SIMD)", linewidth=linewidth)
    plt.vlines(pi_beta_I, plt.ylim()[0], 4, ls=":")
    plt.vlines(pi4_beta_I, plt.ylim()[0], 16, ls=":")

    W = 9 * T * K * N * N - 5 * K * N * N + N * N + 8 * T * K * N + 3 * K * N + K + 2 * K * N * M + 2 * T * K + N + N * M
    for i, imp in enumerate(impls):
        Q = (
                    N + N * N + N * M + 2 * K * T + max_iterations + 3 * K * T * N + K * T * N * N + K * N + K * N * N) * 8 + 144 + \
            imp["additional-memory"](N, M, K, T, max_iterations)
        I = W / Q
        data = df[(df['K'] == K) & (df['N'] == N) & (df['M'] == M) & (df['T'] == T)]
        y = data[data['Implementation'] == imp["name"]]['Performance']
        plt.plot(I, np.minimum(y, beta_i(I)), color=colors[i], marker=markers[i], markersize=12,
                 label=imp["name"] + (" (SIMD)" if imp["vector"] else ""))

    plt.legend(loc="upper right")
    #plt.show()
    plt.savefig(os.path.join("roofline", filename + "_roofline_{}-{}-{}-{}.png".format(K,N,M,T)), orientation='landscape')
    plt.close()


def main():
    for file in filenames.keys():
        graphs(file, 16, 16, 16, 32, implementations)


main()
