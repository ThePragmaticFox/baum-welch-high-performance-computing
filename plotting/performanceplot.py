import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path

colors = ['royalblue', 'firebrick', 'darkgreen', 'dimgray', 'darkslateblue', 'darkmagenta', 'orangered']
color_counter = 0

implementations = ['Baseline', 'jc-reordered', 'scalar-reorder', 'scalar-blocking', 'other-unroll', 'vector_optimized',
                   'combined']
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

def get_csv(filename):
    return pd.read_csv(os.path.join(path_to_csv, filename), delimiter=';')


def create_graph(data, key, x_label, filename):
    linewidth = 2
    rest = list(variables.difference(key))
    title = "Performance Plot  -  ({}={}, {}={}, {}={})\n{} {}\n".format(rest[0], 32 if rest[0] == "T" else 16, rest[1],
                                                                         32 if rest[1] == "T" else 16, rest[2],
                                                                         32 if rest[2] == "T" else 16,
                                                                         filenames[filename]["compiler"],
                                                                         filenames[filename]["flags"])
    title_y = 1.03
    x_label = x_label
    y_label = "Performance [flops/cycle]"
    y_labelpad = 0
    y_y = 1.03

    mul = 1.5
    xf = mul * (6.4 + 0.6)
    yf = mul * (4.8 - 0.8625)
    plt.figure(figsize=[xf, yf], dpi=200, constrained_layout=True)
    ax = plt.axes()
    ax.set_facecolor("lightgray")
    plt.grid(b=True, which="major", ls="-", color="white")
    plt.ylim(0, 4)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='white', linestyle='-', alpha=0.2)

    if key == 'M':
        plt.xscale("log", basex=2)

    plt.title(title, y=title_y, x=-0.03, loc='left', weight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label, labelpad=y_labelpad, rotation=0, horizontalalignment="left", y=y_y)

    for i, imp in enumerate(implementations):
        x = data[data['Implementation'] == imp][key]
        y = data[data['Implementation'] == imp]['Performance']
        plt.plot(x, y, colors[i], label=imp, linewidth=linewidth)

    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(os.path.join("plots", filename + "_" + x_label + ".png"), orientation='landscape')
    plt.close()


def create_graph_baseline(data, key, x_label, filename):
    linewidth = 2
    rest = list(variables.difference(key))
    if len(x_label) > 1:
        title = "Performance Plot  -  (T has a step size of 32)\n{} {}\n".format(
            filenames[filename]["compiler"],
            filenames[filename]["flags"])
    else:
        title = "Performance Plot  -  ({}={}, {}={}, {}={})\n{} {}\n".format(
            rest[0], 32 if rest[0] == "T" else 16,
            rest[1],
            32 if rest[1] == "T" else 16, rest[2],
            32 if rest[2] == "T" else 16,
            filenames[filename]["compiler"],
            filenames[filename]["flags"])

    title_y = 1.03
    x_label = x_label
    y_label = "Performance [flops/cycle]"
    y_labelpad = 0
    y_y = 1.03

    mul = 1
    xf = mul * (6.4 + 0.6)
    yf = mul * (4.8 - 0.8625)
    plt.figure(figsize=[xf, yf], dpi=200, constrained_layout=True)
    ax = plt.axes()
    ax.set_facecolor("lightgray")
    plt.grid(b=True, which="major", ls="-", color="white")
    plt.ylim(0, 1.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='white', linestyle='-', alpha=0.2)

    if key == 'M':
        plt.xscale("log", basex=2)

    plt.title(title, y=title_y, x=-0.05, loc='left', weight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label, labelpad=y_labelpad, rotation=0, horizontalalignment="left", y=y_y)

    x = data[data['Implementation'] == 'Baseline'][key]
    y = data[data['Implementation'] == 'Baseline']['Performance']
    plt.plot(x, y, colors[0], label='Baseline', linewidth=linewidth)

    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(os.path.join("plots", filename + "_" + x_label + "_baseline.png"), orientation='landscape')
    plt.close()


def graphs(filename):
    df = get_csv(filename)

    for key in variables:
        rest = list(variables.difference(key))
        data = df[(df[rest[0]] == (32 if rest[0] == "T" else 16)) &
                  (df[rest[1]] == (32 if rest[1] == "T" else 16)) &
                  (df[rest[2]] == (32 if rest[2] == "T" else 16))]
        create_graph(data, key, key, filename)
        create_graph_baseline(data, key, key, filename)

    data = df[(df['K'] == df['M']) & (df['K'] == df['N']) & ((df['T'] == df['K']) | (df['T'] == df['K'] + 16))]
    create_graph(data, 'K', "K,N,M,T", filename)
    create_graph_baseline(data, 'K', "K,N,M,T", filename)


def main():
    for filename in filenames.keys():
        graphs(filename)


main()
