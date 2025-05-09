#!/usr/bin/env python3
import re
import os
import matplotlib.pyplot as plt

# ————— CONFIG —————
LOG_FILE = "C:/Users/duarte.neves/PycharmProjects/EvolucionaryComputation/project/results/random_structure/Bridge_final_run_insert_mutation_mask_20250509_062612/log.txt"


# this regex matches your "Gen XX | Best: ... | Avg: ... +- ... | All-Time Best: ..." lines
LOG_PATTERN = re.compile(
    r"Gen\s*(\d+)\s*\|\s*Best:\s*([0-9.\-]+)\s*"
    r"\|\s*Avg:\s*([0-9.\-]+)\s*\+-\s*([0-9.\-]+)\s*"
    r"\|\s*All-Time Best:\s*([0-9.\-]+)"
)

def parse_log_file(path):
    gen_avg    = []
    gen_std    = []
    gen_best   = []
    all_time   = []
    with open(path, 'r') as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            # groups: 1=gen, 2=best, 3=avg, 4=std, 5=all-time
            gen_best.append(float(m.group(2)))
            gen_avg.append( float(m.group(3)) )
            gen_std.append( float(m.group(4)) )
            all_time.append( float(m.group(5)) )
    return gen_avg, gen_std, gen_best, all_time

if __name__ == "__main__":
    # parse
    gen_avg_fitness, gen_std_fitness, gen_best_per_gen, best_per_gen_ever = parse_log_file(LOG_FILE)
    NUM_GENERATIONS = len(gen_avg_fitness)

    # plot _exactly_ as in your EA
    plt.figure()
    plt.errorbar(
        range(NUM_GENERATIONS),
        gen_avg_fitness,
        yerr=gen_std_fitness,
        label="mean ± std"
    )
    plt.plot(
        range(NUM_GENERATIONS),
        gen_best_per_gen,
        label="generation best"
    )
    plt.plot(
        range(NUM_GENERATIONS),
        best_per_gen_ever,
        label="all-time best"
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()   # PyCharm will pop this up or render inline
