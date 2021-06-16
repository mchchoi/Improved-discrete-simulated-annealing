# Based on https://github.com/jedrazb/python-tsp-simulated-annealing
from anneal import SimAnneal
import matplotlib.pyplot as plt
import random
import numpy as np


def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


def generate_random_coords(num_nodes):
    return [[random.uniform(0, 100), random.uniform(0, 100)] for i in range(num_nodes)]


if __name__ == "__main__":
    output = []
    random.seed(15)
    seed = []
    for i in range(1000):
        seed.append(random.randint(0,10200))

    for i in range(1000):
        print(i)
        random.seed(seed[i])
        #coords = read_coords("coord.txt")  # generate_random_coords(100)
        coords = generate_random_coords(50)
        random.seed(200)
        sa = SimAnneal(coords,c=19000, stopping_iter=100000)
        # simulated annealing
        sa.anneal()
        #sa.visualize_routes()
        #sa.plot_learning()

        # improved annealing
        random.seed(200)
        isa = SimAnneal(coords, c=19000, stopping_iter=100000)
        isa.improved_anneal()
        improve_percentage = (sa.best_fitness - isa.best_fitness)/sa.best_fitness
        output.append(improve_percentage)

    np_output = np.array(output)
    print(np.mean(output))
    print(np.amax(output))
    print(np.amin(output))
    print(np.median(output))
    print(np.sum(np_output >= 0))
    print(np.sum(np_output < 0))

    plt.hist(np.array(output) * 100, bins=50)
    plt.ylabel("Frequency")
    plt.xlabel("Improvement percentage")
    plt.title("Histogram of improvement percentage of ISA over SA on 100 randomly generated TSP instances")
    plt.show()
