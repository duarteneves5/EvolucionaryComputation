# Adaptation of the Tiny-GP code available at https://github.com/moshesipper/tiny_gp/blob/master/tiny_gp.py
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

# Definition of Terminals and Non-Terminal Symbols

# TODO update and add code here
# ---------------------------
# Function Definitions
# ---------------------------

def add(x, y):
    return x + y
add.arity = 2

def sub(x, y):
    return x - y
sub.arity = 2

def mul(x, y):
    return x * y
mul.arity = 2

# ---------------------------
# Terminals and Function List
# ---------------------------
TERMINALS = ['x', -2, -1, 0, 1, 2]
FUNCTIONS  = [add, sub, mul]

# expanded set
#EXPANDED_FUNCTIONS = [...]
#EXPANDED_TERMINALS = [...]

FUNCTIONS = FUNCTIONS #EXPANDED_FUNCTIONS
TERMINALS = TERMINALS #EXPANDED_TERMINALS


# Parameters

POP_SIZE        = 50   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 6   # maximal initial random tree depth
GENERATIONS     = 100  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability
ELISTIM = 0.1 #0


# Representation of a Program using a GP Tree

import random
from random import randint, random

class GPTree:
    def __init__(self, data=None, children=None):
        self.data = data
        # Use a list for children; if not provided, initialize as empty.
        self.children = children if children is not None else []

    def node_label(self):
        if self.data in FUNCTIONS:
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):
        print(f"{prefix}{self.node_label()}")
        for child in self.children:
            child.print_tree(prefix + "   ")

    def compute_tree(self, x):
        # If the node holds a function, compute all its children first.
        if self.data in FUNCTIONS:
            args = [child.compute_tree(x) for child in self.children]
            return self.data(*args)
        elif self.data == 'x':
            return x
        else:
            return self.data

    def random_tree(self, grow, max_depth, depth=0):
        # Decide whether this node will be a function or a terminal.
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        # If the node is a function, create as many children as its arity
        if self.data in FUNCTIONS:
            self.children = []
            for _ in range(self.data.arity):
                child = GPTree()
                child.random_tree(grow, max_depth, depth + 1)
                self.children.append(child)

    def mutation(self):
        if random() < PROB_MUTATION:
            # Replace this subtree with a new random tree (of limited depth)
            self.random_tree(grow=True, max_depth=2)
        else:
            for child in self.children:
                child.mutation()

    def size(self):
        if self.data not in FUNCTIONS:
            return 1
        return 1 + sum(child.size() for child in self.children)

    def build_subtree(self):
        t = GPTree()
        t.data = self.data
        t.children = [child.build_subtree() for child in self.children]
        return t

    def scan_tree(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second:
                return self.build_subtree()
            else:
                self.data = second.data
                self.children = second.children
        else:
            ret = None
            for child in self.children:
                if count[0] > 1:
                    ret = child.scan_tree(count, second)
            return ret

    def crossover(self, other):
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None)
            self.scan_tree([randint(1, self.size())], second)



# Fitness Evaluation

# *** YOUR CODE HERE ***
# You are going to need to add another function in order to introduce a bloat control mechanism. Adapt the existing one.
def calculate_error(individual, dataset):
    return mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset])

def fitness(individual, dataset, bloat_control = False): # mean absolute error over dataset normalized to [0,1]
        return calculate_error(individual, dataset)


# Parent Selection

def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]])


# Survivor Selection

def survivors_selection_elite(parents,offspring, elite):
    #This Function Should then be called in the Main Cycle line 23, i.e. it should replace the generational mechanism that is currectly in place
    size = len(parents)
    elite_size = int(size* elite)
    new_population = []
    # ******* YOUR CODE HERE *******
    return new_population


# Generate Inital Population using Ramped Half-and-Half
# Generate an individual: method full or grow
# Field Guide Genetic Programming: algorithm 2.1, pg.14
def init_population():
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE / 6)):
            t = GPTree()
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(POP_SIZE / 6)):
            t = GPTree()
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    # If we are missing some inviduals, just fill the rest of population with random trees with the full method
    print(len(pop))
    if len(pop) < POP_SIZE:
        for i in range(POP_SIZE - len(pop)):
            t = GPTree()
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    elif len(pop) > POP_SIZE:
        pop = pop[:POP_SIZE]

    print(len(pop))
    return pop


# Main Cycle

def standard_gp(dataset, verbose=False):
    # init stuff
    seed()  # init internal state of random number generator
    population = init_population()
    best_of_run = None
    best_of_run_f = 999999.0
    best_of_run_gen = 999999.0
    best_per_generation = []
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

    # go evolution!
    for gen in range(GENERATIONS):
        if verbose: print("Generation: %d" % gen)
        nextgen_population = []
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)

        # Generational Strategy
        population = nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        best_per_generation.append(min(fitnesses))

        if min(fitnesses) < best_of_run_f:
            best_of_run_f = min(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(min(fitnesses))])
            if verbose:
                print("________________________")
                print("gen:", gen, ", best_of_run_f:", round(min(fitnesses), 3), ", best_of_run:")
                best_of_run.print_tree()
        if best_of_run_f <= 1e-6:
            break

    if verbose:
        print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(
            best_of_run_gen) + \
              " and has f=" + str(round(best_of_run_f, 3)))
        best_of_run.print_tree()
    return best_per_generation, best_of_run

# Load DataSet

def load_dataset(file_name):
    from numpy import genfromtxt
    my_data = genfromtxt(file_name, delimiter=',')
    return my_data

dataset = load_dataset('datasetHard.csv')
print(dataset)

best_by_generation, overall_best = standard_gp(dataset, verbose=True)

overall_best.print_tree()
erro = calculate_error(overall_best,dataset)
print(erro)

generations = list(range(len(best_by_generation)))
plt.plot(generations, best_by_generation, label='Best by Generation')
plt.title('Performance over generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()

