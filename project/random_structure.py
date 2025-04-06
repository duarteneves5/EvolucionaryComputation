import os
import datetime
import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *

# ---- PARAMETERS ----
NUM_GENERATIONS = 50  # Number of generations to evolve
POPULATION_SIZE = 25  # Number of robots per generation
NUM_ELITE_ROBOTS = max(1, int(POPULATION_SIZE * 0.05))  # 5% elitism
MUTATION_RATE = 0.2
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
STEPS = 250
SCENARIO = 'Walker-v0'
# For dynamic mutation adjustments
STAGNATION_LIMIT = 5  # # of gens without improvement before we do something
MUTATION_RATE_INCREASE = 2.0  # Factor to multiply mutation rate when stagnant
RANDOM_INJECTION_FRACTION = 0.2  # Replace 20% of the population with random new ones if stagnant
STD_THRESHOLD = 0.5  # Additional parameter for adaptive control based on fitness diversity:

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

# Choose a controller
CONTROLLER = alternating_gait

# CONTROLLER = hopping_motion

OUTPUT_ROBOT_GIFS = True                # this serves to output the robot gifs on the evogym so we can better
USE_REFINED_MUTATE = False
USE_DIVERSITY_PRESERVATION = False
USE_ADAPTIVE_PARAMETER_CONTROL = False  # this ensures the population doesn't become too homogeneous, which is key to avoiding premature convergence.
TEST_NAME = "DEFAULT"                   # this serves for when testing for example a new feature to be easily tracked in the results folder

# ------------------ Run Directory and Logger Setup ------------------
def setup_run_directory():
    base_dir = os.path.join("results", "random_structure")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Use a timestamp to create a unique run folder.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{TEST_NAME}_{timestamp}")
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = setup_run_directory()
LOG_FILE = os.path.join(RUN_DIR, "log.txt")

def log(message):
    # Print to console and also write to log file.
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

# ------------------ Function to Capture Simulation Frame ------------------
def capture_simulation_frame(robot, generation, scenario, steps, controller):

    #utils.simulate_best_robot(robot, scenario, steps)

    filename = os.path.join(RUN_DIR, f"simulation_frame_gen_{generation}.gif")

    utils.create_gif(robot, filename=filename, scenario=scenario, steps=steps, controller=controller)

# ------------------ Evaluation and Variation Functions ------------------
def evaluate_fitness(robot_structure, view=False):
    """
    Evaluate the fitness of a robot by simulating it in the environment.
    Returns the cumulative reward.
    """
    try:
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim

        if view:
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')

        total_reward = 0
        action_size = sim.get_dim_action_space('robot')  # correct action size

        for t in range(STEPS):
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render('screen')

            ob, reward, terminated, truncated, info = env.step(actuation)
            total_reward += reward

            if terminated or truncated:
                env.reset()
                break

        if view:
            viewer.close()
        env.close()

        return total_reward
    except (ValueError, IndexError):
        return 0.0


def create_random_robot():
    """Generate a valid random robot structure (connected and has at least one actuator)."""
    while True:
        grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]),
                     random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        random_robot, _ = sample_robot(grid_size)
        if is_connected(random_robot) and np.any(random_robot == 3):  # Must have at least one actuator
            return random_robot


def select_parent(population, fitness_scores, tournament_size=3):
    """
    Tournament selection. Returns (robot, index_in_population).
    """
    candidates = random.sample(range(len(population)), tournament_size)
    best = candidates[0]
    for c in candidates[1:]:
        if fitness_scores[c] > fitness_scores[best]:
            best = c
    return population[best], best


def is_valid_robot(robot):
    """Check if the robot is connected and has at least one actuator."""
    if not is_connected(robot):
        return False
    return np.any(robot == 3)  # must have actuator(s)


def is_critical_voxel(robot, i, j):
    """
    Check if removing voxel (i, j) disconnects the robot.
    """
    temp = np.copy(robot)
    temp[i, j] = 0
    return not is_connected(temp)


def is_adjacent_to_existing_voxel(robot, i, j):
    """
    Check if (i, j) is adjacent to a non-empty voxel in the grid.
    """
    x_shape, y_shape = robot.shape
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < x_shape and 0 <= nj < y_shape and robot[ni, nj] != 0:
            return True
    return False


def mutate(robot, mutation_rate=0.1):
    """
    Mutate the robot by randomly adding, removing, or changing voxel types.
    Ensures the result remains connected with at least one actuator if possible.
    """
    mutated_robot = np.copy(robot)
    x_shape, y_shape = robot.shape

    for i in range(x_shape):
        for j in range(y_shape):
            if random.random() > mutation_rate:
                continue  # skip mutation at this voxel

            if mutated_robot[i, j] != 0:
                # either change type or remove voxel
                if random.random() < 0.5 and not is_critical_voxel(mutated_robot, i, j):
                    # change type
                    new_type = random.choice([t for t in VOXEL_TYPES if t != mutated_robot[i, j]])
                    mutated_robot[i, j] = new_type
                else:
                    # remove voxel (if not critical)
                    if not is_critical_voxel(mutated_robot, i, j):
                        mutated_robot[i, j] = 0
            else:
                # empty voxel, possibility to add if adjacent to existing
                if is_adjacent_to_existing_voxel(mutated_robot, i, j):
                    mutated_robot[i, j] = random.choice(VOXEL_TYPES[1:])  # any non-empty type

    # Validate final mutated robot; if invalid, revert
    if is_valid_robot(mutated_robot):
        return mutated_robot
    else:
        return robot


# ----- Refined Mutation Operator -----
# This server to target changes to the robot structure more strategically by focusing on edge voxels and nearby empty cells that could become
# part of the structure, rather then aplying random mutations uniformly across all voxels.
# it selectively mutates parts of the structure that are more likely to yield meaningful variations while preserving connectivity and ensuring the robot remains viable.

def refined_mutate(robot, mutation_rate=0.1):
    mutated_robot = np.copy(robot)
    x_shape, y_shape = robot.shape

    # Identify edge voxels: non-empty voxels adjacent to at least one empty cell.
    edge_indices = [(i, j) for i in range(x_shape) for j in range(y_shape)
                    if mutated_robot[i, j] != 0 and is_adjacent_to_existing_voxel(mutated_robot, i, j)]

    # Identify empty voxels adjacent to an existing voxel.
    empty_adjacent = [(i, j) for i in range(x_shape) for j in range(y_shape)
                      if mutated_robot[i, j] == 0 and is_adjacent_to_existing_voxel(mutated_robot, i, j)]

    # Mutate edge voxels: either change type or remove.
    for (i, j) in edge_indices:
        if random.random() < mutation_rate:
            if random.random() < 0.5 and not is_critical_voxel(mutated_robot, i, j):
                new_type = random.choice([t for t in VOXEL_TYPES if t != mutated_robot[i, j]])
                mutated_robot[i, j] = new_type
            else:
                if not is_critical_voxel(mutated_robot, i, j):
                    mutated_robot[i, j] = 0

    # For empty adjacent voxels, consider adding a new voxel.
    for (i, j) in empty_adjacent:
        if random.random() < mutation_rate:
            mutated_robot[i, j] = random.choice(VOXEL_TYPES[1:])

    if is_valid_robot(mutated_robot):
        return mutated_robot
    else:
        return robot

def crossover(parent1, parent2):
    """
    Example: mask-based crossover. For each voxel, pick from parent1 or parent2
    with 50% probability. Returns (child1, child2).
    Will attempt multiple times to get valid children.
    """
    attempts = 10
    for _ in range(attempts):
        # Create random mask of the same shape
        mask = np.random.randint(0, 2, size=parent1.shape)
        child1 = np.where(mask == 0, parent1, parent2)
        child2 = np.where(mask == 0, parent2, parent1)

        if is_valid_robot(child1) and is_valid_robot(child2):
            return child1, child2

    # If all attempts fail, just return copies of the parents
    return parent1, parent2


def random_injection(population, fraction=0.2):
    """
    Replace the bottom fraction of population with fresh random robots.
    E.g., if fraction=0.2 and population size=25, replace ~5 with random new ones.
    """
    n_replace = max(1, int(len(population) * fraction))
    # Sort by fitness ascending, replace the worst n_replace
    fitnesses = [evaluate_fitness(r) for r in population]
    sort_idx = np.argsort(fitnesses)
    # replace worst
    for i in range(n_replace):
        population[sort_idx[i]] = create_random_robot()
    return population

# ---------------- Diversity Preservation: Fitness Sharing ----------------
def robot_distance(robot1, robot2):
    """Compute a simple Hamming-like distance between two robot structures."""
    # Assumes both robots have the same shape.
    return np.sum(robot1 != robot2)

def compute_shared_fitnesses(population, raw_fitnesses, sigma_share=5):
    """
    Adjust raw fitness scores using fitness sharing.
    For each individual, compute:
       f_shared(i) = f(i) / (1 + sum_{j != i} sh(d(i,j)))
    where sh(d) = max(0, 1 - d / sigma_share) for d < sigma_share, 0 otherwise.
    """
    n = len(population)
    shared = np.zeros(n)
    for i in range(n):
        sharing_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            d = robot_distance(population[i], population[j])
            sharing_value = max(0, 1 - d / sigma_share)
            sharing_sum += sharing_value
        shared[i] = raw_fitnesses[i] / (1 + sharing_sum)
    return shared

# ---------------- Main Evolutionary Loop ----------------
best_fitness = -float('inf')
best_robot = None

# Initial population
population = [create_random_robot() for _ in range(POPULATION_SIZE)]

# Track stagnation
stagnation_counter = 0
current_mutation_rate = MUTATION_RATE

for gen in range(NUM_GENERATIONS):
    # Evaluate fitness
    population_fitness = [evaluate_fitness(robot) for robot in population]

    if USE_ADAPTIVE_PARAMETER_CONTROL:
        fitness_std = np.std(population_fitness)

    if USE_DIVERSITY_PRESERVATION:
        # Compute shared fitnesses for diversity preservation
        shared_fitnesses = compute_shared_fitnesses(population, population_fitness, sigma_share=5)

    # Find best in current population
    best_idx = np.argmax(population_fitness)
    gen_best_fit = population_fitness[best_idx]

    # Check for improvement
    if gen_best_fit > best_fitness:
        best_fitness = gen_best_fit
        best_robot = population[best_idx]
        stagnation_counter = 0
        current_mutation_rate = MUTATION_RATE  # reset to default
    else:
        stagnation_counter += 1

    if OUTPUT_ROBOT_GIFS:
        capture_simulation_frame(best_robot, gen, SCENARIO, STEPS, CONTROLLER)

    # Adaptive parameter control: Increase mutation rate if fitness diversity is low. This mechanism dynamically adjusts the mutation rate during the evolution by monitoring the fitness diversity, the algorithm increases dthe mutation when diversity is low.
    if USE_ADAPTIVE_PARAMETER_CONTROL:
        if fitness_std < STD_THRESHOLD:
            adjustment_factor = 1 + (STD_THRESHOLD - fitness_std) / STD_THRESHOLD
            current_mutation_rate = min(1.0, current_mutation_rate * adjustment_factor)

        log(f"Gen {gen} | Best Fitness: {best_fitness:.3f} | Current Gen Best: {gen_best_fit:.3f} | Mutation Rate: {current_mutation_rate:.3f} | Fitness Std: {fitness_std:.3f}")

    else:
        log(f"Gen {gen} | Best Fitness: {best_fitness:.3f} | Current Gen Best: {gen_best_fit:.3f} | Mutation Rate: {current_mutation_rate:.3f}")

    # If we've been stagnant, increase mutation or inject random
    if stagnation_counter >= STAGNATION_LIMIT:
        log(f"> Stagnation reached {STAGNATION_LIMIT} generations: Increasing mutation and injecting random individuals.")
        current_mutation_rate = min(1.0, current_mutation_rate * MUTATION_RATE_INCREASE)
        # random injection
        population = random_injection(population, RANDOM_INJECTION_FRACTION)
        # reset stagnation counter so we wait for new improvements
        stagnation_counter = 0

    # Sort population by fitness to get the top individuals
    sorted_indices = np.argsort(population_fitness)
    new_population = []

    # Elitism: keep top N
    for i in sorted_indices[-NUM_ELITE_ROBOTS:]:
        new_population.append(population[i])

    # Fill the rest of the new population
    # Make a copy so we don't remove from original 'population' while selecting
    pop_copy = population[:]

    if USE_DIVERSITY_PRESERVATION:
        fitness_copy = list(shared_fitnesses)
    else:
        fitness_copy = population_fitness[:]

    while len(new_population) < POPULATION_SIZE:
        # Parent selection
        parent1, idx1 = select_parent(pop_copy, fitness_copy)
        # Remove it from the "pool" so we don't select the exact same index again
        pop_copy.pop(idx1)
        fitness_copy.pop(idx1)

        parent2, idx2 = select_parent(pop_copy, fitness_copy)
        # Important: do not pop idx2 from pop_copy yet, because removing idx1 changes indexing
        # but for simplicity, we won't re-use the same parent in one iteration, so it's okay.

        # Crossover
        child1, child2 = crossover(parent1, parent2)

        # Refined Mutation
        if USE_REFINED_MUTATE:
            child1 = refined_mutate(child1, current_mutation_rate)
            child2 = refined_mutate(child2, current_mutation_rate)
        # Mutation
        else:
            child1 = mutate(child1, current_mutation_rate)
            child2 = mutate(child2, current_mutation_rate)



        # Add children
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population

# After evolution, demonstrate the best robot
log(f"\n=== EVOLUTION COMPLETE ===\nBest Fitness Found: {best_fitness}")
log(f"Best Robot:\n{best_robot}")

# Optional: run best robot a few times, or create a GIF, etc.
for i in range(3):
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)

utils.create_gif(best_robot, filename='random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
