import os
import datetime
from fileinput import filename

import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *

import matplotlib.pyplot as plt
# ---- PARAMETERS ----
NUM_GENERATIONS = 50  # Number of generations to evolve
POPULATION_SIZE = 50  # Number of robots per generation
NUM_ELITE_ROBOTS = max(1, int(POPULATION_SIZE * 0.06))  # 6% elitism
MUTATION_RATE = 0.05
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
STEPS = 400
SCENARIO = 'Walker-v0'
# For dynamic mutation adjustments
STAGNATION_LIMIT = 51  # # of gens without improvement before we do something, if greater than 50 it doesnt act
MUTATION_RATE_INCREASE = 2.0  # Factor to multiply mutation rate when stagnant
RANDOM_INJECTION_FRACTION = 0.1  # Replace 20% of the population with random new ones if stagnant


# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

# Choose a controller
CONTROLLER = alternating_gait

# CONTROLLER = hopping_motion

OUTPUT_ROBOT_GIFS = True                # this serves to output the robot gifs on the evogym so we can better
OUTPUT_POPULATION = True

# These will be overridden inside each experiment loop:
MUTATION_METHOD = 'random'     # 'random', 'swap' or 'insert'
CROSSOVER_METHOD = 'mask'      # 'mask', 'one_point' or 'two_point'
TEST_NAME = "baseline"

# ------------------ Run Directory and Logger Setup and Helper Functions ------------------
def setup_run_directory():
    base_dir = os.path.join("results", "random_structure")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Use a timestamp to create a unique run folder.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{TEST_NAME}_{timestamp}")
    os.makedirs(run_dir)

    # Save the parameters to a file
    params = {
        "NUM_GENERATIONS": NUM_GENERATIONS,
        "POPULATION_SIZE": POPULATION_SIZE,
        "NUM_ELITE_ROBOTS": NUM_ELITE_ROBOTS,
        "MUTATION_RATE": MUTATION_RATE,
        "MIN_GRID_SIZE": MIN_GRID_SIZE,
        "MAX_GRID_SIZE": MAX_GRID_SIZE,
        "STEPS": STEPS,
        "SCENARIO": SCENARIO,
        "STAGNATION_LIMIT": STAGNATION_LIMIT,
        "MUTATION_RATE_INCREASE": MUTATION_RATE_INCREASE,
        "RANDOM_INJECTION_FRACTION": RANDOM_INJECTION_FRACTION,
        "MUTATION_METHOD": MUTATION_METHOD,
        "CROSSOVER_METHOD": CROSSOVER_METHOD,
        "VOXEL_TYPES": VOXEL_TYPES,
        "CONTROLLER": CONTROLLER.__name__,
        "OUTPUT_ROBOT_GIFS": OUTPUT_ROBOT_GIFS,
        "OUTPUT_POPULATION": OUTPUT_POPULATION,
        "TEST_NAME": TEST_NAME,
    }

    params_file = os.path.join(run_dir, "parameters.txt")
    with open(params_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    return run_dir


# these will be set each run:
RUN_DIR = None
LOG_FILE = None

def log(message):
    # Print to console and also write to log file.
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def apply_mutation(robot, rate):
    if MUTATION_METHOD == 'random':
        return mutate(robot, rate)
    elif MUTATION_METHOD == 'swap':
        # single swap per robot
        return swap_mutation(robot)
    elif MUTATION_METHOD == 'insert':
        # single insert per robot
        return insert_mutation(robot)
    else:
        raise ValueError(f"Unknown MUTATION_METHOD: {MUTATION_METHOD}")

def apply_crossover(p1, p2):
    if CROSSOVER_METHOD == 'mask':
        return crossover(p1, p2)
    elif CROSSOVER_METHOD == 'one_point':
        return one_point_crossover(p1, p2)
    elif CROSSOVER_METHOD == 'two_point':
        return two_point_crossover(p1, p2)
    else:
        raise ValueError(f"Unknown CROSSOVER_METHOD {CROSSOVER_METHOD}")


# ------------------ Function to Capture Simulation Frame ------------------
def capture_simulation_frame(robot, generation, scenario, steps, controller):

    #utils.simulate_best_robot(robot, scenario, steps)

    filename = os.path.join(RUN_DIR, f"simulation_frame_gen_{generation}.gif")

    utils.create_gif(robot, filename=filename, scenario=scenario, steps=steps, controller=controller)

# ------------------ Function to Capture All Robots Frame ------------------
def capture_population_frames(population, generation, scenario, steps, controller):
    """
    For the given population, capture a simulation GIF for each robot
    and store them in a subfolder corresponding to the current generation.
    """
    # Create a folder for this generation if it doesn't exist.
    gen_folder = os.path.join(RUN_DIR, f"gen_{generation}")
    if not os.path.exists(gen_folder):
        os.makedirs(gen_folder)

    # Loop over each robot in the population.
    for idx, robot in enumerate(population):
        filename = os.path.join(gen_folder, f"robot_{idx}.gif")
        # Capture a GIF for each robot.
        utils.create_gif(robot, filename=filename, scenario=scenario, steps=2, controller=controller)


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



# ------------------ Mutation Methods ------------------
def swap_mutation(robot, attempts=10):
    """
    Swap the contents of two random voxels.
    Tries up to `attempts` times to produce a valid robot.
    """
    x_shape, y_shape = robot.shape
    for _ in range(attempts):
        mutated = robot.copy()
        # pick two random positions
        i1, j1 = random.randrange(x_shape), random.randrange(y_shape)
        i2, j2 = random.randrange(x_shape), random.randrange(y_shape)
        # swap them
        mutated[i1, j1], mutated[i2, j2] = mutated[i2, j2], mutated[i1, j1]
        if is_valid_robot(mutated):
            return mutated
    return robot


def is_valid_robot(robot):
    return is_connected(robot) and np.any(robot == 3)



def insert_mutation(robot, attempts=10):
    """
    Remove one voxel and re-insert it elsewhere (adjacent to existing structure).
    Tries up to `attempts` times to produce a valid robot.
    """
    x_shape, y_shape = robot.shape

    def adjacent_empty_positions(r):
        empties = []
        for i in range(x_shape):
            for j in range(y_shape):
                if r[i, j] == 0:
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < x_shape and 0 <= nj < y_shape and r[ni, nj] != 0:
                            empties.append((i, j))
                            break
        return empties

    for _ in range(attempts):
        mutated = robot.copy()
        # choose a non-critical voxel to move
        nonzeros = [(i,j) for i in range(x_shape) for j in range(y_shape)
                    if mutated[i,j] != 0 and not is_critical_voxel(mutated, i, j)]
        if not nonzeros:
            break
        i1, j1 = random.choice(nonzeros)
        # find empty positions adjacent to structure
        empties = adjacent_empty_positions(mutated)
        if not empties:
            break
        i2, j2 = random.choice(empties)
        # perform the "insert": move type from (i1,j1) to (i2,j2)
        voxel_type = mutated[i1, j1]
        mutated[i1, j1] = 0
        mutated[i2, j2] = voxel_type

        if is_valid_robot(mutated):
            return mutated

    return robot


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

# ------------------ Crossover Methos ------------------

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

def one_point_crossover(parent1, parent2, attempts=10):
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    n = len(flat1)
    for _ in range(attempts):
        cut = random.randrange(1, n)
        c1 = np.concatenate([flat1[:cut], flat2[cut:]]).reshape(parent1.shape)
        c2 = np.concatenate([flat2[:cut], flat1[cut:]]).reshape(parent1.shape)
        if is_valid_robot(c1) and is_valid_robot(c2):
            return c1, c2
    # fallback
    return parent1.copy(), parent2.copy()


def two_point_crossover(parent1, parent2, attempts=10):
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    n = len(flat1)
    for _ in range(attempts):
        i, j = sorted(random.sample(range(1, n), 2))
        c1 = np.concatenate([flat1[:i], flat2[i:j], flat1[j:]]).reshape(parent1.shape)
        c2 = np.concatenate([flat2[:i], flat1[i:j], flat2[j:]]).reshape(parent1.shape)
        if is_valid_robot(c1) and is_valid_robot(c2):
            return c1, c2
    return parent1.copy(), parent2.copy()

# ------------------ Extra Variation Operators ------------------

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


# ---------------- Main Evolutionary Loop ----------------
def main_loop():
    best_fitness = -float('inf')
    best_robot = None

    # keep track of per‐generation stats
    gen_avg_fitness    = []
    gen_std_fitness    = []
    gen_best_per_gen   = []
    best_per_gen_ever  = []

    # Initial population
    population = [create_random_robot() for _ in range(POPULATION_SIZE)]

    # Track stagnation
    stagnation_counter = 0
    current_mutation_rate = MUTATION_RATE
    prev_best_fitness = None

    for gen in range(NUM_GENERATIONS):
        # Evaluate fitness
        population_fitness = [evaluate_fitness(robot) for robot in population]

        # Compute and record generation statistics
        avg_fit = float(np.mean(population_fitness))
        std_fit = float(np.std(population_fitness))
        gen_avg_fitness.append(avg_fit)
        gen_std_fitness.append(std_fit)

        # 3) Per‐gen best
        gen_best = float(np.max(population_fitness))
        gen_best_per_gen.append(gen_best)

        # 4) All‐time best
        if gen_best > best_fitness:
            best_fitness = gen_best
        best_per_gen_ever.append(best_fitness)

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

        # --- Increase mutation rate by 0.05 if this generation's best fitness hasn't changed ---
        if prev_best_fitness is not None and abs(gen_best_fit - prev_best_fitness) < 1e-6:
            current_mutation_rate = min(1.0, current_mutation_rate + 0.05)
            log(f"> No improvement from previous generation; increased mutation rate by 0.05 to {current_mutation_rate:.3f}")

        # Save the current best fitness value for the next generation.
        prev_best_fitness = gen_best_fit

        if OUTPUT_ROBOT_GIFS:
            capture_simulation_frame(best_robot, gen, SCENARIO, STEPS, CONTROLLER)

        if OUTPUT_POPULATION:
            capture_population_frames(population, gen, SCENARIO, STEPS, CONTROLLER)


        log(f"Gen {gen} | Best Fitness: {best_fitness:.3f} | Current Gen Best: {gen_best_fit:.3f} | Mutation Rate: {current_mutation_rate:.3f}")
        log(f"Gen {gen:2d} | Best: {gen_best_fit:.3f} | Avg: {avg_fit:.3f} ± {std_fit:.3f} | All‐Time Best: {best_fitness:.3f} | MutRate: {current_mutation_rate:.3f}")
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


        fitness_copy = population_fitness[:]

        while len(new_population) < POPULATION_SIZE:

            # Parent selection
            while True:
                parent1, idx1 = select_parent(pop_copy, fitness_copy)
                parent2, idx2 = select_parent(pop_copy, fitness_copy)
                if idx2 != idx1:
                    break

            # Parent selection
            #parent1, idx1 = select_parent(pop_copy, fitness_copy)
            # Remove it from the "pool" so we don't select the exact same index again
            #pop_copy.pop(idx1)
            #fitness_copy.pop(idx1)

            #parent2, idx2 = select_parent(pop_copy, fitness_copy)
            # Important: do not pop idx2 from pop_copy yet, because removing idx1 changes indexing
            # but for simplicity, we won't re-use the same parent in one iteration, so it's okay.

            # Crossover
            child1, child2 = apply_crossover(parent1, parent2)


            # Mutation
            child1 = apply_mutation(child1, current_mutation_rate)
            child2 = apply_mutation(child2, current_mutation_rate)


            # Add children
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    # After evolution, demonstrate the best robot
    log(f"\n=== EVOLUTION COMPLETE ===\nBest Fitness Found: {best_fitness}")
    log(f"Best Robot:\n{best_robot}")

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
    plt.show()

    # Optional: run best robot a few times, or create a GIF, etc.
    for i in range(3):
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)

    utils.create_gif(best_robot, filename='random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)


# ---------------- Run All Experiments ----------------
if __name__ == "__main__":
    experiments = []
    # 5 runs each, default (mask) crossover
    for method in ['random', 'insert', 'swap']:
        for run_id in range(1, 6):
            TEST_NAME = f"{method}_mutation"
            experiments.append((method, 'mask', run_id))
    # 5 runs each, one-point crossover
    for method in ['random', 'insert', 'swap']:
        for run_id in range(1, 6):
            TEST_NAME = f"{method}_mutation"
            experiments.append((method, 'one_point', run_id))
    # 5 runs each, two-point crossover
    for method in ['random', 'insert', 'swap']:
        for run_id in range(1, 6):
            TEST_NAME = f"{method}_mutation"
            experiments.append((method, 'two_point', run_id))

    for mut_method, cross_method, run_id in experiments:
        MUTATION_METHOD = mut_method
        CROSSOVER_METHOD = cross_method
        TEST_NAME = f"{mut_method}_mutation_{cross_method}"
        RUN_DIR = setup_run_directory()
        LOG_FILE = os.path.join(RUN_DIR, "log.txt")
        main_loop()