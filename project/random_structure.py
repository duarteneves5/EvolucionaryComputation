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
POPULATION_SIZE = 25    # number of robots per generation
NUM_ELITE_ROBOTS = max(1, int(POPULATION_SIZE * 0.1))   # number of top fitness robots to pass to the next population
MUTATION_RATE = 0.2
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 250
SCENARIO = 'Walker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait
#CONTROLLER = hopping_motion

def evaluate_fitness(robot_structure, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0


def create_random_robot():
    """Generate a valid random robot structure."""
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot


def random_search():
    """Perform a random search to find the best robot structure."""
    best_robot = None
    best_fitness = -float('inf')
    
    for it in range(NUM_GENERATIONS):
        robot = create_random_robot() 
        fitness_score = evaluate_fitness(robot)
        
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot
        
        print(f"Iteration {it + 1}: Fitness = {fitness_score}")
    
    return best_robot, best_fitness


def select_parent(
        population,
        fitness_scores,
        tournament_size=3
) -> tuple[np.ndarray, int]:
    candidates = random.sample(range(len(population)), tournament_size)
    best = candidates[0]
    for c in candidates[1:]:
        if fitness_scores[c] > fitness_scores[best]:
            best = c
    return population[best], best


def is_valid_robot(robot) -> bool:
    """Check if the robot is fully connected."""
    return is_connected(robot)

# Check if the robot has at least one actuator
def has_actuator(robot) -> bool:
    """Check if the robot has at least one actuator."""
    return bool(np.any(robot == 3))


def mutate(
        robot: np.ndarray,
        mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Mutate the robot by changing voxel types, removing voxels, or adding voxels.
    Args:
        robot: The robot structure to mutate.
        mutation_rate: Probability of mutating a voxel.
    Returns:
        The mutated robot structure.
    """
    mutated_robot = np.copy(robot)
    x_shape, y_shape = robot.shape

    for i in range(x_shape):
        for j in range(y_shape):
            if random.random() > mutation_rate:
                continue  # Skip this voxel based on mutation rate

            if mutated_robot[i, j] != 0:  # Non-empty voxel
                # Change voxel type or remove it
                if random.random() < 0.5 and not is_critical_voxel(mutated_robot, i, j):  # 50% chance to change type
                    new_type = random.choice([t for t in VOXEL_TYPES if t != mutated_robot[i, j]])
                    mutated_robot[i, j] = new_type
                else:  # 50% chance to remove voxel
                    #if not is_critical_voxel(mutated_robot, i, j):  # Check if voxel is not critical
                    mutated_robot[i, j] = 0  # Remove the voxel
            else:  # Empty voxel
                # Add a new voxel if it's adjacent to an existing voxel
                if is_adjacent_to_existing_voxel(mutated_robot, i, j):
                    mutated_robot[i, j] = random.choice(VOXEL_TYPES[1:])  # Add a non-empty voxel

        # Check if the mutated robot is still connected
        if is_connected(mutated_robot):
            return mutated_robot

    # If no valid mutation is found after multiple attempts, return the original robot
    return robot

def is_critical_voxel(robot: np.ndarray, i: int, j: int) -> bool:
    """
    Check if a voxel is critical (i.e., its removal would disconnect the robot).
    Args:
        robot: The robot structure.
        i, j: The coordinates of the voxel to check.
    Returns:
        True if the voxel is critical, False otherwise.
    """
    temp_robot = np.copy(robot)
    temp_robot[i, j] = 0  # Temporarily remove the voxel
    return not is_connected(temp_robot)

def is_adjacent_to_existing_voxel(robot: np.ndarray, i: int, j: int) -> bool:
    """
    Check if an empty voxel is adjacent to an existing voxel.
    Args:
        robot: The robot structure.
        i, j: The coordinates of the voxel to check.
    Returns:
        True if the voxel is adjacent to an existing voxel, False otherwise.
    """
    x_shape, y_shape = robot.shape
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check all four directions
        ni, nj = i + di, j + dj
        if 0 <= ni < x_shape and 0 <= nj < y_shape and robot[ni, nj] != 0:
            return True
    return False



# Crossover
def crossover(parent1, parent2):
    """Crossover two parents while ensuring the offspring are connected."""
    for _ in range(10):  # Try up to 10 times to find a valid crossover
        if random.random() < 0.5:
            crossover_point = random.randint(1, parent1.shape[0] - 1)
            child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            crossover_point = random.randint(1, parent1.shape[1] - 1)
            child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
            child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))
        if is_valid_robot(child1) and is_valid_robot(child2):
            return child1, child2
    return parent1, parent2

best_fitness = -float('inf')
previous_fitness = -float('inf')
best_robot = None
population = [create_random_robot() for _ in range(POPULATION_SIZE)]
#fitness_window = [.0, .0, .0]
default_mutation_rate = MUTATION_RATE
mutation_rate = MUTATION_RATE

for gen in range(NUM_GENERATIONS):
    # get the population fitness
    population_fitness = [evaluate_fitness(robot) for robot in population]

    # sort the population fitness
    sorted_population_fitness_idxs = np.argsort(population_fitness)
    # store the index of the most fit robot
    biggest_fitness_idx = sorted_population_fitness_idxs[-1]
    # if the fitness is bigger than the current best fitness, update the best fitness and robot
    if population_fitness[biggest_fitness_idx] > best_fitness:
        best_fitness = population_fitness[biggest_fitness_idx]
        best_robot = population[biggest_fitness_idx]

    #fitness_window[gen%FITNESS_SLIDING_WINDOW_SIZE] = best_fitness
    #if fitness_window
    #if best_fitness == previous_fitness:
    #    mutation_rate = min(1.0, mutation_rate*1.5)
    #    print(f"Stagnant generation, increasing mutation rate to {mutation_rate}")
    #else:
    #    print(f"Generation improved, fallback to default mutation rate {default_mutation_rate}")
    #    mutation_rate = default_mutation_rate

    print(f"Gen {gen}: Best fitness: {population_fitness[biggest_fitness_idx]} - Best robot structure: {best_robot}")

    new_population = []
    # pass the elitists to the next generation
    for i in sorted_population_fitness_idxs[-NUM_ELITE_ROBOTS:]:
        new_population.append(population[i])

    while len(new_population) < POPULATION_SIZE:
        parent1, parent1_idx = select_parent(population, population_fitness)
        population.pop(parent1_idx)
        parent2, _ = select_parent(population, population_fitness)
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([mutate(child1), mutate(child2)])

    previous_fitness = best_fitness
    population = new_population

i = 0
while i < 10:
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    i += 1
utils.create_gif(best_robot, filename='random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)