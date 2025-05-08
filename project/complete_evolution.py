import multiprocessing
import os
import random
import numpy as np
import torch

import utils
import secrets
from concurrent.futures import ProcessPoolExecutor
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from evogym.envs import *
import itertools
from neural_controller import *
from project.random_controler import NUM_GENERATIONS

# 'random', 'swap', or 'insert'
MUTATION_METHOD = 'insert'
# 'mask', 'one_point', 'two_point'
CROSSOVER_METHOD = 'one_point'

SCENARIO = "CaveCrawler-v0"
#SCENARIO = "GapJumper-v0"

SEED = secrets.randbelow(1_000_000_000)
print(f"SEEDING WITH {SEED}")
np.random.seed(SEED)
random.seed(SEED)

MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

NUM_GENERATIONS = 100
CMA_ITERS = 3 # 3 controller optimizations for 1 structure optimization
POPULATION_SIZE = 25
NUM_ELITE_ROBOTS = max(1, int(POPULATION_SIZE * 0.06))  # 6% elitism
STEPS = 1000

def apply_mutation(robot, rate):
    if MUTATION_METHOD == 'random':
        return mutate(robot, rate)
    elif MUTATION_METHOD == 'swap':
        if random.random() < rate:
            return swap_mutation(robot)
        return robot
    elif MUTATION_METHOD == 'insert':
        if random.random() < rate:
            return insert_mutation(robot)
        return robot
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


# ------------------ Evaluation and Variation Functions ------------------
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

## CONTROLLER EVOLUTION

class CMAESOptimizer:
    def __init__(
            self,
            brain,
            population_size = None,
            fitness_function = None
    ):
        if population_size is None or not isinstance(population_size, int) or population_size < 1:
            raise ValueError("population_size must be a positive integer")
        if fitness_function is None or not callable(fitness_function):
            raise ValueError("fitness_function must be provided and be a callable")

        # strategy parameters
        self.pop_size = population_size
        self.mu = population_size // 2

        # expert recombination weights: log-linear scheme
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        #mu_eff = 1.0 / np.sum(self.weights ** 2)
        #print("μ_eff =", mu_eff)

        self.INITIAL_SPREAD = 0.7
        self.MIN_SPREAD = 0.05
        self.MAX_SPREAD = 1.0
        self.COOLING_RATE = 0.85
        self.STAGNATION_WINDOW = 5  # generations without improvement
        self.STAGNATION_BOOST = 1.2  # spread multiplier on stagnation
        self.IMPROVEMENT_THRESHOLD = 0.05  # minimum relative improvement
        self.stagnation_counter = 0
        self.spread = self.INITIAL_SPREAD
        #self.c_cov = 2 / ((len(self.get_mean_vector(brain)) + 2) ** 2)
        #self.c_cov = 0.2
        self.c_cov = 0.1 / (len(self.get_mean_vector(brain)) ** 0.5)

        self.m = self.get_mean_vector(brain) # mean vector
        self.C = (self.spread**2) * np.eye(len(self.m)) # initial covariance
        if population_size:
            self.population_size = population_size
        else:
            self.population_size = int(4 + 3 * np.log(len(self.m)))
        self.mu = self.population_size // 2

        self.executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        self.fitness_function = fitness_function

        self.previous_gen_best_fitness = -np.inf
        self.current_gen_best_fitness = -np.inf

        # statistics
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.path_of_means = []
        self.all_samples = []


    def get_mean_vector(self, brain):
        # extracts flattened parameters from the neural controller
        return get_weights(brain, flatten=True)


    def update_spread(
            self,
            prev_best,
            current_best
    ):
        if prev_best == -np.inf: return

        # raw improvement (could be negative)
        delta = (current_best - prev_best) / (abs(prev_best) + 1e-8)

        if delta > self.IMPROVEMENT_THRESHOLD:
            # we made progress: exploit harder
            self.spread = max(self.spread * self.COOLING_RATE, self.MIN_SPREAD)
            self.stagnation_counter = 0
        elif delta < -self.IMPROVEMENT_THRESHOLD:
            # no progress: explore more
            self.spread = min(self.spread * self.STAGNATION_BOOST, self.MAX_SPREAD)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.STAGNATION_WINDOW:
                self.spread = min(self.spread * self.STAGNATION_BOOST ** 2, self.MAX_SPREAD)
                self.stagnation_counter = 0


    def sample_population(self):
        cov = self.C * (self.spread**2)
        # multivariate normal with current mean and covariance
        return np.random.multivariate_normal(self.m, cov, size=self.pop_size)


    def update_distribution(
            self,
            population,
            fitnesses,
            prev_best,
            curr_best,
    ):
        # select top individuals
        idx_sorted = np.argsort(fitnesses)[::-1]
        best_idx = idx_sorted[: self.mu]
        elites = population[best_idx]

        # recombine mean with expert weights
        new_m = np.sum(elites.T * self.weights, axis=1)

        # recombine covariance with expert weights
        diffs = elites - new_m
        weighted_cov = np.zeros_like(self.C)
        for w, diff in zip(self.weights, diffs):
            weighted_cov += w * np.outer(diff, diff)

        # update params
        self.m = new_m
        self.path_of_means.append(self.m.copy()) # store current mean vector
        self.C = (1 - self.c_cov) * self.C + self.c_cov * weighted_cov

        # adapt step size
        self.update_spread(prev_best, curr_best)
        #self.C *= self.spread ** 2


    def step(
            self,
            generation
    ):
        pop = self.sample_population()
        # calculate samples from current population
        for i in range(self.pop_size):
            self.all_samples.append((generation, pop[i, 0], pop[i, 1]))

        fitnesses = list(self.executor.map(
            self.fitness_function,
            pop,
            itertools.repeat(generation))
        )

        self.previous_gen_best_fitness = self.current_gen_best_fitness
        self.current_gen_best_fitness = max(fitnesses)
        self.best_fitness_history.append(self.current_gen_best_fitness)
        self.mean_fitness_history.append(np.mean(fitnesses))

        self.update_distribution(pop, fitnesses, self.previous_gen_best_fitness, self.current_gen_best_fitness)
        return pop, fitnesses


class BodyFitness:
    def __init__(self, body, mask):
        self.body = body
        self.mask = mask

    # TODO this fitness function is horse shit, take more into account on the fitness calculation
    # TODO especialy stats related to the environment, but mainly, distance, speed and efficiency (the basics)
    def __call__(self, weights, generation=0):
        brain = NeuralController(MAX_BRAIN_INPUT_SIZE, MAX_BRAIN_OUTPUT_SIZE)
        set_weights(brain, weights, reconstruct_weights=True)
        env = gym.make(SCENARIO, max_episode_steps=STEPS,
                       body=self.body, connections=get_full_connectivity(self.body))
        obs, _ = env.reset()
        start_x = env.sim.object_pos_at_time(0, 'robot')[0].mean()
        for _ in range(STEPS):
            pad = np.zeros(MAX_BRAIN_INPUT_SIZE, dtype=np.float32)
            pad[:len(obs)] = obs
            with torch.no_grad():
                logits = brain(torch.from_numpy(pad).unsqueeze(0)).squeeze(0).numpy()
            act = logits[self.mask]
            obs, *_ = env.step(act)
        end_x = env.sim.object_pos_at_time(env.sim.get_time(), 'robot')[0].mean()
        env.close()

        print(f"distance walked: {end_x - start_x}")
        return end_x - start_x


MAX_BRAIN_INPUT_SIZE = 98 # 86
MAX_BRAIN_OUTPUT_SIZE = 25
class Genotype:
    def __init__(self, structure=None, weights=None):
        self.structure = create_random_robot() if structure is None else structure
        self.act_mask = None
        self.update_mask()

        self.brain = NeuralController(MAX_BRAIN_INPUT_SIZE, MAX_BRAIN_OUTPUT_SIZE, init_params=True)
        if weights is not None:
            set_weights(self.brain, weights, reconstruct_weights=True)
        fitness_wrapper = BodyFitness(self.structure, self.act_mask)
        self.cma = CMAESOptimizer(self.brain, POPULATION_SIZE, fitness_wrapper)

    def update_mask(self):
        flat = self.structure.flatten()
        self.act_mask = (flat == 3) | (flat == 4)

    def update_structure(self, new_structure):
        self.structure = new_structure
        self.update_mask()
        self.cma = CMAESOptimizer(self.brain, POPULATION_SIZE, evaluate_fitness)

    def update_weights(self, weights, reconstruct_weights=False):
        set_weights(self.brain, weights, reconstruct_weights)

    def act(self, obs_tensor):
        with torch.no_grad():
            logits = self.brain(obs_tensor).squeeze(0).cpu().numpy()
        return logits[self.act_mask]


## EVALUTATION
def evaluate_fitness(
        genotype,
        view=False,
        w_distance=0.5,
        w_efficiency=0.25,
        w_speed=0.25,
        w_survival=0.0,
        w_fall=0.0,
        return_components=False
):
    DIST_REF = 80.0 # best possible distance
    EFF_REF = 0.6  # good efficiency
    SPD_REF = 0.07  # good speed in m/s

    print(f"GENO: {genotype}")

    # build env for this structure
    connections = get_full_connectivity(genotype.structure)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=genotype.structure, connections=connections)
    sim = env.sim
    obs, _ = env.reset()
    obs_pad = np.zeros(MAX_BRAIN_INPUT_SIZE, dtype=np.float32)
    obs_pad[: len(obs)] = obs

    viewer = EvoViewer(sim) if view else None
    if view:
        viewer.track_objects('robot')

    state = env.reset()[0]  # Get initial state
    t_reward = 0
    start_com = sim.object_pos_at_time(0, 'robot').mean(axis=1)
    prev_act = np.zeros(env.action_space.shape[0])
    prev_acc = np.zeros_like(prev_act)
    total_energy = 0.0
    total_actuation = 0.0
    jerk_sum = 0.0
    upright_accum = 0.0
    survived_steps = STEPS
    fell_over = False
    MAX_TILT = 99999999
    prev_x = start_com[0]
    progress_reward = 0.0

    for t in range(1, STEPS+1):
        # Update actuation before stepping
        #state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Convert to tensor
        #with torch.no_grad():
        #    action = brain(state_tensor).detach().numpy().flatten()  # Get action

        action = genotype.act(obs_pad)

        #action_magnitude = np.sum(np.abs(action))
        #energy = (np.sum(np.abs(action - prev_act)) +  0.1*action_magnitude) / t

        # energy: torque·velocity proxy
        velocity = (action - prev_act)
        total_energy += np.sum(np.abs(action * velocity))
        total_actuation += np.sum(np.abs(action))

        # jerk: change in acceleration
        acc = velocity
        jerk_sum += np.sum((acc - prev_acc) ** 2)

        prev_act = action
        prev_acc = acc

        cur_x = sim.object_pos_at_time(sim.get_time(), 'robot')[0].mean()
        progress_reward += max(0.0, cur_x - prev_x)  # no reward for sliding back
        prev_x = cur_x

        # posture: robot tilt w.r.t. x‐axis
        angle = sim.object_orientation_at_time(sim.get_time(), 'robot')
        fell_over = abs(angle) > MAX_TILT

        upright_accum += abs(np.cos(angle))  # 1.0 if perfectly aligned

        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, info = env.step(action)
        #t_reward += reward
        if terminated or truncated:
            survived_steps = t
            break

    tf = sim.get_time()
    end_com = sim.object_pos_at_time(tf, 'robot').mean(axis=1)
    distance = end_com[0] - start_com[0] # allow negative distance values for backwards moving penalties
    efficiency = distance / (total_energy + 1e-6) * 1e6
    speed = distance / max(1, tf)
    survival_ratio = (survived_steps / STEPS)

    '''
    if SCENARIO == "ObstacleTraverser-v0":
        dist_norm = np.clip(progress_reward / DIST_REF, 0.0, 1.0)
    elif SCENARIO == "DownStepper-v0":
        dist_norm = np.clip(distance / DIST_REF, 0.0, 1.0)
    '''
    dist_norm = np.clip(progress_reward / DIST_REF, 0.0, 1.0)

    eff_norm = np.clip(efficiency / EFF_REF, 0.0, 1.0)
    spd_norm = np.clip(speed / SPD_REF, 0.0, 1.0)
    #death_penalty = (1 - survival_ratio) * 50.0

    if view:
        viewer.close()

    env.close()

    fitness = (
        w_distance * dist_norm +
        w_efficiency * eff_norm +
        w_speed * spd_norm
        - w_survival * (1 - survival_ratio)
    )

    if fell_over:
        fitness -= w_fall

    if return_components:
        return fitness, w_distance * dist_norm, w_efficiency * eff_norm, w_speed * spd_norm, w_survival * (1 - survival_ratio), int(fell_over) * w_fall, total_energy

    return fitness


POOL = ProcessPoolExecutor(max_workers=max(1, os.cpu_count()))
def parallel_fit_eval(population):
    return list(POOL.map(evaluate_fitness, population))


def main():
    population = [Genotype() for _ in range(POPULATION_SIZE)]

    for gen in range(NUM_GENERATIONS):
        for i in range(CMA_ITERS):
            for genotype in population:
                genotype.cma.step(i)
            for genotype in population:
                genotype.update_weights(genotype.cma.m, reconstruct_weights=True)

        fits = parallel_fit_eval(population)
        print(fits)
        print(f"Gen {gen:03d}  best={max(fits):.2f}  mean={np.mean(fits):.2f}")

        # Sort population by fitness to get the top individuals
        sorted_indices = np.argsort(fits)
        new_population = []

        # Elitism: keep top N
        for i in sorted_indices[-NUM_ELITE_ROBOTS:]:
            new_population.append(population[i])

        # Fill the rest of the new population
        # Make a copy so we don't remove from original 'population' while selecting
        pop_copy = population[:]

        fitness_copy = fits[:]

        while len(new_population) < POPULATION_SIZE:

            # Parent selection
            while True:
                parent1, idx1 = select_parent(pop_copy, fitness_copy)
                parent2, idx2 = select_parent(pop_copy, fitness_copy)
                if idx2 != idx1:
                    break

            # Parent selection
            # parent1, idx1 = select_parent(pop_copy, fitness_copy)
            # Remove it from the "pool" so we don't select the exact same index again
            # pop_copy.pop(idx1)
            # fitness_copy.pop(idx1)

            # parent2, idx2 = select_parent(pop_copy, fitness_copy)
            # Important: do not pop idx2 from pop_copy yet, because removing idx1 changes indexing
            # but for simplicity, we won't re-use the same parent in one iteration, so it's okay.

            # Crossover
            child1, child2 = apply_crossover(parent1, parent2)

            # Mutation
            child1 = apply_mutation(child1, 0.05) # TODO implement dynamic stagnation handling and mutation rate adjustment
            child2 = apply_mutation(child2, 0.05)

            # Add children
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()