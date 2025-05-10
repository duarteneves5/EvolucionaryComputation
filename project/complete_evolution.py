import multiprocessing
import os
import random
import numpy as np
import torch
import datetime
import time
import matplotlib.pyplot as plt
import imageio
import csv
from pathlib import Path
import secrets
from concurrent.futures import ProcessPoolExecutor
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from evogym.envs import *
import utils
import itertools
from neural_controller import *
from fixed_controllers import *
import cProfile, pstats

GLOBAL_EXEC = ProcessPoolExecutor(max_workers=os.cpu_count())

# 'random', 'swap', or 'insert'
MUTATION_METHOD = 'swap'
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
POPULATION_SIZE = 15
NUM_ELITE_ROBOTS = 1  # keep only the best robot
STEPS = 500

CONTROLLER = alternating_gait

# Dynamic stagnation handling parameters
STAGNATION_LIMIT = 5          # generations without all-time best improvement
BASE_MUTATION_RATE = 0.2
MUTATION_RATE_INCREASE = 0.05  # additive increase (e.g., +0.2)
OUTPUT_POPULATION = True   # put with the other globals

TEST_NAME = "baseline"

OUTPUT_ROBOT_GIFS = True                # this serves to output the robot gifs on the evogym so we can better

def soft_norm(x, ref, k=4.0):
    """smoothly maps x ≥ 0 to (0, 1); reaches 0.98 around x≈ref"""
    x = np.maximum(0.0, x)
    return 1.0 - np.exp(-k * x / (ref + 1e-8))

def goal_sigmoid(x, centre, width=0.5):
    """sharp 0→1 transition once x passes `centre` (for GapJumper)."""
    return 1.0 / (1.0 + np.exp(-(x - centre) / width))


# ------------------ Run Directory and Logger Setup and Helper Functions ------------------
def setup_run_directory():
    base_dir = os.path.join("results", "complete_evolution")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Use a timestamp to create a unique run folder.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"CoEvolutionary_run_{TEST_NAME}_{timestamp}_{SEED}")
    os.makedirs(run_dir)

    # Save the parameters to a file
    params = {
        "NUM_GENERATIONS": NUM_GENERATIONS,
        "POPULATION_SIZE": POPULATION_SIZE,
        "NUM_ELITE_ROBOTS": NUM_ELITE_ROBOTS,
        "MUTATION_RATE": BASE_MUTATION_RATE,
        "MIN_GRID_SIZE": MIN_GRID_SIZE,
        "MAX_GRID_SIZE": MAX_GRID_SIZE,
        "STEPS": STEPS,
        "SCENARIO": SCENARIO,
        "STAGNATION_LIMIT": STAGNATION_LIMIT,
        "MUTATION_RATE_INCREASE": MUTATION_RATE_INCREASE,
        "MUTATION_METHOD": MUTATION_METHOD,
        "CROSSOVER_METHOD": CROSSOVER_METHOD,
        "VOXEL_TYPES": VOXEL_TYPES,
        "CONTROLLER": "Custom",
        "TEST_NAME": TEST_NAME,
        "SEED": SEED,
    }

    params_file = os.path.join(run_dir, "parameters.txt")
    with open(params_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    return run_dir


# these will be set each run:
RUN_DIR = None
LOG_FILE = "log.txt"

def log(message):
    # Print to console and also write to log file.
    print(message)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(message + "\n")

# ------------------ Function to Capture Simulation Frame ------------------
def capture_simulation_frame(robot, generation, scenario, steps, controller):

    #utils.simulate_best_robot(robot, scenario, steps)

    filename = os.path.join(RUN_DIR, f"simulation_frame_gen_{generation}.gif")

    utils.create_gif(robot.structure, filename=filename, scenario=scenario, steps=2, controller=controller)


# -------------------------------- SAVE FULL POPULATION --------------------------------
def capture_population_frames(population, generation, scenario, steps,controller):
    """
    Save a tiny GIF for every robot in the current population.
    Each GIF lives in   <RUN_DIR>/gen_<generation>/robot_<idx>.gif
    """
    # Create a folder for this generation if it doesn't exist.
    gen_folder = os.path.join(RUN_DIR, f"gen_{generation}")
    if not os.path.exists(gen_folder):
        os.makedirs(gen_folder)

    # Loop over each robot in the population.
    for idx, robot in enumerate(population):
        filename = os.path.join(gen_folder, f"robot_{idx}.gif")
        # Capture a GIF for each robot.
        utils.create_gif(robot.structure, filename=filename, scenario=scenario, steps=2, controller=controller)

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

        #self.executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        self.executor = GLOBAL_EXEC
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


'''
class BodyFitness:
    """
    Scenario-aware fitness usable both by CMA-ES (optimising weights)
    and by higher-level genotype evaluation.  Works for CaveCrawler-v0
    and GapJumper-v0.
    """
    def __init__(self, body, mask):
        self.body = body
        self.mask = mask          # boolean mask selecting actuator indices

        # scenario-specific reference values
        if SCENARIO.startswith("Cave"):
            self.DIST_REF  = 60.0
            self.SPEED_REF = 0.05
            self.GAP_X     = None
        elif SCENARIO.startswith("Gap"):
            self.DIST_REF  = 40.0
            self.SPEED_REF = 0.08
            self.GAP_X     = 25.0     # centre of the chasm in env metres
        else:
            raise ValueError(f"Unsupported scenario {SCENARIO}")

    # ------------------------------------------------------------------
    def __call__(self, weights, generation: int = 0):
        # ------------- rebuild the controller -------------
        brain = NeuralController(MAX_BRAIN_INPUT_SIZE, MAX_BRAIN_OUTPUT_SIZE)
        set_weights(brain, weights, reconstruct_weights=True)

        env = gym.make(
            SCENARIO,
            max_episode_steps=STEPS,
            body=self.body,
            connections=get_full_connectivity(self.body)
        )
        sim = env.sim

        obs, _ = env.reset()
        pad = np.zeros(MAX_BRAIN_INPUT_SIZE, dtype=np.float32)
        pad[:len(obs)] = obs

        # ------------- accumulators -------------
        start_x      = sim.object_pos_at_time(0, "robot")[0].mean()
        prev_act     = np.zeros(np.count_nonzero(self.mask))
        prev_vel     = np.zeros_like(prev_act)
        total_energy = 0.0
        jerk_sum     = 0.0
        fell         = False

        # ------------- episode roll-out -------------
        for _ in range(STEPS):
            with torch.no_grad():
                logits = brain(torch.from_numpy(pad).unsqueeze(0)).squeeze(0).numpy()
            act = logits[self.mask]

            vel          = act - prev_act
            total_energy += np.sum(np.abs(act * vel))
            jerk_sum     += np.sum((vel - prev_vel) ** 2)
            prev_act, prev_vel = act, vel

            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                fell = True
                break
            pad[:len(obs)] = obs

        tf       = sim.get_time()
        end_x    = sim.object_pos_at_time(tf, "robot")[0].mean()
        distance = max(0.0, end_x - start_x)
        speed    = distance / max(1e-6, tf)
        env.close()

        # ------------- term scores -------------
        progress   = soft_norm(distance, self.DIST_REF)
        velocity   = soft_norm(speed,     self.SPEED_REF)
        efficiency = soft_norm(distance / (total_energy + 1e-6) * 1e5, 0.4)
        smoothness = 1.0 / (1.0 + jerk_sum / (STEPS * 10.0))

        if self.GAP_X is None:        # CaveCrawler
            goal_bonus = distance / self.DIST_REF
        else:                         # GapJumper
            goal_bonus = goal_sigmoid(end_x, self.GAP_X)

        if fell:
            progress   *= 0.2
            velocity   *= 0.2
            efficiency *= 0.2
            smoothness *= 0.2
            goal_bonus  = 0.0

        # ------------- weighted blend -------------
        fitness = (
            0.45 * progress   +
            0.25 * velocity   +
            0.15 * efficiency +
            0.10 * smoothness +
            0.05 * goal_bonus
        )
        return fitness
'''
class OLDBodyFitness:
    """
    Reference-free fitness.
    Positive terms:
        • distance travelled           (metres)
        • average horizontal speed     (m/s)
        • goal bonus                   (0 or 1)

    Cost terms (subtracted):
        • total actuation energy proxy (|τ·Δτ|)
        • jerk of actuation sequence   (Σ(Δacc)²)

    Final formula
        fitness = + 1.0 * distance
                  + 5.0 * speed
                  + 50.0 * goal_bonus
                  - 1e-6 * total_energy
                  - 1e-3 * jerk_sum
    """
    def __init__(self, body, mask):
        self.body = body
        self.mask = mask

        # determine whether we can compute a goal bonus
        if SCENARIO.startswith("Gap"):
            self.GAP_X = 25.0       # centre of chasm, adjust if env differs
        else:
            self.GAP_X = None       # CaveCrawler uses only distance

    # ------------------------------------------------------------
    def __call__(self, weights, generation=0, return_components=False):
        # ---------- rebuild brain ----------
        brain = NeuralController(MAX_BRAIN_INPUT_SIZE, MAX_BRAIN_OUTPUT_SIZE)
        set_weights(brain, weights, reconstruct_weights=True)

        env = gym.make(
            SCENARIO,
            max_episode_steps=STEPS,
            body=self.body,
            connections=get_full_connectivity(self.body)
        )
        sim = env.sim

        obs, _ = env.reset()
        pad = np.zeros(MAX_BRAIN_INPUT_SIZE, dtype=np.float32)
        pad[:len(obs)] = obs  # ① initialise with first obs
        pad_t = torch.from_numpy(pad).unsqueeze(0)

        start_x = sim.object_pos_at_time(0, "robot")[0].mean()
        prev_x = start_x
        prev_act = np.zeros(np.count_nonzero(self.mask))
        prev_acc = np.zeros_like(prev_act)

        dist_acc = 0.0
        energy = 0.0
        jerk_sum = 0.0
        fell = False

        t_reward = 0.0
        stalled = 0
        # ---------- episode ----------
        for _ in range(STEPS):
            with torch.no_grad():
                act = brain(pad_t).squeeze(0).numpy()[self.mask]

            vel = act - prev_act
            energy += np.sum(np.abs(act * vel))
            jerk_sum += np.sum((vel - prev_acc) ** 2)
            prev_act, prev_acc = act, vel

            obs, reward, term, trunc, _ = env.step(act)
            t_reward += reward
            cur_x = sim.object_pos_at_time(sim.get_time(), "robot")[0].mean()
            dist_acc = cur_x - start_x  # always non-negative; we clamp later
            walked_distance = cur_x - prev_x
            prev_x = cur_x

            if walked_distance < 0.01:
                stalled += 1
            else:
                stalled = 0
            if stalled > 100:
                break


            if term or trunc:
                fell = True
                break

            pad[:] = 0.0  # clear and refill (same buffer)
            pad[:len(obs)] = obs

        env.close()

        distance = max(0.0, dist_acc)
        speed = distance / max(1e-6, sim.get_time())
        efficiency = distance / (energy + 1e-6)

        goal_bonus = 0.0
        if self.GAP_X is not None and (start_x + distance) >= self.GAP_X:
            goal_bonus = 50.0

        if fell:
            goal_bonus *= 0.0  # no bonus if it crashes
            speed *= 0.2
            efficiency *= 0.2

        # ---------- weighted sum ----------
        fitness = (
                + 3.0 * distance  # meters
                + 8.0 * speed  # m/s (typical 0–0.1)
                + 20.0 * efficiency  # m / (act-energy)
                + goal_bonus
                - 1e-7 * energy  # much lighter penalty
                - 1e-4 * jerk_sum
        )

        if return_components:
            return (t_reward, distance, speed, efficiency,
                    goal_bonus, energy, jerk_sum, fell)
        return t_reward

# ------------------------------------------------------------------------
#  Safe, task-agnostic fitness for EvoGym CaveCrawler / GapJumper
# ------------------------------------------------------------------------
class BodyFitness:
    """
    Main term
        R_env         – the cumulative reward returned by the EvoGym env
                        (Δx for CaveCrawler, landing bonus for GapJumper).

    Early-stage shaping (fades out after N_SHAPING_GEN generations)
        + c_d * distance
        + c_s * mean speed

    Soft costs
        − c_E * total actuation energy proxy   (|τ·Δτ|)
        − c_J * jerk of actuation sequence     (Σ(Δacc)²)

    Hard penalties
        − CRASH_PENALTY   if the robot terminates early
        + GAP_BONUS       once it clearly clears the gap (GapJumper only)
    """
    # ---------- hyper-parameters ----------
    N_SHAPING_GEN  = 20      # generations over which shaping fades to zero
    c_d            = 0.20    # distance shaping coefficient
    c_s            = 0.40    # speed shaping coefficient
    c_E            = 1e-7    # energy cost coefficient
    c_J            = 1e-4    # jerk  cost coefficient
    CRASH_PENALTY  = 20.0
    GAP_BONUS      = 100.0   # added once end_x passes the gap centre

    def __init__(self, body, mask):
        self.body = body
        self.mask = mask

        # centre of the chasm for the stock GapJumper map
        self.gap_x = 25.0 if SCENARIO.startswith("Gap") else None

    # ------------------------------------------------------------------
    def __call__(self, weights, generation=0, return_components=False):
        # ---- rebuild the tiny network (fast) ----
        brain = NeuralController(MAX_BRAIN_INPUT_SIZE, MAX_BRAIN_OUTPUT_SIZE)
        set_weights(brain, weights, reconstruct_weights=True)

        # ---- make env ----
        env = gym.make(
            SCENARIO,
            max_episode_steps=STEPS,
            body=self.body,
            connections=get_full_connectivity(self.body)
        )
        sim  = env.sim
        obs, _ = env.reset()

        pad   = np.zeros(MAX_BRAIN_INPUT_SIZE, dtype=np.float32)
        pad[:len(obs)] = obs
        pad_t = torch.from_numpy(pad).unsqueeze(0)

        start_x   = sim.object_pos_at_time(0, "robot")[0].mean()
        prev_act  = np.zeros(np.count_nonzero(self.mask))
        prev_acc  = np.zeros_like(prev_act)

        # ---- accumulators ----
        R_env       = 0.0
        distance    = 0.0
        speed       = 0.0     # will compute at the end
        energy      = 0.0
        jerk_sum    = 0.0
        fell        = False

        for _ in range(STEPS):
            with torch.no_grad():
                act = brain(pad_t).squeeze(0).numpy()[self.mask]

            # physics proxies
            vel       = act - prev_act
            energy   += np.sum(np.abs(act * vel))
            jerk_sum += np.sum((vel - prev_acc) ** 2)
            prev_act, prev_acc = act, vel

            # step env
            obs, r, terminated, truncated, _ = env.step(act)
            R_env += r

            if terminated or truncated:
                fell = True
                break

            pad[:] = 0.0
            pad[:len(obs)] = obs

        # ---- episode stats ----
        tf        = sim.get_time()
        end_x     = sim.object_pos_at_time(tf, "robot")[0].mean()
        distance  = max(0.0, end_x - start_x)
        speed     = distance / max(1e-6, tf)
        env.close()

        # ---- shaping scale (decays linearly) ----
        shaping_scale = max(0.0, 1.0 - generation / self.N_SHAPING_GEN)

        # ---- fitness components ----
        fitness  = R_env
        fitness += shaping_scale * (self.c_d * distance + self.c_s * speed)
        fitness += self.GAP_BONUS if (self.gap_x and end_x >= self.gap_x) else 0.0
        fitness -= self.c_E * energy
        fitness -= self.c_J * jerk_sum
        if fell:
            fitness -= self.CRASH_PENALTY

        if return_components:
            return (
                fitness,
                R_env,
                distance,
                speed,
                energy,
                jerk_sum,
                int(fell),
            )

        return fitness


MAX_BRAIN_INPUT_SIZE = 102 # 86
MAX_BRAIN_OUTPUT_SIZE = 25
class Genotype:
    def __init__(self, structure=None, weights=None):
        self.structure = create_random_robot() if structure is None else structure
        self.act_mask = None
        self.update_mask()

        self.brain = NeuralController(MAX_BRAIN_INPUT_SIZE, MAX_BRAIN_OUTPUT_SIZE, init_params=True)
        if weights is not None:
            set_weights(self.brain, weights, reconstruct_weights=True)
        self.fitness_wrapper = BodyFitness(self.structure, self.act_mask)
        self.cma = CMAESOptimizer(self.brain, POPULATION_SIZE, self.fitness_wrapper)

    def update_mask(self):
        flat = self.structure.flatten()
        self.act_mask = (flat == 3) | (flat == 4)

    def update_structure(self, new_structure):
        self.structure = new_structure
        self.update_mask()
        self.fitness_wrapper = BodyFitness(self.structure, self.act_mask)
        self.cma = CMAESOptimizer(self.brain, POPULATION_SIZE, self.fitness_wrapper)

    def update_weights(self, weights, reconstruct_weights=False):
        set_weights(self.brain, weights, reconstruct_weights)

    def act(self, obs):
        # Make sure the net always sees a tensor
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0)  # shape (1, D)

        # if it’s already a 1-D tensor, add batch dim
        if isinstance(obs, torch.Tensor) and obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            logits = self.brain(obs).squeeze(0).cpu().numpy()
        return logits[self.act_mask]

    def fitness(self):
        """Evaluate this genotype with its current weights."""
        flat_w = get_weights(self.brain, flatten=True)
        return self.cma.fitness_function(flat_w)


def calc_fitness(structure: np.ndarray, flat_weights: np.ndarray):
    """
    Stand-alone worker for multiprocessing.

    Parameters
    ----------
    structure : (H, W) int8 array
        Voxel grid of the robot body.
    flat_weights : (N,) float32 array
        Flattened neural-network parameters.
    """
    # actuator mask: voxels 3 or 4 are actuators
    mask = (structure.flatten() == 3) | (structure.flatten() == 4)

    # BodyFitness is cheap to construct (just stores numpy arrays)
    fitness_fn = BodyFitness(structure, mask)
    fitness, distance, speed, efficiency, goal_bonus, energy, jerk_sum, fell,  = fitness_fn(flat_weights, return_components=True)
    return fitness, distance, speed, efficiency, goal_bonus, energy, jerk_sum, fell



def main():
    global RUN_DIR, LOG_FILE
    RUN_DIR = setup_run_directory()
    LOG_FILE = os.path.join(RUN_DIR, "log.txt")
    population = [Genotype() for _ in range(POPULATION_SIZE)]

    # per‐generation stats
    gen_avg_fitness = []
    gen_std_fitness = []
    gen_best_per_gen = []
    best_per_gen_ever = []

    # Stagnation tracking
    best_fitness = -float('inf')

    stagnation_counter = 0
    current_mutation_rate = BASE_MUTATION_RATE

    csv_path = Path(RUN_DIR) / "generation_log.csv"
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "generation", "ind_idx",
            "fitness", "base_reward", "distance", "speed", "energy", "jerk_sum",
        ])

    best_weights = None

    try:
        for gen in range(NUM_GENERATIONS):
            start = time.time()
            for i in range(CMA_ITERS):
                for genotype in population:
                    genotype.cma.step(i)
                    print("Stepped!")
                for genotype in population:
                    genotype.update_weights(genotype.cma.m, reconstruct_weights=True)
                    print("Updated!")
            end = time.time()
            print(f'ControllerEvo took {end - start} seconds with {CMA_ITERS} iterations')

            start = time.time()
            # ---- gather arguments for all individuals ----
            structures = [g.structure for g in population]
            weights = [get_weights(g.brain, flatten=True) for g in population]

            # ---- run them in parallel ----
            fits_vals = list(GLOBAL_EXEC.map(calc_fitness, structures, weights))
            fits, base_rw_vals, distance_vals, speed_vals, energy_vals, jerk_vals, fell_vals = map(
                np.array, zip(*fits_vals)
            )

            with csv_path.open("a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for idx, (fit, br, d, s, eg, j, f) in enumerate(zip(
                        fits, base_rw_vals, distance_vals,
                        speed_vals, energy_vals, jerk_vals, fell_vals)):
                    writer.writerow([fit, br, d, s, eg, j, int(f)])


            print(fits)
            end = time.time()
            print(f'Fitness Evaluation took {end - start} seconds')

            gen_mean = float(np.mean(fits))
            gen_std = float(np.std(fits))
            gen_best = float(np.max(fits))

            best_idx = int(np.argmax(fits))
            best_structure = population[best_idx].structure
            best_robot = population[best_idx]

            if gen_best > best_fitness:
                best_genotype = population[best_idx]

            plt.figure(figsize=(3, 3))
            plt.imshow(best_structure, cmap="viridis", vmin=0, vmax=4)
            plt.axis("off")
            plt.title(f"Gen {gen} best")
            struct_dir = Path(RUN_DIR) / "structures"
            struct_dir.mkdir(exist_ok=True)
            plt.savefig(struct_dir / f"gen_{gen:03d}.png", bbox_inches="tight")
            plt.close()

            if gen % 5 == 0:
                gif_dir = Path(RUN_DIR) / "gifs"
                gif_dir.mkdir(exist_ok=True)
                gif_path = gif_dir / f"gen_{gen:03d}.gif"

                # run one episode & capture frames
                frames = []
                g = population[best_idx]
                env = gym.make(
                    SCENARIO, max_episode_steps=STEPS,
                    body=g.structure,
                    connections=get_full_connectivity(g.structure)
                )
                sim = env.sim
                viewer = EvoViewer(sim)
                viewer.track_objects("robot")
                obs, _ = env.reset()
                pad = np.zeros(MAX_BRAIN_INPUT_SIZE, dtype=np.float32)

                for _ in range(STEPS):
                    pad[:len(obs)] = obs
                    action = g.act(pad)
                    viewer.render("screen")  # draw on buffer
                    frames.append(viewer.render("rgb_array"))
                    obs, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break

                viewer.close();
                env.close()
                #imageio.mimsave(gif_path, frames, fps=30)
                GLOBAL_EXEC.submit(imageio.mimsave, gif_path, frames, fps=15)

            gen_avg_fitness.append(gen_mean)
            gen_std_fitness.append(gen_std)
            gen_best_per_gen.append(gen_best)

            # update all‐time best
            best_all_time = max(best_all_time, gen_best)
            best_per_gen_ever.append(best_all_time)

            #print(f"Gen {gen:03d} | Best: {gen_best:.3f} | Mean: {gen_best:.3f} | MutRate: {current_mutation_rate:.3f}")
            log(f"Gen {gen:2d} | Best: {gen_best:.3f} | Avg: {gen_mean:.3f} ±{gen_std:.3f} | All‐Time Best: {best_all_time:.3f} | MutRate: {current_mutation_rate:.3f}")

            # === Check stagnation ===
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_genotype = population[best_idx]
                stagnation_counter = 0
                current_mutation_rate = BASE_MUTATION_RATE
            else:
                stagnation_counter += 1

            # update all‐time best
            best_fitness = max(best_fitness, gen_best)
            best_per_gen_ever.append(best_fitness)

            if OUTPUT_ROBOT_GIFS:
                capture_simulation_frame(best_robot, gen, SCENARIO, STEPS, CONTROLLER)

            if OUTPUT_POPULATION:
                capture_population_frames(population, gen, SCENARIO, STEPS, CONTROLLER)

            #print(f"Gen {gen:03d} | Best: {gen_best:.3f} | Mean: {gen_best:.3f} | MutRate: {current_mutation_rate:.3f}")
            log(f"Gen {gen:2d} | Best: {gen_best:.3f} | Avg: {gen_mean:.3f} ±{gen_std:.3f} | All‐Time Best: {best_fitness:.3f} | MutRate: {current_mutation_rate:.3f}")

            # === Handle stagnation ===
            if stagnation_counter >= STAGNATION_LIMIT:
                log(f"> Stagnation reached {STAGNATION_LIMIT} generations: Increasing mutation rate by 0.05 to {current_mutation_rate:.3f}")
                # Increase mutation rate
                current_mutation_rate = min(1.0, current_mutation_rate + MUTATION_RATE_INCREASE)

                stagnation_counter = 0

            start = time.time()
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
                child1, child2 = apply_crossover(parent1.structure, parent2.structure)

                # Mutation
                child1_struct = apply_mutation(child1, current_mutation_rate)
                child2_struct = apply_mutation(child2, current_mutation_rate)

                new_child1 = Genotype(structure=child1_struct, weights=get_weights(parent1.brain, flatten=True))
                new_population.append(new_child1)

                if len(new_population) < POPULATION_SIZE:
                    new_child2 = Genotype(structure=child2_struct, weights=get_weights(parent2.brain, flatten=True))
                    new_population.append(new_child2)

            population = new_population
            end = time.time()
            #print(f'StructureEvo took {end - start} seconds')

    except KeyboardInterrupt:
        pass

    save_weights(best_genotype.brain, RUN_DIR+"/best_weights.pth")

    gens = np.arange(len(gen_avg_fitness))
    # Plot & save
    plt.figure()
    plt.errorbar(
        gens,
        gen_avg_fitness,
        yerr=gen_std_fitness,
        label="mean ± std"
    )
    plt.plot(
        gens,
        gen_best_per_gen,
        label="generation best"
    )
    plt.plot(
        gens,
        best_per_gen_ever,
        label="all‐time best"
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()

    plot_path = os.path.join(RUN_DIR, "fitness_plot.png")
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    prof = cProfile.Profile()
    prof.enable()
    main()
    prof.disable()
    pstats.Stats(prof).sort_stats("cumtime").print_stats(20)