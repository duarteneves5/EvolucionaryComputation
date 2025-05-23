import multiprocessing

import numpy as np
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity

from neural_controller import *
from concurrent.futures import ProcessPoolExecutor
import os
import time
import csv
from pathlib import Path
from datetime import datetime
import imageio
import secrets
import itertools

from utils import (
    plot_fitness_over_generations,
    animate_ackley_optimization,
)


NUM_WORKERS = os.cpu_count()
STEPS = 1500
#SCENARIO = 'DownStepper-v0'
SCENARIO = 'ObstacleTraverser-v0'


robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])


# ---- VISUALIZATION ----
def visualize_policy(weights, reconstruct_weights=False):
    set_weights(brain, weights, reconstruct_weights=reconstruct_weights)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    for t in range(STEPS):
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten()  # Get action
        viewer.render('screen')
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()


def create_gif_of_best_policy(weights, filename="best_policy.gif", fps=30):
    """
    Runs one episode using the best weights, captures frames, and saves to a GIF.
    """
    set_weights(brain, weights, reconstruct_weights=True)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]

    frames = []
    for t in range(STEPS):
        # Render in "rgb_array" mode to get raw frame data
        frame = viewer.render('rgb_array')
        frames.append(frame)

        # Compute action
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            action = brain(state_tensor).detach().numpy().flatten()

        # Step
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    viewer.close()
    env.close()

    # Save frames as a GIF
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved to {filename}")

robot_structure = np.array([
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size, init_params=True)

CURRICULUM_STEPS = STEPS * 0.2
def ramp_factor(generation: int) -> float:
    """
    Linear schedule in [0, 1].
    gen 0 .. CURRICULUM_STEPS :  ramps up
    beyond                 :  stays at 1.0
    """
    return min(1.0, generation / CURRICULUM_STEPS)


def evaluate_fitness_curriculum(weights, generation, return_components=False, view=False):
    """
    Calls evaluate_fitness with
    curriculum-scaled survival / fall weights
    """
    r = ramp_factor(generation)

    return evaluate_fitness(
        weights,
        w_distance=0.5,
        w_efficiency=0.25,
        w_speed=0.25,
        # scaled terms
        w_survival=r * 50,
        w_fall    =r * 100,
        return_components=return_components,
        view=view,
    )


# ---- FITNESS FUNCTION ----
def evaluate_fitness(
        weights,
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

    set_weights(brain, weights, reconstruct_weights=True)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim

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
        #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Convert to tensor
        with torch.no_grad():
            action = brain(state_tensor).detach().numpy().flatten()  # Get action

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

    if SCENARIO == "ObstacleTraverser-v0":
        dist_norm = np.clip(progress_reward / DIST_REF, 0.0, 1.0)
    elif SCENARIO == "DownStepper-v0":
        dist_norm = np.clip(distance / DIST_REF, 0.0, 1.0)

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
        mu_eff = 1.0 / np.sum(self.weights ** 2)
        print("μ_eff =", mu_eff)

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


# ---- CMA-ES SEARCH ALGORITHM ----

def main():
    best_fitness = -np.inf
    best_weights = None

    POPULATION_SIZE = int(4 + 3*np.log(len(get_weights(brain, flatten=True))))
    print(f"Population Size: {POPULATION_SIZE}")
    mu = POPULATION_SIZE // 2
    NUM_GENERATIONS = POPULATION_SIZE * 10
    NUM_GENERATIONS = 100
    print(f"Number of Generations: {NUM_GENERATIONS}")

    for _ in range(5):
        SEED = secrets.randbelow(1_000_000_000)
        print(f"SEEDING WITH {SEED}")
        np.random.seed(SEED)
        random.seed(SEED)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = Path("results") / f"EC_{SCENARIO}_{POPULATION_SIZE}pop_{STEPS}step_{SEED}"
        results_root.mkdir(parents=True, exist_ok=True)

        # CSV file for generation statistics
        csv_path = results_root / "generation_log.csv"
        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([  # header
            "generation",
            "gen_best_fitness",
            "global_best_fitness",
            "spread",
            "mean_fitness",
            # fitness components:
            "distance",
            "efficiency",
            "speed",
            "survival",
            "fell_over",
            "total_energy"
        ])

        optimizer = CMAESOptimizer(brain, POPULATION_SIZE, evaluate_fitness_curriculum)

        try:
            for generation in range(NUM_GENERATIONS):
                start = time.time()
                pop, fitnesses = optimizer.step(generation)
                gen_best = optimizer.current_gen_best_fitness
                best_idx = fitnesses.index(gen_best)

                if gen_best > best_fitness:
                    best_fitness = gen_best
                    best_weights = pop[best_idx]

                fitness, distance, efficiency, speed, survival, fell_over, total_energy = evaluate_fitness_curriculum(pop[best_idx], generation, return_components=True)

                mean_fit = optimizer.mean_fitness_history[-1]
                csv_writer.writerow([
                    generation,
                    f"{gen_best:.6f}",
                    f"{best_fitness:.6f}",
                    f"{optimizer.spread:.4f}",
                    f"{mean_fit:.6f}",
                    f"{distance:.6f}",
                    f"{efficiency:.6f}",
                    f"{speed:.6f}",
                    f"{survival:.6f}",
                    f"{fell_over:.6f}",
                    f"{total_energy:.6f}"
                ])
                csv_file.flush()

                if generation % 10 == 0:
                    gif_path = results_root / f"gen_{generation:03d}_best.gif"
                    create_gif_of_best_policy(best_weights, filename=str(gif_path))

                end = time.time()
                length = end - start
                print(f"[GEN {generation + 1}/{NUM_GENERATIONS}] Best Fitness: {optimizer.current_gen_best_fitness} / Took {length:.2f} seconds / SPREAD: {optimizer.spread}")


        except KeyboardInterrupt:
            set_weights(brain, best_weights, reconstruct_weights=True)
            save_weights(brain, filename=results_root / "best_weights.pth")

        finally:
            csv_file.close()
            optimizer.executor.shutdown(wait=False)

        path_of_means = np.array(optimizer.path_of_means)
        all_samples = np.array(optimizer.all_samples)

        # Set the best weights found
        set_weights(brain, best_weights, reconstruct_weights=True)
        print(f"Best Fitness: {best_fitness}")


        plot_fitness_over_generations(optimizer.best_fitness_history, optimizer.mean_fitness_history, filename=str(results_root / "best_fitness_over_generations.png"))
        animate_ackley_optimization(all_samples, path_of_means, filename=str(results_root / "ackley_animation.gif"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
