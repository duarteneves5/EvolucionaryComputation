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
import matplotlib.pyplot as plt
import imageio
import imageio.v3 as iio
import secrets

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


connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size, init_params=True)


# ---- FITNESS FUNCTION ----
def evaluate_fitness(
        weights,
        view=False,
        w_distance=1,
        w_efficiency=0.2,
        w_jerk=0.0,
        w_upright=0.0,
        w_speed=0.3,
        w_survival=0.1,
        return_components=False
):
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

        # posture: robot tilt w.r.t. x‐axis
        angle = sim.object_orientation_at_time(sim.get_time(), 'robot')
        upright_accum += abs(np.cos(angle))  # 1.0 if perfectly aligned :contentReference[oaicite:0]{index=0}

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

    # normalized energy per meter (avoid div0)
    efficiency = distance / (total_energy + 1e-6) * 1e6
    #print(efficiency)

    #if total_actuation < STEPS * output_size * 0.01:
    #    actuation_penalty = -100.0
    #else:
    #    actuation_penalty = 0.0

    # average upright score
    upright = upright_accum / survived_steps

    # survival bonus
    survival_bonus = (survived_steps / STEPS)

    #survival_bonus = (survived_steps / STEPS) * w_survival
    speed = distance / max(1, tf)

    if view:
        viewer.close()

    env.close()

    fitness = (
        w_distance * distance
        + w_efficiency * efficiency
        - w_jerk * jerk_sum
        + w_upright * upright
        + w_survival * survival_bonus
        + w_speed * speed
        #+ actuation_penalty
    )
    #t_reward = w_distance*distance - w_energy*energy + w_speed*speed + survival_bonus
    #print(f"reward: {t_reward} - distance: {distance} - average speed: {average_speed} - energy: {energy}")
    if return_components:
        return fitness, w_distance * distance, w_efficiency * efficiency, w_survival * survival_bonus, w_speed * speed

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
        self.MIN_SPREAD = 0.01
        self.MAX_SPREAD = 1.0
        self.COOLING_RATE = 0.7
        self.STAGNATION_WINDOW = 5  # generations without improvement
        self.STAGNATION_BOOST = 1.2  # spread multiplier on stagnation
        self.IMPROVEMENT_THRESHOLD = 0.2  # minimum relative improvement
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

        fitnesses = list(self.executor.map(self.fitness_function, pop))

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

    for SCENARIO in ["DownStepper-v0", "ObstacleTraverser-v0"]:
        for _ in range(5):
            SEED = secrets.randbelow(1_000_000_000)
            np.random.seed(SEED)
            random.seed(SEED)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_root = Path("results") / f"EC_{SCENARIO}_{POPULATION_SIZE}pop_{STEPS}step_{timestamp}"
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
                "survival",
                "speed"
            ])

            optimizer = CMAESOptimizer(brain, POPULATION_SIZE, evaluate_fitness)

            try:
                for generation in range(NUM_GENERATIONS):
                    start = time.time()
                    pop, fitnesses = optimizer.step(generation)
                    gen_best = optimizer.current_gen_best_fitness
                    best_idx = fitnesses.index(gen_best)

                    if gen_best > best_fitness:
                        best_fitness = gen_best
                        best_weights = pop[best_idx]

                    fitness, distance, efficiency, survival, speed = evaluate_fitness(pop[best_idx], return_components=True)

                    mean_fit = optimizer.mean_fitness_history[-1]
                    csv_writer.writerow([
                        generation,
                        f"{gen_best:.6f}",
                        f"{best_fitness:.6f}",
                        f"{optimizer.spread:.4f}",
                        f"{mean_fit:.6f}",
                        f"{distance:.6f}",
                        f"{efficiency:.6f}",
                        f"{survival:.6f}",
                        f"{speed:.6f}"
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
