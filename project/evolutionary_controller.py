import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
from concurrent.futures import ProcessPoolExecutor
import os
import time


NUM_WORKERS = os.cpu_count()
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size, init_params=True)


# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, view=False):
    set_weights(brain, weights, reconstruct_weights=True)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    t_reward = 0
    for t in range(STEPS):
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten()  # Get action
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()
    return t_reward


# ---- CMA-ES SEARCH ALGORITHM ----
best_fitness = -np.inf
best_weights = None

SPREAD = 0.1    # [0.01, 1.0]; initial step size that sets the scale of your random perturbations
m = get_weights(brain, flatten=True) # mean vector
C = SPREAD**2 * np.eye(len(m)) # initial covariance
POPULATION_SIZE = int(4 + 3*np.log(len(m)))
print(f"Population Size: {POPULATION_SIZE}")
mu = POPULATION_SIZE // 2
NUM_GENERATIONS = POPULATION_SIZE * 10
print(f"Nuber of Generations: {NUM_GENERATIONS}")
def sample_normal(_):
    return np.random.multivariate_normal(m, C)


with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for generation in range(NUM_GENERATIONS):
        start = time.time()
        population = np.random.multivariate_normal(mean=m, cov=C, size=POPULATION_SIZE)

        fitnesses = list(executor.map(evaluate_fitness, population))

        sorted_indices = np.argsort(fitnesses)[::-1]
        best_indices = sorted_indices[:mu]

        new_m = np.mean([population[i] for i in best_indices], axis=0)

        new_C = np.zeros_like(C)
        for i in best_indices:
            diff = population[i] - new_m
            new_C += np.outer(diff, diff)
        new_C /= mu

        alpha = 0.1
        C = (1 - alpha) * C + alpha * new_C

        m = new_m

        if fitnesses[best_indices[0]] > best_fitness:
            best_fitness = fitnesses[sorted_indices[0]]
            best_weights = population[sorted_indices[0]]

        end = time.time()
        length = end - start
        print(f"[GEN {generation + 1}/{NUM_GENERATIONS}] Best Fitness: {fitnesses[best_indices[0]]} / Took {length:.2f} seconds")

# Set the best weights found
set_weights(brain, best_weights, reconstruct_weights=True)
print(f"Best Fitness: {best_fitness}")


# ---- VISUALIZATION ----
def visualize_policy(weights):
    set_weights(brain, weights)  # Load weights into the network
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


i = 0
while i < 10:
    visualize_policy(get_weights(brain))
    i += 1