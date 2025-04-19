import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
from concurrent.futures import ProcessPoolExecutor
import os
import time
import matplotlib.pyplot as plt
import imageio


NUM_WORKERS = os.cpu_count()
STEPS = 500
SCENARIO = 'DownStepper-v0'
#SCENARIO = 'ObstacleTraverser-v0'
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
def evaluate_fitness(
        weights,
        view=False,
        w_distance=1,
        w_energy=0.0,
        w_speed=0.1
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
    n_act = sim.get_dim_action_space('robot')
    prev_act = np.zeros(n_act)
    energy = .0

    for t in range(STEPS):
        # Update actuation before stepping
        #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Convert to tensor
        with torch.no_grad():
            action = brain(state_tensor).detach().numpy().flatten()  # Get action

            energy += np.sum(np.abs(action - prev_act))
            prev_act = action.copy()

            if view:
                viewer.render('screen')
            state, reward, terminated, truncated, info = env.step(action)
            #t_reward += reward
            if terminated or truncated:
                break


    tf = sim.get_time()
    end_com = sim.object_pos_at_time(tf, 'robot').mean(axis=1)
    distance = end_com[0] - start_com[0] # only x, maybe account for y as well
    speed = distance / max(1, tf)

    if view:
        viewer.close()

    env.close()

    t_reward = w_distance*distance - w_energy*energy + w_speed*speed
    #print(f"reward: {t_reward} - distance: {distance} - average speed: {average_speed} - energy: {energy}")
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
NUM_GENERATIONS = 20
print(f"Number of Generations: {NUM_GENERATIONS}")

best_fitness_history = []
mean_fitness_history = []

path_of_means = []
all_samples = []

def sample_normal(_):
    return np.random.multivariate_normal(m, C)

try:
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for generation in range(NUM_GENERATIONS):
            start = time.time()
            population = np.random.multivariate_normal(mean=m, cov=C, size=POPULATION_SIZE)

            fitnesses = list(executor.map(evaluate_fitness, population))

            sorted_indices = np.argsort(fitnesses)[::-1]
            best_indices = sorted_indices[:mu]

            new_m = np.mean([population[i] for i in best_indices], axis=0)

            #new_C = np.zeros_like(C)
            #for i in best_indices:
            #    diff = population[i] - new_m
            #    new_C += np.outer(diff, diff)
            #new_C /= mu
            diffs = population[best_indices] - new_m
            new_C = np.einsum('ni,nj->ij', diffs, diffs) / mu

            alpha = 0.1
            C = (1 - alpha) * C + alpha * new_C

            m = new_m

            path_of_means.append(m.copy())
            for i in range(POPULATION_SIZE):
                all_samples.append((generation, population[i, 0], population[i, 1]))

            current_gen_best_fitness = fitnesses[best_indices[0]]
            if current_gen_best_fitness > best_fitness:
                best_fitness = current_gen_best_fitness
                best_weights = population[sorted_indices[0]]

            best_fitness_history.append(current_gen_best_fitness)
            mean_fitness_history.append(np.mean(fitnesses))

            end = time.time()
            length = end - start
            print(f"[GEN {generation + 1}/{NUM_GENERATIONS}] Best Fitness: {fitnesses[best_indices[0]]} / Took {length:.2f} seconds")

except KeyboardInterrupt:
    pass

path_of_means = np.array(path_of_means)
all_samples = np.array(all_samples)

# Set the best weights found
set_weights(brain, best_weights, reconstruct_weights=True)
print(f"Best Fitness: {best_fitness}")


# ---- VISUALIZATION: PLOT FITNESS OVER GENERATIONS ----
plt.plot(best_fitness_history, label='Best Fitness')
plt.plot(mean_fitness_history, label='Mean Fitness')
plt.title("Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()


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


def ackley(x, y):
    """
    Ackley function in 2D.
    Global minimum = 0 at (0,0).
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    # Calculate each part
    part1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    part2 = -np.exp((np.cos(c*x) + np.cos(c*y)) / 2)
    return part1 + part2 + a + np.e


X_MIN, X_MAX = -0.4, 0.4
Y_MIN, Y_MAX = -0.4, 0.4
res = 1000
x_vals = np.linspace(X_MIN, X_MAX, res)
y_vals = np.linspace(Y_MIN, Y_MAX, res)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute Ackley values on the grid
Z = ackley(X, Y)


# Plot contours
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=30)
plt.colorbar(label='Ackley value')

# Plot CMA-ES samples
for g in range(int(all_samples[:, 0].max()) + 1):
    # All samples in generation g
    points_g = all_samples[all_samples[:, 0] == g]
    plt.scatter(points_g[:, 1], points_g[:, 2], s=10, alpha=0.5)

# Plot path of the CMA-ES mean
plt.plot(path_of_means[:, 0], path_of_means[:, 1], '-s', label='CMA-ES mean path', linewidth=2)

plt.title("CMA-ES on Ackley Function (2D)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# ---- CREATE A GIF OF THE BEST POLICY ----
def create_gif_of_best_policy(weights, filename="best_policy.gif", fps=30):
    """
    Runs one episode using the best weights, captures frames, and saves to a GIF.
    """
    set_weights(brain, weights, reconstruct_weights=True)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env)
    viewer.track_objects('robot')
    state, _ = env.reset()

    frames = []
    for t in range(STEPS):
        # Render in "rgb_array" mode to get raw frame data
        frame = viewer.render(mode="rgb_array")
        frames.append(frame)

        # Compute action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()

        # Step
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()

    # Save frames as a GIF
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved to {filename}")

#create_gif_of_best_policy(best_weights, filename="best_policy.gif", fps=30)

i = 0
while i < 10:
    visualize_policy(get_weights(brain))
    i += 1

