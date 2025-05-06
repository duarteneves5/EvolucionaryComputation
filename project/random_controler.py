import csv
import multiprocessing
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
from concurrent.futures import ProcessPoolExecutor
import imageio
from utils import (
    plot_fitness_over_generations
)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_GENERATIONS = 100
STEPS = 1500
SCENARIO = 'DownStepper-v0'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])


connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size)


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
        action = brain(state_tensor).detach().numpy().flatten() # Get action
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
    set_weights(brain, weights)
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


def ackley(x, y):
    a, b, c = 20, 0.2, 2*np.pi
    part1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    part2 = -np.exp((np.cos(c*x) + np.cos(c*y)) / 2)
    return part1 + part2 + a + np.e


def animate_ackley_optimization(all_samples,
                                path_of_means,
                                filename="ackley_optimization.gif"):

    # -------------------------------------------------------------
    # 1.  Pull out x/y columns from the arrays you stored earlier
    # -------------------------------------------------------------
    sample_x = all_samples[:, 1]
    sample_y = all_samples[:, 2]
    path_x   = path_of_means[:, 0]
    path_y   = path_of_means[:, 1]

    # -------------------------------------------------------------
    # 2.  Define sensible plotting bounds (+10 % padding)
    # -------------------------------------------------------------
    x_min, x_max = sample_x.min(), sample_x.max()
    y_min, y_max = sample_y.min(), sample_y.max()
    pad_x = 0.1 * (x_max - x_min)
    pad_y = 0.1 * (y_max - y_min)
    x = np.linspace(x_min - pad_x, x_max + pad_x, 300)
    y = np.linspace(y_min - pad_y, y_max + pad_y, 300)
    X, Y = np.meshgrid(x, y)

    # -------------------------------------------------------------
    # 3.  ***Actually evaluate Ackley here!***
    # -------------------------------------------------------------
    Z = ackley(X, Y)

    # -------------------------------------------------------------
    # 4.  Build the figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Ackley landscape and random‑search trace")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # filled contours for the landscape
    ax.contourf(X, Y, Z, levels=30, cmap='viridis')

    # empty artists that we’ll update every frame
    scat = ax.scatter([], [], c=[], cmap='plasma', s=20, alpha=0.7,
                      vmin=0, vmax=path_of_means.shape[0]-1)
    line, = ax.plot([], [], 'w-', lw=2)

    # -------------------------------------------------------------
    # 5.  Frame‑update function
    # -------------------------------------------------------------
    def update(gen_idx):
        mask = all_samples[:, 0] == gen_idx
        current = all_samples[mask]

        scat.set_offsets(current[:, 1:3])
        scat.set_array(np.full(current.shape[0], gen_idx))

        line.set_data(path_x[:gen_idx+1], path_y[:gen_idx+1])
        ax.set_title(f"Generation {gen_idx}")
        return scat, line

    # -------------------------------------------------------------
    # 6.  Run the animation
    # -------------------------------------------------------------
    n_gens = int(all_samples[:, 0].max()) + 1
    ani = FuncAnimation(fig, update, frames=range(n_gens),
                        interval=120, blit=True)
    ani.save(filename, writer='pillow', fps=15, dpi=90)
    plt.close(fig)
    print(f"Animation saved to {filename}")



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
    set_weights(brain, weights)  # Load weights into the network
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


def main():
    best_fitness = -np.inf
    best_weights = None
    POPULATION_SIZE = 50

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path("results") / f"RC_{SCENARIO}_{POPULATION_SIZE}pop_{STEPS}step_{timestamp}"
    results_root.mkdir(parents=True, exist_ok=True)

    # CSV file for generation statistics
    csv_path = results_root / "generation_log.csv"
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([  # header
        "generation",
        "gen_best_fitness",
        "global_best_fitness",
        "mean_fitness",
        # fitness components:
        "distance",
        "efficiency",
        "survival",
        "speed"
    ])

    path_of_means = []
    all_samples = []
    best_fitness_history = []
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for generation in range(NUM_GENERATIONS):
                start = time.time()
                population = [[np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(POPULATION_SIZE)]
                fitnesses = list(executor.map(evaluate_fitness, population))
                fitness_mean = np.mean(fitnesses)

                xs, ys = [], []
                for ind in population:
                    flat = np.concatenate([p.flatten() for p in ind])
                    x, y = flat[0], flat[1]  # 2‑D projection
                    all_samples.append([generation, x, y])
                    xs.append(x)
                    ys.append(y)
                path_of_means.append([np.mean(xs), np.mean(ys)])

                best_fitness_idx = np.argsort(fitnesses)[-1]

                if fitnesses[best_fitness_idx] > best_fitness:
                    best_fitness = fitnesses[best_fitness_idx]
                    best_weights = population[best_fitness_idx]

                fitness, distance, efficiency, survival, speed = evaluate_fitness(population[best_fitness_idx], return_components=True)
                best_fitness_history.append(fitness)
                csv_writer.writerow([
                    generation,
                    f"{fitness:.6f}",
                    f"{best_fitness:.6f}",
                    f"{fitness_mean:.6f}",
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
                print(f"[GEN {generation + 1}/{NUM_GENERATIONS}] Best Fitness: {fitnesses[best_fitness_idx]} / Took {length:.2f} seconds")


    except KeyboardInterrupt:
        set_weights(brain, best_weights)
        save_weights(brain, filename=results_root / "best_weights.pth")

    finally:
        executor.shutdown()
        csv_file.close()


    path_of_means = np.array(path_of_means)
    all_samples = np.array(all_samples)


    plot_fitness_over_generations(best_fitness_history, path_of_means, filename=str(results_root / "best_fitness_over_generations.png"))
    animate_ackley_optimization(all_samples, path_of_means, filename=str(results_root / "ackley_animation.gif"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()