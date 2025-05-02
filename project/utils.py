import numpy as np
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
from fixed_controllers import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---- SIMULATE BEST ROBOT ----
def simulate_best_robot(robot_structure, scenario=None, steps=500, controller = alternating_gait):
    
    connectivity = get_full_connectivity(robot_structure)
    #if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0
    
    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        actuation = controller(action_size,t)

        ob, reward, terminated, truncated, info = env.step(actuation)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return t_reward #(max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance


def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        #if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        t_reward = 0

        frames = []
        for t in range(200):
            actuation = controller(action_size,t)
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid')
        

def plot_fitness_over_generations(
        best_fitness_history,
        mean_fitness_history,
        filename
):
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(mean_fitness_history, label='Mean Fitness')
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


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


def animate_ackley_optimization(
        all_samples,
        path_of_means,
        filename="ackley_optimization.gif"
):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initial setup
    sample_x = all_samples[:, 1]
    sample_y = all_samples[:, 2]
    path_x = path_of_means[:, 0]
    path_y = path_of_means[:, 1]

    # Dynamic bounds calculation
    x_min = min(sample_x.min(), path_x.min())
    x_max = max(sample_x.max(), path_x.max())
    y_min = min(sample_y.min(), path_y.min())
    y_max = max(sample_y.max(), path_y.max())

    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)

    # Create grid
    x = np.linspace(x_min - x_pad, x_max + x_pad, 100)
    y = np.linspace(y_min - y_pad, y_max + y_pad, 100)
    X, Y = np.meshgrid(x, y)
    Z = ackley(X, Y)

    # Initial plot
    ax.contourf(X, Y, Z, levels=15, cmap='viridis')
    scat = ax.scatter([], [], c=[], cmap='plasma', alpha=0.6, s=10)
    line, = ax.plot([], [], 'w-', linewidth=2)

    def update(frame):
        # Update samples up to current frame
        current_samples = all_samples[all_samples[:, 0] <= frame]
        scat.set_offsets(current_samples[:, 1:3])
        scat.set_array(current_samples[:, 0])

        # Update mean path up to current frame
        current_path = path_of_means[:frame + 1]
        line.set_data(current_path[:, 0], current_path[:, 1])

        ax.set_title(f"Generation {frame}")
        return scat, line

    ani = FuncAnimation(fig, update, frames=int(all_samples[-1, 0]),
                        interval=100, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    print(f"Animation saved to {filename}")

    plt.close()
