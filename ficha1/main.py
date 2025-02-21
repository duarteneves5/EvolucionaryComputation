import numpy as np
import random
import math
import matplotlib.pyplot as plt

""" 
=============================================
 1. Define City Model
=============================================
"""
np.random.seed(2025)  # Ensure reproducibility
random.seed(2025)  # Ensure reproducibility

# City dimensions (1 km x 1 km)
CITY_WIDTH, CITY_HEIGHT = 1000, 1000  # in meters

# Number of food carts
N = 10

# Number of customer hotspots
M = 15

# Generate random customer hotspots with demand values
customer_hotspots = np.random.rand(M, 2) * [CITY_WIDTH, CITY_HEIGHT]
customer_demand = np.random.randint(50, 300, size=M)  # Each hotspot has a demand

# Generate random initial positions for food carts
food_carts = np.random.rand(N, 2) * [CITY_WIDTH, CITY_HEIGHT]

""" 
=============================================
 2. Define Objective Function
=============================================
"""
# Function to compute Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Compute total weighted distance from hotspots to nearest food cart
def total_customer_distance(food_carts, customer_hotspots, customer_demand):
    total_distance = 0
    for i in range(M):
        # Find the nearest food cart to the customer hotspot
        min_distance = min(euclidean_distance(customer_hotspots[i], food_carts[j]) for j in range(N))
        total_distance += customer_demand[i] * min_distance  # Weighted by demand
    return total_distance

""" 
=============================================
 3. Implement Simulated Annealing
=============================================
"""
# Simulated Annealing implementation
def simulated_annealing(food_carts, customer_hotspots, customer_demand, T0, alpha, iterations=2000):
    T = T0
    current_solution = np.copy(food_carts)
    current_cost = total_customer_distance(current_solution, customer_hotspots, customer_demand)
    best_solution = np.copy(food_carts)
    best_cost = total_customer_distance(current_solution, customer_hotspots, customer_demand)

    # TODO: YOUR CODE HERE
    for i in range(iterations):
        candidate_solution = np.copy(current_solution)
        # Select a random food cart to move
        idx = np.random.randint(0, N)
        # Apply a random displacement (scale can be adjusted)
        displacement = np.random.normal(0, 20, 2)
        candidate_solution[idx] += displacement
        # Ensure new position is within city bounds
        candidate_solution[idx, 0] = np.clip(candidate_solution[idx, 0], 0, CITY_WIDTH)
        candidate_solution[idx, 1] = np.clip(candidate_solution[idx, 1], 0, CITY_HEIGHT)

        candidate_cost = total_customer_distance(candidate_solution, customer_hotspots, customer_demand)
        delta = candidate_cost - current_cost

        # Accept new solution if it improves the cost
        if delta < 0:
            current_solution = candidate_solution
            current_cost = candidate_cost
            if candidate_cost < best_cost:
                best_solution = candidate_solution
                best_cost = candidate_cost
        else:
            # Accept with a probability exp(-delta/T)
            if np.random.rand() < np.exp(-delta / T):
                current_solution = candidate_solution
                current_cost = candidate_cost

        # Cool down
        T *= alpha
        # Optional: Reset temperature if it gets too low (or break)
        if T < 1e-5:
            T = T0

    return best_solution, best_cost

""" 
=============================================
 4. Run Simulated Annealing and Visualize Results
=============================================
"""
# Function to plot city layout
def plot_city(food_carts, customer_hotspots, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(food_carts[:, 0], food_carts[:, 1], c='blue', marker='o', label='Food Carts', s=100)
    plt.scatter(customer_hotspots[:, 0], customer_hotspots[:, 1], c='red', marker='x', label='Customer Hotspots', s=80)

    # Draw lines from hotspots to the nearest food cart
    for i in range(M):
        nearest_cart = min(food_carts, key=lambda c: euclidean_distance(customer_hotspots[i], c))
        plt.plot([customer_hotspots[i][0], nearest_cart[0]],
                 [customer_hotspots[i][1], nearest_cart[1]], 'gray', linestyle='dotted')

    plt.xlim(0, CITY_WIDTH)
    plt.ylim(0, CITY_HEIGHT)
    plt.xlabel("City Width (m)")
    plt.ylabel("City Height (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot initial city layout
plot_city(food_carts, customer_hotspots, "Initial Food Cart Locations")
'''
# Run SA optimization
optimized_food_carts, optimized_distance = simulated_annealing(food_carts, customer_hotspots, customer_demand, T0=1000, alpha=0.95) #TODO T0 and alpha implemented

# Plot optimized city layout
plot_city(optimized_food_carts, customer_hotspots, "Optimized Food Cart Locations (SA)")

# Return final optimized total distance
optimized_distance
'''

""" 
=============================================
 5. Implement Random Search
=============================================
"""
def random_search(customer_hotspots, customer_demand, num_iterations=5000):
    best_solution = np.random.rand(N, 2) * [CITY_WIDTH, CITY_HEIGHT]
    best_cost = total_customer_distance(best_solution, customer_hotspots, customer_demand)

    # TODO: YOUR CODE HERE
    for i in range(num_iterations):
        candidate_solution = np.random.rand(N, 2) * [CITY_WIDTH, CITY_HEIGHT]  # put cart in random position in the city
        candidate_cost = total_customer_distance(candidate_solution, customer_hotspots, customer_demand)  # sees the quality of position
        # if the new random position is better than the best one at the moment makes the new position the new best
        if candidate_cost < best_cost:
            best_solution = candidate_solution
            best_cost = candidate_cost

    return best_solution, best_cost

""" 
=============================================
 6. Implement Hill Climber
=============================================
"""
def hill_climber(food_carts, customer_hotspots, customer_demand, iterations=1000):
    current_solution = np.copy(food_carts)
    best_solution = np.copy(food_carts)
    #current cost
    best_cost = total_customer_distance(current_solution, customer_hotspots, customer_demand)

    # TODO: YOUR CODE HERE
    for i in range(iterations):
        candidate_solution = np.copy(current_solution)

        # choose a random cart
        random_cart = np.random.randint(0,N)

        # apply a displacement each iteration
        random_displacement = np.random.uniform(-40,40,2)  # ATTENTION this is the most import aspect to get better results
        candidate_solution[random_cart] += random_displacement

        # Ensure candidate remains within bounds
        candidate_solution[random_cart, 0] = np.clip(candidate_solution[random_cart, 0], 0, CITY_WIDTH)
        candidate_solution[random_cart, 1] = np.clip(candidate_solution[random_cart, 1], 0, CITY_HEIGHT)

        # calculate the cost of the new place
        candidate_cost = total_customer_distance(candidate_solution, customer_hotspots, customer_demand)

        # accept if the candidate improves its solution
        if candidate_cost < best_cost:

            # updates the current solution to
            current_solution = candidate_solution
            current_cost = candidate_cost

            # updates the best of all
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

    return best_solution, best_cost



def hill_climber_with_evolution(food_carts, customer_hotspots, customer_demand, iterations=1000, snapshot_interval=100):
    current_solution = np.copy(food_carts)
    best_solution = np.copy(food_carts)
    best_cost = total_customer_distance(current_solution, customer_hotspots, customer_demand)

    # Lists to record snapshots and iteration markers
    snapshots = [np.copy(current_solution)]
    iteration_marks = [0]

    for i in range(1, iterations + 1):
        candidate_solution = np.copy(current_solution)
        random_cart = np.random.randint(0, N)
        random_displacement = np.random.uniform(-30, 30, 2)
        candidate_solution[random_cart] += random_displacement

        # Ensure candidate remains within bounds
        candidate_solution[random_cart, 0] = np.clip(candidate_solution[random_cart, 0], 0, CITY_WIDTH)
        candidate_solution[random_cart, 1] = np.clip(candidate_solution[random_cart, 1], 0, CITY_HEIGHT)


        candidate_cost = total_customer_distance(candidate_solution, customer_hotspots, customer_demand)
        # accept if the candidate improves its solution
        if candidate_cost < best_cost:

            # updates the current solution to
            current_solution = candidate_solution
            current_cost = candidate_cost

            # updates the best of all
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        # Record a snapshot every 'snapshot_interval' iterations
        if i % snapshot_interval == 0:
            snapshots.append(np.copy(current_solution))
            iteration_marks.append(i)

    return best_solution, best_cost, snapshots, iteration_marks

"""
# Run the hill climber with evolution tracking
hill_climb_best_solution, hill_climb_best_cost, snapshots, iteration_marks = hill_climber_with_evolution(
    food_carts, customer_hotspots, customer_demand, iterations=1000, snapshot_interval=100
)
for idx, snapshot in enumerate(snapshots):
    plt.figure(figsize=(8, 8))
    # Plot customer hotspots
    plt.scatter(customer_hotspots[:, 0], customer_hotspots[:, 1], c='red', marker='x', s=80, label='Customer Hotspots')
    # Plot the food carts positions from this snapshot
    plt.scatter(snapshot[:, 0], snapshot[:, 1], c='blue', marker='o', s=100, label='Food Carts')
    plt.xlim(0, CITY_WIDTH)
    plt.ylim(0, CITY_HEIGHT)
    plt.xlabel("City Width (m)")
    plt.ylabel("City Height (m)")
    plt.title(f"Hill Climber Evolution - Iteration {iteration_marks[idx]}")
    plt.legend()
    plt.grid(True)
    plt.show()
"""
""" 
=============================================
 7. Run All Approaches and Compare Results
=============================================
"""
# Function to plot all approaches in a single figure
def plot_all_methods(random_solution, hill_solution, sim_annealing_solution, customer_hotspots):
    plt.figure(figsize=(8, 8))

    # Plot customer hotspots
    plt.scatter(customer_hotspots[:, 0], customer_hotspots[:, 1], c='red', marker='x', label='Customer Hotspots', s=80)

    # Plot Random Search locations
    plt.scatter(random_solution[:, 0], random_solution[:, 1], c='blue', marker='o', label='Random Search', s=100)

    # Plot Hill Climber locations
    plt.scatter(hill_solution[:, 0], hill_solution[:, 1], c='green', marker='s', label='Hill Climber', s=100)

    # Plot Simulated Annealing locations
    plt.scatter(sim_annealing_solution[:, 0], sim_annealing_solution[:, 1], c='purple', marker='D',
                label='Simulated Annealing', s=100)

    plt.xlim(0, CITY_WIDTH)
    plt.ylim(0, CITY_HEIGHT)
    plt.xlabel("City Width (m)")
    plt.ylabel("City Height (m)")
    plt.title("Comparison of Optimization Approaches")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run Random Search
random_solution, random_cost = random_search(customer_hotspots, customer_demand)

# Run Hill Climber
hill_solution, hill_cost = hill_climber(food_carts, customer_hotspots, customer_demand)

# Run Simulated Annealing
sim_annealing_solution, sim_annealing_cost = simulated_annealing(food_carts, customer_hotspots, customer_demand, T0=300, alpha=0.95) #TODO T0 and alpha implemented

# Print comparison results
print("Comparison of Optimization Methods:")
print(f"Random Search Distance: {random_cost:.2f}")
print(f"Hill Climber Distance: {hill_cost:.2f}")
print(f"Simulated Annealing Distance: {sim_annealing_cost:.2f}")

# Plot all methods together
plot_all_methods(random_solution, hill_solution, sim_annealing_solution, customer_hotspots)

# Plot results
plot_city(hill_solution, customer_hotspots, "Hill Climber - Optimized Food Cart Locations")
plot_city(random_solution, customer_hotspots, "Random Search - Optimized Food Cart Locations")
plot_city(sim_annealing_solution, customer_hotspots, "Simulated Annealing - Optimized Food Cart Locations")
