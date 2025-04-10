# ---------------- Diversity Preservation: Fitness Sharing ----------------
def robot_distance(robot1, robot2):
    """Compute a simple Hamming-like distance between two robot structures."""
    # Assumes both robots have the same shape.
    return np.sum(robot1 != robot2)


def compute_shared_fitnesses(population, raw_fitnesses, sigma_share=5):
    """
    Adjust raw fitness scores using fitness sharing.
    For each individual, compute:
       f_shared(i) = f(i) / (1 + sum_{j != i} sh(d(i,j)))
    where sh(d) = max(0, 1 - d / sigma_share) for d < sigma_share, 0 otherwise.
    """
    n = len(population)
    shared = np.zeros(n)
    for i in range(n):
        sharing_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            d = robot_distance(population[i], population[j])
            sharing_value = max(0, 1 - d / sigma_share)
            sharing_sum += sharing_value
        shared[i] = raw_fitnesses[i] / (1 + sharing_sum)
    return shared


# ------------------ Main Evolutionary Loop (with Diversity Preservation) ------------------
best_fitness = -float('inf')
best_robot = None

# Initial population
population = [create_random_robot() for _ in range(POPULATION_SIZE)]
stagnation_counter = 0
current_mutation_rate = MUTATION_RATE

for gen in range(NUM_GENERATIONS):
    # Evaluate raw fitnesses
    population_fitness = [evaluate_fitness(robot) for robot in population]
    fitness_std = np.std(population_fitness)

    # Compute shared fitnesses for diversity preservation
    shared_fitnesses = compute_shared_fitnesses(population, population_fitness, sigma_share=5)

    # Find best (raw) fitness in the current generation
    best_idx = np.argmax(population_fitness)
    gen_best_fit = population_fitness[best_idx]
    if gen_best_fit > best_fitness:
        best_fitness = gen_best_fit
        best_robot = population[best_idx]
        stagnation_counter = 0
        current_mutation_rate = MUTATION_RATE  # reset to default
    else:
        stagnation_counter += 1

    if OUTPUT_ROBOT_GIFS:
        capture_simulation_frame(best_robot, gen, SCENARIO, STEPS, CONTROLLER)

    # Adaptive parameter control: increase mutation rate if fitness diversity is low.
    if fitness_std < STD_THRESHOLD:
        adjustment_factor = 1 + (STD_THRESHOLD - fitness_std) / STD_THRESHOLD
        current_mutation_rate = min(1.0, current_mutation_rate * adjustment_factor)

    log(f"Gen {gen} | Best Fitness: {best_fitness:.3f} | Current Gen Best: {gen_best_fit:.3f} | Mutation Rate: {current_mutation_rate:.3f} | Fitness Std: {fitness_std:.3f}")

    if stagnation_counter >= STAGNATION_LIMIT:
        log(f"> Stagnation reached {STAGNATION_LIMIT} generations: Increasing mutation and injecting random individuals.")
        current_mutation_rate = min(1.0, current_mutation_rate * MUTATION_RATE_INCREASE)
        population = random_injection(population, RANDOM_INJECTION_FRACTION)
        stagnation_counter = 0

    # Elitism based on raw fitness (you could choose to use shared fitness here too)
    sorted_indices = np.argsort(population_fitness)
    new_population = [population[i] for i in sorted_indices[-NUM_ELITE_ROBOTS:]]

    # For parent selection, use the shared fitness values.
    pop_copy = population[:]
    fitness_copy = list(shared_fitnesses)
    while len(new_population) < POPULATION_SIZE:
        parent1, idx1 = select_parent(pop_copy, fitness_copy)
        pop_copy.pop(idx1)
        fitness_copy.pop(idx1)
        parent2, idx2 = select_parent(pop_copy, fitness_copy)
        child1, child2 = crossover(parent1, parent2)
        child1 = refined_mutate(child1, current_mutation_rate)
        child2 = refined_mutate(child2, current_mutation_rate)
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)
    population = new_population

log(f"\n=== EVOLUTION COMPLETE ===\nBest Fitness Found: {best_fitness}")
log(f"Best Robot:\n{best_robot}")

for i in range(3):
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
utils.create_gif(best_robot, filename=os.path.join(RUN_DIR, 'random_search.gif'),
                 scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
