# import libraries and load data from json
import random
import json
from datetime import datetime
import copy
import pandas as pd
import matplotlib.pyplot as plt

NUMBER_OF_CLASSES_PER_WEEK = 11

# Load class data from JSON file
with open('classes.json', 'r', encoding='utf-8') as f:
    class_data = json.load(f)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
times = ["09:00", "11:00", "14:00", "16:00", "18:00"]
courses_classe = {}
for i in class_data:
    courses_classe.setdefault(i['Course'], set()).add(i['Class'])

total_courses = len(courses_classe)
max_classes = max(len(classes) for classes in courses_classe.values())
total_days = len(days)
total_time_slots = len(times)


# check the contents:

## to check the given options for timetable selection
for c in class_data:
    print(c)

print("....")
## to access courses and classes
for cor in courses_classe:
    print(cor, " -> ", courses_classe[cor])


 #---- Generate random chromosome ----#

def generate_chromosome():
    # TODO YOUR CODE HERE

    # Definir a distribuição de cada cadeira
    # onde as chaves são as cadeiras e os valores o numero necessário

    requirements = {
        "Tópicos de Física Moderna": {"T1": 2, "TP": 1},
        "Princípios de Programação Procedimental": {"TP": 2},
        "Comunicação Técnica": {"T1": 1, "PL": 1},
        "Estatística": {"TP": 2},
        "Análise Matemática II": {"TP": 2}
    }

    chromosome = []

    # Iterate over each course and its required class types.
    for course, reqs in requirements.items():
        for class_type, count in reqs.items():
            for _ in range(count):
                # Filter available options from class_data for the given course.
                # Use startswith for types like "TP" or "PL", while for "T1" an exact match might be desired.
                options = [cls for cls in class_data if cls["Course"] == course and
                           ((class_type in ["TP", "PL"] and cls["Class"].startswith(class_type)) or
                            (class_type not in ["TP", "PL"] and cls["Class"] == class_type))]

                # Shuffle the options to randomize selection.
                random.shuffle(options)
                selected = None
                # Try to select an option that does not conflict (i.e. same Day and Start Time) with already scheduled classes.
                for option in options:
                    conflict = any(
                        (option["Day"] == scheduled["Day"] and option["Start Time"] == scheduled["Start Time"])
                        for scheduled in chromosome)
                    if not conflict:
                        selected = option
                        break
                # If all options conflict, we simply select one (conflict will be penalized in fitness).
                if selected is None:
                    if not options:
                        raise ValueError(f"No available options for course {course} and class type {class_type}")
                    selected = random.choice(options)
                chromosome.append(selected)
    return chromosome


def generate_random_chromosome():
    """
    Generates a chromosome (a candidate timetable) completely at random.
    It randomly selects NUMBER_OF_CLASSES_PER_WEEK (11) classes from class_data,
    without any filtering or enforcing constraints.
    """
    # Sample 11 classes uniformly at random from class_data (without replacement)
    return random.sample(class_data, NUMBER_OF_CLASSES_PER_WEEK)



# helper function to export timetables
def save_timetable(timetable, filename='a_timetable.json',fitness_value="not calculated"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sorted(timetable, key=lambda x: x['Course']) + [{'fitness' : fitness_value}], f, indent=4, ensure_ascii=False)


#---- Fitness ----#

def fitness(timetable):
    # TODO YOUR CODE HERE

    reward = 0
    penalty = 0

    allowed_times = {"09:00", "09:30", "11:00", "14:00", "16:00", "18:00"}

    # Check Constraint 1 - Total classes must be exactly 11
    # but given that in the chromosome we already set to always 11 I think this will be always false
    if len(timetable) != NUMBER_OF_CLASSES_PER_WEEK:
        penalty += abs(NUMBER_OF_CLASSES_PER_WEEK - len(timetable)) * 30

    # Check Constraint 2 -Course Distribution
    # Define the required distribution.
    required = {
        "Tópicos de Física Moderna": {"T1": 2, "TP": 1},
        "Princípios de Programação Procedimental": {"TP": 2},
        "Comunicação Técnica": {"T1": 1, "PL": 1},
        "Estatística": {"TP": 2},
        "Análise Matemática II": {"TP": 2}
    }

    # For each course, check that the count of each required class type matches.
    for course, reqs in required.items():

        # Get all timetable entries for the course.
        course_entries = [entry for entry in timetable if entry["Course"] == course]

        # Total required for this course.
        total_required = sum(reqs.values())

        if len(course_entries) != total_required:   # same here, in the chromosome I already define the maximum in there so this will be always false
            penalty += (len(course_entries) - total_required)*30

        # Check each required class type.
        for class_type, count_required in reqs.items():
            # Use startswith for "TP" and "PL" requirements.
            entries = [entry for entry in course_entries if entry["Class"].startswith(class_type)]
            if len(entries) != count_required:
                penalty += abs(count_required - len(entries)) * 30
            # If more than one class is required for the same type, check that they have the same subtype.
            if count_required > 1 and len(entries) == count_required:
                subtypes = set(entry["Class"] for entry in entries)
                if len(subtypes) > 1:
                    penalty += 50  # Extra penalty for inconsistent subtypes

    # Check Constraint 3 - No Overlaps
    time_slots = {}

    # Fill the time_slots dict where it will have the day,hour as key and the class as values,
    # if a key as more than 1 value it means there is overlap
    for entry in timetable:
        key = (entry["Day"], entry["Start Time"])
        time_slots.setdefault(key, []).append(entry)
    for key, entries in time_slots.items():
        if len(entries) > 1:
            # Each extra class in a timeslot is a conflict.
            penalty += (len(entries) - 1) * 50



    # Constraint 4: Valid Time Slots
    for entry in timetable:
        if entry["Start Time"] not in allowed_times:
            penalty += 20


    # Check Constraint 5 - Balanced Distribution across Days: TODO THIS NEEDS TO BE CHECKED AND MADE TO PENALTY EXTREME CASES LIKE HAVING A DAY FROM 9 TO 18 AND ANOTHER JUST AT 9
    valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}

    # Here we measure the spread of classes over the valid days.
    day_counts = {day: 0 for day in valid_days}
    for entry in timetable:
        if entry["Day"] in day_counts:
            day_counts[entry["Day"]] += 1
    # Penalize based on the difference between the most- and least-populated days.
    max_classes_day = max(day_counts.values())
    min_classes_day = min(day_counts.values())
    penalty += (max_classes_day - min_classes_day) * 10


    # Additional Reward/Penalty: Morning classes.
    # We define morning classes as those scheduled before 14:00.
    for entry in timetable:
        if entry["Start Time"] in {"09:00","09:30", "11:00"}:
            if entry["Start Time"] == "11:00":
                reward += 15  # Reward: prefer 11:00 over 9:00
            elif entry["Start Time"] == "09:30":
                penalty += 6  # Penalty: discourage very early classes
            elif entry["Start Time"] == "09:00":
                penalty += 20  # Penalty: discourage very early classes

    # Additional Reward/Penalty: Noon/Afternoon classes.
    # We consider classes with start times 14:00, 16:00, or 18:00 as "noon" classes.
    for entry in timetable:
        if entry["Start Time"] in {"14:00", "16:00", "18:00"}:
            if entry["Start Time"] == "14:00":
                reward += 15  # Reward: ideal for noon classes
            elif entry["Start Time"] == "16:00":  # if scheduled at 16:00 or 18:00, we apply a penalty.
                penalty += 6
            else:
                penalty += 20
  
    # New Constraint: Avoid Holes in Daily Schedule.
    ordered_times = ["09:00","09:30", "11:00", "14:00", "16:00", "18:00"]
    for day in valid_days:
        day_entries = [entry for entry in timetable if entry["Day"] == day]
        if len(day_entries) > 1:
            day_entries.sort(key=lambda entry: ordered_times.index(entry["Start Time"]))
            for i in range(len(day_entries) - 1):
                current_time = day_entries[i]["Start Time"]
                next_time = day_entries[i + 1]["Start Time"]
                index_current = ordered_times.index(current_time)
                index_next = ordered_times.index(next_time)
                gap = index_next - index_current
                if gap > 1:
                    penalty += (gap - 1) * 3
                    if next_time in {"16:00", "18:00"}:
                        penalty += 10
    

    # New Constraint: Penalize TP or PL classes scheduled on Wednesday or Friday.
    for entry in timetable:
        if entry["Day"] in {"Wednesday", "Friday"}:
            if entry["Class"].startswith("TP") or entry["Class"].startswith("PL"):
                penalty += 20  # Adjust the value as needed


    fitness = penalty - reward

    return fitness

def chromosome_to_timetable(chromosome):
    """
    Converts a chromosome (list of genes) into a timetable.
    Each gene is expected to be a dictionary with keys:
    "Course", "Class", "Day", "Start Time".
    Returns a sorted list (by Course) of these dictionaries.
    """
    timetable = [gene.copy() for gene in chromosome]
    timetable.sort(key=lambda x: x["Course"])
    return timetable


#---- This code blocks serves to analyze the fitness function ----#

def visualize_timetable(timetable, fitness_value,
                        days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]):

    # Create a full timeline in half-hour increments from 09:00 to 20:00.
    times = []
    start_minutes = 9 * 60  # 09:00 in minutes
    end_minutes = 20 * 60  # 20:00 in minutes
    for t in range(start_minutes, end_minutes + 1, 30):
        hour = t // 60
        minute = t % 60
        times.append(f"{hour:02d}:{minute:02d}")

    # Create a DataFrame with these times as rows and days as columns.
    schedule = pd.DataFrame('', index=times, columns=days)

    # Allowed start times (when a class can begin).
    allowed_starts = {"09:00", "09:30", "11:00", "14:00", "16:00", "18:00"}
    duration_slots = 4  # 2 hours = 4 half-hour slots

    # For each class in the timetable, fill the corresponding block.
    for entry in timetable:
        day = entry["Day"]
        start_time = entry["Start Time"]
        # Only consider classes that start at one of the allowed times.
        if start_time not in allowed_starts:
            continue
        if start_time in times:
            start_idx = times.index(start_time)
            # Place the class information in the first cell.
            text = f"{entry['Course']}\n{entry['Class']}"
            schedule.at[times[start_idx], day] = text
            # For the next (duration_slots - 1) rows, mark as continuation.
            for j in range(1, duration_slots):
                if start_idx + j < len(times):
                    schedule.at[times[start_idx + j], day] = "..."

    # Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')

    # Create the table using the DataFrame values.
    table = ax.table(cellText=schedule.values,
                     rowLabels=schedule.index,
                     colLabels=schedule.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.title(f"Timetable Visual Representation\nFitness: {fitness_value}", fontweight="bold")
    plt.show()


def simple_visualize_timetable(timetable,fitness_value, days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                               times=["09:00", "09:30", "11:00", "14:00", "16:00", "18:00"]):
    # Create a DataFrame with times as rows and days as columns
    schedule = pd.DataFrame('', index=times, columns=days)

    # Populate the DataFrame with schedule entries.
    # If more than one class occupies the same timeslot, we join them with a separator.
    for entry in timetable:
        day = entry["Day"]
        time = entry["Start Time"]
        text = f"{entry['Course']}\n{entry['Class']}"
        # If a cell already has an entry, append to it.
        if schedule.loc[time, day]:
            schedule.loc[time, day] += "\n---\n" + text
        else:
            schedule.loc[time, day] = text

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table using the DataFrame values
    table = ax.table(cellText=schedule.values,
                     rowLabels=schedule.index,
                     colLabels=schedule.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.title(f"Timetable Visual Representation\nFitness: {fitness_value}", fontweight="bold")
    plt.show()


def check_constraints(timetable):
    messages = []

    # Constraint 1: Total Classes must be exactly 11.
    if len(timetable) == NUMBER_OF_CLASSES_PER_WEEK:
        messages.append(f"Total Classes: {len(timetable)} ✓")
    else:
        messages.append(f"Total Classes: {len(timetable)} X")

    # Constraint 2: Course Distribution.
    required = {
        "Tópicos de Física Moderna": {"T1": 2, "TP": 1},
        "Princípios de Programação Procedimental": {"TP": 2},
        "Comunicação Técnica": {"T1": 1, "PL": 1},
        "Estatística": {"TP": 2},
        "Análise Matemática II": {"TP": 2}
    }
    for course, reqs in required.items():
        course_entries = [entry for entry in timetable if entry["Course"] == course]
        total_required = sum(reqs.values())
        if len(course_entries) == total_required:
            dist_ok = True
            for class_type, count_required in reqs.items():
                # Use startswith for "TP" and "PL".
                count = sum(1 for entry in course_entries if entry["Class"].startswith(class_type))
                if count != count_required:
                    dist_ok = False
                    break
            if dist_ok:
                messages.append(f"{course} distribution: ✓")
            else:
                messages.append(f"{course} distribution: X")
        else:
            messages.append(f"{course} distribution: X")

    # Constraint 3: No Overlaps.
    time_slots = {}
    overlaps = False
    for entry in timetable:
        key = (entry["Day"], entry["Start Time"])
        time_slots.setdefault(key, []).append(entry)
    for key, entries in time_slots.items():
        if len(entries) > 1:
            overlaps = True
            break
    messages.append("No Overlaps: " + ("✓" if not overlaps else "X"))

    # Constraint 4: Valid Time Slots.
    allowed_times = {"09:00","09:30", "11:00", "14:00", "16:00", "18:00"}
    invalid = any(entry["Start Time"] not in allowed_times for entry in timetable)
    messages.append("Valid Time Slots: " + ("✓" if not invalid else "X"))

    return messages


def visualize_constraints(timetable, fitness_value):
    # Prepare a summary of constraints.
    constraints_status = "\n".join(check_constraints(timetable))

    # Prepare a summary list of chosen classes.
    classes_list = "\n".join([f"{entry['Course']} | {entry['Class']} | {entry['Day']} @ {entry['Start Time']}"
                              for entry in sorted(timetable, key=lambda x: (x["Course"], x["Day"], x["Start Time"]))])

    summary_text = f"Fitness: {fitness_value}\n\nConstraint Status:\n{constraints_status}\n\nChosen Classes:\n{classes_list}"

    # Create a new figure for the summary.
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, summary_text, fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, pad=10))
    plt.axis('off')
    plt.title("Constraints and Chosen Classes Summary", fontweight="bold")
    plt.show()

#-------------------#

#---- Crossover ----#

def crossover(parent1, parent2):
    """
    One-point crossover with repair.

    Steps:
    1. Randomly choose a crossover point.
    2. Create a child by combining the first part of parent1 with the second part of parent2.
    3. Repair the child to enforce the required distribution for each course.
       - For each course and class type requirement, count the genes.
         If there are too many, randomly remove extras.
         If there are too few, randomly add missing genes from the available options.
    4. Shuffle the repaired child and return it.
    """
    # Step 1 & 2: One-point crossover.
    cp = random.randint(1, len(parent1) - 1)
    child = parent1[:cp] + parent2[cp:]

    # Required distribution for each course.
    requirements = {
        "Tópicos de Física Moderna": {"T1": 2, "TP": 1},
        "Princípios de Programação Procedimental": {"TP": 2},
        "Comunicação Técnica": {"T1": 1, "PL": 1},
        "Estatística": {"TP": 2},
        "Análise Matemática II": {"TP": 2}
    }

    # Step 3: Repair the child so that the course distribution is correct.
    repaired_child = []

    # For each course and required class type, fix the genes.
    for course, reqs in requirements.items():
        for class_type, req_count in reqs.items():
            # Depending on the class type, filter using exact match (for T1)
            # or prefix match (for TP and PL).
            if class_type in {"TP", "PL"}:
                genes = [gene for gene in child if gene["Course"] == course and gene["Class"].startswith(class_type)]
            else:
                genes = [gene for gene in child if gene["Course"] == course and gene["Class"] == class_type]

            # If there are too many genes for this requirement, randomly pick the required number.
            if len(genes) > req_count:
                genes = random.sample(genes, req_count)
            # If there are too few, add missing genes.
            elif len(genes) < req_count:
                # Get available options from class_data.
                if class_type in {"TP", "PL"}:
                    options = [cls for cls in class_data
                               if cls["Course"] == course and cls["Class"].startswith(class_type)]
                else:
                    options = [cls for cls in class_data
                               if cls["Course"] == course and cls["Class"] == class_type]
                # Remove those already selected.
                options = [opt for opt in options if opt not in genes]
                missing_count = req_count - len(genes)
                if len(options) < missing_count:
                    # If not enough options remain, add all (this should be rare if data is consistent).
                    missing = options
                else:
                    missing = random.sample(options, missing_count)
                genes.extend(missing)
            # Append the genes for this course/type to the repaired child.
            repaired_child.extend(genes)

    # Step 4: Shuffle the repaired child to remove any ordering bias.
    random.shuffle(repaired_child)
    return repaired_child


def crossover_by_day(parent1, parent2, day_split=["Monday", "Tuesday", "Wednesday"]):
    """
    Creates a child chromosome by taking all classes from Parent1 on the days in day_split
    and all classes from Parent2 on the remaining days.
    """
    child = []
    # Inherit classes for the chosen days from parent1.
    for gene in parent1:
        if gene["Day"] in day_split:
            child.append(gene)
    # Inherit classes for the other days from parent2.
    for gene in parent2:
        if gene["Day"] not in day_split:
            child.append(gene)

    # (Optional) If the total number of classes isn't 11, you might fill in randomly.
    if len(child) < NUMBER_OF_CLASSES_PER_WEEK:
        # Add additional classes from parent1 (or random) until 11 classes are reached.
        missing = NUMBER_OF_CLASSES_PER_WEEK - len(child)
        additional = random.sample(parent1, missing)
        child.extend(additional)
    elif len(child) > NUMBER_OF_CLASSES_PER_WEEK:
        child = random.sample(child, NUMBER_OF_CLASSES_PER_WEEK)

    return child


def crossover_by_time(parent1, parent2):
    """
    Creates a child chromosome by taking all classes that start before 14:00 from Parent1,
    and all classes that start at or after 14:00 from Parent2.
    """
    child = []
    for gene in parent1:
        # Consider 14:00 as the cutoff.
        if gene["Start Time"] < "14:00":
            child.append(gene)
    for gene in parent2:
        if gene["Start Time"] >= "14:00":
            child.append(gene)

    # Ensure we have exactly 11 classes; if not, fill in randomly from the union.
    if len(child) < NUMBER_OF_CLASSES_PER_WEEK:
        union = list({tuple(gene.items()): gene for gene in (parent1 + parent2)}.values())
        missing = NUMBER_OF_CLASSES_PER_WEEK - len(child)
        child.extend(random.sample(union, missing))
    elif len(child) > NUMBER_OF_CLASSES_PER_WEEK:
        child = random.sample(child, NUMBER_OF_CLASSES_PER_WEEK)

    return child


#---- Mutation ----#

def mutate(individual, mutation_rate):
    """
    Mutates the given individual (chromosome) in place.
    For each gene in the chromosome, with probability mutation_rate,
    the gene is replaced by another gene (class) that satisfies the same
    course and class type constraint.

    The mutation operator preserves the required distribution by ensuring
    that the replacement gene belongs to the same course and matches the
    class type criteria.
    """
    # Create a copy of the individual to avoid modifying the original in-place.
    mutated = individual.copy()

    for i, gene in enumerate(mutated):
        if random.random() < mutation_rate:
            course = gene["Course"]
            class_type = gene["Class"]
            # Determine if we are matching with prefix or exact.
            if class_type.startswith("TP") or class_type.startswith("PL"):
                # Use prefix matching (e.g., if gene is "TP3", allow any "TP" class)
                options = [cls for cls in class_data if
                           cls["Course"] == course and cls["Class"].startswith(class_type[:2])]
            else:
                options = [cls for cls in class_data if cls["Course"] == course and cls["Class"] == class_type]

            # To avoid selecting the same gene, remove the current gene if alternatives exist.
            if len(options) > 1:
                options = [opt for opt in options if opt != gene]
            # Only mutate if there is at least one alternative.
            if options:
                new_gene = random.choice(options)
                mutated[i] = new_gene
    return mutated


#---- Parent Selection ----#


def tournament_selection(population, k=5):
    """
    Performs tournament selection on the given population and returns two parents.

    For each parent:
      1. Randomly select k individuals from the population.
      2. Choose the individual with the best (lowest) fitness among those k.

    Returns:
      A tuple containing two selected parent individuals.
    """

    def select_one():
        # Randomly choose k individuals from the population.
        competitors = random.sample(population, k)
        # Sort competitors by their fitness (lower is better)
        competitors.sort(key=lambda ind: fitness(ind))
        return competitors[0]

    parent1 = select_one()
    parent2 = select_one()
    return parent1, parent2


def genetic_algorithm(pop_size=100, generations=2500, mutation_rate=0.05):
    population = [generate_random_chromosome() for _ in range(pop_size)]
    for gen in range(generations):
        population = sorted(population, key=fitness)
        if fitness(population[0]) == 0:
            break
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = tournament_selection(population)
            child = crossover_by_day(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
        print(f"Generation {gen + 1}, Best Fitness: {fitness(population[0])}")
    return population[0]

for i in range(5):
    best_timetable = genetic_algorithm()

    # Calculate fitness for the timetable
    fit = fitness(best_timetable)
    print("Fitness:", fit)
    print("-" * 40)

    # Example usage:
    # Assume 'timetable' is the sorted list you get from chromosome_to_timetable(chromosome)
    timetable = chromosome_to_timetable(best_timetable)
    simple_visualize_timetable(timetable, fit)
    visualize_constraints(timetable, fit)















