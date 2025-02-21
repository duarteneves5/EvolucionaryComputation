# import libraries and load data from json
import random
import json
from datetime import datetime
import copy

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


def generate_chromosome_random():
    # Define the required distribution for each course.
    requirements = {
        "Tópicos de Física Moderna": {"T1": 2, "TP": 1},
        "Princípios de Programação Procedimental": {"TP": 2},
        "Comunicação Técnica": {"T1": 1, "PL": 1},
        "Estatística": {"TP": 2},
        "Análise Matemática II": {"TP": 2}
    }

    chromosome = []  # List to hold the randomly chosen classes (genes)

    # Iterate over each course and its required class types.
    for course, reqs in requirements.items():
        for class_type, count in reqs.items():
            # Filter available options for the given course.
            options = [cls for cls in class_data
                       if cls["Course"] == course and
                       ((class_type in ["TP", "PL"] and cls["Class"].startswith(class_type)) or
                        (class_type not in ["TP", "PL"] and cls["Class"] == class_type))]

            # For a requirement needing more than one instance, we want all selected classes to be of the same subtype.
            if count > 1:
                # Build a dictionary counting available classes per subtype.
                subtype_counts = {}
                for option in options:
                    subtype = option["Class"]
                    subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
                # Filter to subtypes that have at least 'count' available.
                valid_subtypes = [sub for sub, cnt in subtype_counts.items() if cnt >= count]
                if not valid_subtypes:
                    raise ValueError(f"Not enough available classes for {course} with type {class_type}")
                # Randomly choose one valid subtype.
                chosen_subtype = random.choice(valid_subtypes)
                # Filter options to those exactly matching the chosen subtype.
                subtype_options = [option for option in options if option["Class"] == chosen_subtype]
                # Select the required number randomly (without replacement).
                selected = random.sample(subtype_options, count)
            else:
                if len(options) < count:
                    raise ValueError(f"Not enough available classes for {course} and type {class_type}")
                selected = random.sample(options, count)
            chromosome.extend(selected)

    return chromosome

def chromosome_to_timetable(): # generate the timetable object from the representation
    ''' from the genes you should generate the a list with elements that
    follow a structure similar to classes.json as follows:
        {
            "Course": course,
            "Class": classe,
            "Day": day,
            "Start Time": time
        }'''


    pass

# helper function to export timetables
def save_timetable(timetable, filename='a_timetable.json',fitness_value="not calculated"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sorted(timetable, key=lambda x: x['Course']) + [{'fitness' : fitness_value}], f, indent=4, ensure_ascii=False)


#---- Fitness ----#

def fitness(timetable):
    # TODO YOUR CODE HERE

    reward = 0
    penalty = 0

    # Check Constraint 1 - Total classes must be exactly 11
    # but given that in the chromosome we already set to always 11 I think this will be always false
    if len(timetable) != NUMBER_OF_CLASSES_PER_WEEK:
        penalty += NUMBER_OF_CLASSES_PER_WEEK - len(timetable)

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
            penalty += len(course_entries) - total_required

        # Check each required class type.
        for class_type, count_required in reqs.items():
            # We use startswith to match prefixes ("TP" or "PL").
            count = sum(1 for entry in course_entries if entry["Class"].startswith(class_type))
            if count != count_required:  # same here, in the chromosome I already define the correct class type in there so this will be always false
                penalty += count_required - count

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
            penalty += (len(entries) - 1) * 10

    # Check Constraint 5 - Balanced Distribution across Days:
    valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    valid_times = {"09:00", "11:00", "14:00", "16:00", "18:00"}

    # Here we measure the spread of classes over the valid days.
    day_counts = {day: 0 for day in valid_days}
    for entry in timetable:
        if entry["Day"] in day_counts:
            day_counts[entry["Day"]] += 1
    # Penalize based on the difference between the most- and least-populated days.
    max_classes_day = max(day_counts.values())
    min_classes_day = min(day_counts.values())
    penalty += (max_classes_day - min_classes_day) * 2

    return penalty

#---- Crossover ----#

def crossover(parent1, parent2):
    # TODO YOUR CODE HERE
    pass

#---- Mutation ----#

def mutate(individual, mutation_rate):
    # TODO YOUR CODE HERE
    pass

#---- Parent Selection ----#


def tournament_selection(population, k=5):
    # TODO YOUR CODE HERE
    # Note: use sorted(<...>, key=fitness) to sort by value of the fitness function call
    pass


def genetic_algorithm(pop_size=100, generations=2500, mutation_rate=0.05):
    population = [generate_chromosome_random() for _ in range(pop_size)]
    for gen in range(generations):
        population = sorted(population, key=fitness)
        if fitness(population[0]) == 0:
            break
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
        print(f"Generation {gen + 1}, Best Fitness: {fitness(population[0])}")
    return population[0]

best_timetable = genetic_algorithm()















