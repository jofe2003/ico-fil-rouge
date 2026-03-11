import math
import random

def euclidean(a, b):
    return math.dist(a, b)

def route_distance(route, coords, depot=0):
    if not route:
        return 0.0
    distance = euclidean(coords[depot], coords[route[0]])
    for i in range(len(route) - 1):
        distance += euclidean(coords[route[i]], coords[route[i + 1]])
    distance += euclidean(coords[route[-1]], coords[depot])
    return distance

def split_routes(chromosome, demands, capacity):
    routes = []
    current_route = []
    current_load = 0

    for client in chromosome:
        demand = demands[client]
        if current_load + demand <= capacity:
            current_route.append(client)
            current_load += demand
        else:
            routes.append(current_route)
            current_route = [client]
            current_load = demand

    if current_route:
        routes.append(current_route)

    return routes

def solution_cost(chromosome, coords, demands, capacity):
    routes = split_routes(chromosome, demands, capacity)
    return sum(route_distance(route, coords) for route in routes)

def create_individual(num_clients):
    chromosome = list(range(1, num_clients + 1))
    random.shuffle(chromosome)
    return chromosome

def create_population(pop_size, num_clients):
    return [create_individual(num_clients) for _ in range(pop_size)]

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0][:]

def order_crossover(parent1, parent2):
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = parent1[a:b + 1]

    fill_values = [x for x in parent2 if x not in child]
    fill_index = 0

    for i in range(n):
        if child[i] is None:
            child[i] = fill_values[fill_index]
            fill_index += 1

    return child

def mutate_swap(individual, mutation_rate):
    child = individual[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
    return child

def genetic_algorithm(coords, demands, capacity, pop_size=100, generations=200, crossover_rate=0.9, mutation_rate=0.1):
    num_clients = len(coords) - 1
    population = create_population(pop_size, num_clients)

    best_solution = None
    best_cost = float("inf")

    history = []

    for _ in range(generations):
        fitnesses = [solution_cost(ind, coords, demands, capacity) for ind in population]

        for ind, fit in zip(population, fitnesses):
            if fit < best_cost:
                best_cost = fit
                best_solution = ind[:]

        history.append(best_cost)

        new_population = []

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            if random.random() < crossover_rate:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                child1 = parent1[:]
                child2 = parent2[:]

            child1 = mutate_swap(child1, mutation_rate)
            child2 = mutate_swap(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    best_routes = split_routes(best_solution, demands, capacity)
    return best_solution, best_routes, best_cost, history

random.seed(42)

coords = {
    0: (50, 50),
    1: (10, 20),
    2: (20, 40),
    3: (30, 10),
    4: (40, 30),
    5: (60, 20),
    6: (70, 40),
    7: (80, 10),
    8: (20, 80),
    9: (50, 90),
    10: (80, 70)
}

demands = {
    0: 0,
    1: 2,
    2: 3,
    3: 4,
    4: 2,
    5: 5,
    6: 3,
    7: 4,
    8: 2,
    9: 3,
    10: 2
}

capacity = 10

best_solution, best_routes, best_cost, history = genetic_algorithm(
    coords,
    demands,
    capacity,
    pop_size=80,
    generations=200,
    crossover_rate=0.9,
    mutation_rate=0.1
)

print("Best chromosome:", best_solution)
print("Best routes:", best_routes)
print("Best cost:", round(best_cost, 2))