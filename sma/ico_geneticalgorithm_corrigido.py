import math

# -*- coding: utf-8 -*-
"""
ICO — Algorithme Génétique pour le VRP
Centrale Lille — Fil Rouge

Fonction de coût standardisée (formule du cours) :
    f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij          (1)

  - c_ij  : coût de l'arc (i,j) — temps de trajet calculé par Haversine/vitesse
  - E     : ensemble des arcs de la solution x
  - K(x)  : nombre de véhicules utilisés dans la solution x
  - ω     : facteur de pénalité arbitraire non négatif important

Corrections apportées :
  1. fitnessFunction      → ajoute le terme ω·K(x) manquant
  2. fitnessFunctionWithPenalties → même ajout + pénalités de capacité séparées
  3. OMEGA est un paramètre centralisé, cohérent avec la BD (fichier 3_vehicles)
"""

import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

random.seed()

# ─────────────────────────────────────────────
# PARAMÈTRES VÉHICULES
# ─────────────────────────────────────────────
truckKg  = 20000   # capacité poids (kg)
truckVol = 20      # capacité volume (m³)
truckSpd = 0.6     # vitesse (degrés/min — unité interne)

# ω : facteur de pénalité pour chaque véhicule utilisé  ← NOUVEAU PARAMÈTRE
# Calibré pour être supérieur au coût moyen d'une tournée (évite de "gaspiller"
# des véhicules inutiles). Doit être grand mais pas explosif.
OMEGA = 500.0   # euros ou unités de coût par véhicule utilisé


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
customersDf = pd.read_excel("petit/2_detail_table_customers_petit.xlsx")
depotsDf    = pd.read_excel("petit/4_detail_table_depots_petit.xlsx")
trucksDf    = pd.read_excel("petit/3_detail_table_vehicles_petit.xlsx")
routes      = customersDf["ROUTE_ID"].unique()


def getData(route, customersDf, depotsDf, trucksDf):
    """Retourne (nb_trucks, liste_clients, dict_coût, dict_demande) pour une route."""
    C_df = customersDf[customersDf['ROUTE_ID'] == route].set_index(
        "CUSTOMER_NUMBER", drop=False).copy()
    D_df = depotsDf[depotsDf['ROUTE_ID'] == route].reset_index()

    numberOfTrucks = trucksDf[trucksDf['ROUTE_ID'] == route]['VEHICLE_NUMBER'].max()
    customersId    = list(C_df['CUSTOMER_NUMBER'].unique())
    V              = [0] + customersId
    edges          = [(i, j) for i in V for j in V if i != j]

    cost              = getCostDict(C_df, D_df, edges)
    demandForCustomer = {
        i: (C_df.loc[i, "TOTAL_WEIGHT_KG"], C_df.loc[i, "TOTAL_VOLUME_M3"])
        for i in C_df.index.tolist()
    }
    return numberOfTrucks, customersId, cost, demandForCustomer

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em quilômetros entre dois pontos geográficos
    utilizando a fórmula de Haversine.
    """
    R = 6371.0  # Raio da Terra em km

    # Conversão de graus para radianos
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Diferenças das coordenadas
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Fórmula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def getCostDict(customers, depot, edges):
    """
    Retourne un dict {(i,j): c_ij} où c_ij est le temps de trajet entre
    le nœud i et le nœud j (dépôt = indice 0, clients = CUSTOMER_NUMBER).
    c_ij = distance_Haversine_km / vitesse   → correspond à Σ c_ij dans f(x)
    """
    cost = {}
    depot_lat = depot.loc[0, "DEPOT_LATITUDE"]
    depot_lon = depot.loc[0, "DEPOT_LONGITUDE"]
    
    for (loc1, loc2) in edges:
        # Coordenadas do ponto de origem (loc1)
        if loc1 == 0:
            lat1, lon1 = depot_lat, depot_lon
        else:
            lat1 = customers.loc[loc1, "CUSTOMER_LATITUDE"]
            lon1 = customers.loc[loc1, "CUSTOMER_LONGITUDE"]
            
        # Coordenadas do ponto de destino (loc2)
        if loc2 == 0:
            lat2, lon2 = depot_lat, depot_lon
        else:
            lat2 = customers.loc[loc2, "CUSTOMER_LATITUDE"]
            lon2 = customers.loc[loc2, "CUSTOMER_LONGITUDE"]
            
        # Calcula a distância real em Km
        distancia_km = haversine(lat1, lon1, lat2, lon2)
        
        # Calcula o custo (tempo) usando a velocidade
        cost[(loc1, loc2)] = round(distancia_km / truckSpd, 4)
        
    return cost


# ─────────────────────────────────────────────
# UTILITAIRES POPULATION
# ─────────────────────────────────────────────

def _canAddCustomerToTruck(truck, truckCapacityKg, truckCapacityVol,
                           singleDemand, allCustomersDemands):
    total_weight = sum(allCustomersDemands[c][0] for c in truck if c != 0) + singleDemand[0]
    total_volume = sum(allCustomersDemands[c][1] for c in truck if c != 0) + singleDemand[1]
    return total_weight <= truckCapacityKg and total_volume <= truckCapacityVol


def initializePopulation(population_size, numberOfTrucks, customers, demandForCustomer):
    population = []
    random.seed()
    for _ in range(population_size):
        trucks             = [[0] for _ in range(numberOfTrucks)]
        remainingCustomers = set(customers)
        while remainingCustomers:
            for truck in trucks:
                if not remainingCustomers:
                    break
                customerChosen = random.choice(list(remainingCustomers))
                demand         = demandForCustomer[customerChosen]
                if _canAddCustomerToTruck(truck, truckKg, truckVol, demand, demandForCustomer):
                    truck.append(customerChosen)
                    remainingCustomers.remove(customerChosen)
        for truck in trucks:
            if len(truck) > 1:
                truck.append(0)
        population.append(trucks)
    return population


def initializeChaosPopulation(population_size, numberOfTrucks, customers):
    population = []
    for _ in range(population_size):
        trucks             = [[0] for _ in range(numberOfTrucks)]
        shuffled_customers = list(customers)
        random.shuffle(shuffled_customers)
        for i, customer in enumerate(shuffled_customers):
            trucks[i % numberOfTrucks].append(customer)
        for truck in trucks:
            truck.append(0)
        population.append(trucks)
    return population


# ─────────────────────────────────────────────
# ══════════════════════════════════════════════
#  FONCTIONS DE COÛT  — formule du cours f(x)
# ══════════════════════════════════════════════
# ─────────────────────────────────────────────

def count_vehicles_used(solution):
    """K(x) : nombre de véhicules avec au moins un client."""
    return sum(1 for route in solution if len(route) > 2)


def calculate_route_cost(route, cost):
    """Σ_{(i,j)∈route} c_ij  — coût des arcs d'une tournée."""
    total = 0.0
    for i in range(len(route) - 1):
        total += cost[(route[i], route[i + 1])]
    return total


def fitnessFunction(solution, cost, omega=OMEGA):
    """
    Fonction objectif exacte du cours (équation 1) :
        f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij

    Paramètres
    ----------
    solution : list[list[int]]  — liste de tournées ([0, c1, c2, …, 0])
    cost     : dict             — dict des coûts c_ij
    omega    : float            — pénalité par véhicule utilisé (≥ 0)

    Retour
    ------
    f(x) : float
    """
    # ── Σ c_ij  (coût total des arcs) ───────────────────────────────
    arc_cost = sum(calculate_route_cost(route, cost) for route in solution)

    # ── ω · K(x)  (pénalité pour le nombre de véhicules utilisés) ───
    k_x      = count_vehicles_used(solution)
    vehicle_penalty = omega * k_x

    return vehicle_penalty + arc_cost


def fitnessFunctionWithPenalties(solution, costDict, demandForCustomer,
                                 truckCapacityKg, truckCapacityVolume,
                                 penaltyFactor, omega=OMEGA):
    """
    Version étendue : f(x) + pénalités de violation de capacité.

        f(x) = ω·K(x)  +  Σ c_ij  +  pénalités_capacité

    Les pénalités de capacité ne font PAS partie de f(x) du cours ;
    elles sont ajoutées pour guider la recherche vers des solutions
    réalisables lors de l'évolution génétique.

    Paramètres
    ----------
    penaltyFactor : float — facteur dynamique (croît au fil des générations)
    omega         : float — même ω que fitnessFunction
    """
    # ── Coût objectif de base : ω·K(x) + Σ c_ij ────────────────────
    total_cost = fitnessFunction(solution, costDict, omega)

    # ── Pénalités de capacité (infaisabilité) ────────────────────────
    capacity_penalty = 0.0
    for route in solution:
        total_weight = 0.0
        total_volume = 0.0
        for node in route:
            if node != 0:
                w, v         = demandForCustomer[node]
                total_weight += w
                total_volume += v

        excess_weight = max(0.0, total_weight - truckCapacityKg)
        excess_volume = max(0.0, total_volume - truckCapacityVolume)

        if excess_weight > 0 or excess_volume > 0:
            capacity_penalty += (excess_weight + excess_volume) * penaltyFactor

    return total_cost + capacity_penalty


# ─────────────────────────────────────────────
# OPÉRATEURS GÉNÉTIQUES
# ─────────────────────────────────────────────

def tournamentSelection(population, tournament_size, mating_pool_size, cost, fitnessScores):
    matingPool = []
    while len(matingPool) < mating_pool_size:
        battle = random.sample(list(zip(population, fitnessScores)), tournament_size)
        winner = min(battle, key=lambda x: x[1])
        matingPool.append(winner[0])
    return matingPool


def flatten(routes):
    return [node for route in routes for node in route[1:-1]]


def orderCrossover(parent1, parent2):
    dad    = flatten(parent1)
    mom    = flatten(parent2)
    child1 = [None] * len(dad)
    child2 = [None] * len(dad)

    cx1, cx2 = sorted(random.sample(range(len(dad)), 2))

    child1[cx1:cx2 + 1] = dad[cx1:cx2 + 1]
    child2[cx1:cx2 + 1] = mom[cx1:cx2 + 1]

    position = 0
    for node in mom:
        if position == cx1:
            position = cx2 + 1
        if node not in child1:
            child1[position] = node
            position += 1

    position = 0
    for node in dad:
        if position == cx1:
            position = cx2 + 1
        if node not in child2:
            child2[position] = node
            position += 1

    return child1, child2


def reconstruct_routes_for_penalties(flat_sequence, numberOfTrucks):
    chunk_size = len(flat_sequence) // numberOfTrucks + 1
    trucks = []
    for i in range(numberOfTrucks):
        start  = i * chunk_size
        end    = start + chunk_size
        fatia  = flat_sequence[start:end]
        trucks.append([0] + fatia + [0] if fatia else [0, 0])
    return trucks


def treatCrossOver(parents, truckCapacityKg, truckCapacityVol,
                   demandForCustomer, numberOfTrucks):
    if len(parents) % 2 != 0:
        return parents   # protection

    parents1   = parents[:len(parents) // 2]
    parents2   = parents[len(parents) // 2:]
    population = []

    for (dad, mom) in zip(parents1, parents2):
        child1_flat, child2_flat = orderCrossover(dad, mom)
        route1 = reconstruct_routes_for_penalties(child1_flat, numberOfTrucks)
        route2 = reconstruct_routes_for_penalties(child2_flat, numberOfTrucks)
        population.append(route1)
        population.append(route2)

    return population


def mutation(population, mutationRate):
    for individual in population:
        if random.random() < mutationRate:
            t1 = random.randint(0, len(individual) - 1)
            t2 = random.randint(0, len(individual) - 1)
            truck1 = individual[t1]
            truck2 = individual[t2]
            if len(truck1) > 2 and len(truck2) > 2:
                i1 = random.randint(1, len(truck1) - 2)
                i2 = random.randint(1, len(truck2) - 2)
                truck1[i1], truck2[i2] = truck2[i2], truck1[i1]
    return population


def apply_2opt(route, costDict):
    best_route = route.copy()
    improved   = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                old_cost = (costDict[(best_route[i - 1], best_route[i])] +
                            costDict[(best_route[j],     best_route[j + 1])])
                new_cost = (costDict[(best_route[i - 1], best_route[j])] +
                            costDict[(best_route[i],     best_route[j + 1])])
                if new_cost < old_cost:
                    best_route[i:j + 1] = best_route[i:j + 1][::-1]
                    improved = True
        if improved:
            break
    return best_route


def local_search_mutation(population, costDict, mutationRate):
    new_population = []
    for individual in population:
        new_individual = [route.copy() for route in individual]
        if random.random() < mutationRate:
            for k in range(len(new_individual)):
                if len(new_individual[k]) > 4:
                    new_individual[k] = apply_2opt(new_individual[k], costDict)
        new_population.append(new_individual)
    return new_population


def inter_truck_local_search(population, costDict, mutationRate):
    new_population = []
    for individual in population:
        new_ind = [route.copy() for route in individual]
        if random.random() < mutationRate:
            valid_sources = [i for i in range(len(new_ind)) if len(new_ind[i]) > 2]
            if valid_sources:
                t_from   = random.choice(valid_sources)
                cust_idx = random.randint(1, len(new_ind[t_from]) - 2)
                customer = new_ind[t_from][cust_idx]

                valid_dests = [i for i in range(len(new_ind)) if i != t_from]
                if valid_dests:
                    t_to              = random.choice(valid_dests)
                    new_ind[t_from].pop(cust_idx)
                    best_insert_idx   = 1
                    best_cost_diff    = float('inf')

                    for insert_idx in range(1, len(new_ind[t_to])):
                        prev_node  = new_ind[t_to][insert_idx - 1]
                        next_node  = new_ind[t_to][insert_idx]
                        cost_added = costDict[(prev_node, customer)] + costDict[(customer, next_node)]
                        cost_removed = costDict[(prev_node, next_node)]
                        diff       = cost_added - cost_removed
                        if diff < best_cost_diff:
                            best_cost_diff  = diff
                            best_insert_idx = insert_idx

                    new_ind[t_to].insert(best_insert_idx, customer)
        new_population.append(new_ind)
    return new_population


# ─────────────────────────────────────────────
# ALGORITHME GÉNÉTIQUE PRINCIPAL
# ─────────────────────────────────────────────

def genetic_algorithm(populationSize, numberOfTrucks, truckCapacityKg,
                      truckCapacityVol, customersId, cost, demandForCustomer,
                      maxGenNumber=340, baseMutationRate=0.05, omega=OMEGA):
    """
    Algorithme génétique minimisant f(x) = ω·K(x) + Σ c_ij.

    Paramètres supplémentaires
    -------------------------
    omega : float — pénalité par véhicule utilisé (équation 1 du cours)
    """
    # ── Initialisation hybride ───────────────────────────────────────
    pop_legal   = initializePopulation(int(populationSize * 0.5),
                                       numberOfTrucks, customersId, demandForCustomer)
    pop_caotica = initializeChaosPopulation(int(populationSize * 0.5),
                                            numberOfTrucks, customersId)
    population  = pop_legal + pop_caotica
    while len(population) < populationSize:
        population.append(initializeChaosPopulation(1, numberOfTrucks, customersId)[0])

    print("Population initialisée :", len(population), "individus")

    generations          = 0
    best_solution        = None
    best_fitness         = float('inf')
    history              = []
    estagnacao           = 0
    current_mutation_rate = baseMutationRate

    while generations < maxGenNumber:

        # Pénalité de capacité dynamique (croît en cours d'évolution)
        progress        = generations / maxGenNumber
        current_penalty = 100.0 + progress * 9900.0

        # ── Calcul du fitness avec la formule standardisée ───────────
        # f(x) = ω·K(x) + Σ c_ij + pénalités_capacité (guide évolution)
        fitnesses = [
            fitnessFunctionWithPenalties(
                sol, cost, demandForCustomer,
                truckCapacityKg, truckCapacityVol,
                current_penalty, omega=omega
            )
            for sol in population
        ]

        # ── Mise à jour du meilleur ──────────────────────────────────
        generation_best = float('inf')
        for i, fitness in enumerate(fitnesses):
            if fitness < best_fitness:
                best_fitness  = fitness
                best_solution = population[i].copy()
            if fitness < generation_best:
                generation_best = fitness

        # ── Mutación adaptative ──────────────────────────────────────
        if history and generation_best >= history[-1]:
            estagnacao += 1
        else:
            estagnacao            = 0
            current_mutation_rate = baseMutationRate

        if estagnacao > 15:
            current_mutation_rate = 0.30

        # ── Sélection + Croisement ───────────────────────────────────
        winners    = tournamentSelection(population, 2, len(population), cost, fitnesses)
        population = treatCrossOver(winners, truckCapacityKg, truckCapacityVol,
                                    demandForCustomer, numberOfTrucks)

        # ── Mutations ────────────────────────────────────────────────
        population = mutation(population, current_mutation_rate)
        population = inter_truck_local_search(population, cost, current_mutation_rate)
        population = local_search_mutation(population, cost, current_mutation_rate)

        # ── Élitisme ─────────────────────────────────────────────────
        population[0] = best_solution.copy()

        generations += 1
        history.append(best_fitness)

        if generations % 50 == 0 or generations == maxGenNumber:
            k_x = count_vehicles_used(best_solution)
            arc = best_fitness - omega * k_x   # coût des arcs seul
            print(f"Génération {generations:>4} | "
                  f"f(x) = ω·K(x)+Σc_ij = {omega:.0f}×{k_x} + {arc:.2f} = {best_fitness:.2f} | "
                  f"Pénalité dyn. = {current_penalty:.0f}")

    print(f"\nMeilleure solution trouvée : {best_solution}")
    print(f"f(x) final = {best_fitness:.2f}  "
          f"(ω·K = {omega * count_vehicles_used(best_solution):.2f} + "
          f"Σc_ij = {best_fitness - omega * count_vehicles_used(best_solution):.2f})")

    return best_solution, best_fitness, history


# ─────────────────────────────────────────────
# AFFICHAGE DES RÉSULTATS
# ─────────────────────────────────────────────

def gerar_tabela_resultados(best_solution, cost_dict, demand, truck_kg, truck_vol, code, omega=OMEGA):
    summary    = []
    total_arc_cost = 0.0

    for i, route in enumerate(best_solution):
        if len(route) > 2:
            route_weight = sum(demand[node][0] for node in route if node != 0)
            route_vol    = sum(demand[node][1] for node in route if node != 0)
            route_cost   = calculate_route_cost(route, cost_dict)
            total_arc_cost += route_cost

            summary.append({
                "Véhicule"         : f"Camion {i + 1}",
                "Nb Clients"       : len(route) - 2,
                "Coût arcs Σc_ij"  : round(route_cost, 2),
                "Poids Total (kg)" : round(route_weight, 2),
                "Occ. Poids"       : f"{round((route_weight / truck_kg) * 100, 1)}%",
                "Volume Total (m³)": round(route_vol, 2),
                "Occ. Volume"      : f"{round((route_vol / truck_vol) * 100, 1)}%",
            })

    k_x     = len(summary)   # K(x)
    f_total = omega * k_x + total_arc_cost

    totals = {
        "Véhicule"         : f"TOTAL  f(x)={f_total:.2f}",
        "Nb Clients"       : sum(r["Nb Clients"] for r in summary),
        "Coût arcs Σc_ij"  : round(total_arc_cost, 2),
        "Poids Total (kg)" : round(sum(r["Poids Total (kg)"] for r in summary), 2),
        "Occ. Poids"       : f"ω·K(x)={omega:.0f}×{k_x}={omega * k_x:.0f}",
        "Volume Total (m³)": round(sum(r["Volume Total (m³)"] for r in summary), 2),
        "Occ. Volume"      : "-",
    }

    df_summary = pd.DataFrame(summary)
    df_summary = pd.concat([df_summary, pd.DataFrame([totals])], ignore_index=True)
    print(df_summary.to_string())
    return df_summary


def plot_evolucao_fitness(history, route_id, bdd_type=""):
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.lineplot(x=range(1, len(history) + 1), y=history, color='purple', linewidth=2)
    plt.title(f'Évolution de f(x) = ω·K(x)+Σc_ij — Route : {route_id} ({bdd_type})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Génération')
    plt.ylabel('f(x)  [ω·K(x) + Σ c_ij]')
    plt.tight_layout()
    plt.savefig(f"resultats_images/fitness_route_{route_id}_{bdd_type}.png", dpi=300)
    plt.show()


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ROUTE_PETITE = 2939484.0
    num_trucks, customers, cost, demand = getData(ROUTE_PETITE, customersDf, depotsDf, trucksDf)

    best_sol, best_fit, history = genetic_algorithm(
        populationSize  = 50,
        numberOfTrucks  = num_trucks,
        truckCapacityKg = truckKg,
        truckCapacityVol= truckVol,
        customersId     = customers,
        cost            = cost,
        demandForCustomer=demand,
        maxGenNumber    = 100,
        baseMutationRate= 0.10,
        omega           = OMEGA,
    )

    print("\n--- TABLEAU DE RÉSULTATS (PETITE TAILLE) ---")
    gerar_tabela_resultados(best_sol, cost, demand, truckKg, truckVol, ROUTE_PETITE)
    plot_evolucao_fitness(history, ROUTE_PETITE, "Petite")

    # ── Grande taille ────────────────────────────────────────────────
    customersDf = pd.read_excel("2_detail_table_customers.xls")
    depotsDf    = pd.read_excel("4_detail_table_depots.xls")
    trucksDf    = pd.read_excel("3_detail_table_vehicles.xls")

    for ROUTE_GRANDE in [2946091.0, 2922001.0]:
        num_trucks_g, customers_g, cost_g, demand_g = getData(
            ROUTE_GRANDE, customersDf, depotsDf, trucksDf)
        best_sol_g, best_fit_g, history_g = genetic_algorithm(
            populationSize  = 50,
            numberOfTrucks  = num_trucks_g,
            truckCapacityKg = truckKg,
            truckCapacityVol= truckVol,
            customersId     = customers_g,
            cost            = cost_g,
            demandForCustomer=demand_g,
            maxGenNumber    = 100,
            baseMutationRate= 0.10,
            omega           = OMEGA,
        )
        print(f"\n--- TABLEAU DE RÉSULTATS — Route {ROUTE_GRANDE} ---")
        gerar_tabela_resultados(best_sol_g, cost_g, demand_g, truckKg, truckVol, ROUTE_GRANDE)
        plot_evolucao_fitness(history_g, ROUTE_GRANDE, "Grande")
