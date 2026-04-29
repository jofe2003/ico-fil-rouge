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

Modification v2 :
  - Remplacement de orderCrossover (permutation plate) par vrp_route_crossover
    (héritage de routes complètes + insertion optimale des clients manquants)
"""

import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

random.seed()

IMAGE_ROOT_AG = Path("images_ag")

def format_route_id(route_id):
    try:
        value = float(route_id)
        return str(int(value)) if value.is_integer() else str(value).replace(".", "_")
    except Exception:
        return str(route_id).replace(".", "_")

def get_route_image_dir_ag(route_id):
    out_dir = IMAGE_ROOT_AG / f"route_{format_route_id(route_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# ─────────────────────────────────────────────
# PARAMÈTRES VÉHICULES
# ─────────────────────────────────────────────
truckKg  = 20000
truckVol = 20
truckSpd = 0.6

OMEGA = 5000.0


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
customersDf = pd.read_excel("..\\database\\petit\\2_detail_table_customers_petit.xlsx")
depotsDf    = pd.read_excel("..\\database\\petit\\4_detail_table_depots_petit.xlsx")
trucksDf    = pd.read_excel("..\\database\\petit\\3_detail_table_vehicles_petit.xlsx")
routes      = customersDf["ROUTE_ID"].unique()


def getData(route, customersDf, depotsDf, trucksDf):
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
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def getCostDict(customers, depot, edges):
    cost = {}
    depot_lat = depot.loc[0, "DEPOT_LATITUDE"]
    depot_lon = depot.loc[0, "DEPOT_LONGITUDE"]

    for (loc1, loc2) in edges:
        if loc1 == 0:
            lat1, lon1 = depot_lat, depot_lon
        else:
            lat1 = customers.loc[loc1, "CUSTOMER_LATITUDE"]
            lon1 = customers.loc[loc1, "CUSTOMER_LONGITUDE"]

        if loc2 == 0:
            lat2, lon2 = depot_lat, depot_lon
        else:
            lat2 = customers.loc[loc2, "CUSTOMER_LATITUDE"]
            lon2 = customers.loc[loc2, "CUSTOMER_LONGITUDE"]

        distancia_km = haversine(lat1, lon1, lat2, lon2)
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
# FONCTIONS DE COÛT
# ─────────────────────────────────────────────

def count_vehicles_used(solution):
    return sum(1 for route in solution if len(route) > 2)


def calculate_route_cost(route, cost):
    total = 0.0
    for i in range(len(route) - 1):
        total += cost.get((route[i], route[i + 1]), 0.0)
    return total


def fitnessFunction(solution, cost, omega=OMEGA):
    arc_cost        = sum(calculate_route_cost(route, cost) for route in solution)
    k_x             = count_vehicles_used(solution)
    vehicle_penalty = omega * k_x
    return vehicle_penalty + arc_cost


def fitnessFunctionWithPenalties(solution, costDict, demandForCustomer,
                                 truckCapacityKg, truckCapacityVolume,
                                 penaltyFactor, omega=OMEGA):
    total_cost       = fitnessFunction(solution, costDict, omega)
    capacity_penalty = 0.0
    for route in solution:
        total_weight = 0.0
        total_volume = 0.0
        for node in route:
            if node != 0:
                w, v          = demandForCustomer[node]
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


# ══════════════════════════════════════════════════════════════════════
#  CROSSOVER ADAPTÉ AU VRP  — remplace orderCrossover
# ══════════════════════════════════════════════════════════════════════

def vrp_route_crossover(parent1, parent2, numberOfTrucks,
                        demandForCustomer, truckCapacityKg, truckCapacityVol,
                        costDict):
    """
    Crossover orienté routes pour le VRP.

    Principe (3 étapes) :
    ─────────────────────
    1. HÉRITAGE DE ROUTES COMPLÈTES
       On mélange les routes non-vides des deux parents et on en accepte
       aléatoirement (p = 0.5) en vérifiant que :
         • aucun client n'est déjà placé (pas de doublon)
         • la capacité du camion receveur est respectée
       La route héritée est affectée au camion le moins chargé qui peut
       l'accueillir.

    2. INSERTION OPTIMALE DES CLIENTS MANQUANTS
       Pour chaque client absent, on l'insère à la position qui minimise
       l'augmentation de coût Δc = c(prev, c) + c(c, next) - c(prev, next)
       en respectant les contraintes de capacité.
       L'ordre de traitement des clients manquants suit la séquence de
       parent2 (transmission de l'information génétique du second parent).

    3. AFFECTATION FORCÉE
       Si aucun camion ne peut accueillir le client sans violer la capacité,
       il est affecté au camion le moins chargé (faisabilité garantie ;
       la pénalité dynamique corrigera la violation en cours d'évolution).

    Retour
    ──────
    child : list[list[int]]  — solution au format [[0, c1, …, 0], …]
    """

    # ── Ensemble de tous les clients ──────────────────────────────────
    all_customers = set()
    for route in parent1:
        for node in route:
            if node != 0:
                all_customers.add(node)

    # ── Initialisation de l'enfant (chaque camion part du dépôt) ─────
    child  = [[0] for _ in range(numberOfTrucks)]
    used   = set()

    # ── Étape 1 : héritage de routes complètes ────────────────────────
    # On mélange routes des deux parents pour éviter le biais d'ordre
    donor_routes = [r[:] for r in parent1 + parent2 if len(r) > 2]
    random.shuffle(donor_routes)

    for route in donor_routes:
        clients_in_route = [n for n in route if n != 0]

        # Ignorer si tous les clients sont déjà placés
        nouveaux = [c for c in clients_in_route if c not in used]
        if not nouveaux:
            continue

        # Accepter la route entière avec probabilité 0.5
        if random.random() >= 0.5:
            continue

        # Vérifier capacité pour les clients réellement nouveaux
        poids_total = sum(demandForCustomer[c][0] for c in nouveaux)
        vol_total   = sum(demandForCustomer[c][1] for c in nouveaux)

        # Chercher un camion capable d'accueillir ces clients
        camions_dispo = list(range(numberOfTrucks))
        random.shuffle(camions_dispo)
        for k in camions_dispo:
            charge_w = sum(demandForCustomer[n][0] for n in child[k] if n != 0)
            charge_v = sum(demandForCustomer[n][1] for n in child[k] if n != 0)
            if (charge_w + poids_total <= truckCapacityKg and
                    charge_v + vol_total   <= truckCapacityVol):
                for c in nouveaux:
                    child[k].append(c)
                    used.add(c)
                break   # route affectée → passer à la suivante

    # ── Étape 2 : clients manquants triés selon l'ordre de parent2 ────
    # Extraire l'ordre de visite dans parent2 pour les clients absents
    ordre_parent2 = []
    for route in parent2:
        for node in route:
            if node != 0 and node not in used:
                if node not in ordre_parent2:
                    ordre_parent2.append(node)

    # Ajouter les éventuels clients absents de parent2 (sécurité)
    manquants_restants = all_customers - used
    for c in manquants_restants:
        if c not in ordre_parent2:
            ordre_parent2.append(c)

    # ── Insertion position-optimale ───────────────────────────────────
    for c in ordre_parent2:
        if c in used:
            continue

        w_c = demandForCustomer[c][0]
        v_c = demandForCustomer[c][1]

        meilleur_k    = None
        meilleur_pos  = 1
        meilleur_cout = float('inf')

        for k in range(numberOfTrucks):
            charge_w = sum(demandForCustomer[n][0] for n in child[k] if n != 0)
            charge_v = sum(demandForCustomer[n][1] for n in child[k] if n != 0)

            # Vérifier capacité
            if charge_w + w_c > truckCapacityKg or charge_v + v_c > truckCapacityVol:
                continue

            route_k = child[k]   # [0, c1, c2, …]  (pas encore de 0 final)

            # Tester toutes les positions d'insertion dans cette route
            for pos in range(1, len(route_k) + 1):
                prev      = route_k[pos - 1]
                next_node = route_k[pos] if pos < len(route_k) else 0

                # Δcoût = c(prev→c) + c(c→next) - c(prev→next)
                delta = (costDict.get((prev, c),      0.0)
                       + costDict.get((c, next_node), 0.0)
                       - costDict.get((prev, next_node), 0.0))

                if delta < meilleur_cout:
                    meilleur_cout = delta
                    meilleur_k   = k
                    meilleur_pos = pos

        # ── Étape 3 : affectation forcée si aucun camion disponible ───
        if meilleur_k is None:
            meilleur_k = min(
                range(numberOfTrucks),
                key=lambda k: sum(demandForCustomer[n][0] for n in child[k] if n != 0)
            )
            meilleur_pos = len(child[meilleur_k])

        child[meilleur_k].insert(meilleur_pos, c)
        used.add(c)

    # ── Fermeture des routes avec le dépôt ────────────────────────────
    for k in range(numberOfTrucks):
        child[k].append(0)

    return child


def treatCrossOver(parents, truckCapacityKg, truckCapacityVol,
                   demandForCustomer, numberOfTrucks, costDict):
    """
    Applique vrp_route_crossover sur toutes les paires de parents.

    Signature mise à jour : reçoit costDict en paramètre supplémentaire
    (nécessaire pour le calcul du Δcoût dans vrp_route_crossover).
    """
    if len(parents) % 2 != 0:
        return parents

    parents1   = parents[:len(parents) // 2]
    parents2   = parents[len(parents) // 2:]
    population = []

    for (dad, mom) in zip(parents1, parents2):
        child1 = vrp_route_crossover(
            dad, mom, numberOfTrucks,
            demandForCustomer, truckCapacityKg, truckCapacityVol, costDict)
        child2 = vrp_route_crossover(
            mom, dad, numberOfTrucks,
            demandForCustomer, truckCapacityKg, truckCapacityVol, costDict)
        population.append(child1)
        population.append(child2)

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
                old_cost = (costDict.get((best_route[i - 1], best_route[i]),   0.0) +
                            costDict.get((best_route[j],     best_route[j + 1]), 0.0))
                new_cost = (costDict.get((best_route[i - 1], best_route[j]),     0.0) +
                            costDict.get((best_route[i],     best_route[j + 1]), 0.0))
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
                    t_to            = random.choice(valid_dests)
                    new_ind[t_from].pop(cust_idx)
                    best_insert_idx = 1
                    best_cost_diff  = float('inf')

                    for insert_idx in range(1, len(new_ind[t_to])):
                        prev_node    = new_ind[t_to][insert_idx - 1]
                        next_node    = new_ind[t_to][insert_idx]
                        cost_added   = costDict.get((prev_node, customer), 0.0) + costDict.get((customer, next_node), 0.0)
                        cost_removed = costDict.get((prev_node, next_node), 0.0)
                        diff         = cost_added - cost_removed
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

    pop_legal   = initializePopulation(int(populationSize * 0.5),
                                       numberOfTrucks, customersId, demandForCustomer)
    pop_caotica = initializeChaosPopulation(int(populationSize * 0.5),
                                            numberOfTrucks, customersId)
    population  = pop_legal + pop_caotica
    while len(population) < populationSize:
        population.append(initializeChaosPopulation(1, numberOfTrucks, customersId)[0])

    print("Population initialisée :", len(population), "individus")

    generations           = 0
    best_solution         = None
    best_fitness          = float('inf')
    history               = []
    estagnacao            = 0
    current_mutation_rate = baseMutationRate

    while generations < maxGenNumber:

        progress        = generations / maxGenNumber
        current_penalty = 100.0 + progress * 9900.0

        fitnesses = [
            fitnessFunctionWithPenalties(
                sol, cost, demandForCustomer,
                truckCapacityKg, truckCapacityVol,
                current_penalty, omega=omega)
            for sol in population
        ]

        generation_best = float('inf')
        for i, fitness in enumerate(fitnesses):
            if fitness < best_fitness:
                best_fitness  = fitness
                best_solution = population[i].copy()
            if fitness < generation_best:
                generation_best = fitness

        if history and generation_best >= history[-1]:
            estagnacao += 1
        else:
            estagnacao            = 0
            current_mutation_rate = baseMutationRate

        if estagnacao > 15:
            current_mutation_rate = 0.30

        winners    = tournamentSelection(population, 2, len(population), cost, fitnesses)

        # ── Crossover VRP (costDict passé en argument) ────────────────
        population = treatCrossOver(
            winners, truckCapacityKg, truckCapacityVol,
            demandForCustomer, numberOfTrucks, costDict=cost)

        population = mutation(population, current_mutation_rate)
        population = inter_truck_local_search(population, cost, current_mutation_rate)
        population = local_search_mutation(population, cost, current_mutation_rate)

        population[0] = best_solution.copy()

        generations += 1
        history.append(best_fitness)

        if generations % 50 == 0 or generations == maxGenNumber:
            k_x = count_vehicles_used(best_solution)
            arc = best_fitness - omega * k_x
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
    summary        = []
    total_arc_cost = 0.0

    for i, route in enumerate(best_solution):
        if len(route) > 2:
            route_weight    = sum(demand[node][0] for node in route if node != 0)
            route_vol       = sum(demand[node][1] for node in route if node != 0)
            route_cost      = calculate_route_cost(route, cost_dict)
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

    k_x     = len(summary)
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

    out_dir   = get_route_image_dir_ag(code)
    fig_h     = 1.5 + len(df_summary) * 0.5
    fig, ax   = plt.subplots(figsize=(12, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=df_summary.values,
                   colLabels=df_summary.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4c72b0')
            cell.set_text_props(weight='bold', color='white')
    plt.title(f"Résultats AG — Route {format_route_id(code)} | ω={omega:.0f}",
              fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    path = out_dir / f"ag_results_table_{format_route_id(code)}.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Tabela salva: {path}")
    # ─────────────────────────────────────────────────────────────

    print(df_summary.to_string())
    return df_summary


def plot_evolucao_fitness(history, route_id, bdd_type=""):
    out_dir  = get_route_image_dir_ag(route_id)
    safe_bdd = str(bdd_type).strip().lower().replace(" ", "_") or "dataset"
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.lineplot(x=range(1, len(history) + 1), y=history, color='purple', linewidth=2)
    plt.title(f'Évolution de f(x) = ω·K(x)+Σc_ij — Route : {format_route_id(route_id)} ({bdd_type})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Génération')
    plt.ylabel('f(x)  [ω·K(x) + Σ c_ij]')
    plt.tight_layout()
    path = out_dir / f"ag_cost_evolution_{safe_bdd}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Image sauvegardée : {path}")


def plot_solution_routes_ag(solution, route_id, customers_df, depots_df, bdd_type=""):
    out_dir  = get_route_image_dir_ag(route_id)
    safe_bdd = str(bdd_type).strip().lower().replace(" ", "_") or "dataset"
    route_customers = customers_df[customers_df["ROUTE_ID"] == route_id].set_index(
        "CUSTOMER_NUMBER", drop=False)
    route_depots = depots_df[depots_df["ROUTE_ID"] == route_id].reset_index(drop=True)
    if route_depots.empty or route_customers.empty:
        return None

    depot_lat = route_depots.loc[0, "DEPOT_LATITUDE"]
    depot_lon = route_depots.loc[0, "DEPOT_LONGITUDE"]

    plt.figure(figsize=(10, 8))
    for idx, route in enumerate(solution):
        if len(route) <= 2:
            continue
        lats = []
        lons = []
        for node in route:
            if node == 0:
                lats.append(depot_lat)
                lons.append(depot_lon)
            else:
                lats.append(route_customers.loc[node, "CUSTOMER_LATITUDE"])
                lons.append(route_customers.loc[node, "CUSTOMER_LONGITUDE"])
        plt.plot(lons, lats, marker="o", linewidth=1.8, markersize=4, label=f"Véhicule {idx + 1}")

    plt.scatter([depot_lon], [depot_lat], marker="s", s=120, label="Dépôt")
    plt.title(f"Routes optimisées AG — Route {format_route_id(route_id)} ({bdd_type})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"ag_optimized_routes_{safe_bdd}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Image sauvegardée : {path}")
    return path


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ROUTE_PETITE = 2939484.0
    num_trucks, customers, cost, demand = getData(ROUTE_PETITE, customersDf, depotsDf, trucksDf)

    best_sol, best_fit, history = genetic_algorithm(
        populationSize   = 50,
        numberOfTrucks   = num_trucks,
        truckCapacityKg  = truckKg,
        truckCapacityVol = truckVol,
        customersId      = customers,
        cost             = cost,
        demandForCustomer= demand,
        maxGenNumber     = 100,
        baseMutationRate = 0.10,
        omega            = OMEGA,
    )

    print("\n--- TABLEAU DE RÉSULTATS (PETITE TAILLE) ---")
    gerar_tabela_resultados(best_sol, cost, demand, truckKg, truckVol, ROUTE_PETITE)
    plot_evolucao_fitness(history, ROUTE_PETITE, "Petite")
    plot_solution_routes_ag(best_sol, ROUTE_PETITE, customersDf, depotsDf, "Petite")

    # ── Grande taille ────────────────────────────────────────────────
    customersDf = pd.read_excel("..\\database\\2_detail_table_customers.xls")
    depotsDf    = pd.read_excel("..\\database\\4_detail_table_depots.xls")
    trucksDf    = pd.read_excel("..\\database\\3_detail_table_vehicles.xls")

    ALL_ROUTES = sorted(customersDf["ROUTE_ID"].unique().tolist())

    for ROUTE_GRANDE in ALL_ROUTES:
        num_trucks_g, customers_g, cost_g, demand_g = getData(
            ROUTE_GRANDE, customersDf, depotsDf, trucksDf)
        best_sol_g, best_fit_g, history_g = genetic_algorithm(
            populationSize   = 50,
            numberOfTrucks   = num_trucks_g,
            truckCapacityKg  = truckKg,
            truckCapacityVol = truckVol,
            customersId      = customers_g,
            cost             = cost_g,
            demandForCustomer= demand_g,
            maxGenNumber     = 100,
            baseMutationRate = 0.10,
            omega            = OMEGA,
        )
        print(f"\n--- TABLEAU DE RÉSULTATS — Route {ROUTE_GRANDE} ---")
        gerar_tabela_resultados(best_sol_g, cost_g, demand_g, truckKg, truckVol, ROUTE_GRANDE)
        plot_evolucao_fitness(history_g, ROUTE_GRANDE, "Grande")
        plot_solution_routes_ag(best_sol_g, ROUTE_GRANDE, customersDf, depotsDf, "Grande")