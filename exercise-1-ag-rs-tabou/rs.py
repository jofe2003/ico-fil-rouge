# ══════════════════════════════════════════════════════════════════════
# ICO — Recuit Simulé (PTVFT) pour le VRP
# Centrale Lille — Fil Rouge
#
# Fonction de coût standardisée (formule du cours) :
#     f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij               (1)
#
#   - c_ij  = VEHICLE_VARIABLE_COST_KM_k × distance_km(i,j)
#   - K(x)  = nombre de véhicules utilisés
#   - ω     = VEHICLE_FIXED_COST_KM_k  (par véhicule utilisé)
#   - E     = ensemble des arcs de la solution x
#
# Corrections apportées :
#   1. compute_cost   → séparation explicite ω·K(x) et Σc_ij
#                       (avant : mélange coût + pénalités dans une seule valeur)
#   2. compute_objective → retourne f(x) PUR sans pénalités (pour comparaison)
#   3. Pénalités de contrainte séparées dans compute_penalties
#      (fenêtres de temps, capacité, SDVRP)
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import random
import math
import time
import copy
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
OMEGA = 5000.0   # euros ou unités de coût par véhicule utilisé
TRUCK_KG     = 20000.0  
TRUCK_VOL    = 20.0     
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR  = ROOT_DIR / "ico-fil-rouge" / "BaseDeDonnees" / "BaseExcel"
PETIT_DIR = ROOT_DIR / "ico-fil-rouge" / "BaseDeDonnees" / "PetitBaseExcel"
OUT_DIR   = Path("images_rs")

SA_PARAMS = dict(
    T0       = 2000.0,
    alpha    = 0.9990,
    T_min    = 0.5,
    max_iter = 60000,
    seed     = 2024,
)

SA_PARAMS_PETIT = dict(
    T0       = 500.0,
    alpha    = 0.9985,
    T_min    = 0.1,
    max_iter = 20000,
    seed     = 2024,
)

COLORS = [
    "#E63946", "#2A9D8F", "#F4A261", "#457B9D", "#A8DADC",
    "#264653", "#E9C46A", "#6A4C93", "#1982C4", "#8AC926", "#FF595E",
]

# ══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def load_all_data():
    df_customers   = pd.read_excel("..\\database\\2_detail_table_customers.xls")
    df_vehicles    = pd.read_excel("..\\database\\3_detail_table_vehicles.xls")
    df_depots      = pd.read_excel("..\\database\\4_detail_table_depots.xls")
    df_distances   = pd.read_excel("..\\database\\6_detail_table_cust_depots_distances.xls")
    df_constraints = pd.read_excel("..\\database\\5_detail_table_constraints_sdvrp.xls")
    return df_customers, df_vehicles, df_depots, df_distances, df_constraints


def load_petit_data():
    df_customers   = pd.read_excel("..\\database\\petit\\2_detail_table_customers_petit.xlsx")
    df_vehicles    = pd.read_excel("..\\database\\petit\\3_detail_table_vehicles_petit.xlsx")
    df_depots      = pd.read_excel("..\\database\\petit\\4_detail_table_depots_petit.xlsx")
    df_distances   = pd.read_excel("..\\database\\petit\\6_detail_table_cust_depots_distance_petit.xlsx")
    df_constraints = pd.read_excel("..\\database\\petit\\5_detail_table_constraints_sdvrp_petit.xlsx")
    return df_customers, df_vehicles, df_depots, df_distances, df_constraints


def extract_route(route_id, df_customers, df_vehicles, df_depots, df_distances, df_constraints):
    customers   = df_customers  [df_customers  ["ROUTE_ID"] == route_id].reset_index(drop=True).copy()
    vehicles    = df_vehicles   [df_vehicles   ["ROUTE_ID"] == route_id].reset_index(drop=True).copy()
    depot       = df_depots     [df_depots     ["ROUTE_ID"] == route_id].iloc[0]
    distances   = df_distances  [df_distances  ["ROUTE_ID"] == route_id]
    constraints = df_constraints[df_constraints["ROUTE_ID"] == route_id]

    customers["CUSTOMER_CODE"] = customers["CUSTOMER_CODE"].astype(str)
    vehicles ["VEHICLE_CODE"]  = vehicles ["VEHICLE_CODE"].astype(str)

    forbidden = set(
        zip(constraints["SDVRP_CONSTRAINT_CUSTOMER_CODE"].astype(str),
            constraints["SDVRP_CONSTRAINT_VEHICLE_CODE"].astype(str))
    )
    return customers, vehicles, depot, distances, forbidden


# ══════════════════════════════════════════════════════════════════════
# 2. MATRICES DE DISTANCES
# ══════════════════════════════════════════════════════════════════════

def build_distance_matrix(customers, distances, depot):
    n        = len(customers)
    codes    = customers["CUSTOMER_CODE"].tolist()
    code2idx = {c: i for i, c in enumerate(codes)}
    lats     = customers["CUSTOMER_LATITUDE"].values
    lons     = customers["CUSTOMER_LONGITUDE"].values

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a    = math.sin(dphi / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    SPEED    = 40.0
    dist_cc  = np.zeros((n, n));  time_cc  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d            = haversine(lats[i], lons[i], lats[j], lons[j])
            dist_cc[i][j] = d
            time_cc[i][j] = d / SPEED * 60.0

    dist_dc = np.zeros(n);  time_dc = np.zeros(n)
    dist_cd = np.zeros(n);  time_cd = np.zeros(n)

    for _, row in distances.iterrows():
        code = str(row["CUSTOMER_CODE"])
        if code not in code2idx:
            continue
        idx = code2idx[code]
        if str(row["DIRECTION"]).strip() == "DEPOT->CUSTOMER":
            dist_dc[idx] = row["DISTANCE_KM"]
            time_dc[idx] = row["TIME_DISTANCE_MIN"]
        else:
            dist_cd[idx] = row["DISTANCE_KM"]
            time_cd[idx] = row["TIME_DISTANCE_MIN"]

    return dist_cc, time_cc, dist_dc, time_dc, dist_cd, time_cd


# ══════════════════════════════════════════════════════════════════════
# 3. SOLUTION
# ══════════════════════════════════════════════════════════════════════

class Solution:
    def __init__(self, routes):
        self.routes = [list(r) for r in routes]
        self.cost   = None   # f(x) = ω·K(x) + Σ c_ij + pénalités


# ══════════════════════════════════════════════════════════════════════
# 4. FONCTIONS DE COÛT STANDARDISÉES
# ══════════════════════════════════════════════════════════════════════

def compute_arc_cost(solution, customers, vehicles, dist_cc, dist_dc, dist_cd):
    """
    Σ_{(i,j)∈E} c_ij  =  Σ_k  VARIABLE_COST_k × distance_k

    C'est la seconde composante de f(x) = ω·K(x) + Σ c_ij.
    """
    total = 0.0
    for k, route in enumerate(solution.routes):
        if not route:
            continue
        var_cost = vehicles.iloc[k]["VEHICLE_VARIABLE_COST_KM"]
        dist_r   = dist_dc[route[0]]
        for pos in range(len(route) - 1):
            i, j   = route[pos], route[pos + 1]
            dist_r += dist_cc[i][j]
        dist_r  += dist_cd[route[-1]]
        total   += var_cost * dist_r
    return total


def compute_vehicle_cost(solution, vehicles):
    """
    ω · K(x)  =  Σ_{k utilisé} FIXED_COST_k

    C'est la première composante de f(x) = ω·K(x) + Σ c_ij.
    Les véhicules sans client (route vide) ne sont PAS comptés.
    """
    total = 0.0
    for k, route in enumerate(solution.routes):
        if route:   # véhicule utilisé
            total += OMEGA
    return total


def compute_objective(solution, customers, vehicles,
                      dist_cc, dist_dc, dist_cd):
    """
    Objectif pur du cours (équation 1) :
        f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij

    SANS pénalités de contrainte.
    Utilisé pour comparer les solutions en fin d'optimisation.
    """
    return compute_vehicle_cost(solution, vehicles) + \
           compute_arc_cost(solution, customers, vehicles, dist_cc, dist_dc, dist_cd)


def compute_penalties(solution, customers, vehicles,
                      time_cc, time_dc, time_cd, forbidden):
    """
    Pénalités pour violations de contraintes (séparées de f(x)) :
      - Fenêtres de temps [a_i, b_i]
      - Capacité poids et volume
      - Contraintes SDVRP (interdictions client-véhicule)

    Ces pénalités guident le Recuit Simulé vers des solutions réalisables
    mais ne font PAS partie de la fonction objectif f(x) du cours.
    """
    PEN_FACTOR = 500.0   # calibré pour la base de données
    penalty    = 0.0

    for k, route in enumerate(solution.routes):
        if not route:
            continue
        veh      = vehicles.iloc[k]
        # cap_w    = veh["VEHICLE_TOTAL_WEIGHT_KG"]
        # cap_v    = veh["VEHICLE_TOTAL_VOLUME_M3"]
        # const
        cap_w = TRUCK_KG
        cap_v = TRUCK_VOL
        t_start  = veh["VEHICLE_AVAILABLE_TIME_FROM_MIN"]
        veh_code = veh["VEHICLE_CODE"]

        tw = 0.0;  tv = 0.0
        cur_t = t_start + time_dc[route[0]]

        for pos, i in enumerate(route):
            ci        = customers.iloc[i]
            tw       += ci["TOTAL_WEIGHT_KG"]
            tv       += ci["TOTAL_VOLUME_M3"]
            tw_from   = ci["CUSTOMER_TIME_WINDOW_FROM_MIN"]
            tw_to     = ci["CUSTOMER_TIME_WINDOW_TO_MIN"]
            tw_width  = max(1.0, tw_to - tw_from)

            if cur_t < tw_from:
                cur_t = tw_from
            if cur_t > tw_to:
                penalty += PEN_FACTOR * (cur_t - tw_to) / tw_width

            cur_t += ci["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"]

            if (ci["CUSTOMER_CODE"], veh_code) in forbidden:
                penalty += PEN_FACTOR

            if pos < len(route) - 1:
                cur_t += time_cc[i][route[pos + 1]]
            else:
                cur_t += time_cd[i]

        # Pénalités de capacité
        if tw > cap_w:
            penalty += PEN_FACTOR * (tw - cap_w) / cap_w
        if tv > cap_v:
            penalty += PEN_FACTOR * (tv - cap_v) / cap_v

    return penalty


def compute_cost(solution, customers, vehicles,
                 dist_cc, time_cc, dist_dc, time_dc, dist_cd, time_cd, forbidden):
    """
    Coût total utilisé par le Recuit Simulé :

        coût_RS(x) = f(x) + pénalités_contraintes
                   = [ω·K(x) + Σ c_ij] + pénalités

    La valeur retournée guide l'exploration ; seul f(x) est rapporté
    comme résultat final.
    """
    objective = compute_objective(
        solution, customers, vehicles, dist_cc, dist_dc, dist_cd)
    penalties = compute_penalties(
        solution, customers, vehicles, time_cc, time_dc, time_cd, forbidden)
    return objective + penalties


# ══════════════════════════════════════════════════════════════════════
# 5. SOLUTION INITIALE GLOUTONNE
# ══════════════════════════════════════════════════════════════════════

def greedy_solution(customers, vehicles, dist_dc, forbidden):
    n_v    = len(vehicles)
    routes = [[] for _ in range(n_v)]
    load_w = [0.0] * n_v
    load_v = [0.0] * n_v
    order  = customers["CUSTOMER_TIME_WINDOW_FROM_MIN"].argsort().tolist()

    for i in order:
        ci   = customers.iloc[i]
        w, v = ci["TOTAL_WEIGHT_KG"], ci["TOTAL_VOLUME_M3"]
        code = ci["CUSTOMER_CODE"]

        best_k, best_s = -1, float("inf")
        for k in range(n_v):
            veh = vehicles.iloc[k]
            if (code, veh["VEHICLE_CODE"]) in forbidden: continue
            # if load_w[k] + w > veh["VEHICLE_TOTAL_WEIGHT_KG"]: continue
            # if load_v[k] + v > veh["VEHICLE_TOTAL_VOLUME_M3"]: continue
            # const
            if load_w[k] + w > TRUCK_KG: continue
            if load_v[k] + v > TRUCK_VOL: continue
            if dist_dc[i] < best_s:
                best_s, best_k = dist_dc[i], k

        if best_k == -1:
            best_k = i % n_v
        routes[best_k].append(i)
        load_w[best_k] += w
        load_v[best_k] += v

    return Solution(routes)


# ══════════════════════════════════════════════════════════════════════
# 6. OPÉRATEURS DE VOISINAGE
# ══════════════════════════════════════════════════════════════════════

def op_swap_intra(sol):
    s          = Solution(sol.routes)
    candidates = [i for i, r in enumerate(s.routes) if len(r) >= 2]
    if not candidates:
        return s
    k    = random.choice(candidates)
    i, j = random.sample(range(len(s.routes[k])), 2)
    s.routes[k][i], s.routes[k][j] = s.routes[k][j], s.routes[k][i]
    return s


def op_relocate(sol):
    s  = Solution(sol.routes)
    ne = [i for i, r in enumerate(s.routes) if r]
    if not ne:
        return s
    k1 = random.choice(ne)
    k2 = random.randint(0, len(s.routes) - 1)
    p1 = random.randint(0, len(s.routes[k1]) - 1)
    c  = s.routes[k1].pop(p1)
    p2 = random.randint(0, len(s.routes[k2]))
    s.routes[k2].insert(p2, c)
    return s


def op_swap_inter(sol):
    s  = Solution(sol.routes)
    ne = [i for i, r in enumerate(s.routes) if r]
    if len(ne) < 2:
        return s
    k1, k2 = random.sample(ne, 2)
    i1      = random.randint(0, len(s.routes[k1]) - 1)
    i2      = random.randint(0, len(s.routes[k2]) - 1)
    s.routes[k1][i1], s.routes[k2][i2] = s.routes[k2][i2], s.routes[k1][i1]
    return s


def op_2opt(sol):
    s  = Solution(sol.routes)
    ne = [i for i, r in enumerate(s.routes) if len(r) >= 3]
    if not ne:
        return s
    k = random.choice(ne)
    r = s.routes[k]
    i = random.randint(0, len(r) - 2)
    j = random.randint(i + 1, len(r) - 1)
    r[i:j + 1] = r[i:j + 1][::-1]
    return s


OPERATORS = [op_swap_intra, op_relocate, op_swap_inter, op_2opt]


# ══════════════════════════════════════════════════════════════════════
# 7. RECUIT SIMULÉ
# ══════════════════════════════════════════════════════════════════════

def simulated_annealing(customers, vehicles, dist_cc, time_cc,
                        dist_dc, time_dc, dist_cd, time_cd, forbidden,
                        T0=8000.0, alpha=0.9975, T_min=0.5,
                        max_iter=60000, seed=42, verbose=True):
    """
    Recuit Simulé PTVFT minimisant :
        coût_RS(x) = f(x) + pénalités
                   = [ω·K(x) + Σ c_ij] + pénalités_contraintes

    Le résultat final reporté est f(x) pur (sans pénalités).
    """
    random.seed(seed);  np.random.seed(seed)

    def cost(s):
        return compute_cost(s, customers, vehicles,
                            dist_cc, time_cc, dist_dc, time_dc,
                            dist_cd, time_cd, forbidden)

    current      = greedy_solution(customers, vehicles, dist_dc, forbidden)
    current.cost = cost(current)
    best         = Solution(current.routes);  best.cost = current.cost

    # Calcul f(x) pur de la solution initiale (pour comparaison)
    obj_init = compute_objective(current, customers, vehicles, dist_cc, dist_dc, dist_cd)

    T           = T0
    hist_cost   = [best.cost]
    hist_temp   = [T]
    hist_accept = []
    n_accept    = 0
    n_improve   = 0
    t0          = time.time()

    if verbose:
        k_x    = sum(1 for r in current.routes if r)
        arc_c  = obj_init - sum(vehicles.iloc[k]["VEHICLE_FIXED_COST_KM"]
                                for k, r in enumerate(current.routes) if r)
        print(f"    Coût initial (glouton) :")
        print(f"      f(x) = ω·K(x) + Σc_ij = {obj_init:,.2f}  "
              f"[K(x)={k_x}, Σc_ij={arc_c:,.2f}]")
        print(f"      coût_RS (avec pénalités) = {current.cost:,.2f}")

    for it in range(1, max_iter + 1):
        if T < T_min:
            break

        neighbor      = random.choice(OPERATORS)(current)
        neighbor.cost = cost(neighbor)
        delta         = neighbor.cost - current.cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            n_accept += 1
            if current.cost < best.cost:
                best      = Solution(current.routes)
                best.cost = current.cost
                n_improve += 1

        T *= alpha
        hist_cost.append(best.cost)
        hist_temp.append(T)

        if it % 1000 == 0:
            hist_accept.append(n_accept / 1000)
            n_accept = 0

    elapsed  = time.time() - t0
    obj_final = compute_objective(best, customers, vehicles, dist_cc, dist_dc, dist_cd)

    stats = {
        "n_iter"    : it,
        "n_improve" : n_improve,
        "elapsed_s" : elapsed,
        "cost_init" : obj_init,      # f(x) pur initial
        "cost_final": obj_final,     # f(x) pur final  ← résultat rapporté
        "gain_pct"  : (obj_init - obj_final) / obj_init * 100 if obj_init else 0,
    }

    if verbose:
        k_x   = sum(1 for r in best.routes if r)
        fix_c = sum(vehicles.iloc[k]["VEHICLE_FIXED_COST_KM"]
                    for k, r in enumerate(best.routes) if r)
        arc_c = obj_final - fix_c
        print(f"    ✔ {it} itérations | {elapsed:.1f}s")
        print(f"      f(x) final = ω·K(x)+Σc_ij = {fix_c:,.2f} + {arc_c:,.2f} = {obj_final:,.2f}")
        print(f"      Gain vs. initial : {stats['gain_pct']:.1f}% | Améliorations : {n_improve}")

    return best, hist_cost, hist_temp, hist_accept, stats


# ══════════════════════════════════════════════════════════════════════
# 8. MÉTRIQUES DE SOLUTION
# ══════════════════════════════════════════════════════════════════════

def solution_metrics(solution, customers, vehicles,
                     dist_cc, dist_dc, dist_cd, forbidden):
    rows       = []
    total_dist = 0.0

    for k, route in enumerate(solution.routes):
        veh      = vehicles.iloc[k]
        veh_code = veh["VEHICLE_CODE"]
        # cap_w    = veh["VEHICLE_TOTAL_WEIGHT_KG"]
        # cap_v    = veh["VEHICLE_TOTAL_VOLUME_M3"]
        # const 
        cap_w = TRUCK_KG
        cap_v = TRUCK_VOL
        fix_cost = veh["VEHICLE_FIXED_COST_KM"]
        var_cost = veh["VEHICLE_VARIABLE_COST_KM"]

        if not route:
            rows.append({"Véhicule": f"V{k+1} ({veh_code})", "Clients": 0,
                         "Poids (kg)": 0, "Cap. Poids": cap_w,
                         "Volume (m³)": 0, "Cap. Vol.": cap_v,
                         "Distance (km)": 0, "ω (fixe)": 0,
                         "Σc_ij (variable)": 0,
                         "f_route(x)": 0,
                         "Violations": 0})
            continue

        rw = sum(customers.iloc[i]["TOTAL_WEIGHT_KG"] for i in route)
        rv = sum(customers.iloc[i]["TOTAL_VOLUME_M3"]  for i in route)
        d  = dist_dc[route[0]]
        for p in range(len(route) - 1):
            d += dist_cc[route[p]][route[p + 1]]
        d += dist_cd[route[-1]]
        total_dist += d

        arc_c    = var_cost * d
        f_route  = fix_cost + arc_c   # ω_k + Σ c_ij pour cette tournée

        viol = sum(1 for i in route
                   if (customers.iloc[i]["CUSTOMER_CODE"], veh_code) in forbidden)
        viol += int(rw > cap_w) + int(rv > cap_v)

        rows.append({
            "Véhicule"         : f"V{k+1} ({veh_code})",
            "Clients"          : len(route),
            "Poids (kg)"       : round(rw, 1),
            "Cap. Poids"       : round(cap_w, 1),
            "Volume (m³)"      : round(rv, 3),
            "Cap. Vol."        : round(cap_v, 3),
            "Distance (km)"    : round(d, 2),
            "ω (fixe)"         : 5000,
            "Σc_ij (variable)" : round(arc_c, 2),
            "f_route(x)"       : round(f_route, 2),   # ω_k + Σc_ij_k
            "Violations"       : viol,
        })

    df = pd.DataFrame(rows)
    return df, total_dist


# ══════════════════════════════════════════════════════════════════════
# 9. FIGURES
# ══════════════════════════════════════════════════════════════════════

def fig_convergence(hist_cost, hist_temp, hist_accept, stats, route_id, tag, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Convergence du Recuit Simulé | Route {route_id}\n"
        f"Tracé : coût_RS(x) = f(x) + pénalités  "
        f"[f(x) final = {stats['cost_final']:,.1f}]",
        fontsize=12, fontweight="bold")

    iters = range(len(hist_cost))

    ax = axes[0]
    ax.plot(iters, hist_cost, color="#E63946", linewidth=1.3)
    ax.axhline(stats["cost_final"], color="#2A9D8F", linestyle="--", linewidth=1.2,
               label=f"f(x)={stats['cost_final']:,.1f}")
    ax.set_title("Évolution du coût RS (meilleur)");  ax.set_xlabel("Itération")
    ax.set_ylabel("coût_RS = f(x)+pénalités");  ax.legend();  ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(len(hist_temp)), hist_temp, color="#457B9D", linewidth=1.3)
    ax.set_title("Refroidissement T ← α·T");  ax.set_xlabel("Itération")
    ax.set_ylabel("Température (log)");  ax.set_yscale("log");  ax.grid(alpha=0.3)

    ax = axes[2]
    if hist_accept:
        ax.plot(range(len(hist_accept)), hist_accept, color="#F4A261", linewidth=1.3)
    ax.set_title("Taux d'acceptation (1 000 iter.)");  ax.set_xlabel("Tranche ×1 000")
    ax.set_ylabel("Taux");  ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"rs_cost_temperature_acceptance_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight");  plt.close()
    return path


def fig_routes(solution, customers, vehicles, depot, stats, route_id, tag, out_dir):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(
        f"Routes optimisées — Route {route_id}\n"
        f"f(x) = ω·K(x)+Σc_ij = {stats['cost_final']:,.1f}  |  "
        f"Gain : {stats['gain_pct']:.1f}%  |  "
        f"Véhicules : {stats.get('n_used', '?')}/{len(vehicles)}",
        fontsize=10, fontweight="bold")

    dlat, dlon = depot["DEPOT_LATITUDE"], depot["DEPOT_LONGITUDE"]
    patches    = []

    for k, route in enumerate(solution.routes):
        if not route:
            continue
        color = COLORS[k % len(COLORS)]
        lats  = [dlat] + [customers.iloc[i]["CUSTOMER_LATITUDE"]  for i in route] + [dlat]
        lons  = [dlon] + [customers.iloc[i]["CUSTOMER_LONGITUDE"] for i in route] + [dlon]
        ax.plot(lons, lats, "-o", color=color, linewidth=1.6, markersize=4, alpha=0.85)
        patches.append(mpatches.Patch(color=color,
                       label=f"V{k+1} {vehicles.iloc[k]['VEHICLE_CODE']} ({len(route)} cl.)"))

    ax.plot(dlon, dlat, "s", color="#1D3557", markersize=14, zorder=5)
    ax.annotate("Dépôt", (dlon, dlat), xytext=(8, 6),
                textcoords="offset points", fontsize=9, fontweight="bold")
    ax.legend(handles=patches, loc="upper left", fontsize=7.5, ncol=2)
    ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude");  ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"rs_optimized_routes_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight");  plt.close()
    return path

def save_route_table(df_metrics, stats, route_id, tag, route_out_dir):
    """Salva a tabela de métricas da rota em route_out_dir/tableau_{tag}.png"""
    fig_h   = 1.5 + len(df_metrics) * 0.5
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=df_metrics.values,
                   colLabels=df_metrics.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2A9D8F')
            cell.set_text_props(weight='bold', color='white')

    plt.title(
        f"Métriques Route {route_id} — {tag.upper()}\n"
        f"f(x) = {stats['cost_final']:,.1f}  |  Gain : {stats['gain_pct']:.1f}%",
        fontsize=10, fontweight='bold', pad=12
    )
    plt.tight_layout()
    path = route_out_dir / f"tableau_{tag}.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Tableau sauvegardé : {path}")
    return path

def save_summary_table(all_stats, tag, out_dir):
    df = pd.DataFrame(all_stats)
    if df.empty:
        return None

    df_out = df.rename(columns={
        "route_id"   : "Route ID",
        "n_clients"  : "Clients",
        "n_vehicles" : "Véhicules",
        "n_used"     : "Véh. utilisés",
        "cost_init"  : "f(x) initial",
        "cost_final" : "f(x) final",
        "gain_pct"   : "Gain (%)",
        "total_dist" : "Distance (km)",
        "elapsed_s"  : "Temps (s)",
        "n_iter"     : "Itérations",
        "n_improve"  : "Améliorations",
        "n_violations": "Violations",
    })

    route_id = str(df_out["Route ID"].iloc[0])

    for col in ["f(x) initial", "f(x) final", "Distance (km)", "Gain (%)", "Temps (s)"]:
        if col in df_out:
            df_out[col] = df_out[col].round(2)

    path_csv = out_dir / f"tableau_{tag}.csv"
    df_out.to_csv(path_csv, index=False, sep=";", decimal=",")
    print(f"\n  Tableau sauvegardé : {path_csv}")

    fig_h   = 1.5 + len(df_out) * 0.5
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=df_out.values,
                   colLabels=df_out.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2A9D8F')
            cell.set_text_props(weight='bold', color='white')
    plt.title(f"Résultats Recuit Simulé — {tag.upper()}",
              fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    path_img = out_dir / f"route_{route_id}" / f"tableau_{tag}.png"
    plt.savefig(path_img, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Tabela salva: {path_img}")
    # ─────────────────────────────────────────────────────────────
    return path_csv


# ══════════════════════════════════════════════════════════════════════
# 10. PIPELINE PAR ROUTE
# ══════════════════════════════════════════════════════════════════════

def run_route(route_id, df_customers, df_vehicles, df_depots, df_distances,
              df_constraints, tag, out_dir, params, verbose=True):
    route_out_dir = out_dir / f"route_{int(float(route_id)) if float(route_id).is_integer() else str(route_id).replace('.', '_')}"
    route_out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n  ── Route {route_id} ──")

    customers, vehicles, depot, distances, forbidden = extract_route(
        route_id, df_customers, df_vehicles, df_depots, df_distances, df_constraints)

    n_clients  = len(customers)
    n_vehicles = len(vehicles)
    if verbose:
        print(f"     {n_clients} clients | {n_vehicles} véhicules | {len(forbidden)} contraintes SDVRP")

    dist_cc, time_cc, dist_dc, time_dc, dist_cd, time_cd = \
        build_distance_matrix(customers, distances, depot)

    best, hist_cost, hist_temp, hist_accept, stats = simulated_annealing(
        customers, vehicles, dist_cc, time_cc,
        dist_dc, time_dc, dist_cd, time_cd, forbidden,
        verbose=verbose, **params)

    df_metrics, total_dist = solution_metrics(
        best, customers, vehicles, dist_cc, dist_dc, dist_cd, forbidden)

    n_used       = sum(1 for r in best.routes if r)
    n_violations = df_metrics["Violations"].sum()
    stats.update({
        "route_id"    : route_id,
        "n_clients"   : n_clients,
        "n_vehicles"  : n_vehicles,
        "n_used"      : n_used,
        "total_dist"  : total_dist,
        "n_violations": n_violations,
    })

    fig_convergence(hist_cost, hist_temp, hist_accept, stats, route_id, tag, route_out_dir)
    fig_routes(best, customers, vehicles, depot, stats, route_id, tag, route_out_dir)
    save_route_table(df_metrics, stats, route_id, tag, route_out_dir)

    return best, hist_cost, stats, df_metrics


def run_bdd(tag, route_ids, df_customers, df_vehicles, df_depots,
            df_distances, df_constraints, params):
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  BDD {tag.upper()} — {len(route_ids)} route(s)")
    print(f"{'═'*65}")

    all_stats     = []
    all_metrics   = []
    all_hist_cost = []

    for rid in route_ids:
        best, hist_cost, stats, df_metrics = run_route(
            rid, df_customers, df_vehicles, df_depots,
            df_distances, df_constraints, tag, out_dir, params)
        all_stats.append(stats)
        all_metrics.append(df_metrics)
        all_hist_cost.append(hist_cost)

    save_summary_table(all_stats, tag, out_dir)
    print(f"\n  ✅ BDD {tag} terminée. Images dans : {out_dir}")
    return all_stats


# ══════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  VRP — Recuit Simulé | ICO Centrale Lille")
    print("  Fonction objectif : f(x) = ω·K(x) + Σ c_ij")
    print("=" * 65)

    # BDD Petite taille
    print("\n[1] Chargement BDD petite taille...")
    df_cp, df_vp, df_dp, df_distp, df_conp = load_petit_data()
    PETIT_ROUTES = sorted(df_cp["ROUTE_ID"].unique().tolist())

    stats_petit = run_bdd(
        tag="petit", route_ids=PETIT_ROUTES,
        df_customers=df_cp, df_vehicles=df_vp, df_depots=df_dp,
        df_distances=df_distp, df_constraints=df_conp,
        params=SA_PARAMS_PETIT,
    )

    # BDD Grande taille
    print("\n[2] Chargement BDD grande taille...")
    df_c, df_v, df_d, df_dist, df_con = load_all_data()
    ALL_ROUTES = sorted(df_c["ROUTE_ID"].unique().tolist())

    stats_grand = run_bdd(
        tag="grand", route_ids=ALL_ROUTES,
        df_customers=df_c, df_vehicles=df_v, df_depots=df_d,
        df_distances=df_dist, df_constraints=df_con,
        params=SA_PARAMS,
    )

    print("\n" + "=" * 65)
    print("  Terminé. f(x) rapporté = ω·K(x) + Σ c_ij (sans pénalités).")
    print("=" * 65)
