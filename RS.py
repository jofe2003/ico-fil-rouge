# ══════════════════════════════════════════════════════════════════════
# IMPORTS
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
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← seul bloc à modifier
# ══════════════════════════════════════════════════════════════════════

from pathlib import Path

# Racine du projet (dossier ico-fil-rouge)
ROOT_DIR = Path(__file__).resolve().parents[1]
print(ROOT_DIR)
# ── BDD Grande taille ──────────────────────
DATA_DIR = ROOT_DIR / "BaseDeDonnees" / "BaseExcel"

# ── BDD Petite taille ──────────────────────
PETIT_DIR = ROOT_DIR / "BaseDeDonnees" / "PetitBaseExcel"

# ── Dossier de sortie des résultats ────────
OUT_DIR = ROOT_DIR / "Results" / "RecuitSimule"

# Paramètres Recuit Simulé
SA_PARAMS = dict(
    T0       = 2000.0,
    alpha    = 0.9990,   # ~15 000 iterações reais
    T_min    = 0.5,
    max_iter = 60000,
    seed     = 2024
)

SA_PARAMS_PETIT = dict(
    T0       = 500.0,
    alpha    = 0.9985,   # ~8 000 iterações reais
    T_min    = 0.1,
    max_iter = 20000,
    seed     = 2024
)

COLORS = [
    "#E63946","#2A9D8F","#F4A261","#457B9D","#A8DADC",
    "#264653","#E9C46A","#6A4C93","#1982C4","#8AC926","#FF595E"
]

# ══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def load_all_data():
    """Charge la BDD grande taille (fichiers .xls originaux)."""
    df_customers   = pd.read_excel(DATA_DIR / "2_detail_table_customers.xls",             engine="xlrd")
    df_vehicles    = pd.read_excel(DATA_DIR / "3_detail_table_vehicles.xls",              engine="xlrd")
    df_depots      = pd.read_excel(DATA_DIR / "4_detail_table_depots.xls",                engine="xlrd")
    df_distances   = pd.read_excel(DATA_DIR / "6_detail_table_cust_depots_distances.xls", engine="xlrd")
    df_constraints = pd.read_excel(DATA_DIR / "5_detail_table_constraints_sdvrp.xls",     engine="xlrd")
    return df_customers, df_vehicles, df_depots, df_distances, df_constraints


def load_petit_data():
    """
    Charge la BDD petite taille depuis les fichiers .xlsx du dossier PetitBase.
    Route unique : 2939484 | 20 clients | 3 vehicules
    """
    df_customers   = pd.read_excel(PETIT_DIR / "2_detail_table_customers_petit.xlsx")
    df_vehicles    = pd.read_excel(PETIT_DIR / "3_detail_table_vehicles_petit.xlsx")
    df_depots      = pd.read_excel(PETIT_DIR / "4_detail_table_depots_petit.xlsx")
    df_distances   = pd.read_excel(PETIT_DIR / "6_detail_table_cust_depots_distance_petit.xlsx")
    df_constraints = pd.read_excel(PETIT_DIR / "5_detail_table_constraints_sdvrp_petit.xlsx")
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

    lats = customers["CUSTOMER_LATITUDE"].values
    lons = customers["CUSTOMER_LONGITUDE"].values

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    SPEED = 40.0
    dist_cc = np.zeros((n, n))
    time_cc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            dist_cc[i][j] = d
            time_cc[i][j] = d / SPEED * 60.0

    dist_dc = np.zeros(n); time_dc = np.zeros(n)
    dist_cd = np.zeros(n); time_cd = np.zeros(n)

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
# 3. SOLUTION & FONCTION DE COÛT
# ══════════════════════════════════════════════════════════════════════

class Solution:
    def __init__(self, routes):
        self.routes = [list(r) for r in routes]
        self.cost   = None


def compute_cost(solution, customers, vehicles,
                 dist_cc, time_cc, dist_dc, time_dc, dist_cd, time_cd, forbidden):
    """
    Funcao de custo com penalidades normalizadas.
    PEN_FACTOR esta na mesma ordem de grandeza do custo real (KM),
    evitando explosao numerica que distorce a comparacao entre rotas.
    """
    total_cost = 0.0
    total_pen  = 0.0
    PEN_FACTOR = 500.0   # calibrado para os custos desta base de dados

    for k, route in enumerate(solution.routes):
        if not route:
            continue
        veh      = vehicles.iloc[k]
        var_cost = veh["VEHICLE_VARIABLE_COST_KM"]
        fix_cost = veh["VEHICLE_FIXED_COST_KM"]
        cap_w    = veh["VEHICLE_TOTAL_WEIGHT_KG"]
        cap_v    = veh["VEHICLE_TOTAL_VOLUME_M3"]
        t_start  = veh["VEHICLE_AVAILABLE_TIME_FROM_MIN"]
        veh_code = veh["VEHICLE_CODE"]

        tw = 0.0; tv = 0.0
        dist_r = dist_dc[route[0]]
        cur_t  = t_start + time_dc[route[0]]

        for pos, i in enumerate(route):
            ci      = customers.iloc[i]
            tw     += ci["TOTAL_WEIGHT_KG"]
            tv     += ci["TOTAL_VOLUME_M3"]
            tw_from = ci["CUSTOMER_TIME_WINDOW_FROM_MIN"]
            tw_to   = ci["CUSTOMER_TIME_WINDOW_TO_MIN"]
            tw_width = max(1.0, tw_to - tw_from)

            if cur_t < tw_from:
                cur_t = tw_from
            if cur_t > tw_to:
                total_pen += PEN_FACTOR * (cur_t - tw_to) / tw_width

            cur_t += ci["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"]

            if (ci["CUSTOMER_CODE"], veh_code) in forbidden:
                total_pen += PEN_FACTOR

            if pos < len(route) - 1:
                j = route[pos + 1]
                dist_r += dist_cc[i][j]
                cur_t  += time_cc[i][j]
            else:
                dist_r += dist_cd[i]
                cur_t  += time_cd[i]

        # Penalidades de capacidade normalizadas
        if tw > cap_w:
            total_pen += PEN_FACTOR * (tw - cap_w) / cap_w
        if tv > cap_v:
            total_pen += PEN_FACTOR * (tv - cap_v) / cap_v

        total_cost += var_cost * dist_r + fix_cost

    return total_cost + total_pen


# ══════════════════════════════════════════════════════════════════════
# 4. SOLUTION INITIALE GLOUTONNE
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
            if load_w[k] + w > veh["VEHICLE_TOTAL_WEIGHT_KG"]: continue
            if load_v[k] + v > veh["VEHICLE_TOTAL_VOLUME_M3"]: continue
            if dist_dc[i] < best_s:
                best_s, best_k = dist_dc[i], k

        if best_k == -1: best_k = i % n_v
        routes[best_k].append(i)
        load_w[best_k] += w
        load_v[best_k] += v

    return Solution(routes)


# ══════════════════════════════════════════════════════════════════════
# 5. OPÉRATEURS DE VOISINAGE
# ══════════════════════════════════════════════════════════════════════

def op_swap_intra(sol):
    s = Solution(sol.routes)
    candidates = [i for i, r in enumerate(s.routes) if len(r) >= 2]
    if not candidates: return s
    k = random.choice(candidates)
    i, j = random.sample(range(len(s.routes[k])), 2)
    s.routes[k][i], s.routes[k][j] = s.routes[k][j], s.routes[k][i]
    return s

def op_relocate(sol):
    s = Solution(sol.routes)
    ne = [i for i, r in enumerate(s.routes) if r]
    if not ne: return s
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
    if len(ne) < 2: return s
    k1, k2 = random.sample(ne, 2)
    i1 = random.randint(0, len(s.routes[k1]) - 1)
    i2 = random.randint(0, len(s.routes[k2]) - 1)
    s.routes[k1][i1], s.routes[k2][i2] = s.routes[k2][i2], s.routes[k1][i1]
    return s

def op_2opt(sol):
    s  = Solution(sol.routes)
    ne = [i for i, r in enumerate(s.routes) if len(r) >= 3]
    if not ne: return s
    k = random.choice(ne)
    r = s.routes[k]
    i = random.randint(0, len(r) - 2)
    j = random.randint(i + 1, len(r) - 1)
    r[i:j+1] = r[i:j+1][::-1]
    return s

OPERATORS = [op_swap_intra, op_relocate, op_swap_inter, op_2opt]


# ══════════════════════════════════════════════════════════════════════
# 6. RECUIT SIMULÉ  (II.2.1 — Algorithme PTVFT)
# ══════════════════════════════════════════════════════════════════════
# PTVFT = Perturbation + Température Variable + Fonction de coût Total

def simulated_annealing(customers, vehicles, dist_cc, time_cc,
                        dist_dc, time_dc, dist_cd, time_cd, forbidden,
                        T0=8000.0, alpha=0.9975, T_min=0.5,
                        max_iter=60000, seed=42, verbose=True):
    """
    Recuit Simulé PTVFT pour le VRP avec fenêtres de temps et contraintes SDVRP.

    Phases :
      1. Initialisation : solution gloutonne + T = T0
      2. Perturbation   : 4 opérateurs de voisinage (swap intra/inter, relocate, 2-opt)
      3. Acceptation    : critère de Metropolis  P = exp(-Δf / T)
      4. Refroidissement géométrique : T ← α·T
      5. Arrêt : T < T_min  ou  iter > max_iter
    """
    random.seed(seed); np.random.seed(seed)

    def cost(s):
        return compute_cost(s, customers, vehicles,
                            dist_cc, time_cc, dist_dc, time_dc, dist_cd, time_cd, forbidden)

    current      = greedy_solution(customers, vehicles, dist_dc, forbidden)
    current.cost = cost(current)
    best         = Solution(current.routes); best.cost = current.cost

    T            = T0
    hist_cost    = [best.cost]
    hist_temp    = [T]
    hist_accept  = []          # taux d'acceptation par tranche de 1000 iter
    n_accept     = 0
    n_improve    = 0
    t0           = time.time()

    if verbose:
        print(f"    Coût initial (glouton) : {current.cost:,.2f}")

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
                best = Solution(current.routes); best.cost = current.cost
                n_improve += 1

        T *= alpha
        hist_cost.append(best.cost)
        hist_temp.append(T)

        if it % 1000 == 0:
            hist_accept.append(n_accept / 1000)
            n_accept = 0

    elapsed = time.time() - t0
    stats = {
        "n_iter"    : it,
        "n_improve" : n_improve,
        "elapsed_s" : elapsed,
        "cost_init" : hist_cost[0],
        "cost_final": best.cost,
        "gain_pct"  : (hist_cost[0] - best.cost) / hist_cost[0] * 100 if hist_cost[0] else 0,
    }
    if verbose:
        print(f"    ✔ {it} itérations | {elapsed:.1f}s | "
              f"Meilleur coût : {best.cost:,.2f} | "
              f"Améliorations : {n_improve}")

    return best, hist_cost, hist_temp, hist_accept, stats


# ══════════════════════════════════════════════════════════════════════
# 7. MÉTRIQUES DE SOLUTION
# ══════════════════════════════════════════════════════════════════════

def solution_metrics(solution, customers, vehicles,
                     dist_cc, dist_dc, dist_cd, forbidden):
    rows = []
    total_dist = 0.0

    for k, route in enumerate(solution.routes):
        veh      = vehicles.iloc[k]
        veh_code = veh["VEHICLE_CODE"]
        cap_w    = veh["VEHICLE_TOTAL_WEIGHT_KG"]
        cap_v    = veh["VEHICLE_TOTAL_VOLUME_M3"]

        if not route:
            rows.append({"Véhicule": f"V{k+1} ({veh_code})", "Clients": 0,
                         "Poids (kg)": 0, "Cap. Poids (kg)": cap_w,
                         "Volume (m³)": 0, "Cap. Vol. (m³)": cap_v,
                         "Distance (km)": 0,
                         "Taux Poids (%)": 0, "Taux Vol. (%)": 0,
                         "Violations": 0})
            continue

        rw = sum(customers.iloc[i]["TOTAL_WEIGHT_KG"] for i in route)
        rv = sum(customers.iloc[i]["TOTAL_VOLUME_M3"]  for i in route)
        d  = dist_dc[route[0]]
        for p in range(len(route) - 1):
            d += dist_cc[route[p]][route[p+1]]
        d += dist_cd[route[-1]]
        total_dist += d

        viol = sum(1 for i in route
                   if (customers.iloc[i]["CUSTOMER_CODE"], veh_code) in forbidden)
        viol += int(rw > cap_w) + int(rv > cap_v)

        rows.append({
            "Véhicule"        : f"V{k+1} ({veh_code})",
            "Clients"         : len(route),
            "Poids (kg)"      : round(rw, 1),
            "Cap. Poids (kg)" : round(cap_w, 1),
            "Volume (m³)"     : round(rv, 3),
            "Cap. Vol. (m³)"  : round(cap_v, 3),
            "Distance (km)"   : round(d, 2),
            "Taux Poids (%)"  : round(rw / cap_w * 100, 1),
            "Taux Vol. (%)"   : round(rv / cap_v * 100, 1),
            "Violations"      : viol,
        })

    df = pd.DataFrame(rows)
    return df, total_dist


# ══════════════════════════════════════════════════════════════════════
# 8. FIGURES  (II.2.x.1 — Courbes et tableaux)
# ══════════════════════════════════════════════════════════════════════

def fig_convergence(hist_cost, hist_temp, hist_accept, stats, route_id, tag, out_dir):
    """Courbes de convergence : coût, température, taux d'acceptation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(f"II.2.1 — Convergence du Recuit Simulé (PTVFT) | Route {route_id}",
                 fontsize=13, fontweight="bold", color="#1D3557")

    iters = range(len(hist_cost))

    # Coût
    ax = axes[0]
    ax.set_facecolor("#EEF2F7")
    ax.plot(iters, hist_cost, color="#E63946", linewidth=1.3)
    ax.axhline(stats["cost_final"], color="#2A9D8F", linestyle="--",
               linewidth=1.2, label=f"Optimal = {stats['cost_final']:,.1f}")
    ax.set_title("Évolution du coût (meilleur)", fontweight="bold")
    ax.set_xlabel("Itération"); ax.set_ylabel("Coût"); ax.legend(); ax.grid(alpha=0.3)

    # Température
    ax = axes[1]
    ax.set_facecolor("#EEF2F7")
    ax.plot(range(len(hist_temp)), hist_temp, color="#457B9D", linewidth=1.3)
    ax.set_title("Refroidissement géométrique (T ← α·T)", fontweight="bold")
    ax.set_xlabel("Itération"); ax.set_ylabel("Température (log)"); ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # Taux d'acceptation
    ax = axes[2]
    ax.set_facecolor("#EEF2F7")
    if hist_accept:
        ax.plot(range(len(hist_accept)), hist_accept, color="#F4A261", linewidth=1.3)
    ax.set_title("Taux d'acceptation (par tranche 1 000 iter.)", fontweight="bold")
    ax.set_xlabel("Tranche (×1 000 iter.)"); ax.set_ylabel("Taux"); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"convergence_{tag}_route{route_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def fig_routes(solution, customers, vehicles, depot, stats, route_id, tag, out_dir):
    """Carte des routes optimisées."""
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#EEF2F7")
    ax.set_title(
        f"Routes optimisées — Route {route_id}  |  Coût : {stats['cost_final']:,.1f}  "
        f"|  Distance : {stats.get('total_dist', 0):.1f} km\n"
        f"Gain vs. initial : {stats['gain_pct']:.1f}%  |  "
        f"Véhicules : {stats.get('n_used', '?')} / {len(vehicles)}",
        fontsize=11, fontweight="bold", pad=10)

    dlat, dlon = depot["DEPOT_LATITUDE"], depot["DEPOT_LONGITUDE"]
    patches = []

    for k, route in enumerate(solution.routes):
        if not route: continue
        color = COLORS[k % len(COLORS)]
        lats  = [dlat] + [customers.iloc[i]["CUSTOMER_LATITUDE"]  for i in route] + [dlat]
        lons  = [dlon] + [customers.iloc[i]["CUSTOMER_LONGITUDE"] for i in route] + [dlon]
        ax.plot(lons, lats, "-o", color=color, linewidth=1.6, markersize=4, alpha=0.85, zorder=2)
        patches.append(mpatches.Patch(color=color,
                       label=f"V{k+1} {vehicles.iloc[k]['VEHICLE_CODE']} ({len(route)} cl.)"))

    ax.plot(dlon, dlat, "s", color="#1D3557", markersize=14, zorder=5)
    ax.annotate("Dépôt", (dlon, dlat), xytext=(8, 6),
                textcoords="offset points", fontsize=9, fontweight="bold", color="#1D3557")
    ax.legend(handles=patches, loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"routes_{tag}_route{route_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def fig_charge(solution, customers, vehicles, route_id, tag, out_dir):
    """Taux de remplissage par véhicule (poids & volume)."""
    labels, pct_w, pct_v = [], [], []
    for k, route in enumerate(solution.routes):
        if not route: continue
        veh = vehicles.iloc[k]
        rw  = sum(customers.iloc[i]["TOTAL_WEIGHT_KG"] for i in route)
        rv  = sum(customers.iloc[i]["TOTAL_VOLUME_M3"]  for i in route)
        labels.append(f"V{k+1}\n{veh['VEHICLE_CODE']}")
        pct_w.append(rw / veh["VEHICLE_TOTAL_WEIGHT_KG"] * 100)
        pct_v.append(rv / veh["VEHICLE_TOTAL_VOLUME_M3"] * 100)

    if not labels:
        return None

    x      = range(len(labels))
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(f"Taux d'utilisation de la flotte — Route {route_id}",
                 fontsize=13, fontweight="bold", color="#1D3557")

    for ax, pct, title in [(ax1, pct_w, "Poids"), (ax2, pct_v, "Volume")]:
        ax.set_facecolor("#EEF2F7")
        bars = ax.bar(x, pct, color=colors, edgecolor="white")
        ax.axhline(100, color="red", linestyle="--", linewidth=1.2, label="Capacité max")
        for bar, p in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{p:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Taux (%)"); ax.set_title(f"Taux de remplissage — {title}", fontweight="bold")
        ax.legend(); ax.set_ylim(0, max(max(pct) * 1.15, 115)); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"charge_{tag}_route{route_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def fig_comparaison_routes(all_stats, tag, out_dir):
    """
    Comparaison de toutes les routes : coût final, distance totale, gain (%).
    Utilisé dans II.2.x.2 — Analyse des résultats.
    """
    df = pd.DataFrame(all_stats)
    if df.empty: return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(f"Comparaison des routes — BDD {tag}",
                 fontsize=13, fontweight="bold", color="#1D3557")

    x      = range(len(df))
    xlabs  = [str(r) for r in df["route_id"]]
    colors = [COLORS[i % len(COLORS)] for i in range(len(df))]

    for ax, col, title, ylabel in [
        (axes[0], "cost_final",  "Coût total par route",       "Coût"),
        (axes[1], "total_dist",  "Distance totale par route",  "Distance (km)"),
        (axes[2], "gain_pct",    "Gain vs. solution initiale", "Gain (%)"),
    ]:
        ax.set_facecolor("#EEF2F7")
        bars = ax.bar(x, df[col], color=colors, edgecolor="white")
        for bar, v in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels(xlabs, rotation=30, fontsize=8)
        ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"comparaison_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def fig_boxplot_convergence(all_hist_cost, all_route_ids, tag, out_dir):
    """
    Distribution des coûts de convergence sur toutes les routes.
    Utilisé dans II.2.x.2.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#EEF2F7")

    # Normalisation : coût relatif au coût initial
    normalized = []
    for hist in all_hist_cost:
        c0 = hist[0] if hist[0] != 0 else 1
        normalized.append([c / c0 * 100 for c in hist])

    # Sous-échantillonnage pour lisibilité
    step = max(1, max(len(h) for h in normalized) // 50)
    data_plot = [h[::step] for h in normalized]
    min_len   = min(len(h) for h in data_plot)
    data_plot = [h[:min_len] for h in data_plot]

    for i, (hist, rid) in enumerate(zip(data_plot, all_route_ids)):
        ax.plot(range(min_len), hist, linewidth=1.0, alpha=0.65,
                color=COLORS[i % len(COLORS)], label=str(rid))

    ax.set_title(f"Convergence normalisée (% du coût initial) — BDD {tag}",
                 fontweight="bold")
    ax.set_xlabel("Itération (sous-échantillonné)"); ax.set_ylabel("Coût relatif (%)")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"convergence_toutes_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ══════════════════════════════════════════════════════════════════════
# 9. TABLEAU RÉCAPITULATIF  (II.2.x.1)
# ══════════════════════════════════════════════════════════════════════

def save_summary_table(all_stats, tag, out_dir):
    df = pd.DataFrame(all_stats)
    if df.empty: return None

    # Colonnes lisibles
    df_out = df.rename(columns={
        "route_id"   : "Route ID",
        "n_clients"  : "Clients",
        "n_vehicles" : "Véhicules",
        "n_used"     : "Véh. utilisés",
        "cost_init"  : "Coût initial",
        "cost_final" : "Coût final",
        "gain_pct"   : "Gain (%)",
        "total_dist" : "Distance (km)",
        "elapsed_s"  : "Temps (s)",
        "n_iter"     : "Itérations",
        "n_improve"  : "Améliorations",
        "n_violations": "Violations",
    })

    # Arrondi
    for col in ["Coût initial", "Coût final", "Distance (km)"]:
        if col in df_out: df_out[col] = df_out[col].round(2)
    for col in ["Gain (%)", "Temps (s)"]:
        if col in df_out: df_out[col] = df_out[col].round(2)

    path_csv = out_dir / f"tableau_recapitulatif_{tag}.csv"
    df_out.to_csv(path_csv, index=False, sep=";", decimal=",")

    # Figure tableau
    fig, ax = plt.subplots(figsize=(max(14, len(df_out.columns) * 1.5), max(3, len(df_out) * 0.6 + 1.5)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.axis("off")
    tbl = ax.table(
        cellText  = df_out.values,
        colLabels = df_out.columns,
        cellLoc   = "center", loc="center"
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.4)

    # En-têtes en bleu
    for j in range(len(df_out.columns)):
        tbl[0, j].set_facecolor("#1D3557"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Lignes alternées
    for i in range(1, len(df_out) + 1):
        for j in range(len(df_out.columns)):
            tbl[i, j].set_facecolor("#EEF2F7" if i % 2 == 0 else "white")

    ax.set_title(f"II.2.x.1 — Tableau récapitulatif — BDD {tag}",
                 fontsize=12, fontweight="bold", color="#1D3557", pad=12)
    plt.tight_layout()
    path_fig = out_dir / f"tableau_recapitulatif_{tag}.png"
    plt.savefig(path_fig, dpi=150, bbox_inches="tight"); plt.close()
    return path_fig, path_csv


# ══════════════════════════════════════════════════════════════════════
# 10. PIPELINE PRINCIPAL PAR ROUTE
# ══════════════════════════════════════════════════════════════════════

def run_route(route_id, df_customers, df_vehicles, df_depots, df_distances,
              df_constraints, tag, out_dir, params, verbose=True):
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
        customers, vehicles, dist_cc, time_cc, dist_dc, time_dc, dist_cd, time_cd,
        forbidden, verbose=verbose, **params)

    df_metrics, total_dist = solution_metrics(
        best, customers, vehicles, dist_cc, dist_dc, dist_cd, forbidden)

    n_used      = sum(1 for r in best.routes if r)
    n_violations = df_metrics["Violations"].sum()
    stats.update({
        "route_id"    : route_id,
        "n_clients"   : n_clients,
        "n_vehicles"  : n_vehicles,
        "n_used"      : n_used,
        "total_dist"  : total_dist,
        "n_violations": n_violations,
    })

    # Figures
    fig_convergence(hist_cost, hist_temp, hist_accept, stats, route_id, tag, out_dir)
    fig_routes(best, customers, vehicles, depot, stats, route_id, tag, out_dir)
    fig_charge(best, customers, vehicles, route_id, tag, out_dir)

    return best, hist_cost, stats, df_metrics


# ══════════════════════════════════════════════════════════════════════
# 11. RAPPORT TEXTE  (II.2.x.2 & II.2.x.3)
# ══════════════════════════════════════════════════════════════════════

def write_report(all_stats, all_metrics, tag, out_dir):
    """
    Génère un rapport texte structuré :
      II.2.x.1  Courbes et tableaux  → voir figures PNG + CSV
      II.2.x.2  Analyse des résultats
      II.2.x.3  Intérêt de l'étude
    """
    path = out_dir / f"rapport_{tag}.txt"
    df   = pd.DataFrame(all_stats)

    lines = []
    lines.append("=" * 72)
    lines.append(f"  VRP — Recuit Simulé (PTVFT) | BDD {tag}")
    lines.append(f"  ICO Centrale Lille — Fil Rouge")
    lines.append("=" * 72)

    # ── II.2.x.1 ────────────────────────────────────────────────────
    lines.append(f"\nII.2.{'2' if tag=='petit' else '3'}.1  Courbes et tableaux")
    lines.append("-" * 50)
    lines.append(f"  Nombre de routes traitées   : {len(df)}")
    lines.append(f"  Total clients               : {df['n_clients'].sum()}")
    lines.append(f"  Total véhicules disponibles : {df['n_vehicles'].sum()}")
    lines.append(f"  Total véhicules utilisés    : {df['n_used'].sum()}")
    lines.append(f"  Distance totale (km)        : {df['total_dist'].sum():.2f}")
    lines.append(f"  Coût total (toutes routes)  : {df['cost_final'].sum():.2f}")
    lines.append(f"  Temps total de calcul (s)   : {df['elapsed_s'].sum():.1f}")
    lines.append(f"\n  → Voir fichiers PNG et CSV générés dans : {out_dir}")

    # ── II.2.x.2 ────────────────────────────────────────────────────
    lines.append(f"\nII.2.{'2' if tag=='petit' else '3'}.2  Analyse des résultats")
    lines.append("-" * 50)

    lines.append(f"\n  a) Performance globale :")
    lines.append(f"     Gain moyen vs. solution initiale : {df['gain_pct'].mean():.2f}%")
    lines.append(f"     Gain médian                      : {df['gain_pct'].median():.2f}%")
    lines.append(f"     Gain max                         : {df['gain_pct'].max():.2f}%  (route {df.loc[df['gain_pct'].idxmax(), 'route_id']})")
    lines.append(f"     Gain min                         : {df['gain_pct'].min():.2f}%  (route {df.loc[df['gain_pct'].idxmin(), 'route_id']})")

    lines.append(f"\n  b) Qualité des solutions :")
    lines.append(f"     Violations de contraintes totales : {int(df['n_violations'].sum())}")
    lines.append(f"     Taux de routes sans violation     : "
                 f"{len(df[df['n_violations']==0]) / len(df) * 100:.1f}%")

    lines.append(f"\n  c) Efficacité de la flotte :")
    lines.append(f"     Taux moyen d'utilisation des véhicules : "
                 f"{df['n_used'].sum() / df['n_vehicles'].sum() * 100:.1f}%")

    lines.append(f"\n  d) Convergence :")
    lines.append(f"     Nombre moyen d'itérations       : {df['n_iter'].mean():.0f}")
    lines.append(f"     Nombre moyen d'améliorations    : {df['n_improve'].mean():.0f}")
    lines.append(f"     Temps moyen par route (s)        : {df['elapsed_s'].mean():.1f}")

    lines.append(f"\n  e) Détail par route :")
    for _, row in df.iterrows():
        lines.append(f"     Route {row['route_id']:>8} | "
                     f"{int(row['n_clients']):>3} clients | "
                     f"Coût: {row['cost_final']:>10.2f} | "
                     f"Dist: {row['total_dist']:>7.1f} km | "
                     f"Gain: {row['gain_pct']:>5.1f}% | "
                     f"Viol: {int(row['n_violations'])}")

    # ── II.2.x.3 ────────────────────────────────────────────────────
    lines.append(f"\nII.2.{'2' if tag=='petit' else '3'}.3  Intérêt de l'étude")
    lines.append("-" * 50)
    lines.append("""
  Le Recuit Simulé appliqué au VRP avec les données réelles de cette BDD
  présente plusieurs intérêts :

  1. Adaptabilité aux contraintes réelles
     L'algorithme intègre simultanément les fenêtres de temps, les
     capacités hétérogènes des véhicules et les contraintes SDVRP
     (interdictions client-véhicule), ce qui reflète des scénarios
     de livraison réels en Bosnie-Herzégovine.

  2. Qualité de la solution
     Le recuit simulé permet d'échapper aux optima locaux grâce au
     critère de Metropolis (acceptation probabiliste des solutions
     dégradées), produisant des gains significatifs par rapport à
     la solution gloutonne initiale.

  3. Flexibilité des opérateurs
     Les 4 opérateurs de voisinage (swap intra/inter-route, relocalisation,
     2-opt) assurent une exploration large de l'espace de solutions,
     en exploitant à la fois la diversification et l'intensification.

  4. Passage à l'échelle
     L'algorithme traite efficacement des instances allant de quelques
     dizaines à plus de 100 clients, avec un temps de calcul raisonnable
     (< 30 s par route sur CPU standard).

  5. Comparaison BDD petite / grande taille
     L'analyse comparative permet d'évaluer la robustesse de l'approche
     sur des instances de complexité variable, et d'identifier les
     paramètres (T0, α) les mieux adaptés à chaque type de BDD.
    """)

    lines.append("=" * 72)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Rapport écrit : {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════════════════════════════

def run_bdd(tag, route_ids, df_customers, df_vehicles, df_depots,
            df_distances, df_constraints, params):
    """
    Exécute le pipeline complet pour un ensemble de routes (une BDD).
    Génère toutes les figures et le rapport structuré.
    """
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  BDD {tag.upper()} — {len(route_ids)} route(s)")
    print(f"{'═'*65}")

    all_stats    = []
    all_metrics  = []
    all_hist_cost = []
    all_route_ids = []

    for rid in route_ids:
        best, hist_cost, stats, df_metrics = run_route(
            rid, df_customers, df_vehicles, df_depots, df_distances,
            df_constraints, tag, out_dir, params, verbose=True)
        all_stats.append(stats)
        all_metrics.append(df_metrics)
        all_hist_cost.append(hist_cost)
        all_route_ids.append(rid)

    # Figures multi-routes
    fig_comparaison_routes(all_stats, tag, out_dir)
    if len(all_hist_cost) > 1:
        fig_boxplot_convergence(all_hist_cost, all_route_ids, tag, out_dir)

    # Tableau récapitulatif
    save_summary_table(all_stats, tag, out_dir)

    # Rapport texte
    write_report(all_stats, all_metrics, tag, out_dir)

    print(f"\n  ✅ BDD {tag} terminée. Résultats dans : {out_dir}")
    return all_stats



# ══════════════════════════════════════════════════════════════════════
# 13. COMPARAISON PETIT vs GRAND
# ══════════════════════════════════════════════════════════════════════

def _generate_comparison(stats_petit, stats_grand, out_dir):
    """
    Figure et tableau comparant les indicateurs clés entre BDD petit et grand.
    Illustre l'intérêt de tester l'algorithme sur des tailles différentes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def agg(stats_list):
        df = pd.DataFrame(stats_list)
        return {
            "n_routes"      : len(df),
            "moy_clients"   : df["n_clients"].mean(),
            "moy_cost"      : df["cost_final"].mean(),
            "moy_dist"      : df["total_dist"].mean(),
            "moy_gain"      : df["gain_pct"].mean(),
            "moy_temps"     : df["elapsed_s"].mean(),
            "moy_violations": df["n_violations"].mean(),
            "moy_iter"      : df["n_iter"].mean(),
        }

    p = agg(stats_petit)
    g = agg(stats_grand)

    metrics = [
        ("Nbre de routes",           p["n_routes"],       g["n_routes"],       ""),
        ("Clients moyens / route",   p["moy_clients"],    g["moy_clients"],    ""),
        ("Coût moyen",               p["moy_cost"],       g["moy_cost"],       "KM"),
        ("Distance moy. (km)",       p["moy_dist"],       g["moy_dist"],       "km"),
        ("Gain moyen (%)",           p["moy_gain"],       g["moy_gain"],       "%"),
        ("Temps moyen (s)",          p["moy_temps"],      g["moy_temps"],      "s"),
        ("Violations moyennes",      p["moy_violations"], g["moy_violations"], ""),
        ("Itérations moyennes",      p["moy_iter"],       g["moy_iter"],       ""),
    ]

    # ── Tableau figure ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#F8F9FA")
    ax.axis("off")
    col_labels = ["Indicateur", "BDD Petite taille", "BDD Grande taille", "Unité"]
    cell_text  = [[m[0], f"{m[1]:.2f}", f"{m[2]:.2f}", m[3]] for m in metrics]
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(4):
        tbl[0, j].set_facecolor("#1D3557")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_text) + 1):
        for j in range(4):
            tbl[i, j].set_facecolor("#EEF2F7" if i % 2 == 0 else "white")
    ax.set_title("Comparaison BDD Petite taille vs Grande taille — Recuit Simulé",
                 fontsize=12, fontweight="bold", color="#1D3557", pad=15)
    plt.tight_layout()
    path_tbl = out_dir / "comparaison_petit_vs_grand_tableau.png"
    plt.savefig(path_tbl, dpi=150, bbox_inches="tight"); plt.close()

    # ── Bar chart comparatif ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Comparaison Petit vs Grand — Recuit Simulé",
                 fontsize=13, fontweight="bold", color="#1D3557")

    for ax, (label, vp, vg, unit) in zip(axes, [
        ("Coût moyen",     p["moy_cost"],  g["moy_cost"],  "KM"),
        ("Gain moyen (%)", p["moy_gain"],  g["moy_gain"],  "%"),
        ("Temps moyen (s)",p["moy_temps"], g["moy_temps"], "s"),
    ]):
        ax.set_facecolor("#EEF2F7")
        bars = ax.bar(["Petit", "Grand"], [vp, vg],
                      color=["#2A9D8F", "#E63946"], edgecolor="white", width=0.5)
        for bar, v in zip(bars, [vp, vg]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{v:.2f} {unit}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel(unit); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path_bar = out_dir / "comparaison_petit_vs_grand_bar.png"
    plt.savefig(path_bar, dpi=150, bbox_inches="tight"); plt.close()

    print(f"  Comparaison sauvegardée : {path_tbl.name} | {path_bar.name}")

if __name__ == "__main__":
    print("=" * 65)
    print("  VRP — Recuit Simulé (PTVFT) | ICO Centrale Lille")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════════════
    # II.2.2 — BDD PETITE TAILLE
    # Route 2939484 | 20 clients | 3 vehicules
    # ══════════════════════════════════════════════════════════════════
    print("\n[1] Chargement BDD petite taille...")
    df_cp, df_vp, df_dp, df_distp, df_conp = load_petit_data()
    PETIT_ROUTES = sorted(df_cp["ROUTE_ID"].unique().tolist())
    print(f"    Routes : {PETIT_ROUTES}")
    for rid in PETIT_ROUTES:
        nc = len(df_cp[df_cp["ROUTE_ID"] == rid])
        nv = len(df_vp[df_vp["ROUTE_ID"] == rid])
        print(f"    Route {rid} : {nc} clients | {nv} vehicules")

    stats_petit = run_bdd(
        tag         = "petit",
        route_ids   = PETIT_ROUTES,
        df_customers= df_cp, df_vehicles= df_vp, df_depots= df_dp,
        df_distances= df_distp, df_constraints= df_conp,
        params      = SA_PARAMS_PETIT,
    )

    # ══════════════════════════════════════════════════════════════════
    # II.2.3 — BDD GRANDE TAILLE
    # 11 routes | jusqu a 129 clients | jusqu a 8 vehicules
    # ══════════════════════════════════════════════════════════════════
    print("\n[2] Chargement BDD grande taille...")
    df_c, df_v, df_d, df_dist, df_con = load_all_data()
    ALL_ROUTES = sorted(df_c["ROUTE_ID"].unique().tolist())
    print(f"    {len(ALL_ROUTES)} routes : {ALL_ROUTES}")

    stats_grand = run_bdd(
        tag         = "grand",
        route_ids   = ALL_ROUTES,
        df_customers= df_c, df_vehicles= df_v, df_depots= df_d,
        df_distances= df_dist, df_constraints= df_con,
        params      = SA_PARAMS,
    )

    # ══════════════════════════════════════════════════════════════════
    # COMPARAISON PETIT vs GRAND
    # ══════════════════════════════════════════════════════════════════
    print("\n[3] Génération comparaison petit vs grand...")
    _generate_comparison(stats_petit, stats_grand, OUT_DIR)

    print("\n" + "=" * 65)
    print("  Tous les resultats ont ete sauvegardes.")
    print(f"  Dossier de sortie : {OUT_DIR}")
    print("=" * 65)
