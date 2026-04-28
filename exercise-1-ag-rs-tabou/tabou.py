# -*- coding: utf-8 -*-
"""
ICO — Algorithme Tabou pour le VRP (Grande Taille + Petite Taille)
Centrale Lille — Fil Rouge

Fonction de coût standardisée (formule du cours) :
    f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij                (1)

  - c_ij  = VEHICLE_VARIABLE_COST_KM_k × distance_km(i,j)
  - K(x)  = nombre de véhicules utilisés (routes non vides)
  - ω_k   = VEHICLE_FIXED_COST_KM_k  (par véhicule utilisé)
  - E     = ensemble des arcs de la solution x

Modifications v2 :
  1. Lecture des fichiers .xlsx (petite taille) via openpyxl
  2. Lecture des fichiers .xls  (grande taille) via parseur BIFF8 intégré
  3. Paramètre DATASET = 'petit' | 'grand' | 'les_deux'
  4. Chemins configurables en haut du fichier
"""

import struct
import math
import random
import time
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════════════
# PARAMÈTRES GLOBAUX
# ══════════════════════════════════════════════════════════════════════

# Choisir le dataset à traiter : 'petit', 'grand', ou 'les_deux'
DATASET = 'les_deux'

# ── Chemins des fichiers GRANDE taille (.xls) ──────────────────────
PATH_DATA_GRAND      = '..\\database\\'
PATH_DISTANCES_GRAND = '..\\database\\6_detail_table_cust_depots_distances.xls'

ALL_ROUTE_IDS_GRAND = [
    2604001, 2922001, 2939484, 2946091, 2958047,
    2970877, 2990001, 3005971, 3016355, 3027038, 3044702,
]

# ── Chemins des fichiers PETITE taille (.xlsx) ──────────────────────
PATH_DATA_PETIT = '..\\database\\petit\\'       # dossier contenant les fichiers _petit.xlsx

ALL_ROUTE_IDS_PETIT = [2939484]   # route(s) présente(s) dans la petite BD

# ── Paramètres Tabou ────────────────────────────────────────────────
TAILLE_TABOU = 7
NB_MAX_ITER  = 500
SEED         = 42

IMAGE_ROOT_TABOU = Path("images_tabou")
COLORS_TABOU = ["#E63946", "#2A9D8F", "#F4A261", "#457B9D", "#6A4C93", "#1982C4", "#8AC926", "#FF595E"]

def format_route_id(route_id):
    try:
        value = float(route_id)
        return str(int(value)) if value.is_integer() else str(value).replace(".", "_")
    except Exception:
        return str(route_id).replace(".", "_")

def get_route_image_dir_tabou(route_id):
    out_dir = IMAGE_ROOT_TABOU / f"route_{format_route_id(route_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ══════════════════════════════════════════════════════════════════════
# BLOC 1A : LECTURE DES FICHIERS XLS — BIFF8 (grande taille)
# ══════════════════════════════════════════════════════════════════════

def read_xls(filepath):
    """Parseur BIFF8 bas-niveau — aucune dépendance externe."""
    with open(filepath, 'rb') as f:
        data = f.read()

    sector_size  = 512
    difat_sectors = [
        struct.unpack_from('<I', data, 76 + i * 4)[0]
        for i in range(109)
        if struct.unpack_from('<I', data, 76 + i * 4)[0] < 0xFFFFFFFD
    ]
    fat = {}
    for k, fs in enumerate(difat_sectors):
        off = (fs + 1) * sector_size
        for i in range(sector_size // 4):
            if off + i * 4 + 4 <= len(data):
                fat[k * (sector_size // 4) + i] = struct.unpack_from('<I', data, off + i * 4)[0]

    def chain(start):
        r = b'';  cur = start;  vis = set()
        while cur < 0xFFFFFFFD and cur not in vis:
            vis.add(cur)
            r  += data[(cur + 1) * sector_size:(cur + 2) * sector_size]
            cur = fat.get(cur, 0xFFFFFFFE)
        return r

    dir_data = chain(struct.unpack_from('<I', data, 48)[0])
    wb_start = wb_size = None
    for i in range(len(dir_data) // 128):
        e  = dir_data[i * 128:(i + 1) * 128]
        if len(e) < 128: break
        nl = struct.unpack_from('<H', e, 64)[0]
        if 0 < nl <= 64:
            name = e[:nl].decode('utf-16-le', errors='ignore').rstrip('\x00')
            if name in ('Workbook', 'Book') and e[66] == 2:
                wb_start = struct.unpack_from('<I', e, 116)[0]
                wb_size  = struct.unpack_from('<I', e, 120)[0]
                break
    if wb_start is None:
        return []

    wb      = chain(wb_start)[:wb_size]
    records = [];  i = 0
    while i < len(wb) - 4:
        try:
            rt = struct.unpack_from('<H', wb, i)[0]
            rl = struct.unpack_from('<H', wb, i + 2)[0]
            if rl > 200000: break
            records.append((rt, wb[i + 4:i + 4 + rl]));  i += 4 + rl
        except:
            break

    merged = []
    for rt, rd in records:
        if rt == 0x003C and merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + rd)
        else:
            merged.append((rt, rd))

    sst = []
    for rt, rd in merged:
        if rt == 0x00FC and len(rd) >= 8:
            n   = struct.unpack_from('<I', rd, 4)[0];  pos = 8
            for _ in range(n):
                if pos + 2 >= len(rd): break
                try:
                    sl = struct.unpack_from('<H', rd, pos)[0]
                    fl = rd[pos + 2];  pos += 3
                    iu = fl & 1
                    if (fl >> 3) & 1: pos += 2
                    if (fl >> 2) & 1: pos += 4
                    bl = sl * 2 if iu else sl
                    s  = rd[pos:pos + bl].decode('utf-16-le' if iu else 'latin-1', errors='ignore')
                    pos += bl;  sst.append(s.strip())
                except:
                    sst.append('');  break

    cells = {}
    for rt, rd in merged:
        try:
            if rt == 0x0203 and len(rd) >= 14:
                r, c = struct.unpack_from('<H', rd, 0)[0], struct.unpack_from('<H', rd, 2)[0]
                cells[(r, c)] = struct.unpack_from('<d', rd, 6)[0]
            elif rt == 0x00FD and len(rd) >= 10:
                r, c = struct.unpack_from('<H', rd, 0)[0], struct.unpack_from('<H', rd, 2)[0]
                idx  = struct.unpack_from('<I', rd, 6)[0]
                if idx < len(sst): cells[(r, c)] = sst[idx]
            elif rt == 0x027E and len(rd) >= 10:
                r, c = struct.unpack_from('<H', rd, 0)[0], struct.unpack_from('<H', rd, 2)[0]
                rk   = struct.unpack_from('<I', rd, 6)[0]
                v    = float(rk >> 2) if rk & 2 else struct.unpack(
                    '<d', b'\x00\x00\x00\x00' + struct.pack('<I', rk & 0xFFFFFFFC))[0]
                if rk & 1: v /= 100.0
                cells[(r, c)] = v
            elif rt == 0x00BD and len(rd) >= 6:
                r  = struct.unpack_from('<H', rd, 0)[0]
                c0 = struct.unpack_from('<H', rd, 2)[0]
                pos = 4
                while pos + 6 <= len(rd) - 2:
                    rk = struct.unpack_from('<I', rd, pos + 2)[0]
                    v  = float(rk >> 2) if rk & 2 else struct.unpack(
                        '<d', b'\x00\x00\x00\x00' + struct.pack('<I', rk & 0xFFFFFFFC))[0]
                    if rk & 1: v /= 100.0
                    cells[(r, c0)] = v;  c0 += 1;  pos += 6
            elif rt == 0x0204 and len(rd) >= 8:
                r, c = struct.unpack_from('<H', rd, 0)[0], struct.unpack_from('<H', rd, 2)[0]
                sl   = struct.unpack_from('<H', rd, 6)[0]
                cells[(r, c)] = rd[8:8 + sl].decode('latin-1', errors='ignore')
        except:
            pass

    if not cells:
        return []
    mr = max(r for r, c in cells);  mc = max(c for r, c in cells)
    table   = [[cells.get((r, c), '') for c in range(mc + 1)] for r in range(mr + 1)]
    headers = table[0]
    rows    = []
    for row in table[1:]:
        d = {h: row[j] for j, h in enumerate(headers) if h and j < len(row)}
        if any(v != '' for v in d.values()):
            rows.append(d)
    return rows


# ══════════════════════════════════════════════════════════════════════
# BLOC 1B : LECTURE DES FICHIERS XLSX (petite taille) via openpyxl
# ══════════════════════════════════════════════════════════════════════

def read_xlsx(filepath):
    """
    Lit un fichier .xlsx avec openpyxl et retourne une liste de dicts
    (même format que read_xls) pour compatibilité avec le reste du code.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "La bibliothèque 'openpyxl' est requise pour lire les fichiers .xlsx.\n"
            "Installez-la avec : pip install openpyxl"
        )

    wb   = openpyxl.load_workbook(filepath, data_only=True)
    ws   = wb.active
    rows_raw = list(ws.iter_rows(values_only=True))
    if not rows_raw:
        return []

    headers = [str(h).strip() if h is not None else '' for h in rows_raw[0]]
    result  = []
    for row in rows_raw[1:]:
        d = {}
        for j, h in enumerate(headers):
            if h and j < len(row):
                val = row[j]
                # Normalise : None → ''
                d[h] = val if val is not None else ''
        if any(v != '' for v in d.values()):
            result.append(d)
    return result


# ══════════════════════════════════════════════════════════════════════
# BLOC 2 : CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def charger_toutes_donnees_grand():
    """Charge les 5 fichiers XLS de la grande taille."""
    print("  Chargement des fichiers XLS (grande taille)...")
    customers_all   = read_xls(PATH_DATA_GRAND + '2_detail_table_customers.xls')
    vehicles_all    = read_xls(PATH_DATA_GRAND + '3_detail_table_vehicles.xls')
    depots_all      = read_xls(PATH_DATA_GRAND + '4_detail_table_depots.xls')
    constraints_all = read_xls(PATH_DATA_GRAND + '5_detail_table_constraints_sdvrp.xls')
    distances_all   = read_xls(PATH_DISTANCES_GRAND)
    print("  Fichiers XLS chargés.")
    return customers_all, vehicles_all, depots_all, constraints_all, distances_all


def charger_toutes_donnees_petit():
    """Charge les 5 fichiers XLSX de la petite taille."""
    print("  Chargement des fichiers XLSX (petite taille)...")
    customers_all   = read_xlsx(PATH_DATA_PETIT + '2_detail_table_customers_petit.xlsx')
    vehicles_all    = read_xlsx(PATH_DATA_PETIT + '3_detail_table_vehicles_petit.xlsx')
    depots_all      = read_xlsx(PATH_DATA_PETIT + '4_detail_table_depots_petit.xlsx')
    constraints_all = read_xlsx(PATH_DATA_PETIT + '5_detail_table_constraints_sdvrp_petit.xlsx')
    distances_all   = read_xlsx(PATH_DATA_PETIT + '6_detail_table_cust_depots_distance_petit.xlsx')
    print("  Fichiers XLSX chargés.")
    return customers_all, vehicles_all, depots_all, constraints_all, distances_all


# ══════════════════════════════════════════════════════════════════════
# BLOC 3 : EXTRACTION DES DONNÉES PAR ROUTE
# ══════════════════════════════════════════════════════════════════════

def extraire_route(route_id, customers_all, vehicles_all, depots_all,
                   constraints_all, distances_all):
    """
    Extrait et normalise toutes les données d'une ROUTE_ID donnée.
    Compatibilité totale avec les deux formats (xls / xlsx).
    """
    # Le ROUTE_ID peut être stocké comme float ou int selon le format
    rid_float = float(route_id)
    rid_int   = int(route_id)

    def match_rid(val):
        try:
            v = float(val)
            return v == rid_float
        except (TypeError, ValueError):
            return str(val).strip() == str(rid_int)

    def filtre(rows):
        return [r for r in rows if match_rid(r.get('ROUTE_ID', ''))]

    def fval(d, key, default):
        v = d.get(key, '')
        if v == '' or v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    customers   = filtre(customers_all)
    vehicles    = filtre(vehicles_all)
    depot_rows  = filtre(depots_all)
    constraints = filtre(constraints_all)
    distances   = filtre(distances_all)

    if not depot_rows:
        raise ValueError(f"ROUTE_ID={route_id} introuvable dans les dépôts.")
    if not customers:
        raise ValueError(f"ROUTE_ID={route_id} : aucun client trouvé.")
    if not vehicles:
        raise ValueError(f"ROUTE_ID={route_id} : aucun véhicule trouvé.")

    depot = {
        'code'    : str(depot_rows[0].get('DEPOT_CODE', 'D')),
        'lat'     : fval(depot_rows[0], 'DEPOT_LATITUDE',                0.0),
        'lon'     : fval(depot_rows[0], 'DEPOT_LONGITUDE',               0.0),
        'tw_from' : fval(depot_rows[0], 'DEPOT_AVAILABLE_TIME_FROM_MIN', 0.0),
        'tw_to'   : fval(depot_rows[0], 'DEPOT_AVAILABLE_TIME_TO_MIN', 1440.0),
    }

    # Clé client : CUSTOMER_CODE (grand) ou CUSTOMER_NUMBER (petit — fallback)
    def get_cust_code(c):
        for key in ('CUSTOMER_CODE', 'CUSTOMER_NUMBER'):
            if key in c and c[key] != '':
                return str(c[key])
        return str(list(c.values())[0])

    clients = [{
        'code'       : get_cust_code(c),
        'lat'        : fval(c, 'CUSTOMER_LATITUDE',                   0.0),
        'lon'        : fval(c, 'CUSTOMER_LONGITUDE',                  0.0),
        'poids_kg'   : fval(c, 'TOTAL_WEIGHT_KG',                     0.0),
        'volume_m3'  : fval(c, 'TOTAL_VOLUME_M3',                     0.0),
        'tw_from'    : fval(c, 'CUSTOMER_TIME_WINDOW_FROM_MIN',        0.0),
        'tw_to'      : fval(c, 'CUSTOMER_TIME_WINDOW_TO_MIN',       1440.0),
        'service_min': fval(c, 'CUSTOMER_DELIVERY_SERVICE_TIME_MIN',   0.0),
    } for c in customers]

    # Clé véhicule : VEHICLE_CODE (grand) ou VEHICLE_NUMBER (petit — fallback)
    def get_veh_code(v):
        for key in ('VEHICLE_CODE', 'VEHICLE_NUMBER'):
            if key in v and v[key] != '':
                return str(v[key])
        return str(list(v.values())[0])

    veh_list = [{
        'code'        : get_veh_code(v),
        'cap_kg'      : fval(v, 'VEHICLE_TOTAL_WEIGHT_KG',         9999.0),
        'cap_m3'      : fval(v, 'VEHICLE_TOTAL_VOLUME_M3',         9999.0),
        'cout_fixe'   : fval(v, 'VEHICLE_FIXED_COST_KM',              0.0),
        'cout_var_km' : fval(v, 'VEHICLE_VARIABLE_COST_KM',           1.0),
        'dispo_from'  : fval(v, 'VEHICLE_AVAILABLE_TIME_FROM_MIN',     0.0),
        'dispo_to'    : fval(v, 'VEHICLE_AVAILABLE_TIME_TO_MIN',    1440.0),
    } for v in vehicles]

    # ── Matrice distances réelles dépôt↔client ───────────────────────
    dist_reel  = {}
    temps_reel = {}
    for d in distances:
        # Compatibilité CUSTOMER_CODE / CUSTOMER_NUMBER
        code = str(d.get('CUSTOMER_CODE', d.get('CUSTOMER_NUMBER', ''))).strip()
        direction = str(d.get('DIRECTION', '')).strip()
        dist_val  = d.get('DISTANCE_KM', '')
        time_val  = d.get('TIME_DISTANCE_MIN', '')
        if code and direction:
            dist_reel [(code, direction)] = float(dist_val)  if dist_val  not in ('', None) else 0.0
            temps_reel[(code, direction)] = float(time_val)  if time_val  not in ('', None) else 0.0

    # Vitesse moyenne (km/h) calculée à partir des vraies distances
    vitesses    = [dist_reel[(k, d)] / (temps_reel[(k, d)] / 60.0)
                   for (k, d) in dist_reel
                   if temps_reel.get((k, d), 0) > 0 and dist_reel[(k, d)] > 0]
    vitesse_moy = sum(vitesses) / len(vitesses) if vitesses else 50.0

    # ── Contraintes SDVRP ────────────────────────────────────────────
    sdvrp = defaultdict(set)
    for c in constraints:
        cust_key = str(c.get('SDVRP_CONSTRAINT_CUSTOMER_CODE',
                              c.get('CUSTOMER_CODE', ''))).strip()
        veh_key  = str(c.get('SDVRP_CONSTRAINT_VEHICLE_CODE',
                              c.get('VEHICLE_CODE', ''))).strip()
        if cust_key and veh_key:
            sdvrp[cust_key].add(veh_key)

    return depot, clients, veh_list, dist_reel, temps_reel, vitesse_moy, sdvrp


# ══════════════════════════════════════════════════════════════════════
# BLOC 4 : MATRICES DISTANCES ET TEMPS
# ══════════════════════════════════════════════════════════════════════

R_TERRE = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R_TERRE * math.asin(math.sqrt(a))


def construire_matrices(depot, clients, dist_reel, temps_reel, vitesse_moy):
    """
    dist_mat[i][j]  : distance en km   (indice 0 = dépôt)
    temps_mat[i][j] : temps en minutes
    Priorité aux vraies distances PTV ; Haversine en fallback.
    """
    n         = len(clients)
    dist_mat  = [[0.0] * (n + 1) for _ in range(n + 1)]
    temps_mat = [[0.0] * (n + 1) for _ in range(n + 1)]

    for i, ci in enumerate(clients, 1):
        code = ci['code']
        d_dc = dist_reel.get((code, 'DEPOT->CUSTOMER'),
                             haversine(depot['lat'], depot['lon'], ci['lat'], ci['lon']))
        t_dc = temps_reel.get((code, 'DEPOT->CUSTOMER'), d_dc / vitesse_moy * 60.0 if vitesse_moy > 0 else 0.0)
        d_cd = dist_reel.get((code, 'CUSTOMER->DEPOT'),
                             haversine(ci['lat'], ci['lon'], depot['lat'], depot['lon']))
        t_cd = temps_reel.get((code, 'CUSTOMER->DEPOT'), d_cd / vitesse_moy * 60.0 if vitesse_moy > 0 else 0.0)

        dist_mat[0][i]  = d_dc;  temps_mat[0][i]  = t_dc
        dist_mat[i][0]  = d_cd;  temps_mat[i][0]  = t_cd

        for j, cj in enumerate(clients, 1):
            if i != j:
                d               = haversine(ci['lat'], ci['lon'], cj['lat'], cj['lon'])
                dist_mat[i][j]  = d
                temps_mat[i][j] = d / vitesse_moy * 60.0 if vitesse_moy > 0 else 0.0

    return dist_mat, temps_mat


# ══════════════════════════════════════════════════════════════════════
# BLOC 5 : FONCTIONS DE COÛT  f(x) = ω·K(x) + Σ c_ij
# ══════════════════════════════════════════════════════════════════════

def calculer_distance_route(route, dist_mat):
    """Distance totale d'une tournée en km."""
    return sum(dist_mat[route[p]][route[p + 1]] for p in range(len(route) - 1))


def calculer_f_objectif(routes, dist_mat, veh_list):
    """
    Objectif pur (équation 1 du cours) :
        f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij

    Où :
        ω·K(x)   = Σ_{k utilisé} cout_fixe_k
        Σ c_ij   = Σ_{k utilisé} cout_var_km_k × dist_route_k
    """
    omega_k_x = 0.0
    sum_c_ij  = 0.0
    for k, route in enumerate(routes):
        if not any(c != 0 for c in route):
            continue
        dist_route  = calculer_distance_route(route, dist_mat)
        omega_k_x  += veh_list[k]['cout_fixe']
        sum_c_ij   += veh_list[k]['cout_var_km'] * dist_route
    return omega_k_x + sum_c_ij


# Alias
calculer_cout = calculer_f_objectif


def calculer_distance_totale(routes, dist_mat):
    return sum(calculer_distance_route(route, dist_mat) for route in routes)


def decomposer_cout(routes, dist_mat, veh_list):
    """Retourne (omega_k_x, sum_c_ij, f_total)."""
    omega_k_x = sum(veh_list[k]['cout_fixe']
                    for k, route in enumerate(routes)
                    if any(c != 0 for c in route))
    sum_c_ij  = sum(veh_list[k]['cout_var_km'] * calculer_distance_route(route, dist_mat)
                    for k, route in enumerate(routes)
                    if any(c != 0 for c in route))
    return omega_k_x, sum_c_ij, omega_k_x + sum_c_ij


# ══════════════════════════════════════════════════════════════════════
# BLOC 6 : VÉRIFICATION DES CONTRAINTES
# ══════════════════════════════════════════════════════════════════════

def sdvrp_ok(client_idx, veh_code, clients, sdvrp):
    if client_idx == 0:
        return True
    return veh_code not in sdvrp.get(clients[client_idx - 1]['code'], set())


def fenetres_temps_ok(route, clients, temps_mat, heure_depart):
    t = heure_depart
    for k in range(len(route) - 1):
        i = route[k];  j = route[k + 1]
        t += temps_mat[i][j]
        if j == 0:
            break
        client = clients[j - 1]
        if t > client['tw_to']:
            return False
        t = max(t, client['tw_from'])
        t += client['service_min']
    return True


# ══════════════════════════════════════════════════════════════════════
# BLOC 7 : SOLUTION INITIALE ALÉATOIRE
# ══════════════════════════════════════════════════════════════════════

def solution_initiale(n_clients, veh_list, clients, sdvrp, seed):
    random.seed(seed)
    n_veh      = len(veh_list)
    ordre      = list(range(1, n_clients + 1))
    random.shuffle(ordre)
    routes     = [[0] for _ in range(n_veh)]
    charges    = [0.0] * n_veh
    non_places = []

    for client_idx in ordre:
        place     = False
        veh_ordre = list(range(n_veh))
        random.shuffle(veh_ordre)
        for v in veh_ordre:
            poids_client = clients[client_idx - 1]['poids_kg']
            veh_code     = veh_list[v]['code']
            if (charges[v] + poids_client <= veh_list[v]['cap_kg'] and
                    sdvrp_ok(client_idx, veh_code, clients, sdvrp)):
                routes[v].append(client_idx)
                charges[v] += poids_client
                place = True
                break
        if not place:
            non_places.append(client_idx)

    for v in range(n_veh):
        routes[v].append(0)

    if non_places:
        print(f"  ATTENTION : {len(non_places)} clients non placés !")
    return routes


# ══════════════════════════════════════════════════════════════════════
# BLOC 8 : VOISINAGE PAR RELOCATION
# ══════════════════════════════════════════════════════════════════════

def generer_voisins(routes, clients, veh_list, sdvrp, temps_mat):
    voisins = []
    for v_src in range(len(routes)):
        for pos_src, client_idx in enumerate(routes[v_src]):
            if client_idx == 0:
                continue
            poids_client = clients[client_idx - 1]['poids_kg']
            for v_dst in range(len(routes)):
                if v_dst == v_src:
                    continue
                veh_code_dst = veh_list[v_dst]['code']
                cap_dst      = veh_list[v_dst]['cap_kg']
                heure_depart = veh_list[v_dst]['dispo_from']
                charge_dst   = sum(clients[c - 1]['poids_kg'] for c in routes[v_dst] if c != 0)

                if charge_dst + poids_client > cap_dst:
                    continue
                if not sdvrp_ok(client_idx, veh_code_dst, clients, sdvrp):
                    continue

                for pos_dst in range(1, len(routes[v_dst])):
                    route_test = (routes[v_dst][:pos_dst] + [client_idx] + routes[v_dst][pos_dst:])
                    if fenetres_temps_ok(route_test, clients, temps_mat, heure_depart):
                        voisins.append((client_idx, v_src, pos_src, v_dst, pos_dst))
    return voisins


def appliquer_mouvement(routes, mouvement):
    client_idx, v_src, pos_src, v_dst, pos_dst = mouvement
    r = [route[:] for route in routes]
    r[v_src].pop(pos_src)
    r[v_dst].insert(pos_dst, client_idx)
    return r


def est_tabou(mouvement, liste_tabou):
    client_idx, v_src, _, v_dst, _ = mouvement
    return (client_idx, v_src, v_dst) in liste_tabou


def mouvement_inverse(mouvement):
    client_idx, v_src, _, v_dst, _ = mouvement
    return (client_idx, v_dst, v_src)


# ══════════════════════════════════════════════════════════════════════
# BLOC 9 : ALGORITHME TABOU
# Minimise f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij
# ══════════════════════════════════════════════════════════════════════

def tabou_vrp(routes_init, dist_mat, temps_mat, clients, veh_list, sdvrp):
    """
    Algorithme Tabou — relocation inter-routes.

    À chaque itération :
      1. Générer tous les voisins valides
      2. Choisir le meilleur voisin non tabou
         (ou tabou si critère d'aspiration : f(x) < meilleur global)
      3. Mettre à jour la liste tabou et la meilleure solution
    """
    routes    = [r[:] for r in routes_init]
    cout      = calculer_f_objectif(routes, dist_mat, veh_list)
    routes_s  = [r[:] for r in routes]
    cout_s    = cout
    tabou     = []
    nbiter    = 0
    meil_iter = 0
    history = [cout_s]

    while nbiter < NB_MAX_ITER:
        nbiter += 1
        voisins = generer_voisins(routes, clients, veh_list, sdvrp, temps_mat)
        if not voisins:
            print(f"  Aucun voisin valide → arrêt à l'iter {nbiter}")
            break

        best_mv     = None
        best_cout   = float('inf')
        best_routes = None

        for mv in voisins:
            new_routes = appliquer_mouvement(routes, mv)
            new_cout   = calculer_f_objectif(new_routes, dist_mat, veh_list)
            is_tab     = est_tabou(mv, tabou)
            aspiration = new_cout < cout_s   # critère d'aspiration

            if (not is_tab or aspiration) and new_cout < best_cout:
                best_cout   = new_cout
                best_mv     = mv
                best_routes = new_routes

        if best_mv is None:
            continue

        routes = best_routes;  cout = best_cout
        tabou.append(mouvement_inverse(best_mv))
        if len(tabou) > TAILLE_TABOU:
            tabou.pop(0)

        if cout < cout_s:
            cout_s    = cout
            routes_s  = [r[:] for r in routes]
            meil_iter = nbiter

        history.append(cout_s)

    return routes_s, cout_s, meil_iter, history


# ══════════════════════════════════════════════════════════════════════
# BLOC 10 : AFFICHAGE DÉTAIL D'UNE SOLUTION
# ══════════════════════════════════════════════════════════════════════

def plot_tabou_convergence(history, route_id, out_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, linewidth=1.8)
    plt.title(f"Évolution du coût Tabou — Route {format_route_id(route_id)}")
    plt.xlabel("Itération")
    plt.ylabel("f(x) = ω·K(x) + Σc_ij")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "tabou_cost_evolution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Image sauvegardée : {path}")
    return path


def plot_tabou_routes(routes, depot, clients, route_id, out_dir, suffix):
    plt.figure(figsize=(10, 8))
    for v, route in enumerate(routes):
        if not any(c != 0 for c in route):
            continue
        lats = []
        lons = []
        for node in route:
            if node == 0:
                lats.append(depot["lat"])
                lons.append(depot["lon"])
            else:
                client = clients[node - 1]
                lats.append(client["lat"])
                lons.append(client["lon"])
        plt.plot(lons, lats, marker="o", linewidth=1.8, markersize=4,
                 color=COLORS_TABOU[v % len(COLORS_TABOU)], label=f"Véhicule {v + 1}")

    plt.scatter([depot["lon"]], [depot["lat"]], marker="s", s=120, label="Dépôt")
    plt.title(f"Routes {suffix} Tabou — Route {format_route_id(route_id)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"tabou_routes_{suffix}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Image sauvegardée : {path}")
    return path


def afficher_solution(routes, clients, veh_list, dist_mat, temps_mat, titre):
    SEP = "=" * 70
    print("\n" + SEP)
    print(titre)
    print(SEP)

    total_omega, total_arc, total_f = decomposer_cout(routes, dist_mat, veh_list)
    print(f"  f(x) = ω·K(x) + Σc_ij = {total_omega:.2f} + {total_arc:.2f} = {total_f:.2f}")
    print()

    for v, route in enumerate(routes):
        n_cli     = sum(1 for c in route if c != 0)
        charge_kg = sum(clients[c - 1]['poids_kg']  for c in route if c != 0)
        charge_m3 = sum(clients[c - 1]['volume_m3'] for c in route if c != 0)
        cap_kg    = veh_list[v]['cap_kg']
        cap_m3    = veh_list[v]['cap_m3']
        hdep      = veh_list[v]['dispo_from']
        statut    = "OK" if fenetres_temps_ok(route, clients, temps_mat, hdep) else "VIOLATION TW"
        codes     = ['Dépôt' if c == 0 else clients[c - 1]['code'] for c in route]
        dist_rte  = calculer_distance_route(route, dist_mat)
        omega_k   = veh_list[v]['cout_fixe']   if n_cli > 0 else 0.0
        arc_k     = veh_list[v]['cout_var_km'] * dist_rte if n_cli > 0 else 0.0

        print(f"  V{v+1} [{veh_list[v]['code']}] ({n_cli} clients) "
              f"poids={charge_kg:.1f}/{cap_kg:.0f}kg "
              f"vol={charge_m3:.2f}/{cap_m3:.2f}m³ "
              f"dist={dist_rte:.1f}km "
              f"f_k=ω+Σc={omega_k:.2f}+{arc_k:.2f}={omega_k+arc_k:.2f} "
              f"[{statut}]")
        print("    " + " → ".join(codes))
    print()


# ══════════════════════════════════════════════════════════════════════
# BLOC 11 : TRAITEMENT COMPLET D'UNE ROUTE
# ══════════════════════════════════════════════════════════════════════

def traiter_route(route_id, customers_all, vehicles_all, depots_all,
                  constraints_all, distances_all, verbose=False):
    """
    Pipeline complet :
      1. Extraire les données
      2. Construire les matrices
      3. Générer la solution initiale (seed fixe)
      4. Lancer l'algorithme Tabou
      5. Retourner les résultats
    """
    depot, clients, veh_list, dist_reel, temps_reel, vitesse_moy, sdvrp = \
        extraire_route(route_id, customers_all, vehicles_all, depots_all,
                       constraints_all, distances_all)

    n_clients = len(clients)
    dist_mat, temps_mat = construire_matrices(depot, clients, dist_reel, temps_reel, vitesse_moy)

    routes_init = solution_initiale(n_clients, veh_list, clients, sdvrp, SEED)
    cout_init   = calculer_f_objectif(routes_init, dist_mat, veh_list)
    dist_init   = calculer_distance_totale(routes_init, dist_mat)
    omega_init, arc_init, _ = decomposer_cout(routes_init, dist_mat, veh_list)

    if verbose:
        afficher_solution(routes_init, clients, veh_list, dist_mat, temps_mat,
                          f"SOLUTION INITIALE — Route {route_id}")

    temps_debut = time.time()
    routes_fin, cout_final, meil_iter, history = tabou_vrp(
        routes_init, dist_mat, temps_mat, clients, veh_list, sdvrp)
    temps_exec = time.time() - temps_debut

    dist_fin   = calculer_distance_totale(routes_fin, dist_mat)
    amelio     = (cout_init - cout_final) / cout_init * 100 if cout_init > 0 else 0.0
    violations = sum(
        1 for v, route in enumerate(routes_fin)
        if not fenetres_temps_ok(route, clients, temps_mat, veh_list[v]['dispo_from'])
    )
    omega_fin, arc_fin, _ = decomposer_cout(routes_fin, dist_mat, veh_list)

    out_dir = get_route_image_dir_tabou(route_id)
    plot_tabou_convergence(history, route_id, out_dir)
    plot_tabou_routes(routes_init, depot, clients, route_id, out_dir, "initial")
    plot_tabou_routes(routes_fin, depot, clients, route_id, out_dir, "final")

    if verbose:
        afficher_solution(routes_fin, clients, veh_list, dist_mat, temps_mat,
                          f"SOLUTION FINALE — Route {route_id}")

    return {
        'route_id'   : route_id,
        'n_clients'  : n_clients,
        'n_veh'      : len(veh_list),
        'cout_init'  : cout_init,
        'cout_final' : cout_final,
        'omega_init' : omega_init,
        'arc_init'   : arc_init,
        'omega_final': omega_fin,
        'arc_final'  : arc_fin,
        'dist_init'  : dist_init,
        'dist_fin'   : dist_fin,
        'amelio'     : amelio,
        'meil_iter'  : meil_iter,
        'temps_exec' : temps_exec,
        'violations' : violations,
    }


# ══════════════════════════════════════════════════════════════════════
# BLOC 12 : BILAN GLOBAL (affichage tableau)
# ══════════════════════════════════════════════════════════════════════

def afficher_bilan(resultats, titre):
    SEP1 = "=" * 78
    SEP2 = "-" * 78
    print("\n" + SEP1)
    print(titre)
    print(f"Paramètres : TAILLE_TABOU={TAILLE_TABOU}, NB_MAX_ITER={NB_MAX_ITER}, SEED={SEED}")
    print("Fonction objectif : f(x) = ω·K(x) + Σ_{(i,j)∈E} c_ij")
    print(SEP2)
    print(f"{'ROUTE_ID':<12} {'Clients':>7} {'Veh':>4} "
          f"{'f_init':>12} {'f_final':>12} "
          f"{'ω·K':>10} {'Σc_ij':>10} "
          f"{'Dist(km)':>9} {'Amélio':>8} {'Temps(s)':>9} {'ViolTW':>7}")
    print(SEP2)

    total_init  = total_final = total_dist = total_temps = 0.0
    for r in resultats:
        print(f"{r['route_id']:<12} {r['n_clients']:>7} {r['n_veh']:>4} "
              f"{r['cout_init']:>12.2f} {r['cout_final']:>12.2f} "
              f"{r['omega_final']:>10.2f} {r['arc_final']:>10.2f} "
              f"{r['dist_fin']:>9.2f} {r['amelio']:>7.1f}% "
              f"{r['temps_exec']:>9.2f} {r['violations']:>7}")
        total_init  += r['cout_init']
        total_final += r['cout_final']
        total_dist  += r['dist_fin']
        total_temps += r['temps_exec']

    amelio_globale = (total_init - total_final) / total_init * 100 if total_init > 0 else 0.0
    print(SEP2)
    print(f"{'TOTAL':<12} {'':>7} {'':>4} "
          f"{total_init:>12.2f} {total_final:>12.2f} "
          f"{'':>10} {'':>10} "
          f"{total_dist:>9.2f} {amelio_globale:>7.1f}% "
          f"{total_temps:>9.2f}")
    print(SEP1)
    print("\nLégende :")
    print("  f(x) = ω·K(x) + Σc_ij  (équation 1 du cours)")
    print("  ω·K(x) = Σ VEHICLE_FIXED_COST_KM    (coût fixe par véhicule utilisé)")
    print("  Σc_ij  = Σ VEHICLE_VARIABLE_COST_KM × distance_km  (coût des arcs)")
    print("  ViolTW = nombre de routes avec violation de fenêtre de temps")
    print(SEP1)


# ══════════════════════════════════════════════════════════════════════
# BLOC 13 : PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run_dataset(label, route_ids, charger_fn, verbose_first=True):
    """Lance le Tabou sur toutes les routes d'un dataset et affiche le bilan."""
    SEP = "=" * 78
    print("\n" + SEP)
    print(f"DATASET : {label}  ({len(route_ids)} route(s))")
    print(SEP)

    (customers_all, vehicles_all, depots_all,
     constraints_all, distances_all) = charger_fn()

    resultats = []
    for i, route_id in enumerate(route_ids, 1):
        print(f"\n[{i:>2}/{len(route_ids)}] ROUTE_ID={route_id}...")
        try:
            res = traiter_route(
                route_id,
                customers_all, vehicles_all, depots_all,
                constraints_all, distances_all,
                verbose=(verbose_first and i == 1),   # détail sur la 1ʳᵉ route seulement
            )
            resultats.append(res)
            print(f"  → {res['n_clients']} clients | {res['n_veh']} veh | "
                  f"f_init={res['cout_init']:.2f} | f_final={res['cout_final']:.2f} | "
                  f"ω·K={res['omega_final']:.2f} + Σc_ij={res['arc_final']:.2f} | "
                  f"dist={res['dist_fin']:.1f} km | amélio={res['amelio']:.1f}% | "
                  f"{res['temps_exec']:.1f}s | TW_viol={res['violations']}")
        except Exception as e:
            print(f"  ERREUR sur route {route_id} : {e}")

    if resultats:
        afficher_bilan(resultats, f"BILAN — {label}")
    return resultats


if __name__ == "__main__":
    print("=" * 78)
    print("VRP TABOU — minimisation de f(x) = ω·K(x) + Σ c_ij")
    print(f"Mode : DATASET = '{DATASET}'")
    print("=" * 78)

    if DATASET in ('petit', 'les_deux'):
        run_dataset(
            label       = "PETITE TAILLE (.xlsx)",
            route_ids   = ALL_ROUTE_IDS_PETIT,
            charger_fn  = charger_toutes_donnees_petit,
            verbose_first=True,
        )

    if DATASET in ('grand', 'les_deux'):
        run_dataset(
            label       = "GRANDE TAILLE (.xls)",
            route_ids   = ALL_ROUTE_IDS_GRAND,
            charger_fn  = charger_toutes_donnees_grand,
            verbose_first=False,
        )
