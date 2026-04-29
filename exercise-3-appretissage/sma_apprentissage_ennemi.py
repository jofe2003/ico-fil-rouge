import math
import os
import random
import struct
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OMEGA = 500.0
DEFAULT_ROUTE_ID = 2939484

GA_OUTER_CYCLES = 10
GA_POPULATION_SIZE = 16
GA_OFFSPRING_PER_CYCLE = 12
GA_MUTATION_RATE = 0.25
GA_ENEMY_RATE = 0.30

SA_OUTER_CYCLES = 10
SA_INNER_ITER = 120
SA_T0 = 600.0
SA_ALPHA = 0.985
SA_ENEMY_RATE = 0.25

TABU_OUTER_CYCLES = 10
TABU_ITER_PER_CYCLE = 25
TABU_TENURE = 8
TABU_ENEMY_RATE = 0.25


def first_existing(df, names, default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


@dataclass
class VRPInstance:
    route_id: int
    customers: pd.DataFrame
    vehicles: pd.DataFrame
    depot: pd.Series
    distances: pd.DataFrame
    constraints: pd.DataFrame
    customer_codes: list
    code_to_idx: dict
    idx_to_code: dict
    vehicle_codes: list
    customer_weight: np.ndarray
    customer_volume: np.ndarray
    tw_from: np.ndarray
    tw_to: np.ndarray
    service_time: np.ndarray
    cap_weight: np.ndarray
    cap_volume: np.ndarray
    available_from: np.ndarray
    fixed_cost: np.ndarray
    variable_cost: np.ndarray
    forbidden: set
    forbidden_matrix: np.ndarray
    dist_cc: np.ndarray
    time_cc: np.ndarray
    dist_dc: np.ndarray
    time_dc: np.ndarray
    dist_cd: np.ndarray
    time_cd: np.ndarray
    customer_number: np.ndarray


@dataclass
class Solution:
    routes: list
    objective: float = float('inf')
    penalized: float = float('inf')
    penalties: float = float('inf')
    source: str = ''
    route_objectives: list = None
    route_penalties: list = None
    assignment_penalty: float = 0.0

    def clone(self):
        return Solution(
            [r[:] for r in self.routes],
            self.objective,
            self.penalized,
            self.penalties,
            self.source,
            self.route_objectives[:] if self.route_objectives is not None else None,
            self.route_penalties[:] if self.route_penalties is not None else None,
            self.assignment_penalty,
        )


class EnemyPool:
    def __init__(self, max_size=18, pool_radius=6.0):
        self.max_size = max_size
        self.pool_radius = pool_radius
        self.items = []
        self.history_size = []
        self.history_best = []
        self.history_diversity = []

    def arcs_of(self, solution):
        arcs = set()
        for route in solution.routes:
            if not route:
                continue
            prev = -1
            for node in route:
                arcs.add((prev, node))
                prev = node
            arcs.add((prev, -1))
        return arcs

    def lambda_distance(self, s1, s2):
        a1 = self.arcs_of(s1)
        a2 = self.arcs_of(s2)
        return len(a1.symmetric_difference(a2))

    def phi(self, lam):
        if lam <= self.pool_radius:
            return 1.0 - (lam / self.pool_radius)
        return 0.0

    def diversity_score(self, candidate):
        if not self.items:
            return 0.0
        return sum(self.phi(self.lambda_distance(candidate, sol)) for sol in self.items)

    def add(self, candidate):
        cand = candidate.clone()
        if not self.items:
            self.items.append(cand)
            self._record()
            return True
        duplicate = any(self.lambda_distance(cand, s) == 0 for s in self.items)
        if duplicate:
            self._record()
            return False
        self.items.append(cand)
        ranked = []
        for idx, sol in enumerate(self.items):
            score = self.diversity_score(sol)
            ranked.append((sol.penalized, score, idx, sol))
        ranked.sort(key=lambda x: (x[0], x[1]))
        self.items = [x[3] for x in ranked[:self.max_size]]
        self._record()
        return True

    def best(self):
        if not self.items:
            return None
        return min(self.items, key=lambda s: s.penalized).clone()

    def sample_enemy_score(self, exclude_source=None):
        choices = [s for s in self.items if s.source != exclude_source] if exclude_source else self.items[:]
        if not choices:
            return None
        return min(s.penalized for s in choices)

    def _record(self):
        self.history_size.append(len(self.items))
        self.history_best.append(min((s.penalized for s in self.items), default=np.nan))
        if len(self.items) >= 2:
            divs = []
            for i in range(len(self.items)):
                for j in range(i + 1, len(self.items)):
                    divs.append(self.lambda_distance(self.items[i], self.items[j]))
            self.history_diversity.append(float(np.mean(divs)) if divs else 0.0)
        else:
            self.history_diversity.append(0.0)


class AgentResult:
    def __init__(self, name):
        self.name = name
        self.best = None
        self.history = []
        self.enemy_observations = 0
        self.enemy_improvements = 0


def read_xls_biff8(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    sector_size = 512
    difat = []
    for i in range(109):
        val = struct.unpack_from('<I', data, 76 + i * 4)[0]
        if val < 0xFFFFFFFD:
            difat.append(val)

    fat = {}
    for k, fs in enumerate(difat):
        off = (fs + 1) * sector_size
        for i in range(sector_size // 4):
            pos = off + i * 4
            if pos + 4 <= len(data):
                fat[k * (sector_size // 4) + i] = struct.unpack_from('<I', data, pos)[0]

    def chain(start):
        out = b''
        cur = start
        seen = set()
        while cur < 0xFFFFFFFD and cur not in seen:
            seen.add(cur)
            out += data[(cur + 1) * sector_size:(cur + 2) * sector_size]
            cur = fat.get(cur, 0xFFFFFFFE)
        return out

    dir_data = chain(struct.unpack_from('<I', data, 48)[0])
    wb_start, wb_size = None, None
    for i in range(len(dir_data) // 128):
        entry = dir_data[i * 128:(i + 1) * 128]
        if len(entry) < 128:
            break
        name_len = struct.unpack_from('<H', entry, 64)[0]
        if 0 < name_len <= 64:
            name = entry[:name_len].decode('utf-16-le', errors='ignore').rstrip('\x00')
            if name in ('Workbook', 'Book') and entry[66] == 2:
                wb_start = struct.unpack_from('<I', entry, 116)[0]
                wb_size = struct.unpack_from('<I', entry, 120)[0]
                break

    if wb_start is None:
        return pd.DataFrame()

    wb = chain(wb_start)[:wb_size]
    records = []
    i = 0
    while i < len(wb) - 4:
        try:
            rt = struct.unpack_from('<H', wb, i)[0]
            rl = struct.unpack_from('<H', wb, i + 2)[0]
            if rl > 200000:
                break
            records.append((rt, wb[i + 4:i + 4 + rl]))
            i += 4 + rl
        except Exception:
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
            n = struct.unpack_from('<I', rd, 4)[0]
            pos = 8
            for _ in range(n):
                if pos + 2 >= len(rd):
                    break
                try:
                    sl = struct.unpack_from('<H', rd, pos)[0]
                    fl = rd[pos + 2]
                    pos += 3
                    iu = fl & 1
                    if (fl >> 3) & 1:
                        pos += 2
                    if (fl >> 2) & 1:
                        pos += 4
                    bl = sl * 2 if iu else sl
                    s = rd[pos:pos + bl].decode('utf-16-le' if iu else 'latin-1', errors='ignore')
                    pos += bl
                    sst.append(s.strip())
                except Exception:
                    sst.append('')
                    break

    cells = {}
    for rt, rd in merged:
        try:
            if rt == 0x0203 and len(rd) >= 14:
                r = struct.unpack_from('<H', rd, 0)[0]
                c = struct.unpack_from('<H', rd, 2)[0]
                cells[(r, c)] = struct.unpack_from('<d', rd, 6)[0]
            elif rt == 0x00FD and len(rd) >= 10:
                r = struct.unpack_from('<H', rd, 0)[0]
                c = struct.unpack_from('<H', rd, 2)[0]
                idx = struct.unpack_from('<I', rd, 6)[0]
                if idx < len(sst):
                    cells[(r, c)] = sst[idx]
            elif rt == 0x027E and len(rd) >= 10:
                r = struct.unpack_from('<H', rd, 0)[0]
                c = struct.unpack_from('<H', rd, 2)[0]
                rk = struct.unpack_from('<I', rd, 6)[0]
                v = float(rk >> 2) if rk & 2 else struct.unpack('<d', b'\x00\x00\x00\x00' + struct.pack('<I', rk & 0xFFFFFFFC))[0]
                if rk & 1:
                    v /= 100.0
                cells[(r, c)] = v
            elif rt == 0x00BD and len(rd) >= 6:
                r = struct.unpack_from('<H', rd, 0)[0]
                c0 = struct.unpack_from('<H', rd, 2)[0]
                pos = 4
                while pos + 6 <= len(rd) - 2:
                    rk = struct.unpack_from('<I', rd, pos + 2)[0]
                    v = float(rk >> 2) if rk & 2 else struct.unpack('<d', b'\x00\x00\x00\x00' + struct.pack('<I', rk & 0xFFFFFFFC))[0]
                    if rk & 1:
                        v /= 100.0
                    cells[(r, c0)] = v
                    c0 += 1
                    pos += 6
            elif rt == 0x0204 and len(rd) >= 8:
                r = struct.unpack_from('<H', rd, 0)[0]
                c = struct.unpack_from('<H', rd, 2)[0]
                sl = struct.unpack_from('<H', rd, 6)[0]
                cells[(r, c)] = rd[8:8 + sl].decode('latin-1', errors='ignore')
        except Exception:
            pass

    if not cells:
        return pd.DataFrame()

    mr = max(r for r, _ in cells)
    mc = max(c for _, c in cells)
    table = [[cells.get((r, c), '') for c in range(mc + 1)] for r in range(mr + 1)]
    headers = [str(h).strip() if h is not None else '' for h in table[0]]

    rows = []
    for row in table[1:]:
        d = {headers[j]: row[j] for j in range(min(len(headers), len(row))) if headers[j]}
        if any(v != '' for v in d.values()):
            rows.append(d)
    return pd.DataFrame(rows)


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == '.xls':
        try:
            return pd.read_excel(path)
        except Exception:
            return read_xls_biff8(path)
    raise ValueError(f'Extension non supportée: {path.suffix}')


def detect_dataset(root):
    root = Path(root)
    return {
        'customers': root / '2_detail_table_customers.xls',
        'vehicles': root / '3_detail_table_vehicles.xls',
        'depots': root / '4_detail_table_depots.xls',
        'constraints': root / '5_detail_table_constraints_sdvrp.xls',
        'distances': root / '6_detail_table_cust_depots_distances.xls'
    }


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlam / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_instance(customers_df, vehicles_df, depots_df, constraints_df, distances_df, route_id):
    customers = customers_df[customers_df['ROUTE_ID'] == route_id].copy().reset_index(drop=True)
    vehicles = vehicles_df[vehicles_df['ROUTE_ID'] == route_id].copy().reset_index(drop=True)
    depots = depots_df[depots_df['ROUTE_ID'] == route_id].copy().reset_index(drop=True)
    constraints = constraints_df[constraints_df['ROUTE_ID'] == route_id].copy().reset_index(drop=True)
    distances = distances_df[distances_df['ROUTE_ID'] == route_id].copy().reset_index(drop=True)

    if customers.empty or vehicles.empty or depots.empty:
        raise ValueError(f'ROUTE_ID {route_id} incomplet dans la base')

    depot = depots.iloc[0]

    cust_code_col = first_existing(customers, ['CUSTOMER_CODE', 'CUSTOMER_NUMBER'])
    veh_code_col = first_existing(vehicles, ['VEHICLE_CODE', 'VEHICLE_NUMBER'])
    cust_num_col = first_existing(customers, ['CUSTOMER_NUMBER'], None)

    customers[cust_code_col] = customers[cust_code_col].astype(str)
    vehicles[veh_code_col] = vehicles[veh_code_col].astype(str)

    customer_codes = customers[cust_code_col].tolist()
    code_to_idx = {c: i for i, c in enumerate(customer_codes)}
    idx_to_code = {i: c for c, i in code_to_idx.items()}
    vehicle_codes = vehicles[veh_code_col].tolist()

    n = len(customers)
    lats = customers['CUSTOMER_LATITUDE'].astype(float).to_numpy()
    lons = customers['CUSTOMER_LONGITUDE'].astype(float).to_numpy()

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    dist_cc = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    np.fill_diagonal(dist_cc, 0.0)

    dist_dc = np.zeros(n, dtype=float)
    dist_cd = np.zeros(n, dtype=float)
    time_dc = np.zeros(n, dtype=float)
    time_cd = np.zeros(n, dtype=float)

    dist_customer_col = first_existing(distances, ['CUSTOMER_CODE', 'CUSTOMER_NUMBER'])
    if not distances.empty and dist_customer_col and 'DIRECTION' in distances.columns:
        for _, row in distances.iterrows():
            code = str(row[dist_customer_col])
            if code not in code_to_idx:
                continue
            idx = code_to_idx[code]
            direction = str(row['DIRECTION']).strip().upper()
            if 'DEPOT->CUSTOMER' in direction:
                dist_dc[idx] = float(row.get('DISTANCE_KM', 0.0) or 0.0)
                time_dc[idx] = float(row.get('TIME_DISTANCE_MIN', 0.0) or 0.0)
            elif 'CUSTOMER->DEPOT' in direction:
                dist_cd[idx] = float(row.get('DISTANCE_KM', 0.0) or 0.0)
                time_cd[idx] = float(row.get('TIME_DISTANCE_MIN', 0.0) or 0.0)

    dlat = float(depot['DEPOT_LATITUDE'])
    dlon = float(depot['DEPOT_LONGITUDE'])
    for i in range(n):
        d = haversine(dlat, dlon, lats[i], lons[i])
        if dist_dc[i] <= 0:
            dist_dc[i] = d
        if dist_cd[i] <= 0:
            dist_cd[i] = d

    speeds = []
    for d, t in list(zip(dist_dc, time_dc)) + list(zip(dist_cd, time_cd)):
        if d > 0 and t > 0:
            speeds.append(d / (t / 60.0))
    avg_speed_kmh = float(np.mean(speeds)) if speeds else 40.0

    time_cc = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                time_cc[i, j] = dist_cc[i, j] / avg_speed_kmh * 60.0

    for i in range(n):
        if time_dc[i] <= 0:
            time_dc[i] = dist_dc[i] / avg_speed_kmh * 60.0
        if time_cd[i] <= 0:
            time_cd[i] = dist_cd[i] / avg_speed_kmh * 60.0

    forbidden = set()
    ccol = first_existing(constraints, ['SDVRP_CONSTRAINT_CUSTOMER_CODE', 'CUSTOMER_CODE', 'CUSTOMER_NUMBER'])
    vcol = first_existing(constraints, ['SDVRP_CONSTRAINT_VEHICLE_CODE', 'VEHICLE_CODE', 'VEHICLE_NUMBER'])
    if ccol and vcol:
        forbidden = set(zip(constraints[ccol].astype(str), constraints[vcol].astype(str)))

    forbidden_matrix = np.zeros((n, len(vehicle_codes)), dtype=bool)
    vehicle_to_idx = {v: k for k, v in enumerate(vehicle_codes)}
    for cust_code, veh_code in forbidden:
        if cust_code in code_to_idx and veh_code in vehicle_to_idx:
            forbidden_matrix[code_to_idx[cust_code], vehicle_to_idx[veh_code]] = True

    def get_col(df, candidates, default):
        col = first_existing(df, candidates)
        if col is not None:
            return df[col].astype(float).to_numpy()
        return np.full(len(df), default, dtype=float)

    if cust_num_col is not None:
        customer_number = customers[cust_num_col].astype(float).to_numpy()
    else:
        customer_number = np.arange(1, len(customers) + 1, dtype=float)

    return VRPInstance(
        route_id=int(route_id),
        customers=customers,
        vehicles=vehicles,
        depot=depot,
        distances=distances,
        constraints=constraints,
        customer_codes=customer_codes,
        code_to_idx=code_to_idx,
        idx_to_code=idx_to_code,
        vehicle_codes=vehicle_codes,
        customer_weight=get_col(customers, ['TOTAL_WEIGHT_KG'], 0.0),
        customer_volume=get_col(customers, ['TOTAL_VOLUME_M3'], 0.0),
        tw_from=get_col(customers, ['CUSTOMER_TIME_WINDOW_FROM_MIN'], 0.0),
        tw_to=get_col(customers, ['CUSTOMER_TIME_WINDOW_TO_MIN'], 1e12),
        service_time=get_col(customers, ['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'], 0.0),
        cap_weight=get_col(vehicles, ['VEHICLE_TOTAL_WEIGHT_KG'], 1e12),
        cap_volume=get_col(vehicles, ['VEHICLE_TOTAL_VOLUME_M3'], 1e12),
        available_from=get_col(vehicles, ['VEHICLE_AVAILABLE_TIME_FROM_MIN'], 0.0),
        fixed_cost=np.full(len(vehicles), OMEGA, dtype=float),
        variable_cost=np.ones(len(vehicles), dtype=float),
        forbidden=forbidden,
        forbidden_matrix=forbidden_matrix,
        dist_cc=dist_cc,
        time_cc=time_cc,
        dist_dc=dist_dc,
        time_dc=time_dc,
        dist_cd=dist_cd,
        time_cd=time_cd,
        customer_number=customer_number,
    )


def route_objective(instance, route, vehicle_idx):
    if not route:
        return 0.0
    total_arc = instance.dist_dc[route[0]]
    for i in range(len(route) - 1):
        total_arc += instance.dist_cc[route[i], route[i + 1]]
    total_arc += instance.dist_cd[route[-1]]
    return OMEGA + total_arc


def route_penalty(instance, route, vehicle_idx, penalty_factor=500.0):
    if not route:
        return 0.0

    total = 0.0
    total_w = 0.0
    total_v = 0.0
    current_time = instance.available_from[vehicle_idx] + instance.time_dc[route[0]]

    for pos, i in enumerate(route):
        total_w += instance.customer_weight[i]
        total_v += instance.customer_volume[i]

        if current_time < instance.tw_from[i]:
            current_time = instance.tw_from[i]

        if current_time > instance.tw_to[i]:
            width = max(1.0, instance.tw_to[i] - instance.tw_from[i])
            total += penalty_factor * (current_time - instance.tw_to[i]) / width

        if instance.forbidden_matrix[i, vehicle_idx]:
            total += penalty_factor

        current_time += instance.service_time[i]

        if pos < len(route) - 1:
            current_time += instance.time_cc[route[pos], route[pos + 1]]
        else:
            current_time += instance.time_cd[i]

    if total_w > instance.cap_weight[vehicle_idx]:
        total += penalty_factor * (total_w - instance.cap_weight[vehicle_idx]) / max(1.0, instance.cap_weight[vehicle_idx])

    if total_v > instance.cap_volume[vehicle_idx]:
        total += penalty_factor * (total_v - instance.cap_volume[vehicle_idx]) / max(1.0, instance.cap_volume[vehicle_idx])

    return total


def assignment_penalty(instance, routes, penalty_factor=500.0):
    assigned = [c for route in routes for c in route]
    assigned_set = set(assigned)
    duplicates = len(assigned) - len(assigned_set)
    missing = len(instance.customer_codes) - len(assigned_set)

    total = 0.0
    if duplicates > 0:
        total += penalty_factor * 10 * duplicates
    if missing > 0:
        total += penalty_factor * 10 * missing
    return total


def objective(instance, routes):
    return sum(route_objective(instance, route, k) for k, route in enumerate(routes))


def penalties(instance, routes, penalty_factor=500.0):
    route_penalties = [
        route_penalty(instance, route, k, penalty_factor=penalty_factor)
        for k, route in enumerate(routes)
    ]
    return sum(route_penalties) + assignment_penalty(instance, routes, penalty_factor=penalty_factor)


def evaluate(instance, routes, penalty_factor=500.0, source=''):
    clean_routes = [r[:] for r in routes]
    route_objectives = [
        route_objective(instance, route, k)
        for k, route in enumerate(clean_routes)
    ]
    route_penalties = [
        route_penalty(instance, route, k, penalty_factor=penalty_factor)
        for k, route in enumerate(clean_routes)
    ]
    assign_pen = assignment_penalty(instance, clean_routes, penalty_factor=penalty_factor)

    sol = Solution(clean_routes, source=source)
    sol.route_objectives = route_objectives
    sol.route_penalties = route_penalties
    sol.assignment_penalty = assign_pen
    sol.objective = float(sum(route_objectives))
    sol.penalties = float(sum(route_penalties) + assign_pen)
    sol.penalized = sol.objective + sol.penalties
    return sol


def rebuild_solution_from_routes(instance, routes, route_objectives, route_penalties, assignment_pen, source=''):
    sol = Solution([r[:] for r in routes], source=source)
    sol.route_objectives = route_objectives[:]
    sol.route_penalties = route_penalties[:]
    sol.assignment_penalty = assignment_pen
    sol.objective = float(sum(route_objectives))
    sol.penalties = float(sum(route_penalties) + assignment_pen)
    sol.penalized = sol.objective + sol.penalties
    return sol

def empty_routes_like(instance):
    return [[] for _ in range(len(instance.vehicle_codes))]


def route_distance(route, vehicle_idx, instance):
    if not route:
        return 0.0
    d = instance.dist_dc[route[0]]
    for i in range(len(route) - 1):
        d += instance.dist_cc[route[i], route[i + 1]]
    d += instance.dist_cd[route[-1]]
    return d


def two_opt_route(route, vehicle_idx, instance):
    if len(route) < 4:
        return route[:]
    best = route[:]
    best_cost = route_distance(best, vehicle_idx, instance)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 1, len(best)):
                cand = best[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                cand_cost = route_distance(cand, vehicle_idx, instance)
                if cand_cost + 1e-9 < best_cost:
                    best = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


def excel_order_initial_solution(instance):
    routes = empty_routes_like(instance)
    ordered_customers = sorted(range(len(instance.customer_codes)), key=lambda i: instance.customer_number[i])

    for idx, c in enumerate(ordered_customers):
        routes[idx % len(routes)].append(c)

    for k in range(len(routes)):
        if len(routes[k]) >= 4:
            routes[k] = two_opt_route(routes[k], k, instance)

    return evaluate(instance, routes, source='excel_order')


def greedy_initial_solution(instance, seed=None):
    if seed is not None:
        random.seed(seed)

    routes = empty_routes_like(instance)
    ordered_customers = sorted(range(len(instance.customer_codes)), key=lambda i: instance.customer_number[i])

    for c in ordered_customers:
        best_k = None
        best_score = float('inf')
        for k in range(len(routes)):
            trial = [r[:] for r in routes]
            trial[k].append(c)
            score = evaluate(instance, trial).penalized
            if score < best_score:
                best_score = score
                best_k = k
        routes[best_k].append(c)

    for k in range(len(routes)):
        if len(routes[k]) >= 4:
            routes[k] = two_opt_route(routes[k], k, instance)

    return evaluate(instance, routes, source='greedy_excel')


def random_initial_solution(instance):
    base = excel_order_initial_solution(instance)
    routes = [r[:] for r in base.routes]
    for _ in range(max(1, len(instance.customer_codes) // 4)):
        routes = mutate_routes(routes, instance)
    return evaluate(instance, routes, source='random_from_excel')


def relocate_move(routes):
    cand = [r[:] for r in routes]
    non_empty = [k for k, r in enumerate(cand) if r]
    if not non_empty:
        return cand
    ks = random.choice(non_empty)
    pos = random.randrange(len(cand[ks]))
    node = cand[ks].pop(pos)
    kd = random.randrange(len(cand))
    insert_pos = random.randrange(len(cand[kd]) + 1)
    cand[kd].insert(insert_pos, node)
    return cand


def swap_move(routes):
    cand = [r[:] for r in routes]
    candidates = [k for k, r in enumerate(cand) if r]
    if len(candidates) < 2:
        return cand
    k1, k2 = random.sample(candidates, 2)
    i = random.randrange(len(cand[k1]))
    j = random.randrange(len(cand[k2]))
    cand[k1][i], cand[k2][j] = cand[k2][j], cand[k1][i]
    return cand


def intra_route_swap_move(routes):
    cand = [r[:] for r in routes]
    candidates = [k for k, r in enumerate(cand) if len(r) >= 2]
    if not candidates:
        return cand
    k = random.choice(candidates)
    i, j = random.sample(range(len(cand[k])), 2)
    cand[k][i], cand[k][j] = cand[k][j], cand[k][i]
    return cand


def inter_route_swap_move(routes):
    return swap_move(routes)


def intra_route_shift_move(routes):
    cand = [r[:] for r in routes]
    candidates = [k for k, r in enumerate(cand) if len(r) >= 2]
    if not candidates:
        return cand
    k = random.choice(candidates)
    i = random.randrange(len(cand[k]))
    node = cand[k].pop(i)
    j = random.randrange(len(cand[k]) + 1)
    cand[k].insert(j, node)
    return cand


def inter_route_shift_move(routes):
    cand = [r[:] for r in routes]
    sources = [k for k, r in enumerate(cand) if r]
    if not sources:
        return cand
    ks = random.choice(sources)
    destinations = [k for k in range(len(cand)) if k != ks]
    if not destinations:
        return cand
    kd = random.choice(destinations)
    i = random.randrange(len(cand[ks]))
    node = cand[ks].pop(i)
    j = random.randrange(len(cand[kd]) + 1)
    cand[kd].insert(j, node)
    return cand


def two_intra_route_swap_move(routes):
    cand = [r[:] for r in routes]
    candidates = [k for k, r in enumerate(cand) if len(r) >= 4]
    if not candidates:
        return cand
    k = random.choice(candidates)
    n = len(cand[k])
    valid = [(a, b) for a in range(n - 1) for b in range(n - 1) if abs(a - b) >= 2]
    if not valid:
        return cand
    a, b = random.choice(valid)
    if a > b:
        a, b = b, a
    first = cand[k][a:a + 2]
    second = cand[k][b:b + 2]
    cand[k][a:a + 2] = second
    cand[k][b:b + 2] = first
    return cand


def two_intra_route_shift_move(routes):
    cand = [r[:] for r in routes]
    candidates = [k for k, r in enumerate(cand) if len(r) >= 4]
    if not candidates:
        return cand
    k = random.choice(candidates)
    i = random.randrange(len(cand[k]) - 1)
    pair = cand[k][i:i + 2]
    del cand[k][i:i + 2]
    j = random.randrange(len(cand[k]) + 1)
    cand[k][j:j] = pair
    return cand


def insert_clients_best(cand, removed, instance):
    for c in removed:
        best_routes = None
        best_score = float('inf')
        for k in range(len(cand)):
            for pos in range(len(cand[k]) + 1):
                trial = [r[:] for r in cand]
                trial[k].insert(pos, c)
                sc = evaluate(instance, trial).penalized
                if sc < best_score:
                    best_score = sc
                    best_routes = trial
        cand = best_routes
    return cand


def eliminate_smallest_route_move(routes, instance):
    cand = [r[:] for r in routes]
    non_empty = [k for k, r in enumerate(cand) if r]
    if len(non_empty) < 2:
        return cand
    k = min(non_empty, key=lambda idx: len(cand[idx]))
    removed = cand[k][:]
    cand[k] = []
    return insert_clients_best(cand, removed, instance)


def eliminate_random_route_move(routes, instance):
    cand = [r[:] for r in routes]
    non_empty = [k for k, r in enumerate(cand) if r]
    if len(non_empty) < 2:
        return cand
    k = random.choice(non_empty)
    removed = cand[k][:]
    cand[k] = []
    return insert_clients_best(cand, removed, instance)


def intra_two_opt_move(routes, instance):
    cand = [r[:] for r in routes]
    candidates = [k for k, r in enumerate(cand) if len(r) >= 4]
    if not candidates:
        return cand
    k = random.choice(candidates)
    cand[k] = two_opt_route(cand[k], k, instance)
    return cand


def destroy_repair_move(routes, instance):
    cand = [r[:] for r in routes]
    assigned = [c for r in cand for c in r]
    if len(assigned) < 3:
        return cand
    remove_n = max(1, len(assigned) // 10)
    removed = []
    for _ in range(remove_n):
        non_empty = [k for k, r in enumerate(cand) if r]
        if not non_empty:
            break
        k = random.choice(non_empty)
        pos = random.randrange(len(cand[k]))
        removed.append(cand[k].pop(pos))
    return insert_clients_best(cand, removed, instance)


QL_ACTIONS = ['intra_swap', 'inter_swap', 'intra_shift', 'inter_shift', 'two_intra_swap', 'two_intra_shift', 'eliminate_smallest_route', 'eliminate_random_route']


class QLearningController:
    def __init__(self, alpha=0.10, gamma=0.90, epsilon=0.20, epsilon_decay=0.995, min_epsilon=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q = {}

    def values(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(len(QL_ACTIONS), dtype=float)
        return self.q[state]

    def choose(self, state):
        values = self.values(state)
        if random.random() < self.epsilon:
            return random.randrange(len(QL_ACTIONS))
        best = np.flatnonzero(values == values.max())
        return int(random.choice(best))

    def update(self, state, action_idx, reward, next_state):
        values = self.values(state)
        next_values = self.values(next_state)
        target = reward + self.gamma * float(np.max(next_values))
        values[action_idx] = (1.0 - self.alpha) * values[action_idx] + self.alpha * target
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def q_state(solution, instance):
    used = sum(1 for r in solution.routes if r)
    nveh = max(1, len(instance.vehicle_codes))
    used_bucket = min(2, int(3 * used / nveh))
    penalty_bucket = 1 if solution.penalties > 1e-9 else 0
    cost_bucket = min(4, int(solution.objective // 500.0))
    return used_bucket, penalty_bucket, cost_bucket


def apply_neighbor_action(routes, instance, action):
    if action == 'intra_swap':
        return intra_route_swap_move(routes)
    if action == 'inter_swap':
        return inter_route_swap_move(routes)
    if action == 'intra_shift':
        return intra_route_shift_move(routes)
    if action == 'inter_shift':
        return inter_route_shift_move(routes)
    if action == 'two_intra_swap':
        return two_intra_route_swap_move(routes)
    if action == 'two_intra_shift':
        return two_intra_route_shift_move(routes)
    if action == 'eliminate_smallest_route':
        return eliminate_smallest_route_move(routes, instance)
    if action == 'eliminate_random_route':
        return eliminate_random_route_move(routes, instance)
    return destroy_repair_move(routes, instance)


def q_learning_neighbor(current, instance, ql, source=''):
    state = q_state(current, instance)
    action_idx = ql.choose(state)
    candidate_routes = apply_neighbor_action(current.routes, instance, QL_ACTIONS[action_idx])
    candidate = evaluate(instance, candidate_routes, source=source)
    reward = current.penalized - candidate.penalized
    next_state = q_state(candidate, instance)
    ql.update(state, action_idx, reward, next_state)
    return candidate


def mutate_routes(routes, instance):
    action = random.choice(QL_ACTIONS)
    return apply_neighbor_action(routes, instance, action)


def route_based_crossover(parent1, parent2, instance):
    n_customers = len(instance.customer_codes)
    child = empty_routes_like(instance)
    used = set()
    donor_routes = parent1.routes + parent2.routes
    random.shuffle(donor_routes)

    for route in donor_routes:
        if not route:
            continue
        if random.random() < 0.45:
            for node in route:
                if node not in used:
                    k = min(range(len(child)), key=lambda kk: len(child[kk]))
                    child[k].append(node)
                    used.add(node)

    missing = [c for c in range(n_customers) if c not in used]
    missing.sort(key=lambda i: instance.customer_number[i])

    for c in missing:
        best_child = None
        best_score = float('inf')
        for k in range(len(child)):
            for pos in range(len(child[k]) + 1):
                trial = [r[:] for r in child]
                trial[k].insert(pos, c)
                sc = evaluate(instance, trial).penalized
                if sc < best_score:
                    best_score = sc
                    best_child = trial
        child = best_child

    for k in range(len(child)):
        if len(child[k]) >= 4:
            child[k] = two_opt_route(child[k], k, instance)

    return evaluate(instance, child, source='ga')



def genetic_agent(instance, pool, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    result = AgentResult('genetic')
    ql = QLearningController()

    population = [excel_order_initial_solution(instance), greedy_initial_solution(instance, seed)]
    while len(population) < GA_POPULATION_SIZE:
        population.append(random_initial_solution(instance))

    for sol in population:
        sol.source = 'genetic'
        pool.add(sol)

    best = min(population, key=lambda s: s.penalized).clone()

    for _ in range(GA_OUTER_CYCLES):
        population.sort(key=lambda s: s.penalized)
        elites = [s.clone() for s in population[:max(2, GA_POPULATION_SIZE // 5)]]
        offspring = elites[:]

        while len(offspring) < GA_OFFSPRING_PER_CYCLE:
            p1, p2 = random.sample(population[:max(6, GA_POPULATION_SIZE // 2)], 2)
            child = route_based_crossover(p1, p2, instance)

            if random.random() < GA_MUTATION_RATE:
                child = q_learning_neighbor(child, instance, ql, source='genetic')

            if random.random() < GA_ENEMY_RATE:
                enemy_score = pool.sample_enemy_score(exclude_source='genetic')
                if enemy_score is not None:
                    result.enemy_observations += 1
                    candidate = child
                    steps = 2 if enemy_score + 1e-9 < child.penalized else 1
                    for _ in range(steps):
                        candidate = q_learning_neighbor(candidate, instance, ql, source='genetic')
                    if candidate.penalized + 1e-9 < child.penalized:
                        result.enemy_improvements += 1
                        child = candidate

            child.source = 'genetic'
            offspring.append(child)
            pool.add(child)

        population = sorted(offspring, key=lambda s: s.penalized)[:GA_POPULATION_SIZE]
        if population[0].penalized + 1e-9 < best.penalized:
            best = population[0].clone()

        result.history.append(best.penalized)
        pool.add(best)

    best.source = 'genetic'
    result.best = best
    return result


def simulated_annealing_agent(instance, pool, seed=43):
    random.seed(seed)
    np.random.seed(seed)
    result = AgentResult('annealing')
    ql = QLearningController()

    current = excel_order_initial_solution(instance)
    current.source = 'annealing'
    best = current.clone()
    pool.add(best)
    temp = SA_T0

    for _ in range(SA_OUTER_CYCLES):
        for _ in range(SA_INNER_ITER):
            candidate = q_learning_neighbor(current, instance, ql, source='annealing')
            delta = candidate.penalized - current.penalized
            if delta < 0 or random.random() < math.exp(-delta / max(1e-9, temp)):
                current = candidate
            if current.penalized + 1e-9 < best.penalized:
                best = current.clone()
                pool.add(best)
            temp *= SA_ALPHA

        if random.random() < SA_ENEMY_RATE:
            enemy_score = pool.sample_enemy_score(exclude_source='annealing')
            if enemy_score is not None:
                result.enemy_observations += 1
                candidate = current
                steps = 3 if enemy_score + 1e-9 < current.penalized else 1
                for _ in range(steps):
                    candidate = q_learning_neighbor(candidate, instance, ql, source='annealing')
                if candidate.penalized + 1e-9 < current.penalized:
                    result.enemy_improvements += 1
                    current = candidate
                    if current.penalized + 1e-9 < best.penalized:
                        best = current.clone()
                pool.add(candidate)

        result.history.append(best.penalized)

    best.source = 'annealing'
    result.best = best
    return result


class TabuMemory:
    def __init__(self, tenure=12):
        self.tenure = tenure
        self.q = deque()
        self.s = set()

    def add(self, move):
        if move is None:
            return
        self.q.append(move)
        self.s.add(move)
        if len(self.q) > self.tenure:
            old = self.q.popleft()
            self.s.discard(old)

    def contains(self, move):
        return move in self.s


def best_relocate_neighbor(instance, current, tabu, aspiration):
    routes = current.routes
    best_data = None
    best_move = None
    nveh = len(routes)

    if current.route_objectives is None or current.route_penalties is None:
        current = evaluate(instance, routes, source=current.source)

    base_obj = current.objective
    base_pen_routes = sum(current.route_penalties)
    assign_pen = current.assignment_penalty

    for ks in range(nveh):
        source_route = routes[ks]
        if not source_route:
            continue

        for i, node in enumerate(source_route):
            for kd in range(nveh):
                dest_route = routes[kd]
                for pos in range(len(dest_route) + 1):
                    if ks == kd and (pos == i or pos == i + 1):
                        continue

                    move = (node, ks, kd)

                    if ks == kd:
                        new_route = source_route[:]
                        removed = new_route.pop(i)
                        insert_pos = pos - 1 if pos > i else pos
                        new_route.insert(insert_pos, removed)

                        new_obj_k = route_objective(instance, new_route, ks)
                        new_pen_k = route_penalty(instance, new_route, ks)

                        obj = base_obj - current.route_objectives[ks] + new_obj_k
                        pen_routes = base_pen_routes - current.route_penalties[ks] + new_pen_k
                        penalized = obj + pen_routes + assign_pen

                        if tabu.contains(move) and penalized >= aspiration - 1e-9:
                            continue

                        if best_data is None or penalized + 1e-9 < best_data[0]:
                            new_routes = [r[:] for r in routes]
                            new_routes[ks] = new_route
                            new_route_objectives = current.route_objectives[:]
                            new_route_penalties = current.route_penalties[:]
                            new_route_objectives[ks] = new_obj_k
                            new_route_penalties[ks] = new_pen_k
                            best_data = (penalized, new_routes, new_route_objectives, new_route_penalties)
                            best_move = (node, kd, ks)

                    else:
                        new_source = source_route[:]
                        removed = new_source.pop(i)
                        new_dest = dest_route[:]
                        new_dest.insert(pos, removed)

                        new_obj_ks = route_objective(instance, new_source, ks)
                        new_obj_kd = route_objective(instance, new_dest, kd)
                        new_pen_ks = route_penalty(instance, new_source, ks)
                        new_pen_kd = route_penalty(instance, new_dest, kd)

                        obj = (
                            base_obj
                            - current.route_objectives[ks]
                            - current.route_objectives[kd]
                            + new_obj_ks
                            + new_obj_kd
                        )
                        pen_routes = (
                            base_pen_routes
                            - current.route_penalties[ks]
                            - current.route_penalties[kd]
                            + new_pen_ks
                            + new_pen_kd
                        )
                        penalized = obj + pen_routes + assign_pen

                        if tabu.contains(move) and penalized >= aspiration - 1e-9:
                            continue

                        if best_data is None or penalized + 1e-9 < best_data[0]:
                            new_routes = [r[:] for r in routes]
                            new_routes[ks] = new_source
                            new_routes[kd] = new_dest
                            new_route_objectives = current.route_objectives[:]
                            new_route_penalties = current.route_penalties[:]
                            new_route_objectives[ks] = new_obj_ks
                            new_route_objectives[kd] = new_obj_kd
                            new_route_penalties[ks] = new_pen_ks
                            new_route_penalties[kd] = new_pen_kd
                            best_data = (penalized, new_routes, new_route_objectives, new_route_penalties)
                            best_move = (node, kd, ks)

    if best_data is None:
        return None, None

    _, best_routes, best_route_objectives, best_route_penalties = best_data
    best_sol = rebuild_solution_from_routes(
        instance,
        best_routes,
        best_route_objectives,
        best_route_penalties,
        assign_pen,
        source='tabu',
    )
    return best_sol, best_move

def tabu_agent(instance, pool, seed=44):
    random.seed(seed)
    np.random.seed(seed)
    result = AgentResult('tabu')
    ql = QLearningController()

    current = excel_order_initial_solution(instance)
    current.source = 'tabu'
    best = current.clone()
    pool.add(best)
    tabu = TabuMemory(tenure=TABU_TENURE)

    for _ in range(TABU_OUTER_CYCLES):
        for _ in range(TABU_ITER_PER_CYCLE):
            neighbor, inverse_move = best_relocate_neighbor(instance, current, tabu, best.penalized)
            if neighbor is None:
                break
            current = neighbor
            tabu.add(inverse_move)
            if random.random() < 0.35:
                current = q_learning_neighbor(current, instance, ql, source='tabu')
            if current.penalized + 1e-9 < best.penalized:
                best = current.clone()
                pool.add(best)

        if random.random() < TABU_ENEMY_RATE:
            enemy_score = pool.sample_enemy_score(exclude_source='tabu')
            if enemy_score is not None:
                result.enemy_observations += 1
                candidate = current
                steps = 2 if enemy_score + 1e-9 < current.penalized else 1
                for _ in range(steps):
                    candidate = q_learning_neighbor(candidate, instance, ql, source='tabu')
                if candidate.penalized + 1e-9 < current.penalized:
                    result.enemy_improvements += 1
                    current = candidate
                    if current.penalized + 1e-9 < best.penalized:
                        best = current.clone()
                pool.add(candidate)

        result.history.append(best.penalized)

    best.source = 'tabu'
    result.best = best
    return result


def plot_convergence(results, output_dir):
    plt.figure(figsize=(10, 6))
    for res in results:
        if len(res.history) > 0:
            plt.plot(range(1, len(res.history) + 1), res.history, label=res.name)
    if results and all(len(r.history) > 0 for r in results):
        global_best = np.minimum.reduce([np.array(r.history, dtype=float) for r in results])
        plt.plot(range(1, len(global_best) + 1), global_best, linewidth=2.5, label='global_best')
    plt.xlabel('Cycle')
    plt.ylabel('Penalized objective')
    plt.title('Convergence des agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '01_convergence_agents.png', dpi=180)
    plt.close()


def plot_pool_metrics(pool, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(pool.history_size)
    axes[0].set_title('Taille du pool')
    axes[0].set_xlabel('Mise à jour')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pool.history_best)
    axes[1].set_title('Meilleur coût dans le pool')
    axes[1].set_xlabel('Mise à jour')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pool.history_diversity)
    axes[2].set_title('Diversité moyenne du pool')
    axes[2].set_xlabel('Mise à jour')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / '02_pool_metrics.png', dpi=180)
    plt.close()


def plot_agent_scores(results, output_dir):
    names = [r.name for r in results]
    objectives = [r.best.objective for r in results]
    penalized = [r.best.penalized for r in results]
    penalties_vals = [r.best.penalties for r in results]

    x = np.arange(len(results))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, objectives, width, label='f(x)')
    plt.bar(x, penalized, width, label='Penalized')
    plt.bar(x + width, penalties_vals, width, label='Penalties')
    plt.xticks(x, names)
    plt.ylabel('Value')
    plt.title('Best score breakdown by agent')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '03_agent_scores.png', dpi=180)
    plt.close()


def plot_enemy_activity(results, output_dir):
    names = [r.name for r in results]
    observations = [r.enemy_observations for r in results]
    gains = [r.enemy_improvements for r in results]
    x = np.arange(len(results))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.18, observations, 0.36, label='Enemy observations')
    plt.bar(x + 0.18, gains, 0.36, label='Q-learning gains')
    plt.xticks(x, names)
    plt.ylabel('Count')
    plt.title('Enemy score competition activity')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '04_enemy_activity.png', dpi=180)
    plt.close()


def _route_time(instance, route, vehicle_idx):
    if not route:
        return 0.0
    t = float(instance.time_dc[route[0]])
    for pos, cust in enumerate(route):
        t += float(instance.service_time[cust])
        if pos < len(route) - 1:
            t += float(instance.time_cc[route[pos], route[pos + 1]])
        else:
            t += float(instance.time_cd[cust])
    return t


def _route_weight(instance, route):
    return float(sum(instance.customer_weight[i] for i in route)) if route else 0.0


def _route_volume(instance, route):
    return float(sum(instance.customer_volume[i] for i in route)) if route else 0.0


def _arrival_profile(instance, route, vehicle_idx):
    arrivals = []
    opens = []
    closes = []
    if not route:
        return arrivals, opens, closes

    t = float(instance.available_from[vehicle_idx]) + float(instance.time_dc[route[0]])
    for pos, cust in enumerate(route):
        arrivals.append(t)
        opens.append(float(instance.tw_from[cust]))
        closes.append(float(instance.tw_to[cust]))
        start_service = max(t, float(instance.tw_from[cust]))
        t = start_service + float(instance.service_time[cust])
        if pos < len(route) - 1:
            t += float(instance.time_cc[route[pos], route[pos + 1]])
        else:
            t += float(instance.time_cd[cust])
    return arrivals, opens, closes


def plot_solution_diagnostics(instance, best, output_dir):
    used_routes = [(k, r) for k, r in enumerate(best.routes) if len(r) > 0]
    if not used_routes:
        return

    distances = []
    times = []
    stops = []
    weights = []
    volumes = []
    cap_w = []
    cap_v = []

    for k, route in used_routes:
        distances.append(route_distance(route, k, instance))
        times.append(_route_time(instance, route, k))
        stops.append(len(route))
        weights.append(_route_weight(instance, route))
        volumes.append(_route_volume(instance, route))
        cap_w.append(float(instance.cap_weight[k]))
        cap_v.append(float(instance.cap_volume[k]))

    x = np.arange(len(used_routes))
    labels = [f'V{k + 1}' for k, _ in used_routes]

    plt.figure(figsize=(10, 6))
    plt.bar(x, distances)
    plt.xticks(x, labels)
    plt.ylabel('Distance (km)')
    plt.title('Distance by used vehicle')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '05_route_distances.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(x, times)
    plt.xticks(x, labels)
    plt.ylabel('Time (min)')
    plt.title('Route time by used vehicle')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '06_route_times.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(x, stops)
    plt.xticks(x, labels)
    plt.ylabel('Number of customers')
    plt.title('Stops per used vehicle')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '07_route_stops.png', dpi=180)
    plt.close()

    util_w = [100.0 * w / cw if cw > 0 else 0.0 for w, cw in zip(weights, cap_w)]
    util_v = [100.0 * v / cv if cv > 0 else 0.0 for v, cv in zip(volumes, cap_v)]

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, util_w, 0.4, label='Weight')
    plt.bar(x + 0.2, util_v, 0.4, label='Volume')
    plt.axhline(100.0, linestyle='--')
    plt.xticks(x, labels)
    plt.ylabel('Utilization (%)')
    plt.title('Capacity utilization by used vehicle')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '08_capacity_utilization.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(distances, stops, s=80)
    for i in range(len(distances)):
        plt.annotate(labels[i], (distances[i], stops[i]))
    plt.xlabel('Distance (km)')
    plt.ylabel('Number of customers')
    plt.title('Distance vs stops')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '09_distance_vs_stops.png', dpi=180)
    plt.close()

    edge_lengths = []
    for _, route in used_routes:
        edge_lengths.append(float(instance.dist_dc[route[0]]))
        for i in range(len(route) - 1):
            edge_lengths.append(float(instance.dist_cc[route[i], route[i + 1]]))
        edge_lengths.append(float(instance.dist_cd[route[-1]]))

    plt.figure(figsize=(10, 6))
    plt.hist(edge_lengths, bins=min(20, max(5, len(edge_lengths) // 2)))
    plt.xlabel('Arc length (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of arc lengths')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / '10_arc_length_distribution.png', dpi=180)
    plt.close()

    all_arrivals = []
    all_opens = []
    all_closes = []
    for k, route in used_routes:
        arr, opn, cls = _arrival_profile(instance, route, k)
        all_arrivals.extend(arr)
        all_opens.extend(opn)
        all_closes.extend(cls)

    if all_arrivals:
        y = np.arange(len(all_arrivals))
        plt.figure(figsize=(12, max(6, 0.25 * len(all_arrivals))))
        plt.hlines(y, all_opens, all_closes)
        plt.scatter(all_arrivals, y, s=20, label='Arrival')
        plt.xlabel('Time (min)')
        plt.ylabel('Visited customers')
        plt.title('Arrival times vs time windows')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / '12_time_windows.png', dpi=180)
        plt.close()


def plot_routes(instance, solution, output_dir, name='11_best_routes_map'):
    depot_lat = float(instance.depot['DEPOT_LATITUDE'])
    depot_lon = float(instance.depot['DEPOT_LONGITUDE'])
    clat = instance.customers['CUSTOMER_LATITUDE'].astype(float).to_numpy()
    clon = instance.customers['CUSTOMER_LONGITUDE'].astype(float).to_numpy()

    plt.figure(figsize=(9, 7))
    plt.scatter([depot_lon], [depot_lat], marker='s', s=100, label='Depot')
    plt.scatter(clon, clat, s=30, label='Clients')

    for k, route in enumerate(solution.routes):
        if not route:
            continue
        xs = [depot_lon] + [clon[i] for i in route] + [depot_lon]
        ys = [depot_lat] + [clat[i] for i in route] + [depot_lat]
        plt.plot(xs, ys, linewidth=1.6, label=f'Vehicle {k + 1}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Routes finales - route_id {instance.route_id}')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{name}.png', dpi=180)
    plt.close()


def save_summary(instance, results, best, output_dir):
    rows = []
    for res in results:
        rows.append({
            'agent': res.name,
            'best_objective_fx': res.best.objective,
            'best_penalties': res.best.penalties,
            'best_penalized': res.best.penalized,
            'enemy_observations': res.enemy_observations,
            'enemy_improvements': res.enemy_improvements,
        })
    rows.append({
        'agent': 'global_best',
        'best_objective_fx': best.objective,
        'best_penalties': best.penalties,
        'best_penalized': best.penalized,
        'enemy_observations': np.nan,
        'enemy_improvements': np.nan,
    })
    pd.DataFrame(rows).to_csv(Path(output_dir) / 'summary_agents.csv', index=False)

    route_rows = []
    for k, route in enumerate(best.routes):
        weight = float(np.sum(instance.customer_weight[route])) if route else 0.0
        volume = float(np.sum(instance.customer_volume[route])) if route else 0.0
        dist = route_distance(route, k, instance)
        route_rows.append({
            'vehicle_idx': k,
            'vehicle_code': instance.vehicle_codes[k],
            'n_customers': len(route),
            'distance_km': dist,
            'fixed_cost_omega_k': instance.fixed_cost[k] if route else 0.0,
            'load_weight_kg': weight,
            'cap_weight_kg': instance.cap_weight[k],
            'load_volume_m3': volume,
            'cap_volume_m3': instance.cap_volume[k],
            'customer_sequence': ' -> '.join(instance.idx_to_code[i] for i in route)
        })
    pd.DataFrame(route_rows).to_csv(Path(output_dir) / 'best_routes_detail.csv', index=False)


def run_sma_enemie(instance, output_dir, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    pr = max(3.0, round(len(instance.customer_codes) * 0.20))
    pool = EnemyPool(max_size=18, pool_radius=pr)

    ga = genetic_agent(instance, pool, seed=seed)
    sa = simulated_annealing_agent(instance, pool, seed=seed + 1)
    tb = tabu_agent(instance, pool, seed=seed + 2)

    results = [ga, sa, tb]
    pool_best = pool.best()
    candidates = [r.best for r in results]
    if pool_best is not None:
        candidates.append(pool_best)
    best = min(candidates, key=lambda s: s.penalized)

    plot_convergence(results, output_dir)
    plot_pool_metrics(pool, output_dir)
    plot_agent_scores(results, output_dir)
    plot_enemy_activity(results, output_dir)
    plot_solution_diagnostics(instance, best, output_dir)
    plot_routes(instance, best, output_dir)
    save_summary(instance, results, best, output_dir)

    return results, best, pool


def choose_route_id(customers_df, route_id=None):
    routes = sorted(pd.Series(customers_df['ROUTE_ID']).dropna().unique().tolist())
    if not routes:
        raise ValueError('Nenhuma ROUTE_ID encontrada')
    if route_id is None:
        return int(DEFAULT_ROUTE_ID)
    if route_id not in routes:
        raise ValueError(f'ROUTE_ID {route_id} não encontrada. Disponíveis: {routes}')
    return int(route_id)


def load_instance_from_folder(route_id=None):
    files = detect_dataset('..\\database\\')
    customers = read_table(files['customers'])
    vehicles = read_table(files['vehicles'])
    depots = read_table(files['depots'])
    constraints = read_table(files['constraints'])
    distances = read_table(files['distances'])
    rid = choose_route_id(customers, route_id=route_id)
    return build_instance(customers, vehicles, depots, constraints, distances, rid)


def print_report(instance, results, best, elapsed):
    print('=' * 72)
    print('SMA VRP - Concurrence Ennemie')
    print('=' * 72)
    print(f'ROUTE_ID: {instance.route_id}')
    print(f'Clients: {len(instance.customer_codes)} | Vehicles: {len(instance.vehicle_codes)}')
    print(f'Temps total: {elapsed:.2f}s')
    print('-' * 72)
    for res in results:
        print(f"{res.name:<12} penalized={res.best.penalized:>12.2f} | f(x)={res.best.objective:>12.2f} | penalties={res.best.penalties:>10.2f} | enemy_obs={res.enemy_observations:>3d} | q_gains={res.enemy_improvements:>3d}")
    print('-' * 72)
    print(f"global_best   penalized={best.penalized:>12.2f} | f(x)={best.objective:>12.2f} | penalties={best.penalties:>10.2f}")
    print('=' * 72)


# if __name__ == '__main__':
#     root = Path(__file__).resolve().parent
#     route_id_env = os.getenv('ROUTE_ID')
#     route_id = int(route_id_env) if route_id_env else DEFAULT_ROUTE_ID
#     output_dir = root / 'resultats_images' / f'sma_enemie_qlearning_route_{route_id}'

#     t0 = time.time()
#     instance = load_instance_from_folder(route_id=route_id)
#     results, best, pool = run_sma_enemie(instance, output_dir=output_dir, seed=42)
#     elapsed = time.time() - t0
#     print_report(instance, results, best, elapsed)


def save_report_table_image(instance, results, best, elapsed, out_path):
    """
    Gera uma imagem de tabela com o relatório de resultados e salva no disco.
    """
    # 1. Preparar os dados para a tabela
    data = []
    for res in results:
        data.append([
            res.name,
            f"{res.best.penalized:.2f}",
            f"{res.best.objective:.2f}",
            f"{res.best.penalties:.2f}",
            str(res.enemy_imports),
            str(res.enemy_improvements)
        ])
    
    # Adicionar o "global_best" como a última linha
    data.append([
        "global_best",
        f"{best.penalized:.2f}",
        f"{best.objective:.2f}",
        f"{best.penalties:.2f}",
        "-",  # Sem imports para o global best
        "-"   # Sem gains para o global best
    ])
    
    # Nomes das colunas
    columns = ["Agent", "Penalized", "f(x) Objective", "Penalties", "Imports", "Gains"]
    df = pd.DataFrame(data, columns=columns)
    
    # 2. Configurar o plot usando matplotlib
    # Ajustar a altura da imagem dinamicamente baseada na quantidade de agentes
    fig_height = 2 + len(data) * 0.5
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Esconder os eixos normais do gráfico
    ax.axis('off')
    ax.axis('tight')
    
    # Desenhar a tabela
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8) # Estica um pouco as células para ficar legível
    
    # Colorir os cabeçalhos para ficar mais bonito
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
    
    # 3. Adicionar o cabeçalho (Metadados) como título da imagem
    title_text = (
        f"SMA VRP - Collaboration Amis\n"
        f"ROUTE_ID: {instance.route_id} | "
        f"Clients: {len(instance.customer_codes)} | "
        f"Vehicles: {len(instance.vehicle_codes)} | "
        f"Temps total: {elapsed:.2f}s"
    )
    plt.title(title_text, fontsize=12, pad=20, fontweight='bold')
    
    # 4. Salvar a imagem no diretório
    # out_path = Path(output_dir)
    # out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"13_report_table_route_{instance.route_id}.png"
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Tabela de resumo salva como imagem em: {file_path}")



if __name__ == '__main__':
    root = Path(__file__).resolve().parent

    # ── Grande taille ────────────────────────────────────────────────
    customersDf = pd.read_excel("..\\database\\2_detail_table_customers.xls")

    ALL_ROUTES = sorted(customersDf["ROUTE_ID"].unique().tolist())

    customers = pd.read_excel('..\\database\\2_detail_table_customers.xls')
    vehicles = pd.read_excel('..\\database\\3_detail_table_vehicles.xls')
    depots = pd.read_excel('..\\database\\4_detail_table_depots.xls')
    constraints = pd.read_excel('..\\database\\5_detail_table_constraints_sdvrp.xls')
    distances = pd.read_excel('..\\database\\6_detail_table_cust_depots_distances.xls')

    for route_id in ALL_ROUTES:
        # route_id_env = os.getenv('ROUTE_ID')
        # route_id = int(route_id_env) if route_id_env else DEFAULT_ROUTE_ID
        rid = choose_route_id(customers, route_id=route_id)
        output_dir = root / 'resultats_images' / 'amis' / f'sma_appr_amis_route_{route_id}'

        t0 = time.time()
        instance = build_instance(customers, vehicles, depots, constraints, distances, rid)
        results, best, pool = run_sma_enemie(instance, output_dir=output_dir, seed=42)
        elapsed = time.time() - t0
        save_report_table_image(instance, results, best, elapsed, out_path=Path(output_dir))
        print_report(instance, results, best, elapsed)