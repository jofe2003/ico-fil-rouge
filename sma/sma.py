# -*- coding: utf-8 -*-
"""
SMA (Sistema Multiagente) - Hibridização de Metaheurísticas para VRP
Agentes: Algoritmo Genético (AG), Recozimento Simulado (RS), Busca Tabu (TS)
Estratégia: Interação "Amis" (Pool de Soluções Compartilhado / EMP)
"""

import pandas as pd
import numpy as np
import random
import math
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# CONFIGURAÇÕES E PARÂMETROS GLOBAIS
# =====================================================================
OMEGA = 500.0  # Fator de penalidade por veículo utilizado
PENALTY_FACTOR = 500.0 # Fator para violações (Janela de tempo, Capacidade)
POOL_RADIUS_PR = 15    # Parâmetro 'pr' do slide: raio mínimo de diferença de arcos
POOL_MAX_SIZE = 10     # Capacidade máxima do Espace Mémoire Partagé (EMP)

MACRO_ITERATIONS = 30  # Quantas vezes o coordenador SMA vai acionar os agentes
MICRO_ITERATIONS_GA = 10 # Gerações do AG por macro-iteração
MICRO_ITERATIONS_RS = 500 # Iterações do RS por macro-iteração
MICRO_ITERATIONS_TS = 20 # Iterações do Tabu por macro-iteração

OUT_DIR = "resultats_images"
os.makedirs(OUT_DIR, exist_ok=True)

def build_matrices(customers, vehicles, depot, distances):
    n = len(customers)
    lats = customers["CUSTOMER_LATITUDE"].values
    lons = customers["CUSTOMER_LONGITUDE"].values
    codes = customers["CUSTOMER_CODE"].astype(str).tolist()
    code2idx = {c: i+1 for i, c in enumerate(codes)} # 1-based (0 is depot)
    
    # Haversine fallback
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dist_cc = np.zeros((n+1, n+1)); time_cc = np.zeros((n+1, n+1))
    
    # Clientes entre si
    for i in range(n):
        for j in range(n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            dist_cc[i+1][j+1] = d
            time_cc[i+1][j+1] = d / 40.0 * 60.0

    # Depósito
    d_lat, d_lon = depot["DEPOT_LATITUDE"], depot["DEPOT_LONGITUDE"]
    for i in range(n):
        d = haversine(d_lat, d_lon, lats[i], lons[i])
        dist_cc[0][i+1] = dist_cc[i+1][0] = d
        time_cc[0][i+1] = time_cc[i+1][0] = d / 40.0 * 60.0

    # Sobrescreve com distâncias reais (se existirem)
    for _, row in distances.iterrows():
        code = str(row["CUSTOMER_CODE"])
        if code in code2idx:
            idx = code2idx[code]
            if "DEPOT->CUSTOMER" in str(row["DIRECTION"]):
                dist_cc[0][idx] = row["DISTANCE_KM"]
                time_cc[0][idx] = row["TIME_DISTANCE_MIN"]
            else:
                dist_cc[idx][0] = row["DISTANCE_KM"]
                time_cc[idx][0] = row["TIME_DISTANCE_MIN"]
                
    return dist_cc, time_cc

# =====================================================================
# FUNÇÕES DE CUSTO E PENALIDADE (f(x) = wK(x) + E c_ij)
# =====================================================================
def get_solution_cost(routes, vehicles, dist_cc, time_cc, customers, constraints):
    objective = 0.0
    penalty = 0.0
    
    for k, route in enumerate(routes):
        if len(route) <= 2: # Apenas [0, 0]
            continue
            
        veh = vehicles.iloc[k]
        objective += veh["VEHICLE_FIXED_COST_KM"] # w * K(x)
        
        tw_w, tw_v = 0.0, 0.0
        cur_t = veh["VEHICLE_AVAILABLE_TIME_FROM_MIN"]
        var_cost = veh["VEHICLE_VARIABLE_COST_KM"]
        
        for pos in range(len(route) - 1):
            i, j = route[pos], route[pos+1]
            dist_arc = dist_cc[i][j]
            objective += var_cost * dist_arc # c_ij
            
            cur_t += time_cc[i][j]
            
            if j != 0:
                cust = customers.iloc[j-1]
                tw_w += cust["TOTAL_WEIGHT_KG"]
                tw_v += cust["TOTAL_VOLUME_M3"]
                
                # Penalidade Janela de Tempo
                if cur_t > cust["CUSTOMER_TIME_WINDOW_TO_MIN"]:
                    penalty += PENALTY_FACTOR * (cur_t - cust["CUSTOMER_TIME_WINDOW_TO_MIN"])
                cur_t = max(cur_t, cust["CUSTOMER_TIME_WINDOW_FROM_MIN"])
                cur_t += cust["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"]
                
                # Penalidade SDVRP OTIMIZADA
                if constraints:
                    if (str(cust['CUSTOMER_CODE']), str(veh['VEHICLE_CODE'])) in constraints:
                        penalty += PENALTY_FACTOR * 2

        # Penalidade Capacidade
        if tw_w > veh["VEHICLE_TOTAL_WEIGHT_KG"]:
            penalty += PENALTY_FACTOR * (tw_w - veh["VEHICLE_TOTAL_WEIGHT_KG"])
        if tw_v > veh["VEHICLE_TOTAL_VOLUME_M3"]:
            penalty += PENALTY_FACTOR * (tw_v - veh["VEHICLE_TOTAL_VOLUME_M3"])

    return objective, penalty

# =====================================================================
# ESPACE MÉMOIRE PARTAGÉ (EMP) - SMA
# =====================================================================
def get_arcs(routes):
    """Extrai o conjunto de arcos de uma solução."""
    arcs = set()
    for route in routes:
        if len(route) > 2:
            for i in range(len(route) - 1):
                arcs.add((route[i], route[i+1]))
    return arcs

def calc_lambda(arcs1, arcs2):
    """Calcula lambda_ij: quantidade de arcos não comuns entre i e j."""
    return len(arcs1 ^ arcs2) # Diferença simétrica

class EMP:
    def __init__(self, pr_radius, max_size):
        self.pool = [] # Lista de dicionários: {'routes': [], 'cost': 0.0, 'penalty': 0.0, 'arcs': set()}
        self.pr = pr_radius
        self.max_size = max_size
        
    def evaluate_diversity_g(self, new_arcs):
        """Calcula g(phi_i) conforme os slides."""
        g_val = 0.0
        for sol in self.pool:
            lambda_ij = calc_lambda(new_arcs, sol['arcs'])
            if lambda_ij <= self.pr:
                phi = 1.0 - (lambda_ij / self.pr)
                g_val += phi
        return g_val

    def try_add(self, routes, cost, penalty):
        arcs = get_arcs(routes)
        total_fit = cost + penalty
        
        # Ignora se for idêntica (lambda == 0) a alguma
        for s in self.pool:
            if calc_lambda(arcs, s['arcs']) == 0:
                return False

        g_diversity = self.evaluate_diversity_g(arcs)
        new_entry = {'routes': [r[:] for r in routes], 'cost': cost, 'penalty': penalty, 'total': total_fit, 'arcs': arcs}

        if len(self.pool) < self.max_size:
            self.pool.append(new_entry)
            self.pool.sort(key=lambda x: x['total'])
            return True
        else:
            # Se compromete a diversidade (g > 0), só entra se o custo for absurdamente melhor
            # Substitui a pior solução
            if total_fit < self.pool[-1]['total']:
                # Penaliza inserções que prejudicam muito a diversidade
                if g_diversity < 1.0 or (total_fit < self.pool[0]['total']):
                    self.pool[-1] = new_entry
                    self.pool.sort(key=lambda x: x['total'])
                    return True
        return False

    def get_friend_solution(self):
        """Retorna uma solução boa e aleatória do pool para colaboração (Amis)."""
        if not self.pool: return None
        # Escolhe tendenciosamente entre as melhores
        idx = int(random.triangular(0, len(self.pool)-1, 0))
        return [r[:] for r in self.pool[idx]['routes']]

# =====================================================================
# FUNÇÕES COMUNS DE VIZINHANÇA
# =====================================================================
def op_relocate(routes):
    s = [r[:] for r in routes]
    valid_routes = [i for i, r in enumerate(s) if len(r) > 2]
    if not valid_routes: return s
    r1 = random.choice(valid_routes)
    r2 = random.choice(range(len(s)))
    if len(s[r1]) <= 2: return s
    idx_from = random.randint(1, len(s[r1])-2)
    client = s[r1].pop(idx_from)
    idx_to = random.randint(1, len(s[r2])-1)
    s[r2].insert(idx_to, client)
    return s

def op_swap_inter(routes):
    s = [r[:] for r in routes]
    valid_routes = [i for i, r in enumerate(s) if len(r) > 2]
    if len(valid_routes) < 2: return s
    r1, r2 = random.sample(valid_routes, 2)
    idx1 = random.randint(1, len(s[r1])-2)
    idx2 = random.randint(1, len(s[r2])-2)
    s[r1][idx1], s[r2][idx2] = s[r2][idx2], s[r1][idx1]
    return s

# =====================================================================
# AGENTES METAHEURÍSTICOS
# =====================================================================
class AgentGA:
    def __init__(self, customers, vehicles, dist_cc, time_cc, constraints):
        self.pop_size = 20
        self.customers, self.vehicles = customers, vehicles
        self.dist_cc, self.time_cc = dist_cc, time_cc
        self.constraints = constraints
        self.population = []
        
    def initialize(self):
        # Cria população inicial caótica
        n_clients = len(self.customers)
        for _ in range(self.pop_size):
            clientes = list(range(1, n_clients + 1))
            random.shuffle(clientes)
            routes = [[0] for _ in range(len(self.vehicles))]
            for i, c in enumerate(clientes):
                routes[i % len(self.vehicles)].append(c)
            for r in routes: r.append(0)
            self.population.append(routes)

    def step(self, emp: EMP, iterations):
        best_local = None
        best_fit = float('inf')
        
        # Colaboração Amis: Puxa do EMP para injetar na população
        friend = emp.get_friend_solution()
        if friend:
            self.population[random.randint(0, self.pop_size-1)] = friend

        for _ in range(iterations):
            # Avaliação
            fits = []
            for sol in self.population:
                obj, pen = get_solution_cost(sol, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
                fits.append(obj + pen)
                if obj + pen < best_fit:
                    best_fit, best_local = obj + pen, sol

            # Seleção por torneio e mutação simples (Relocate)
            new_pop = [best_local] # Elitismo
            while len(new_pop) < self.pop_size:
                c1, c2 = random.sample(list(zip(self.population, fits)), 2)
                parent = c1[0] if c1[1] < c2[1] else c2[0]
                child = op_relocate(parent) if random.random() < 0.6 else op_swap_inter(parent)
                new_pop.append(child)
            self.population = new_pop

        # Interação Amis: Tenta enviar a melhor para o EMP
        if best_local:
            obj, pen = get_solution_cost(best_local, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
            emp.try_add(best_local, obj, pen)
        return best_local, best_fit

class AgentRS:
    def __init__(self, customers, vehicles, dist_cc, time_cc, constraints):
        self.customers, self.vehicles = customers, vehicles
        self.dist_cc, self.time_cc = dist_cc, time_cc
        self.constraints = constraints
        self.current = None
        self.current_fit = float('inf')
        self.T = 1000.0

    def initialize(self):
        n_clients = len(self.customers)
        clientes = list(range(1, n_clients + 1))
        routes = [[0] for _ in range(len(self.vehicles))]
        for i, c in enumerate(clientes): routes[i % len(self.vehicles)].append(c)
        for r in routes: r.append(0)
        self.current = routes
        obj, pen = get_solution_cost(self.current, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
        self.current_fit = obj + pen

    def step(self, emp: EMP, iterations):
        # Colaboração Amis: Puxa do EMP se estiver estagnado
        if random.random() < 0.1:
            friend = emp.get_friend_solution()
            if friend:
                obj, pen = get_solution_cost(friend, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
                self.current, self.current_fit = friend, obj + pen
                self.T = 500.0 # Re-heat

        best_local = self.current
        best_fit = self.current_fit

        for _ in range(iterations):
            neighbor = op_relocate(self.current) if random.random() < 0.5 else op_swap_inter(self.current)
            obj, pen = get_solution_cost(neighbor, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
            n_fit = obj + pen
            
            delta = n_fit - self.current_fit
            if delta < 0 or random.random() < math.exp(-delta / max(0.001, self.T)):
                self.current = neighbor
                self.current_fit = n_fit
                if n_fit < best_fit:
                    best_local, best_fit = neighbor, n_fit
            
            self.T *= 0.99 # Resfriamento

        emp.try_add(best_local, *get_solution_cost(best_local, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints))
        return best_local, best_fit

class AgentTabou:
    def __init__(self, customers, vehicles, dist_cc, time_cc, constraints):
        self.customers, self.vehicles = customers, vehicles
        self.dist_cc, self.time_cc = dist_cc, time_cc
        self.constraints = constraints
        self.current = None
        self.tabu_list = []
        self.tabu_size = 10

    def initialize(self):
        n_clients = len(self.customers)
        clientes = list(range(1, n_clients + 1))
        routes = [[0] for _ in range(len(self.vehicles))]
        for i, c in enumerate(clientes): routes[i % len(self.vehicles)].append(c)
        for r in routes: r.append(0)
        self.current = routes

    def step(self, emp: EMP, iterations):
        # Colaboração Amis
        friend = emp.get_friend_solution()
        if friend and random.random() < 0.2:
            self.current = friend
            self.tabu_list.clear()

        best_local = self.current
        obj, pen = get_solution_cost(best_local, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
        best_fit = obj + pen

        for _ in range(iterations):
            neighbors = [op_relocate(self.current) for _ in range(5)]
            best_n = None
            best_n_fit = float('inf')
            
            for n in neighbors:
                obj, pen = get_solution_cost(n, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints)
                fit = obj + pen
                # Simplificação da lista Tabu focada no fitness do vizinho
                if fit not in self.tabu_list or fit < best_fit: 
                    if fit < best_n_fit:
                        best_n = n
                        best_n_fit = fit
                        
            if best_n:
                self.current = best_n
                self.tabu_list.append(best_n_fit)
                if len(self.tabu_list) > self.tabu_size: self.tabu_list.pop(0)
                if best_n_fit < best_fit:
                    best_local, best_fit = best_n, best_n_fit

        emp.try_add(best_local, *get_solution_cost(best_local, self.vehicles, self.dist_cc, self.time_cc, self.customers, self.constraints))
        return best_local, best_fit
    
# =====================================================================
# FUNÇÕES DE PLOTAGEM (GRÁFICOS)
# =====================================================================
def plotar_convergencia(history, titulo, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, color='purple', linewidth=2, marker='o')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel("Macro Iterações (AG + RS + TS)")
    plt.ylabel("Custo Total (Fitness com Penalidades)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300) # Salva a imagem
    plt.close() # FECHA a imagem da memória para não travar o loop

def plotar_rotas(routes, customers, depot, titulo, filepath):
    plt.figure(figsize=(12, 8))
    
    # Plota o depósito (Quadrado Vermelho)
    d_lat = depot["DEPOT_LATITUDE"]
    d_lon = depot["DEPOT_LONGITUDE"]
    plt.scatter(d_lon, d_lat, c='red', marker='s', s=150, label='Depósito', zorder=5)
    
    # Pega as coordenadas dos clientes (Bolinhas Pretas)
    c_lats = customers["CUSTOMER_LATITUDE"].values
    c_lons = customers["CUSTOMER_LONGITUDE"].values
    plt.scatter(c_lons, c_lats, c='black', marker='o', s=30, label='Clientes', zorder=2)
    
    # Plota cada veículo com uma cor diferente
    colors = plt.cm.tab20.colors
    patches = []
    
    for k, route in enumerate(routes):
        if len(route) > 2: # Ignora veículos que não saíram da garagem
            color = colors[k % len(colors)]
            rota_lons, rota_lats = [], []
            
            for node in route:
                if node == 0:
                    rota_lons.append(d_lon)
                    rota_lats.append(d_lat)
                else:
                    # node - 1 porque o cliente 1 está no index 0 da lista c_lons
                    rota_lons.append(c_lons[node-1])
                    rota_lats.append(c_lats[node-1])
            
            # Desenha a linha da rota
            plt.plot(rota_lons, rota_lats, color=color, linewidth=2, zorder=3, alpha=0.8)
            patches.append(mpatches.Patch(color=color, label=f'Veículo {k+1}'))
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Organiza a legenda (misturando os pontos e as linhas)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles + patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300) # Salva o mapa
    plt.close() # Limpa a memória

# =====================================================================
# CÉREBRO: O SISTEMA MULTIAGENTE (SMA)
# =====================================================================
def run_sma_all_routes():
    print("="*60)
    print("Iniciando SMA - Processamento de Múltiplas Rotas")
    print("="*60)
    
    # 1. Carregamento dos dados 
    df_customers = pd.read_excel("2_detail_table_customers.xls")
    df_vehicles  = pd.read_excel("3_detail_table_vehicles.xls")
    df_depots    = pd.read_excel("4_detail_table_depots.xls")
    df_dist      = pd.read_excel("6_detail_table_cust_depots_distances.xls")
    df_const     = pd.read_excel("5_detail_table_constraints_sdvrp.xls")
    
    # 2. Pegar a lista de todos os IDs de rotas únicos
    route_ids = df_customers["ROUTE_ID"].unique()
    print(f"Total de rotas identificadas: {len(route_ids)}")
    
    # 3. Iniciar o Loop Principal
    for ROUTE_ID in route_ids:
        print("\n" + "="*60)
        print(f"🚀 PROCESSANDO A ROTA: {ROUTE_ID}")
        print("="*60)
        
        # Filtra os dados EXCLUSIVOS desta rota
        customers = df_customers[df_customers["ROUTE_ID"] == ROUTE_ID]
        vehicles = df_vehicles[df_vehicles["ROUTE_ID"] == ROUTE_ID]
        
        depots_route = df_depots[df_depots["ROUTE_ID"] == ROUTE_ID]
        if depots_route.empty:
            print(f"⚠️ Nenhum depósito encontrado para a rota {ROUTE_ID}. Pulando...")
            continue
        depot = depots_route.iloc[0]
        
        # ====================================================================
        # 👇 É EXATAMENTE AQUI QUE VOCÊ COLA O CÓDIGO QUE MANDOU 👇
        # ====================================================================
        
        # Constrói as matrizes de distância apenas com os dados filtrados
        dist_cc, time_cc = build_matrices(customers, vehicles, depot, df_dist)
        
        # ==========================================================
        # NOVO: Converte o DataFrame de restrições em um Set otimizado
        # ==========================================================
        const_set = set()
        # Garante que só pegaremos as restrições desta rota específica
        route_const = df_const[df_const["ROUTE_ID"] == ROUTE_ID] if "ROUTE_ID" in df_const.columns else df_const
            
        for _, row in route_const.iterrows():
            cliente = str(row['SDVRP_CONSTRAINT_CUSTOMER_CODE'])
            veiculo = str(row['SDVRP_CONSTRAINT_VEHICLE_CODE'])
            const_set.add((cliente, veiculo))
        # ==========================================================

        # Reinicia o EMP e os Agentes passando o 'const_set' (muito mais rápido que o df_const)
        emp = EMP(pr_radius=POOL_RADIUS_PR, max_size=POOL_MAX_SIZE)
        ag = AgentGA(customers, vehicles, dist_cc, time_cc, const_set)
        rs = AgentRS(customers, vehicles, dist_cc, time_cc, const_set)
        tb = AgentTabou(customers, vehicles, dist_cc, time_cc, const_set)

        # ====================================================================
        # 👆 O SEU CÓDIGO TERMINA AQUI 👆
        # ====================================================================

        # Agora continua a inicialização e o loop da rota normalmente...
        ag.initialize()
        rs.initialize()
        tb.initialize()
        
        history_best = []
        
        # Roda o SMA para a rota atual
        for i in range(MACRO_ITERATIONS):
            best_ag, fit_ag = ag.step(emp, MICRO_ITERATIONS_GA)
            best_rs, fit_rs = rs.step(emp, MICRO_ITERATIONS_RS)
            best_tb, fit_tb = tb.step(emp, MICRO_ITERATIONS_TS)
            
            if emp.pool:
                global_best = emp.pool[0]['total']
            else:
                global_best = min(fit_ag, fit_rs, fit_tb)
                
            history_best.append(global_best)
            
        # Pega a melhor solução desta rota
        best_overall = emp.pool[0]
        print(f"✅ Rota {ROUTE_ID} Otimizada! Custo: {best_overall['cost']:.2f}")

        # ==========================================================
        # GERAÇÃO DOS GRÁFICOS
        # ==========================================================
        print(f"Gerando imagens para a rota {ROUTE_ID}...")
        
        # Cria os nomes de arquivos únicos para esta rota
        caminho_conv = os.path.join(OUT_DIR, f"Convergencia_{ROUTE_ID}.png")
        caminho_mapa = os.path.join(OUT_DIR, f"Mapa_{ROUTE_ID}.png")
        
        # Chama as funções
        plotar_convergencia(history_best, titulo=f"Convergência SMA - Rota {ROUTE_ID}", filepath=caminho_conv)
        plotar_rotas(best_overall['routes'], customers, depot, titulo=f"Mapa de Rotas SMA - {ROUTE_ID}", filepath=caminho_mapa)
        
        print(f"Gráficos salvos na pasta '{OUT_DIR}'!\n")

if __name__ == "__main__":
    run_sma_all_routes()
