import math
import random

# 1. Definindo o problema: Coordenadas (x, y) de 5 pontos de entrega
pontos = [(0, 0), (2, 3), (5, 2), (6, 6), (1, 5)]

def calcular_tempo_total(rota):
    tempo = 0
    for i in range(len(rota) - 1):
        p1, p2 = rota[i], rota[i+1]
        # Distância euclidiana simples como representação do tempo
        tempo += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return tempo

def simulated_annealing(pontos_iniciais):
    # Configurações iniciais
    solucao_atual = pontos_iniciais[:]
    random.shuffle(solucao_atual) # Começa com uma rota aleatória
    temp = 100.0
    resfriamento = 0.95 # Fator de redução da temperatura
    temp_minima = 0.01
    print(solucao_atual)
    while temp > temp_minima:
        # Criar vizinho: trocando dois pontos de lugar
        nova_solucao = solucao_atual[:]
        idx1, idx2 = random.sample(range(len(nova_solucao)), 2)
        nova_solucao[idx1], nova_solucao[idx2] = nova_solucao[idx2], nova_solucao[idx1]
        
        # Calcular custos
        custo_atual = calcular_tempo_total(solucao_atual)
        novo_custo = calcular_tempo_total(nova_solucao)
        
        # Critério de aceitação
        if novo_custo < custo_atual:
            solucao_atual = nova_solucao
        else:
            # Se for pior, aceita com uma probabilidade
            probabilidade = math.exp((custo_atual - novo_custo) / temp)
            if random.random() < probabilidade:
                solucao_atual = nova_solucao
        
        # Esfriar
        temp *= resfriamento
        
    return solucao_atual, calcular_tempo_total(solucao_atual)

# Execução
melhor_rota, menor_tempo = simulated_annealing(pontos)

print(f"Melhor rota encontrada: {melhor_rota}")
print(f"Tempo total: {menor_tempo:.2f}")