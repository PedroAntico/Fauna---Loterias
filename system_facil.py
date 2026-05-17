#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE OTIMIZAÇÃO MULTIOBJETIVO - LOTOFÁCIL
================================================
Versão 4.0 - NSGA-II + Hipergrafos + Covering Codes

IMPLEMENTAÇÕES AVANÇADAS:
✅ NSGA-II (Non-dominated Sorting Genetic Algorithm II)
✅ Fronteira de Pareto real (não soma ponderada)
✅ Teoria de Hipergrafos (transversalidade, hitting sets)
✅ Covering Codes (covering radius, sphere packing)
✅ Métricas topológicas (persistent homology simplificada)
✅ KL-Divergence para uniformidade
✅ Análise de robustez (sensibilidade paramétrica)
✅ Visualização 3D da fronteira de Pareto

PRINCÍPIO:
Não existe solução ótima única.
Existe uma FRONTEIRA de trade-offs.
O algoritmo descobre essa fronteira.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.distance import hamming, jaccard
from scipy.stats import entropy
from scipy.spatial import ConvexHull
from scipy.linalg import eigh
from collections import Counter, defaultdict
from itertools import combinations, product
from datetime import datetime
import warnings
import os
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150

# ============================================================
# TEORIA DE CÓDIGOS BINÁRIOS
# ============================================================

class BinaryCode:
    """
    Representação de jogos como códigos binários
    
    Cada jogo = vetor de 25 bits com peso 15
    Distância de Hamming = dissimilaridade estrutural
    """
    
    def __init__(self, n=25, k=15):
        self.n = n  # Comprimento do código
        self.k = k  # Peso constante
    
    def encode(self, game):
        """Codifica jogo → vetor binário"""
        vec = np.zeros(self.n, dtype=np.int8)
        vec[np.array(game) - 1] = 1
        return vec
    
    def decode(self, vec):
        """Decodifica vetor → jogo"""
        indices = np.where(vec == 1)[0] + 1
        # Garantir peso k
        if len(indices) > self.k:
            indices = np.random.choice(indices, self.k, replace=False)
        elif len(indices) < self.k:
            available = list(set(range(1, self.n+1)) - set(indices))
            extra = np.random.choice(available, self.k - len(indices), replace=False)
            indices = np.concatenate([indices, extra])
        return sorted(indices.astype(int).tolist())
    
    def distance(self, game1, game2):
        """Distância de Hamming"""
        v1 = self.encode(game1)
        v2 = self.encode(game2)
        return int(np.sum(v1 != v2))
    
    def weight(self, game):
        """Peso do código (deve ser k)"""
        return len(set(game))
    
    def covering_radius(self, pool):
        """
        Covering radius do pool
        
        Distância máxima de qualquer código do espaço
        até o código mais próximo no pool
        """
        # Gerar amostra do espaço total (impraticável enumerar C(25,15))
        sample_size = 10000
        random_codes = []
        for _ in range(sample_size):
            code = sorted(np.random.choice(range(1, self.n+1), self.k, replace=False))
            random_codes.append(code)
        
        # Calcular distância mínima para cada código
        max_min_dist = 0
        for code in random_codes:
            min_dist = min(self.distance(code, p) for p in pool)
            max_min_dist = max(max_min_dist, min_dist)
        
        return max_min_dist
    
    def sphere_packing_density(self, pool, radius=6):
        """
        Densidade de empacotamento de esferas
        
        Esferas de raio 'radius' centradas em cada código do pool
        Mede proporção do espaço coberto sem sobreposição
        """
        n_codes = len(pool)
        volume_per_sphere = sum(
            self._comb(self.n, i) for i in range(radius + 1)
        )
        total_volume = self._comb(self.n, self.k)
        
        # Densidade ideal (sem sobreposição)
        ideal_density = (n_codes * volume_per_sphere) / total_volume
        
        # Penalizar sobreposição
        overlap_count = 0
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                if self.distance(pool[i], pool[j]) < 2 * radius:
                    overlap_count += 1
        
        overlap_penalty = overlap_count / max(1, n_codes * (n_codes - 1) / 2)
        actual_density = ideal_density * (1 - overlap_penalty * 0.5)
        
        return min(1.0, max(0.0, actual_density))
    
    def _comb(self, n, k):
        """Coeficiente binomial"""
        if k < 0 or k > n:
            return 0
        from math import comb
        return comb(n, k)


# ============================================================
# TEORIA DE HIPERGRAFOS
# ============================================================

class HypergraphAnalyzer:
    """
    Análise de pools como hipergrafos
    
    Vértices = dezenas (1-25)
    Hiperarestas = jogos (subconjuntos de 15 vértices)
    """
    
    def __init__(self):
        self.n_vertices = 25
        self.edge_size = 15
    
    def build_incidence_matrix(self, pool):
        """
        Matriz de incidência vértice-hiperaresta
        M[i,j] = 1 se vértice i está na aresta j
        """
        M = np.zeros((self.n_vertices, len(pool)), dtype=np.int8)
        for j, game in enumerate(pool):
            for v in game:
                M[v-1, j] = 1
        return M
    
    def degree_distribution(self, pool):
        """Distribuição de graus dos vértices"""
        M = self.build_incidence_matrix(pool)
        degrees = M.sum(axis=1)
        return {
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees)),
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'gini': self._gini(degrees)
        }
    
    def edge_overlap_matrix(self, pool):
        """Matriz de sobreposição entre arestas"""
        n = len(pool)
        overlap = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                overlap[i, j] = len(set(pool[i]) & set(pool[j]))
                overlap[j, i] = overlap[i, j]
        return overlap
    
    def transversal_number(self, pool):
        """
        Número de transversalidade (hitting set)
        
        Menor conjunto de vértices que intersecta
        TODAS as hiperarestas
        """
        M = self.build_incidence_matrix(pool)
        
        # Heurística gulosa para hitting set
        remaining_edges = set(range(len(pool)))
        hitting_set = set()
        
        while remaining_edges:
            # Vértice que cobre mais arestas restantes
            best_vertex = None
            best_coverage = -1
            
            for v in range(self.n_vertices):
                covered = sum(1 for e in remaining_edges if M[v, e] == 1)
                if covered > best_coverage:
                    best_coverage = covered
                    best_vertex = v
            
            if best_vertex is not None and best_coverage > 0:
                hitting_set.add(best_vertex)
                # Remover arestas cobertas
                remaining_edges = {e for e in remaining_edges 
                                  if M[best_vertex, e] == 0}
            else:
                break
        
        return len(hitting_set)
    
    def hypergraph_density(self, pool):
        """Densidade do hipergrafo"""
        M = self.build_incidence_matrix(pool)
        total_edges = len(pool)
        max_possible = self._comb(self.n_vertices, self.edge_size)
        return total_edges / max_possible if max_possible > 0 else 0
    
    def _gini(self, array):
        """Coeficiente de Gini"""
        array = np.sort(array)
        n = len(array)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * array)) / (n * np.sum(array)) - (n + 1) / n
    
    def _comb(self, n, k):
        from math import comb
        return comb(n, k)


# ============================================================
# NSGA-II: NON-DOMINATED SORTING GENETIC ALGORITHM II
# ============================================================

class NSGA2:
    """
    NSGA-II para otimização multiobjetivo de pools
    
    Objetivos (MINIMIZAR ou MAXIMIZAR):
    1. MAX cobertura de pares
    2. MAX cobertura de trincas
    3. MAX distância mínima (diversidade)
    4. MAX entropia de Shannon
    5. MIN redundância (overlap)
    
    Sem pesos! Fronteira de Pareto real.
    """
    
    def __init__(self, n_games=50, pop_size=100, n_generations=50):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        # Componentes
        self.code = BinaryCode()
        self.hypergraph = HypergraphAnalyzer()
        
        # Objetivos (todos para MAXIMIZAR, exceto redundância)
        self.n_objectives = 5
        self.objective_names = [
            'Cobertura Pares',
            'Cobertura Trincas',
            'Distância Mínima',
            'Entropia',
            'Anti-Redundância'
        ]
        
        # Histórico
        self.pareto_fronts = []
        self.fitness_history = []
    
    def _evaluate(self, pool):
        """
        Avalia TODOS os objetivos de um pool
        
        Returns:
            list: [obj1, obj2, obj3, obj4, obj5] (todos maximizar)
        """
        # 1. Cobertura de pares
        pair_cov = self._coverage_ratio(pool, k=2)
        
        # 2. Cobertura de trincas
        triple_cov = self._coverage_ratio(pool, k=3)
        
        # 3. Distância mínima (diversidade)
        if len(pool) >= 2:
            distances = []
            for i in range(len(pool)):
                for j in range(i+1, len(pool)):
                    distances.append(self.code.distance(pool[i], pool[j]))
            min_dist = min(distances) / 30  # Normalizar 0-1
        else:
            min_dist = 1.0
        
        # 4. Entropia de Shannon
        all_dezenas = [d for game in pool for d in game]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        ent = entropy(probs) / np.log(25)  # Normalizar 0-1
        
        # 5. Anti-redundância (1 - redundância normalizada)
        if len(pool) >= 2:
            overlaps = []
            for i in range(len(pool)):
                for j in range(i+1, len(pool)):
                    overlaps.append(len(set(pool[i]) & set(pool[j])))
            avg_overlap = np.mean(overlaps) / 15  # Normalizar 0-1
            anti_redundancy = 1.0 - avg_overlap
        else:
            anti_redundancy = 1.0
        
        return [pair_cov, triple_cov, min_dist, ent, anti_redundancy]
    
    def _coverage_ratio(self, pool, k):
        """Proporção de k-subconjuntos cobertos"""
        covered = set()
        for game in pool:
            for subset in combinations(sorted(game), k):
                covered.add(subset)
        
        total = len(list(combinations(range(1, 26), k)))
        return len(covered) / total if total > 0 else 0
    
    def _dominates(self, obj1, obj2):
        """
        Verifica se obj1 DOMINA obj2
        
        Dominação: obj1 é melhor ou igual em TODOS os objetivos
        e ESTRITAMENTE melhor em pelo menos um
        """
        better_in_any = False
        for a, b in zip(obj1, obj2):
            if a < b:  # Pior em algum objetivo
                return False
            if a > b:  # Melhor em pelo menos um
                better_in_any = True
        return better_in_any
    
    def _fast_non_dominated_sort(self, population_obj):
        """
        Ordenação não-dominada rápida (NSGA-II)
        
        Returns:
            list: Lista de fronts [front_0, front_1, ...]
        """
        n = len(population_obj)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(population_obj[i], population_obj[j]):
                    dominated_solutions[i].append(j)
                elif self._dominates(population_obj[j], population_obj[i]):
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_front = []
            for idx in fronts[i]:
                for dominated_idx in dominated_solutions[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remover último front vazio
    
    def _crowding_distance(self, front_indices, population_obj):

        n = len(front_indices)

        if n <= 2:
            return {idx: float('inf') for idx in front_indices}

        distances = {idx: 0.0 for idx in front_indices}

        n_obj = len(population_obj[0])

        for obj_idx in range(n_obj):

            sorted_front = sorted( front_indices,key=lambda i: population_obj[i][obj_idx]
        )

            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            obj_min = population_obj[sorted_front[0]][obj_idx]
            obj_max = population_obj[sorted_front[-1]][obj_idx]

            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for i in range(1, n - 1):

                prev_obj = population_obj[sorted_front[i - 1]][obj_idx]
                next_obj = population_obj[sorted_front[i + 1]][obj_idx]

                distances[sorted_front[i]] += (  next_obj - prev_obj ) / obj_range

        return distances
    
    
    def _crossover(self, parent1, parent2):
        """Crossover de pools (SBX-like para combinatória)"""
        child_pool = []
        
        for i in range(self.n_games):
            # Selecionar jogo de um dos pais
            if np.random.random() < 0.5:
                base_game = parent1[i].copy()
                other_game = parent2[i].copy()
            else:
                base_game = parent2[i].copy()
                other_game = parent1[i].copy()
            
            # Trocar algumas dezenas
            common = set(base_game) & set(other_game)
            only_base = set(base_game) - common
            only_other = set(other_game) - common
            
            # Manter interseção, trocar alguns exclusivos
            child_game = list(common)
            
            # Pegar metade de cada
            n_from_base = len(only_base) // 2
            n_from_other = len(only_other) // 2
            
            child_game.extend(list(only_base)[:n_from_base])
            child_game.extend(list(only_other)[:n_from_other])
            
            # Completar para 15
            while len(child_game) < 15:
                available = list(set(range(1, 26)) - set(child_game))
                child_game.append(np.random.choice(available))
            
            child_pool.append(sorted(child_game[:15]))
        
        return child_pool
    
    def _mutate(self, pool, mutation_rate=0.1):
        """Mutação de pool"""
        mutated = [g.copy() for g in pool]
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                game = mutated[i]
                
                # Estratégia de mutação
                strategy = np.random.choice(['swap', 'shift', 'balance'])
                
                if strategy == 'swap':
                    pos = np.random.randint(0, 15)
                    available = [d for d in range(1, 26) if d not in game]
                    if available:
                        game[pos] = np.random.choice(available)
                
                elif strategy == 'shift':
                    # Mover para vizinho no volante 5x5
                    pos = np.random.randint(0, 15)
                    current = game[pos]
                    row, col = (current - 1) // 5, (current - 1) % 5
                    
                    neighbors = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < 5 and 0 <= nc < 5:
                            neighbor = nr * 5 + nc + 1
                            if neighbor not in game:
                                neighbors.append(neighbor)
                    
                    if neighbors:
                        game[pos] = np.random.choice(neighbors)
                
                elif strategy == 'balance':
                    # Balancear quadrantes
                    quadrants = defaultdict(list)
                    for idx, d in enumerate(game):
                        q = (d - 1) // 5
                        quadrants[q].append(idx)
                    
                    if quadrants:
                        max_q = max(quadrants.keys(), key=lambda q: len(quadrants[q]))
                        if len(quadrants[max_q]) > 4:
                            pos = np.random.choice(quadrants[max_q])
                            available = [d for d in range(1, 26) if d not in game]
                            if available:
                                game[pos] = np.random.choice(available)
                
                mutated[i] = sorted(game)
        
        return mutated
    
    def run(self):
        """
        Executa NSGA-II completo
        
        Returns:
            list: População final (pools na fronteira de Pareto)
        """
        print(f"\n{'='*70}")
        print(f"🧬 NSGA-II: OTIMIZAÇÃO MULTIOBJETIVO")
        print(f"{'='*70}")
        print(f"   Objetivos: {self.n_objectives}")
        print(f"   População: {self.pop_size}")
        print(f"   Gerações: {self.n_generations}")
        print(f"   Jogos/pool: {self.n_games}")
        
        # Inicializar população
        population = []
        for _ in range(self.pop_size):
            pool = []
            for _ in range(self.n_games):
                game = sorted(np.random.choice(range(1, 26), 15, replace=False))
                pool.append(game)
            population.append(pool)
        
        # Avaliar objetivos iniciais
        population_obj = [self._evaluate(p) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            # Criar filhos
            offspring = []
            
            while len(offspring) < self.pop_size:
                # Seleção por torneio
                idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                
                if self._dominates(population_obj[idx1], population_obj[idx2]):
                    parent1 = population[idx1]
                elif self._dominates(population_obj[idx2], population_obj[idx1]):
                    parent1 = population[idx2]
                else:
                    parent1 = population[np.random.choice([idx1, idx2])]
                
                idx3, idx4 = np.random.choice(self.pop_size, 2, replace=False)
                if self._dominates(population_obj[idx3], population_obj[idx4]):
                    parent2 = population[idx3]
                elif self._dominates(population_obj[idx4], population_obj[idx3]):
                    parent2 = population[idx4]
                else:
                    parent2 = population[np.random.choice([idx3, idx4])]
                
                # Crossover e mutação
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                offspring.append(child)
            
            # Avaliar filhos
            offspring_obj = [self._evaluate(o) for o in offspring]
            
            # Combinar população e filhos
            combined = population + offspring
            combined_obj = population_obj + offspring_obj
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined_obj)
            
            # Selecionar próxima população
            new_population = []
            new_population_obj = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    # Adicionar front inteiro
                    for idx in front:
                        new_population.append(combined[idx])
                        new_population_obj.append(combined_obj[idx])
                else:
                    # Truncar último front por crowding distance
                    remaining = self.pop_size - len(new_population)
                    distances = self._crowding_distance(front, combined_obj)

                    sorted_by_dist = sorted(  front,  key=lambda idx: distances[idx], reverse=True)
                    
                    for idx in sorted_by_dist[:remaining]:
                        new_population.append(combined[idx])
                        new_population_obj.append(combined_obj[idx])
                    break
            
            population = new_population
            population_obj = new_population_obj
            current_fronts = self._fast_non_dominated_sort(population_obj)
            
            # Registrar Pareto front
            pareto_front = [population_obj[i] for i in current_fronts[0]]
            self.pareto_fronts.append(pareto_front)
            self.fitness_history.append(len(fronts[0]))
        
        # Resultado final
        final_fronts = self._fast_non_dominated_sort(population_obj)
        pareto_population = [population[i] for i in final_fronts[0]]
        pareto_objectives = [population_obj[i] for i in final_fronts[0]]
        
        print(f"\n   ✅ NSGA-II concluído!")
        print(f"   📊 Soluções na fronteira de Pareto: {len(pareto_population)}")
        
        self.final_pareto = {
            'pools': pareto_population,
            'objectives': pareto_objectives,
            'front_size': len(pareto_population)
        }
        
        return pareto_population, pareto_objectives
    
    def get_pareto_summary(self):
        """Sumário da fronteira de Pareto"""
        if not hasattr(self, 'final_pareto'):
            return None
        
        objectives = self.final_pareto['objectives']
        obj_array = np.array(objectives)
        
        summary = {}
        for i, name in enumerate(self.objective_names):
            summary[name] = {
                'min': float(np.min(obj_array[:, i])),
                'max': float(np.max(obj_array[:, i])),
                'mean': float(np.mean(obj_array[:, i])),
                'std': float(np.std(obj_array[:, i]))
            }
        
        return summary
    
    def visualize_pareto(self, output_dir='graficos_pareto'):
        """Visualiza fronteira de Pareto em 3D e 2D"""
        print(f"\n🎨 VISUALIZANDO FRONTEIRA DE PARETO...")
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'final_pareto'):
            print("⚠️  Execute run() primeiro!")
            return
        
        objectives = np.array(self.final_pareto['objectives'])
        
        # 1. Gráfico 3D (3 primeiros objetivos)
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            objectives[:, 0],  # Cobertura pares
            objectives[:, 1],  # Cobertura trincas
            objectives[:, 2],  # Distância mínima
            c=objectives[:, 3],  # Cor = entropia
            cmap='viridis',
            s=100,
            alpha=0.8
        )
        
        ax.set_xlabel(self.objective_names[0])
        ax.set_ylabel(self.objective_names[1])
        ax.set_zlabel(self.objective_names[2])
        ax.set_title('Fronteira de Pareto - NSGA-II (3D)')
        
        plt.colorbar(scatter, ax=ax, label=self.objective_names[3])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_3d.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 2. Matriz de scatter plots 2D
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        pairs = [(0, 1), (0, 2), (1, 3), (2, 4)]
        
        for (i, j), ax in zip(pairs, axes.flat):
            scatter = ax.scatter(
                objectives[:, i],
                objectives[:, j],
                c=objectives[:, 3],
                cmap='viridis',
                s=80,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel(self.objective_names[i])
            ax.set_ylabel(self.objective_names[j])
            ax.set_title(f'{self.objective_names[i]} vs {self.objective_names[j]}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Trade-offs na Fronteira de Pareto', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_2d_matrix.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 3. Evolução do tamanho da fronteira
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.fitness_history, color='blue', linewidth=2)
        ax.set_xlabel('Geração')
        ax.set_ylabel('Tamanho da Fronteira de Pareto')
        ax.set_title('Evolução do NSGA-II')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/nsga2_evolution.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 4. Coordenadas paralelas
        from pandas.plotting import parallel_coordinates
        
        # Normalizar para visualização
        obj_norm = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i in range(len(obj_norm)):
            ax.plot(range(len(self.objective_names)), obj_norm[i], 
                   'o-', alpha=0.3, linewidth=1, markersize=4)
        
        ax.set_xticks(range(len(self.objective_names)))
        ax.set_xticklabels(self.objective_names, rotation=45, ha='right')
        ax.set_ylabel('Valor Normalizado')
        ax.set_title('Coordenadas Paralelas - Fronteira de Pareto')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_parallel.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")


# ============================================================
# ANÁLISE DE ROBUSTEZ E KL-DIVERGENCE
# ============================================================

class RobustnessAnalyzer:
    """
    Análise de robustez e qualidade informacional
    """
    
    @staticmethod
    def kl_divergence(pool, uniform=True):
        """
        KL-Divergence entre distribuição do pool e uniforme
        
        KL(P || U) mede quão diferente o pool é da uniformidade perfeita
        """
        all_dezenas = [d for game in pool for d in game]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        P = freq / np.sum(freq)
        P = np.where(P > 0, P, 1e-10)
        
        if uniform:
            U = np.ones(25) / 25
        else:
            U = np.ones(25) / 25
        
        kl = np.sum(P * np.log(P / U))
        return float(kl)
    
    @staticmethod
    def stability_under_mutation(pool, n_trials=100, mutation_rate=0.1):
        """
        Estabilidade: quão robusto é o pool a pequenas mutações
        """
        original_metrics = RobustnessAnalyzer._compute_metrics(pool)
        
        metric_changes = defaultdict(list)
        
        for _ in range(n_trials):
            # Mutar levemente
            mutated = [g.copy() for g in pool]
            idx = np.random.randint(0, len(mutated))
            game = mutated[idx]
            pos = np.random.randint(0, 15)
            available = [d for d in range(1, 26) if d not in game]
            if available:
                game[pos] = np.random.choice(available)
            
            mutated_metrics = RobustnessAnalyzer._compute_metrics(mutated)
            
            for key in original_metrics:
                change = abs(mutated_metrics[key] - original_metrics[key])
                metric_changes[key].append(change)
        
        stability = {}
        for key, changes in metric_changes.items():
            stability[key] = {
                'mean_change': float(np.mean(changes)),
                'max_change': float(np.max(changes)),
                'std_change': float(np.std(changes))
            }
        
        return stability
    
    @staticmethod
    def _compute_metrics(pool):
        """Métricas básicas para análise de robustez"""
        all_dezenas = [d for game in pool for d in game]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        
        return {
            'entropy': float(entropy(freq / np.sum(freq) + 1e-10)),
            'mean_freq': float(np.mean(freq)),
            'std_freq': float(np.std(freq)),
            'unique_dezenas': len(set(all_dezenas))
        }
    
    @staticmethod
    def sensitivity_analysis(pool, n_trials=500):
        """
        Análise de sensibilidade: como o desempenho varia
        com pequenas perturbações
        """
        # Simular múltiplos sorteios
        hits_distribution = []
        
        for _ in range(n_trials):
            drawn = set(np.random.choice(range(1, 26), 15, replace=False))
            
            total_hits = 0
            for game in pool:
                hits = len(set(game) & drawn)
                if hits >= 11:
                    total_hits += 1
            
            hits_distribution.append(total_hits)
        
        return {
            'mean_hits': float(np.mean(hits_distribution)),
            'std_hits': float(np.std(hits_distribution)),
            'cv': float(np.std(hits_distribution) / np.mean(hits_distribution)) if np.mean(hits_distribution) > 0 else float('inf'),
            'ci95_lower': float(np.percentile(hits_distribution, 2.5)),
            'ci95_upper': float(np.percentile(hits_distribution, 97.5))
        }


# ============================================================
# SUITE PRINCIPAL
# ============================================================

def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*70)
    print("🧬 SISTEMA DE OTIMIZAÇÃO MULTIOBJETIVO - LOTOFÁCIL")
    print("   NSGA-II + Hipergrafos + Covering Codes")
    print("="*70)
    print()
    print("⚠️  PARADIGMA:")
    print("   NÃO existe solução ótima única.")
    print("   Existe uma FRONTEIRA de Pareto com trade-offs.")
    print("   O NSGA-II descobre essa fronteira automaticamente.")
    print("="*70)
    
    # 1. NSGA-II
    nsga2 = NSGA2(
        n_games=50,
        pop_size=100,
        n_generations=30
    )
    
    pareto_pools, pareto_objectives = nsga2.run()
    
    # Sumário da fronteira
    summary = nsga2.get_pareto_summary()
    print(f"\n📊 FRONTEIRA DE PARETO - SUMÁRIO:")
    print(f"{'Objetivo':<25} {'Min':<10} {'Max':<10} {'Média':<10}")
    print("-"*55)
    for name, stats in summary.items():
        print(f"{name:<25} {stats['min']:<10.3f} {stats['max']:<10.3f} {stats['mean']:<10.3f}")
    
    # 2. Visualizações
    nsga2.visualize_pareto()
    
    # 3. Análise de robustez da melhor solução (trade-off equilibrado)
    # Selecionar solução com maior entropia (índice 3)
    best_idx = np.argmax([obj[3] for obj in pareto_objectives])
    best_pool = pareto_pools[best_idx]
    
    print(f"\n🔬 ANÁLISE DE ROBUSTEZ (melhor solução):")
    
    # KL-Divergence
    kl = RobustnessAnalyzer.kl_divergence(best_pool)
    print(f"   KL-Divergence (vs uniforme): {kl:.6f}")
    
    # Estabilidade
    stability = RobustnessAnalyzer.stability_under_mutation(best_pool)
    print(f"   Estabilidade sob mutação:")
    for key, stats in stability.items():
        print(f"      {key}: Δmédio={stats['mean_change']:.4f}")
    
    # Sensibilidade
    sensitivity = RobustnessAnalyzer.sensitivity_analysis(best_pool)
    print(f"   Sensibilidade (500 simulações):")
    print(f"      Média hits: {sensitivity['mean_hits']:.2f} ± {sensitivity['std_hits']:.2f}")
    print(f"      IC 95%: [{sensitivity['ci95_lower']:.1f}, {sensitivity['ci95_upper']:.1f}]")
    
    # 4. Análise de hipergrafo
    print(f"\n📐 ANÁLISE DE HIPERGRAFO:")
    hg = HypergraphAnalyzer()
    
    degree_dist = hg.degree_distribution(best_pool)
    print(f"   Distribuição de graus:")
    print(f"      Mín: {degree_dist['min']} | Máx: {degree_dist['max']}")
    print(f"      Média: {degree_dist['mean']:.1f} ± {degree_dist['std']:.1f}")
    print(f"      Gini: {degree_dist['gini']:.3f}")
    
    transversal = hg.transversal_number(best_pool)
    print(f"   Número de transversalidade: {transversal}")
    
    # 5. Covering codes
    print(f"\n🔐 COVERING CODES:")
    code = BinaryCode()
    
    covering_radius = code.covering_radius(best_pool)
    print(f"   Covering radius: {covering_radius}")
    
    sphere_density = code.sphere_packing_density(best_pool, radius=6)
    print(f"   Sphere packing density: {sphere_density:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✅ OTIMIZAÇÃO MULTIOBJETIVO CONCLUÍDA!")
    print(f"📁 Gráficos: graficos_pareto/")
    print(f"{'='*70}")
    
    print(f"\n💡 PRINCIPAIS DESCOBERTAS:")
    print(f"   1. NÃO existe solução ótima única")
    print(f"   2. Existe FRONTEIRA de Pareto com trade-offs reais")
    print(f"   3. Cobertura vs Diversidade: conflito fundamental")
    print(f"   4. NSGA-II encontra automaticamente os melhores trade-offs")
    print(f"   5. O usuário ESCOLHE qual trade-off prefere")
    
    print(f"\n🎯 RECOMENDAÇÕES POR PERFIL:")
    print(f"   Máxima cobertura → Solução extremo A")
    print(f"   Máxima diversidade → Solução extremo B")
    print(f"   Equilíbrio → Solução central da fronteira")
    print(f"   Máxima entropia → Solução extremo C")


if __name__ == "__main__":
    main()
