#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE OTIMIZAÇÃO MULTIOBJETIVO - VERSÃO CORRIGIDA
=======================================================
Versão 7.1 - Bitwise Distance + Pareto Elite + Cache Binário

CORREÇÕES:
✅ min_johnson_distance() RESTAURADO
✅ Distâncias BITWISE (não sets Python)
✅ Elite por dominância Pareto (não soma)
✅ Cache com hash binário (MD5 de bytes)
✅ Pool key otimizado (BinarySerializer)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
from tqdm import tqdm
import zlib
import struct
from math import comb
import hashlib

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 120

# ============================================================
# JOHNSON SPACE J(25,15) - BITWISE OTIMIZADO
# ============================================================

class JohnsonSpace:
    """
    Espaço de Johnson com operações BITWISE
    
    Cada jogo = inteiro 32-bit (25 bits usados)
    Distância = k - popcount(bits1 & bits2)
    """
    
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.total_codes = comb(n, k)
        self.min_intersection = max(0, 2*k - n)
        self.max_distance = k - self.min_intersection
        
        # Cache com hash binário
        self._dist_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache = 15000
    
    def game_to_bits(self, game):
        """Converte jogo para inteiro 32-bit (RÁPIDO)"""
        bits = 0
        for d in game:
            bits |= (1 << (d - 1))
        return bits
    
    def bits_to_game(self, bits):
        """Converte bits para jogo"""
        game = []
        for i in range(self.n):
            if bits & (1 << i):
                game.append(i + 1)
        return sorted(game)
    
    def johnson_distance(self, game1, game2):
        """
        Distância de Johnson BITWISE
        
        d = k - popcount(bits1 & bits2)
        """
        # Converter para bits (cache interno)
        if isinstance(game1, list):
            bits1 = self.game_to_bits(game1)
        else:
            bits1 = game1
        
        if isinstance(game2, list):
            bits2 = self.game_to_bits(game2)
        else:
            bits2 = game2
        
        # Cache key (usar bits como chave)
        cache_key = (bits1, bits2) if bits1 < bits2 else (bits2, bits1)
        
        if cache_key in self._dist_cache:
            self._cache_hits += 1
            return self._dist_cache[cache_key]
        
        self._cache_misses += 1
        
        # Operação BITWISE (muito mais rápida que sets)
        intersection = (bits1 & bits2).bit_count()
        d = self.k - intersection
        
        # Cache com limite
        if len(self._dist_cache) < self._max_cache:
            self._dist_cache[cache_key] = d
        
        return d
    
    def min_johnson_distance(self, pool):
        """
        Distância mínima de Johnson no pool (RESTAURADO)
        """
        if len(pool) < 2:
            return self.max_distance
        
        # Converter todos para bits uma vez
        bits_list = [self.game_to_bits(g) for g in pool]
        
        min_d = self.max_distance
        
        for i in range(len(bits_list)):
            for j in range(i + 1, len(bits_list)):
                d = self.johnson_distance(bits_list[i], bits_list[j])
                
                if d < min_d:
                    min_d = d
                
                # Early stop
                if min_d == 0:
                    return 0
        
        return min_d
    
    def avg_johnson_distance(self, pool):
        """Distância média de Johnson"""
        if len(pool) < 2:
            return 0
        
        bits_list = [self.game_to_bits(g) for g in pool]
        distances = []
        
        for i in range(len(bits_list)):
            for j in range(i + 1, len(bits_list)):
                distances.append(self.johnson_distance(bits_list[i], bits_list[j]))
        
        return np.mean(distances)
    
    def distance_matrix_sparse(self, pool, k_neighbors=5):
        """
        Matriz de distâncias ESPARSA (k-NN)
        Usa bits para velocidade
        """
        n = len(pool)
        bits_list = [self.game_to_bits(g) for g in pool]
        
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    d = self.johnson_distance(bits_list[i], bits_list[j])
                    distances.append((d, j))
            
            distances.sort(key=lambda x: x[0])
            
            for d, j in distances[:k_neighbors]:
                row_indices.append(i)
                col_indices.append(j)
                data.append(d)
        
        return csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    def covering_radius_fast(self, pool, sample_size=500):
        """Covering radius com early stopping"""
        bits_list = [self.game_to_bits(g) for g in pool]
        
        # Gerar amostras como bits
        samples = set()
        while len(samples) < sample_size:
            game = sorted(np.random.choice(range(1, self.n+1), self.k, replace=False))
            samples.add(self.game_to_bits(game))
        
        samples = list(samples)
        
        max_min_dist = 0
        early_stop_threshold = self.max_distance * 0.8
        
        for sample_bits in samples:
            min_dist = self.max_distance
            
            for pool_bits in bits_list:
                d = self.johnson_distance(sample_bits, pool_bits)
                min_dist = min(min_dist, d)
                
                if min_dist == 0:
                    break
            
            max_min_dist = max(max_min_dist, min_dist)
            
            if max_min_dist >= early_stop_threshold:
                break
        
        return max_min_dist / self.max_distance


# ============================================================
# SERIALIZAÇÃO BINÁRIA (para cache hash)
# ============================================================

class BinarySerializer:
    """Serialização binária para hash rápido"""
    
    @staticmethod
    def pool_to_bytes(pool):
        """Converte pool para bytes (4 bytes por jogo)"""
        data = bytearray()
        for game in pool:
            bits = 0
            for d in game:
                bits |= (1 << (d - 1))
            data.extend(struct.pack('>I', bits))
        return bytes(data)
    
    @staticmethod
    def pool_hash(pool):
        """Hash MD5 do pool (para cache key)"""
        return hashlib.md5(BinarySerializer.pool_to_bytes(pool)).hexdigest()
    
    @staticmethod
    def compressibility(pool):
        """Compressibilidade binária"""
        raw = BinarySerializer.pool_to_bytes(pool)
        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)


# ============================================================
# LAPLACIAN ESPECTRAL ESPARSO
# ============================================================

class LaplacianSpectrumSparse:
    """Análise espectral com matriz esparsa"""
    
    def __init__(self, johnson_space):
        self.js = johnson_space
        self._spectral_cache = {}
        self._max_cache = 150
    
    def similarity_sparse(self, pool, k_neighbors=5):
        """Matriz de similaridade esparsa (k-NN)"""
        n = len(pool)
        D_sparse = self.js.distance_matrix_sparse(pool, k_neighbors)
        
        sigma = 3.0
        S = D_sparse.copy()
        S.data = np.exp(-S.data**2 / (2 * sigma**2))
        S = S + eye(n, format='csr')
        
        return S
    
    def laplacian_sparse(self, pool, k_neighbors=5):
        """Laplaciana normalizada esparsa"""
        S = self.similarity_sparse(pool, k_neighbors)
        n = len(pool)
        
        degrees = np.array(S.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-10)
        D_inv_sqrt = diags(d_inv_sqrt, format='csr')
        
        I = eye(n, format='csr')
        L = I - D_inv_sqrt @ S @ D_inv_sqrt
        
        return L
    
    def spectral_analysis_fast(self, pool, k_eigenvalues=3):
        """Análise espectral rápida (eigsh)"""
        pool_hash = BinarySerializer.pool_hash(pool)
        
        if pool_hash in self._spectral_cache:
            return self._spectral_cache[pool_hash]
        
        n = len(pool)
        
        try:
            L = self.laplacian_sparse(pool, k_neighbors=5)
            eigenvalues, _ = eigsh(L, k=min(k_eigenvalues, n-1), which='SM')
            eigenvalues = np.sort(eigenvalues)
            
            result = {
                'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0,
                'algebraic_connectivity': float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
                'n_components': int(np.sum(eigenvalues < 1e-6)),
                'eigenvalues': eigenvalues.tolist()
            }
        except:
            result = {
                'spectral_gap': 0.0,
                'algebraic_connectivity': 0.0,
                'n_components': 1,
                'eigenvalues': [0.0]
            }
        
        if len(self._spectral_cache) < self._max_cache:
            self._spectral_cache[pool_hash] = result
        
        return result
    
    def connectivity_score(self, pool):
        """Conectividade algébrica (0-1)"""
        analysis = self.spectral_analysis_fast(pool)
        return min(1.0, analysis['algebraic_connectivity'] / 2.0)


# ============================================================
# OBJETIVOS COM AVALIAÇÃO SELETIVA (CORRIGIDA)
# ============================================================

class SelectiveObjectives:
    """Objetivos com elite por dominância Pareto"""
    
    def __init__(self, historical_data=None):
        self.johnson = JohnsonSpace()
        self.laplacian = LaplacianSpectrumSparse(self.johnson)
        self.serializer = BinarySerializer()
        self.historical_data = historical_data or []
        
        self.n_objectives = 6
        self.objective_names = [
            'Cobertura Pares',
            'Dist Johnson Mín',
            'Covering Radius',
            'Compressibilidade',
            'Indep Histórica',
            'Conectividade Alg'
        ]
        self.directions = ['max', 'max', 'min', 'min', 'min', 'max']
        
        self._spectral_eval_freq = 3
        self._elite_ratio = 0.3
        
        # Cache com hash binário
        self._eval_cache = {}
        self._max_eval_cache = 300
    
    def evaluate_all(self, pool, is_elite=False, force_spectral=False):
        """Avaliação completa"""
        pool_hash = BinarySerializer.pool_hash(pool)
        
        if pool_hash in self._eval_cache:
            return self._eval_cache[pool_hash].copy()
        
        objectives = np.zeros(self.n_objectives)
        
        # Baratos (sempre)
        objectives[0] = self._pair_coverage(pool)
        objectives[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        objectives[2] = self.johnson.covering_radius_fast(pool, sample_size=500)
        objectives[3] = self.serializer.compressibility(pool)
        objectives[4] = self._historical_independence(pool)
        
        # Caro (apenas elite)
        if is_elite or force_spectral:
            objectives[5] = self.laplacian.connectivity_score(pool)
        else:
            objectives[5] = 0.5
        
        if len(self._eval_cache) < self._max_eval_cache:
            self._eval_cache[pool_hash] = objectives.copy()
        
        return objectives
    
    def evaluate_light(self, pool):
        """Avaliação leve (sem espectro)"""
        objectives = np.zeros(self.n_objectives)
        objectives[0] = self._pair_coverage(pool)
        objectives[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        objectives[2] = self.johnson.covering_radius_fast(pool, sample_size=300)
        objectives[3] = self.serializer.compressibility(pool)
        objectives[4] = self._historical_independence(pool)
        objectives[5] = 0.5
        return objectives
    
    def _pair_coverage(self, pool):
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        return len(covered) / comb(25, 2)
    
    def _historical_independence(self, pool):
        if not self.historical_data:
            return 0.5
        recent = self.historical_data[-50:] if len(self.historical_data) > 50 else self.historical_data
        total_sim = 0
        for game in pool:
            max_sim = max(len(set(game) & set(h)) / 15 for h in recent)
            total_sim += max_sim
        return total_sim / len(pool)


# ============================================================
# NSGA-II COM ELITE PARETO (CORRIGIDO)
# ============================================================

class FastNSGA2:
    """NSGA-II com seleção de elite por dominância Pareto"""
    
    def __init__(self, n_games=30, pop_size=80, n_generations=30, historical_data=None):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        self.objectives = SelectiveObjectives(historical_data)
        self.johnson = JohnsonSpace()
        
        self.n_obj = self.objectives.n_objectives
        self.directions = self.objectives.directions
    
    def _dominates(self, obj1, obj2):
        better_in_any = False
        for i, direction in enumerate(self.directions):
            if direction == 'max':
                if obj1[i] < obj2[i]: return False
                if obj1[i] > obj2[i]: better_in_any = True
            else:
                if obj1[i] > obj2[i]: return False
                if obj1[i] < obj2[i]: better_in_any = True
        return better_in_any
    
    def _get_pareto_elite(self, population, population_obj, ratio=0.3):
        """
        Seleciona elite por DOMINÂNCIA PARETO (não soma!)
        """
        n_elite = int(self.pop_size * ratio)
        
        # Ordenar por fronts de Pareto
        fronts = self._fast_non_dominated_sort(population_obj)
        
        elite_indices = []
        for front in fronts:
            elite_indices.extend(front)
            if len(elite_indices) >= n_elite:
                break
        
        return elite_indices[:n_elite]
    
    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            pool = []
            for _ in range(self.n_games):
                pool.append(sorted(np.random.choice(range(1, 26), 15, replace=False)))
            population.append(pool)
        return population
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II BITWISE OTIMIZADO")
        print(f"{'='*60}")
        print(f"   Pop: {self.pop_size} | Gen: {self.n_generations} | Jogos: {self.n_games}")
        print(f"   Distância: BITWISE | Cache: Binário MD5")
        print(f"   Elite: Pareto (não soma) | Espectro: cada {self.objectives._spectral_eval_freq} gens")
        
        population = self._initialize_population()
        population_obj = [self.objectives.evaluate_light(p) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            offspring = []
            while len(offspring) < self.pop_size:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1], population_obj[i2]) else population[i2]
                
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3], population_obj[i4]) else population[i4]
                
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                offspring.append(child)
            
            offspring_obj = [self.objectives.evaluate_light(o) for o in offspring]
            
            combined = population + offspring
            combined_obj = population_obj + offspring_obj
            
            fronts = self._fast_non_dominated_sort(combined_obj)
            
            new_pop = []
            new_obj = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined[idx])
                        new_obj.append(combined_obj[idx])
                else:
                    remaining = self.pop_size - len(new_pop)
                    distances = self._crowding_distance(front, combined_obj)
                    sorted_f = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                    for idx, _ in sorted_f[:remaining]:
                        new_pop.append(combined[idx])
                        new_obj.append(combined_obj[idx])
                    break
            
            population = new_pop
            population_obj = new_obj
            
            # Avaliação espectral seletiva na ELITE PARETO
            force_spectral = (gen % self.objectives._spectral_eval_freq == 0)
            
            if force_spectral:
                elite_indices = self._get_pareto_elite(population, population_obj, self.objectives._elite_ratio)
                
                for idx in elite_indices:
                    population_obj[idx] = self.objectives.evaluate_all(
                        population[idx], is_elite=True, force_spectral=True
                    )
        
        # Final: reavaliar Pareto front completo
        final_fronts = self._fast_non_dominated_sort(population_obj)
        pareto_idx = final_fronts[0]
        
        for idx in pareto_idx:
            population_obj[idx] = self.objectives.evaluate_all(
                population[idx], is_elite=True, force_spectral=True
            )
        
        pareto_pools = [population[i] for i in pareto_idx]
        pareto_obj = [population_obj[i] for i in pareto_idx]
        
        print(f"\n   ✅ Fronteira de Pareto: {len(pareto_pools)} soluções")
        print(f"   📊 Cache distâncias: {self.johnson._cache_hits} hits / {self.johnson._cache_misses} misses")
        
        return pareto_pools, pareto_obj
    
    def _crossover(self, p1, p2):
        mid = self.n_games // 2
        if np.random.random() < 0.5:
            return p1[:mid] + p2[mid:]
        else:
            child = []
            for i in range(self.n_games):
                g1, g2 = p1[i], p2[i]
                common = list(set(g1) & set(g2))
                only1 = list(set(g1) - set(g2))
                only2 = list(set(g2) - set(g1))
                cg = common + only1[:len(only1)//2] + only2[:len(only2)//2]
                while len(cg) < 15:
                    avail = [d for d in range(1, 26) if d not in cg]
                    cg.append(np.random.choice(avail))
                child.append(sorted(cg[:15]))
            return child
    
    def _mutate(self, pool, rate=0.1):
        mutated = [g.copy() for g in pool]
        n_mut = max(1, int(self.n_games * rate))
        indices = np.random.choice(self.n_games, n_mut, replace=False)
        
        for idx in indices:
            game = mutated[idx]
            pos = np.random.randint(0, 15)
            avail = [d for d in range(1, 26) if d not in game]
            if avail:
                game[pos] = np.random.choice(avail)
            mutated[idx] = sorted(game)
        
        return mutated
    
    def _fast_non_dominated_sort(self, pop_obj):
        n = len(pop_obj)
        dom_count = np.zeros(n, dtype=int)
        dom_sol = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(pop_obj[i], pop_obj[j]):
                    dom_sol[i].append(j)
                elif self._dominates(pop_obj[j], pop_obj[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_f = []
            for idx in fronts[i]:
                for didx in dom_sol[idx]:
                    dom_count[didx] -= 1
                    if dom_count[didx] == 0:
                        next_f.append(didx)
            i += 1
            fronts.append(next_f)
        
        return fronts[:-1]
    
    def _crowding_distance(self, front_idx, pop_obj):
        n = len(front_idx)
        if n <= 2:
            return [float('inf')] * n
        
        distances = np.zeros(n)
        for obj_i in range(self.n_obj):
            sorted_i = sorted(front_idx, key=lambda i: pop_obj[i][obj_i])
            obj_range = pop_obj[sorted_i[-1]][obj_i] - pop_obj[sorted_i[0]][obj_i]
            if obj_range > 0:
                distances[0] = float('inf')
                distances[-1] = float('inf')
                for i in range(1, n-1):
                    distances[i] += (pop_obj[sorted_i[i+1]][obj_i] - pop_obj[sorted_i[i-1]][obj_i]) / obj_range
        
        return distances.tolist()


def main():
    print("="*60)
    print("🧬 NSGA-II BITWISE + PARETO ELITE")
    print("="*60)
    
    historical = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(100)]
    
    nsga2 = FastNSGA2(
        n_games=30,
        pop_size=80,
        n_generations=30,
        historical_data=historical
    )
    
    pareto_pools, pareto_obj = nsga2.run()
    
    obj_array = np.array(pareto_obj)
    
    print(f"\n📊 DIVERSIDADE NA FRONTEIRA:")
    for i, name in enumerate(nsga2.objectives.objective_names):
        vals = obj_array[:, i]
        print(f"   {name:<25}: [{np.min(vals):.3f}, {np.max(vals):.3f}] Δ={np.max(vals)-np.min(vals):.3f}")
    
    print(f"\n📊 CORRELAÇÕES:")
    for i in range(nsga2.n_obj):
        for j in range(i+1, nsga2.n_obj):
            corr = np.corrcoef(obj_array[:, i], obj_array[:, j])[0, 1]
            s = "🔴 CONFLITO" if corr < -0.3 else "🟡 NEUTRO" if abs(corr) < 0.3 else "🟢 ALINHADO"
            print(f"   {nsga2.objectives.objective_names[i][:18]} vs {nsga2.objectives.objective_names[j][:18]}: r={corr:+.3f} {s}")
    
    # Métricas Johnson
    js = JohnsonSpace()
    best = pareto_pools[0]
    print(f"\n🔐 JOHNSON SPACE (melhor solução):")
    print(f"   Dist mín: {js.min_johnson_distance(best)}/10")
    print(f"   Dist méd: {js.avg_johnson_distance(best):.1f}/10")
    
    print(f"\n✅ CONCLUÍDO!")

if __name__ == "__main__":
    main()
