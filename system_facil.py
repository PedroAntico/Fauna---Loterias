#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE OTIMIZAÇÃO MULTIOBJETIVO - VERSÃO OTIMIZADA
=======================================================
Versão 7.0 - Espectro Esparso + Cache + Avaliação Seletiva

OTIMIZAÇÕES CRÍTICAS:
✅ scipy.sparse.linalg.eigsh (não eigh denso)
✅ k-NN graph (esparso, não totalmente conectado)
✅ Cache espectral agressivo
✅ Avaliação espectral apenas na elite (top 30%)
✅ Cálculo espectral a cada 3 gerações
✅ Early stopping para covering radius
✅ Matriz Laplaciana esparsa (não densa)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
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
from functools import lru_cache

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 120

# ============================================================
# JOHNSON SPACE J(25,15)
# ============================================================

class JohnsonSpace:
    """Espaço de Johnson com cache eficiente"""
    
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.total_codes = comb(n, k)
        self.min_intersection = max(0, 2*k - n)
        self.max_distance = k - self.min_intersection
        
        # Cache LRU para distâncias
        self._dist_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def johnson_distance(self, game1, game2):
        """Distância de Johnson com cache"""
        key = (tuple(sorted(game1)), tuple(sorted(game2)))
        if key in self._dist_cache:
            self._cache_hits += 1
            return self._dist_cache[key]
        
        self._cache_misses += 1
        intersection = len(set(game1) & set(game2))
        d = self.k - intersection
        self._dist_cache[key] = d
        
        # Limitar cache
        if len(self._dist_cache) > 10000:
            # Remover 20% mais antigos
            keys = list(self._dist_cache.keys())[:2000]
            for k in keys:
                del self._dist_cache[k]
        
        return d
    
    def distance_matrix_sparse(self, pool, k_neighbors=5):
        """
        Matriz de distâncias ESPARSA (k-NN graph)
        
        Apenas armazena os k vizinhos mais próximos
        Reduz O(n²) para O(nk)
        """
        n = len(pool)
        
        # Listas para matriz esparsa CSR
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(n):
            # Calcular distâncias para todos os outros
            distances = []
            for j in range(n):
                if i != j:
                    d = self.johnson_distance(pool[i], pool[j])
                    distances.append((d, j))
            
            # Ordenar por distância e pegar k mais próximos
            distances.sort(key=lambda x: x[0])
            
            for d, j in distances[:k_neighbors]:
                row_indices.append(i)
                col_indices.append(j)
                data.append(d)
        
        D_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        return D_sparse
    
    def covering_radius_fast(self, pool, sample_size=1000):
        """
        Covering radius com EARLY STOPPING
        
        Para se já encontrou raio grande
        """
        # Gerar amostras
        samples = set()
        while len(samples) < sample_size:
            code = tuple(sorted(np.random.choice(range(1, self.n+1), self.k, replace=False)))
            samples.add(code)
        
        samples = [list(s) for s in samples]
        
        max_min_dist = 0
        early_stop_threshold = self.max_distance * 0.8  # 80% do máximo
        
        for code in samples:
            min_dist = self.max_distance
            
            for p in pool:
                d = self.johnson_distance(code, p)
                min_dist = min(min_dist, d)
                
                # Early stop: se já achou distância 0
                if min_dist == 0:
                    break
            
            max_min_dist = max(max_min_dist, min_dist)
            
            # Early stop: se já atingiu threshold alto
            if max_min_dist >= early_stop_threshold:
                break
        
        return max_min_dist / self.max_distance


# ============================================================
# LAPLACIAN SPECTRUM - OTIMIZADO (ESPARRSO)
# ============================================================

class LaplacianSpectrumSparse:
    """
    Análise Espectral com matriz ESPARSA
    
    Otimizações:
    - k-NN graph (não denso)
    - eigsh (não eigh)
    - Cache agressivo
    - Apenas 3 autovalores (k=3)
    """
    
    def __init__(self, johnson_space):
        self.js = johnson_space
        self._spectral_cache = {}
        self._cache_max_size = 200
    
    def similarity_sparse(self, pool, k_neighbors=5):
        """
        Matriz de similaridade ESPARSA (k-NN)
        
        S(i,j) = exp(-d(i,j)² / (2σ²)) apenas para vizinhos
        """
        n = len(pool)
        
        # Calcular distâncias apenas para k vizinhos
        D_sparse = self.js.distance_matrix_sparse(pool, k_neighbors)
        
        # Converter para similaridade
        sigma = 3.0  # Parâmetro fixo para estabilidade
        
        # Construir matriz de similaridade esparsa
        S = D_sparse.copy()
        S.data = np.exp(-S.data**2 / (2 * sigma**2))
        
        # Adicionar diagonal
        S = S + eye(n, format='csr')
        
        return S
    
    def laplacian_sparse(self, pool, k_neighbors=5):
        """
        Laplaciana normalizada ESPARSA
        
        L = I - D^(-1/2) S D^(-1/2)
        """
        S = self.similarity_sparse(pool, k_neighbors)
        n = len(pool)
        
        # Graus (vetor)
        degrees = np.array(S.sum(axis=1)).flatten()
        
        # D^(-1/2) como diagonal esparsa
        d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-10)
        D_inv_sqrt = diags(d_inv_sqrt, format='csr')
        
        # Laplaciana normalizada esparsa
        I = eye(n, format='csr')
        L = I - D_inv_sqrt @ S @ D_inv_sqrt
        
        return L
    
    def spectral_analysis_fast(self, pool, k_eigenvalues=3):
        """
        Análise espectral RÁPIDA
        
        Usa eigsh para obter apenas k menores autovalores
        """
        # Verificar cache
        pool_key = hashlib.md5(
            str([tuple(sorted(g)) for g in pool]).encode()
        ).hexdigest()
        
        if pool_key in self._spectral_cache:
            return self._spectral_cache[pool_key]
        
        n = len(pool)
        
        try:
            # Construir Laplaciana esparsa
            L = self.laplacian_sparse(pool, k_neighbors=5)
            
            # Obter apenas k menores autovalores (eigsh)
            eigenvalues, _ = eigsh(L, k=min(k_eigenvalues, n-1), which='SM')
            eigenvalues = np.sort(eigenvalues)
            
            # Métricas
            spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
            algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
            
            # Cache
            result = {
                'spectral_gap': float(spectral_gap),
                'algebraic_connectivity': float(algebraic_connectivity),
                'n_components': int(np.sum(eigenvalues < 1e-6)),
                'eigenvalues': eigenvalues.tolist()
            }
            
        except Exception as e:
            # Fallback: valores padrão
            result = {
                'spectral_gap': 0.0,
                'algebraic_connectivity': 0.0,
                'n_components': 1,
                'eigenvalues': [0.0]
            }
        
        # Armazenar no cache (limitar tamanho)
        if len(self._spectral_cache) < self._cache_max_size:
            self._spectral_cache[pool_key] = result
        else:
            # Limpar metade do cache
            keys = list(self._spectral_cache.keys())[:self._cache_max_size//2]
            for k in keys:
                del self._spectral_cache[k]
            self._spectral_cache[pool_key] = result
        
        return result
    
    def connectivity_score(self, pool):
        """Conectividade algébrica (0-1)"""
        analysis = self.spectral_analysis_fast(pool)
        return min(1.0, analysis['algebraic_connectivity'] / 2.0)


# ============================================================
# SERIALIZAÇÃO BINÁRIA (mantida)
# ============================================================

class BinarySerializer:
    """Serialização binária eficiente"""
    
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
    def compressibility(pool):
        """Compressibilidade binária"""
        raw = BinarySerializer.pool_to_bytes(pool)
        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)


# ============================================================
# OBJETIVOS COM AVALIAÇÃO SELETIVA
# ============================================================

class SelectiveObjectives:
    """
    Objetivos com AVALIAÇÃO SELETIVA
    
    - Espectro: apenas elite (top 30%)
    - Espectro: apenas a cada 3 gerações
    - Covering radius: cache agressivo
    """
    
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
        
        # Controle de avaliação
        self._eval_counter = 0
        self._spectral_eval_freq = 3  # A cada 3 gerações
        self._elite_ratio = 0.3  # Top 30%
        
        # Caches
        self._eval_cache = {}
        self._spectral_cache = {}
    
    def evaluate_all(self, pool, is_elite=False, force_spectral=False):
        """
        Avaliação completa com seleção condicional
        """
        pool_key = tuple(tuple(sorted(g)) for g in pool)
        
        # Verificar cache completo
        if pool_key in self._eval_cache:
            return self._eval_cache[pool_key].copy()
        
        objectives = np.zeros(self.n_objectives)
        
        # Objetivos BARATOS (sempre calculados)
        objectives[0] = self._pair_coverage(pool)
        objectives[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        objectives[2] = self.johnson.covering_radius_fast(pool, sample_size=500)
        objectives[3] = self.serializer.compressibility(pool)
        objectives[4] = self._historical_independence(pool)
        
        # Objetivo CARO (apenas elite ou forçado)
        if is_elite or force_spectral:
            objectives[5] = self.laplacian.connectivity_score(pool)
        else:
            # Usar valor padrão ou cache
            objectives[5] = self._spectral_cache.get(pool_key, 0.5)
        
        # Cache (limitar)
        if len(self._eval_cache) < 300:
            self._eval_cache[pool_key] = objectives.copy()
        
        return objectives
    
    def evaluate_light(self, pool):
        """
        Avaliação LEVE (sem espectro)
        """
        objectives = np.zeros(self.n_objectives)
        objectives[0] = self._pair_coverage(pool)
        objectives[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        objectives[2] = self.johnson.covering_radius_fast(pool, sample_size=300)
        objectives[3] = self.serializer.compressibility(pool)
        objectives[4] = self._historical_independence(pool)
        objectives[5] = 0.5  # Default
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
# NSGA-II OTIMIZADO
# ============================================================

class FastNSGA2:
    """NSGA-II com avaliação seletiva"""
    
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
    
    def _initialize_population(self):
        """População inicial diversificada"""
        population = []
        for _ in range(self.pop_size):
            pool = []
            for _ in range(self.n_games):
                pool.append(sorted(np.random.choice(range(1, 26), 15, replace=False)))
            population.append(pool)
        return population
    
    def run(self):
        """Executa NSGA-II otimizado"""
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II OTIMIZADO")
        print(f"{'='*60}")
        print(f"   Pop: {self.pop_size} | Gen: {self.n_generations} | Jogos: {self.n_games}")
        print(f"   Espectro: cada {self.objectives._spectral_eval_freq} gens | Elite: {self.objectives._elite_ratio*100:.0f}%")
        print(f"   Matriz: ESPARSA (k-NN) | Autovalores: eigsh (k=3)")
        
        population = self._initialize_population()
        
        # Avaliação inicial (leve para todos)
        population_obj = [self.objectives.evaluate_light(p) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            # Criar offspring
            offspring = []
            while len(offspring) < self.pop_size:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1], population_obj[i2]) else population[i2]
                
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3], population_obj[i4]) else population[i4]
                
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                offspring.append(child)
            
            # Avaliação dos filhos (leve)
            offspring_obj = [self.objectives.evaluate_light(o) for o in offspring]
            
            # Combinar
            combined = population + offspring
            combined_obj = population_obj + offspring_obj
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined_obj)
            
            # Selecionar próxima população
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
            
            # AVALIAÇÃO ESPECTRAL SELETIVA
            force_spectral = (gen % self.objectives._spectral_eval_freq == 0)
            
            if force_spectral:
                # Ordenar por fitness (soma dos objetivos)
                fitness_scores = [np.sum(obj) for obj in population_obj]
                elite_indices = np.argsort(fitness_scores)[-int(self.pop_size * self.objectives._elite_ratio):]
                
                # Reavaliar elite com espectro
                for idx in elite_indices:
                    population_obj[idx] = self.objectives.evaluate_all(
                        population[idx], is_elite=True, force_spectral=True
                    )
        
        # Avaliação final completa da elite
        final_fronts = self._fast_non_dominated_sort(population_obj)
        pareto_idx = final_fronts[0]
        
        # Reavaliar Pareto com espectro
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
        """Crossover rápido"""
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
        """Mutação leve"""
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
        """Non-dominated sorting"""
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
        """Crowding distance"""
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
    print("🧬 NSGA-II OTIMIZADO - ESPARRSO + CACHE")
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
    
    print(f"\n✅ CONCLUÍDO!")
    print(f"   ⚡ O sistema agora é VIÁVEL computacionalmente!")

if __name__ == "__main__":
    main()
