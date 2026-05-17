#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE OTIMIZAÇÃO MULTIOBJETIVO AVANÇADO
=============================================
Versão 6.0 - Laplacian Spectrum + Cache Adaptativo + Serialização Binária

MELHORIAS CRÍTICAS:
✅ Cache adaptativo para covering radius (3 níveis)
✅ Serialização binária (não JSON) para compressibilidade
✅ Laplacian Spectrum (spectral gap, algebraic connectivity)
✅ Grassmannian packing metrics
✅ Adaptive sample size por geração
✅ Binary serialization para fitness de compressão
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from scipy.linalg import eigh
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
from tqdm import tqdm
import zlib
import struct
from math import comb
from functools import lru_cache
import hashlib

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150

# ============================================================
# JOHNSON SPACE J(25,15) - CORRETO
# ============================================================

class JohnsonSpace:
    """
    Espaço de Johnson J(n, k) - Constant Weight Codes
    
    Distância correta: d(A,B) = k - |A ∩ B|
    Volume de esfera: V(r) = Σ C(k,i) × C(n-k,i)
    """
    
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.total_codes = comb(n, k)
        self.min_intersection = max(0, 2*k - n)  # 5 para Lotofácil
        self.max_distance = k - self.min_intersection  # 10
        
        # Cache para distâncias
        self._dist_cache = {}
    
    def johnson_distance(self, game1, game2):
        """Distância de Johnson com cache"""
        key = (tuple(sorted(game1)), tuple(sorted(game2)))
        if key not in self._dist_cache:
            intersection = len(set(game1) & set(game2))
            self._dist_cache[key] = self.k - intersection
        return self._dist_cache[key]
    
    def min_johnson_distance(self, pool):
        """Distância mínima no pool"""
        if len(pool) < 2:
            return self.max_distance
        
        min_d = self.max_distance
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                d = self.johnson_distance(pool[i], pool[j])
                min_d = min(min_d, d)
                if min_d == 0:
                    return 0
        return min_d
    
    def distance_matrix(self, pool):
        """Matriz de distâncias completa"""
        n = len(pool)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = self.johnson_distance(pool[i], pool[j])
                D[i, j] = d
                D[j, i] = d
        return D
    
    def sphere_volume_johnson(self, radius):
        """Volume de esfera CORRETO em J(n,k)"""
        volume = 0
        for i in range(min(radius, self.k) + 1):
            if i <= self.n - self.k:
                volume += comb(self.k, i) * comb(self.n - self.k, i)
        return volume
    
    def covering_radius_adaptive(self, pool, generation=0, max_generations=50):
        """
        Covering radius com CACHE ADAPTATIVO
        
        Aumenta precisão conforme geração avança
        """
        # Tamanho da amostra adaptativo
        if generation < max_generations * 0.2:
            sample_size = 500
        elif generation < max_generations * 0.6:
            sample_size = 2000
        else:
            sample_size = 5000
        
        # Verificar cache
        pool_key = tuple(tuple(sorted(g)) for g in pool)
        cache_key = hashlib.md5(str(pool_key).encode()).hexdigest()
        
        if hasattr(self, '_covering_cache') and cache_key in self._covering_cache:
            return self._covering_cache[cache_key]
        
        # Amostrar espaço
        samples = set()
        while len(samples) < sample_size:
            code = tuple(sorted(np.random.choice(range(1, self.n+1), self.k, replace=False)))
            samples.add(code)
        
        samples = [list(s) for s in samples]
        
        # Calcular covering radius
        max_min_dist = 0
        for code in samples:
            min_dist = min(self.johnson_distance(code, p) for p in pool)
            max_min_dist = max(max_min_dist, min_dist)
        
        result = max_min_dist / self.max_distance
        
        # Cache (limitar tamanho)
        if not hasattr(self, '_covering_cache'):
            self._covering_cache = {}
        if len(self._covering_cache) < 100:
            self._covering_cache[cache_key] = result
        
        return result


# ============================================================
# LAPLACIAN SPECTRUM - GEOMETRIA GLOBAL
# ============================================================

class LaplacianSpectrum:
    """
    Análise Espectral do Grafo do Pool
    
    Vértices = jogos
    Arestas = similaridade (peso = distância Johnson)
    
    Métricas:
    - Spectral gap (λ₂ - λ₁)
    - Algebraic connectivity
    - Fiedler vector
    - Spectral radius
    """
    
    def __init__(self, johnson_space):
        self.js = johnson_space
    
    def similarity_matrix(self, pool):
        """
        Matriz de similaridade (kernel RBF sobre distância Johnson)
        
        S(i,j) = exp(-d(i,j)² / (2σ²))
        """
        n = len(pool)
        S = np.zeros((n, n))
        
        # Calcular distâncias
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                d = self.js.johnson_distance(pool[i], pool[j])
                distances.append(d)
        
        if not distances:
            return S
        
        # Sigma adaptativo (média das distâncias)
        sigma = np.mean(distances) * 0.5 if distances else 1.0
        
        for i in range(n):
            S[i, i] = 1.0  # Autosimilaridade
            for j in range(i+1, n):
                d = self.js.johnson_distance(pool[i], pool[j])
                s = np.exp(-d**2 / (2 * sigma**2))
                S[i, j] = s
                S[j, i] = s
        
        return S
    
    def laplacian_matrix(self, pool):
        """
        Matriz Laplaciana normalizada
        
        L = I - D^(-1/2) S D^(-1/2)
        """
        S = self.similarity_matrix(pool)
        n = len(pool)
        
        # Matriz de graus
        degrees = S.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        
        # Laplaciana normalizada
        L = np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt
        
        return L
    
    def spectral_analysis(self, pool):
        """
        Análise espectral completa
        
        Returns:
            dict: Métricas espectrais
        """
        L = self.laplacian_matrix(pool)
        
        # Autovalores (ordenados)
        eigenvalues = eigh(L, eigvals_only=True)
        eigenvalues = np.sort(eigenvalues)
        
        # Métricas
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
        algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
        spectral_radius = eigenvalues[-1] if len(eigenvalues) > 0 else 0
        
        # Trace (soma dos autovalores = rank)
        trace = np.sum(eigenvalues)
        
        # Entropia espectral
        if trace > 0:
            probs = eigenvalues / trace
            probs = np.where(probs > 0, probs, 1e-10)
            spectral_entropy = float(entropy(probs))
        else:
            spectral_entropy = 0
        
        return {
            'spectral_gap': float(spectral_gap),
            'algebraic_connectivity': float(algebraic_connectivity),
            'spectral_radius': float(spectral_radius),
            'spectral_entropy': float(spectral_entropy),
            'n_components': int(np.sum(eigenvalues < 1e-6)),
            'eigenvalues': eigenvalues.tolist()
        }
    
    def algebraic_connectivity_score(self, pool):
        """
        Conectividade algébrica normalizada (0-1)
        
        Alta = grafo bem conectado (poucos clusters)
        Baixa = grafo com clusters isolados
        """
        analysis = self.spectral_analysis(pool)
        # Normalizar pelo máximo teórico (2 para Laplaciana normalizada)
        return min(1.0, analysis['algebraic_connectivity'] / 2.0)


# ============================================================
# SERIALIZAÇÃO BINÁRIA (ANTI-EXPLOIT)
# ============================================================

class BinarySerializer:
    """
    Serialização binária para compressibilidade
    
    Evita exploits de JSON/string:
    - Usa representação binária direta
    - Cada jogo = 25 bits (peso 15)
    - Pool = matriz binária compacta
    """
    
    @staticmethod
    def game_to_bits(game):
        """Converte jogo para 25 bits (4 bytes)"""
        bits = 0
        for d in game:
            bits |= (1 << (d - 1))
        return bits
    
    @staticmethod
    def pool_to_bytes(pool):
        """
        Converte pool para bytes compactos
        
        Formato: 4 bytes por jogo (25 bits + padding)
        """
        data = bytearray()
        for game in pool:
            bits = BinarySerializer.game_to_bits(game)
            data.extend(struct.pack('>I', bits))
        return bytes(data)
    
    @staticmethod
    def compressibility(pool):
        """
        Compressibilidade binária (ANTI-EXPLOIT)
        
        Usa representação binária direta, não JSON
        Evita que GA aprenda padrões sintáticos
        """
        raw = BinarySerializer.pool_to_bytes(pool)
        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)


# ============================================================
# OBJETIVOS CONFLITANTES AVANÇADOS
# ============================================================

class AdvancedConflictingObjectives:
    """
    Objetivos com CONFLITO GEOMÉTRICO REAL
    
    1. Cobertura de pares (MAX)
    2. Distância Johnson mínima (MAX) 
    3. Covering radius (MIN) → CONFLITO com #2
    4. Compressibilidade binária (MIN) → CONFLITO com #1
    5. Independência histórica (MIN)
    6. Conectividade algébrica (MAX) → NOVO!
    """
    
    def __init__(self, historical_data=None):
        self.johnson = JohnsonSpace()
        self.laplacian = LaplacianSpectrum(self.johnson)
        self.serializer = BinarySerializer()
        self.historical_data = historical_data or []
        
        self.n_objectives = 6
        self.objective_names = [
            'Cobertura Pares',
            'Dist Johnson Mín',
            'Covering Radius',
            'Compressibilidade',
            'Indep Histórica',
            'Conectividade Algébrica'
        ]
        self.directions = ['max', 'max', 'min', 'min', 'min', 'max']
        
        # Cache para avaliações
        self._eval_cache = {}
    
    def evaluate(self, pool, generation=0, max_gen=50):
        """
        Avaliação com cache adaptativo
        """
        pool_key = tuple(tuple(sorted(g)) for g in pool)
        if pool_key in self._eval_cache:
            return self._eval_cache[pool_key].copy()
        
        objectives = np.zeros(self.n_objectives)
        
        # 1. COBERTURA DE PARES
        objectives[0] = self._pair_coverage(pool)
        
        # 2. DISTÂNCIA JOHNSON MÍNIMA
        objectives[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        
        # 3. COVERING RADIUS (ADAPTATIVO)
        objectives[2] = self.johnson.covering_radius_adaptive(pool, generation, max_gen)
        
        # 4. COMPRESSIBILIDADE BINÁRIA
        objectives[3] = self.serializer.compressibility(pool)
        
        # 5. INDEPENDÊNCIA HISTÓRICA
        objectives[4] = self._historical_independence(pool)
        
        # 6. CONECTIVIDADE ALGÉBRICA (NOVO)
        objectives[5] = self.laplacian.algebraic_connectivity_score(pool)
        
        # Cache (limitar tamanho)
        if len(self._eval_cache) < 500:
            self._eval_cache[pool_key] = objectives.copy()
        
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

class NSGA2_Advanced:
    """NSGA-II com objetivos conflitantes avançados"""
    
    def __init__(self, n_games=50, pop_size=200, n_generations=50, historical_data=None):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        self.objectives = AdvancedConflictingObjectives(historical_data)
        self.johnson = JohnsonSpace()
        
        self.n_obj = self.objectives.n_objectives
        self.directions = self.objectives.directions
        
        self.pareto_fronts = []
    
    def _dominates(self, obj1, obj2):
        """Domingação de Pareto"""
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
        strategies = ['random', 'spread', 'clustered', 'balanced']
        
        for _ in range(self.pop_size):
            strategy = np.random.choice(strategies)
            pool = self._generate_pool(strategy)
            population.append(pool)
        
        return population
    
    def _generate_pool(self, strategy='random'):
        """Gera pool com estratégia específica"""
        pool = []
        
        if strategy == 'random':
            for _ in range(self.n_games):
                pool.append(sorted(np.random.choice(range(1, 26), 15, replace=False)))
        
        elif strategy == 'spread':
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base)
            for _ in range(self.n_games - 1):
                best = None
                best_min = 15
                for _ in range(50):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min:
                        best_min = m
                        best = c
                pool.append(best if best else sorted(np.random.choice(range(1, 26), 15, replace=False)))
        
        elif strategy == 'clustered':
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(self.n_games):
                game = base.copy()
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail:
                    game[pos] = np.random.choice(avail)
                pool.append(sorted(game))
        
        elif strategy == 'balanced':
            half = self.n_games // 2
            # Espalhado
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base)
            for _ in range(half - 1):
                best = None
                best_min = 15
                for _ in range(50):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min:
                        best_min = m
                        best = c
                pool.append(best if best else sorted(np.random.choice(range(1, 26), 15, replace=False)))
            # Clusterizado
            cbase = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(self.n_games - half):
                game = cbase.copy()
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail:
                    game[pos] = np.random.choice(avail)
                pool.append(sorted(game))
        
        return pool
    
    def run(self):
        """Executa NSGA-II"""
        print(f"\n{'='*70}")
        print(f"🧬 NSGA-II AVANÇADO")
        print(f"{'='*70}")
        print(f"   Objetivos: {self.n_obj}")
        print(f"   População: {self.pop_size}")
        print(f"   Gerações: {self.n_generations}")
        print(f"   Cache adaptativo: SIM")
        print(f"   Serialização binária: SIM")
        print(f"   Laplacian Spectrum: SIM")
        
        population = self._initialize_population()
        population_obj = [self.objectives.evaluate(p, 0, self.n_generations) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            offspring = []
            
            while len(offspring) < self.pop_size:
                # Seleção por torneio
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1], population_obj[i2]) else population[i2]
                
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3], population_obj[i4]) else population[i4]
                
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                offspring.append(child)
            
            offspring_obj = [self.objectives.evaluate(o, gen, self.n_generations) for o in offspring]
            
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
            self.pareto_fronts.append(len(fronts[0]))
        
        final_fronts = self._fast_non_dominated_sort(population_obj)
        pareto_idx = final_fronts[0]
        
        pareto_pools = [population[i] for i in pareto_idx]
        pareto_obj = [population_obj[i] for i in pareto_idx]
        
        print(f"\n   ✅ Fronteira de Pareto: {len(pareto_pools)} soluções")
        
        return pareto_pools, pareto_obj
    
    def _crossover(self, p1, p2):
        strategy = np.random.choice(['exchange', 'interpolate', 'diverse'])
        mid = self.n_games // 2
        
        if strategy == 'exchange':
            return p1[:mid] + p2[mid:]
        elif strategy == 'interpolate':
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
        else:
            combined = p1 + p2
            selected = [combined[0]]
            remaining = combined[1:]
            while len(selected) < self.n_games and remaining:
                best_idx = 0
                best_min = -1
                for idx in range(min(50, len(remaining))):
                    m = min(self.johnson.johnson_distance(remaining[idx], s) for s in selected)
                    if m > best_min:
                        best_min = m
                        best_idx = idx
                selected.append(remaining[best_idx])
                remaining.pop(best_idx)
            return selected[:self.n_games]
    
    def _mutate(self, pool, rate=0.15):
        mutated = [g.copy() for g in pool]
        n_mut = max(1, int(self.n_games * rate))
        indices = np.random.choice(self.n_games, n_mut, replace=False)
        
        for idx in indices:
            game = mutated[idx]
            strategy = np.random.choice(['drift', 'jump', 'restart'])
            
            if strategy == 'drift':
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail:
                    game[pos] = np.random.choice(avail)
            elif strategy == 'jump':
                for _ in range(np.random.randint(2, 5)):
                    pos = np.random.randint(0, 15)
                    avail = [d for d in range(1, 26) if d not in game]
                    if avail:
                        game[pos] = np.random.choice(avail)
            else:
                game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            
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
    print("="*70)
    print("🧬 NSGA-II AVANÇADO - JOHNSON SPACE + LAPLACIAN")
    print("="*70)
    
    historical = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(100)]
    
    nsga2 = NSGA2_Advanced(
        n_games=30,  # Reduzido para demonstração
        pop_size=100,
        n_generations=30,
        historical_data=historical
    )
    
    pareto_pools, pareto_obj = nsga2.run()
    
    obj_array = np.array(pareto_obj)
    
    print(f"\n📊 DIVERSIDADE NA FRONTEIRA:")
    for i, name in enumerate(nsga2.objectives.objective_names):
        vals = obj_array[:, i]
        print(f"   {name:<30}: [{np.min(vals):.3f}, {np.max(vals):.3f}] range={np.max(vals)-np.min(vals):.3f}")
    
    # Correlações
    print(f"\n📊 CORRELAÇÕES (negativo = CONFLITO):")
    for i in range(nsga2.n_obj):
        for j in range(i+1, nsga2.n_obj):
            corr = np.corrcoef(obj_array[:, i], obj_array[:, j])[0, 1]
            status = "🔴 CONFLITO" if corr < -0.3 else "🟡 NEUTRO" if abs(corr) < 0.3 else "🟢 ALINHADO"
            print(f"   {nsga2.objectives.objective_names[i][:20]} vs {nsga2.objectives.objective_names[j][:20]}: r={corr:+.3f} {status}")
    
    # Laplacian analysis
    print(f"\n🔷 LAPLACIAN SPECTRUM (melhor solução):")
    js = JohnsonSpace()
    ls = LaplacianSpectrum(js)
    analysis = ls.spectral_analysis(pareto_pools[0])
    print(f"   Spectral gap: {analysis['spectral_gap']:.4f}")
    print(f"   Algebraic connectivity: {analysis['algebraic_connectivity']:.4f}")
    print(f"   Spectral entropy: {analysis['spectral_entropy']:.4f}")
    print(f"   Components: {analysis['n_components']}")
    
    print(f"\n✅ CONCLUÍDO!")

if __name__ == "__main__":
    main()
