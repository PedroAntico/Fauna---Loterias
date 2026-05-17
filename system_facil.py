#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOTOR DE OTIMIZAÇÃO - SYSTEM_FACIL.PY
======================================
Versão 1.0 - Gera e Exporta Fronteira de Pareto

FUNCIONALIDADES:
✅ NSGA-II completo com Johnson Space
✅ Distâncias bitwise otimizadas
✅ Entropia posicional + Mutual Information
✅ Exporta pareto_frontier.json
✅ Salva métricas e embeddings
✅ Prepara dados para o frontend
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
import struct
import zlib
import heapq
from math import comb
import hashlib
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# JOHNSON SPACE J(25,15)
# ============================================================

class JohnsonSpace:
    """Espaço de Johnson com operações bitwise"""
    
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.total_codes = comb(n, k)
        self.min_intersection = max(0, 2*k - n)
        self.max_distance = k - self.min_intersection
        self._dist_cache = {}
        self._cover_samples = None
        self._cache_hits = 0
        self._cache_misses = 0
    
    def game_to_bits(self, game):
        bits = 0
        for d in game:
            bits |= (1 << (d - 1))
        return bits
    
    def bits_to_game(self, bits):
        game = []
        for i in range(self.n):
            if bits & (1 << i):
                game.append(i + 1)
        return sorted(game)
    
    def johnson_distance(self, game1, game2):
        if isinstance(game1, list): bits1 = self.game_to_bits(game1)
        else: bits1 = game1
        if isinstance(game2, list): bits2 = self.game_to_bits(game2)
        else: bits2 = game2
        
        key = (bits1, bits2) if bits1 < bits2 else (bits2, bits1)
        if key in self._dist_cache:
            self._cache_hits += 1
            return self._dist_cache[key]
        
        self._cache_misses += 1
        d = self.k - (bits1 & bits2).bit_count()
        
        if len(self._dist_cache) < 15000:
            self._dist_cache[key] = d
        
        return d
    
    def min_johnson_distance(self, pool):
        if len(pool) < 2: return self.max_distance
        bits_list = [self.game_to_bits(g) for g in pool]
        min_d = self.max_distance
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                d = self.johnson_distance(bits_list[i], bits_list[j])
                min_d = min(min_d, d)
                if min_d == 0: return 0
        return min_d
    
    def avg_johnson_distance(self, pool):
        if len(pool) < 2: return 0
        bits_list = [self.game_to_bits(g) for g in pool]
        distances = []
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                distances.append(self.johnson_distance(bits_list[i], bits_list[j]))
        return np.mean(distances)
    
    def pre_generate_cover_samples(self, n_samples=2000):
        if self._cover_samples is not None:
            return self._cover_samples
        samples = set()
        while len(samples) < n_samples:
            game = tuple(sorted(np.random.choice(range(1, self.n+1), self.k, replace=False)))
            samples.add(game)
        self._cover_samples = [list(s) for s in samples]
        return self._cover_samples
    
    def covering_radius_fast(self, pool):
        if self._cover_samples is None:
            self.pre_generate_cover_samples()
        bits_list = [self.game_to_bits(g) for g in pool]
        max_min_dist = 0
        for sample in self._cover_samples[:500]:
            sample_bits = self.game_to_bits(sample)
            min_dist = self.max_distance
            for pool_bits in bits_list:
                d = self.johnson_distance(sample_bits, pool_bits)
                min_dist = min(min_dist, d)
                if min_dist == 0: break
            max_min_dist = max(max_min_dist, min_dist)
        return max_min_dist / self.max_distance
    
    def sphere_packing_bound(self, pool):
        d = self.min_johnson_distance(pool)
        if d == 0: return 0
        radius = (d - 1) // 2
        if radius < 0: radius = 0
        sphere_vol = 0
        for i in range(min(radius, self.k) + 1):
            if i <= self.n - self.k:
                sphere_vol += comb(self.k, i) * comb(self.n - self.k, i)
        if sphere_vol == 0: return 1.0
        max_codes = self.total_codes / sphere_vol
        return min(1.0, len(pool) / max_codes)
    
    def distance_matrix_sparse(self, pool, k_neighbors=5):
        n = len(pool)
        bits_list = [self.game_to_bits(g) for g in pool]
        row_indices, col_indices, data = [], [], []
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


# ============================================================
# GARANTIA DE UNICIDADE
# ============================================================

def ensure_unique_pool(pool, n_games, johnson=None):
    if johnson is None:
        johnson = JohnsonSpace()
    unique = []
    seen = set()
    for game in pool:
        key = tuple(sorted(game))
        if key not in seen:
            seen.add(key)
            unique.append(sorted(game))
    while len(unique) < n_games:
        candidates = []
        for _ in range(100):
            candidate = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(candidate) in seen: continue
            bits_c = johnson.game_to_bits(candidate)
            min_dist = johnson.max_distance
            for existing in unique:
                bits_e = johnson.game_to_bits(existing)
                d = johnson.johnson_distance(bits_c, bits_e)
                min_dist = min(min_dist, d)
            if len(candidates) < 10:
                heapq.heappush(candidates, (-min_dist, tuple(candidate)))
            else:
                heapq.heappushpop(candidates, (-min_dist, tuple(candidate)))
        if candidates:
            best = max(candidates, key=lambda x: -x[0])
            game = list(best[1])
            seen.add(tuple(game))
            unique.append(game)
        else:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                unique.append(game)
    return unique[:n_games]


# ============================================================
# ENTROPIA POSICIONAL + MUTUAL INFORMATION
# ============================================================

def positional_entropy(pool):
    if not pool: return 0.0
    n_games = len(pool)
    pos_entropies = []
    for pos in range(15):
        pos_values = [sorted(g)[pos] for g in pool]
        freq = np.bincount(pos_values, minlength=26)[1:]
        probs = freq / n_games
        probs = np.where(probs > 0, probs, 1e-10)
        pos_entropies.append(entropy(probs))
    max_entropy = np.log(25)
    return float(np.mean(pos_entropies) / max_entropy)


def mutual_information_positions(pool):
    if len(pool) < 2: return 0.0
    mi_values = []
    for pos in range(14):
        pos1_vals = [sorted(g)[pos] for g in pool]
        pos2_vals = [sorted(g)[pos+1] for g in pool]
        contingency = np.zeros((25, 25))
        for v1, v2 in zip(pos1_vals, pos2_vals):
            contingency[v1-1, v2-1] += 1
        joint = contingency / len(pool)
        marginal1 = joint.sum(axis=1)
        marginal2 = joint.sum(axis=0)
        mi = 0.0
        for i in range(25):
            for j in range(25):
                if joint[i, j] > 0:
                    expected = marginal1[i] * marginal2[j]
                    if expected > 0:
                        mi += joint[i, j] * np.log(joint[i, j] / expected)
        mi_values.append(mi)
    return float(np.mean(mi_values))


# ============================================================
# SERIALIZAÇÃO BINÁRIA
# ============================================================

class BinarySerializer:
    @staticmethod
    def pool_to_bytes(pool):
        data = bytearray()
        for game in pool:
            bits = 0
            for d in game:
                bits |= (1 << (d - 1))
            data.extend(struct.pack('>I', bits))
        return bytes(data)
    
    @staticmethod
    def pool_hash(pool):
        return hashlib.md5(BinarySerializer.pool_to_bytes(pool)).hexdigest()
    
    @staticmethod
    def compressibility(pool):
        raw = BinarySerializer.pool_to_bytes(pool)
        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)


# ============================================================
# LAPLACIAN SPECTRUM
# ============================================================

class LaplacianSpectrumSparse:
    def __init__(self, johnson_space):
        self.js = johnson_space
        self._spectral_cache = {}
    
    def similarity_sparse(self, pool, k_neighbors=5):
        n = len(pool)
        D_sparse = self.js.distance_matrix_sparse(pool, k_neighbors)
        sigma = 3.0
        S = D_sparse.copy()
        S.data = np.exp(-S.data**2 / (2 * sigma**2))
        S = S + eye(n, format='csr')
        return S
    
    def laplacian_sparse(self, pool, k_neighbors=5):
        S = self.similarity_sparse(pool, k_neighbors)
        n = len(pool)
        degrees = np.array(S.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-10)
        D_inv_sqrt = diags(d_inv_sqrt, format='csr')
        I = eye(n, format='csr')
        L = I - D_inv_sqrt @ S @ D_inv_sqrt
        return L
    
    def connectivity_score(self, pool):
        pool_hash = BinarySerializer.pool_hash(pool)
        if pool_hash in self._spectral_cache:
            return self._spectral_cache[pool_hash]
        
        n = len(pool)
        try:
            L = self.laplacian_sparse(pool, k_neighbors=5)
            eigenvalues, _ = eigsh(L, k=min(3, n-1), which='SM')
            eigenvalues = np.sort(eigenvalues)
            score = min(1.0, float(eigenvalues[1]) / 2.0) if len(eigenvalues) > 1 else 0.0
        except:
            score = 0.5
        
        if len(self._spectral_cache) < 100:
            self._spectral_cache[pool_hash] = score
        
        return score


# ============================================================
# OBJETIVOS CONFLITANTES
# ============================================================

class ConflictingObjectives:
    def __init__(self, historical_data=None):
        self.johnson = JohnsonSpace()
        self.johnson.pre_generate_cover_samples(2000)
        self.laplacian = LaplacianSpectrumSparse(self.johnson)
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
        self._eval_cache = {}
    
    def evaluate(self, pool, generation=0, max_gen=50, is_elite=False):
        pool_hash = BinarySerializer.pool_hash(pool)
        if pool_hash in self._eval_cache:
            return self._eval_cache[pool_hash].copy()
        
        objectives = np.zeros(6)
        
        # 1. Cobertura de pares
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        objectives[0] = len(covered) / comb(25, 2)
        
        # 2. Distância Johnson mínima
        objectives[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        
        # 3. Covering radius
        objectives[2] = self.johnson.covering_radius_fast(pool)
        
        # 4. Compressibilidade
        objectives[3] = BinarySerializer.compressibility(pool)
        
        # 5. Independência histórica
        if self.historical_data:
            recent = self.historical_data[-50:]
            total_sim = 0
            for game in pool:
                max_sim = max(len(set(game) & set(h)) / 15 for h in recent)
                total_sim += max_sim
            objectives[4] = total_sim / len(pool)
        else:
            objectives[4] = 0.5
        
        # 6. Conectividade algébrica (apenas elite)
        if is_elite:
            objectives[5] = self.laplacian.connectivity_score(pool)
        else:
            objectives[5] = 0.5
        
        if len(self._eval_cache) < 300:
            self._eval_cache[pool_hash] = objectives.copy()
        
        return objectives


# ============================================================
# NSGA-II
# ============================================================

class FastNSGA2:
    def __init__(self, n_games=30, pop_size=80, n_generations=30, historical_data=None):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.objectives = ConflictingObjectives(historical_data)
        self.johnson = JohnsonSpace()
        self.n_obj = 6
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
        population = []
        for _ in range(self.pop_size):
            pool = []
            seen = set()
            for _ in range(self.n_games):
                game = tuple(sorted(np.random.choice(range(1, 26), 15, replace=False)))
                if game not in seen:
                    seen.add(game)
                    pool.append(list(game))
            population.append(pool)
        return population
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II - GERANDO FRONTEIRA DE PARETO")
        print(f"{'='*60}")
        print(f"   Pop: {self.pop_size} | Gen: {self.n_generations} | Jogos: {self.n_games}")
        
        population = self._initialize_population()
        population_obj = [self.objectives.evaluate(p, 0, self.n_generations) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            offspring = []
            while len(offspring) < self.pop_size:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1], population_obj[i2]) else population[i2]
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3], population_obj[i4]) else population[i4]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                child = ensure_unique_pool(child, self.n_games, self.johnson)
                offspring.append(child)
            
            offspring_obj = [self.objectives.evaluate(o, gen, self.n_generations) for o in offspring]
            combined = population + offspring
            combined_obj = population_obj + offspring_obj
            fronts = self._fast_non_dominated_sort(combined_obj)
            
            new_pop, new_obj = [], []
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
            
            # Avaliação espectral na elite a cada 3 gerações
            if gen % 3 == 0:
                fronts = self._fast_non_dominated_sort(population_obj)
                elite_indices = []
                for front in fronts:
                    elite_indices.extend(front)
                    if len(elite_indices) >= int(self.pop_size * 0.3):
                        break
                for idx in elite_indices:
                    population_obj[idx] = self.objectives.evaluate(population[idx], gen, self.n_generations, is_elite=True)
        
        # Resultado final
        final_fronts = self._fast_non_dominated_sort(population_obj)
        pareto_idx = final_fronts[0]
        
        for idx in pareto_idx:
            population_obj[idx] = self.objectives.evaluate(population[idx], self.n_generations, self.n_generations, is_elite=True)
        
        pareto_pools = [ensure_unique_pool(population[i], self.n_games, self.johnson) for i in pareto_idx]
        pareto_obj = [population_obj[i] for i in pareto_idx]
        
        print(f"\n   ✅ Fronteira: {len(pareto_pools)} soluções")
        return pareto_pools, pareto_obj
    
    def _crossover(self, p1, p2):
        mid = self.n_games // 2
        if np.random.random() < 0.5:
            return p1[:mid] + p2[mid:]
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
        if n <= 2: return [float('inf')] * n
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


# ============================================================
# ANÁLISE DE SINAIS
# ============================================================

def compute_all_signals(pareto_pools):
    """Calcula todos os sinais estruturais"""
    johnson = JohnsonSpace()
    johnson.pre_generate_cover_samples(2000)
    
    signals_list = []
    
    for pool in tqdm(pareto_pools, desc="Analisando sinais"):
        pool = ensure_unique_pool(pool, len(pool), johnson)
        
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        
        signals = {
            'pair_coverage': float(len(covered) / comb(25, 2)),
            'johnson_min': float(johnson.min_johnson_distance(pool) / johnson.max_distance),
            'johnson_avg': float(johnson.avg_johnson_distance(pool) / johnson.max_distance),
            'covering_radius': float(johnson.covering_radius_fast(pool)),
            'sphere_packing': float(johnson.sphere_packing_bound(pool)),
            'pos_entropy': float(positional_entropy(pool)),
            'mutual_info': float(mutual_information_positions(pool)),
            'compressibility': float(BinarySerializer.compressibility(pool)),
        }
        
        signals_list.append(signals)
    
    return signals_list


# ============================================================
# EXPORTAÇÃO
# ============================================================

def convert_for_json(obj):
    """Converte tipos numpy para JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_for_json(x) for x in obj]
    return obj


def export_pareto_frontier(pareto_pools, pareto_obj, signals_list, filename='pareto_frontier.json'):
    """Exporta fronteira de Pareto completa"""
    print(f"\n💾 Exportando fronteira para {filename}...")
    
    export_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_solutions': len(pareto_pools),
            'n_games_per_pool': len(pareto_pools[0]) if pareto_pools else 0,
            'objective_names': [
                'Cobertura Pares',
                'Dist Johnson Mín',
                'Covering Radius',
                'Compressibilidade',
                'Indep Histórica',
                'Conectividade Alg'
            ],
            'signal_names': list(signals_list[0].keys()) if signals_list else []
        },
        'pareto_pools': [[sorted(g) for g in pool] for pool in pareto_pools],
        'pareto_objectives': [obj.tolist() for obj in pareto_obj],
        'signals': signals_list
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=convert_for_json)
    
    file_size = os.path.getsize(filename)
    print(f"   ✅ Exportado: {file_size:,} bytes")
    return filename


def main():
    print("="*60)
    print("🧬 MOTOR DE OTIMIZAÇÃO - SYSTEM_FACIL.PY")
    print("="*60)
    
    # Dados históricos simulados
    historical = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(100)]
    
    # Executar NSGA-II
    nsga2 = FastNSGA2(
        n_games=30,
        pop_size=80,
        n_generations=30,
        historical_data=historical
    )
    
    pareto_pools, pareto_obj = nsga2.run()
    
    # Calcular sinais
    signals_list = compute_all_signals(pareto_pools)
    
    # Exportar
    export_pareto_frontier(pareto_pools, pareto_obj, signals_list)
    
    # Estatísticas
    obj_array = np.array(pareto_obj)
    print(f"\n📊 ESTATÍSTICAS DA FRONTEIRA:")
    print(f"   Soluções: {len(pareto_pools)}")
    
    # Correlações
    print(f"\n📊 CORRELAÇÕES (negativo = CONFLITO):")
    names = ['Cob Pares', 'Dist Min', 'Cov Radius', 'Compress', 'Indep Hist', 'Conectiv']
    for i in range(6):
        for j in range(i+1, 6):
            corr = np.corrcoef(obj_array[:, i], obj_array[:, j])[0, 1]
            s = "🔴" if corr < -0.3 else "🟡" if abs(corr) < 0.3 else "🟢"
            print(f"   {names[i]:<12} vs {names[j]:<12}: r={corr:+.3f} {s}")
    
    print(f"\n✅ PRONTO PARA O FRONTEND!")
    print(f"📁 Arquivo: pareto_frontier.json")


if __name__ == "__main__":
    main()
