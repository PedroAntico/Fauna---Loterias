#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOTOR DE OTIMIZAÇÃO - EXPORTAÇÃO COMPLETA
==========================================
Versão 3.0 - Perfis + Fronts + Signatures

NOVAS EXPORTAÇÕES:
✅ profiles (pré-classificados, sem recalcular)
✅ front_ids (qual front cada pool pertence)
✅ signatures (vetor contínuo de scores por perfil)
✅ Metadados completos para frontend
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
# JOHNSON SPACE (mantido igual)
# ============================================================

class JohnsonSpace:
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.total_codes = comb(n, k)
        self.min_intersection = max(0, 2*k - n)
        self.max_distance = k - self.min_intersection
        self._dist_cache = {}
        self._cover_samples = None
    
    def game_to_bits(self, game):
        bits = 0
        for d in game:
            bits |= (1 << (d - 1))
        return bits
    
    def johnson_distance(self, game1, game2):
        if isinstance(game1, list): bits1 = self.game_to_bits(game1)
        else: bits1 = game1
        if isinstance(game2, list): bits2 = self.game_to_bits(game2)
        else: bits2 = game2
        key = (bits1, bits2) if bits1 < bits2 else (bits2, bits1)
        if key in self._dist_cache: return self._dist_cache[key]
        d = self.k - (bits1 & bits2).bit_count()
        if len(self._dist_cache) < 20000: self._dist_cache[key] = d
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
        if self._cover_samples is not None: return self._cover_samples
        samples = set()
        while len(samples) < n_samples:
            game = tuple(sorted(np.random.choice(range(1, self.n+1), self.k, replace=False)))
            samples.add(game)
        self._cover_samples = [list(s) for s in samples]
        return self._cover_samples
    
    def covering_radius_fast(self, pool):
        if self._cover_samples is None: self.pre_generate_cover_samples()
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
        return min(1.0, len(pool) / (self.total_codes / sphere_vol))
    
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
                row_indices.append(i); col_indices.append(j); data.append(d)
        return csr_matrix((data, (row_indices, col_indices)), shape=(n, n))


# ============================================================
# UTILITÁRIOS (mantidos)
# ============================================================

def ensure_unique_pool(pool, n_games, johnson=None):
    if johnson is None: johnson = JohnsonSpace()
    unique, seen = [], set()
    for game in pool:
        key = tuple(sorted(game))
        if key not in seen:
            seen.add(key); unique.append(sorted(game))
    while len(unique) < n_games:
        candidates = []
        for _ in range(100):
            candidate = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(candidate) in seen: continue
            bits_c = johnson.game_to_bits(candidate)
            min_dist = johnson.max_distance
            for existing in unique:
                d = johnson.johnson_distance(bits_c, johnson.game_to_bits(existing))
                min_dist = min(min_dist, d)
            if len(candidates) < 10: heapq.heappush(candidates, (-min_dist, tuple(candidate)))
            else: heapq.heappushpop(candidates, (-min_dist, tuple(candidate)))
        if candidates:
            best = max(candidates, key=lambda x: -x[0])
            game = list(best[1]); seen.add(tuple(game)); unique.append(game)
        else:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen: seen.add(tuple(game)); unique.append(game)
    return unique[:n_games]

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
    return float(np.mean(pos_entropies) / np.log(25))

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
        marginal1, marginal2 = joint.sum(axis=1), joint.sum(axis=0)
        mi = 0.0
        for i in range(25):
            for j in range(25):
                if joint[i,j] > 0:
                    expected = marginal1[i] * marginal2[j]
                    if expected > 0: mi += joint[i,j] * np.log(joint[i,j] / expected)
        mi_values.append(mi)
    return float(np.mean(mi_values))

class BinarySerializer:
    @staticmethod
    def pool_to_bytes(pool):
        data = bytearray()
        for game in pool:
            bits = 0
            for d in game: bits |= (1 << (d-1))
            data.extend(struct.pack('>I', bits))
        return bytes(data)
    @staticmethod
    def compressibility(pool):
        raw = BinarySerializer.pool_to_bytes(pool)
        return len(zlib.compress(raw, level=9)) / len(raw)

class LaplacianSpectrumSparse:
    def __init__(self, johnson_space):
        self.js = johnson_space; self._cache = {}
    def connectivity_score(self, pool):
        h = hashlib.md5(BinarySerializer.pool_to_bytes(pool)).hexdigest()
        if h in self._cache: return self._cache[h]
        n = len(pool)
        try:
            D = self.js.distance_matrix_sparse(pool, 5)
            sigma = 3.0; S = D.copy(); S.data = np.exp(-S.data**2/(2*sigma**2))
            S = S + eye(n, format='csr')
            deg = np.array(S.sum(axis=1)).flatten()
            d_inv = 1.0/np.sqrt(deg+1e-10)
            L = eye(n, format='csr') - diags(d_inv, format='csr') @ S @ diags(d_inv, format='csr')
            ev, _ = eigsh(L, k=min(3,n-1), which='SM'); ev = np.sort(ev)
            score = min(1.0, float(ev[1])/2.0) if len(ev)>1 else 0.0
        except: score = 0.5
        if len(self._cache) < 100: self._cache[h] = score
        return score

class ConflictingObjectives:
    def __init__(self, historical_data=None):
        self.johnson = JohnsonSpace(); self.johnson.pre_generate_cover_samples(2000)
        self.laplacian = LaplacianSpectrumSparse(self.johnson)
        self.historical_data = historical_data or []; self._eval_cache = {}
    def evaluate(self, pool, is_elite=False):
        h = hashlib.md5(BinarySerializer.pool_to_bytes(pool)).hexdigest()
        if h in self._eval_cache: return self._eval_cache[h].copy()
        obj = np.zeros(6)
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2): covered.add(pair)
        obj[0] = len(covered)/comb(25,2)
        obj[1] = self.johnson.min_johnson_distance(pool)/self.johnson.max_distance
        obj[2] = self.johnson.covering_radius_fast(pool)
        obj[3] = BinarySerializer.compressibility(pool)
        if self.historical_data:
            recent = self.historical_data[-50:]
            obj[4] = np.mean([max(len(set(g)&set(h))/15 for h in recent) for g in pool])
        else: obj[4] = 0.5
        obj[5] = self.laplacian.connectivity_score(pool) if is_elite else 0.5
        if len(self._eval_cache) < 500: self._eval_cache[h] = obj.copy()
        return obj


# ============================================================
# NSGA-II ROBUSTO (mantido)
# ============================================================

class RobustNSGA2:
    def __init__(self, n_games=30, pop_size=300, n_generations=120, historical_data=None):
        self.n_games = n_games; self.pop_size = pop_size; self.n_generations = n_generations
        self.objectives = ConflictingObjectives(historical_data)
        self.johnson = JohnsonSpace()
        self.n_obj = 6
        self.directions = ['max','max','min','min','min','max']
        self.n_fronts_to_keep = 5
    
    def _dominates(self, o1, o2):
        better = False
        for i,d in enumerate(self.directions):
            if d=='max':
                if o1[i]<o2[i]: return False
                if o1[i]>o2[i]: better = True
            else:
                if o1[i]>o2[i]: return False
                if o1[i]<o2[i]: better = True
        return better
    
    def _initialize_population(self):
        pop = []
        strategies = ['random','spread','clustered','balanced','entropy']
        for i in range(self.pop_size):
            pop.append(self._generate_pool(strategies[i%5]))
        return pop
    
    def _generate_pool(self, strategy):
        pool, seen = [], set()
        if strategy == 'random':
            for _ in range(self.n_games):
                g = tuple(sorted(np.random.choice(range(1,26),15,replace=False)))
                if g not in seen: seen.add(g); pool.append(list(g))
        elif strategy == 'spread':
            base = sorted(np.random.choice(range(1,26),15,replace=False))
            pool.append(base); seen.add(tuple(base))
            for _ in range(self.n_games-1):
                best, bmin = None, 15
                for _ in range(200):
                    c = sorted(np.random.choice(range(1,26),15,replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c)&set(g)) for g in pool)
                    if m < bmin: bmin = m; best = c
                if best: seen.add(tuple(best)); pool.append(best)
        elif strategy == 'clustered':
            base = sorted(np.random.choice(range(1,26),15,replace=False))
            for _ in range(self.n_games):
                g = base.copy()
                for _ in range(np.random.randint(1,4)):
                    pos = np.random.randint(0,15)
                    avail = [d for d in range(1,26) if d not in g]
                    if avail: g[pos] = np.random.choice(avail)
                g = sorted(g)
                if tuple(g) not in seen: seen.add(tuple(g)); pool.append(g)
        elif strategy == 'balanced':
            half = self.n_games//2
            base = sorted(np.random.choice(range(1,26),15,replace=False))
            pool.append(base); seen.add(tuple(base))
            for _ in range(half-1):
                best, bmin = None, 15
                for _ in range(100):
                    c = sorted(np.random.choice(range(1,26),15,replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c)&set(g)) for g in pool)
                    if m < bmin: bmin = m; best = c
                if best: seen.add(tuple(best)); pool.append(best)
            cbase = sorted(np.random.choice(range(1,26),15,replace=False))
            for _ in range(self.n_games-half):
                g = cbase.copy()
                pos = np.random.randint(0,15)
                avail = [d for d in range(1,26) if d not in g]
                if avail: g[pos] = np.random.choice(avail)
                g = sorted(g)
                if tuple(g) not in seen: seen.add(tuple(g)); pool.append(g)
        elif strategy == 'entropy':
            for _ in range(self.n_games):
                g = []
                for pos in range(15):
                    low = max(1, int(pos*1.5))
                    high = min(25, int(25-(14-pos)*1.5))
                    avail = [d for d in range(low,high+1) if d not in g]
                    if avail: g.append(np.random.choice(avail))
                    else:
                        avail_all = [d for d in range(1,26) if d not in g]
                        g.append(np.random.choice(avail_all))
                g = sorted(g)
                if tuple(g) not in seen: seen.add(tuple(g)); pool.append(g)
        return ensure_unique_pool(pool, self.n_games, self.johnson)
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II ROBUSTO v3.0")
        print(f"{'='*60}")
        print(f"   Pop: {self.pop_size} | Gen: {self.n_generations} | Jogos: {self.n_games}")
        print(f"   Fronts: {self.n_fronts_to_keep} | Nichos: 5 estratégias")
        
        population = self._initialize_population()
        population_obj = [self.objectives.evaluate(p) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            offspring = []
            while len(offspring) < self.pop_size:
                i1,i2 = np.random.choice(self.pop_size,2,replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1],population_obj[i2]) else population[i2]
                i3,i4 = np.random.choice(self.pop_size,2,replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3],population_obj[i4]) else population[i4]
                child = self._crossover(p1,p2)
                child = self._mutate(child,gen)
                child = ensure_unique_pool(child,self.n_games,self.johnson)
                offspring.append(child)
            
            offspring_obj = [self.objectives.evaluate(o) for o in offspring]
            combined = population + offspring
            combined_obj = population_obj + offspring_obj
            fronts = self._fast_non_dominated_sort(combined_obj)
            
            new_pop, new_obj = [], []
            for front in fronts:
                if len(new_pop)+len(front) <= self.pop_size:
                    for idx in front: new_pop.append(combined[idx]); new_obj.append(combined_obj[idx])
                else:
                    rem = self.pop_size-len(new_pop)
                    dist = self._crowding_distance(front,combined_obj)
                    sf = sorted(zip(front,dist),key=lambda x:x[1],reverse=True)
                    for idx,_ in sf[:rem]: new_pop.append(combined[idx]); new_obj.append(combined_obj[idx])
                    break
            
            population, population_obj = new_pop, new_obj
            
            if gen % 5 == 0:
                fronts = self._fast_non_dominated_sort(population_obj)
                elite = []
                for front in fronts[:3]:
                    elite.extend(front)
                    if len(elite) >= int(self.pop_size*0.2): break
                for idx in elite: population_obj[idx] = self.objectives.evaluate(population[idx],is_elite=True)
        
        final_fronts = self._fast_non_dominated_sort(population_obj)
        
        # COLETAR COM FRONT IDs
        pareto_idx, front_ids = [], []
        for front_id, front in enumerate(final_fronts[:self.n_fronts_to_keep]):
            for idx in front:
                pareto_idx.append(idx)
                front_ids.append(front_id)
        
        for idx in pareto_idx:
            population_obj[idx] = self.objectives.evaluate(population[idx],is_elite=True)
        
        pareto_pools = [ensure_unique_pool(population[i],self.n_games,self.johnson) for i in pareto_idx]
        pareto_obj = [population_obj[i] for i in pareto_idx]
        
        print(f"\n   ✅ {len(pareto_pools)} soluções (front 0: {len(final_fronts[0])})")
        return pareto_pools, pareto_obj, front_ids
    
    def _crossover(self, p1, p2):
        if np.random.random() < 0.5:
            mid = self.n_games//2; return p1[:mid] + p2[mid:]
        child = []
        for i in range(self.n_games):
            g1,g2 = p1[i],p2[i]
            common = list(set(g1)&set(g2))
            only1 = list(set(g1)-set(g2)); only2 = list(set(g2)-set(g1))
            cg = common + only1[:len(only1)//2] + only2[:len(only2)//2]
            while len(cg) < 15:
                avail = [d for d in range(1,26) if d not in cg]
                cg.append(np.random.choice(avail))
            child.append(sorted(cg[:15]))
        return child
    
    def _mutate(self, pool, gen):
        rate = 0.15*(1-gen/self.n_generations)
        mutated = [g.copy() for g in pool]
        n_mut = max(1,int(self.n_games*rate))
        indices = np.random.choice(self.n_games,n_mut,replace=False)
        for idx in indices:
            game = mutated[idx]
            s = np.random.choice(['drift','jump','restart'],p=[0.6,0.3,0.1])
            if s == 'drift':
                pos = np.random.randint(0,15)
                avail = [d for d in range(1,26) if d not in game]
                if avail: game[pos] = np.random.choice(avail)
            elif s == 'jump':
                for _ in range(np.random.randint(2,5)):
                    pos = np.random.randint(0,15)
                    avail = [d for d in range(1,26) if d not in game]
                    if avail: game[pos] = np.random.choice(avail)
            else: game = sorted(np.random.choice(range(1,26),15,replace=False))
            mutated[idx] = sorted(game)
        return mutated
    
    def _fast_non_dominated_sort(self, pop_obj):
        n = len(pop_obj)
        dom_count = np.zeros(n,dtype=int)
        dom_sol = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                if self._dominates(pop_obj[i],pop_obj[j]): dom_sol[i].append(j)
                elif self._dominates(pop_obj[j],pop_obj[i]): dom_count[i] += 1
            if dom_count[i]==0: fronts[0].append(i)
        i=0
        while fronts[i]:
            nf = []
            for idx in fronts[i]:
                for didx in dom_sol[idx]:
                    dom_count[didx] -= 1
                    if dom_count[didx]==0: nf.append(didx)
            i+=1; fronts.append(nf)
        return fronts[:-1]
    
    def _crowding_distance(self, front_idx, pop_obj):
        n = len(front_idx)
        if n<=2: return [float('inf')]*n
        distances = np.zeros(n)
        for obj_i in range(self.n_obj):
            si = sorted(front_idx,key=lambda i:pop_obj[i][obj_i])
            r = pop_obj[si[-1]][obj_i]-pop_obj[si[0]][obj_i]
            if r>0:
                distances[0]=distances[-1]=float('inf')
                for i in range(1,n-1):
                    distances[i] += (pop_obj[si[i+1]][obj_i]-pop_obj[si[i-1]][obj_i])/r
        return distances.tolist()


# ============================================================
# CLASSIFICAÇÃO DE PERFIS + SIGNATURES
# ============================================================

def classify_profiles_with_signatures(signals_list):
    """
    Classifica perfis E gera signatures contínuas
    
    Returns:
        profiles: lista de dicts com profile, scores, signature
    """
    if not signals_list:
        return []
    
    keys = list(signals_list[0].keys())
    ranges = {}
    for key in keys:
        values = [s[key] for s in signals_list]
        ranges[key] = (np.min(values), np.max(values))
    
    profiles = []
    
    for signals in signals_list:
        norm = {}
        for key in keys:
            vmin, vmax = ranges[key]
            norm[key] = (signals[key]-vmin)/(vmax-vmin) if vmax>vmin else 0.5
        
        # Scores por perfil
        scores = {
            'conservador': (
                (1-norm.get('pos_entropy',0.5))*1.5 +
                norm.get('compressibility',0.5)*1.5 +
                (1-norm.get('johnson_avg',0.5))*1.0
            ),
            'caotico': (
                norm.get('johnson_min',0.5)*1.5 +
                norm.get('johnson_avg',0.5)*1.5 +
                norm.get('pos_entropy',0.5)*1.0
            ),
            'cobertura': (
                norm.get('pair_coverage',0.5)*1.5 +
                (1-norm.get('covering_radius',0.5))*1.5 +
                norm.get('sphere_packing',0.5)*1.0
            ),
            'balanceado': (
                norm.get('pair_coverage',0.5)*1.0 +
                norm.get('johnson_min',0.5)*1.0 +
                (1-norm.get('covering_radius',0.5))*1.0 +
                norm.get('pos_entropy',0.5)*1.0
            )
        }
        
        # Signature contínua (0-1 para cada perfil)
        max_score = max(scores.values()) if scores else 1.0
        signature = {k: v/max_score for k, v in scores.items()} if max_score > 0 else {k:0 for k in scores}
        
        best_profile = max(scores, key=scores.get)
        
        profiles.append({
            'profile': best_profile,
            'scores': scores,
            'signature': signature,
            'signals': signals
        })
    
    return profiles


# ============================================================
# COMPUTAÇÃO DE SINAIS
# ============================================================

def compute_all_signals(pareto_pools):
    johnson = JohnsonSpace(); johnson.pre_generate_cover_samples(2000)
    signals_list = []
    for pool in tqdm(pareto_pools, desc="Analisando sinais"):
        pool = ensure_unique_pool(pool, len(pool), johnson)
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2): covered.add(pair)
        signals = {
            'pair_coverage': float(len(covered)/comb(25,2)),
            'johnson_min': float(johnson.min_johnson_distance(pool)/johnson.max_distance),
            'johnson_avg': float(johnson.avg_johnson_distance(pool)/johnson.max_distance),
            'covering_radius': float(johnson.covering_radius_fast(pool)),
            'sphere_packing': float(johnson.sphere_packing_bound(pool)),
            'pos_entropy': float(positional_entropy(pool)),
            'mutual_info': float(mutual_information_positions(pool)),
            'compressibility': float(BinarySerializer.compressibility(pool)),
        }
        signals_list.append(signals)
    return signals_list


# ============================================================
# EXPORTAÇÃO COMPLETA
# ============================================================

def export_complete_frontier(pareto_pools, pareto_obj, front_ids, signals_list, profiles, filename='pareto_frontier.json'):
    print(f"\n💾 Exportando fronteira COMPLETA...")
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, list): return [convert(x) for x in obj]
        if isinstance(obj, dict): return {k: convert(v) for k,v in obj.items()}
        return obj
    
    export_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_solutions': len(pareto_pools),
            'n_games_per_pool': len(pareto_pools[0]) if pareto_pools else 0,
            'n_fronts': len(set(front_ids)) if front_ids else 0,
            'objective_names': [
                'Cobertura Pares', 'Dist Johnson Mín', 'Covering Radius',
                'Compressibilidade', 'Indep Histórica', 'Conectividade Alg'
            ],
            'signal_names': list(signals_list[0].keys()) if signals_list else [],
            'profile_names': ['conservador', 'caotico', 'cobertura', 'balanceado']
        },
        'pareto_pools': [[sorted(g) for g in pool] for pool in pareto_pools],
        'pareto_objectives': [obj.tolist() for obj in pareto_obj],
        'front_ids': front_ids,
        'signals': signals_list,
        'profiles': [
            {
                'profile': p['profile'],
                'signature': p['signature'],
                'scores': p['scores']
            } for p in profiles
        ]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=convert)
    
    print(f"   ✅ {os.path.getsize(filename):,} bytes")
    print(f"   📊 {len(set(front_ids))} fronts | {len(profiles)} perfis | signatures contínuas")
    return filename


def main():
    print("="*60)
    print("🧬 MOTOR v3.0 - EXPORTAÇÃO COMPLETA")
    print("="*60)
    
    historical = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(100)]
    
    nsga2 = RobustNSGA2(n_games=30, pop_size=300, n_generations=120, historical_data=historical)
    pareto_pools, pareto_obj, front_ids = nsga2.run()
    
    signals_list = compute_all_signals(pareto_pools)
    profiles = classify_profiles_with_signatures(signals_list)
    
    export_complete_frontier(pareto_pools, pareto_obj, front_ids, signals_list, profiles)
    
    # Estatísticas
    profile_counts = Counter(p['profile'] for p in profiles)
    print(f"\n📊 PERFIS:")
    for profile, count in profile_counts.most_common():
        print(f"   {profile:<15} {count} pools")
    
    # Assinatura média por perfil
    print(f"\n📊 SIGNATURES MÉDIAS:")
    by_profile = defaultdict(list)
    for p in profiles:
        by_profile[p['profile']].append(p['signature'])
    for profile, sigs in by_profile.items():
        avg_sig = {k: np.mean([s[k] for s in sigs]) for k in sigs[0]}
        print(f"   {profile}: {json.dumps({k:round(v,3) for k,v in avg_sig.items()})}")
    
    obj_array = np.array(pareto_obj)
    names = ['Cob Pares', 'Dist Min', 'Cov Radius', 'Compress', 'Indep Hist', 'Conectiv']
    print(f"\n📊 CORRELAÇÕES:")
    for i in range(6):
        for j in range(i+1, 6):
            corr = np.corrcoef(obj_array[:,i], obj_array[:,j])[0,1]
            s = "🔴" if corr<-0.3 else "🟡" if abs(corr)<0.3 else "🟢"
            print(f"   {names[i]:<12} vs {names[j]:<12}: r={corr:+.3f} {s}")
    
    print(f"\n✅ PRONTO! pareto_frontier.json")


if __name__ == "__main__":
    main()
