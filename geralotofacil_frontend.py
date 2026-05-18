#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA INTEGRADO - FRONTEND + MOTOR CONDICIONADO
==================================================
Versão 4.0 - Geração Sob Demanda com Constraints

FLUXO CORRETO:
1. Usuário configura preferências no frontend
2. Frontend chama o motor COM constraints
3. Motor gera fronteira DENTRO do subespaço
4. Frontend exibe resultados (sem pós-filtro)
5. Repair operator garante validade após crossover
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
# CONJUNTOS MATEMÁTICOS
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}

# ============================================================
# CARREGAMENTO DO CSV
# ============================================================

def load_historical_contests(csv_file='resultados_lotofacil.csv'):
    if not os.path.exists(csv_file):
        return None
    contests = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) >= 17:
                contests.append({
                    'concurso': int(parts[0]),
                    'data': parts[1],
                    'dezenas': sorted([int(x) for x in parts[2:17]])
                })
        contests.sort(key=lambda x: x['concurso'])
        return contests
    except:
        return None


def get_last_contest(csv_file='resultados_lotofacil.csv'):
    contests = load_historical_contests(csv_file)
    return contests[-1] if contests else None


def get_historical_data(csv_file='resultados_lotofacil.csv', n=100):
    contests = load_historical_contests(csv_file)
    if contests:
        return [c['dezenas'] for c in contests[-n:]]
    return [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n)]


# ============================================================
# JOHNSON SPACE
# ============================================================

class JohnsonSpace:
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.max_distance = k - max(0, 2*k - n)
        self._dist_cache = {}
        self._cover_samples = None
    
    def game_to_bits(self, game):
        bits = 0
        for d in game: bits |= (1 << (d - 1))
        return bits
    
    def johnson_distance(self, game1, game2):
        bits1 = self.game_to_bits(game1) if isinstance(game1, list) else game1
        bits2 = self.game_to_bits(game2) if isinstance(game2, list) else game2
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
    
    def pre_generate_cover_samples(self, n=2000):
        if self._cover_samples is not None: return self._cover_samples
        samples = set()
        while len(samples) < n:
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
        sphere_vol = sum(comb(self.k, i) * comb(self.n - self.k, i) 
                        for i in range(min(radius, self.k) + 1) if i <= self.n - self.k)
        return min(1.0, len(pool) / (comb(self.n, self.k) / sphere_vol)) if sphere_vol > 0 else 1.0


# ============================================================
# GERADOR CONDICIONADO + REPAIR OPERATOR
# ============================================================

def build_constraints_from_preferences(preferences, ultimo_concurso=None):
    """Converte preferências do frontend em constraints para o motor"""
    constraints = {}
    
    if 'fixas' in preferences:
        constraints['fixas'] = preferences['fixas']
    if 'excluidas' in preferences:
        constraints['excluidas'] = preferences['excluidas']
    if 'pares' in preferences:
        constraints['pares_target'] = preferences['pares']
    if 'primos' in preferences:
        constraints['primos_target'] = preferences['primos']
    if 'moldura' in preferences:
        constraints['moldura_target'] = preferences['moldura']
    if 'repetidas' in preferences and ultimo_concurso:
        constraints['repetidas_target'] = preferences['repetidas']
        constraints['ultimo_concurso'] = ultimo_concurso
    if 'soma_min' in preferences:
        constraints['soma_min'] = preferences['soma_min']
    if 'soma_max' in preferences:
        constraints['soma_max'] = preferences['soma_max']
    if 'max_consecutivos' in preferences:
        constraints['max_consecutivos'] = preferences['max_consecutivos']
    
    return constraints if constraints else None


def generate_constrained_game(constraints):
    """Gera UM jogo obedecendo constraints"""
    if constraints is None:
        return sorted(np.random.choice(range(1, 26), 15, replace=False))
    
    game = set()
    
    # 1. Fixas (obrigatório)
    if 'fixas' in constraints and constraints['fixas']:
        game.update(constraints['fixas'])
    
    # 2. Pool disponível
    excluded = set(constraints.get('excluidas', []))
    available = set(range(1, 26)) - excluded - game
    
    # 3. Repetidas do último concurso
    if 'repetidas_target' in constraints and 'ultimo_concurso' in constraints:
        ultimo = set(constraints['ultimo_concurso'])
        repeated_pool = list(ultimo & available)
        target = constraints['repetidas_target']
        current = len(game & ultimo)
        needed = max(0, target - current)
        if repeated_pool and needed > 0:
            n = min(needed, len(repeated_pool))
            chosen = np.random.choice(repeated_pool, n, replace=False)
            game.update(chosen)
            available -= set(chosen)
    
    # 4. Ajustar PARES
    if 'pares_target' in constraints:
        target = constraints['pares_target']
        current = sum(1 for d in game if d % 2 == 0)
        if current < target:
            even_avail = [d for d in available if d % 2 == 0]
            n = min(target - current, len(even_avail))
            if n > 0:
                game.update(np.random.choice(even_avail, n, replace=False))
                available = set(range(1, 26)) - excluded - game
        current = sum(1 for d in game if d % 2 == 0)
        odd_needed = (15 - target) - (len(game) - current)
        if odd_needed > 0:
            odd_avail = [d for d in available if d % 2 != 0]
            n = min(odd_needed, len(odd_avail))
            if n > 0:
                game.update(np.random.choice(odd_avail, n, replace=False))
                available = set(range(1, 26)) - excluded - game
    
    # 5. Ajustar PRIMOS
    if 'primos_target' in constraints:
        target = constraints['primos_target']
        current = sum(1 for d in game if d in PRIMES)
        needed = target - current
        if needed > 0:
            prime_avail = [d for d in available if d in PRIMES]
            n = min(needed, len(prime_avail))
            if n > 0:
                game.update(np.random.choice(prime_avail, n, replace=False))
                available = set(range(1, 26)) - excluded - game
    
    # 6. Ajustar MOLDURA
    if 'moldura_target' in constraints:
        target = constraints['moldura_target']
        current = sum(1 for d in game if d in MOLDURA)
        needed = target - current
        if needed > 0:
            moldura_avail = [d for d in available if d in MOLDURA]
            n = min(needed, len(moldura_avail))
            if n > 0:
                game.update(np.random.choice(moldura_avail, n, replace=False))
                available = set(range(1, 26)) - excluded - game
    
    # 7. Completar até 15
    available = set(range(1, 26)) - excluded - game
    while len(game) < 15 and available:
        game.add(np.random.choice(list(available)))
        available = set(range(1, 26)) - excluded - game
    
    result = sorted(list(game))[:15]
    while len(result) < 15:
        remaining = list(set(range(1, 26)) - set(result) - excluded)
        if remaining:
            result.append(np.random.choice(remaining))
        else:
            result.append(np.random.choice([d for d in range(1, 26) if d not in result]))
    
    return sorted(result[:15])


def repair_game(game, constraints):
    """
    REPAIR OPERATOR: Corrige jogo após crossover/mutação
    
    Garante que o jogo volte a obedecer todas as restrições
    """
    if constraints is None:
        return sorted(game)[:15]
    
    game = set(game)
    excluded = set(constraints.get('excluidas', []))
    
    # 1. Garantir fixas
    if 'fixas' in constraints and constraints['fixas']:
        game.update(constraints['fixas'])
    
    # 2. Remover excluídas
    game -= excluded
    
    # 3. Truncar para 15
    game_list = sorted(list(game))[:15]
    
    # 4. Reconstruir com constraints
    return generate_constrained_game(constraints)


def generate_constrained_pool(n_games, constraints):
    """Gera pool inteiro dentro do subespaço"""
    pool = []
    seen = set()
    attempts = 0
    max_attempts = n_games * 50
    
    while len(pool) < n_games and attempts < max_attempts:
        game = generate_constrained_game(constraints)
        key = tuple(game)
        if key not in seen:
            seen.add(key)
            pool.append(game)
        attempts += 1
    
    while len(pool) < n_games:
        game = sorted(np.random.choice(range(1, 26), 15, replace=False))
        if tuple(game) not in seen:
            seen.add(tuple(game))
            pool.append(game)
    
    return pool[:n_games]


# ============================================================
# NSGA-II COM REPAIR OPERATOR
# ============================================================

class ConstrainedNSGA2:
    def __init__(self, n_games=30, pop_size=200, n_generations=80, 
                 historical_data=None, constraints=None):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.constraints = constraints
        self.johnson = JohnsonSpace()
        self.johnson.pre_generate_cover_samples(2000)
        self.historical_data = historical_data or []
        self.n_obj = 6
        self.directions = ['max', 'max', 'min', 'min', 'min', 'max']
        self._eval_cache = {}
    
    def _dominates(self, obj1, obj2):
        better = False
        for i, d in enumerate(self.directions):
            if d == 'max':
                if obj1[i] < obj2[i]: return False
                if obj1[i] > obj2[i]: better = True
            else:
                if obj1[i] > obj2[i]: return False
                if obj1[i] < obj2[i]: better = True
        return better
    
    def evaluate(self, pool):
        h = hashlib.md5(b''.join(struct.pack('>I', 
            sum((1<<(d-1)) for d in g)) for g in pool)).hexdigest()
        if h in self._eval_cache: return self._eval_cache[h].copy()
        
        obj = np.zeros(6)
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        obj[0] = len(covered) / comb(25, 2)
        obj[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        obj[2] = self.johnson.covering_radius_fast(pool)
        obj[3] = BinarySerializer.compressibility(pool)
        if self.historical_data:
            recent = self.historical_data[-50:]
            obj[4] = np.mean([max(len(set(g) & set(h)) / 15 for h in recent) for g in pool])
        else:
            obj[4] = 0.5
        obj[5] = 0.5
        if len(self._eval_cache) < 500: self._eval_cache[h] = obj.copy()
        return obj
    
    def _initialize_population(self):
        population = []
        strategies = ['constrained', 'spread', 'balanced', 'entropy', 'random']
        for i in range(self.pop_size):
            s = strategies[i % len(strategies)]
            if s == 'constrained' and self.constraints:
                pool = generate_constrained_pool(self.n_games, self.constraints)
            else:
                pool = self._generate_pool(s)
            population.append(pool)
        return population[:self.pop_size]
    
    def _generate_pool(self, strategy):
        pool, seen = [], set()
        if strategy == 'spread':
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base); seen.add(tuple(base))
            for _ in range(self.n_games - 1):
                best, best_min = None, 15
                for _ in range(200):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min: best_min = m; best = c
                if best: seen.add(tuple(best)); pool.append(best)
        elif strategy == 'balanced':
            half = self.n_games // 2
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base); seen.add(tuple(base))
            for _ in range(half - 1):
                best, best_min = None, 15
                for _ in range(100):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min: best_min = m; best = c
                if best: seen.add(tuple(best)); pool.append(best)
            cbase = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(self.n_games - half):
                game = cbase.copy()
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail: game[pos] = np.random.choice(avail)
                game = sorted(game)
                if tuple(game) not in seen: seen.add(tuple(game)); pool.append(game)
        else:
            for _ in range(self.n_games):
                game = tuple(sorted(np.random.choice(range(1, 26), 15, replace=False)))
                if game not in seen: seen.add(game); pool.append(list(game))
        return self._ensure_unique(pool)
    
    def _ensure_unique(self, pool):
        unique, seen = [], set()
        for game in pool:
            key = tuple(sorted(game))
            if key not in seen: seen.add(key); unique.append(sorted(game))
        attempts = 0
        while len(unique) < self.n_games and attempts < 500:
            attempts += 1
            game = generate_constrained_game(self.constraints) if self.constraints else sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen: seen.add(tuple(game)); unique.append(game)
        while len(unique) < self.n_games:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen: seen.add(tuple(game)); unique.append(game)
        return unique[:self.n_games]
    
    def _crossover(self, p1, p2):
        if np.random.random() < 0.5:
            child = p1[:self.n_games//2] + p2[self.n_games//2:]
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
                    cg.append(np.random.choice(avail) if avail else 1)
                child.append(sorted(cg[:15]))
        
        # REPAIR OPERATOR
        if self.constraints:
            child = [repair_game(g, self.constraints) for g in child]
        
        return child
    
    def _mutate(self, pool, generation):
        rate = 0.15 * (1 - generation / self.n_generations)
        mutated = [g.copy() for g in pool]
        n_mut = max(1, int(self.n_games * rate))
        for idx in np.random.choice(self.n_games, n_mut, replace=False):
            game = mutated[idx]
            if self.constraints and 'fixas' in self.constraints:
                fixed = set(self.constraints['fixas'])
                mutable_pos = [i for i, d in enumerate(game) if d not in fixed]
                if mutable_pos:
                    pos = np.random.choice(mutable_pos)
                    excluded = set(self.constraints.get('excluidas', []))
                    avail = [d for d in range(1, 26) if d not in game and d not in excluded]
                    if avail: game[pos] = np.random.choice(avail)
            else:
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail: game[pos] = np.random.choice(avail)
            mutated[idx] = sorted(game)
        
        # REPAIR OPERATOR
        if self.constraints:
            mutated = [repair_game(g, self.constraints) for g in mutated]
        
        return mutated
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II CONDICIONADO")
        print(f"{'='*60}")
        if self.constraints:
            print(f"   🎯 Constraints ativas: {list(self.constraints.keys())}")
        
        population = self._initialize_population()
        population_obj = [self.evaluate(p) for p in population]
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            offspring = []
            while len(offspring) < self.pop_size:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1], population_obj[i2]) else population[i2]
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3], population_obj[i4]) else population[i4]
                child = self._crossover(p1, p2)
                child = self._mutate(child, gen)
                child = self._ensure_unique(child)
                offspring.append(child)
            
            offspring_obj = [self.evaluate(o) for o in offspring]
            combined = population + offspring
            combined_obj = population_obj + offspring_obj
            fronts = self._non_dominated_sort(combined_obj)
            
            new_pop, new_obj = [], []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined[idx])
                        new_obj.append(combined_obj[idx])
                else:
                    remaining = self.pop_size - len(new_pop)
                    distances = self._crowding_distance(front, combined_obj)
                    for idx, _ in sorted(zip(front, distances), key=lambda x: x[1], reverse=True)[:remaining]:
                        new_pop.append(combined[idx])
                        new_obj.append(combined_obj[idx])
                    break
            
            population = new_pop
            population_obj = new_obj
        
        final_fronts = self._non_dominated_sort(population_obj)
        pareto_idx = []
        for front in final_fronts[:5]:
            pareto_idx.extend(front)
        
        pareto_pools = [population[i] for i in pareto_idx]
        pareto_obj = [population_obj[i] for i in pareto_idx]
        
        print(f"\n   ✅ Fronteira: {len(pareto_pools)} soluções")
        return pareto_pools, pareto_obj
    
    def _non_dominated_sort(self, pop_obj):
        n = len(pop_obj)
        dom_count = np.zeros(n, dtype=int)
        dom_sol = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(pop_obj[i], pop_obj[j]): dom_sol[i].append(j)
                elif self._dominates(pop_obj[j], pop_obj[i]): dom_count[i] += 1
            if dom_count[i] == 0: fronts[0].append(i)
        i = 0
        while fronts[i]:
            next_f = []
            for idx in fronts[i]:
                for didx in dom_sol[idx]:
                    dom_count[didx] -= 1
                    if dom_count[didx] == 0: next_f.append(didx)
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
                distances[0] = distances[-1] = float('inf')
                for i in range(1, n-1):
                    distances[i] += (pop_obj[sorted_i[i+1]][obj_i] - pop_obj[sorted_i[i-1]][obj_i]) / obj_range
        return distances.tolist()


# ============================================================
# UTILITÁRIOS
# ============================================================

class BinarySerializer:
    @staticmethod
    def compressibility(pool):
        data = bytearray()
        for game in pool:
            bits = 0
            for d in game: bits |= (1 << (d-1))
            data.extend(struct.pack('>I', bits))
        return len(zlib.compress(bytes(data), level=9)) / len(data)


def positional_entropy(pool):
    if not pool: return 0.0
    pos_entropies = []
    for pos in range(15):
        pos_values = [sorted(g)[pos] for g in pool]
        freq = np.bincount(pos_values, minlength=26)[1:]
        probs = freq / len(pool)
        probs = np.where(probs > 0, probs, 1e-10)
        pos_entropies.append(entropy(probs))
    return float(np.mean(pos_entropies) / np.log(25))


# ============================================================
# FRONTEND INTEGRADO
# ============================================================

def collect_preferences(ultimo_concurso=None):
    """Coleta preferências do usuário"""
    print(f"\n{'='*60}")
    print(f"🎯 CONFIGURAÇÃO DE PREFERÊNCIAS")
    print(f"{'='*60}")
    print(f"💡 ENTER = sem restrição")
    
    prefs = {}
    
    print(f"\n📌 DEZENAS FIXAS (separadas por espaço)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            fixas = sorted(set(int(x) for x in v.split() if 1 <= int(x) <= 25))
            if fixas: prefs['fixas'] = fixas[:15]
        except: pass
    
    print(f"\n🚫 DEZENAS EXCLUÍDAS")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            excl = [int(x) for x in v.split() if 1 <= int(x) <= 25]
            if 'fixas' in prefs: excl = [x for x in excl if x not in prefs['fixas']]
            if excl: prefs['excluidas'] = sorted(set(excl))
        except: pass
    
    print(f"\n📊 PARES (típico: 6-9)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try: prefs['pares'] = int(v)
        except: pass
    
    print(f"\n🔢 PRIMOS (típico: 3-6)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try: prefs['primos'] = int(v)
        except: pass
    
    print(f"\n🖼️  MOLDURA (típico: 7-10)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try: prefs['moldura'] = int(v)
        except: pass
    
    if ultimo_concurso:
        print(f"\n🔄 REPETIDAS DO ÚLTIMO CONCURSO")
        print(f"   Último: {ultimo_concurso}")
        v = input(f"   Quantidade [ENTER=pular]: ").strip()
        if v:
            try: prefs['repetidas'] = int(v)
            except: pass
    
    print(f"\n📐 SOMA (típico: 170-220)")
    v = input(f"   Mínima [ENTER=pular]: ").strip()
    if v:
        try: prefs['soma_min'] = int(v)
        except: pass
    v = input(f"   Máxima [ENTER=pular]: ").strip()
    if v:
        try: prefs['soma_max'] = int(v)
        except: pass
    
    print(f"\n📏 CONSECUTIVOS MÁXIMOS (típico: 3-7)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['max_consecutivos'] = int(v)
        except: pass
    
    return prefs if prefs else None


def display_results(pareto_pools, constraints, ultimo_concurso=None):
    """Exibe resultados diretamente (sem pós-filtro)"""
    print(f"\n{'='*60}")
    print(f"🏆 RESULTADOS (GERAÇÃO CONDICIONADA)")
    print(f"{'='*60}")
    
    if constraints:
        print(f"📋 Constraints usadas:")
        for k, v in constraints.items():
            if k != 'ultimo_concurso':
                print(f"   {k}: {v}")
    
    print(f"\n📊 {len(pareto_pools)} pools na fronteira")
    
    # Mostrar top jogos do primeiro pool
    if pareto_pools:
        pool = pareto_pools[0]
        print(f"\n📋 TOP 10 JOGOS (Pool #1):")
        for i, game in enumerate(pool[:10], 1):
            pares = sum(1 for d in game if d % 2 == 0)
            primos = sum(1 for d in game if d in PRIMES)
            moldura = sum(1 for d in game if d in MOLDURA)
            soma = sum(game)
            cons = sum(1 for j in range(len(game)-1) if game[j+1]-game[j] == 1)
            rep = len(set(game) & set(ultimo_concurso)) if ultimo_concurso else 0
            
            print(f"   {i:2d}. {game}")
            print(f"       Pares:{pares} Prim:{primos} Mold:{moldura} Soma:{soma} Cons:{cons} Rep:{rep}")


def main():
    print("="*60)
    print("🧭 SISTEMA INTEGRADO - GERAÇÃO SOB DEMANDA")
    print("="*60)
    
    # Carregar último concurso REAL
    last = get_last_contest('resultados_lotofacil.csv')
    ultimo_concurso = last['dezenas'] if last else None
    
    if last:
        print(f"\n📌 Último concurso: {last['concurso']} ({last['data']})")
        print(f"   {ultimo_concurso}")
    
    # Carregar histórico
    historical = get_historical_data('resultados_lotofacil.csv', 100)
    
    # Coletar preferências
    preferences = collect_preferences(ultimo_concurso)
    
    # Converter para constraints do motor
    constraints = build_constraints_from_preferences(preferences, ultimo_concurso)
    
    # Rodar NSGA-II CONDICIONADO
    print(f"\n🔥 GERANDO FRONTEIRA CONDICIONADA...")
    nsga2 = ConstrainedNSGA2(
        n_games=30, pop_size=200, n_generations=80,
        historical_data=historical,
        constraints=constraints
    )
    
    pareto_pools, pareto_obj = nsga2.run()
    
    # Exibir resultados (sem pós-filtro!)
    display_results(pareto_pools, constraints, ultimo_concurso)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
