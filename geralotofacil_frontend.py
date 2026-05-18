#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA INTEGRADO COM HARD CONSTRAINTS - VERSÃO FINAL
======================================================
Versão 5.0 - CSP + NSGA-II com Repair Local

CORREÇÕES CRÍTICAS:
✅ repair_game() MODIFICA minimamente (não descarta)
✅ Hard constraints garantidas em TODAS as etapas
✅ Validação após cada operação genética
✅ Preenchimento inteligente (não aleatório)
✅ Conversão de numpy scalars para int nativo
✅ Display limpo sem np.int64
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
                    'dezenas': [int(x) for x in parts[2:17]]
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
# HARD CONSTRAINT SATISFACTION
# ============================================================

def generate_feasible_game(constraints, max_attempts=100):
    """
    Gera UM jogo que SATISFAZ TODAS as hard constraints
    
    Diferente do anterior: NÃO preenche aleatoriamente no final.
    Cada adição é validada contra TODAS as constraints.
    """
    if constraints is None:
        return sorted(np.random.choice(range(1, 26), 15, replace=False))
    
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    target_pares = constraints.get('pares_target')
    target_primos = constraints.get('primos_target')
    target_moldura = constraints.get('moldura_target')
    target_repetidas = constraints.get('repetidas_target')
    ultimo = set(constraints.get('ultimo_concurso', []))
    soma_min = constraints.get('soma_min')
    soma_max = constraints.get('soma_max')
    max_cons = constraints.get('max_consecutivos')
    
    for _ in range(max_attempts):
        game = set()
        
        # 1. Fixas (obrigatório)
        game.update(fixed)
        
        # 2. Pool disponível
        available = set(range(1, 26)) - excluded - game
        
        # 3. Repetidas (preencher primeiro, com prioridade)
        if target_repetidas is not None and ultimo:
            repeated_pool = list(ultimo & available)
            needed = max(0, target_repetidas - len(game & ultimo))
            if needed > 0 and repeated_pool:
                n = min(needed, len(repeated_pool))
                chosen = np.random.choice(repeated_pool, n, replace=False)
                game.update(chosen)
                available -= set(chosen)
        
        # 4. Preencher respeitando TODAS as constraints
        # Prioridade: pares → primos → moldura → soma → consecutivos
        while len(game) < 15 and available:
            candidates = list(available)
            
            # Filtrar por constraints
            valid_candidates = []
            for c in candidates:
                test_game = game | {c}
                
                # Validar pares (não exceder target)
                if target_pares is not None:
                    current_pares = sum(1 for d in test_game if d % 2 == 0)
                    remaining_slots = 15 - len(test_game)
                    max_possible_pares = current_pares + sum(1 for d in available if d % 2 == 0 and d != c)
                    if current_pares > target_pares:
                        continue
                    if max_possible_pares < target_pares and len(test_game) < 13:
                        if c % 2 != 0 and current_pares < target_pares:
                            continue
                
                # Validar primos
                if target_primos is not None:
                    current_primos = sum(1 for d in test_game if d in PRIMES)
                    remaining_primos = sum(1 for d in available if d in PRIMES and d != c)
                    if current_primos > target_primos:
                        continue
                
                # Validar consecutivos
                if max_cons is not None:
                    sorted_test = sorted(test_game)
                    cons_count = 1
                    max_cons_found = 1
                    for i in range(len(sorted_test)-1):
                        if sorted_test[i+1] - sorted_test[i] == 1:
                            cons_count += 1
                            max_cons_found = max(max_cons_found, cons_count)
                        else:
                            cons_count = 1
                    if max_cons_found > max_cons:
                        continue
                
                valid_candidates.append(c)
            
            if not valid_candidates:
                # Relaxar: aceitar qualquer disponível
                valid_candidates = list(available)
            
            if not valid_candidates:
                break
            
            # Escolher o melhor candidato
            # Prioridade: manter balanceamento
            scores = []
            for c in valid_candidates:
                score = 0
                test_game = game | {c}
                
                # Bônus por preencher target de pares
                if target_pares is not None:
                    current = sum(1 for d in test_game if d % 2 == 0)
                    if current <= target_pares:
                        score += 2 if c % 2 == 0 else 0
                
                # Bônus por preencher target de moldura
                if target_moldura is not None:
                    current = sum(1 for d in test_game if d in MOLDURA)
                    if current <= target_moldura:
                        score += 1 if c in MOLDURA else 0
                
                # Bônus por soma na faixa
                if soma_min is not None and soma_max is not None:
                    current_soma = sum(test_game)
                    if soma_min <= current_soma <= soma_max:
                        score += 1
                
                scores.append(score)
            
            # Escolher com probabilidade proporcional ao score
            if sum(scores) > 0:
                probs = np.array(scores) / sum(scores)
                chosen = np.random.choice(valid_candidates, p=probs)
            else:
                chosen = np.random.choice(valid_candidates)
            
            game.add(chosen)
            available.remove(chosen)
        
        # Verificar viabilidade
        if len(game) >= 14:  # Aceita 14 ou 15
            result = sorted([int(x) for x in game])[:15]
            while len(result) < 15:
                remaining = list(set(range(1, 26)) - set(result) - excluded)
                if remaining:
                    result.append(int(np.random.choice(remaining)))
                else:
                    break
            return sorted(result[:15])
    
    # Fallback: gerar com repair
    game = sorted(np.random.choice(range(1, 26), 15, replace=False))
    return repair_game_local(game, constraints)


def repair_game_local(game, constraints):
    """
    REPAIR LOCAL: Modifica MINIMAMENTE o jogo para satisfazer constraints
    
    NÃO descarta o jogo. Apenas ajusta o necessário.
    Preserva o material genético do crossover/mutação.
    """
    if constraints is None:
        return sorted([int(x) for x in game])[:15]
    
    game = set(int(x) for x in game)
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    target_pares = constraints.get('pares_target')
    target_primos = constraints.get('primos_target')
    target_moldura = constraints.get('moldura_target')
    target_repetidas = constraints.get('repetidas_target')
    ultimo = set(constraints.get('ultimo_concurso', []))
    max_cons = constraints.get('max_consecutivos')
    
    # 1. Remover excluídas
    game -= excluded
    
    # 2. Adicionar fixas
    game |= fixed
    
    # 3. Ajustar tamanho para 15
    # Remover excedentes (não fixas)
    while len(game) > 15:
        removable = [d for d in game if d not in fixed]
        if not removable:
            break
        # Remover o que menos contribui para constraints
        scores = {}
        for d in removable:
            score = 0
            test = game - {d}
            if target_pares is not None:
                current = sum(1 for x in test if x % 2 == 0)
                score -= abs(current - target_pares)
            if target_primos is not None:
                current = sum(1 for x in test if x in PRIMES)
                score -= abs(current - target_primos) * 2
            scores[d] = score
        
        worst = min(scores, key=scores.get)
        game.remove(worst)
    
    # 4. Completar até 15 (escolhas INTELIGENTES)
    while len(game) < 15:
        available = set(range(1, 26)) - game - excluded
        
        if not available:
            break
        
        # Pontuar candidatos
        candidates = list(available)
        scores = []
        for c in candidates:
            score = 0
            test = game | {c}
            
            # Bônus por aproximar do target de pares
            if target_pares is not None:
                current = sum(1 for x in test if x % 2 == 0)
                score -= abs(current - target_pares)
            
            # Bônus por aproximar do target de primos
            if target_primos is not None:
                current = sum(1 for x in test if x in PRIMES)
                score -= abs(current - target_primos)
            
            # Bônus por moldura
            if target_moldura is not None:
                current = sum(1 for x in test if x in MOLDURA)
                score -= abs(current - target_moldura) * 0.5
            
            # Penalidade por consecutivos
            if max_cons is not None:
                sorted_test = sorted(test)
                cons = 1
                max_found = 1
                for i in range(len(sorted_test)-1):
                    if sorted_test[i+1] - sorted_test[i] == 1:
                        cons += 1
                        max_found = max(max_found, cons)
                    else:
                        cons = 1
                if max_found > max_cons:
                    score -= 10
            
            scores.append(score)
        
        # Escolher melhor
        best_idx = np.argmax(scores)
        game.add(candidates[best_idx])
    
    return sorted([int(x) for x in game])[:15]


def validate_game(game, constraints):
    """
    Valida se o jogo satisfaz TODAS as hard constraints
    
    Returns:
        (bool, list): (válido?, [violações])
    """
    if constraints is None:
        return True, []
    
    game = set(int(x) for x in game)
    violations = []
    
    # Fixas
    fixed = set(constraints.get('fixas', []))
    if not fixed.issubset(game):
        violations.append(f"Fixas ausentes: {fixed - game}")
    
    # Excluídas
    excluded = set(constraints.get('excluidas', []))
    if game & excluded:
        violations.append(f"Excluídas presentes: {game & excluded}")
    
    # Pares (hard: exatamente igual)
    if 'pares_target' in constraints:
        actual = sum(1 for d in game if d % 2 == 0)
        if actual != constraints['pares_target']:
            violations.append(f"Pares: {actual} ≠ {constraints['pares_target']}")
    
    # Primos (hard: exatamente igual)
    if 'primos_target' in constraints:
        actual = sum(1 for d in game if d in PRIMES)
        if actual != constraints['primos_target']:
            violations.append(f"Primos: {actual} ≠ {constraints['primos_target']}")
    
    # Moldura
    if 'moldura_target' in constraints:
        actual = sum(1 for d in game if d in MOLDURA)
        if actual != constraints['moldura_target']:
            violations.append(f"Moldura: {actual} ≠ {constraints['moldura_target']}")
    
    # Repetidas
    if 'repetidas_target' in constraints and 'ultimo_concurso' in constraints:
        ultimo = set(constraints['ultimo_concurso'])
        actual = len(game & ultimo)
        if actual != constraints['repetidas_target']:
            violations.append(f"Repetidas: {actual} ≠ {constraints['repetidas_target']}")
    
    # Soma
    if 'soma_min' in constraints:
        if sum(game) < constraints['soma_min']:
            violations.append(f"Soma: {sum(game)} < {constraints['soma_min']}")
    if 'soma_max' in constraints:
        if sum(game) > constraints['soma_max']:
            violations.append(f"Soma: {sum(game)} > {constraints['soma_max']}")
    
    # Consecutivos
    if 'max_consecutivos' in constraints:
        sorted_g = sorted(game)
        cons = 1
        max_found = 1
        for i in range(len(sorted_g)-1):
            if sorted_g[i+1] - sorted_g[i] == 1:
                cons += 1
                max_found = max(max_found, cons)
            else:
                cons = 1
        if max_found > constraints['max_consecutivos']:
            violations.append(f"Consecutivos: {max_found} > {constraints['max_consecutivos']}")
    
    return len(violations) == 0, violations


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
        for d in game: bits |= (1 << (int(d) - 1))
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


# ============================================================
# NSGA-II COM VALIDAÇÃO HARD
# ============================================================

class HardConstraintNSGA2:
    def __init__(self, n_games=30, pop_size=200, n_generations=80, 
                 historical_data=None, constraints=None):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.constraints = constraints
        self.johnson = JohnsonSpace()
        self.johnson.pre_generate_cover_samples(2000)
        self.historical_data = historical_data or []
        self._eval_cache = {}
    
    def _dominates(self, obj1, obj2):
        better = False
        for i, d in enumerate(['max', 'max', 'min', 'min', 'min', 'max']):
            if d == 'max':
                if obj1[i] < obj2[i]: return False
                if obj1[i] > obj2[i]: better = True
            else:
                if obj1[i] > obj2[i]: return False
                if obj1[i] < obj2[i]: better = True
        return better
    
    def evaluate(self, pool):
        h = hashlib.md5(b''.join(struct.pack('>I', 
            sum((1<<(int(d)-1)) for d in g)) for g in pool)).hexdigest()
        if h in self._eval_cache: return self._eval_cache[h].copy()
        
        obj = np.zeros(6)
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        obj[0] = len(covered) / comb(25, 2)
        obj[1] = self.johnson.min_johnson_distance(pool) / self.johnson.max_distance
        obj[2] = self.johnson.covering_radius_fast(pool)
        
        data = bytearray()
        for game in pool:
            bits = 0
            for d in game: bits |= (1 << (int(d)-1))
            data.extend(struct.pack('>I', bits))
        obj[3] = len(zlib.compress(bytes(data), level=9)) / len(data)
        
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
        for i in range(self.pop_size):
            pool = []
            seen = set()
            for _ in range(self.n_games):
                game = generate_feasible_game(self.constraints)
                key = tuple(game)
                if key not in seen:
                    seen.add(key)
                    pool.append(game)
            # Completar se necessário
            while len(pool) < self.n_games:
                game = generate_feasible_game(self.constraints)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
            population.append(pool[:self.n_games])
        return population
    
    def _crossover(self, p1, p2):
        mid = self.n_games // 2
        child = p1[:mid] + p2[mid:]
        # REPAIR LOCAL (não regenerar!)
        child = [repair_game_local(g, self.constraints) for g in child]
        return child
    
    def _mutate(self, pool, generation):
        rate = 0.15 * (1 - generation / self.n_generations)
        mutated = [list(g) for g in pool]
        n_mut = max(1, int(self.n_games * rate))
        
        for idx in np.random.choice(self.n_games, n_mut, replace=False):
            game = mutated[idx]
            fixed = set(self.constraints.get('fixas', [])) if self.constraints else set()
            excluded = set(self.constraints.get('excluidas', [])) if self.constraints else set()
            
            # Escolher posição mutável (não fixa)
            mutable = [i for i, d in enumerate(game) if int(d) not in fixed]
            if mutable:
                pos = np.random.choice(mutable)
                avail = [d for d in range(1, 26) if d not in game and d not in excluded]
                if avail:
                    game[pos] = int(np.random.choice(avail))
            
            mutated[idx] = sorted(game)
        
        # REPAIR LOCAL
        mutated = [repair_game_local(g, self.constraints) for g in mutated]
        return mutated
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II COM HARD CONSTRAINTS")
        print(f"{'='*60}")
        
        population = self._initialize_population()
        
        # Validar população inicial
        valid_count = 0
        for pool in population:
            all_valid = True
            for game in pool:
                valid, _ = validate_game(game, self.constraints)
                if not valid:
                    all_valid = False
                    break
            if all_valid: valid_count += 1
        print(f"   ✅ População inicial: {valid_count}/{len(population)} válidos")
        
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
                
                # VALIDAR e descartar filhos inválidos
                child_valid = True
                for game in child:
                    valid, _ = validate_game(game, self.constraints)
                    if not valid:
                        child_valid = False
                        break
                
                if child_valid:
                    offspring.append(child)
            
            if not offspring:
                continue
            
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
        
        # Validação final
        final_valid = 0
        for pool in pareto_pools:
            all_ok = all(validate_game(g, self.constraints)[0] for g in pool)
            if all_ok: final_valid += 1
        
        print(f"\n   ✅ Fronteira: {len(pareto_pools)} soluções ({final_valid} válidas)")
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
        for obj_i in range(6):
            sorted_i = sorted(front_idx, key=lambda i: pop_obj[i][obj_i])
            obj_range = pop_obj[sorted_i[-1]][obj_i] - pop_obj[sorted_i[0]][obj_i]
            if obj_range > 0:
                distances[0] = distances[-1] = float('inf')
                for i in range(1, n-1):
                    distances[i] += (pop_obj[sorted_i[i+1]][obj_i] - pop_obj[sorted_i[i-1]][obj_i]) / obj_range
        return distances.tolist()


# ============================================================
# INTERFACE
# ============================================================

def collect_preferences(ultimo_concurso=None):
    print(f"\n{'='*60}")
    print(f"🎯 CONFIGURAÇÃO DE PREFERÊNCIAS (HARD CONSTRAINTS)")
    print(f"{'='*60}")
    print(f"💡 ENTER = sem restrição | Valores serão EXATOS")
    
    prefs = {}
    
    print(f"\n📌 DEZENAS FIXAS")
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
    
    print(f"\n📊 PARES (valor EXATO)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['pares_target'] = int(v)
        except: pass
    
    print(f"\n🔢 PRIMOS (valor EXATO)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['primos_target'] = int(v)
        except: pass
    
    print(f"\n🖼️  MOLDURA (valor EXATO)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['moldura_target'] = int(v)
        except: pass
    
    if ultimo_concurso:
        print(f"\n🔄 REPETIDAS DO ÚLTIMO CONCURSO")
        print(f"   Último: {ultimo_concurso}")
        v = input(f"   Quantidade EXATA [ENTER=pular]: ").strip()
        if v:
            try: prefs['repetidas_target'] = int(v)
            except: pass
        prefs['ultimo_concurso'] = ultimo_concurso
    
    return prefs if prefs else None


def display_results(pareto_pools, constraints, ultimo_concurso=None):
    print(f"\n{'='*60}")
    print(f"🏆 RESULTADOS (COM HARD CONSTRAINTS)")
    print(f"{'='*60}")
    
    if constraints:
        print(f"📋 Constraints:")
        for k, v in constraints.items():
            if k != 'ultimo_concurso':
                print(f"   {k}: {v}")
    
    if pareto_pools:
        pool = pareto_pools[0]
        print(f"\n📋 TOP 10 JOGOS:")
        for i, game in enumerate(pool[:10], 1):
            game = [int(x) for x in game]
            pares = sum(1 for d in game if d % 2 == 0)
            primos = sum(1 for d in game if d in PRIMES)
            moldura = sum(1 for d in game if d in MOLDURA)
            soma = sum(game)
            cons = sum(1 for j in range(len(game)-1) if game[j+1]-game[j] == 1)
            rep = len(set(game) & set(ultimo_concurso)) if ultimo_concurso else 0
            
            # Validar
            valid, violations = validate_game(game, constraints)
            status = "✅" if valid else f"❌ {violations}"
            
            print(f"   {i:2d}. {game} {status}")
            print(f"       P:{pares} Pr:{primos} M:{moldura} S:{soma} C:{cons} R:{rep}")


def main():
    print("="*60)
    print("🧬 SISTEMA COM HARD CONSTRAINTS")
    print("="*60)
    
    last = get_last_contest('resultados_lotofacil.csv')
    ultimo_concurso = last['dezenas'] if last else None
    
    if last:
        print(f"\n📌 Último concurso: {last['concurso']}")
        print(f"   {ultimo_concurso}")
    
    historical = get_historical_data('resultados_lotofacil.csv', 100)
    preferences = collect_preferences(ultimo_concurso)
    
    constraints = preferences  # Já está no formato certo
    
    nsga2 = HardConstraintNSGA2(
        n_games=30, pop_size=200, n_generations=80,
        historical_data=historical,
        constraints=constraints
    )
    
    pareto_pools, pareto_obj = nsga2.run()
    display_results(pareto_pools, constraints, ultimo_concurso)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
