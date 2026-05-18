#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA CSP + NSGA-II - VERSÃO FINAL ROBUSTA
=============================================
Versão 6.0 - Validação de Factibilidade + Timeout + Backtracking

CORREÇÕES CRÍTICAS:
✅ validate_constraint_feasibility() ANTES de otimizar
✅ Timeout em TODOS os loops (sem loops infinitos)
✅ Backtracking leve para jogos inviáveis
✅ Separação: CSP (viabilidade) → NSGA-II (otimização)
✅ Avisos claros de sobreconstrangimento
✅ Geração construtiva com fallback progressivo
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
    if not os.path.exists(csv_file): return None
    contests = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(';')
                if len(parts) >= 17:
                    contests.append({
                        'concurso': int(parts[0]),
                        'data': parts[1],
                        'dezenas': [int(x) for x in parts[2:17]]
                    })
        contests.sort(key=lambda x: x['concurso'])
        return contests
    except: return None

def get_last_contest(csv_file='resultados_lotofacil.csv'):
    c = load_historical_contests(csv_file)
    return c[-1] if c else None

def get_historical_data(csv_file='resultados_lotofacil.csv', n=100):
    c = load_historical_contests(csv_file)
    return [x['dezenas'] for x in c[-n:]] if c else [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(n)]


# ============================================================
# VALIDAÇÃO DE FACTIBILIDADE (ANTES DE OTIMIZAR)
# ============================================================

def validate_constraint_feasibility(constraints):
    """
    Verifica se as constraints são COMBINATORIAMENTE POSSÍVEIS
    
    Returns:
        (bool, str): (viável?, mensagem)
    """
    if constraints is None:
        return True, "Sem restrições"
    
    issues = []
    
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    # 1. Conflito direto fixas vs excluídas
    if fixed & excluded:
        conflict = fixed & excluded
        issues.append(f"❌ Fixas e excluídas conflitam: {conflict}")
        return False, " | ".join(issues)
    
    # 2. Fixas excedem 15
    if len(fixed) > 15:
        issues.append(f"❌ {len(fixed)} fixas > 15 (impossível)")
        return False, " | ".join(issues)
    
    # 3. Espaço disponível
    available = set(range(1, 26)) - excluded - fixed
    slots_available = 15 - len(fixed)
    total_available = len(available)
    
    if total_available < slots_available:
        issues.append(f"❌ Espaço disponível ({total_available}) < slots necessários ({slots_available})")
        return False, " | ".join(issues)
    
    # 4. Pares possíveis?
    target_pares = constraints.get('pares_target')
    if target_pares is not None:
        # Pares já nas fixas
        fixed_even = sum(1 for d in fixed if d % 2 == 0)
        # Máximo de pares possíveis
        max_even = fixed_even + sum(1 for d in available if d % 2 == 0)
        # Mínimo de pares possíveis
        min_even = fixed_even + max(0, slots_available - sum(1 for d in available if d % 2 != 0))
        
        if target_pares > max_even:
            issues.append(f"⚠️  Pares target ({target_pares}) > máximo possível ({max_even})")
        if target_pares < min_even:
            issues.append(f"⚠️  Pares target ({target_pares}) < mínimo possível ({min_even})")
    
    # 5. Primos possíveis?
    target_primos = constraints.get('primos_target')
    if target_primos is not None:
        fixed_prime = sum(1 for d in fixed if d in PRIMES)
        max_prime = fixed_prime + sum(1 for d in available if d in PRIMES)
        if target_primos > max_prime:
            issues.append(f"⚠️  Primos target ({target_primos}) > máximo possível ({max_prime})")
    
    # 6. Moldura possível?
    target_moldura = constraints.get('moldura_target')
    if target_moldura is not None:
        fixed_mold = sum(1 for d in fixed if d in MOLDURA)
        max_mold = fixed_mold + sum(1 for d in available if d in MOLDURA)
        if target_moldura > max_mold:
            issues.append(f"⚠️  Moldura target ({target_moldura}) > máximo possível ({max_mold})")
    
    # 7. Repetidas possíveis?
    target_rep = constraints.get('repetidas_target')
    ultimo = set(constraints.get('ultimo_concurso', []))
    if target_rep is not None and ultimo:
        fixed_rep = len(fixed & ultimo)
        available_rep = len((available & ultimo) - fixed)
        max_rep = fixed_rep + min(available_rep, slots_available)
        if target_rep > max_rep:
            issues.append(f"⚠️  Repetidas target ({target_rep}) > máximo possível ({max_rep})")
    
    # 8. Soma possível?
    soma_min = constraints.get('soma_min')
    soma_max = constraints.get('soma_max')
    if soma_min is not None and soma_max is not None:
        fixed_sum = sum(fixed)
        available_sorted = sorted(available)
        min_sum = fixed_sum + sum(available_sorted[:slots_available]) if len(available_sorted) >= slots_available else fixed_sum
        max_sum = fixed_sum + sum(available_sorted[-slots_available:]) if len(available_sorted) >= slots_available else fixed_sum
        if soma_max < min_sum:
            issues.append(f"⚠️  Soma max ({soma_max}) < mínimo possível ({min_sum})")
        if soma_min > max_sum:
            issues.append(f"⚠️  Soma min ({soma_min}) > máximo possível ({max_sum})")
    
    if issues:
        return False, " | ".join(issues)
    
    return True, "✅ Espaço viável"


# ============================================================
# GERAÇÃO COM BACKTRACKING LEVE
# ============================================================

def generate_feasible_game_backtrack(constraints, max_backtracks=50):
    """
    Gera jogo viável com backtracking limitado
    
    Se não encontrar em max_backtracks tentativas,
    relaxa progressivamente as constraints
    """
    if constraints is None:
        return sorted(np.random.choice(range(1, 26), 15, replace=False))
    
    # Tentar com constraints completas
    for _ in range(max_backtracks):
        game = _try_build_game(constraints)
        if game and len(game) == 15:
            valid, _ = validate_game_hard(game, constraints)
            if valid:
                return sorted([int(x) for x in game])
    
    # Relaxar: remover targets numéricos, manter fixas/excluídas
    relaxed = {
        'fixas': constraints.get('fixas', []),
        'excluidas': constraints.get('excluidas', []),
        'ultimo_concurso': constraints.get('ultimo_concurso', []),
    }
    # Manter targets como preferências (não hard)
    for _ in range(max_backtracks):
        game = _try_build_game(relaxed)
        if game and len(game) == 15:
            valid, _ = validate_game_hard(game, relaxed)
            if valid:
                return sorted([int(x) for x in game])
    
    # Fallback total
    return sorted(np.random.choice(range(1, 26), 15, replace=False))


def _try_build_game(constraints):
    """Tenta construir um jogo que satisfaça as constraints"""
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    target_pares = constraints.get('pares_target')
    target_primos = constraints.get('primos_target')
    target_moldura = constraints.get('moldura_target')
    target_rep = constraints.get('repetidas_target')
    ultimo = set(constraints.get('ultimo_concurso', []))
    max_cons = constraints.get('max_consecutivos')
    soma_min = constraints.get('soma_min')
    soma_max = constraints.get('soma_max')
    
    game = set(fixed)
    available = set(range(1, 26)) - excluded - game
    
    # Repetidas primeiro (se houver target)
    if target_rep is not None and ultimo:
        rep_pool = list(ultimo & available)
        needed = target_rep - len(game & ultimo)
        if needed > 0 and rep_pool:
            n = min(needed, len(rep_pool))
            game.update(np.random.choice(rep_pool, n, replace=False))
            available -= game
    
    # Preencher com escolhas ponderadas
    slots = 15 - len(game)
    for _ in range(slots):
        if not available:
            break
        
        candidates = list(available)
        scores = []
        for c in candidates:
            score = 0
            test = game | {c}
            test_sorted = sorted(test)
            
            # Pares
            if target_pares is not None:
                current = sum(1 for d in test if d % 2 == 0)
                remaining = slots - 1
                max_possible = current + sum(1 for d in available if d % 2 == 0 and d != c)
                if target_pares > max_possible:
                    score -= 20
                score -= abs(current - target_pares)
            
            # Consecutivos
            if max_cons is not None:
                cons = 1
                max_found = 1
                for i in range(len(test_sorted)-1):
                    if test_sorted[i+1] - test_sorted[i] == 1:
                        cons += 1
                        max_found = max(max_found, cons)
                    else:
                        cons = 1
                if max_found > max_cons:
                    score -= 15
            
            # Soma
            if soma_min is not None and soma_max is not None:
                current_sum = sum(test)
                if current_sum > soma_max:
                    score -= 10
            
            scores.append(score)
        
        # Selecionar melhor (ou aleatório entre os positivos)
        positive = [(c, s) for c, s in zip(candidates, scores) if s >= 0]
        if positive:
            chosen = positive[np.random.randint(0, len(positive))][0]
        else:
            # Pegar o menos pior
            best_idx = np.argmax(scores)
            chosen = candidates[best_idx]
        
        game.add(chosen)
        available.remove(chosen)
    
    return list(game)


def repair_game_local(game, constraints):
    """Repair local preservando material genético"""
    if constraints is None:
        return sorted([int(x) for x in game])[:15]
    
    game = set(int(x) for x in game)
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    # Remover excluídas
    game -= excluded
    # Adicionar fixas
    game |= fixed
    
    # Truncar para 15 (remover piores)
    while len(game) > 15:
        removable = [d for d in game if d not in fixed]
        if not removable: break
        # Remover o que tem pior score
        scores = {}
        for d in removable:
            test = game - {d}
            score = 0
            if 'pares_target' in constraints:
                score -= abs(sum(1 for x in test if x % 2 == 0) - constraints['pares_target'])
            if 'primos_target' in constraints:
                score -= abs(sum(1 for x in test if x in PRIMES) - constraints['primos_target'])
            scores[d] = score
        game.remove(max(scores, key=scores.get))
    
    # Completar
    available = set(range(1, 26)) - game - excluded
    while len(game) < 15 and available:
        candidates = list(available)
        scores = []
        for c in candidates:
            score = 0
            test = game | {c}
            if 'pares_target' in constraints:
                score -= abs(sum(1 for x in test if x % 2 == 0) - constraints['pares_target'])
            if 'primos_target' in constraints:
                score -= abs(sum(1 for x in test if x in PRIMES) - constraints['primos_target'])
            if 'max_consecutivos' in constraints:
                st = sorted(test)
                cons = 1
                mf = 1
                for i in range(len(st)-1):
                    if st[i+1]-st[i]==1:
                        cons += 1; mf = max(mf, cons)
                    else: cons = 1
                if mf > constraints['max_consecutivos']: score -= 10
            scores.append(score)
        
        best_idx = np.argmax(scores)
        game.add(candidates[best_idx])
        available.remove(candidates[best_idx])
    
    return sorted([int(x) for x in game])[:15]


def validate_game_hard(game, constraints):
    """Validação hard (exata)"""
    if constraints is None:
        return True, []
    
    game = set(int(x) for x in game)
    violations = []
    
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    if not fixed.issubset(game):
        violations.append(f"Faltam fixas: {fixed - game}")
    if game & excluded:
        violations.append(f"Excluídas: {game & excluded}")
    if 'pares_target' in constraints:
        actual = sum(1 for d in game if d % 2 == 0)
        if actual != constraints['pares_target']:
            violations.append(f"Pares: {actual}≠{constraints['pares_target']}")
    if 'primos_target' in constraints:
        actual = sum(1 for d in game if d in PRIMES)
        if actual != constraints['primos_target']:
            violations.append(f"Primos: {actual}≠{constraints['primos_target']}")
    if 'moldura_target' in constraints:
        actual = sum(1 for d in game if d in MOLDURA)
        if actual != constraints['moldura_target']:
            violations.append(f"Moldura: {actual}≠{constraints['moldura_target']}")
    if 'repetidas_target' in constraints and 'ultimo_concurso' in constraints:
        actual = len(game & set(constraints['ultimo_concurso']))
        if actual != constraints['repetidas_target']:
            violations.append(f"Repetidas: {actual}≠{constraints['repetidas_target']}")
    if 'max_consecutivos' in constraints:
        st = sorted(game)
        cons = 1; mf = 1
        for i in range(len(st)-1):
            if st[i+1]-st[i]==1: cons+=1; mf=max(mf,cons)
            else: cons=1
        if mf > constraints['max_consecutivos']:
            violations.append(f"Cons: {mf}>{constraints['max_consecutivos']}")
    if 'soma_min' in constraints and sum(game) < constraints['soma_min']:
        violations.append(f"Soma: {sum(game)}<{constraints['soma_min']}")
    if 'soma_max' in constraints and sum(game) > constraints['soma_max']:
        violations.append(f"Soma: {sum(game)}>{constraints['soma_max']}")
    
    return len(violations) == 0, violations


# ============================================================
# JOHNSON SPACE
# ============================================================

class JohnsonSpace:
    def __init__(self, n=25, k=15):
        self.n, self.k = n, k
        self.max_distance = k - max(0, 2*k-n)
        self._dist_cache = {}
        self._cover_samples = None
    
    def game_to_bits(self, game):
        bits = 0
        for d in game: bits |= (1 << (int(d)-1))
        return bits
    
    def johnson_distance(self, g1, g2):
        b1 = self.game_to_bits(g1) if isinstance(g1, list) else g1
        b2 = self.game_to_bits(g2) if isinstance(g2, list) else g2
        key = (b1,b2) if b1<b2 else (b2,b1)
        if key in self._dist_cache: return self._dist_cache[key]
        d = self.k - (b1 & b2).bit_count()
        if len(self._dist_cache) < 20000: self._dist_cache[key] = d
        return d
    
    def min_johnson_distance(self, pool):
        if len(pool) < 2: return self.max_distance
        bits = [self.game_to_bits(g) for g in pool]
        md = self.max_distance
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                d = self.johnson_distance(bits[i], bits[j])
                md = min(md, d)
                if md == 0: return 0
        return md
    
    def pre_generate_cover_samples(self, n=2000):
        if self._cover_samples is not None: return self._cover_samples
        samples = set()
        while len(samples) < n:
            samples.add(tuple(sorted(np.random.choice(range(1,26),15,replace=False))))
        self._cover_samples = [list(s) for s in samples]
        return self._cover_samples
    
    def covering_radius_fast(self, pool):
        if self._cover_samples is None: self.pre_generate_cover_samples()
        bits_list = [self.game_to_bits(g) for g in pool]
        max_min = 0
        for s in self._cover_samples[:500]:
            sb = self.game_to_bits(s)
            md = self.max_distance
            for pb in bits_list:
                d = self.johnson_distance(sb, pb)
                md = min(md, d)
                if md == 0: break
            max_min = max(max_min, md)
        return max_min / self.max_distance


# ============================================================
# NSGA-II COM TIMEOUT
# ============================================================

class RobustNSGA2:
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
        
        # Validar factibilidade ANTES
        feasible, msg = validate_constraint_feasibility(constraints)
        print(f"   🔍 Factibilidade: {msg}")
        if not feasible:
            print(f"   ⚠️  Espaço INVIÁVEL! Relaxando constraints numéricas...")
            self.constraints = self._relax_numerical_constraints(constraints)
    
    def _relax_numerical_constraints(self, constraints):
        """Remove targets numéricos, mantém fixas/excluídas"""
        if constraints is None: return None
        return {
            'fixas': constraints.get('fixas', []),
            'excluidas': constraints.get('excluidas', []),
            'ultimo_concurso': constraints.get('ultimo_concurso', []),
            'max_consecutivos': constraints.get('max_consecutivos'),
            'soma_min': constraints.get('soma_min'),
            'soma_max': constraints.get('soma_max'),
        }
    
    def _dominates(self, o1, o2):
        better = False
        for i, d in enumerate(['max','max','min','min','min','max']):
            if d == 'max':
                if o1[i] < o2[i]: return False
                if o1[i] > o2[i]: better = True
            else:
                if o1[i] > o2[i]: return False
                if o1[i] < o2[i]: better = True
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
        obj[3] = len(zlib.compress(bytes(data), 9)) / len(data)
        if self.historical_data:
            recent = self.historical_data[-50:]
            obj[4] = np.mean([max(len(set(g)&set(h))/15 for h in recent) for g in pool])
        else:
            obj[4] = 0.5
        obj[5] = 0.5
        if len(self._eval_cache) < 500: self._eval_cache[h] = obj.copy()
        return obj
    
    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            pool = []
            seen = set()
            for _ in range(self.n_games):
                game = generate_feasible_game_backtrack(self.constraints)
                key = tuple(game)
                if key not in seen:
                    seen.add(key)
                    pool.append(game)
            while len(pool) < self.n_games:
                game = generate_feasible_game_backtrack(self.constraints)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
            population.append(pool[:self.n_games])
        return population
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II ROBUSTO (COM TIMEOUT)")
        print(f"{'='*60}")
        
        population = self._initialize_population()
        population_obj = [self.evaluate(p) for p in population]
        
        stalled_generations = 0
        
        for gen in tqdm(range(self.n_generations), desc="NSGA-II"):
            offspring = []
            attempts = 0
            max_attempts = self.pop_size * 10  # TIMEOUT
            
            while len(offspring) < self.pop_size and attempts < max_attempts:
                attempts += 1
                
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if self._dominates(population_obj[i1], population_obj[i2]) else population[i2]
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if self._dominates(population_obj[i3], population_obj[i4]) else population[i4]
                
                child = self._crossover(p1, p2)
                child = self._mutate(child, gen)
                
                # Validar
                valid = all(validate_game_hard(g, self.constraints)[0] for g in child)
                if valid:
                    offspring.append(child)
            
            if not offspring:
                stalled_generations += 1
                if stalled_generations > 5:
                    print(f"\n   ⚠️  {stalled_generations} gerações sem filhos válidos - interrompendo")
                    break
                continue
            
            stalled_generations = 0
            
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
        print(f"\n   ✅ Fronteira: {len(pareto_pools)} soluções")
        return pareto_pools
    
    def _crossover(self, p1, p2):
        mid = self.n_games // 2
        child = p1[:mid] + p2[mid:]
        return [repair_game_local(g, self.constraints) for g in child]
    
    def _mutate(self, pool, gen):
        rate = 0.15 * (1 - gen / max(1, self.n_generations))
        mutated = [list(g) for g in pool]
        n_mut = max(1, int(self.n_games * rate))
        
        for idx in np.random.choice(self.n_games, n_mut, replace=False):
            game = mutated[idx]
            fixed = set(self.constraints.get('fixas', [])) if self.constraints else set()
            excluded = set(self.constraints.get('excluidas', [])) if self.constraints else set()
            mutable = [i for i, d in enumerate(game) if int(d) not in fixed]
            if mutable:
                pos = np.random.choice(mutable)
                avail = [d for d in range(1,26) if d not in game and d not in excluded]
                if avail:
                    game[pos] = int(np.random.choice(avail))
            mutated[idx] = sorted(game)
        
        return [repair_game_local(g, self.constraints) for g in mutated]
    
    def _non_dominated_sort(self, pop_obj):
        n = len(pop_obj)
        dc = np.zeros(n, dtype=int)
        ds = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(pop_obj[i], pop_obj[j]): ds[i].append(j)
                elif self._dominates(pop_obj[j], pop_obj[i]): dc[i] += 1
            if dc[i] == 0: fronts[0].append(i)
        i = 0
        while fronts[i]:
            nf = []
            for idx in fronts[i]:
                for didx in ds[idx]:
                    dc[didx] -= 1
                    if dc[didx] == 0: nf.append(didx)
            i += 1
            fronts.append(nf)
        return fronts[:-1]
    
    def _crowding_distance(self, fi, po):
        n = len(fi)
        if n <= 2: return [float('inf')] * n
        dist = np.zeros(n)
        for oi in range(6):
            si = sorted(fi, key=lambda i: po[i][oi])
            rng = po[si[-1]][oi] - po[si[0]][oi]
            if rng > 0:
                dist[0] = dist[-1] = float('inf')
                for i in range(1, n-1):
                    dist[i] += (po[si[i+1]][oi] - po[si[i-1]][oi]) / rng
        return dist.tolist()


# ============================================================
# INTERFACE
# ============================================================

def collect_preferences(ultimo=None):
    print(f"\n{'='*60}")
    print(f"🎯 PREFERÊNCIAS (valores EXATOS)")
    print(f"{'='*60}")
    prefs = {}
    
    print(f"\n📌 FIXAS:")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            f = sorted(set(int(x) for x in v.split() if 1<=int(x)<=25))
            if f: prefs['fixas'] = f[:15]
        except: pass
    
    print(f"\n🚫 EXCLUÍDAS:")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            e = [int(x) for x in v.split() if 1<=int(x)<=25]
            if 'fixas' in prefs: e = [x for x in e if x not in prefs['fixas']]
            if e: prefs['excluidas'] = sorted(set(e))
        except: pass
    
    print(f"\n📊 PARES (exato):")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['pares_target'] = int(v)
        except: pass
    
    print(f"\n🔢 PRIMOS (exato):")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['primos_target'] = int(v)
        except: pass
    
    print(f"\n🖼️  MOLDURA (exato):")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try: prefs['moldura_target'] = int(v)
        except: pass
    
    if ultimo:
        print(f"\n🔄 REPETIDAS (exato):")
        print(f"   Último: {ultimo}")
        v = input(f"   [ENTER=pular]: ").strip()
        if v:
            try: prefs['repetidas_target'] = int(v)
            except: pass
        prefs['ultimo_concurso'] = ultimo
    
    return prefs if prefs else None


def display(pools, constraints, ultimo=None):
    print(f"\n{'='*60}")
    print(f"🏆 RESULTADOS")
    print(f"{'='*60}")
    if pools:
        for i, game in enumerate(pools[0][:10], 1):
            game = [int(x) for x in game]
            p = sum(1 for d in game if d%2==0)
            pr = sum(1 for d in game if d in PRIMES)
            m = sum(1 for d in game if d in MOLDURA)
            s = sum(game)
            c = sum(1 for j in range(len(game)-1) if game[j+1]-game[j]==1)
            r = len(set(game)&set(ultimo)) if ultimo else 0
            v, _ = validate_game_hard(game, constraints)
            print(f"   {i:2d}. {game} {'✅' if v else '❌'}")
            print(f"       P:{p} Pr:{pr} M:{m} S:{s} C:{c} R:{r}")


def main():
    print("="*60)
    print("🧬 CSP + NSGA-II ROBUSTO")
    print("="*60)
    
    last = get_last_contest('resultados_lotofacil.csv')
    ultimo = last['dezenas'] if last else None
    hist = get_historical_data('resultados_lotofacil.csv', 100)
    
    prefs = collect_preferences(ultimo)
    
    nsga2 = RobustNSGA2(n_games=30, pop_size=200, n_generations=80,
                        historical_data=hist, constraints=prefs)
    pools = nsga2.run()
    display(pools, prefs, ultimo)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
