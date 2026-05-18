#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOTOR DE OTIMIZAÇÃO CONDICIONADO - SYSTEM_FACIL.PY
===================================================
Versão 3.0 - Geração Condicionada + Balanceamento de Fronteira

CORREÇÕES CRÍTICAS:
✅ CSV: ordenação automática por número do concurso
✅ Geração condicionada (não pós-filtro)
✅ NSGA-II balanceado (anti-colapso conservador)
✅ Inicialização dentro do subespaço válido
✅ Penalização de dominância de perfil
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
# CARREGAMENTO DO CSV (CORRIGIDO)
# ============================================================

def load_historical_contests(csv_file='resultados_lotofacil.csv'):
    """
    Carrega TODOS os concursos com ordenação automática
    
    Correção: não depende da ordem do arquivo
    """
    if not os.path.exists(csv_file):
        return None
    
    contests = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Pular cabeçalho
            parts = line.strip().split(';')
            if len(parts) >= 17:
                contests.append({
                    'concurso': int(parts[0]),
                    'data': parts[1],
                    'dezenas': sorted([int(x) for x in parts[2:17]])
                })
        
        # Ordenar por número do concurso (crescente)
        contests.sort(key=lambda x: x['concurso'])
        return contests
    except Exception as e:
        print(f"⚠️ Erro ao ler CSV: {e}")
        return None


def get_last_contest(csv_file='resultados_lotofacil.csv'):
    """Retorna o ÚLTIMO concurso (maior número)"""
    contests = load_historical_contests(csv_file)
    if contests:
        return contests[-1]
    return None


def get_historical_data(csv_file='resultados_lotofacil.csv', n_recent=100):
    """Retorna últimos N concursos para análise histórica"""
    contests = load_historical_contests(csv_file)
    if contests:
        return [c['dezenas'] for c in contests[-n_recent:]]
    return [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_recent)]


# ============================================================
# JOHNSON SPACE J(25,15)
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
        for d in game:
            bits |= (1 << (d - 1))
        return bits
    
    def johnson_distance(self, game1, game2):
        if isinstance(game1, list): bits1 = self.game_to_bits(game1)
        else: bits1 = game1
        if isinstance(game2, list): bits2 = self.game_to_bits(game2)
        else: bits2 = game2
        
        key = (bits1, bits2) if bits1 < bits2 else (bits2, bits1)
        if key in self._dist_cache:
            return self._dist_cache[key]
        
        d = self.k - (bits1 & bits2).bit_count()
        if len(self._dist_cache) < 20000:
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
        sphere_vol = sum(comb(self.k, i) * comb(self.n - self.k, i) 
                        for i in range(min(radius, self.k) + 1) if i <= self.n - self.k)
        if sphere_vol == 0: return 1.0
        return min(1.0, len(pool) / (comb(self.n, self.k) / sphere_vol))


# ============================================================
# GERADOR CONDICIONADO (PRÉ-FILTRO)
# ============================================================

def generate_constrained_game(constraints=None):
    """
    Gera UM jogo já obedecendo restrições
    
    Isso evita o problema de gerar → filtrar depois
    O jogo já NASCE dentro do subespaço válido
    
    Args:
        constraints: dict com restrições (fixas, excluídas, pares, etc.)
    
    Returns:
        list: Jogo de 15 dezenas
    """
    if constraints is None:
        return sorted(np.random.choice(range(1, 26), 15, replace=False))
    
    game = set()
    
    # 1. Adicionar FIXAS (obrigatório)
    if 'fixas' in constraints and constraints['fixas']:
        game.update(constraints['fixas'])
    
    # 2. Pool de dezenas disponíveis
    excluded = set(constraints.get('excluidas', []))
    available = set(range(1, 26)) - excluded - game
    
    # 3. Adicionar REPETIDAS do último concurso
    if 'repetidas_target' in constraints and constraints.get('ultimo_concurso'):
        ultimo = set(constraints['ultimo_concurso'])
        repeated_pool = list(ultimo & available)
        
        target_rep = constraints['repetidas_target']
        current_rep = len(game & ultimo)
        needed_rep = max(0, target_rep - current_rep)
        
        if repeated_pool and needed_rep > 0:
            n_rep = min(needed_rep, len(repeated_pool))
            chosen_rep = np.random.choice(repeated_pool, n_rep, replace=False)
            game.update(chosen_rep)
            available -= set(chosen_rep)
    
    # 4. Ajustar PARES
    if 'pares_target' in constraints:
        target_pares = constraints['pares_target']
        current_pares = sum(1 for d in game if d % 2 == 0)
        
        # Se precisa de mais pares
        if current_pares < target_pares:
            even_available = [d for d in available if d % 2 == 0]
            needed = target_pares - current_pares
            if even_available:
                n = min(needed, len(even_available))
                chosen = np.random.choice(even_available, n, replace=False)
                game.update(chosen)
                available -= set(chosen)
        
        # Se precisa de mais ímpares
        current_pares = sum(1 for d in game if d % 2 == 0)
        odd_needed = (15 - target_pares) - (len(game) - current_pares)
        if odd_needed > 0:
            odd_available = [d for d in available if d % 2 != 0]
            if odd_available:
                n = min(odd_needed, len(odd_available))
                chosen = np.random.choice(odd_available, n, replace=False)
                game.update(chosen)
                available -= set(chosen)
    
    # 5. Ajustar PRIMOS
    if 'primos_target' in constraints:
        target_primos = constraints['primos_target']
        current_primos = sum(1 for d in game if d in PRIMES)
        needed = target_primos - current_primos
        
        if needed > 0:
            prime_available = [d for d in available if d in PRIMES]
            if prime_available:
                n = min(needed, len(prime_available))
                chosen = np.random.choice(prime_available, n, replace=False)
                game.update(chosen)
                available -= set(chosen)
    
    # 6. Ajustar MOLDURA
    if 'moldura_target' in constraints:
        target_moldura = constraints['moldura_target']
        current_moldura = sum(1 for d in game if d in MOLDURA)
        needed = target_moldura - current_moldura
        
        if needed > 0:
            moldura_available = [d for d in available if d in MOLDURA]
            if moldura_available:
                n = min(needed, len(moldura_available))
                chosen = np.random.choice(moldura_available, n, replace=False)
                game.update(chosen)
                available -= set(chosen)
    
    # 7. Completar até 15 dezenas
    while len(game) < 15 and available:
        # Preferir dezenas que ajudem no balanceamento
        if len(game) < 8:
            # Preferir dezenas baixas no início
            low_available = [d for d in available if d <= 12]
            if low_available:
                game.add(np.random.choice(low_available))
                available.remove(list(game)[-1])
                continue
        
        game.add(np.random.choice(list(available)))
        available.remove(list(game)[-1])
    
    # 8. Garantir 15 dezenas
    result = sorted(list(game))[:15]
    while len(result) < 15:
        remaining = list(set(range(1, 26)) - set(result))
        if remaining:
            result.append(np.random.choice(remaining))
        else:
            break
    
    return sorted(result[:15])


def generate_constrained_pool(n_games, constraints=None):
    """Gera pool inteiro com restrições"""
    pool = []
    seen = set()
    max_attempts = n_games * 50
    attempts = 0
    
    while len(pool) < n_games and attempts < max_attempts:
        game = generate_constrained_game(constraints)
        key = tuple(game)
        if key not in seen:
            seen.add(key)
            pool.append(game)
        attempts += 1
    
    # Se não gerou suficiente, completar sem restrições
    while len(pool) < n_games:
        game = sorted(np.random.choice(range(1, 26), 15, replace=False))
        if tuple(game) not in seen:
            seen.add(tuple(game))
            pool.append(game)
    
    return pool[:n_games]


# ============================================================
# MÉTRICAS
# ============================================================

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
        mi = sum(joint[i,j] * np.log(joint[i,j] / (marginal1[i] * marginal2[j]))
                for i in range(25) for j in range(25)
                if joint[i,j] > 0 and marginal1[i] > 0 and marginal2[j] > 0)
        mi_values.append(mi)
    return float(np.mean(mi_values))


class BinarySerializer:
    @staticmethod
    def compressibility(pool):
        data = bytearray()
        for game in pool:
            bits = 0
            for d in game: bits |= (1 << (d-1))
            data.extend(struct.pack('>I', bits))
        return len(zlib.compress(bytes(data), level=9)) / len(data)


# ============================================================
# NSGA-II BALANCEADO (ANTI-COLAPSO)
# ============================================================

class BalancedNSGA2:
    def __init__(self, n_games=30, pop_size=300, n_generations=120, 
                 historical_data=None, constraints=None):
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.constraints = constraints  # Restrições para geração condicionada
        self.johnson = JohnsonSpace()
        self.johnson.pre_generate_cover_samples(2000)
        self.historical_data = historical_data or []
        
        self.n_obj = 6
        self.directions = ['max', 'max', 'min', 'min', 'min', 'max']
        self.n_fronts_to_keep = 5
        
        # Cache de avaliação
        self._eval_cache = {}
        
        # Controle de diversidade de perfis
        self._profile_counts = Counter()
    
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
    
    def evaluate(self, pool, is_elite=False):
        pool_hash = hashlib.md5(
            b''.join(struct.pack('>I', sum((1<<(d-1)) for d in g)) for g in pool)
        ).hexdigest()
        
        if pool_hash in self._eval_cache:
            return self._eval_cache[pool_hash].copy()
        
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
        
        obj[5] = 0.5  # Conectividade só na elite
        
        if len(self._eval_cache) < 500:
            self._eval_cache[pool_hash] = obj.copy()
        return obj
    
    def _initialize_population(self):
        """Inicialização DIVERSIFICADA com nichos forçados"""
        population = []
        
        # Estratégias com pesos DIFERENTES para forçar diversidade
        strategies_weighted = [
            ('spread', 60),      # Alta diversidade
            ('clustered', 50),   # Baixa diversidade
            ('balanced', 60),    # Médio
            ('entropy', 50),     # Alta entropia
            ('constrained', 80), # Com restrições (se houver)
        ]
        
        for strategy, weight in strategies_weighted:
            n_pools = max(1, self.pop_size * weight // sum(w for _, w in strategies_weighted))
            for _ in range(n_pools):
                if len(population) >= self.pop_size:
                    break
                pool = self._generate_pool(strategy)
                population.append(pool)
        
        # Completar se necessário
        while len(population) < self.pop_size:
            pool = self._generate_pool('random')
            population.append(pool)
        
        return population[:self.pop_size]
    
    def _generate_pool(self, strategy):
        pool = []
        seen = set()
        
        if strategy == 'constrained' and self.constraints:
            # Geração CONDICIONADA (pré-filtro)
            return generate_constrained_pool(self.n_games, self.constraints)
        
        if strategy == 'spread':
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base)
            seen.add(tuple(base))
            for _ in range(self.n_games - 1):
                best, best_min = None, 15
                for _ in range(200):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min:
                        best_min = m
                        best = c
                if best:
                    seen.add(tuple(best))
                    pool.append(best)
        
        elif strategy == 'clustered':
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(self.n_games):
                game = base.copy()
                for _ in range(np.random.randint(1, 4)):
                    pos = np.random.randint(0, 15)
                    avail = [d for d in range(1, 26) if d not in game]
                    if avail: game[pos] = np.random.choice(avail)
                game = sorted(game)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
        
        elif strategy == 'balanced':
            half = self.n_games // 2
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base)
            seen.add(tuple(base))
            for _ in range(half - 1):
                best, best_min = None, 15
                for _ in range(100):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min:
                        best_min = m
                        best = c
                if best:
                    seen.add(tuple(best))
                    pool.append(best)
            cbase = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(self.n_games - half):
                game = cbase.copy()
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail: game[pos] = np.random.choice(avail)
                game = sorted(game)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
        
        elif strategy == 'entropy':
            for _ in range(self.n_games):
                game = []
                for pos in range(15):
                    low = max(1, int(pos * 1.5))
                    high = min(25, int(25 - (14 - pos) * 1.5))
                    available = [d for d in range(low, high+1) if d not in game]
                    if available:
                        game.append(np.random.choice(available))
                    else:
                        avail_all = [d for d in range(1, 26) if d not in game]
                        game.append(np.random.choice(avail_all) if avail_all else 1)
                game = sorted(set(game))
                while len(game) < 15:
                    avail = [d for d in range(1, 26) if d not in game]
                    game.append(np.random.choice(avail) if avail else 1)
                if tuple(game[:15]) not in seen:
                    seen.add(tuple(game[:15]))
                    pool.append(game[:15])
        
        else:  # random
            for _ in range(self.n_games):
                game = tuple(sorted(np.random.choice(range(1, 26), 15, replace=False)))
                if game not in seen:
                    seen.add(game)
                    pool.append(list(game))
        
        # Garantir unicidade
        return self._ensure_unique(pool)
    
    def _ensure_unique(self, pool):
        unique = []
        seen = set()
        for game in pool:
            key = tuple(sorted(game))
            if key not in seen:
                seen.add(key)
                unique.append(sorted(game))
        
        # Completar
        attempts = 0
        while len(unique) < self.n_games and attempts < 500:
            attempts += 1
            if self.constraints:
                game = generate_constrained_game(self.constraints)
            else:
                game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                unique.append(game)
        
        while len(unique) < self.n_games:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                unique.append(game)
        
        return unique[:self.n_games]
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"🧬 NSGA-II BALANCEADO")
        print(f"{'='*60}")
        print(f"   Pop: {self.pop_size} | Gen: {self.n_generations} | Jogos: {self.n_games}")
        if self.constraints:
            print(f"   🎯 Geração CONDICIONADA ativa")
        
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
        
        # Coletar múltiplas fronts
        final_fronts = self._non_dominated_sort(population_obj)
        pareto_idx = []
        for front in final_fronts[:self.n_fronts_to_keep]:
            pareto_idx.extend(front)
        
        pareto_pools = [population[i] for i in pareto_idx]
        pareto_obj = [population_obj[i] for i in pareto_idx]
        
        print(f"\n   ✅ Fronteira: {len(pareto_pools)} soluções")
        return pareto_pools, pareto_obj
    
    def _crossover(self, p1, p2):
        if np.random.random() < 0.5:
            return p1[:self.n_games//2] + p2[self.n_games//2:]
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
        return child
    
    def _mutate(self, pool, generation):
        rate = 0.15 * (1 - generation / self.n_generations)
        mutated = [g.copy() for g in pool]
        n_mut = max(1, int(self.n_games * rate))
        for idx in np.random.choice(self.n_games, n_mut, replace=False):
            game = mutated[idx]
            if self.constraints:
                # Mutação respeitando restrições
                pos = np.random.randint(0, 15)
                fixed = set(self.constraints.get('fixas', []))
                excluded = set(self.constraints.get('excluidas', []))
                available = [d for d in range(1, 26) if d not in game and d not in excluded]
                if pos < len(game) and game[pos] not in fixed and available:
                    game[pos] = np.random.choice(available)
            else:
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail:
                    game[pos] = np.random.choice(avail)
            mutated[idx] = sorted(game)
        return mutated
    
    def _non_dominated_sort(self, pop_obj):
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
                distances[0] = distances[-1] = float('inf')
                for i in range(1, n-1):
                    distances[i] += (pop_obj[sorted_i[i+1]][obj_i] - pop_obj[sorted_i[i-1]][obj_i]) / obj_range
        return distances.tolist()


# ============================================================
# ANÁLISE DE SINAIS
# ============================================================

def compute_all_signals(pareto_pools):
    johnson = JohnsonSpace()
    johnson.pre_generate_cover_samples(2000)
    signals_list = []
    for pool in tqdm(pareto_pools, desc="Sinais"):
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        signals_list.append({
            'pair_coverage': float(len(covered) / comb(25, 2)),
            'johnson_min': float(johnson.min_johnson_distance(pool) / johnson.max_distance),
            'johnson_avg': float(johnson.avg_johnson_distance(pool) / johnson.max_distance),
            'covering_radius': float(johnson.covering_radius_fast(pool)),
            'sphere_packing': float(johnson.sphere_packing_bound(pool)),
            'pos_entropy': float(positional_entropy(pool)),
            'mutual_info': float(mutual_information_positions(pool)),
            'compressibility': float(BinarySerializer.compressibility(pool)),
        })
    return signals_list


def export_pareto_frontier(pareto_pools, pareto_obj, signals_list, filename='pareto_frontier.json'):
    print(f"\n💾 Exportando {filename}...")
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, list): return [convert(x) for x in obj]
        return obj
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_solutions': len(pareto_pools),
                'n_games_per_pool': len(pareto_pools[0]) if pareto_pools else 0,
            },
            'pareto_pools': [[sorted(g) for g in pool] for pool in pareto_pools],
            'pareto_objectives': [obj.tolist() for obj in pareto_obj],
            'signals': signals_list
        }, f, indent=2, ensure_ascii=False, default=convert)
    
    print(f"   ✅ {os.path.getsize(filename):,} bytes")
    return filename


def main():
    print("="*60)
    print("🧬 MOTOR CONDICIONADO v3.0")
    print("="*60)
    
    # Carregar histórico REAL
    historical = get_historical_data('resultados_lotofacil.csv', 100)
    last = get_last_contest('resultados_lotofacil.csv')
    
    if last:
        print(f"📌 Último concurso: {last['concurso']} ({last['data']})")
        print(f"   {last['dezenas']}")
    
    # Constraints (opcional - para geração condicionada)
    # Deixe vazio para gerar sem restrições
    constraints = None
    
    nsga2 = BalancedNSGA2(
        n_games=30, pop_size=300, n_generations=120,
        historical_data=historical,
        constraints=constraints
    )
    
    pareto_pools, pareto_obj = nsga2.run()
    signals_list = compute_all_signals(pareto_pools)
    export_pareto_frontier(pareto_pools, pareto_obj, signals_list)
    
    print(f"\n✅ PRONTO!")


if __name__ == "__main__":
    main()
