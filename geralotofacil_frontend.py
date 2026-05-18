#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOTOR PROBABILÍSTICO ADAPTATIVO - LOTOFÁCIL
============================================
Versão 7.0 - CSP + Modelagem Temporal + Otimização

ARQUITETURA:
✅ CAMADA 1: CSP (universo permitido)
✅ CAMADA 2: Modelagem Temporal (regime atual)
✅ CAMADA 3: Otimizador de Aderência

NOVAS CAPACIDADES:
✅ Frequência relativa local (momentum)
✅ Coocorrência dinâmica P(A|B)
✅ Vetor de assinatura de concursos
✅ Nearest neighbors históricos
✅ Heatmap temporal posicional
✅ Transição de regimes
✅ Mutation-only evolution (sem crossover)
✅ Repair determinístico orientado por violação
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
import struct
import zlib
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
# CARREGAMENTO DE DADOS
# ============================================================

def load_all_contests(csv_file='resultados_lotofacil.csv'):
    """Carrega TODOS os concursos ordenados"""
    if not os.path.exists(csv_file):
        return None
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
    except:
        return None


# ============================================================
# CAMADA 2: MODELAGEM TEMPORAL
# ============================================================

class TemporalModel:
    """
    Modelo temporal adaptativo
    
    Captura:
    - Frequência relativa local (momentum)
    - Coocorrência dinâmica
    - Assinatura estrutural de concursos
    - Regimes históricos similares
    - Heatmap posicional
    """
    
    def __init__(self, all_contests):
        self.contests = all_contests
        self.n_contests = len(all_contests)
        
        # Pré-computar features de todos os concursos
        self.feature_vectors = self._compute_all_features()
        
        # Últimos concursos para análise local
        self.last_contest = all_contests[-1] if all_contests else None
    
    def _extract_features(self, dezenas):
        """Extrai vetor de features estruturais de um concurso"""
        d = sorted(dezenas)
        return np.array([
            sum(d),                                    # soma
            sum(1 for x in d if x % 2 == 0),          # pares
            sum(1 for x in d if x in PRIMES),         # primos
            sum(1 for x in d if x in MOLDURA),        # moldura
            sum(1 for x in d if x in CENTRO),         # centro
            max(d) - min(d),                           # amplitude
            sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),  # consecutivos
            len(set((x-1)//5 for x in d)),            # quadrantes cobertos
            np.mean([d[i+1]-d[i] for i in range(14)]), # distância média
            np.std([d[i+1]-d[i] for i in range(14)]) if len(d)>1 else 0,  # std distância
        ])
    
    def _compute_all_features(self):
        """Pré-computa features de todos os concursos"""
        return np.array([self._extract_features(c['dezenas']) for c in self.contests])
    
    def get_momentum(self, windows=[5, 10, 20, 50]):
        """
        Calcula momentum (frequência relativa local)
        
        Para cada dezena:
        momentum = freq_curta - freq_longa
        
        Positivo = acelerando (esquentando)
        Negativo = desacelerando (esfriando)
        """
        if self.n_contests < max(windows):
            return {}
        
        momentum = {}
        last_n = {w: self.contests[-w:] for w in windows}
        
        for dezena in range(1, 26):
            freqs = {}
            for w, contests in last_n.items():
                count = sum(1 for c in contests if dezena in c['dezenas'])
                freqs[w] = count / w
            
            # Momentum: diferença entre janela curta e longa
            if 5 in freqs and 50 in freqs:
                momentum[dezena] = freqs[5] - freqs[50]
            else:
                momentum[dezena] = 0
        
        return momentum
    
    def get_cooccurrence_matrix(self, window=20):
        """
        Matriz de coocorrência dinâmica P(A|B)
        
        Dado que B saiu, qual a probabilidade de A sair junto?
        (baseado nos últimos N concursos)
        """
        recent = self.contests[-window:]
        n = len(recent)
        
        # Contagem de coocorrência
        cooccur = np.zeros((26, 26))
        single = np.zeros(26)
        
        for c in recent:
            dezenas = c['dezenas']
            for d in dezenas:
                single[d] += 1
            for d1 in dezenas:
                for d2 in dezenas:
                    if d1 != d2:
                        cooccur[d1, d2] += 1
        
        # P(A|B) = P(A,B) / P(B)
        prob_given = np.zeros((26, 26))
        for b in range(1, 26):
            if single[b] > 0:
                for a in range(1, 26):
                    if a != b:
                        prob_given[a, b] = cooccur[a, b] / single[b]
        
        return prob_given
    
    def get_signature(self, n_recent=10):
        """
        Vetor de assinatura dos últimos N concursos
        
        Agrega features dos concursos recentes
        """
        if self.n_contests < n_recent:
            return None
        
        recent_features = self.feature_vectors[-n_recent:]
        return np.mean(recent_features, axis=0)
    
    def find_similar_regimes(self, n_recent=10, top_k=20):
        """
        Encontra regimes históricos similares ao atual
        
        Compara assinatura atual com todas as janelas históricas
        """
        if self.n_contests < n_recent + 1:
            return []
        
        # Assinatura atual
        current_sig = self.get_signature(n_recent)
        
        # Deslizar janela pelo histórico
        similarities = []
        for i in range(n_recent, self.n_contests - 1):
            window_features = self.feature_vectors[i-n_recent:i]
            hist_sig = np.mean(window_features, axis=0)
            
            # Similaridade (distância euclidiana negativa)
            dist = np.linalg.norm(current_sig - hist_sig)
            similarities.append((dist, i))
        
        # Ordenar por similaridade (menor distância = mais similar)
        similarities.sort(key=lambda x: x[0])
        
        # Retornar os concursos SEGUINTES aos regimes similares
        next_contests = []
        for dist, idx in similarities[:top_k]:
            if idx + 1 < self.n_contests:
                next_contests.append(self.contests[idx + 1])
        
        return next_contests
    
    def get_positional_heatmap(self, window=30):
        """
        Heatmap temporal posicional
        
        Para cada posição (1ª a 15ª dezena), qual a distribuição atual?
        """
        recent = self.contests[-window:]
        
        heatmap = np.zeros((15, 25))
        for c in recent:
            sorted_d = sorted(c['dezenas'])
            for pos, d in enumerate(sorted_d):
                heatmap[pos, d-1] += 1
        
        # Normalizar por linha
        for pos in range(15):
            row_sum = heatmap[pos].sum()
            if row_sum > 0:
                heatmap[pos] /= row_sum
        
        return heatmap
    
    def score_game_temporal(self, game):
        """
        Score temporal de um jogo
        
        Combina:
        - Momentum das dezenas
        - Coocorrência com último concurso
        - Aderência ao heatmap posicional
        - Similaridade com regimes históricos
        """
        if self.last_contest is None:
            return 0.5
        
        game = set(game)
        last = set(self.last_contest['dezenas'])
        score = 0.0
        
        # 1. Momentum (dezenas esquentando = melhor)
        momentum = self.get_momentum()
        if momentum:
            mom_scores = [momentum.get(d, 0) for d in game]
            score += np.mean(mom_scores) * 0.3 + 0.5  # Normalizar
        
        # 2. Coocorrência com último concurso
        cooccur = self.get_cooccurrence_matrix(window=30)
        if self.last_contest:
            cooc_scores = []
            for d in game:
                if d not in last:
                    # Probabilidade de aparecer dado que as dezenas do último saíram
                    probs = [cooccur[d, ld] for ld in last if ld <= 25]
                    if probs:
                        cooc_scores.append(np.mean(probs))
            if cooc_scores:
                score += np.mean(cooc_scores) * 0.3
        
        # 3. Aderência posicional
        heatmap = self.get_positional_heatmap(window=50)
        if heatmap is not None:
            sorted_game = sorted(game)
            pos_scores = []
            for pos, d in enumerate(sorted_game[:15]):
                if d <= 25:
                    pos_scores.append(heatmap[min(pos, 14), d-1])
            if pos_scores:
                score += np.mean(pos_scores) * 0.2
        
        # 4. Repetidas do último (bônus)
        rep = len(game & last)
        score += (rep / 15) * 0.2
        
        return min(1.0, max(0.0, score))


# ============================================================
# CAMADA 1: CSP (UNIVERSO PERMITIDO)
# ============================================================

def validate_constraint_feasibility(constraints):
    """Verifica factibilidade combinatória"""
    if constraints is None:
        return True, "OK"
    
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    if fixed & excluded:
        return False, f"Conflito: {fixed & excluded}"
    if len(fixed) > 15:
        return False, f"Fixas > 15"
    
    available = set(range(1, 26)) - excluded - fixed
    slots = 15 - len(fixed)
    
    if len(available) < slots:
        return False, f"Espaço insuficiente"
    
    return True, "✅"


def generate_feasible_game(constraints, max_attempts=100):
    """Gera jogo viável dentro das constraints"""
    if constraints is None:
        return sorted(np.random.choice(range(1, 26), 15, replace=False).tolist())
    
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    for _ in range(max_attempts):
        game = set(fixed)
        available = list(set(range(1, 26)) - excluded - game)
        
        # Completar aleatoriamente
        needed = 15 - len(game)
        if needed > 0 and len(available) >= needed:
            chosen = np.random.choice(available, needed, replace=False)
            game.update(chosen)
        
        if len(game) == 15:
            # Validar targets exatos
            valid = True
            if 'pares_target' in constraints:
                if sum(1 for d in game if d % 2 == 0) != constraints['pares_target']:
                    valid = False
            if 'primos_target' in constraints:
                if sum(1 for d in game if d in PRIMES) != constraints['primos_target']:
                    valid = False
            if 'moldura_target' in constraints:
                if sum(1 for d in game if d in MOLDURA) != constraints['moldura_target']:
                    valid = False
            if 'repetidas_target' in constraints and 'ultimo_concurso' in constraints:
                if len(game & set(constraints['ultimo_concurso'])) != constraints['repetidas_target']:
                    valid = False
            if 'max_consecutivos' in constraints:
                sg = sorted(game)
                cons = 1; mf = 1
                for i in range(len(sg)-1):
                    if sg[i+1]-sg[i]==1: cons+=1; mf=max(mf,cons)
                    else: cons=1
                if mf > constraints['max_consecutivos']:
                    valid = False
            
            if valid:
                return sorted([int(x) for x in game])
    
    # Fallback: retornar com fixas e excluídas apenas
    game = set(fixed)
    available = list(set(range(1, 26)) - excluded - game)
    needed = 15 - len(game)
    if needed > 0 and len(available) >= needed:
        game.update(np.random.choice(available, needed, replace=False))
    return sorted([int(x) for x in game])


def repair_exact(game, constraints):
    """
    REPAIR DETERMINÍSTICO orientado por violação
    
    Corrige constraints EXATAS uma por uma
    """
    if constraints is None:
        return list(game)
    
    game = set(int(x) for x in game)
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    # Garantir fixas e excluídas
    game |= fixed
    game -= excluded
    
    # Corrigir tamanho
    while len(game) > 15:
        removable = [d for d in game if d not in fixed]
        if not removable: break
        game.remove(removable[0])
    
    available = set(range(1, 26)) - game - excluded
    
    # Corrigir pares exatos
    target_pares = constraints.get('pares_target')
    if target_pares is not None:
        current = sum(1 for d in game if d % 2 == 0)
        while current < target_pares and available:
            even_avail = [d for d in available if d % 2 == 0]
            odd_in_game = [d for d in game if d % 2 != 0 and d not in fixed]
            if even_avail and odd_in_game:
                game.remove(odd_in_game[0])
                game.add(even_avail[0])
                available = set(range(1, 26)) - game - excluded
            else:
                break
            current = sum(1 for d in game if d % 2 == 0)
        
        while current > target_pares:
            odd_avail = [d for d in available if d % 2 != 0]
            even_in_game = [d for d in game if d % 2 == 0 and d not in fixed]
            if odd_avail and even_in_game:
                game.remove(even_in_game[0])
                game.add(odd_avail[0])
                available = set(range(1, 26)) - game - excluded
            else:
                break
            current = sum(1 for d in game if d % 2 == 0)
    
    # Corrigir primos exatos
    target_primos = constraints.get('primos_target')
    if target_primos is not None:
        current = sum(1 for d in game if d in PRIMES)
        while current < target_primos and available:
            prime_avail = [d for d in available if d in PRIMES]
            nonprime_in_game = [d for d in game if d not in PRIMES and d not in fixed]
            if prime_avail and nonprime_in_game:
                game.remove(nonprime_in_game[0])
                game.add(prime_avail[0])
                available = set(range(1, 26)) - game - excluded
            else:
                break
            current = sum(1 for d in game if d in PRIMES)
    
    # Corrigir moldura exata
    target_moldura = constraints.get('moldura_target')
    if target_moldura is not None:
        current = sum(1 for d in game if d in MOLDURA)
        while current < target_moldura and available:
            mold_avail = [d for d in available if d in MOLDURA]
            nonmold_in_game = [d for d in game if d not in MOLDURA and d not in fixed]
            if mold_avail and nonmold_in_game:
                game.remove(nonmold_in_game[0])
                game.add(mold_avail[0])
                available = set(range(1, 26)) - game - excluded
            else:
                break
            current = sum(1 for d in game if d in MOLDURA)
    
    # Corrigir repetidas
    target_rep = constraints.get('repetidas_target')
    ultimo = set(constraints.get('ultimo_concurso', []))
    if target_rep is not None and ultimo:
        current = len(game & ultimo)
        while current < target_rep and available:
            rep_avail = [d for d in available if d in ultimo]
            nonrep_in_game = [d for d in game if d not in ultimo and d not in fixed]
            if rep_avail and nonrep_in_game:
                game.remove(nonrep_in_game[0])
                game.add(rep_avail[0])
                available = set(range(1, 26)) - game - excluded
            else:
                break
            current = len(game & ultimo)
    
    # Garantir 15
    while len(game) < 15 and available:
        game.add(list(available)[0])
        available = set(range(1, 26)) - game - excluded
    
    while len(game) > 15:
        removable = [d for d in game if d not in fixed]
        if not removable: break
        game.remove(removable[0])
    
    return sorted([int(x) for x in game])[:15]


# ============================================================
# CAMADA 3: OTIMIZADOR (MUTATION-ONLY)
# ============================================================

class TemporalOptimizer:
    """
    Otimizador de aderência temporal
    
    Usa MUTATION-ONLY (sem crossover que destrói constraints)
    """
    
    def __init__(self, temporal_model, constraints=None, n_games=30, pop_size=100, n_generations=50):
        self.model = temporal_model
        self.constraints = constraints
        self.n_games = n_games
        self.pop_size = pop_size
        self.n_generations = n_generations
    
    def _fitness(self, pool):
        """Fitness combinado: temporal + estrutural"""
        # Score temporal médio
        temporal_scores = [self.model.score_game_temporal(g) for g in pool]
        avg_temporal = np.mean(temporal_scores)
        
        # Score estrutural (diversidade)
        distances = []
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                common = len(set(pool[i]) & set(pool[j]))
                distances.append(15 - common)
        avg_distance = np.mean(distances) if distances else 0
        
        # Fitness combinado
        return avg_temporal * 0.7 + (avg_distance / 10) * 0.3
    
    def _mutate_pool(self, pool):
        """Mutação de pool preservando constraints"""
        mutated = [list(g) for g in pool]
        
        # Mutar alguns jogos
        n_mut = max(1, self.n_games // 5)
        for idx in np.random.choice(self.n_games, n_mut, replace=False):
            # Gerar novo jogo viável
            new_game = generate_feasible_game(self.constraints)
            mutated[idx] = new_game
        
        # Repair em todos
        mutated = [repair_exact(g, self.constraints) for g in mutated]
        return mutated
    
    def optimize(self):
        """Otimização mutation-only"""
        print(f"\n{'='*60}")
        print(f"🎯 OTIMIZADOR TEMPORAL (MUTATION-ONLY)")
        print(f"{'='*60}")
        
        # População inicial (todos viáveis)
        population = []
        for _ in range(self.pop_size):
            pool = []
            seen = set()
            for _ in range(self.n_games):
                game = generate_feasible_game(self.constraints)
                key = tuple(game)
                if key not in seen:
                    seen.add(key)
                    pool.append(game)
            while len(pool) < self.n_games:
                game = generate_feasible_game(self.constraints)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
            population.append(pool[:self.n_games])
        
        # Avaliar
        fitnesses = [self._fitness(p) for p in population]
        best_idx = np.argmax(fitnesses)
        best_pool = population[best_idx]
        best_fitness = fitnesses[best_idx]
        
        # Evolução mutation-only
        for gen in tqdm(range(self.n_generations), desc="Otimizando"):
            # Gerar mutantes
            offspring = [self._mutate_pool(p) for p in population]
            offspring_fitness = [self._fitness(o) for o in offspring]
            
            # Selecionar melhores (elitismo)
            combined = list(zip(population + offspring, fitnesses + offspring_fitness))
            combined.sort(key=lambda x: x[1], reverse=True)
            
            population = [c[0] for c in combined[:self.pop_size]]
            fitnesses = [c[1] for c in combined[:self.pop_size]]
            
            if fitnesses[0] > best_fitness:
                best_fitness = fitnesses[0]
                best_pool = population[0]
        
        print(f"\n   ✅ Melhor fitness: {best_fitness:.4f}")
        return best_pool


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def collect_preferences(ultimo=None):
    print(f"\n{'='*60}")
    print(f"🎯 CONFIGURAÇÃO DE PREFERÊNCIAS")
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


def display_results(pool, constraints, model, ultimo=None):
    print(f"\n{'='*60}")
    print(f"🏆 RESULTADOS")
    print(f"{'='*60}")
    
    if model.last_contest:
        print(f"📌 Último concurso: {model.last_contest['concurso']} - {model.last_contest['dezenas']}")
    
    # Momentum
    momentum = model.get_momentum()
    if momentum:
        top_momentum = sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🔥 Esquentando: {[(d, f'{s:.3f}') for d, s in top_momentum]}")
    
    # Jogos
    for i, game in enumerate(pool[:10], 1):
        game = [int(x) for x in game]
        p = sum(1 for d in game if d%2==0)
        pr = sum(1 for d in game if d in PRIMES)
        m = sum(1 for d in game if d in MOLDURA)
        s = sum(game)
        r = len(set(game)&set(ultimo)) if ultimo else 0
        temp = model.score_game_temporal(game)
        
        print(f"   {i:2d}. {game}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{s} R:{r} | Temp:{temp:.3f}")


def main():
    print("="*60)
    print("🧬 MOTOR PROBABILÍSTICO ADAPTATIVO")
    print("="*60)
    
    # Carregar dados
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    
    print(f"📂 {len(contests)} concursos carregados")
    
    # Modelo temporal
    model = TemporalModel(contests)
    ultimo = model.last_contest['dezenas'] if model.last_contest else None
    
    # Coletar preferências
    prefs = collect_preferences(ultimo)
    
    # Validar factibilidade
    feasible, msg = validate_constraint_feasibility(prefs)
    print(f"\n🔍 Factibilidade: {msg}")
    
    if not feasible:
        print("⚠️  Relaxando constraints...")
        if prefs:
            prefs = {
                'fixas': prefs.get('fixas', []),
                'excluidas': prefs.get('excluidas', []),
                'ultimo_concurso': prefs.get('ultimo_concurso', []),
            }
    
    # Otimizar
    opt = TemporalOptimizer(model, prefs, n_games=30, pop_size=100, n_generations=50)
    best_pool = opt.optimize()
    
    # Exibir
    display_results(best_pool, prefs, model, ultimo)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
