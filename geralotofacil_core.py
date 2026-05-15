#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOTOR DE OTIMIZAÇÃO COMBINATÓRIA - LOTOFÁCIL
=============================================
Versão 3.0 - Arquitetura Modular Importável

Este módulo contém APENAS o motor de otimização.
O front-end interativo está em geralotofacil_frontend.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
import warnings
import os
from datetime import datetime
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

# ============================================================
# CONSTANTES DA LOTOFÁCIL (exportáveis)
# ============================================================

TOTAL_NUMBERS = 25
NUMBERS_PER_GAME = 15
MIN_PRIZE = 11

# Conjuntos matemáticos
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

# Moldura da Lotofácil (bordas do volante 5x5)
MOLDURA = {
    1, 2, 3, 4, 5,      # linha 1
    6, 10,               # linha 2 bordas
    11, 15,              # linha 3 bordas
    16, 20,              # linha 4 bordas
    21, 22, 23, 24, 25   # linha 5
}

# Centro do volante
CENTRO = {7, 8, 9, 12, 13, 14, 17, 18, 19}

# Quadrantes
QUADRANTES = {
    'Q1': {1, 2, 3, 4, 5},
    'Q2': {6, 7, 8, 9, 10},
    'Q3': {11, 12, 13, 14, 15},
    'Q4': {16, 17, 18, 19, 20},
    'Q5': {21, 22, 23, 24, 25}
}


class LotofacilOptimizerV3:
    """
    Motor de Otimização Lotofácil - Versão 3.0 Modular
    
    Características:
    - Totalmente importável
    - Métodos públicos bem definidos
    - Constantes exportáveis
    - Fitness multiobjetivo calibrado
    - Geração de candidatos eficiente
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """
        Inicializa o motor de otimização
        
        Args:
            historical_csv: Caminho para arquivo CSV histórico
        """
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Simulações pré-geradas para avaliação consistente
        self.reference_draws = None
        self._pre_generate_simulations(5000)
        
        # Constraints calibradas
        self.constraints = self._default_constraints()
        
        # Carregar dados
        self._load_historical_data()
        
        # Cache para fitness
        self._fitness_cache = {}
        
    def _default_constraints(self):
        """Constraints padrão calibradas"""
        return {
            'max_consecutive_block': 5,
            'consecutive_weight': 2.0,
            'max_per_line': 5,
            'line_weight': 3.0,
            'max_per_column': 5,
            'column_weight': 3.0,
            'max_pair_reuse': 4,
            'pair_redundancy_weight': 2.0,
            'target_overlap': 8,
            'overlap_weight': 2.0,
            'acceptance_threshold': 35,
        }
    
    def _pre_generate_simulations(self, n_draws=5000):
        """Pré-gera simulações de referência"""
        self.reference_draws = [
            set(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(n_draws)
        ]
    
    def _load_historical_data(self):
        """Carrega dados históricos"""
        try:
            self.df = pd.read_csv(self.historical_csv, sep=';', encoding='utf-8')
            bola_cols = [f'b{i}' for i in range(1, 16)]
            self.df.columns = ['concurso', 'data'] + bola_cols
            self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
            self.dezenas_historicas = self.df[bola_cols].values
        except FileNotFoundError:
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_games=3000):
        """Gera dados sintéticos se não houver arquivo"""
        synthetic_games = []
        for _ in range(n_games):
            weights = np.ones(25) + np.random.normal(0, 0.05, 25)
            weights = np.abs(weights) / np.sum(np.abs(weights))
            game = sorted(np.random.choice(range(1, 26), 15, replace=False, p=weights))
            synthetic_games.append(game)
        
        self.df = pd.DataFrame({
            'concurso': range(1, n_games + 1),
            'data': pd.date_range(start='2003-09-29', periods=n_games, freq='3D')
        })
        
        for i in range(15):
            self.df[f'b{i+1}'] = [g[i] for g in synthetic_games]
        
        bola_cols = [f'b{i}' for i in range(1, 16)]
        self.dezenas_historicas = self.df[bola_cols].values
    
    # ============================================================
    # MÉTODOS PÚBLICOS PRINCIPAIS
    # ============================================================
    
    def get_last_draw(self):
        """
        Retorna o último concurso como lista de inteiros
        
        Returns:
            list: 15 dezenas do último concurso
        """
        if self.dezenas_historicas is not None and len(self.dezenas_historicas) > 0:
            return [int(x) for x in self.dezenas_historicas[-1]]
        return None
    
    def get_last_draws(self, n=10):
        """
        Retorna os últimos n concursos
        
        Args:
            n: Número de concursos
            
        Returns:
            list: Lista dos últimos n concursos
        """
        if self.dezenas_historicas is not None and len(self.dezenas_historicas) >= n:
            return [[int(x) for x in draw] for draw in self.dezenas_historicas[-n:]]
        return None
    
    def get_historical_frequency(self):
        """
        Retorna frequência histórica das dezenas
        
        Returns:
            dict: Frequência de cada dezena (1-25)
        """
        if self.dezenas_historicas is None:
            return {}
        
        freq = np.bincount(self.dezenas_historicas.flatten(), minlength=26)[1:]
        return {i+1: int(f) for i, f in enumerate(freq)}
    
    def calculate_structural_penalty(self, game):
        """
        Calcula penalidade estrutural de um jogo
        
        Args:
            game: Lista de 15 dezenas
            
        Returns:
            float: Penalidade estrutural (menor = melhor)
        """
        game_set = set(game)
        penalty = 0.0
        c = self.constraints
        
        # Blocos consecutivos
        d = sorted(game)
        current_block = 1
        max_block = 1
        
        for i in range(len(d) - 1):
            if d[i+1] - d[i] == 1:
                current_block += 1
                max_block = max(max_block, current_block)
            else:
                current_block = 1
        
        if max_block > c['max_consecutive_block']:
            penalty += (max_block - c['max_consecutive_block']) * c['consecutive_weight']
        
        # Linhas do volante
        for row_start in [1, 6, 11, 16, 21]:
            row_numbers = set(range(row_start, row_start + 5))
            overlap = len(game_set & row_numbers)
            if overlap > c['max_per_line']:
                penalty += (overlap - c['max_per_line']) * c['line_weight']
        
        # Colunas do volante
        for col_start in range(1, 6):
            col_numbers = set(range(col_start, 26, 5))
            overlap = len(game_set & col_numbers)
            if overlap > c['max_per_column']:
                penalty += (overlap - c['max_per_column']) * c['column_weight']
        
        return penalty
    
    def generate_candidates(self, n_candidates=5000, respect_constraints=True):
        """
        Gera candidatos diversos
        
        Args:
            n_candidates: Número de candidatos a gerar
            respect_constraints: Se True, filtra por penalidade estrutural
            
        Returns:
            list: Lista de jogos candidatos
        """
        candidates = []
        seen = set()
        attempts = 0
        max_attempts = n_candidates * 20
        
        while len(candidates) < n_candidates and attempts < max_attempts:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            game_tuple = tuple(game)
            
            if game_tuple not in seen:
                if respect_constraints:
                    penalty = self.calculate_structural_penalty(game)
                    if penalty < self.constraints['acceptance_threshold']:
                        seen.add(game_tuple)
                        candidates.append(game)
                else:
                    seen.add(game_tuple)
                    candidates.append(game)
            
            attempts += 1
        
        return candidates
    
    def compute_game_fitness(self, game):
        """
        Calcula fitness de um jogo individual
        
        Args:
            game: Lista de 15 dezenas
            
        Returns:
            tuple: (fitness_total, dict_metricas)
        """
        # Verificar cache
        game_key = tuple(sorted(game))
        if game_key in self._fitness_cache:
            return self._fitness_cache[game_key]
        
        # Penalidade estrutural (invertida: menor penalidade = maior score)
        penalty = self.calculate_structural_penalty(game)
        structural_score = max(0, 25 - penalty)
        
        # Diversidade de pares (evitar padrões)
        pair_counts = Counter()
        for pair in combinations(sorted(game), 2):
            pair_counts[pair] += 1
        
        # Entropia local
        dezena_counts = np.bincount(game, minlength=26)[1:]
        dezena_probs = dezena_counts / np.sum(dezena_counts)
        dezena_probs = np.where(dezena_probs > 0, dezena_probs, 1e-10)
        local_entropy = entropy(dezena_probs)
        entropy_score = (local_entropy / np.log(15)) * 10
        
        # Cobertura de quadrantes
        quadrant_coverage = 0
        for q_name, q_numbers in QUADRANTES.items():
            if len(set(game) & q_numbers) >= 2:
                quadrant_coverage += 1
        quadrant_score = (quadrant_coverage / 5) * 15
        
        # Balanceamento par/ímpar
        pares = sum(1 for x in game if x % 2 == 0)
        balance_score = 10 - abs(pares - 7.5) * 2
        balance_score = max(0, balance_score)
        
        # Fitness total
        total_fitness = structural_score + entropy_score + quadrant_score + balance_score
        
        metrics = {
            'structural_score': structural_score,
            'entropy_score': entropy_score,
            'quadrant_score': quadrant_score,
            'balance_score': balance_score,
            'penalty': penalty,
            'pares': pares,
            'impares': 15 - pares
        }
        
        # Cache
        self._fitness_cache[game_key] = (total_fitness, metrics)
        
        return total_fitness, metrics
    
    def compute_pool_fitness(self, pool):
        """
        Calcula fitness de um pool de jogos
        
        Args:
            pool: Lista de jogos
            
        Returns:
            tuple: (fitness_total, dict_metricas)
        """
        n_games = len(pool)
        
        # Fitness médio dos jogos individuais
        individual_scores = []
        total_penalty = 0
        for game in pool:
            score, _ = self.compute_game_fitness(game)
            individual_scores.append(score)
            total_penalty += self.calculate_structural_penalty(game)
        
        avg_individual = np.mean(individual_scores) if individual_scores else 0
        
        # Diversidade do pool (sobreposição)
        overlaps = []
        for i in range(n_games):
            for j in range(i+1, n_games):
                common = len(set(pool[i]) & set(pool[j]))
                overlaps.append(common)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        
        # Penalidade por sobreposição
        overlap_deviation = abs(avg_overlap - self.constraints['target_overlap'])
        overlap_penalty = overlap_deviation * self.constraints['overlap_weight']
        overlap_score = max(0, 20 - overlap_penalty)
        
        # Redundância de pares
        pair_counter = Counter()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                pair_counter[pair] += 1
        
        if pair_counter:
            counts = list(pair_counter.values())
            avg_redundancy = np.mean(counts)
            overused = sum(1 for c in counts if c > self.constraints['max_pair_reuse'])
        else:
            avg_redundancy = 0
            overused = 0
        
        redundancy_penalty = max(0, (avg_redundancy - 1) * self.constraints['pair_redundancy_weight'])
        redundancy_penalty += overused * 3
        redundancy_score = max(0, 25 - redundancy_penalty)
        
        # Cobertura de premiação (proxy rápido)
        avg_11, avg_12, avg_13, avg_14 = self._fast_prize_proxy(pool)
        prize_score = min(20, avg_11 * 4 + avg_12 * 6 + avg_13 * 6 + avg_14 * 4)
        
        # Fitness total
        total_fitness = (
            avg_individual * 0.3 +
            overlap_score * 0.2 +
            redundancy_score * 0.25 +
            prize_score * 0.25
        )
        
        metrics = {
            'avg_individual_score': avg_individual,
            'overlap_score': overlap_score,
            'redundancy_score': redundancy_score,
            'prize_score': prize_score,
            'avg_overlap': avg_overlap,
            'avg_pair_redundancy': avg_redundancy,
            'overused_pairs': overused,
            'avg_11': avg_11,
            'avg_12': avg_12,
            'avg_13': avg_13,
            'avg_14': avg_14
        }
        
        return total_fitness, metrics
    
    def _fast_prize_proxy(self, pool):
        """Proxy rápido para cobertura de premiação"""
        if not self.reference_draws:
            return 0, 0, 0, 0
        
        n_eval = min(1000, len(self.reference_draws))
        eval_draws = self.reference_draws[:n_eval]
        
        total_11, total_12, total_13, total_14 = 0, 0, 0, 0
        
        for drawn in eval_draws:
            for game in pool:
                hits = len(set(game) & drawn)
                if hits == 11:
                    total_11 += 1
                elif hits == 12:
                    total_12 += 1
                elif hits == 13:
                    total_13 += 1
                elif hits >= 14:
                    total_14 += 1
        
        return total_11/n_eval, total_12/n_eval, total_13/n_eval, total_14/n_eval
    
    def rank_games(self, games, top_n=10):
        """
        Rankeia jogos por fitness
        
        Args:
            games: Lista de jogos
            top_n: Número de jogos para retornar
            
        Returns:
            list: Lista de tuplas (fitness, game, metrics) ordenadas
        """
        ranked = []
        for game in games:
            fitness, metrics = self.compute_game_fitness(game)
            ranked.append((fitness, game, metrics))
        
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:top_n]
    
    def optimize_pool(self, n_games=50, candidate_pool_size=2000, iterations=500):
        """
        Otimiza um pool de jogos
        
        Args:
            n_games: Número de jogos no pool
            candidate_pool_size: Tamanho do pool de candidatos
            iterations: Iterações do simulated annealing
            
        Returns:
            list: Pool otimizado de jogos
        """
        # Gerar candidatos
        candidates = self.generate_candidates(candidate_pool_size)
        
        # Selecionar subconjunto diverso
        if len(candidates) <= n_games:
            pool = candidates
        else:
            pool = [candidates[0]]
            remaining = candidates[1:]
            
            for _ in range(n_games - 1):
                if not remaining:
                    break
                
                best_idx = 0
                best_min_dist = -1
                sample_size = min(100, len(remaining))
                sample_indices = np.random.choice(len(remaining), sample_size, replace=False)
                
                for idx in sample_indices:
                    candidate = remaining[idx]
                    min_dist = min(
                        1 - len(set(candidate) & set(s)) / len(set(candidate) | set(s))
                        for s in pool
                    )
                    
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                
                pool.append(remaining[best_idx])
                remaining.pop(best_idx)
        
        # Simulated annealing
        current_pool = pool.copy()
        current_fitness, _ = self.compute_pool_fitness(current_pool)
        best_pool = [g.copy() for g in current_pool]
        best_fitness = current_fitness
        
        for iteration in range(iterations):
            temp = 5.0 * (0.05 / 5.0) ** (iteration / iterations)
            
            new_pool = [g.copy() for g in current_pool]
            idx = np.random.randint(0, len(new_pool))
            
            # Mutação
            game = new_pool[idx]
            pos = np.random.randint(0, 15)
            available = [d for d in range(1, 26) if d not in game]
            if available:
                game[pos] = np.random.choice(available)
                game.sort()
            
            new_fitness, _ = self.compute_pool_fitness(new_pool)
            
            delta = new_fitness - current_fitness
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_pool = new_pool
                current_fitness = new_fitness
                
                if current_fitness > best_fitness:
                    best_pool = [g.copy() for g in current_pool]
                    best_fitness = current_fitness
        
        return best_pool


# ============================================================
# FUNÇÕES UTILITÁRIAS (exportáveis)
# ============================================================

def contar_pares(game):
    """Conta números pares no jogo"""
    return sum(1 for x in game if x % 2 == 0)

def contar_impares(game):
    """Conta números ímpares no jogo"""
    return sum(1 for x in game if x % 2 != 0)

def contar_primos(game):
    """Conta números primos no jogo"""
    return sum(1 for x in game if x in PRIMES)

def contar_moldura(game):
    """Conta números na moldura do volante"""
    return sum(1 for x in game if x in MOLDURA)

def contar_centro(game):
    """Conta números no centro do volante"""
    return sum(1 for x in game if x in CENTRO)

def contar_repetidos(game, reference):
    """Conta números repetidos em relação a uma referência"""
    return len(set(game) & set(reference))

def contar_quadrante(game, quadrante):
    """Conta números em um quadrante específico"""
    return sum(1 for x in game if x in quadrante)

def calcular_soma(game):
    """Calcula soma das dezenas"""
    return sum(game)

def calcular_amplitude(game):
    """Calcula amplitude (max - min)"""
    return max(game) - min(game)

def contar_consecutivos(game):
    """Conta pares consecutivos"""
    d = sorted(game)
    return sum(1 for i in range(len(d)-1) if d[i+1] - d[i] == 1)


# ============================================================
# EXECUÇÃO PRINCIPAL (apenas quando rodado diretamente)
# ============================================================

def main():
    """Execução principal do motor"""
    print("="*70)
    print("🎯 MOTOR DE OTIMIZAÇÃO LOTOFÁCIL v3.0")
    print("   Este é o motor principal. Use o front-end para interação.")
    print("="*70)
    
    optimizer = LotofacilOptimizerV3()
    
    # Demonstração rápida
    print("\n📊 Demonstração de geração de candidatos...")
    candidates = optimizer.generate_candidates(100)
    print(f"   ✅ {len(candidates)} candidatos gerados")
    
    print("\n📊 Demonstração de ranking...")
    ranked = optimizer.rank_games(candidates, 5)
    for i, (fitness, game, metrics) in enumerate(ranked, 1):
        print(f"   {i}. Fitness={fitness:.2f} | Pares={metrics['pares']} | Penalidade={metrics['penalty']:.1f}")
        print(f"      {sorted(game)}")
    
    print("\n✅ Motor funcionando corretamente!")


if __name__ == "__main__":
    main()
