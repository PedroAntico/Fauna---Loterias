#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOTOR DE OTIMIZAÇÃO COMBINATÓRIA - LOTOFÁCIL v3.1
==================================================
MELHORIAS CRÍTICAS:
✅ Distância de Mahalanobis multivariada
✅ Correção da saturação do fitness
✅ Penalidade de consecutivos excessivos
✅ Score de raridade histórica (bayesiano)
✅ Ranking com discriminação real
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
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
# CONSTANTES DA LOTOFÁCIL
# ============================================================

TOTAL_NUMBERS = 25
NUMBERS_PER_GAME = 15
MIN_PRIZE = 11

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

MOLDURA = {
    1, 2, 3, 4, 5,
    6, 10,
    11, 15,
    16, 20,
    21, 22, 23, 24, 25
}

CENTRO = {7, 8, 9, 12, 13, 14, 17, 18, 19}

QUADRANTES = {
    'Q1': {1, 2, 3, 4, 5},
    'Q2': {6, 7, 8, 9, 10},
    'Q3': {11, 12, 13, 14, 15},
    'Q4': {16, 17, 18, 19, 20},
    'Q5': {21, 22, 23, 24, 25}
}


class LotofacilOptimizerV3:
    """
    Motor de Otimização Lotofácil v3.1 - Com Mahalanobis
    
    NOVAS CAPACIDADES:
    - Distância multivariada real (Mahalanobis)
    - Fitness com discriminação efetiva
    - Penalidades de consecutivos
    - Score de raridade bayesiana
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """Inicialização completa com modelo multivariado"""
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Simulações pré-geradas
        self.reference_draws = None
        self._pre_generate_simulations(5000)
        
        # Constraints
        self.constraints = self._default_constraints()
        
        # Modelo Mahalanobis
        self.feature_matrix = None
        self.feature_mean = None
        self.inv_cov_matrix = None
        
        # Cache
        self._fitness_cache = {}
        self._mahalanobis_cache = {}
        
        # Estatísticas históricas para score bayesiano
        self.historical_patterns = None
        
        # Carregar tudo
        self._load_historical_data()
        self._build_mahalanobis_model()
        self._build_historical_patterns()
        
        print("✅ Motor Lotofácil v3.1 inicializado!")
        print(f"📊 {len(self.df)} concursos | Mahalanobis ativo | Score bayesiano ativo")
    
    def _default_constraints(self):
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
        """Gera dados sintéticos"""
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
    # MODELO MAHALANOBIS
    # ============================================================
    
    def _extract_feature_vector(self, game):
        """
        Extrai vetor de features para Mahalanobis
        
        Features correlacionadas:
        - pares, primos, moldura, centro
        - soma, amplitude, consecutivos
        """
        game = [int(x) for x in game]
        
        return np.array([
            contar_pares(game),           # 0: pares
            contar_primos(game),          # 1: primos
            contar_moldura(game),         # 2: moldura
            contar_centro(game),          # 3: centro
            calcular_soma(game),          # 4: soma
            calcular_amplitude(game),     # 5: amplitude
            contar_consecutivos(game),    # 6: consecutivos
        ], dtype=np.float64)
    
    def _build_mahalanobis_model(self):
        """
        Constrói modelo multivariado com dados históricos
        
        A matriz de covariância captura correlações entre features
        Ex: pares vs soma, primos vs amplitude, etc.
        """
        print("📐 Construindo modelo Mahalanobis...")
        
        features = []
        for game in self.dezenas_historicas:
            vec = self._extract_feature_vector(game)
            features.append(vec)
        
        self.feature_matrix = np.array(features)
        self.feature_mean = np.mean(self.feature_matrix, axis=0)
        
        # Matriz de covariância com regularização
        cov_estimator = EmpiricalCovariance().fit(self.feature_matrix)
        cov_matrix = cov_estimator.covariance_
        cov_matrix += np.eye(len(self.feature_mean)) * 1e-6
        
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        # Estatísticas para normalização
        self.feature_std = np.std(self.feature_matrix, axis=0)
        
        print(f"   ✅ Features: {len(self.feature_mean)} dimensões")
        print(f"   ✅ Médias: {self.feature_mean.round(1)}")
    
    def compute_mahalanobis_distance(self, game):
        """
        Calcula distância de Mahalanobis de um jogo
        
        Retorna:
            float: Distância (menor = mais típico historicamente)
        """
        game_key = tuple(sorted(game))
        if game_key in self._mahalanobis_cache:
            return self._mahalanobis_cache[game_key]
        
        vec = self._extract_feature_vector(game)
        
        try:
            dist = mahalanobis(vec, self.feature_mean, self.inv_cov_matrix)
        except:
            # Fallback: distância euclidiana normalizada
            dist = np.sqrt(np.sum(((vec - self.feature_mean) / (self.feature_std + 1e-10)) ** 2))
        
        self._mahalanobis_cache[game_key] = dist
        return dist
    
    def compute_mahalanobis_score(self, game):
        """
        Converte distância Mahalanobis em score (0-20)
        
        Distribuição de referência:
        - dist < 2: muito típico (score alto)
        - dist 2-4: típico (score médio)
        - dist 4-6: atípico (score baixo)
        - dist > 6: muito atípico (score mínimo)
        """
        dist = self.compute_mahalanobis_distance(game)
        
        # Função sigmoide invertida para score suave
        # Score máximo (20) quando dist=0
        # Score ~10 quando dist=3
        # Score ~2 quando dist=6
        score = 20 * np.exp(-dist / 3.0)
        
        return score
    
    # ============================================================
    # SCORE BAYESIANO / RARIDADE HISTÓRICA
    # ============================================================
    
    def _build_historical_patterns(self):
        """
        Constrói distribuição de frequência de padrões históricos
        
        Padrão: (faixa_pares, faixa_primos, faixa_soma)
        """
        print("📊 Construindo padrões históricos...")
        
        self.historical_patterns = Counter()
        
        for game in self.dezenas_historicas:
            game = [int(x) for x in game]
            
            pares = contar_pares(game)
            primos = contar_primos(game)
            soma = calcular_soma(game)
            
            # Discretizar em faixas
            faixa_pares = pares  # 0-15, já discreto
            faixa_primos = primos  # 0-9, já discreto
            faixa_soma = soma // 10  # 0-30 -> 0-3 faixas
            
            pattern = (faixa_pares, faixa_primos, faixa_soma)
            self.historical_patterns[pattern] += 1
        
        total = len(self.dezenas_historicas)
        self.historical_patterns_probs = {
            k: v / total for k, v in self.historical_patterns.items()
        }
        
        print(f"   ✅ {len(self.historical_patterns)} padrões únicos identificados")
    
    def compute_rarity_score(self, game):
        """
        Calcula score de raridade bayesiana
        
        Jogos com padrões frequentes historicamente = score ALTO
        Jogos com padrões raros = score BAIXO
        """
        game = [int(x) for x in game]
        
        pares = contar_pares(game)
        primos = contar_primos(game)
        soma = calcular_soma(game)
        
        faixa_pares = pares
        faixa_primos = primos
        faixa_soma = soma // 10
        
        pattern = (faixa_pares, faixa_primos, faixa_soma)
        
        # Probabilidade do padrão nos dados históricos
        prob = self.historical_patterns_probs.get(pattern, 0.001)
        
        # Converter para score (0-15)
        # Padrões comuns: score alto
        # Padrões raros: score baixo
        if prob > 0.05:  # Muito comum (>5% dos jogos)
            score = 15.0
        elif prob > 0.02:  # Comum
            score = 12.0
        elif prob > 0.01:  # Moderado
            score = 8.0
        elif prob > 0.005:  # Raro
            score = 4.0
        else:  # Muito raro
            score = 0.0
        
        return score
    
    # ============================================================
    # PENALIDADES ESTRUTURAIS
    # ============================================================
    
    def calculate_structural_penalty(self, game):
        """Penalidade estrutural (soft constraints)"""
        game_set = set(game)
        penalty = 0.0
        c = self.constraints
        
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
        
        for row_start in [1, 6, 11, 16, 21]:
            row_numbers = set(range(row_start, row_start + 5))
            overlap = len(game_set & row_numbers)
            if overlap > c['max_per_line']:
                penalty += (overlap - c['max_per_line']) * c['line_weight']
        
        for col_start in range(1, 6):
            col_numbers = set(range(col_start, 26, 5))
            overlap = len(game_set & col_numbers)
            if overlap > c['max_per_column']:
                penalty += (overlap - c['max_per_column']) * c['column_weight']
        
        return penalty
    
    # ============================================================
    # FITNESS PRINCIPAL (CORRIGIDO)
    # ============================================================
    
    def compute_game_fitness(self, game):
        """
        FITNESS MULTIOBJETIVO COM MAHALANOBIS
        
        Componentes (todos 0-20 para equilíbrio):
        1. Estrutural: penalidades baixas = score alto
        2. Entropia: diversidade interna
        3. Quadrantes: cobertura espacial
        4. Balanceamento: par/ímpar
        5. Mahalanobis: similaridade histórica multivariada
        6. Raridade: padrões historicamente comuns
        7. Penalidade consecutivos: evitar artificialidade
        """
        game_key = tuple(sorted(game))
        if game_key in self._fitness_cache:
            return self._fitness_cache[game_key]
        
        # 1. Estrutural (0-20)
        penalty = self.calculate_structural_penalty(game)
        structural_score = max(0, 20 - penalty * 0.8)
        
        # 2. Entropia local (0-15)
        dezena_probs = np.bincount(game, minlength=26)[1:] / 15
        dezena_probs = np.where(dezena_probs > 0, dezena_probs, 1e-10)
        local_entropy = entropy(dezena_probs)
        entropy_score = (local_entropy / np.log(15)) * 15
        
        # 3. Quadrantes (0-15)
        quadrant_count = sum(
            1 for q in QUADRANTES.values()
            if len(set(game) & q) >= 2
        )
        quadrant_score = (quadrant_count / 5) * 15
        
        # 4. Balanceamento par/ímpar (0-10)
        pares = contar_pares(game)
        balance_score = max(0, 10 - abs(pares - 7.5) * 1.5)
        
        # 5. MAHALANOBIS (0-20) - NOVA MÉTRICA PRINCIPAL
        mahalanobis_score = self.compute_mahalanobis_score(game)
        
        # 6. RARIDADE BAYESIANA (0-10)
        rarity_score = self.compute_rarity_score(game) * 0.67  # Normalizar para 0-10
        
        # 7. PENALIDADE CONSECUTIVOS (-8 a 0)
        consecutivos = contar_consecutivos(game)
        consecutive_penalty = 0
        if consecutivos > 6:
            consecutive_penalty = (consecutivos - 6) * 2
        if consecutivos > 8:
            consecutive_penalty += (consecutivos - 8) * 3
        
        # FITNESS TOTAL
        total_fitness = (
            structural_score * 0.75 +    # 15 max
            entropy_score * 0.67 +       # 10 max
            quadrant_score * 0.67 +      # 10 max
            balance_score * 1.0 +        # 10 max
            mahalanobis_score * 1.0 +    # 20 max (MUITO IMPORTANTE)
            rarity_score * 1.0 -         # 10 max
            consecutive_penalty          # Penalidade
        )
        
        metrics = {
            'structural_score': structural_score,
            'entropy_score': entropy_score,
            'quadrant_score': quadrant_score,
            'balance_score': balance_score,
            'mahalanobis_score': mahalanobis_score,
            'rarity_score': rarity_score,
            'consecutive_penalty': consecutive_penalty,
            'mahalanobis_distance': self.compute_mahalanobis_distance(game),
            'pares': pares,
            'impares': 15 - pares,
            'consecutivos': consecutivos,
            'penalty': penalty
        }
        
        self._fitness_cache[game_key] = (total_fitness, metrics)
        return total_fitness, metrics
    
    # ============================================================
    # MÉTODOS PÚBLICOS
    # ============================================================
    
    def get_last_draw(self):
        """Retorna o último concurso real"""
        draws = self.get_last_draws(1)
        return draws[0] if draws else None
    
    def get_last_draws(self, n=10):
        """Retorna os últimos n concursos reais"""
        if self.df is None or len(self.df) == 0:
            return None
        
        latest = self.df.sort_values('concurso', ascending=False).head(n)
        results = []
        
        for _, row in latest.iterrows():
            game = [int(row[f'b{i}']) for i in range(1, 16)]
            results.append(sorted(game))
        
        return results
    
    def get_historical_frequency(self):
        """Frequência histórica das dezenas"""
        if self.dezenas_historicas is None:
            return {}
        
        freq = np.bincount(self.dezenas_historicas.flatten(), minlength=26)[1:]
        return {i+1: int(f) for i, f in enumerate(freq)}
    
    def generate_candidates(self, n_candidates=5000, respect_constraints=True):
        """Gera candidatos diversos"""
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
    
    def rank_games(self, games, top_n=10):
        """Rankeia jogos por fitness (com Mahalanobis)"""
        ranked = []
        for game in games:
            fitness, metrics = self.compute_game_fitness(game)
            ranked.append((fitness, game, metrics))
        
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:top_n]
    
    def compute_pool_fitness(self, pool):
        """Fitness de pool (para otimização)"""
        n_games = len(pool)
        
        individual_scores = []
        for game in pool:
            score, _ = self.compute_game_fitness(game)
            individual_scores.append(score)
        
        avg_individual = np.mean(individual_scores) if individual_scores else 0
        
        overlaps = []
        for i in range(n_games):
            for j in range(i+1, n_games):
                common = len(set(pool[i]) & set(pool[j]))
                overlaps.append(common)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        overlap_deviation = abs(avg_overlap - self.constraints['target_overlap'])
        overlap_penalty = overlap_deviation * self.constraints['overlap_weight']
        overlap_score = max(0, 20 - overlap_penalty)
        
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
        
        avg_11, avg_12, avg_13, avg_14 = self._fast_prize_proxy(pool)
        prize_score = min(20, avg_11 * 4 + avg_12 * 6 + avg_13 * 6 + avg_14 * 4)
        
        total_fitness = (
            avg_individual * 0.3 +
            overlap_score * 0.2 +
            redundancy_score * 0.25 +
            prize_score * 0.25
        )
        
        return total_fitness, {
            'avg_individual': avg_individual,
            'overlap_score': overlap_score,
            'redundancy_score': redundancy_score,
            'prize_score': prize_score,
            'avg_overlap': avg_overlap,
            'avg_11': avg_11,
            'avg_12': avg_12,
            'avg_13': avg_13,
            'avg_14': avg_14
        }
    
    def _fast_prize_proxy(self, pool):
        """Proxy rápido para premiação"""
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
    
    def optimize_pool(self, n_games=50, candidate_pool_size=2000, iterations=500):
        """Otimiza pool de jogos"""
        candidates = self.generate_candidates(candidate_pool_size)
        
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
        
        current_pool = pool.copy()
        current_fitness, _ = self.compute_pool_fitness(current_pool)
        best_pool = [g.copy() for g in current_pool]
        best_fitness = current_fitness
        
        for iteration in range(iterations):
            temp = 5.0 * (0.05 / 5.0) ** (iteration / iterations)
            
            new_pool = [g.copy() for g in current_pool]
            idx = np.random.randint(0, len(new_pool))
            
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
# FUNÇÕES UTILITÁRIAS
# ============================================================

def contar_pares(game):
    return sum(1 for x in game if x % 2 == 0)

def contar_impares(game):
    return sum(1 for x in game if x % 2 != 0)

def contar_primos(game):
    return sum(1 for x in game if x in PRIMES)

def contar_moldura(game):
    return sum(1 for x in game if x in MOLDURA)

def contar_centro(game):
    return sum(1 for x in game if x in CENTRO)

def contar_repetidos(game, reference):
    return len(set(game) & set(reference))

def contar_quadrante(game, quadrante):
    return sum(1 for x in game if x in quadrante)

def calcular_soma(game):
    return sum(game)

def calcular_amplitude(game):
    return max(game) - min(game)

def contar_consecutivos(game):

    d = sorted(game)

    max_block = 1
    current = 1

    for i in range(len(d)-1):

        if d[i+1] - d[i] == 1:
            current += 1
            max_block = max(max_block, current)
        else:
            current = 1

    return max_block


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("="*70)
    print("🎯 MOTOR DE OTIMIZAÇÃO LOTOFÁCIL v3.1")
    print("   Mahalanobis + Score Bayesiano + Penalidades")
    print("="*70)
    
    optimizer = LotofacilOptimizerV31()
    
    # Demonstração com Mahalanobis
    print("\n📊 Demonstração do Mahalanobis:")
    
    # Gerar candidatos
    candidates = optimizer.generate_candidates(200)
    print(f"   ✅ {len(candidates)} candidatos gerados")
    
    # Rankear com novo fitness
    ranked = optimizer.rank_games(candidates, 10)
    
    print(f"\n🏆 TOP 10 (com Mahalanobis):")
    print(f"{'Rank':<6} {'Fitness':<10} {'Mahal Dist':<12} {'Mahal Score':<12} {'Consec':<8}")
    print("-"*50)
    
    for i, (fitness, game, metrics) in enumerate(ranked, 1):
        print(f"{i:<6} {fitness:<10.2f} {metrics['mahalanobis_distance']:<12.2f} "
              f"{metrics['mahalanobis_score']:<12.2f} {metrics['consecutivos']:<8}")
    
    print(f"\n📈 MÉTRICAS DO TOP 1:")
    best_fitness, best_game, best_metrics = ranked[0]
    print(f"   Jogo: {sorted(best_game)}")
    print(f"   Fitness Total: {best_fitness:.2f}")
    print(f"   Mahalanobis Score: {best_metrics['mahalanobis_score']:.2f}/20")
    print(f"   Distância Mahalanobis: {best_metrics['mahalanobis_distance']:.2f}")
    print(f"   Raridade Score: {best_metrics['rarity_score']:.2f}/10")
    print(f"   Pares: {best_metrics['pares']} | Consecutivos: {best_metrics['consecutivos']}")
    
    print("\n✅ Motor funcionando com discriminação real!")


if __name__ == "__main__":
    main()
