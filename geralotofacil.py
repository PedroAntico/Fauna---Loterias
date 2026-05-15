#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA AVANÇADO DE OTIMIZAÇÃO COMBINATÓRIA - LOTOFÁCIL
========================================================
Versão 1.0 - Otimização Global com Cobertura Explícita

CARACTERÍSTICAS DA LOTOFÁCIL:
✅ 25 números (01 a 25)
✅ 15 números por aposta
✅ Alta probabilidade de acertos parciais
✅ Menor espaço amostral que Mega-Sena
✅ Foco em cobertura de 11, 12, 13, 14 pontos

ESTRATÉGIA:
Maximizar cobertura de combinações parciais
Otimizar diversidade estrutural
Minimizar sobreposição entre bilhetes
Manter estatísticas historicamente típicas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis, hamming, euclidean
from scipy.stats import entropy
from scipy.linalg import det
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from collections import Counter, defaultdict
from itertools import combinations, product
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

class LotofacilCoverageOptimizer:
    """
    Otimizador Avançado de Cobertura para Lotofácil
    
    Específico para 25 números, 15 por aposta
    Foco em maximizar cobertura de acertos parciais (11-14 pontos)
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """
        Inicializa o otimizador para Lotofácil
        
        Args:
            historical_csv: Arquivo CSV com resultados históricos
                           Formato: concurso;data;b1;b2;...;b15
        """
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Parâmetros específicos da Lotofácil
        self.TOTAL_NUMBERS = 25
        self.NUMBERS_PER_GAME = 15
        self.MIN_PRIZE = 11  # Mínimo para premiação
        
        # Features para análise multivariada
        self.feature_names = [
            'soma', 'pares', 'impares', 'primos', 'fibonacci',
            'q1', 'q2', 'q3', 'q4', 'q5',  # 5 quadrantes de 5 números
            'amplitude', 'distancia_media', 'consecutivas',
            'multiplos_3', 'multiplos_5', 'numeros_baixos', 'numeros_altos'
        ]
        
        # Conjuntos matemáticos
        self.primes = self._gen_primes(25)
        self.fibonacci = self._gen_fibonacci(25)
        
        # Estatísticas
        self.reference_stats = {}
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.feature_means = None
        
        # Padrões humanos
        self.human_patterns = self._define_human_patterns()
        
        # Carregar dados
        self._load_historical_data()
        self._compute_multivariate_reference()
        
        print("✅ Otimizador Lotofácil inicializado!")
        print(f"📊 {len(self.df)} concursos históricos analisados")
        print(f"🎯 Otimizando para {self.NUMBERS_PER_GAME} números em {self.TOTAL_NUMBERS}")
    
    def _gen_primes(self, limit):
        """Gera números primos até o limite"""
        return {n for n in range(2, limit+1) 
                if all(n % i != 0 for i in range(2, int(n**0.5)+1))}
    
    def _gen_fibonacci(self, limit):
        """Gera números Fibonacci até o limite"""
        fib = [0, 1]
        while fib[-1] <= limit:
            fib.append(fib[-1] + fib[-2])
        return set(f for f in fib[2:] if f <= limit)
    
    def _define_human_patterns(self):
        """
        Define padrões humanos comuns na Lotofácil
        """
        # Sequências óbvias
        sequences = [
            set(range(1, 16)),      # 1-15
            set(range(11, 26)),     # 11-25
            set(range(1, 26, 2)),   # Todos ímpares
            set(range(2, 26, 2)),   # Todos pares
            {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},  # Primeira metade
            {11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},  # Segunda metade
        ]
        
        # Números "da sorte" populares
        lucky_numbers = {1, 3, 7, 8, 13, 17, 21, 23, 25}
        
        # Padrões de data (dia + mês <= 25)
        date_numbers = set(range(1, 32)) & set(range(1, 26))
        
        # Padrões de linha/coluna no volante (5x5)
        line_patterns = [
            {1,2,3,4,5},      # Linha 1
            {6,7,8,9,10},     # Linha 2
            {11,12,13,14,15}, # Linha 3
            {16,17,18,19,20}, # Linha 4
            {21,22,23,24,25}, # Linha 5
        ]
        
        column_patterns = [
            {1,6,11,16,21},   # Coluna 1
            {2,7,12,17,22},   # Coluna 2
            {3,8,13,18,23},   # Coluna 3
            {4,9,14,19,24},   # Coluna 4
            {5,10,15,20,25},  # Coluna 5
        ]
        
        return {
            'sequences': sequences,
            'lucky_numbers': lucky_numbers,
            'date_numbers': date_numbers,
            'line_patterns': line_patterns,
            'column_patterns': column_patterns,
            'common_pairs': self._generate_common_pairs()
        }
    
    def _generate_common_pairs(self):
        """Pares frequentemente escolhidos juntos"""
        common_pairs = set()
        
        # Consecutivos
        for i in range(1, 25):
            common_pairs.add((i, i+1))
        
        # Mesma coluna (diferença 5)
        for i in range(1, 21):
            common_pairs.add((i, i+5))
        
        # Datas comemorativas (pares de dia/mês)
        date_pairs = [(1,1), (25,12), (7,9), (12,10), (15,11), (2,11)]
        for d1, d2 in date_pairs:
            if d1 <= 25 and d2 <= 25:
                common_pairs.add((min(d1, d2), max(d1, d2)))
        
        return common_pairs
    
    def _load_historical_data(self):
        """Carrega dados históricos da Lotofácil"""
        print("📂 Carregando dados históricos da Lotofácil...")
        
        # Tentar carregar, se não existir, gerar dados sintéticos para demonstração
        try:
            self.df = pd.read_csv(self.historical_csv, sep=';', encoding='utf-8')
            print("   ✅ Arquivo histórico encontrado")
        except FileNotFoundError:
            print("   ⚠️  Arquivo não encontrado. Gerando dados sintéticos...")
            self._generate_synthetic_data()
            return
        
        # Identificar colunas
        bola_cols = [f'b{i}' for i in range(1, 16)]
        self.df.columns = ['concurso', 'data'] + bola_cols
        self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
        self.dezenas_historicas = self.df[bola_cols].values
        
        print(f"   ✅ {len(self.df)} concursos carregados")
    
    def _generate_synthetic_data(self, n_games=2000):
        """
        Gera dados sintéticos baseados em distribuições realistas
        para demonstração quando não há arquivo histórico
        """
        print("   🔧 Gerando dados sintéticos realistas...")
        
        synthetic_games = []
        
        for _ in range(n_games):
            # Simular distribuição realista
            # Na Lotofácil, as dezenas têm distribuição aproximadamente uniforme
            # mas com pequenas variações
            
            # Gerar pesos ligeiramente diferentes para simular não-uniformidade
            weights = np.ones(25)
            # Pequenas variações aleatórias nos pesos
            weights += np.random.normal(0, 0.1, 25)
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            game = sorted(np.random.choice(
                range(1, 26), 
                size=15, 
                replace=False, 
                p=weights
            ))
            synthetic_games.append(game)
        
        # Criar DataFrame sintético
        self.df = pd.DataFrame({
            'concurso': range(1, n_games + 1),
            'data': pd.date_range(start='2003-09-29', periods=n_games, freq='3D')
        })
        
        for i in range(15):
            self.df[f'b{i+1}'] = [g[i] for g in synthetic_games]
        
        bola_cols = [f'b{i}' for i in range(1, 16)]
        self.dezenas_historicas = self.df[bola_cols].values
        
        print(f"   ✅ {len(self.df)} jogos sintéticos gerados")
    
    def _extract_features(self, game):
        """
        Extrai vetor de features para um jogo da Lotofácil
        
        Features específicas para 25 números, 15 por jogo
        """
        game = np.array(sorted(game))
        
        # Quadrantes (5 regiões de 5 números cada)
        q1 = np.sum((game >= 1) & (game <= 5))
        q2 = np.sum((game >= 6) & (game <= 10))
        q3 = np.sum((game >= 11) & (game <= 15))
        q4 = np.sum((game >= 16) & (game <= 20))
        q5 = np.sum((game >= 21) & (game <= 25))
        
        return np.array([
            np.sum(game),                                    # soma
            np.sum(game % 2 == 0),                          # pares
            np.sum(game % 2 != 0),                          # impares
            np.sum(np.isin(game, list(self.primes))),       # primos
            np.sum(np.isin(game, list(self.fibonacci))),    # fibonacci
            q1, q2, q3, q4, q5,                             # quadrantes
            game.max() - game.min(),                         # amplitude
            np.mean([game[i+1] - game[i] for i in range(14)]), # distancia_media
            self._count_consecutive(game),                   # consecutivas
            np.sum(game % 3 == 0),                          # multiplos_3
            np.sum(game % 5 == 0),                          # multiplos_5
            np.sum(game <= 12),                             # numeros_baixos
            np.sum(game >= 14),                             # numeros_altos
        ])
    
    def _count_consecutive(self, dezenas):
        """Conta sequências consecutivas"""
        d = sorted(dezenas)
        count = 0
        seq = 1
        for i in range(len(d)-1):
            if d[i+1] - d[i] == 1:
                seq += 1
            else:
                if seq >= 2:
                    count += seq
                seq = 1
        if seq >= 2:
            count += seq
        return count
    
    def _compute_multivariate_reference(self):
        """Calcula referência multivariada completa"""
        print("📊 Calculando referência multivariada...")
        
        # Extrair features de todos os concursos
        X = np.array([self._extract_features(d) for d in self.dezenas_historicas])
        
        # Estatísticas
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        
        # Matriz de covariância para Mahalanobis
        cov_estimator = EmpiricalCovariance().fit(X)
        self.cov_matrix = cov_estimator.covariance_
        self.cov_matrix += np.eye(len(self.feature_names)) * 1e-6
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        
        # PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=0.95)
        self.pca.fit(X_scaled)
        
        # Frequência individual das dezenas
        freq = np.bincount(self.dezenas_historicas.flatten(), minlength=26)[1:]
        self.reference_stats['frequencia'] = freq / len(self.dezenas_historicas)
        
        # Distribuição de pares/ímpares
        pares = np.sum(self.dezenas_historicas % 2 == 0, axis=1)
        self.reference_stats['pares_dist'] = np.bincount(pares, minlength=16) / len(pares)
        
        print(f"   ✅ {len(self.feature_names)} features extraídas")
        print(f"   ✅ PCA: {self.pca.n_components_} componentes principais")
    
    def mahalanobis_distance(self, game):
        """Distância de Mahalanobis multivariada"""
        features = self._extract_features(game)
        return mahalanobis(features, self.feature_means, self.inv_cov_matrix)
    
    def _compute_human_score(self, game):
        """
        Calcula quão "humano" é um jogo (para EV Optimizer)
        Quanto maior, mais provável que humanos escolham
        """
        game_set = set(game)
        score = 0
        
        # Números de data
        date_count = len(game_set & self.human_patterns['date_numbers'])
        score += date_count * 2
        
        # Números da sorte
        lucky_count = len(game_set & self.human_patterns['lucky_numbers'])
        score += lucky_count * 3
        
        # Sequências conhecidas
        for seq in self.human_patterns['sequences']:
            overlap = len(game_set & seq)
            if overlap >= 10:  # Mais rigoroso por ser 15 números
                score += overlap * 3
        
        # Padrões de linha
        for line in self.human_patterns['line_patterns']:
            overlap = len(game_set & line)
            if overlap >= 4:
                score += overlap * 2
        
        # Padrões de coluna
        for col in self.human_patterns['column_patterns']:
            overlap = len(game_set & col)
            if overlap >= 4:
                score += overlap * 2
        
        # Pares comuns
        for pair in combinations(sorted(game), 2):
            if pair in self.human_patterns['common_pairs']:
                score += 0.5
        
        return score
    
    def _compute_coverage_metrics(self, pool):
        """
        Calcula métricas de cobertura específicas para Lotofácil
        
        Foco em cobertura de 11, 12, 13, 14 pontos
        """
        n_games = len(pool)
        
        # Cobertura de dezenas
        all_dezenas = set()
        for game in pool:
            all_dezenas.update(game)
        
        dezenas_cobertas = len(all_dezenas)
        coverage_pct = dezenas_cobertas / self.TOTAL_NUMBERS
        
        # Cobertura de pares
        all_pairs = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                all_pairs.add(pair)
        
        total_pairs = self.TOTAL_NUMBERS * (self.TOTAL_NUMBERS - 1) // 2
        pair_coverage = len(all_pairs) / total_pairs
        
        # Cobertura de trincas
        all_triples = set()
        for game in pool:
            for triple in combinations(sorted(game), 3):
                all_triples.add(triple)
        
        total_triples = self.TOTAL_NUMBERS * (self.TOTAL_NUMBERS - 1) * (self.TOTAL_NUMBERS - 2) // 6
        triple_coverage = len(all_triples) / total_triples
        
        # Simulação de cobertura de premiação
        # Gerar jogos de teste aleatórios e verificar quantos acertos parciais
        prize_coverage = self._simulate_prize_coverage(pool, n_simulations=100)
        
        return {
            'dezenas_cobertas': dezenas_cobertas,
            'coverage_pct': coverage_pct,
            'pares_cobertos': len(all_pairs),
            'pair_coverage': pair_coverage,
            'trincas_cobertas': len(all_triples),
            'triple_coverage': triple_coverage,
            'avg_11_points': prize_coverage.get(11, 0),
            'avg_12_points': prize_coverage.get(12, 0),
            'avg_13_points': prize_coverage.get(13, 0),
            'avg_14_points': prize_coverage.get(14, 0)
        }
    
    def _simulate_prize_coverage(self, pool, n_simulations=100):
        """
        Simula cobertura de premiação
        
        Gera jogos aleatórios e verifica quantos bilhetes do pool
        acertariam 11, 12, 13, 14 pontos
        """
        prize_counts = defaultdict(list)
        success = defaultdict(int)
        
        for _ in range(n_simulations):
            # Gerar jogo "sorteado" aleatório
            drawn = set(sorted(np.random.choice(range(1, 26), 15, replace=False)))
            found = {11:False,12:False,13:False,14:False}
            
            # Verificar acertos de cada bilhete do pool
            for game in pool:
                hits = len(set(game) & drawn)
                
                if hits >= 11:
                    found[hits] = True
        
            for k,v in found.items():
                if v:
                    success[k] += 1
        
        return {k: success[k]/n_simulations}
        
        # Média de bilhetes premiados por simulação
        avg_prizes = {}
        for hits in [11, 12, 13, 14]:
            if prize_counts[hits]:
                avg_prizes[hits] = np.mean(prize_counts[hits])
            else:
                avg_prizes[hits] = 0
        
        return avg_prizes
    
    def _compute_pool_global_fitness(self, pool):
        """
        FITNESS GLOBAL DO POOL PARA LOTOFÁCIL
        
        Adaptado para 25 números, 15 por jogo
        Maior peso em cobertura de premiação parcial
        """
        # Métricas de cobertura
        coverage = self._compute_coverage_metrics(pool)
        
        # 1. COBERTURA DE DEZENAS (15 pontos)
        dezenas_score = coverage['coverage_pct'] * 15
        
        # 2. COBERTURA COMBINATÓRIA (25 pontos)
        pair_score = coverage['pair_coverage'] * 12
        triple_score = coverage['triple_coverage'] * 13
        combinatorial_score = pair_score + triple_score
        
        # 3. COBERTURA DE PREMIAÇÃO (25 pontos)
        # Foco em 11-14 pontos
        prize_score = (
            coverage['avg_11_points'] * 5 +
            coverage['avg_12_points'] * 7 +
            coverage['avg_13_points'] * 8 +
            coverage['avg_14_points'] * 5
        )
        prize_score = min(prize_score, 25)  # Cap
        
        # 4. DIVERSIDADE MULTIVARIADA (20 pontos)
        features_matrix = np.array([self._extract_features(g) for g in pool])
        
        distances = []
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                dist = np.linalg.norm(features_matrix[i] - features_matrix[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        max_dist = np.linalg.norm(self.feature_stds * 3)
        diversity_score = (avg_distance / max_dist) * 20
        
        # 5. ANTI-HUMAN SCORE (10 pontos)
        human_scores = [self._compute_human_score(g) for g in pool]
        avg_human = np.mean(human_scores)
        anti_human_score = max(0, 10 - avg_human * 0.3)
        
        # 6. ENTROPIA (5 pontos)
        dezena_counts = np.bincount([d for game in pool for d in game], minlength=26)[1:]
        dezena_probs = dezena_counts / np.sum(dezena_counts)
        pool_entropy = entropy(dezena_probs + 1e-10)
        entropy_score = (pool_entropy / np.log(self.TOTAL_NUMBERS)) * 5
        
        # FITNESS TOTAL
        total_fitness = (
            dezenas_score +
            combinatorial_score +
            prize_score +
            diversity_score +
            anti_human_score +
            entropy_score
        )
        
        return total_fitness, {
            'dezenas': dezenas_score,
            'combinatorial': combinatorial_score,
            'prize_coverage': prize_score,
            'diversity': diversity_score,
            'anti_human': anti_human_score,
            'entropy': entropy_score,
            **coverage
        }
    
    def _dpp_sampling(self, n_games, pool_size=1000):
        """
        DPP (Determinantal Point Process) para diversidade máxima
        
        Adaptado para Lotofácil (espaço menor)
        """
        print("🎯 Aplicando DPP para diversidade máxima...")
        
        # Gerar pool inicial
        initial_pool = []
        seen = set()
        
        while len(initial_pool) < pool_size:
            game = tuple(sorted(np.random.choice(range(1, 26), 15, replace=False)))
            if game not in seen:
                seen.add(game)
                initial_pool.append(list(game))
        
        # Features
        X = np.array([self._extract_features(g) for g in initial_pool])
        X_scaled = self.scaler.transform(X)
        
        # Kernel RBF
        def rbf_kernel(x1, x2, sigma=2.0):
            dist = np.linalg.norm(x1 - x2)
            return np.exp(-dist**2 / (2 * sigma**2))
        
        # Matriz de kernel
        n = len(initial_pool)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = rbf_kernel(X_scaled[i], X_scaled[j])
        
        # Seleção DPP
        selected_indices = []
        remaining = list(range(n))
        
        for _ in range(n_games):
            if not remaining:
                break
            
            volumes = []
            for idx in remaining:
                indices = selected_indices + [idx]
                if len(indices) == 1:
                    volumes.append(K[idx, idx])
                else:
                    sub_K = K[np.ix_(indices, indices)]
                    try:
                        vol = np.linalg.det(sub_K)
                        volumes.append(max(0, vol))
                    except:
                        volumes.append(0)
            
            if volumes and np.max(volumes) > 0:
                best_idx = remaining[np.argmax(volumes)]
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                idx = np.random.choice(remaining)
                selected_indices.append(idx)
                remaining.remove(idx)
        
        return [initial_pool[i] for i in selected_indices[:n_games]]
    
    def _simulated_annealing(self, initial_pool, iterations=1000, temp_start=10, temp_end=0.1):
        """
        Simulated Annealing para otimização global
        """
        print(f"🔥 Simulated Annealing ({iterations} iterações)...")
        
        current_pool = [list(g) for g in initial_pool]
        current_fitness, _ = self._compute_pool_global_fitness(current_pool)
        
        best_pool = current_pool.copy()
        best_fitness = current_fitness
        
        fitness_history = [current_fitness]
        
        for iteration in tqdm(range(iterations), desc="Annealing"):
            temp = temp_start * (temp_end / temp_start) ** (iteration / iterations)
            
            # Modificar pool
            new_pool = current_pool.copy()
            idx = np.random.randint(0, len(new_pool))
            
            # Trocar uma dezena
            game = new_pool[idx].copy()
            old_value = game[np.random.randint(0, 15)]
            
            available = [d for d in range(1, 26) if d not in game]
            if available:
                game[game.index(old_value)] = np.random.choice(available)
                game.sort()
                new_pool[idx] = game
            
            # Avaliar
            new_fitness, new_metrics = self._compute_pool_global_fitness(new_pool)
            
            # Metropolis criterion
            delta = new_fitness - current_fitness
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_pool = new_pool
                current_fitness = new_fitness
                
                if current_fitness > best_fitness:
                    best_pool = current_pool.copy()
                    best_fitness = current_fitness
            
            fitness_history.append(current_fitness)
        
        print(f"   ✅ Fitness final: {best_fitness:.2f}")
        return best_pool, fitness_history
    
    def optimize_pool(self, n_games=50, dpp_pool_size=500, annealing_iterations=500):
        """
        Pipeline completo de otimização
        """
        print("\n" + "="*60)
        print(f"🎯 OTIMIZANDO POOL DE {n_games} JOGOS PARA LOTOFÁCIL")
        print("="*60)
        
        # 1. DPP para diversidade inicial
        initial_pool = self._dpp_sampling(n_games, dpp_pool_size)
        
        # 2. Simulated Annealing
        optimized_pool, fitness_history = self._simulated_annealing(
            initial_pool, 
            iterations=annealing_iterations
        )
        
        # 3. Avaliação final
        final_fitness, final_metrics = self._compute_pool_global_fitness(optimized_pool)
        
        print(f"\n📊 MÉTRICAS FINAIS DO POOL:")
        print(f"   • Fitness Total: {final_fitness:.2f}/100")
        print(f"   • Cobertura de Dezenas: {final_metrics['dezenas_cobertas']}/25")
        print(f"   • Cobertura de Pares: {final_metrics['pares_cobertos']}/300")
        print(f"   • Cobertura de Trincas: {final_metrics['trincas_cobertas']}/2300")
        print(f"   • Média 11 pts: {final_metrics['avg_11_points']:.2f} bilhetes/sorteio")
        print(f"   • Média 12 pts: {final_metrics['avg_12_points']:.2f} bilhetes/sorteio")
        print(f"   • Média 13 pts: {final_metrics['avg_13_points']:.2f} bilhetes/sorteio")
        print(f"   • Média 14 pts: {final_metrics['avg_14_points']:.2f} bilhetes/sorteio")
        
        self.final_metrics = final_metrics
        self.fitness_history = fitness_history
        self.final_pool = optimized_pool
        
        return optimized_pool
    
    def visualize_pool(self, output_dir='graficos_lotofacil'):
        """Visualizações específicas para Lotofácil"""
        print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'final_pool'):
            print("   ⚠️  Execute optimize_pool primeiro")
            return
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Evolução do fitness
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.fitness_history, color='blue', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Fitness Global')
        ax1.set_title('Evolução do Fitness (Simulated Annealing)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cobertura de dezenas (heatmap 5x5)
        ax2 = plt.subplot(2, 3, 2)
        coverage_matrix = np.zeros((5, 5))
        all_dezenas = []
        for game in self.final_pool:
            all_dezenas.extend(game)
        
        for d in all_dezenas:
            row = (d - 1) // 5
            col = (d - 1) % 5
            coverage_matrix[row, col] += 1
        
        sns.heatmap(coverage_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'C{i+1}' for i in range(5)],
                   yticklabels=[f'L{i+1}' for i in range(5)],
                   ax=ax2, cbar_kws={'label': 'Frequência'})
        ax2.set_title('Cobertura no Volante 5x5')
        ax2.set_xlabel('Coluna')
        ax2.set_ylabel('Linha')
        
        # 3. Frequência individual das dezenas
        ax3 = plt.subplot(2, 3, 3)
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        colors = ['green' if f > np.mean(freq) else 'red' if f < np.mean(freq) else 'yellow' 
                 for f in freq]
        ax3.bar(range(1, 26), freq, color=colors, edgecolor='black')
        ax3.axhline(y=np.mean(freq), color='blue', linestyle='--', 
                   label=f'Média: {np.mean(freq):.1f}')
        ax3.set_xlabel('Dezena')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Frequência das Dezenas no Pool')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribuição de Pares vs Ímpares
        ax4 = plt.subplot(2, 3, 4)
        pool_pares = [np.sum(np.array(g) % 2 == 0) for g in self.final_pool]
        hist_pares = np.sum(self.dezenas_historicas % 2 == 0, axis=1)
        
        ax4.hist(hist_pares, bins=range(17), alpha=0.5, label='Histórico', 
                color='blue', density=True)
        ax4.hist(pool_pares, bins=range(17), alpha=0.7, label='Pool', 
                color='green', density=True)
        ax4.set_xlabel('Quantidade de Pares')
        ax4.set_ylabel('Densidade')
        ax4.set_title('Distribuição de Pares')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Radar de métricas
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        metrics = self.final_metrics
        categories = ['Dezenas', 'Pares', 'Trincas', '11 pts', '12 pts', '13 pts', '14 pts']
        values = [
            metrics['coverage_pct'] * 100,
            metrics['pair_coverage'] * 100,
            metrics['triple_coverage'] * 100,
            min(metrics['avg_11_points'] * 10, 100),
            min(metrics['avg_12_points'] * 20, 100),
            min(metrics['avg_13_points'] * 30, 100),
            min(metrics['avg_14_points'] * 50, 100)
        ]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=2, color='green')
        ax5.fill(angles, values, alpha=0.25, color='green')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=9)
        ax5.set_ylim(0, 100)
        ax5.set_title('Métricas de Cobertura (%)')
        ax5.grid(True)
        
        # 6. Comparação Mahalanobis
        ax6 = plt.subplot(2, 3, 6)
        pool_mahal = [self.mahalanobis_distance(g) for g in self.final_pool]
        random_mahal = [self.mahalanobis_distance(
            sorted(np.random.choice(range(1, 26), 15, replace=False))
        ) for _ in range(1000)]
        
        ax6.hist(random_mahal, bins=30, alpha=0.7, label='Aleatório', 
                color='orange', density=True)
        ax6.hist(pool_mahal, bins=15, alpha=0.7, label='Pool', 
                color='green', density=True)
        ax6.axvline(np.mean(random_mahal), color='orange', linestyle='--', 
                   label=f'Média Aleatória: {np.mean(random_mahal):.1f}')
        ax6.axvline(np.mean(pool_mahal), color='green', linestyle='--', 
                   label=f'Média Pool: {np.mean(pool_mahal):.1f}')
        ax6.set_xlabel('Distância de Mahalanobis')
        ax6.set_ylabel('Densidade')
        ax6.set_title('Distribuição Multivariada')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Análise do Pool Otimizado - Lotofácil', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analise_pool_lotofacil.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def export_pool(self, games, filename=None):
        """Exporta pool otimizado"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lotofacil_otimizado_{timestamp}.csv'
        
        df_games = pd.DataFrame(games, columns=[f'D_{i+1:02d}' for i in range(15)])
        df_games.index = [f'Jogo_{i+1}' for i in range(len(games))]
        
        # Métricas por jogo
        df_games['Soma'] = [sum(g) for g in games]
        df_games['Pares'] = [np.sum(np.array(g) % 2 == 0) for g in games]
        df_games['Mahalanobis'] = [self.mahalanobis_distance(g) for g in games]
        df_games['Human_Score'] = [self._compute_human_score(g) for g in games]
        
        df_games.to_csv(filename)
        print(f"\n💾 Pool exportado: {filename}")
        
        return filename
    
    def generate_report(self, games, output_dir='relatorio_lotofacil'):
        """Gera relatório completo"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Métricas
        _, metrics = self._compute_pool_global_fitness(games)
        
        # Cobertura de dezenas específicas
        all_dezenas = []
        for game in games:
            all_dezenas.extend(game)
        
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        top_dezenas = np.argsort(freq)[-5:] + 1
        bottom_dezenas = np.argsort(freq)[:5] + 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_games': len(games),
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()},
            'top_dezenas': [int(x) for x in top_dezenas],
            'bottom_dezenas': [int(x) for x in bottom_dezenas],
            'games': [[int(x) for x in sorted(g)] for g in games]
        }
        
        json_path = f'{output_dir}/relatorio_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Relatório: {json_path}")
        return json_path


def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*70)
    print("🎯 SISTEMA AVANÇADO DE OTIMIZAÇÃO - LOTOFÁCIL")
    print("   25 números | 15 por jogo | Foco em 11-14 pontos")
    print("="*70)
    
    # Inicializar
    optimizer = LotofacilCoverageOptimizer()
    
    # Otimizar
    n_games = 50
    optimized_pool = optimizer.optimize_pool(
        n_games=n_games,
        dpp_pool_size=500,
        annealing_iterations=500
    )
    
    # Visualizar
    optimizer.visualize_pool()
    
    # Exportar
    optimizer.export_pool(optimized_pool)
    
    # Relatório
    optimizer.generate_report(optimized_pool)
    
    # Mostrar jogos
    print(f"\n🎯 POOL OTIMIZADO ({n_games} jogos):")
    print("="*70)
    for i, game in enumerate(optimized_pool, 1):
        sorted_game = sorted(game)
        soma = sum(game)
        pares = sum(1 for n in game if n % 2 == 0)
        print(f"Jogo {i:2d}: {sorted_game}")
        print(f"        Soma={soma:3d} | Pares={pares:2d} | Ímpares={15-pares:2d}")
    
    print("\n" + "="*70)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("📁 Resultados salvos em:")
    print("   • graficos_lotofacil/ - Visualizações")
    print("   • relatorio_lotofacil/ - Relatório JSON")
    print("   • lotofacil_otimizado_*.csv - Jogos exportados")
    print("="*70)
    
    print("\n💡 CARACTERÍSTICAS DO OTIMIZADOR LOTOFÁCIL:")
    print("   1. Cobertura otimizada para 11-14 pontos")
    print("   2. Máxima diversidade entre bilhetes")
    print("   3. Estatísticas historicamente consistentes")
    print("   4. Anti-padrões humanos (EV Optimizer)")
    print("   5. DPP + Simulated Annealing")
    
    print("\n⚠️  DISCLAIMER:")
    print("   Este sistema NÃO prevê resultados futuros.")
    print("   Otimiza cobertura combinatória dentro de")
    print("   restrições estatísticas observadas.")


if __name__ == "__main__":
    main()
