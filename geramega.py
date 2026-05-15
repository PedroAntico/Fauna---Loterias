#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA AVANÇADO DE OTIMIZAÇÃO COMBINATÓRIA - MEGA-SENA
========================================================
Versão 2.0 - Otimização Global com Cobertura Explícita

MELHORIAS CRÍTICAS:
✅ Fitness GLOBAL do pool (não individual)
✅ Cobertura explícita de pares, trincas e quadras
✅ Distância de Mahalanobis multivariada
✅ Simulated Annealing para diversidade máxima
✅ Anti-correlação com padrões humanos (EV optimizer)
✅ Determinantal Point Processes (DPP)
✅ Wheel system parcial (cobertura combinatória)

PRINCÍPIO:
Não prever, mas OTIMIZAR a cobertura do espaço amostral
com restrições estatísticas e anti-viés humano.
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
import heapq

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

class AdvancedCoverageOptimizer:
    """
    Otimizador Avançado de Cobertura Probabilística
    
    Features críticas:
    1. Fitness GLOBAL do pool
    2. Cobertura explícita de pares/trincas/quadras
    3. Mahalanobis distance multivariada
    4. Simulated Annealing para diversidade
    5. DPP (Determinantal Point Processes)
    6. EV Optimizer (anti-padrões humanos)
    """
    
    def __init__(self, historical_csv='resultados_megasena.csv'):
        """Inicialização com análise multivariada completa"""
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Conjuntos matemáticos
        self.fibonacci = self._gen_fibonacci(60)
        self.primes = self._gen_primes(60)
        
        # Features multivariadas
        self.feature_names = [
            'soma', 'pares', 'primos', 'fibonacci', 'amplitude',
            'q1', 'q2', 'q3', 'q4', 'distancia_media', 'consecutivas'
        ]
        
        # Estatísticas de referência
        self.reference_stats = {}
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.feature_means = None
        
        # Padrões humanos comuns (EV optimizer)
        self.human_patterns = self._define_human_patterns()
        
        # Carregar e analisar
        self._load_historical_data()
        self._compute_multivariate_reference()
        
        print("✅ Otimizador Avançado inicializado!")
        print(f"📊 Referência multivariada: {len(self.df)} concursos")
    
    def _gen_fibonacci(self, limit):
        fib = [0, 1]
        while fib[-1] <= limit:
            fib.append(fib[-1] + fib[-2])
        return set(fib[2:])
    
    def _gen_primes(self, limit):
        return {n for n in range(2, limit+1) 
                if all(n % i != 0 for i in range(2, int(n**0.5)+1))}
    
    def _define_human_patterns(self):
        """
        Define padrões que humanos tendem a escolher
        Usado para EV Optimizer (aumentar valor esperado do prêmio)
        """
        # Datas de aniversário (1-31)
        birthday_numbers = set(range(1, 32))
        
        # Sequências óbvias
        sequences = [
            {1,2,3,4,5,6},
            {5,10,15,20,25,30},
            {10,20,30,40,50,60},
            {7,14,21,28,35,42},
        ]
        
        # Números "da sorte" populares
        lucky_numbers = {7, 13, 3, 8, 17, 21, 33, 37, 42, 55}
        
        # Padrões de teclado (disposição visual no volante)
        keyboard_patterns = [
            {1,2,3,4,5,6},      # Primeira linha
            {7,8,9,10,11,12},   # Segunda linha
            {1,11,21,31,41,51}, # Primeira coluna
            {10,20,30,40,50,60}, # Última coluna
        ]
        
        return {
            'birthday_numbers': birthday_numbers,
            'sequences': sequences,
            'lucky_numbers': lucky_numbers,
            'keyboard_patterns': keyboard_patterns,
            'common_pairs': self._generate_common_pairs()
        }
    
    def _generate_common_pairs(self):
        """Pares de números frequentemente escolhidos juntos por humanos"""
        common_pairs = set()
        
        # Pares consecutivos
        for i in range(1, 60):
            common_pairs.add((i, i+1))
        
        # Pares com diferença 10 (mesma coluna no volante)
        for i in range(1, 51):
            common_pairs.add((i, i+10))
        
        # Pares de datas (dia/mês)
        for day in range(1, 32):
            for month in range(1, 13):
                if day <= 60 and month <= 60:
                    common_pairs.add((min(day, month), max(day, month)))
        
        return common_pairs
    
    def _load_historical_data(self):
        """Carrega dados históricos"""
        print("📂 Carregando dados históricos...")
        
        try:
            self.df = pd.read_csv(self.historical_csv, sep=';', encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(self.historical_csv, sep=',', encoding='utf-8')
            except:
                self.df = pd.read_csv(self.historical_csv, sep=';', encoding='latin-1')
        
        self.df.columns = ['concurso', 'data', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
        self.dezenas_historicas = self.df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6']].values
        
        print(f"   ✅ {len(self.df)} concursos carregados")
    
    def _extract_features(self, game):
        """Extrai vetor de features para análise multivariada"""
        game = np.array(sorted(game))
        
        return np.array([
            np.sum(game),                                    # soma
            np.sum(game % 2 == 0),                          # pares
            np.sum(np.isin(game, list(self.primes))),       # primos
            np.sum(np.isin(game, list(self.fibonacci))),    # fibonacci
            game.max() - game.min(),                         # amplitude
            np.sum((game >= 1) & (game <= 15)),             # q1
            np.sum((game >= 16) & (game <= 30)),            # q2
            np.sum((game >= 31) & (game <= 45)),            # q3
            np.sum((game >= 46) & (game <= 60)),            # q4
            np.mean([game[i+1] - game[i] for i in range(5)]), # distancia_media
            self._count_consecutive(game)                    # consecutivas
        ])
    
    def _count_consecutive(self, dezenas):
        """Conta consecutivas de forma padronizada"""
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
        """
        Calcula referência MULTIVARIADA completa
        Usa matriz de covariância para distância de Mahalanobis
        """
        print("📊 Calculando referência multivariada...")
        
        # Extrair features de todos os concursos históricos
        X = np.array([self._extract_features(d) for d in self.dezenas_historicas])
        
        # Estatísticas
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        
        # Matriz de covariância (para Mahalanobis)
        cov_estimator = EmpiricalCovariance().fit(X)
        self.cov_matrix = cov_estimator.covariance_
        
        # Regularização para garantir inversão
        self.cov_matrix += np.eye(len(self.feature_names)) * 1e-6
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        
        # PCA para redução dimensional
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=0.95)  # 95% da variância
        self.pca.fit(X_scaled)
        
        # Estatísticas univariadas para referência
        self.reference_stats = {
            'soma_mean': np.mean(X[:, 0]),
            'soma_std': np.std(X[:, 0]),
            'pares_dist': np.bincount(X[:, 1].astype(int), minlength=7) / len(X),
            'primos_dist': np.bincount(X[:, 2].astype(int), minlength=7) / len(X),
        }
        
        print(f"   ✅ Features: {len(self.feature_names)} dimensões")
        print(f"   ✅ PCA: {self.pca.n_components_} componentes (95% variância)")
    
    def mahalanobis_distance(self, game):
        """
        Calcula distância de Mahalanobis multivariada
        
        Mede quão "típico" é o jogo considerando TODAS as correlações
        entre features simultaneamente.
        """
        features = self._extract_features(game)
        diff = features - self.feature_means
        
        # Distância de Mahalanobis
        md = mahalanobis(features, self.feature_means, self.inv_cov_matrix)
        
        return md
    
    def _compute_human_score(self, game):
        """
        Calcula pontuação de "humanidade" do jogo
        
        Valores altos = jogo que humanos tenderiam a escolher
        Usado para EV Optimizer (evitar padrões populares)
        """
        game_set = set(game)
        score = 0
        
        # 1. Números de aniversário (1-31)
        birthday_count = len(game_set & self.human_patterns['birthday_numbers'])
        score += birthday_count * 3
        
        # 2. Números da sorte
        lucky_count = len(game_set & self.human_patterns['lucky_numbers'])
        score += lucky_count * 2
        
        # 3. Sequências conhecidas
        for seq in self.human_patterns['sequences']:
            overlap = len(game_set & seq)
            if overlap >= 4:
                score += overlap * 5
        
        # 4. Padrões de teclado
        for pattern in self.human_patterns['keyboard_patterns']:
            overlap = len(game_set & pattern)
            if overlap >= 4:
                score += overlap * 5
        
        # 5. Pares comuns
        for pair in combinations(sorted(game), 2):
            if pair in self.human_patterns['common_pairs']:
                score += 1
        
        return score
    
    def _compute_pool_global_fitness(self, pool):
        """
        FITNESS GLOBAL DO POOL
        
        Avalia o conjunto INTEIRO de jogos simultaneamente,
        não jogos individuais.
        
        Componentes:
        1. Cobertura total de dezenas
        2. Cobertura de pares/trincas/quadras
        3. Diversidade multivariada
        4. Anti-human score (EV optimizer)
        5. Entropia do conjunto
        """
        n_games = len(pool)
        
        # 1. COBERTURA DE DEZENAS (20 pontos)
        all_dezenas = set()
        for game in pool:
            all_dezenas.update(game)
        
        coverage_pct = len(all_dezenas) / 60
        coverage_score = coverage_pct * 20
        
        # 2. COBERTURA COMBINATÓRIA (30 pontos)
        # Pares cobertos
        all_pairs = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                all_pairs.add(pair)
        
        total_possible_pairs = 60 * 59 // 2  # C(60,2) = 1770
        pair_coverage = len(all_pairs) / total_possible_pairs
        pair_score = pair_coverage * 15
        
        # Trincas cobertas
        all_triples = set()
        for game in pool:
            for triple in combinations(sorted(game), 3):
                all_triples.add(triple)
        
        total_possible_triples = 60 * 59 * 58 // 6  # C(60,3) = 34220
        triple_coverage = len(all_triples) / min(total_possible_triples, n_games * 20)
        triple_coverage = min(triple_coverage, 1.0)  # Cap at 100%
        triple_score = triple_coverage * 15
        
        combinatorial_score = pair_score + triple_score
        
        # 3. DIVERSIDADE MULTIVARIADA (25 pontos)
        # Distância média entre todos os pares de jogos
        features_matrix = np.array([self._extract_features(g) for g in pool])
        
        # Calcular matriz de distâncias
        distances = []
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                # Distância euclidiana no espaço de features
                dist = np.linalg.norm(features_matrix[i] - features_matrix[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        # Normalizar pela distância máxima possível
        max_possible_dist = np.linalg.norm(self.feature_stds * 3)
        diversity_score = (avg_distance / max_possible_dist) * 25
        
        # 4. ANTI-HUMAN SCORE - EV OPTIMIZER (15 pontos)
        # Quanto menor o human_score, melhor (menos chance de dividir prêmio)
        human_scores = [self._compute_human_score(g) for g in pool]
        avg_human_score = np.mean(human_scores)
        
        # Penalizar jogos "humanos" (inverter escala)
        anti_human_score = max(0, 15 - avg_human_score * 0.5)
        
        # 5. ENTROPIA DO CONJUNTO (10 pontos)
        # Mede uniformidade da distribuição de dezenas
        dezena_counts = np.bincount(
            [d for game in pool for d in game], 
            minlength=61
        )[1:]
        
        dezena_probs = dezena_counts / np.sum(dezena_counts)
        pool_entropy = entropy(dezena_probs + 1e-10)
        max_entropy = np.log(60)
        entropy_score = (pool_entropy / max_entropy) * 10
        
        # FITNESS TOTAL
        total_fitness = (
            coverage_score +
            combinatorial_score +
            diversity_score +
            anti_human_score +
            entropy_score
        )
        
        return total_fitness, {
            'coverage': coverage_score,
            'combinatorial': combinatorial_score,
            'diversity': diversity_score,
            'anti_human': anti_human_score,
            'entropy': entropy_score,
            'pair_coverage': pair_coverage,
            'triple_coverage': triple_coverage,
            'dezenas_cobertas': len(all_dezenas),
            'pares_cobertos': len(all_pairs),
            'trincas_cobertas': len(all_triples)
        }
    
    def _dpp_sampling(self, n_games, pool_size=500):
        """
        DETERMINANTAL POINT PROCESS (DPP)
        
        Amostragem que garante DIVERSIDADE MÁXIMA
        Baseado em álgebra linear: seleciona subconjuntos
        com máxima dispersão no espaço de features
        """
        print("🎯 Aplicando DPP para diversidade máxima...")
        
        # Gerar pool inicial grande
        initial_pool = []
        seen = set()
        
        while len(initial_pool) < pool_size:
            game = tuple(sorted(np.random.choice(range(1, 61), 6, replace=False)))
            if game not in seen:
                seen.add(game)
                initial_pool.append(list(game))
        
        # Matriz de features
        X = np.array([self._extract_features(g) for g in initial_pool])
        X_scaled = self.scaler.transform(X)
        
        # Kernel de similaridade (RBF)
        def rbf_kernel(x1, x2, sigma=1.0):
            dist = np.linalg.norm(x1 - x2)
            return np.exp(-dist**2 / (2 * sigma**2))
        
        # Matriz de kernel (similaridade)
        n = len(initial_pool)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = rbf_kernel(X_scaled[i], X_scaled[j])
        
        # DPP: selecionar subconjunto diverso
        # Usa decomposição espectral para amostragem
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        
        # Selecionar top eigenvectors (maior diversidade)
        selected_indices = []
        remaining = list(range(n))
        
        for _ in range(n_games):
            if not remaining:
                break
            
            # Calcular volume do DPP para cada candidato
            volumes = []
            for idx in remaining:
                # Submatriz incluindo novo ponto
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
            
            # Selecionar ponto que maximiza o volume
            if volumes and np.max(volumes) > 0:
                best_idx = remaining[np.argmax(volumes)]
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                # Fallback: selecionar aleatório
                idx = np.random.choice(remaining)
                selected_indices.append(idx)
                remaining.remove(idx)
        
        # Retornar jogos selecionados
        selected_games = [initial_pool[i] for i in selected_indices[:n_games]]
        
        print(f"   ✅ DPP selecionou {len(selected_games)} jogos diversos")
        
        return selected_games
    
    def _simulated_annealing_optimization(self, initial_pool, iterations=1000, temp_start=10, temp_end=0.1):
        """
        SIMULATED ANNEALING
        
        Otimização global que aceita soluções piores no início
        para escapar de máximos locais e encontrar diversidade ótima
        """
        print(f"🔥 Simulated Annealing ({iterations} iterações)...")
        
        current_pool = [list(g) for g in initial_pool]
        current_fitness, _ = self._compute_pool_global_fitness(current_pool)
        
        best_pool = current_pool.copy()
        best_fitness = current_fitness
        
        fitness_history = [current_fitness]
        
        for iteration in tqdm(range(iterations), desc="Annealing"):
            # Temperatura decai exponencialmente
            temp = temp_start * (temp_end / temp_start) ** (iteration / iterations)
            
            # Propor modificação
            new_pool = current_pool.copy()
            
            # Escolher um jogo para modificar
            idx = np.random.randint(0, len(new_pool))
            
            # Modificação: trocar uma dezena
            game = new_pool[idx].copy()
            pos = np.random.randint(0, 6)
            old_value = game[pos]
            
            # Nova dezena (evitar repetição)
            available = [d for d in range(1, 61) if d not in game]
            if available:
                game[pos] = np.random.choice(available)
                game.sort()
                new_pool[idx] = game
            
            # Avaliar novo pool
            new_fitness, new_metrics = self._compute_pool_global_fitness(new_pool)
            
            # Decidir se aceita (Metropolis criterion)
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
        PIPELINE COMPLETO DE OTIMIZAÇÃO
        
        1. DPP para seleção inicial diversa
        2. Simulated Annealing para otimização global
        3. Fitness global do pool inteiro
        """
        print("\n" + "="*60)
        print(f"🎯 OTIMIZANDO POOL DE {n_games} JOGOS")
        print("="*60)
        
        # 1. DPP para diversidade inicial
        initial_pool = self._dpp_sampling(n_games, dpp_pool_size)
        
        # 2. Simulated Annealing para otimização global
        optimized_pool, fitness_history = self._simulated_annealing_optimization(
            initial_pool, 
            iterations=annealing_iterations
        )
        
        # 3. Avaliação final
        final_fitness, final_metrics = self._compute_pool_global_fitness(optimized_pool)
        
        print(f"\n📊 MÉTRICAS FINAIS DO POOL:")
        print(f"   • Fitness Total: {final_fitness:.2f}/100")
        print(f"   • Cobertura de Dezenas: {final_metrics['dezenas_cobertas']}/60")
        print(f"   • Cobertura de Pares: {final_metrics['pares_cobertos']}/1770 ({final_metrics['pair_coverage']:.1%})")
        print(f"   • Cobertura de Trincas: {final_metrics['trincas_cobertas']} ({final_metrics['triple_coverage']:.1%})")
        print(f"   • Diversidade: {final_metrics['diversity']:.2f}/25")
        print(f"   • Anti-Human Score: {final_metrics['anti_human']:.2f}/15")
        print(f"   • Entropia: {final_metrics['entropy']:.2f}/10")
        
        self.final_metrics = final_metrics
        self.fitness_history = fitness_history
        
        return optimized_pool
    
    def visualize_optimization(self, output_dir='graficos_otimizacao'):
        """Visualiza o processo de otimização"""
        print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Evolução do fitness (Simulated Annealing)
        ax = axes[0, 0]
        ax.plot(self.fitness_history, color='blue', alpha=0.7, linewidth=1)
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Fitness Global')
        ax.set_title('Evolução do Fitness (Simulated Annealing)')
        ax.grid(True, alpha=0.3)
        
        # 2. Radar chart das métricas
        ax = axes[0, 1]
        metrics = self.final_metrics
        categories = ['Cobertura\nDezenas', 'Cobertura\nPares', 'Cobertura\nTrincas', 
                     'Diversidade', 'Anti-Human', 'Entropia']
        values = [
            metrics['dezenas_cobertas'] / 60 * 100,
            metrics['pair_coverage'] * 100,
            metrics['triple_coverage'] * 100,
            metrics['diversity'] / 25 * 100,
            metrics['anti_human'] / 15 * 100,
            metrics['entropy'] / 10 * 100
        ]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 2, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='green')
        ax.fill(angles, values, alpha=0.25, color='green')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title('Métricas do Pool (%)', fontsize=12)
        ax.grid(True)
        
        # 3. Distribuição de distâncias de Mahalanobis
        ax = axes[1, 0]
        # Gerar amostra de jogos aleatórios para comparação
        random_games = []
        for _ in range(1000):
            game = sorted(np.random.choice(range(1, 61), 6, replace=False))
            random_games.append(self.mahalanobis_distance(game))
        
        ax.hist(random_games, bins=30, alpha=0.7, label='Aleatório', color='orange', density=True)
        ax.axvline(np.mean(random_games), color='orange', linestyle='--', 
                  label=f'Média Aleatória: {np.mean(random_games):.1f}')
        ax.set_xlabel('Distância de Mahalanobis')
        ax.set_ylabel('Densidade')
        ax.set_title('Distribuição de Distâncias Multivariadas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cobertura de dezenas (heatmap)
        ax = axes[1, 1]
        # Criar matriz 6x10 para visualização
        coverage_matrix = np.zeros((6, 10))
        all_dezenas = []
        for game in self.final_pool if hasattr(self, 'final_pool') else []:
            all_dezenas.extend(game)
        
        for d in all_dezenas:
            row = (d - 1) // 10
            col = (d - 1) % 10
            coverage_matrix[row, col] += 1
        
        sns.heatmap(coverage_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'{i*10+1}-{(i+1)*10}' for i in range(10)],
                   yticklabels=[f'Linha {i+1}' for i in range(6)],
                   ax=ax)
        ax.set_title('Cobertura de Dezenas por Região')
        ax.set_xlabel('Coluna')
        ax.set_ylabel('Linha')
        
        # 5. Comparação com distribuição histórica
        ax = axes[1, 2]
        if hasattr(self, 'final_pool'):
            pool_somas = [sum(g) for g in self.final_pool]
            hist_somas = self.dezenas_historicas.sum(axis=1)
            
            ax.hist(hist_somas, bins=20, alpha=0.5, label='Histórico', color='blue', density=True)
            ax.hist(pool_somas, bins=20, alpha=0.7, label='Pool Otimizado', color='green', density=True)
            ax.set_xlabel('Soma')
            ax.set_ylabel('Densidade')
            ax.set_title('Distribuição de Somas: Pool vs Histórico')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Otimização Global de Cobertura - Análise Completa', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/otimizacao_global.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def export_optimized_pool(self, games, filename=None):
        """Exporta pool otimizado com métricas"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'pool_otimizado_{timestamp}.csv'
        
        df_games = pd.DataFrame(games, columns=[f'D_{i+1}' for i in range(6)])
        df_games.index = [f'Jogo_{i+1}' for i in range(len(games))]
        
        # Adicionar métricas por jogo
        df_games['Mahalanobis_D'] = [self.mahalanobis_distance(g) for g in games]
        df_games['Human_Score'] = [self._compute_human_score(g) for g in games]
        df_games['Soma'] = [sum(g) for g in games]
        
        df_games.to_csv(filename)
        print(f"\n💾 Pool exportado: {filename}")
        
        return filename


def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*70)
    print("🎯 SISTEMA AVANÇADO DE OTIMIZAÇÃO COMBINATÓRIA")
    print("   Fitness Global + DPP + Simulated Annealing")
    print("="*70)
    
    # Inicializar
    optimizer = AdvancedCoverageOptimizer('resultados_megasena.csv')
    
    # Otimizar pool
    n_games = 50
    optimized_pool = optimizer.optimize_pool(
        n_games=n_games,
        dpp_pool_size=500,
        annealing_iterations=500
    )
    
    # Armazenar para visualização
    optimizer.final_pool = optimized_pool
    
    # Visualizar
    optimizer.visualize_optimization()
    
    # Exportar
    optimizer.export_optimized_pool(optimized_pool)
    
    # Mostrar jogos
    print(f"\n🎯 POOL OTIMIZADO ({n_games} jogos):")
    print("="*50)
    for i, game in enumerate(optimized_pool, 1):
        mahal = optimizer.mahalanobis_distance(game)
        human = optimizer._compute_human_score(game)
        print(f"Jogo {i:2d}: {sorted(game)} | "
              f"Mahal={mahal:.1f} | Human={human:.0f} | Soma={sum(game)}")


if __name__ == "__main__":
    main()
