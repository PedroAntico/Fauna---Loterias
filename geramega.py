#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR OTIMIZADO DE COBERTURA PROBABILÍSTICA - MEGA-SENA
=========================================================
Versão 1.0 - Otimização Combinatória com Restrições Estatísticas

OBJETIVO:
✅ Maximizar cobertura do espaço amostral
✅ Minimizar sobreposição entre bilhetes
✅ Manter aderência às distribuições históricas
✅ Otimizar diversidade estrutural
✅ Aumentar eficiência relativa das apostas

PRINCÍPIO MATEMÁTICO:
Não prevê dezenas, mas otimiza a cobertura do espaço de possibilidades
dentro de restrições estatísticas historicamente observadas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean, hamming, cosine
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

class CoverageOptimizer:
    """
    Otimizador de Cobertura Probabilística para Mega-Sena
    
    Princípios:
    1. MAXIMIZAR distância entre bilhetes (diversidade)
    2. MINIMIZAR sobreposição de dezenas
    3. COBRIR máximo de padrões estruturais
    4. MANTER estatísticas historicamente típicas
    5. EVITAR padrões humanos comuns (vieses cognitivos)
    """
    
    def __init__(self, historical_csv='resultados_megasena.csv'):
        """
        Inicializa o otimizador com dados históricos
        
        Args:
            historical_csv: Caminho para arquivo CSV com resultados históricos
        """
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Conjuntos matemáticos
        self.fibonacci = self._gen_fibonacci(60)
        self.primes = self._gen_primes(60)
        
        # Estatísticas de referência
        self.reference_stats = {}
        
        # Carregar e analisar dados históricos
        self._load_historical_data()
        self._compute_reference_statistics()
        
        print("✅ Otimizador de Cobertura inicializado!")
        print(f"📊 Referência: {len(self.df)} concursos históricos analisados")
    
    def _gen_fibonacci(self, limit):
        """Gera números Fibonacci"""
        fib = [0, 1]
        while fib[-1] <= limit:
            fib.append(fib[-1] + fib[-2])
        return set(fib[2:])
    
    def _gen_primes(self, limit):
        """Gera números primos"""
        return {n for n in range(2, limit+1) 
                if all(n % i != 0 for i in range(2, int(n**0.5)+1))}
    
    def _load_historical_data(self):
        """Carrega dados históricos da Mega-Sena"""
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
    
    def _compute_reference_statistics(self):
        """
        Calcula estatísticas de referência dos dados históricos
        Estas serão usadas como restrições para o otimizador
        """
        print("📊 Calculando estatísticas de referência...")
        
        dezenas = self.dezenas_historicas
        
        # 1. Estatísticas básicas
        self.reference_stats['soma'] = {
            'mean': np.mean(dezenas.sum(axis=1)),
            'std': np.std(dezenas.sum(axis=1)),
            'min': np.min(dezenas.sum(axis=1)),
            'max': np.max(dezenas.sum(axis=1)),
            'q25': np.percentile(dezenas.sum(axis=1), 25),
            'q75': np.percentile(dezenas.sum(axis=1), 75)
        }
        
        # 2. Distribuição de pares
        pares_historicos = np.sum(dezenas % 2 == 0, axis=1)
        self.reference_stats['pares'] = {
            'distribution': np.bincount(pares_historicos, minlength=7) / len(pares_historicos),
            'mean': np.mean(pares_historicos),
            'std': np.std(pares_historicos)
        }
        
        # 3. Distribuição de primos
        primos_historicos = np.sum(np.isin(dezenas, list(self.primes)), axis=1)
        self.reference_stats['primos'] = {
            'distribution': np.bincount(primos_historicos, minlength=7) / len(primos_historicos),
            'mean': np.mean(primos_historicos),
            'std': np.std(primos_historicos)
        }
        
        # 4. Frequência individual das dezenas
        freq = np.bincount(dezenas.flatten(), minlength=61)[1:]
        self.reference_stats['frequencia'] = freq / len(dezenas)
        
        # 5. Distribuição por quadrantes
        quadrantes = np.zeros((len(dezenas), 4))
        quadrantes[:, 0] = np.sum((dezenas >= 1) & (dezenas <= 15), axis=1)  # Q1
        quadrantes[:, 1] = np.sum((dezenas >= 16) & (dezenas <= 30), axis=1)  # Q2
        quadrantes[:, 2] = np.sum((dezenas >= 31) & (dezenas <= 45), axis=1)  # Q3
        quadrantes[:, 3] = np.sum((dezenas >= 46) & (dezenas <= 60), axis=1)  # Q4
        
        self.reference_stats['quadrantes'] = {
            'mean': np.mean(quadrantes, axis=0),
            'std': np.std(quadrantes, axis=0)
        }
        
        # 6. Amplitude
        amplitudes = dezenas.max(axis=1) - dezenas.min(axis=1)
        self.reference_stats['amplitude'] = {
            'mean': np.mean(amplitudes),
            'std': np.std(amplitudes)
        }
        
        # 7. Média de distância entre dezenas consecutivas
        distancias = np.array([
            np.mean([sorted(d)[i+1] - sorted(d)[i] for i in range(5)])
            for d in dezenas
        ])
        self.reference_stats['distancia_media'] = {
            'mean': np.mean(distancias),
            'std': np.std(distancias)
        }
        
        # 8. Fibonacci
        fib_historicos = np.sum(np.isin(dezenas, list(self.fibonacci)), axis=1)
        self.reference_stats['fibonacci'] = {
            'mean': np.mean(fib_historicos),
            'std': np.std(fib_historicos)
        }
        
        print("   ✅ Estatísticas de referência calculadas")
    
    def _game_to_features(self, game):
        """
        Converte um jogo em vetor de features para avaliação
        
        Args:
            game: Lista/array com 6 dezenas
            
        Returns:
            dict: Features estruturais do jogo
        """
        game = np.array(sorted(game))
        
        return {
            'soma': np.sum(game),
            'pares': np.sum(game % 2 == 0),
            'impares': np.sum(game % 2 != 0),
            'primos': np.sum(np.isin(game, list(self.primes))),
            'fibonacci': np.sum(np.isin(game, list(self.fibonacci))),
            'amplitude': game.max() - game.min(),
            'q1': np.sum((game >= 1) & (game <= 15)),
            'q2': np.sum((game >= 16) & (game <= 30)),
            'q3': np.sum((game >= 31) & (game <= 45)),
            'q4': np.sum((game >= 46) & (game <= 60)),
            'distancia_media': np.mean([game[i+1] - game[i] for i in range(5)]),
            'vector': game,
            'binary_vector': np.eye(60)[game - 1].sum(axis=0)
        }
    
    def _compute_game_fitness(self, game, pool_features=None):
        """
        Calcula o fitness de um jogo individual
        
        O fitness mede quão bem o jogo:
        1. Se adequa às distribuições históricas
        2. Se distancia de outros jogos (diversidade)
        3. Cobre padrões estruturais importantes
        
        Args:
            game: Lista com 6 dezenas
            pool_features: Lista de features dos jogos já selecionados
            
        Returns:
            float: Score de fitness (maior = melhor)
        """
        features = self._game_to_features(game)
        score = 0.0
        
        # 1. ADERÊNCIA ÀS DISTRIBUIÇÕES HISTÓRICAS (40% do score)
        
        # Soma dentro do intervalo interquartil
        soma_ref = self.reference_stats['soma']
        soma_z = abs(features['soma'] - soma_ref['mean']) / soma_ref['std'] if soma_ref['std'] > 0 else 0
        score += 15 * np.exp(-soma_z)  # Bônus decai exponencialmente
        
        # Pares próximos à média histórica
        pares_ref = self.reference_stats['pares']
        pares_z = abs(features['pares'] - pares_ref['mean']) / pares_ref['std'] if pares_ref['std'] > 0 else 0
        score += 10 * np.exp(-pares_z)
        
        # Primos dentro do esperado
        primos_ref = self.reference_stats['primos']
        primos_z = abs(features['primos'] - primos_ref['mean']) / primos_ref['std'] if primos_ref['std'] > 0 else 0
        score += 8 * np.exp(-primos_z)
        
        # Amplitude típica
        amp_ref = self.reference_stats['amplitude']
        amp_z = abs(features['amplitude'] - amp_ref['mean']) / amp_ref['std'] if amp_ref['std'] > 0 else 0
        score += 7 * np.exp(-amp_z)
        
        # 2. COBERTURA DE QUADRANTES (20% do score)
        quadrantes = np.array([features['q1'], features['q2'], features['q3'], features['q4']])
        q_ref = self.reference_stats['quadrantes']['mean']
        
        # Penalizar quadrantes vazios
        score += np.sum(quadrantes > 0) * 3
        
        # Recompensar distribuição balanceada
        q_balance = 1 - np.std(quadrantes) / np.max(quadrantes) if np.max(quadrantes) > 0 else 0
        score += q_balance * 5
        
        # 3. DIVERSIDADE EM RELAÇÃO AO POOL (30% do score)
        if pool_features and len(pool_features) > 0:
            # Distância média para os outros jogos
            distances = []
            for pf in pool_features:
                # Distância de Hamming (dezenas diferentes)
                hamming_dist = hamming(features['binary_vector'], pf['binary_vector'])
                distances.append(hamming_dist)
            
            if distances:
                avg_distance = np.mean(distances)
                score += avg_distance * 20  # Mais distante = melhor
        
        # 4. PENALIDADES POR PADRÕES INDESEJADOS (10% do score)
        
        # Penalizar jogos muito ordenados (ex: 1,2,3,4,5,6)
        sorted_game = sorted(game)
        consecutivas = sum(1 for i in range(5) if sorted_game[i+1] - sorted_game[i] == 1)
        score -= consecutivas * 5
        
        # Penalizar todos números baixos ou todos altos
        if features['q4'] == 6 or features['q1'] == 6:
            score -= 30
        
        # Penalizar Fibonacci excessivo
        if features['fibonacci'] >= 4:
            score -= 10
        
        # 5. BÔNUS POR PADRÕES FAVORÁVEIS
        
        # Recompensar distância média saudável entre dezenas
        if 7 <= features['distancia_media'] <= 13:
            score += 5
        
        # Recompensar mistura de quadrantes
        unique_quadrantes = sum(1 for q in quadrantes if q > 0)
        if unique_quadrantes >= 3:
            score += 5
        
        return score
    
    def generate_optimized_pool(self, n_games=50, population_size=500, generations=200):
        """
        GERA POOL OTIMIZADO DE JOGOS
        
        Usa algoritmo genético com:
        - Fitness multi-objetivo
        - Diversidade forçada
        - Restrições estatísticas
        
        Args:
            n_games: Número de jogos a gerar
            population_size: Tamanho da população no algoritmo genético
            generations: Número de gerações
            
        Returns:
            list: Lista de jogos otimizados
        """
        print(f"\n🧬 GERANDO POOL OTIMIZADO DE {n_games} JOGOS...")
        print(f"   População: {population_size} | Gerações: {generations}")
        
        # Pool final de jogos selecionados
        selected_games = []
        selected_features = []
        
        # Para cada jogo a ser gerado
        for game_idx in tqdm(range(n_games), desc="Otimizando jogos"):
            
            # Inicializar população aleatória
            population = []
            seen = set()
            
            while len(population) < population_size:
                game = tuple(sorted(np.random.choice(range(1, 61), 6, replace=False)))
                if game not in seen:
                    seen.add(game)
                    population.append(list(game))
            
            # Evoluir população
            for gen in range(generations):
                # Calcular fitness para cada indivíduo
                fitness_scores = []
                for game in population:
                    fitness = self._compute_game_fitness(game, selected_features)
                    fitness_scores.append(fitness)
                
                # Selecionar elite (top 20%)
                elite_size = population_size // 5
                elite_indices = np.argsort(fitness_scores)[-elite_size:]
                elite = [population[i] for i in elite_indices]
                
                # Nova população começa com elite
                new_population = elite.copy()
                
                # Crossover e mutação para preencher o resto
                while len(new_population) < population_size:
                    # Selecionar dois pais da elite
                    parent1 = elite[np.random.randint(0, len(elite))]
                    parent2 = elite[np.random.randint(0, len(elite))]
                    
                    # Crossover: misturar metades
                    child = list(set(parent1[:3] + parent2[3:]))
                    
                    # Completar se necessário
                    while len(child) < 6:
                        new_num = np.random.randint(1, 61)
                        if new_num not in child:
                            child.append(new_num)
                    
                    child = sorted(child[:6])
                    
                    # Mutação (15% de chance)
                    if np.random.random() < 0.15:
                        # Mutação adaptativa: trocar dezena por vizinha
                        idx = np.random.randint(0, 6)
                        current = child[idx]
                        
                        # Escolher nova dezena próxima
                        candidates = []
                        for delta in [-3, -2, -1, 1, 2, 3]:
                            new_val = current + delta
                            if 1 <= new_val <= 60 and new_val not in child:
                                candidates.append(new_val)
                        
                        if candidates:
                            child[idx] = np.random.choice(candidates)
                            child = sorted(child)
                    
                    # Sangue novo (10% de chance)
                    if np.random.random() < 0.10:
                        child = sorted(np.random.choice(range(1, 61), 6, replace=False))
                    
                    new_population.append(child)
                
                population = new_population
            
            # Selecionar o melhor jogo da geração final
            final_fitness = [self._compute_game_fitness(g, selected_features) for g in population]
            best_idx = np.argmax(final_fitness)
            best_game = population[best_idx]
            
            # Adicionar ao pool
            selected_games.append(best_game)
            selected_features.append(self._game_to_features(best_game))
        
        print(f"\n✅ Pool de {n_games} jogos otimizados gerado!")
        
        # Análise do pool gerado
        self._analyze_pool(selected_games)
        
        return selected_games
    
    def _analyze_pool(self, games):
        """
        Analisa a qualidade do pool gerado
        
        Args:
            games: Lista de jogos
        """
        print("\n📊 ANÁLISE DO POOL GERADO:")
        print("="*50)
        
        features_list = [self._game_to_features(g) for g in games]
        
        # 1. Diversidade interna
        binary_vectors = np.array([f['binary_vector'] for f in features_list])
        distances = []
        for i in range(len(binary_vectors)):
            for j in range(i+1, len(binary_vectors)):
                distances.append(hamming(binary_vectors[i], binary_vectors[j]))
        
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        
        print(f"📏 DISTÂNCIA MÉDIA ENTRE JOGOS: {avg_distance:.4f}")
        print(f"📏 DISTÂNCIA MÍNIMA: {min_distance:.4f}")
        print(f"   (Hamming: 0=idênticos, 1=totalmente diferentes)")
        
        # 2. Cobertura de dezenas
        all_dezenas = []
        for game in games:
            all_dezenas.extend(game)
        
        dezena_coverage = len(set(all_dezenas))
        print(f"\n🎯 COBERTURA DE DEZENAS: {dezena_coverage}/60 ({dezena_coverage/60*100:.1f}%)")
        
        # Frequência das dezenas
        freq = Counter(all_dezenas)
        print(f"📈 Dezenas mais usadas: {freq.most_common(5)}")
        print(f"📉 Dezenas menos usadas: {freq.most_common()[-5:]}")
        
        # 3. Estatísticas estruturais
        somas = [f['soma'] for f in features_list]
        print(f"\n📊 SOMA: média={np.mean(somas):.1f}, std={np.std(somas):.1f}")
        print(f"   Ref histórica: {self.reference_stats['soma']['mean']:.1f} ± {self.reference_stats['soma']['std']:.1f}")
        
        pares = [f['pares'] for f in features_list]
        print(f"📊 PARES: média={np.mean(pares):.2f}")
        print(f"   Ref histórica: {self.reference_stats['pares']['mean']:.2f}")
        
        # 4. Sobreposição
        overlap_count = 0
        for i in range(len(games)):
            for j in range(i+1, len(games)):
                common = len(set(games[i]) & set(games[j]))
                if common >= 4:
                    overlap_count += 1
        
        print(f"\n⚠️  SOBREPOSIÇÃO (≥4 dezenas iguais): {overlap_count} pares")
        
        # 5. Score de qualidade geral
        quality_score = (
            avg_distance * 30 +
            dezena_coverage / 60 * 20 +
            (1 - min_distance) * 20 +
            (1 - overlap_count / max(1, len(games)*(len(games)-1)/2)) * 30
        )
        
        print(f"\n⭐ SCORE DE QUALIDADE: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("   🏆 Excelente cobertura e diversidade!")
        elif quality_score >= 60:
            print("   👍 Boa cobertura, pode ser melhorada")
        else:
            print("   ⚠️  Cobertura subótima, ajuste os parâmetros")

    def compare_with_historical(self, games):
        """
        Compara o pool gerado com concursos históricos reais
        
        Args:
            games: Lista de jogos gerados
        """
        print("\n🔬 COMPARAÇÃO COM DADOS HISTÓRICOS:")
        print("="*50)
        
        features_pool = [self._game_to_features(g) for g in games]
        
        # Comparar distribuições
        metrics = ['soma', 'pares', 'primos', 'amplitude']
        
        for metric in metrics:
            pool_values = [f[metric] for f in features_pool]
            historical_values = []
            
            if metric == 'soma':
                historical_values = self.dezenas_historicas.sum(axis=1)
            elif metric == 'pares':
                historical_values = np.sum(self.dezenas_historicas % 2 == 0, axis=1)
            elif metric == 'primos':
                historical_values = np.sum(np.isin(self.dezenas_historicas, list(self.primes)), axis=1)
            elif metric == 'amplitude':
                historical_values = self.dezenas_historicas.max(axis=1) - self.dezenas_historicas.min(axis=1)
            
            # Teste KS
            ks_stat, ks_pvalue = stats.ks_2samp(pool_values, historical_values)
            
            similarity = "✅ SIMILAR" if ks_pvalue > 0.05 else "⚠️  DIFERENTE"
            print(f"   {metric}: p={ks_pvalue:.4f} {similarity}")
    
    def visualize_pool(self, games, output_dir='graficos_pool'):
        """
        Gera visualizações do pool otimizado
        
        Args:
            games: Lista de jogos gerados
            output_dir: Diretório para salvar gráficos
        """
        print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
        os.makedirs(output_dir, exist_ok=True)
        
        features_list = [self._game_to_features(g) for g in games]
        
        # 1. Mapa de calor de cobertura de dezenas
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cobertura de dezenas
        ax = axes[0, 0]
        all_dezenas = []
        for game in games:
            all_dezenas.extend(game)
        
        freq = np.bincount(all_dezenas, minlength=61)[1:]
        colors = ['red' if f == 0 else 'green' if f >= 3 else 'yellow' for f in freq]
        ax.bar(range(1, 61), freq, color=colors, edgecolor='black')
        ax.set_xlabel('Dezena')
        ax.set_ylabel('Frequência')
        ax.set_title('Cobertura de Dezenas no Pool')
        ax.grid(True, alpha=0.3)
        
        # Distribuição de somas
        ax = axes[0, 1]
        somas = [f['soma'] for f in features_list]
        ax.hist(somas, bins=20, alpha=0.7, color='blue', edgecolor='black', label='Pool')
        ax.axvline(self.reference_stats['soma']['mean'], color='red', linestyle='--', 
                  label=f"Ref: {self.reference_stats['soma']['mean']:.0f}")
        ax.set_xlabel('Soma')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Somas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pares vs Primos
        ax = axes[1, 0]
        pares = [f['pares'] for f in features_list]
        primos = [f['primos'] for f in features_list]
        
        scatter = ax.scatter(pares, primos, c=somas, cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel('Pares')
        ax.set_ylabel('Primos')
        ax.set_title('Distribuição Pares vs Primos')
        plt.colorbar(scatter, ax=ax, label='Soma')
        ax.grid(True, alpha=0.3)
        
        # Distância entre jogos (matriz de similaridade)
        ax = axes[1, 1]
        binary_vectors = np.array([f['binary_vector'] for f in features_list])
        similarity = np.zeros((len(games), len(games)))
        
        for i in range(len(games)):
            for j in range(len(games)):
                similarity[i, j] = 1 - hamming(binary_vectors[i], binary_vectors[j])
        
        sns.heatmap(similarity, cmap='RdYlGn', ax=ax, 
                   xticklabels=5, yticklabels=5,
                   cbar_kws={'label': 'Similaridade'})
        ax.set_title('Matriz de Similaridade entre Jogos')
        ax.set_xlabel('Jogo')
        ax.set_ylabel('Jogo')
        
        plt.suptitle('Análise do Pool Otimizado de Jogos', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analise_pool.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # 2. Gráfico de radar com features estruturais
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Médias das features normalizadas
        features_radar = {
            'Soma Típica': np.mean([abs(f['soma'] - self.reference_stats['soma']['mean']) / 
                                   self.reference_stats['soma']['std'] for f in features_list]),
            'Pares Balanceados': np.mean([abs(f['pares'] - self.reference_stats['pares']['mean']) / 
                                         self.reference_stats['pares']['std'] for f in features_list]),
            'Cobertura Q1': np.mean([f['q1'] for f in features_list]) / 3,
            'Cobertura Q2': np.mean([f['q2'] for f in features_list]) / 3,
            'Cobertura Q3': np.mean([f['q3'] for f in features_list]) / 3,
            'Cobertura Q4': np.mean([f['q4'] for f in features_list]) / 3,
            'Distância Média': np.mean([f['distancia_media'] for f in features_list]) / 15,
            'Diversidade': np.mean([1 - hamming(binary_vectors[i], binary_vectors[j]) 
                                   for i in range(len(games)) for j in range(i+1, len(games))])
        }
        
        categories = list(features_radar.keys())
        values = list(features_radar.values())
        
        # Fechar o polígono
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Perfil Estrutural do Pool', fontsize=14, pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/radar_pool.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def export_games(self, games, filename=None):
        """
        Exporta os jogos gerados para CSV
        
        Args:
            games: Lista de jogos
            filename: Nome do arquivo (opcional)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'jogos_otimizados_{timestamp}.csv'
        
        # Criar DataFrame
        df_games = pd.DataFrame(games, columns=[f'Dezena_{i+1}' for i in range(6)])
        df_games.index = [f'Jogo_{i+1}' for i in range(len(games))]
        df_games.index.name = 'Jogo'
        
        # Adicionar métricas
        features_list = [self._game_to_features(g) for g in games]
        df_games['Soma'] = [f['soma'] for f in features_list]
        df_games['Pares'] = [f['pares'] for f in features_list]
        df_games['Primos'] = [f['primos'] for f in features_list]
        df_games['Amplitude'] = [f['amplitude'] for f in features_list]
        
        # Salvar
        df_games.to_csv(filename)
        print(f"\n💾 Jogos exportados para: {filename}")
        
        return filename
    
    def generate_report(self, games, output_dir='relatorio_pool'):
        """
        Gera relatório completo do pool otimizado
        
        Args:
            games: Lista de jogos
            output_dir: Diretório para salvar relatório
        """
        print(f"\n📄 GERANDO RELATÓRIO COMPLETO...")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Análise quantitativa
        features_list = [self._game_to_features(g) for g in games]
        binary_vectors = np.array([f['binary_vector'] for f in features_list])
        
        # Calcular métricas
        distances = []
        for i in range(len(binary_vectors)):
            for j in range(i+1, len(binary_vectors)):
                distances.append(hamming(binary_vectors[i], binary_vectors[j]))
        
        all_dezenas = []
        for game in games:
            all_dezenas.extend(game)
        
        dezena_coverage = len(set(all_dezenas))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_games': len(games),
            'metrics': {
                'avg_distance': float(np.mean(distances)),
                'min_distance': float(np.min(distances)),
                'dezena_coverage': dezena_coverage,
                'coverage_pct': float(dezena_coverage / 60 * 100),
                'avg_sum': float(np.mean([f['soma'] for f in features_list])),
                'std_sum': float(np.std([f['soma'] for f in features_list])),
                'avg_pares': float(np.mean([f['pares'] for f in features_list])),
                'avg_primos': float(np.mean([f['primos'] for f in features_list]))
            },
            'games': [sorted(g) for g in games]
        }
        
        # Salvar JSON
        json_path = f'{output_dir}/relatorio_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Relatório salvo: {json_path}")
        
        return json_path


def main():
    """
    EXECUÇÃO PRINCIPAL DO OTIMIZADOR DE COBERTURA
    """
    print("="*70)
    print("🎯 GERADOR OTIMIZADO DE COBERTURA PROBABILÍSTICA")
    print("   Mega-Sena - Otimização Combinatória")
    print("="*70)
    
    # 1. Inicializar otimizador
    optimizer = CoverageOptimizer('resultados_megasena.csv')
    
    # 2. Gerar pool otimizado
    print("\n" + "="*50)
    print("GERANDO POOL OTIMIZADO...")
    print("="*50)
    
    n_games = 50  # Número de jogos a gerar
    games = optimizer.generate_optimized_pool(
        n_games=n_games,
        population_size=300,
        generations=150
    )
    
    # 3. Comparar com histórico
    optimizer.compare_with_historical(games)
    
    # 4. Gerar visualizações
    optimizer.visualize_pool(games)
    
    # 5. Exportar jogos
    optimizer.export_games(games)
    
    # 6. Gerar relatório
    optimizer.generate_report(games)
    
    # 7. Mostrar jogos gerados
    print("\n" + "="*50)
    print("🎯 JOGOS OTIMIZADOS GERADOS:")
    print("="*50)
    
    for i, game in enumerate(games, 1):
        features = optimizer._game_to_features(game)
        print(f"\nJogo {i:2d}: {sorted(game)}")
        print(f"        Soma={features['soma']:3d} | Pares={features['pares']} | "
              f"Primos={features['primos']} | Amplitude={features['amplitude']}")
    
    print("\n" + "="*70)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("📁 Resultados salvos em:")
    print("   • graficos_pool/ - Visualizações")
    print("   • relatorio_pool/ - Relatório JSON")
    print("   • jogos_otimizados_*.csv - Jogos exportados")
    print("="*70)
    
    print("\n💡 PRINCÍPIOS DO OTIMIZADOR:")
    print("   1. Maximiza distância entre bilhetes")
    print("   2. Cobre máximo de dezenas diferentes")
    print("   3. Mantém estatísticas historicamente típicas")
    print("   4. Evita padrões humanos comuns")
    print("   5. Otimiza diversidade combinatória")
    
    print("\n⚠️  DISCLAIMER:")
    print("   Este sistema NÃO prevê resultados futuros.")
    print("   Apenas otimiza a cobertura do espaço amostral")
    print("   dentro de restrições estatísticas observadas.")


if __name__ == "__main__":
    main()
