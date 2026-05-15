#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA AVANÇADO DE ANÁLISE ESTRUTURAL DA MEGA-SENA
===================================================
Versão 3.0 - Análise Vetorial, Markoviana e de Anomalias

Autor: Cientista de Dados Estatístico
Abordagem: Fingerprint estrutural + Cadeias de Markov + Detecção de Anomalias
Objetivo: Separar coincidência visual de comportamento recorrente REAL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, entropy, chi2_contingency
from scipy.spatial.distance import euclidean, hamming, cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter, defaultdict
from itertools import combinations, product
from datetime import datetime
import warnings
import os
import json
from tqdm import tqdm
import hashlib

warnings.filterwarnings('ignore')

# Configurações de alta qualidade
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

class MegaSenaStructuralAnalyzer:
    """
    Analisador estrutural avançado que vai MUITO além de soma
    Usa fingerprints vetoriais, cadeias de Markov e detecção de anomalias
    """
    
    def __init__(self, csv_path='resultados_megasena.csv'):
        """Inicialização com carregamento de dados"""
        self.csv_path = csv_path
        self.df = None
        self.dezenas = None
        self.universo_aleatorio = None
        self.markov_states = None
        self.anomalies = None
        
        # Conjuntos matemáticos
        self.fibonacci = self._generate_fibonacci(60)
        self.primes = self._generate_primes(60)
        self.quadrados = {1, 4, 9, 16, 25, 36, 49}
        self.triangulares = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55}
        
        self.load_and_prepare_data()
        
    def _generate_fibonacci(self, limit):
        """Gera Fibonacci até limite"""
        fib = [0, 1]
        while fib[-1] <= limit:
            fib.append(fib[-1] + fib[-2])
        return set(fib[2:])
    
    def _generate_primes(self, limit):
        """Gera primos até limite"""
        primes = set()
        for num in range(2, limit + 1):
            if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                primes.add(num)
        return primes
    
    def load_and_prepare_data(self):
        """Carrega e prepara dados com engenharia de features avançada"""
        print("🔬 CARREGANDO E PREPARANDO DADOS COM FEATURES ESTRUTURAIS...")
        
        # Carregar CSV
        try:
            self.df = pd.read_csv(self.csv_path, sep=';', encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
            except:
                self.df = pd.read_csv(self.csv_path, sep=';', encoding='latin-1')
        
        # Padronizar colunas
        self.df.columns = ['concurso', 'data', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
        
        # Extrair dezenas
        self.dezenas = self.df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6']].values
        
        print(f"📊 Dados carregados: {len(self.df)} concursos")
        
        # Criar TODAS as features estruturais
        self._create_structural_fingerprints()
        self._create_vector_representations()
        self._create_markov_states()
        
        print("✅ Features estruturais criadas com sucesso!")
        
    def _create_structural_fingerprints(self):
        """
        CRIA FINGERPRINTS ESTRUTURAIS COMPLETOS
        Vai MUITO além de soma - captura a ESTRUTURA do jogo
        """
        print("🎯 Criando fingerprints estruturais...")
        
        # 1. Features básicas melhoradas
        self.df['soma'] = self.dezenas.sum(axis=1)
        self.df['produto_escalar'] = np.prod(self.dezenas, axis=1) % 1000000
        self.df['amplitude'] = self.dezenas.max(axis=1) - self.dezenas.min(axis=1)
        self.df['media'] = self.dezenas.mean(axis=1)
        self.df['mediana'] = np.median(self.dezenas, axis=1)
        self.df['desvio_padrao'] = self.dezenas.std(axis=1)
        
        # 2. Contagens categóricas
        self.df['pares'] = np.sum(self.dezenas % 2 == 0, axis=1)
        self.df['primos'] = np.sum(np.isin(self.dezenas, list(self.primes)), axis=1)
        self.df['fibonacci'] = np.sum(np.isin(self.dezenas, list(self.fibonacci)), axis=1)
        self.df['quadrados'] = np.sum(np.isin(self.dezenas, list(self.quadrados)), axis=1)
        self.df['triangulares'] = np.sum(np.isin(self.dezenas, list(self.triangulares)), axis=1)
        self.df['multiplos_3'] = np.sum(self.dezenas % 3 == 0, axis=1)
        self.df['multiplos_5'] = np.sum(self.dezenas % 5 == 0, axis=1)
        
        # 3. Distribuição espacial AVANÇADA
        # Moldura, centro, linhas, colunas
        moldura = {1,2,3,4,5,6,7,8,9,10,19,20,28,29,37,38,46,47,55,56,57,58,59,60}
        self.df['moldura'] = np.sum(np.isin(self.dezenas, list(moldura)), axis=1)
        self.df['centro'] = 6 - self.df['moldura']
        
        # Distribuição por quadrantes
        q1 = set(range(1, 16))   # 1-15
        q2 = set(range(16, 31))  # 16-30
        q3 = set(range(31, 46))  # 31-45
        q4 = set(range(46, 61))  # 46-60
        
        self.df['q1'] = np.sum(np.isin(self.dezenas, list(q1)), axis=1)
        self.df['q2'] = np.sum(np.isin(self.dezenas, list(q2)), axis=1)
        self.df['q3'] = np.sum(np.isin(self.dezenas, list(q3)), axis=1)
        self.df['q4'] = np.sum(np.isin(self.dezenas, list(q4)), axis=1)
        
        # 4. Métricas de dispersão
        self.df['distancia_media'] = [np.mean([dezenas[i+1] - dezenas[i] 
                                              for i in range(len(dezenas)-1)]) 
                                     for dezenas in self.dezenas]
        self.df['distancia_maxima'] = [max([dezenas[i+1] - dezenas[i] 
                                           for i in range(len(dezenas)-1)]) 
                                      for dezenas in self.dezenas]
        self.df['distancia_minima'] = [min([dezenas[i+1] - dezenas[i] 
                                           for i in range(len(dezenas)-1)]) 
                                      for dezenas in self.dezenas]
        
        # 5. Sequências e padrões
        self.df['consecutivas'] = self._count_consecutive_sequences()
        self.df['repetidas_anterior'] = self._count_repeats_from_previous()
        self.df['numeros_espelho'] = self._count_mirror_numbers()
        
        # 6. Características modulares AVANÇADAS
        for mod in [3, 5, 7, 9, 11, 13, 17, 19]:
            self.df[f'soma_mod_{mod}'] = self.df['soma'] % mod
            # Distribuição modular das dezenas
            self.df[f'dez_mod_{mod}_unicos'] = [len(set(d % mod for d in dezenas)) 
                                                for dezenas in self.dezenas]
        
        # 7. Features de equilíbrio
        self.df['indice_gini'] = [self._calculate_gini(dezenas) for dezenas in self.dezenas]
        self.df['entropia_local'] = [self._calculate_local_entropy(dezenas) for dezenas in self.dezenas]
        
        print(f"✅ {len(self.df.columns)} features estruturais criadas")
    
    def _create_vector_representations(self):
        """
        CRIA REPRESENTAÇÕES VETORIAIS GEOMÉTRICAS
        Cada concurso vira um vetor de 60 dimensões
        """
        print("📐 Criando representações vetoriais...")
        
        # Vetor binário (60 dimensões)
        self.vectors_binary = np.zeros((len(self.dezenas), 60))
        for i, dezenas in enumerate(self.dezenas):
            self.vectors_binary[i, dezenas-1] = 1
        
        # Vetor de frequência acumulada
        self.vectors_frequency = np.zeros((len(self.dezenas), 60))
        freq_cumulative = np.zeros(60)
        for i, dezenas in enumerate(self.dezenas):
            freq_cumulative[dezenas-1] += 1
            self.vectors_frequency[i] = freq_cumulative.copy()
        
        # Calcular matriz de similaridade coseno
        self.cosine_similarity_matrix = np.zeros((len(self.dezenas), len(self.dezenas)))
        
        # Calcular distâncias entre concursos consecutivos
        self.distances_euclidean = []
        self.distances_hamming = []
        self.distances_cosine = []
        
        for i in range(len(self.dezenas) - 1):
            v1 = self.vectors_binary[i]
            v2 = self.vectors_binary[i + 1]
            
            self.distances_euclidean.append(euclidean(v1, v2))
            self.distances_hamming.append(hamming(v1, v2))
            self.distances_cosine.append(cosine(v1, v2))
        
        print("✅ Representações vetoriais criadas")
    
    def _create_markov_states(self):
        """
        CRIA ESTADOS DE MARKOV REAIS E AVANÇADOS
        Estado = (paridade, primos, soma_faixa, moldura_faixa, quadrantes)
        """
        print("🔄 Criando estados de Markov avançados...")
        
        # Definir faixas para características contínuas
        soma_faixas = pd.qcut(self.df['soma'], q=5, labels=['Muito_Baixa', 'Baixa', 'Media', 'Alta', 'Muito_Alta'])
        amplitude_faixas = pd.qcut(self.df['amplitude'], q=3, labels=['Pequena', 'Media', 'Grande'])
        
        # Criar estado composto
        self.df['markov_state'] = (
            self.df['pares'].astype(str) + 'P_' +
            self.df['primos'].astype(str) + 'PR_' +
            soma_faixas.astype(str) + '_' +
            self.df['moldura'].astype(str) + 'M_' +
            amplitude_faixas.astype(str)
        )
        
        # Codificar estados
        le = LabelEncoder()
        self.df['markov_state_id'] = le.fit_transform(self.df['markov_state'])
        self.markov_state_labels = dict(zip(range(len(le.classes_)), le.classes_))
        
        # Criar matriz de transição Markoviana REAL
        n_states = len(le.classes_)
        self.transition_count = np.zeros((n_states, n_states))
        
        states = self.df['markov_state_id'].values
        for i in range(len(states) - 1):
            self.transition_count[states[i], states[i+1]] += 1
        
        # Matriz de probabilidade de transição
        row_sums = self.transition_count.sum(axis=1, keepdims=True)
        self.transition_prob = np.divide(self.transition_count, row_sums, 
                                        where=row_sums!=0, out=np.zeros_like(self.transition_count))
        
        # Entropia condicional H(X_t+1 | X_t)
        self.conditional_entropy = self._calculate_conditional_entropy()
        
        print(f"✅ {n_states} estados de Markov identificados")
        print(f"📊 Entropia condicional: {self.conditional_entropy:.4f}")
    
    def _count_consecutive_sequences(self):
        """Conta sequências consecutivas de forma avançada"""
        sequences = []
        for dezenas in self.dezenas:
            sorted_dez = sorted(dezenas)
            count = 0
            seq_length = 1
            for i in range(len(sorted_dez)-1):
                if sorted_dez[i+1] - sorted_dez[i] == 1:
                    seq_length += 1
                else:
                    if seq_length >= 2:
                        count += seq_length
                    seq_length = 1
            if seq_length >= 2:
                count += seq_length
            sequences.append(count)
        return sequences
    
    def _count_repeats_from_previous(self):
        """Conta repetições do concurso anterior"""
        repeats = [0]
        for i in range(1, len(self.dezenas)):
            count = len(set(self.dezenas[i]) & set(self.dezenas[i-1]))
            repeats.append(count)
        return repeats
    
    def _count_mirror_numbers(self):
        """Conta números espelho (diferença de 10)"""
        mirrors = []
        for dezenas in self.dezenas:
            count = 0
            sorted_dez = sorted(dezenas)
            for i in range(len(sorted_dez)):
                for j in range(i+1, len(sorted_dez)):
                    if abs(sorted_dez[j] - sorted_dez[i]) == 10:
                        count += 1
            mirrors.append(count)
        return mirrors
    
    def _calculate_gini(self, numbers):
        """Calcula índice de Gini para distribuição dos números"""
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        cumulative = np.cumsum(sorted_numbers)
        return (2 * np.sum((np.arange(1, n+1) * sorted_numbers)) - (n + 1) * np.sum(sorted_numbers)) / (n * np.sum(sorted_numbers))
    
    def _calculate_local_entropy(self, numbers):
        """Calcula entropia local da distribuição"""
        hist, _ = np.histogram(numbers, bins=6, range=(1, 60))
        hist = hist / np.sum(hist)
        return entropy(hist + 1e-10)
    
    def _calculate_conditional_entropy(self):
        """Calcula entropia condicional H(Y|X) para estados de Markov"""
        # H(Y|X) = -sum_x P(x) * sum_y P(y|x) * log(P(y|x))
        cond_entropy = 0
        for i in range(len(self.transition_prob)):
            p_x = np.sum(self.transition_count[i]) / np.sum(self.transition_count)
            if p_x > 0:
                p_y_given_x = self.transition_prob[i]
                ent_x = entropy(p_y_given_x[p_y_given_x > 0])
                cond_entropy += p_x * ent_x
        
        return cond_entropy
    
    def generate_random_universe(self, n_simulations=100000):
        """
        GERA UNIVERSO ALEATÓRIO COMPLETO PARA COMPARAÇÃO
        Milhões de jogos aleatórios para testar TODAS as métricas
        """
        print(f"\n🎲 GERANDO UNIVERSO ALEATÓRIO ({n_simulations:,} simulações)...")
        
        # Gerar jogos aleatórios
        random_games = np.zeros((n_simulations, 6), dtype=int)
        for i in tqdm(range(n_simulations), desc="Simulando"):
            random_games[i] = sorted(np.random.choice(range(1, 61), 6, replace=False))
        
        # Calcular todas as métricas para o universo aleatório
        self.universo_aleatorio = {
            'jogos': random_games,
            'soma': random_games.sum(axis=1),
            'pares': np.sum(random_games % 2 == 0, axis=1),
            'primos': np.sum(np.isin(random_games, list(self.primes)), axis=1),
            'fibonacci': np.sum(np.isin(random_games, list(self.fibonacci)), axis=1),
            'amplitude': random_games.max(axis=1) - random_games.min(axis=1),
            'consecutivas': [sum(1 for i in range(5) if sorted(g)[i+1] - sorted(g)[i] == 1) 
                           for g in random_games],
            'distancia_media': [np.mean([sorted(g)[i+1] - sorted(g)[i] for i in range(5)]) 
                               for g in random_games]
        }
        
        print("✅ Universo aleatório gerado!")
        return self.universo_aleatorio
    
    def compare_with_random_universe(self):
        """
        COMPARAÇÃO COMPLETA COM UNIVERSO ALEATÓRIO
        Responde: "O concurso real se comporta igual a um RNG puro?"
        """
        if self.universo_aleatorio is None:
            self.generate_random_universe()
        
        print("\n🔬 COMPARANDO MEGA-SENA REAL vs UNIVERSO ALEATÓRIO...")
        
        self.comparison_results = {}
        
        # Métricas a comparar
        metrics = ['soma', 'pares', 'primos', 'fibonacci', 'amplitude', 'consecutivas', 'distancia_media']
        
        for metric in metrics:
            real_data = self.df[metric].values if metric in self.df.columns else self._calculate_metric(metric)
            random_data = self.universo_aleatorio[metric]
            
            # Teste KS
            ks_stat, ks_pvalue = stats.ks_2samp(real_data, random_data)
            
            # Teste de Mann-Whitney
            mw_stat, mw_pvalue = stats.mannwhitneyu(real_data, random_data, alternative='two-sided')
            
            # Estatísticas descritivas
            self.comparison_results[metric] = {
                'real_mean': np.mean(real_data),
                'random_mean': np.mean(random_data),
                'real_std': np.std(real_data),
                'random_std': np.std(random_data),
                'difference_pct': (np.mean(real_data) - np.mean(random_data)) / np.mean(random_data) * 100,
                'ks_pvalue': ks_pvalue,
                'mw_pvalue': mw_pvalue,
                'is_significant': ks_pvalue < 0.05 or mw_pvalue < 0.05
            }
        
        # Resumo das diferenças significativas
        significant_diffs = {k: v for k, v in self.comparison_results.items() if v['is_significant']}
        
        print(f"\n📊 RESULTADOS DA COMPARAÇÃO:")
        print(f"✅ Métricas com diferenças significativas: {len(significant_diffs)}")
        for metric, results in significant_diffs.items():
            print(f"   • {metric}: Real={results['real_mean']:.2f} vs Random={results['random_mean']:.2f} "
                  f"(dif: {results['difference_pct']:.1f}%, p={min(results['ks_pvalue'], results['mw_pvalue']):.4f})")
        
        if len(significant_diffs) == 0:
            print("🎯 Mega-Sena NÃO apresenta diferenças significativas do aleatório")
        else:
            print("⚠️  Foram encontradas diferenças estatisticamente significativas!")
        
        return self.comparison_results
    
    def detect_anomalies(self):
        """
        DETECÇÃO DE ANOMALIAS ESTRUTURAIS
        Usa Isolation Forest, LOF e DBSCAN
        """
        print("\n🔍 DETECTANDO ANOMALIAS ESTRUTURAIS...")
        
        # Features para detecção de anomalias
        anomaly_features = ['soma', 'amplitude', 'pares', 'primos', 'fibonacci', 
                           'consecutivas', 'distancia_media', 'moldura', 'centro',
                           'indice_gini', 'entropia_local']
        
        X = self.df[anomaly_features].values
        
        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.df['anomaly_if'] = iso_forest.fit_predict(X_scaled)
        
        # 2. Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.05, novelty=False)
        self.df['anomaly_lof'] = lof.fit_predict(X_scaled)
        
        # 3. DBSCAN para outliers
        dbscan = DBSCAN(eps=1.5, min_samples=10)
        clusters = dbscan.fit_predict(X_scaled)
        self.df['anomaly_dbscan'] = (clusters == -1).astype(int) * -1 + (clusters != -1).astype(int)
        
        # Consenso de anomalias
        anomaly_score = (
            (self.df['anomaly_if'] == -1).astype(int) +
            (self.df['anomaly_lof'] == -1).astype(int) +
            (self.df['anomaly_dbscan'] == -1).astype(int)
        )
        
        self.df['anomaly_consensus'] = anomaly_score
        self.anomalies = self.df[self.df['anomaly_consensus'] >= 2].copy()
        
        print(f"✅ Anomalias detectadas:")
        print(f"   • Isolation Forest: {(self.df['anomaly_if'] == -1).sum()} anomalias")
        print(f"   • LOF: {(self.df['anomaly_lof'] == -1).sum()} anomalias")
        print(f"   • DBSCAN: {(self.df['anomaly_dbscan'] == -1).sum()} anomalias")
        print(f"   • Consenso (≥2 métodos): {len(self.anomalies)} anomalias")
        
        if len(self.anomalies) > 0:
            print("\n🔴 CONCURSOS ANÔMALOS (mais de 2 métodos concordam):")
            for idx, row in self.anomalies.iterrows():
                dezenas = [row['b1'], row['b2'], row['b3'], row['b4'], row['b5'], row['b6']]
                print(f"   Concurso {row['concurso']}: {sorted(dezenas)}")
        
        return self.anomalies
    
    def analyze_modular_persistence(self):
        """
        ANÁLISE DE PERSISTÊNCIA MODULAR AVANÇADA
        Estuda transições entre restos modulares
        """
        print("\n🔢 ANALISANDO PERSISTÊNCIA MODULAR...")
        
        self.modular_analysis = {}
        
        for mod in [3, 5, 7, 9, 11, 13, 17, 19]:
            # Criar estados modulares
            states = self.df[f'soma_mod_{mod}'].values
            
            # Matriz de transição modular
            n_states = mod
            trans_count = np.zeros((n_states, n_states))
            
            for i in range(len(states) - 1):
                trans_count[states[i], states[i+1]] += 1
            
            # Probabilidades de transição
            row_sums = trans_count.sum(axis=1, keepdims=True)
            trans_prob = np.divide(trans_count, row_sums, where=row_sums!=0)
            
            # Entropia condicional modular
            cond_entropy = 0
            for i in range(n_states):
                p_x = np.sum(trans_count[i]) / np.sum(trans_count)
                if p_x > 0:
                    p_y_given_x = trans_prob[i]
                    ent_x = entropy(p_y_given_x[p_y_given_x > 0])
                    cond_entropy += p_x * ent_x
            
            # Entropia máxima (distribuição uniforme)
            max_entropy = np.log(n_states)
            
            # Persistência (probabilidade de permanecer no mesmo estado)
            persistence = np.trace(trans_prob) / n_states
            
            self.modular_analysis[mod] = {
                'transition_matrix': trans_prob,
                'conditional_entropy': cond_entropy,
                'max_entropy': max_entropy,
                'entropy_ratio': cond_entropy / max_entropy if max_entropy > 0 else 0,
                'persistence': persistence,
                'is_random': abs(cond_entropy - max_entropy) < 0.1
            }
        
        # Mostrar módulos com comportamento não-aleatório
        non_random_mods = {k: v for k, v in self.modular_analysis.items() 
                          if not v['is_random']}
        
        print(f"\n📊 MÓDULOS COM COMPORTAMENTO NÃO-ALEATÓRIO:")
        for mod, analysis in non_random_mods.items():
            print(f"   • Módulo {mod}: Persistência={analysis['persistence']:.3f}, "
                  f"Entropia={analysis['conditional_entropy']:.3f} (max={analysis['max_entropy']:.3f})")
        
        return self.modular_analysis
    
    def cluster_temporal_patterns(self):
        """
        CLUSTERIZAÇÃO TEMPORAL AVANÇADA
        Encontra "famílias" de concursos usando similaridade vetorial
        """
        print("\n🕰️  CLUSTERIZANDO PADRÕES TEMPORAIS...")
        
        # Usar representação vetorial para clustering
        X = self.vectors_binary
        
        # PCA para visualização
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Clustering hierárquico
        linkage_matrix = linkage(X, method='ward')
        
        # Encontrar número ótimo de clusters
        max_clusters = 10
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            if len(set(clusters)) > 1:
                score = silhouette_score(X, clusters)
                silhouette_scores.append(score)
        
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Aplicar clustering com número ótimo
        clusters = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')
        self.df['temporal_cluster'] = clusters
        
        # KMeans também
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.df['kmeans_cluster'] = kmeans.fit_predict(X)
        
        print(f"✅ Clusters temporais identificados: {optimal_clusters}")
        
        # Análise dos clusters
        for i in range(1, optimal_clusters + 1):
            cluster_data = self.df[self.df['temporal_cluster'] == i]
            print(f"   Cluster {i}: {len(cluster_data)} concursos "
                  f"({len(cluster_data)/len(self.df)*100:.1f}%)")
        
        # Identificar transições entre clusters
        cluster_transitions = np.zeros((optimal_clusters, optimal_clusters))
        cluster_seq = self.df['temporal_cluster'].values
        
        for i in range(len(cluster_seq) - 1):
            cluster_transitions[cluster_seq[i]-1, cluster_seq[i+1]-1] += 1
        
        # Normalizar
        row_sums = cluster_transitions.sum(axis=1, keepdims=True)
        self.cluster_transition_prob = np.divide(cluster_transitions, row_sums, 
                                                 where=row_sums!=0)
        
        print(f"\n🔄 Persistência de cluster: {np.trace(self.cluster_transition_prob)/optimal_clusters:.3f}")
        
        return clusters
    
    def create_advanced_visualizations(self, output_dir='graficos_avancados'):
        """
        CRIA VISUALIZAÇÕES AVANÇADAS E REVELADORAS
        """
        print(f"\n🎨 CRIANDO VISUALIZAÇÕES AVANÇADAS em {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Mapa de calor de similaridade coseno entre concursos
        self._plot_cosine_similarity_heatmap(f'{output_dir}/01_similaridade_coseno.png')
        
        # 2. Dendrograma de clusters temporais
        self._plot_dendrogram(f'{output_dir}/02_dendrograma.png')
        
        # 3. Comparação real vs aleatório para TODAS as métricas
        self._plot_random_comparison(f'{output_dir}/03_comparacao_aleatorio.png')
        
        # 4. Gráfico de persistência modular
        self._plot_modular_persistence(f'{output_dir}/04_persistencia_modular.png')
        
        # 5. Matriz de transição Markoviana
        self._plot_markov_transitions(f'{output_dir}/05_markov_transicoes.png')
        
        # 6. Anomalias no espaço PCA
        self._plot_anomalies_pca(f'{output_dir}/06_anomalias_pca.png')
        
        # 7. Evolução temporal da entropia condicional
        self._plot_entropy_evolution(f'{output_dir}/07_evolucao_entropia.png')
        
        # 8. Fingerprints estruturais dos clusters
        self._plot_cluster_fingerprints(f'{output_dir}/08_fingerprints_clusters.png')
        
        # 9. Rede de transições entre estados
        self._plot_state_network(f'{output_dir}/09_rede_estados.png')
        
        # 10. Série temporal de anomalias
        self._plot_anomaly_timeseries(f'{output_dir}/10_serie_anomalias.png')
        
        print(f"✅ Visualizações salvas em {output_dir}/")
    
    def _plot_cosine_similarity_heatmap(self, filename):
        """Heatmap de similaridade coseno entre últimos 100 concursos"""
        # Usar últimos 100 concursos para visualização
        n_recent = min(100, len(self.dezenas))
        recent_vectors = self.vectors_binary[-n_recent:]
        
        # Calcular similaridade coseno
        similarity = np.zeros((n_recent, n_recent))
        for i in range(n_recent):
            for j in range(n_recent):
                similarity[i, j] = 1 - cosine(recent_vectors[i], recent_vectors[j])
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(similarity, cmap='viridis', square=True, 
                   xticklabels=range(len(self.df)-n_recent+1, len(self.df)+1),
                   yticklabels=range(len(self.df)-n_recent+1, len(self.df)+1))
        ax.set_title(f'Similaridade Coseno entre Últimos {n_recent} Concursos')
        ax.set_xlabel('Concurso')
        ax.set_ylabel('Concurso')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_dendrogram(self, filename):
        """Dendrograma hierárquico dos concursos"""
        # Usar amostra para visualização
        sample_size = min(200, len(self.dezenas))
        sample_idx = np.random.choice(len(self.dezenas), sample_size, replace=False)
        sample_vectors = self.vectors_binary[sample_idx]
        
        linkage_matrix = linkage(sample_vectors, method='ward')
        
        fig, ax = plt.subplots(figsize=(20, 10))
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        ax.set_title('Dendrograma Hierárquico dos Concursos (Amostra)')
        ax.set_xlabel('Amostra de Concursos')
        ax.set_ylabel('Distância')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_random_comparison(self, filename):
        """Comparação visual real vs aleatório para métricas principais"""
        if self.universo_aleatorio is None:
            self.generate_random_universe()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = [
            ('soma', 'Soma das Dezenas'),
            ('pares', 'Quantidade de Pares'),
            ('primos', 'Quantidade de Primos'),
            ('amplitude', 'Amplitude'),
            ('fibonacci', 'Números Fibonacci'),
            ('distancia_media', 'Distância Média')
        ]
        
        for ax, (metric, title) in zip(axes.flat, metrics):
            if metric in self.df.columns:
                real_data = self.df[metric].values
            else:
                real_data = self._calculate_metric(metric)
            
            random_data = self.universo_aleatorio[metric]
            
            ax.hist(real_data, bins=30, alpha=0.7, density=True, 
                   label='Real', color='blue', edgecolor='black')
            ax.hist(random_data, bins=30, alpha=0.7, density=True, 
                   label='Aleatório', color='orange', edgecolor='black')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adicionar p-value do teste KS
            ks_stat, ks_pvalue = stats.ks_2samp(real_data, random_data)
            ax.text(0.95, 0.95, f'KS p={ks_pvalue:.4f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle('Comparação Mega-Sena Real vs Universo Aleatório', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_modular_persistence(self, filename):
        """Gráfico de persistência modular"""
        if not hasattr(self, 'modular_analysis'):
            self.analyze_modular_persistence()
        
        mods = list(self.modular_analysis.keys())
        persistence = [self.modular_analysis[m]['persistence'] for m in mods]
        entropy_ratio = [self.modular_analysis[m]['entropy_ratio'] for m in mods]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Persistência
        ax = axes[0]
        bars = ax.bar(mods, persistence, color=['green' if p > 1/len(mods) else 'red' 
                                                for p in persistence])
        ax.axhline(y=np.mean(persistence), color='blue', linestyle='--', 
                  label=f'Média: {np.mean(persistence):.3f}')
        ax.set_xlabel('Módulo')
        ax.set_ylabel('Persistência')
        ax.set_title('Persistência Modular (Prob. de Permanecer no Mesmo Resto)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Entropia
        ax2 = axes[1]
        ax2.bar(mods, entropy_ratio, color='purple', alpha=0.7)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Máxima Entropia (Aleatório)')
        ax2.set_xlabel('Módulo')
        ax2.set_ylabel('Razão de Entropia')
        ax2.set_title('Entropia Condicional / Entropia Máxima')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_markov_transitions(self, filename):
        """Visualização das transições Markovianas"""
        if self.transition_prob is None:
            return
        
        # Usar apenas estados mais frequentes
        state_freq = np.sum(self.transition_count, axis=1)
        top_states_idx = np.argsort(state_freq)[-15:]  # Top 15 estados
        top_states = [self.markov_state_labels[i] for i in top_states_idx]
        
        # Submatriz de transição
        sub_matrix = self.transition_prob[top_states_idx][:, top_states_idx]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(sub_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=top_states, yticklabels=top_states,
                   cbar_kws={'label': 'Probabilidade de Transição'})
        ax.set_title('Matriz de Transição Markoviana (Top 15 Estados)')
        ax.set_xlabel('Estado em t+1')
        ax.set_ylabel('Estado em t')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_anomalies_pca(self, filename):
        """Visualização de anomalias no espaço PCA"""
        if self.anomalies is None:
            self.detect_anomalies()
        
        features = ['soma', 'amplitude', 'pares', 'primos', 'fibonacci', 
                   'consecutivas', 'distancia_media', 'moldura', 'centro']
        
        X = self.df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plotar todos os pontos
        normal = self.df['anomaly_consensus'] < 2
        anomaly = self.df['anomaly_consensus'] >= 2
        
        ax.scatter(X_pca[normal, 0], X_pca[normal, 1], 
                  c='blue', alpha=0.5, label='Normal', s=50)
        ax.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1], 
                  c='red', alpha=0.8, label='Anômalo', s=100, edgecolors='black')
        
        # Anotar anomalias
        for idx in self.anomalies.index:
            if idx < len(self.df):
                ax.annotate(f"{self.df.loc[idx, 'concurso']}", 
                          (X_pca[idx, 0], X_pca[idx, 1]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Detecção de Anomalias Estruturais (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_entropy_evolution(self, filename):
        """Evolução temporal da entropia condicional"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Calcular entropia condicional em janelas deslizantes
        window_size = 200
        entropies = []
        
        for i in range(window_size, len(self.df)):
            window = self.df['markov_state_id'].iloc[i-window_size:i].values
            
            # Matriz de transição local
            n_states = len(set(window))
            trans_count = np.zeros((n_states, n_states))
            
            state_map = {s: idx for idx, s in enumerate(set(window))}
            mapped_states = [state_map[s] for s in window]
            
            for j in range(len(mapped_states) - 1):
                trans_count[mapped_states[j], mapped_states[j+1]] += 1
            
            # Entropia condicional local
            cond_entropy = 0
            for j in range(n_states):
                if np.sum(trans_count[j]) > 0:
                    p_x = np.sum(trans_count[j]) / np.sum(trans_count)
                    p_y_given_x = trans_count[j] / np.sum(trans_count[j])
                    cond_entropy += p_x * entropy(p_y_given_x[p_y_given_x > 0])
            
            entropies.append(cond_entropy)
        
        ax.plot(range(window_size, len(self.df)), entropies, color='darkred', linewidth=1)
        ax.axhline(y=self.conditional_entropy, color='blue', linestyle='--', 
                  label=f'Entropia Global: {self.conditional_entropy:.3f}')
        ax.set_xlabel('Concurso')
        ax.set_ylabel('Entropia Condicional')
        ax.set_title('Evolução Temporal da Entropia Condicional H(X_t+1 | X_t)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_cluster_fingerprints(self, filename):
        """Fingerprints estruturais dos clusters temporais"""
        if 'temporal_cluster' not in self.df.columns:
            self.cluster_temporal_patterns()
        
        features = ['soma', 'pares', 'primos', 'fibonacci', 'amplitude', 
                   'moldura', 'consecutivas', 'distancia_media']
        
        n_clusters = len(set(self.df['temporal_cluster']))
        cluster_means = self.df.groupby('temporal_cluster')[features].mean()
        
        # Normalizar para visualização
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(features))
        width = 0.8 / n_clusters
        
        for i in range(n_clusters):
            ax.bar(x + i*width, cluster_means_norm.iloc[i], width, 
                  label=f'Cluster {i+1}', alpha=0.8)
        
        ax.set_xlabel('Features Estruturais')
        ax.set_ylabel('Valor Normalizado')
        ax.set_title('Fingerprints Estruturais por Cluster Temporal')
        ax.set_xticks(x + width * (n_clusters-1) / 2)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_state_network(self, filename):
        """Rede de transições entre estados principais"""
        # Simplificado - mostrar top estados e transições fortes
        if self.transition_prob is None:
            return
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Identificar top 10 estados
        state_freq = np.sum(self.transition_count, axis=1)
        top_states = np.argsort(state_freq)[-10:]
        
        # Matriz de adjacência para transições fortes
        adj_matrix = np.zeros((10, 10))
        for i, state_i in enumerate(top_states):
            for j, state_j in enumerate(top_states):
                prob = self.transition_prob[state_i, state_j]
                if prob > 0.1:  # Transições com prob > 10%
                    adj_matrix[i, j] = prob
        
        sns.heatmap(adj_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[f'E{i+1}' for i in range(10)],
                   yticklabels=[f'E{i+1}' for i in range(10)])
        ax.set_title('Rede de Transições entre Top 10 Estados')
        ax.set_xlabel('Estado Destino')
        ax.set_ylabel('Estado Origem')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_anomaly_timeseries(self, filename):
        """Série temporal de anomalias"""
        if 'anomaly_consensus' not in self.df.columns:
            self.detect_anomalies()
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        
        # Série temporal das anomalias
        ax = axes[0]
        colors = ['green' if score < 2 else 'red' for score in self.df['anomaly_consensus']]
        ax.scatter(self.df.index, self.df['soma'], c=colors, alpha=0.6, s=30)
        ax.set_xlabel('Índice do Concurso')
        ax.set_ylabel('Soma')
        ax.set_title('Série Temporal com Anomalias Destacadas (Vermelho = Anômalo)')
        ax.grid(True, alpha=0.3)
        
        # Frequência de anomalias ao longo do tempo
        ax2 = axes[1]
        window = 100
        anomaly_rate = self.df['anomaly_consensus'].rolling(window).apply(
            lambda x: (x >= 2).sum() / len(x)
        )
        
        ax2.plot(self.df.index, anomaly_rate, color='purple', linewidth=2)
        ax2.axhline(y=0.05, color='red', linestyle='--', label='Esperado 5%')
        ax2.set_xlabel('Índice do Concurso')
        ax2.set_ylabel('Taxa de Anomalias')
        ax2.set_title(f'Taxa de Anomalias em Janela Móvel de {window} Concursos')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _calculate_metric(self, metric_name):
        """Calcula métrica para os dados reais"""
        if metric_name == 'consecutivas':
            return self.df['consecutivas'].values
        elif metric_name == 'distancia_media':
            return self.df['distancia_media'].values
        else:
            return None
    
    def export_comprehensive_report(self, output_dir='relatorio_final'):
        """
        Exporta relatório completo com TODAS as descobertas
        """
        print(f"\n📄 GERANDO RELATÓRIO COMPLETO em {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Excel com todos os dados processados
        excel_path = f'{output_dir}/dados_completos_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Aba principal
            self.df.to_excel(writer, sheet_name='Dados_Completos', index=False)
            
            # Comparação com aleatório
            if hasattr(self, 'comparison_results'):
                comparison_df = pd.DataFrame(self.comparison_results).T
                comparison_df.to_excel(writer, sheet_name='Comparacao_Aleatorio')
            
            # Análise modular
            if hasattr(self, 'modular_analysis'):
                modular_df = pd.DataFrame(self.modular_analysis).T
                modular_df.to_excel(writer, sheet_name='Analise_Modular')
            
            # Anomalias
            if self.anomalies is not None:
                self.anomalies.to_excel(writer, sheet_name='Anomalias')
        
        print(f"✅ Excel exportado: {excel_path}")
        
        # 2. Relatório JSON com métricas principais
        report = {
            'data_analise': datetime.now().isoformat(),
            'total_concursos': len(self.df),
            'estados_markov': len(set(self.df['markov_state_id'])),
            'entropia_condicional': float(self.conditional_entropy),
            'anomalias_detectadas': len(self.anomalies) if self.anomalies is not None else 0,
            'comparacao_aleatorio': {},
            'persistencia_modular': {}
        }
        
        if hasattr(self, 'comparison_results'):
            report['comparacao_aleatorio'] = {
                k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                    for kk, vv in v.items()}
                for k, v in self.comparison_results.items()
            }
        
        if hasattr(self, 'modular_analysis'):
            report['persistencia_modular'] = {
                str(k): {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                        for kk, vv in v.items()}
                for k, v in self.modular_analysis.items()
            }
        
        json_path = f'{output_dir}/relatorio_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Relatório JSON exportado: {json_path}")
        
        return excel_path, json_path
    
    def generate_evolutionary_recommendations(self, n_recommendations=10):
        """
        GERA RECOMENDAÇÕES USANDO ALGORITMO EVOLUTIVO
        Encontra jogos que maximizam aderência histórica
        """
        print(f"\n🧬 GERANDO RECOMENDAÇÕES EVOLUTIVAS...")
        
        # Função de fitness: combinação de métricas desejáveis
        def fitness(game):
            game = np.array(sorted(game))
            
            # Métricas do jogo
            soma = np.sum(game)
            pares = np.sum(game % 2 == 0)
            primos = np.sum(np.isin(game, list(self.primes)))
            fib = np.sum(np.isin(game, list(self.fibonacci)))
            
            # Comparar com distribuições históricas
            soma_z = (soma - self.df['soma'].mean()) / self.df['soma'].std()
            pares_z = abs(pares - self.df['pares'].mean()) / self.df['pares'].std()
            primos_z = abs(primos - self.df['primos'].mean()) / self.df['primos'].std()
            
            # Penalizar extremos
            score = -abs(soma_z) * 2 - pares_z - primos_z
            
            # Bônus por Fibonacci
            score += fib * 0.5
            
            return score
        
        # População inicial
        population = []
        for _ in range(100):
            game = sorted(np.random.choice(range(1, 61), 6, replace=False))
            population.append((fitness(game), game))
        
        # Evolução
        for generation in range(50):
            # Selecionar melhores
            population.sort(key=lambda x: x[0], reverse=True)
            population = population[:50]
            
            # Crossover
            new_population = population.copy()
            for _ in range(50):
                parent1 = population[np.random.randint(0, len(population))][1]
                parent2 = population[np.random.randint(0, len(population))][1]
                
                # Crossover simples
                child = list(set(list(parent1[:3]) + list(parent2[3:])))
                while len(child) < 6:
                    new_num = np.random.randint(1, 61)
                    if new_num not in child:
                        child.append(new_num)
                child = sorted(child[:6])
                
                # Mutação
                if np.random.random() < 0.1:
                    idx = np.random.randint(0, 6)
                    child[idx] = np.random.randint(1, 61)
                    child = sorted(list(set(child)))
                    while len(child) < 6:
                        new_num = np.random.randint(1, 61)
                        if new_num not in child:
                            child.append(new_num)
                    child = sorted(child[:6])
                
                new_population.append((fitness(child), child))
            
            population = new_population
        
        # Melhores recomendações
        population.sort(key=lambda x: x[0], reverse=True)
        recommendations = population[:n_recommendations]
        
        print(f"\n🎯 TOP {n_recommendations} RECOMENDAÇÕES ESTRUTURAIS:")
        for i, (score, game) in enumerate(recommendations, 1):
            print(f"   {i}. {game} (Score: {score:.3f})")
        
        return recommendations


def main():
    """
    EXECUÇÃO PRINCIPAL DO SISTEMA AVANÇADO
    """
    print("="*80)
    print("🔬 SISTEMA AVANÇADO DE ANÁLISE ESTRUTURAL DA MEGA-SENA")
    print("   Fingerprints Vetoriais + Cadeias de Markov + Detecção de Anomalias")
    print("="*80)
    
    # Inicializar
    analyzer = MegaSenaStructuralAnalyzer('resultados_megasena.csv')
    
    # 1. Gerar universo aleatório para comparação
    analyzer.generate_random_universe(n_simulations=50000)
    
    # 2. Comparar com universo aleatório
    analyzer.compare_with_random_universe()
    
    # 3. Detectar anomalias
    analyzer.detect_anomalies()
    
    # 4. Analisar persistência modular
    analyzer.analyze_modular_persistence()
    
    # 5. Clusterização temporal
    analyzer.cluster_temporal_patterns()
    
    # 6. Criar visualizações avançadas
    analyzer.create_advanced_visualizations()
    
    # 7. Gerar recomendações evolutivas
    analyzer.generate_evolutionary_recommendations(10)
    
    # 8. Exportar relatório completo
    analyzer.export_comprehensive_report()
    
    print("\n" + "="*80)
    print("✅ ANÁLISE AVANÇADA CONCLUÍDA!")
    print("📁 Resultados salvos em:")
    print("   • graficos_avancados/ - Visualizações de alta qualidade")
    print("   • relatorio_final/ - Dados completos em Excel e JSON")
    print("="*80)
    
    # Insights principais
    print("\n🔍 PRINCIPAIS DESCOBERTAS:")
    print(f"   1. Entropia condicional: {analyzer.conditional_entropy:.4f}")
    print(f"   2. Estados de Markov identificados: {len(set(analyzer.df['markov_state_id']))}")
    print(f"   3. Anomalias estruturais: {len(analyzer.anomalies) if analyzer.anomalies is not None else 0}")
    
    if hasattr(analyzer, 'comparison_results'):
        sig_diffs = sum(1 for v in analyzer.comparison_results.values() if v['is_significant'])
        if sig_diffs > 0:
            print(f"   4. ⚠️  {sig_diffs} métricas com diferenças significativas do aleatório!")
        else:
            print(f"   4. ✅ Mega-Sena compatível com aleatoriedade nas métricas testadas")


if __name__ == "__main__":
    main()
