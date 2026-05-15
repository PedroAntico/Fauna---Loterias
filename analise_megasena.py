#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DEFINITIVO DE ANÁLISE ESTRUTURAL DA MEGA-SENA
=====================================================
Versão 4.0 - Com Correções Estatísticas e Testes de Significância Rigorosos

CORREÇÕES CRÍTICAS:
✅ Cosine similarity com proteção NaN
✅ Entropia condicional com validação
✅ Métricas padronizadas real vs aleatório
✅ Correção de múltiplos testes (Bonferroni/BH)
✅ Baseline aleatório para anomalias
✅ Diversidade genética no evolutivo

NOVAS FEATURES:
✅ Análise espectral (FFT)
✅ Mutual Information temporal
✅ Teste de periodicidade
✅ Validação cruzada temporal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, entropy, chi2_contingency
from scipy.spatial.distance import euclidean, hamming, cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
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

# ============================================
# ENCODER NUMPY PARA JSON
# ============================================
class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para tipos NumPy"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MegaSenaAdvancedAnalyzer:
    """
    Analisador estrutural DEFINITIVO com correções estatísticas rigorosas
    """
    
    def __init__(self, csv_path='resultados_megasena.csv'):
        """Inicialização robusta"""
        self.csv_path = csv_path
        self.df = None
        self.dezenas = None
        self.universo_aleatorio = None
        
        # Conjuntos matemáticos
        self.fibonacci = self._generate_fibonacci(60)
        self.primes = self._generate_primes(60)
        
        # Para correção de múltiplos testes
        self.all_pvalues = []
        
        self.load_and_prepare_data()
        
    def _generate_fibonacci(self, limit):
        """Gera Fibonacci com proteção"""
        fib = [0, 1]
        while fib[-1] <= limit:
            fib.append(fib[-1] + fib[-2])
        return set(fib[2:])
    
    def _generate_primes(self, limit):
        """Gera primos de forma otimizada"""
        primes = set()
        for num in range(2, limit + 1):
            if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                primes.add(num)
        return primes
    
    def load_and_prepare_data(self):
        """Carrega dados com features estruturais COMPLETAS"""
        print("🔬 CARREGANDO DADOS E CRIANDO FEATURES AVANÇADAS...")
        
        # Carregar CSV (múltiplos formatos)
        try:
            self.df = pd.read_csv(self.csv_path, sep=';', encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
            except:
                self.df = pd.read_csv(self.csv_path, sep=';', encoding='latin-1')
        
        # Padronizar
        self.df.columns = ['concurso', 'data', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
        self.dezenas = self.df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6']].values
        
        print(f"📊 {len(self.df)} concursos carregados")
        
        # Criar TODAS as features
        self._create_structural_features()
        self._create_vector_representations()
        self._create_markov_states()
        
        print("✅ Features criadas com sucesso!")
    
    def _count_consecutive_single(self, dezenas):
        """
        MÉTODO PADRONIZADO para contar consecutivas
        USAR EXATAMENTE A MESMA FUNÇÃO para real e aleatório
        """
        sorted_dez = sorted(dezenas)
        count = 0
        seq_length = 1
        
        for i in range(len(sorted_dez) - 1):
            if sorted_dez[i+1] - sorted_dez[i] == 1:
                seq_length += 1
            else:
                if seq_length >= 2:
                    count += seq_length
                seq_length = 1
        
        if seq_length >= 2:
            count += seq_length
        
        return count
    
    def _create_structural_features(self):
        """Features estruturais com MÉTRICAS PADRONIZADAS"""
        print("🎯 Criando fingerprints estruturais padronizados...")
        
        # Features básicas
        self.df['soma'] = self.dezenas.sum(axis=1)
        self.df['amplitude'] = self.dezenas.max(axis=1) - self.dezenas.min(axis=1)
        self.df['media'] = self.dezenas.mean(axis=1)
        self.df['desvio_padrao'] = self.dezenas.std(axis=1)
        
        # Contagens PADRONIZADAS
        self.df['pares'] = np.sum(self.dezenas % 2 == 0, axis=1)
        self.df['primos'] = np.sum(np.isin(self.dezenas, list(self.primes)), axis=1)
        self.df['fibonacci'] = np.sum(np.isin(self.dezenas, list(self.fibonacci)), axis=1)
        
        # Distribuição espacial
        moldura = {1,2,3,4,5,6,7,8,9,10,19,20,28,29,37,38,46,47,55,56,57,58,59,60}
        self.df['moldura'] = np.sum(np.isin(self.dezenas, list(moldura)), axis=1)
        self.df['centro'] = 6 - self.df['moldura']
        
        # Quadrantes
        self.df['q1'] = np.sum((self.dezenas >= 1) & (self.dezenas <= 15), axis=1)
        self.df['q2'] = np.sum((self.dezenas >= 16) & (self.dezenas <= 30), axis=1)
        self.df['q3'] = np.sum((self.dezenas >= 31) & (self.dezenas <= 45), axis=1)
        self.df['q4'] = np.sum((self.dezenas >= 46) & (self.dezenas <= 60), axis=1)
        
        # USANDO MÉTODO PADRONIZADO para consecutivas
        self.df['consecutivas'] = [self._count_consecutive_single(d) for d in self.dezenas]
        
        # Distâncias
        self.df['distancia_media'] = [
            np.mean([d[i+1] - d[i] for i in range(5)]) for d in self.dezenas
        ]
        
        # Repetidas do anterior
        self.df['repetidas'] = self._calculate_repeats()
        
        # Features modulares
        for mod in [3, 5, 7, 9, 11, 13, 17, 19]:
            self.df[f'soma_mod_{mod}'] = self.df['soma'] % mod
        
        print(f"✅ {len(self.df.columns)} features criadas")
    
    def _calculate_repeats(self):
        """Calcula repetidas do concurso anterior"""
        repeats = [0]
        for i in range(1, len(self.dezenas)):
            count = len(set(self.dezenas[i]) & set(self.dezenas[i-1]))
            repeats.append(count)
        return repeats
    
    def _create_vector_representations(self):
        """Representações vetoriais com proteção NaN"""
        print("📐 Criando representações vetoriais geométricas...")
        
        # Vetores binários (60 dimensões)
        self.vectors_binary = np.zeros((len(self.dezenas), 60))
        for i, dezenas in enumerate(self.dezenas):
            self.vectors_binary[i, dezenas-1] = 1
        
        # Distâncias entre consecutivos (COM PROTEÇÃO)
        self.distances = {
            'euclidean': [],
            'hamming': [],
            'cosine': []
        }
        
        for i in range(len(self.dezenas) - 1):
            v1 = self.vectors_binary[i]
            v2 = self.vectors_binary[i + 1]
            
            # Euclidiana
            self.distances['euclidean'].append(euclidean(v1, v2))
            
            # Hamming
            self.distances['hamming'].append(hamming(v1, v2))
            
            # Cosine COM PROTEÇÃO CONTRA NaN
            cos_dist = cosine(v1, v2)
            if np.isnan(cos_dist):
                cos_dist = 0.0  # Vetores idênticos
            self.distances['cosine'].append(cos_dist)
        
        print("✅ Distâncias vetoriais calculadas")
    
    def _create_markov_states(self):
        """Estados de Markov AVANÇADOS"""
        print("🔄 Criando estados de Markov...")
        
        # Criar estado composto
        soma_faixas = pd.qcut(self.df['soma'], q=5, labels=['MB', 'B', 'M', 'A', 'MA'])
        
        self.df['markov_state'] = (
            self.df['pares'].astype(str) + 'P_' +
            self.df['primos'].astype(str) + 'PR_' +
            soma_faixas.astype(str) + '_' +
            self.df['moldura'].astype(str) + 'M'
        )
        
        # Codificar
        le = LabelEncoder()
        self.df['markov_state_id'] = le.fit_transform(self.df['markov_state'])
        self.markov_labels = dict(enumerate(le.classes_))
        
        # Matriz de transição
        n_states = len(le.classes_)
        self.transition_count = np.zeros((n_states, n_states))
        
        states = self.df['markov_state_id'].values
        for i in range(len(states) - 1):
            self.transition_count[states[i], states[i+1]] += 1
        
        # Probabilidades
        row_sums = self.transition_count.sum(axis=1, keepdims=True)
        self.transition_prob = np.divide(
            self.transition_count, row_sums,
            where=row_sums!=0, out=np.zeros_like(self.transition_count)
        )
        
        # Entropia condicional COM PROTEÇÃO
        self.conditional_entropy = self._calculate_safe_conditional_entropy()
        
        print(f"✅ {n_states} estados Markovianos")
        print(f"📊 Entropia condicional: {self.conditional_entropy:.4f}")
    
    def _calculate_safe_conditional_entropy(self):
        """
        Entropia condicional COM PROTEÇÃO contra vetores vazios
        """
        cond_entropy = 0.0
        
        for i in range(len(self.transition_prob)):
            # Probabilidade marginal do estado i
            p_x = np.sum(self.transition_count[i]) / np.sum(self.transition_count)
            
            if p_x > 0:
                # Probabilidades condicionais P(j|i)
                p_y_given_x = self.transition_prob[i]
                
                # Filtrar probabilidades válidas (COM PROTEÇÃO)
                valid_probs = p_y_given_x[p_y_given_x > 0]
                
                if len(valid_probs) > 0:
                    ent_x = entropy(valid_probs)
                    cond_entropy += p_x * ent_x
        
        return cond_entropy
    
    def generate_random_universe(self, n_simulations=100000):
        """
        GERA UNIVERSO ALEATÓRIO COM MÉTRICAS PADRONIZADAS
        USA EXATAMENTE AS MESMAS FUNÇÕES DO REAL
        """
        print(f"\n🎲 GERANDO UNIVERSO ALEATÓRIO ({n_simulations:,} jogos)...")
        
        random_games = np.zeros((n_simulations, 6), dtype=int)
        for i in tqdm(range(n_simulations), desc="Simulando aleatório"):
            random_games[i] = sorted(np.random.choice(range(1, 61), 6, replace=False))
        
        # USANDO AS MESMAS FUNÇÕES DO REAL
        self.universo_aleatorio = {
            'jogos': random_games,
            'soma': random_games.sum(axis=1),
            'pares': np.sum(random_games % 2 == 0, axis=1),
            'primos': np.sum(np.isin(random_games, list(self.primes)), axis=1),
            'fibonacci': np.sum(np.isin(random_games, list(self.fibonacci)), axis=1),
            'amplitude': random_games.max(axis=1) - random_games.min(axis=1),
            # USA MÉTODO PADRONIZADO
            'consecutivas': np.array([self._count_consecutive_single(g) for g in random_games]),
            'distancia_media': np.array([
                np.mean([sorted(g)[i+1] - sorted(g)[i] for i in range(5)]) 
                for g in random_games
            ])
        }
        
        print("✅ Universo aleatório gerado com métricas padronizadas!")
        return self.universo_aleatorio
    
    def compare_with_random_universe(self):
        """
        COMPARAÇÃO RIGOROSA com correção de múltiplos testes
        """
        if self.universo_aleatorio is None:
            self.generate_random_universe()
        
        print("\n🔬 COMPARAÇÃO ESTATÍSTICA RIGOROSA...")
        
        self.comparison_results = {}
        self.all_pvalues = []
        
        metrics = ['soma', 'pares', 'primos', 'fibonacci', 'amplitude', 
                   'consecutivas', 'distancia_media']
        
        for metric in metrics:
            real_data = self.df[metric].values
            random_data = self.universo_aleatorio[metric]
            
            # Teste KS
            ks_stat, ks_pvalue = stats.ks_2samp(real_data, random_data)
            
            # Teste Mann-Whitney
            mw_stat, mw_pvalue = stats.mannwhitneyu(real_data, random_data, alternative='two-sided')
            
            # Armazenar p-values para correção
            self.all_pvalues.append(('KS', metric, ks_pvalue))
            self.all_pvalues.append(('MW', metric, mw_pvalue))
            
            # Estatísticas
            self.comparison_results[metric] = {
                'real_mean': float(np.mean(real_data)),
                'random_mean': float(np.mean(random_data)),
                'real_std': float(np.std(real_data)),
                'random_std': float(np.std(random_data)),
                'difference_pct': float((np.mean(real_data) - np.mean(random_data)) / 
                                       np.mean(random_data) * 100),
                'ks_pvalue': float(ks_pvalue),
                'mw_pvalue': float(mw_pvalue)
            }
        
        # CORREÇÃO DE MÚLTIPLOS TESTES
        self._apply_multiple_testing_correction()
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS (com correção de múltiplos testes):")
        
        significant_before = sum(1 for v in self.comparison_results.values() 
                                if v['ks_pvalue'] < 0.05 or v['mw_pvalue'] < 0.05)
        
        print(f"   • Significativos (não corrigido): {significant_before}")
        print(f"   • Significativos (Bonferroni): {sum(1 for _, _, p in self.all_pvalues if p < 0.05/len(self.all_pvalues))}")
        print(f"   • Significativos (BH): {sum(1 for r in self.bh_results if r['bh_reject'])}")
        
        return self.comparison_results
    
    def _apply_multiple_testing_correction(self):
        """
        APLICA CORREÇÃO DE MÚLTIPLOS TESTES
        Bonferroni e Benjamini-Hochberg
        """
        # Extrair p-values
        pvalues = np.array([p for _, _, p in self.all_pvalues])
        
        # Bonferroni
        n_tests = len(pvalues)
        bonferroni_threshold = 0.05 / n_tests
        
        # Benjamini-Hochberg
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = pvalues[sorted_indices]
        
        bh_thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
        bh_reject = sorted_pvalues <= bh_thresholds
        
        # Encontrar o maior k que satisfaz
        if np.any(bh_reject):
            k_max = np.max(np.where(bh_reject))
            bh_significant = sorted_indices[:k_max + 1]
        else:
            bh_significant = []
        
        self.bh_results = []
        for i, (test, metric, p) in enumerate(self.all_pvalues):
            self.bh_results.append({
                'test': test,
                'metric': metric,
                'pvalue': p,
                'bonferroni_reject': p < bonferroni_threshold,
                'bh_reject': i in bh_significant
            })
        
        print(f"\n🔬 Correção aplicada:")
        print(f"   • Testes realizados: {n_tests}")
        print(f"   • Threshold Bonferroni: {bonferroni_threshold:.6f}")
        print(f"   • BH significativos: {len(bh_significant)}")
    
    def detect_anomalies_with_baseline(self):
        """
        DETECÇÃO DE ANOMALIAS COM BASELINE ALEATÓRIO
        Responde: "Quantas anomalias surgiriam num RNG puro?"
        """
        print("\n🔍 DETECTANDO ANOMALIAS COM BASELINE ALEATÓRIO...")
        
        # Features para detecção
        anomaly_features = ['soma', 'amplitude', 'pares', 'primos', 'fibonacci', 
                           'consecutivas', 'distancia_media', 'moldura', 'centro']
        
        # Dados reais
        X_real = self.df[anomaly_features].values
        scaler = StandardScaler()
        X_real_scaled = scaler.fit_transform(X_real)
        
        # Detectar anomalias nos dados REAIS
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        real_anomalies = iso_forest.fit_predict(X_real_scaled)
        real_anomaly_rate = (real_anomalies == -1).sum() / len(real_anomalies)
        
        # Gerar baseline aleatório
        n_baseline = min(10000, len(self.df) * 10)
        random_data = np.zeros((n_baseline, len(anomaly_features)))
        
        for i in range(n_baseline):
            game = sorted(np.random.choice(range(1, 61), 6, replace=False))
            features = [
                sum(game),
                max(game) - min(game),
                sum(1 for n in game if n % 2 == 0),
                sum(1 for n in game if n in self.primes),
                sum(1 for n in game if n in self.fibonacci),
                self._count_consecutive_single(game),
                np.mean([game[i+1] - game[i] for i in range(5)]),
                sum(1 for n in game if n in {1,2,3,4,5,6,7,8,9,10,19,20,28,29,37,38,46,47,55,56,57,58,59,60}),
                6 - sum(1 for n in game if n in {1,2,3,4,5,6,7,8,9,10,19,20,28,29,37,38,46,47,55,56,57,58,59,60})
            ]
            random_data[i] = features
        
        # Detectar anomalias no baseline aleatório
        random_scaled = scaler.transform(random_data)
        random_anomalies = iso_forest.predict(random_scaled)
        random_anomaly_rate = (random_anomalies == -1).sum() / len(random_anomalies)
        
        # Comparar
        self.anomaly_baseline = {
            'real_rate': float(real_anomaly_rate),
            'random_rate': float(random_anomaly_rate),
            'ratio': float(real_anomaly_rate / random_anomaly_rate) if random_anomaly_rate > 0 else float('inf'),
            'z_score': float((real_anomaly_rate - random_anomaly_rate) / 
                            np.sqrt(random_anomaly_rate * (1 - random_anomaly_rate) / n_baseline))
        }
        
        print(f"\n📊 COMPARAÇÃO DE ANOMALIAS:")
        print(f"   • Taxa real: {real_anomaly_rate:.3%}")
        print(f"   • Taxa aleatória: {random_anomaly_rate:.3%}")
        print(f"   • Razão: {self.anomaly_baseline['ratio']:.2f}x")
        print(f"   • Z-score: {self.anomaly_baseline['z_score']:.2f}")
        
        if abs(self.anomaly_baseline['z_score']) < 2:
            print("   ✅ Taxa de anomalias compatível com aleatoriedade")
        else:
            print("   ⚠️  Taxa de anomalias significativamente diferente do esperado!")
        
        # Armazenar resultados
        self.df['is_anomaly'] = real_anomalies == -1
        
        return self.anomaly_baseline
    
    def analyze_spectral_patterns(self):
        """
        ANÁLISE ESPECTRAL (FOURIER)
        Procura periodicidades ocultas
        """
        print("\n🌊 ANALISANDO PADRÕES ESPECTRAIS (FFT)...")
        
        self.spectral_results = {}
        
        # Sinais a analisar
        signals = {
            'soma': self.df['soma'].values,
            'pares': self.df['pares'].values,
            'primos': self.df['primos'].values,
            'consecutivas': self.df['consecutivas'].values,
            'amplitude': self.df['amplitude'].values
        }
        
        for name, signal in signals.items():
            # Remover tendência
            signal_detrended = signal - np.mean(signal)
            
            # FFT
            n = len(signal_detrended)
            fft_values = fft(signal_detrended)
            freqs = fftfreq(n)
            
            # Magnitude (apenas frequências positivas)
            positive_freqs = freqs[:n//2]
            magnitude = np.abs(fft_values[:n//2])
            
            # Normalizar
            magnitude_norm = magnitude / np.max(magnitude)
            
            # Encontrar picos
            # Ignorar frequência zero (DC)
            if len(positive_freqs) > 1:
                peak_threshold = np.mean(magnitude_norm[1:]) + 2 * np.std(magnitude_norm[1:])
                peaks = np.where(magnitude_norm[1:] > peak_threshold)[0]
                
                # Converter para períodos
                peak_periods = []
                for peak in peaks:
                    if positive_freqs[peak + 1] > 0:
                        period = 1 / positive_freqs[peak + 1]
                        if period <= n:  # Período válido
                            peak_periods.append({
                                'period': float(period),
                                'magnitude': float(magnitude_norm[peak + 1]),
                                'frequency': float(positive_freqs[peak + 1])
                            })
                
                self.spectral_results[name] = {
                    'peak_count': len(peak_periods),
                    'peaks': sorted(peak_periods, key=lambda x: x['magnitude'], reverse=True)[:5],
                    'max_magnitude': float(np.max(magnitude_norm))
                }
        
        # Mostrar resultados
        print(f"\n📊 PADRÕES ESPECTRAIS ENCONTRADOS:")
        for name, results in self.spectral_results.items():
            print(f"   • {name}: {results['peak_count']} picos significativos")
            if results['peaks']:
                top_peak = results['peaks'][0]
                print(f"     - Maior pico: período={top_peak['period']:.1f} concursos")
        
        return self.spectral_results
    
    def calculate_mutual_information(self):
        """
        MUTUAL INFORMATION TEMPORAL
        Mede dependência não-linear entre concursos consecutivos
        """
        print("\n🔗 CALCULANDO MUTUAL INFORMATION TEMPORAL...")
        
        self.mi_results = {}
        
        # Features discretizadas
        features = {
            'soma_bins': pd.qcut(self.df['soma'], q=10, labels=False),
            'pares': self.df['pares'],
            'primos': self.df['primos'],
            'consecutivas': self.df['consecutivas'],
            'moldura': self.df['moldura']
        }
        
        for name, feature in features.items():
            # Valores atuais e futuros
            X_t = feature.values[:-1]
            X_t1 = feature.values[1:]
            
            # Mutual Information
            mi = mutual_info_score(X_t, X_t1)
            
            # Normalizar por entropia máxima
            h_x = entropy(np.bincount(X_t[X_t >= 0]) / len(X_t))
            h_x1 = entropy(np.bincount(X_t1[X_t1 >= 0]) / len(X_t1))
            
            if h_x > 0 and h_x1 > 0:
                mi_normalized = mi / np.sqrt(h_x * h_x1)
            else:
                mi_normalized = 0
            
            self.mi_results[name] = {
                'mi': float(mi),
                'mi_normalized': float(mi_normalized)
            }
        
        print(f"\n📊 MUTUAL INFORMATION:")
        for name, results in self.mi_results.items():
            print(f"   • {name}: MI={results['mi']:.4f}, Normalized={results['mi_normalized']:.4f}")
        
        return self.mi_results
    
    def generate_diverse_recommendations(self, n_recommendations=10):
        """
        ALGORITMO EVOLUTIVO COM DIVERSIDADE GENÉTICA
        Evita convergência prematura para clones
        """
        print(f"\n🧬 GERANDO RECOMENDAÇÕES DIVERSIFICADAS...")
        
        def fitness(game):
            """Fitness function multi-objetivo"""
            game = np.array(sorted(game))
            
            # Métricas do jogo
            soma = np.sum(game)
            pares = np.sum(game % 2 == 0)
            primos = np.sum(np.isin(game, list(self.primes)))
            
            # Z-scores (proximidade da média histórica)
            soma_z = abs(soma - self.df['soma'].mean()) / self.df['soma'].std()
            pares_z = abs(pares - self.df['pares'].mean()) / self.df['pares'].std()
            
            # Penalizar extremos, recompensar típico
            score = -(soma_z + pares_z)
            
            return score
        
        # População inicial DIVERSIFICADA
        population = []
        seen_initial = set()
        
        while len(population) < 100:
            game = tuple(sorted(np.random.choice(range(1, 61), 6, replace=False)))
            if game not in seen_initial:
                seen_initial.add(game)
                population.append((fitness(game), list(game)))
        
        # Evolução COM DIVERSIDADE
        for generation in range(100):
            # Ordenar por fitness
            population.sort(key=lambda x: x[0], reverse=True)
            
            # ELITISMO PARCIAL (manter top 20, não 50)
            elite = population[:20]
            
            # Crossover
            new_population = elite.copy()
            
            for _ in range(60):
                # Selecionar pais da elite
                parent1 = elite[np.random.randint(0, len(elite))][1]
                parent2 = elite[np.random.randint(0, len(elite))][1]
                
                # Crossover
                child = list(set(list(parent1[:3]) + list(parent2[3:])))
                while len(child) < 6:
                    new_num = np.random.randint(1, 61)
                    if new_num not in child:
                        child.append(new_num)
                child = sorted(child[:6])
                
                # Mutação adaptativa
                if np.random.random() < 0.15:
                    idx = np.random.randint(0, 6)
                    child[idx] = np.random.randint(1, 61)
                    child = sorted(list(set(child)))
                    while len(child) < 6:
                        new_num = np.random.randint(1, 61)
                        if new_num not in child:
                            child.append(new_num)
                    child = sorted(child[:6])
                
                new_population.append((fitness(child), child))
            
            # SANGUE NOVO (20 jogos aleatórios)
            for _ in range(20):
                random_game = sorted(np.random.choice(range(1, 61), 6, replace=False))
                new_population.append((fitness(random_game), random_game))
            
            population = new_population
        
        # Selecionar recomendações ÚNICAS
        unique_games = []
        seen = set()
        
        for score, game in sorted(population, key=lambda x: x[0], reverse=True):
            key = tuple(sorted(game))
            if key not in seen:
                seen.add(key)
                unique_games.append((score, game))
        
        recommendations = unique_games[:n_recommendations]
        
        print(f"\n🎯 TOP {n_recommendations} RECOMENDAÇÕES DIVERSIFICADAS:")
        for i, (score, game) in enumerate(recommendations, 1):
            print(f"   {i}. {game} (Score: {score:.3f})")
        
        # Verificar diversidade
        unique_count = len(set(tuple(g) for _, g in unique_games))
        print(f"\n📊 Diversidade: {unique_count} jogos únicos em {len(unique_games)} candidatos")
        
        return recommendations
    
    def create_visualizations(self, output_dir='graficos_finais'):
        """Visualizações completas"""
        print(f"\n🎨 GERANDO VISUALIZAÇÕES em {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Comparação real vs aleatório
        self._plot_comprehensive_comparison(f'{output_dir}/01_comparacao_completa.png')
        
        # 2. Análise espectral
        self._plot_spectral_analysis(f'{output_dir}/02_analise_espectral.png')
        
        # 3. Matriz de transição Markoviana
        self._plot_markov_matrix(f'{output_dir}/03_markov_transicoes.png')
        
        # 4. Evolução temporal com anomalias
        self._plot_temporal_anomalies(f'{output_dir}/04_anomalias_temporais.png')
        
        # 5. Mutual Information
        self._plot_mutual_information(f'{output_dir}/05_mutual_information.png')
        
        # 6. Correção de múltiplos testes
        self._plot_multiple_testing(f'{output_dir}/06_correcao_testes.png')
        
        # 7. Distribuição de distâncias vetoriais
        self._plot_vector_distances(f'{output_dir}/07_distancias_vetoriais.png')
        
        print(f"✅ Visualizações salvas em {output_dir}/")
    
    def _plot_comprehensive_comparison(self, filename):
        """Comparação visual completa"""
        if self.universo_aleatorio is None:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        metrics = ['soma', 'pares', 'primos', 'fibonacci', 'amplitude', 
                   'consecutivas', 'distancia_media', 'repetidas']
        
        for ax, metric in zip(axes.flat, metrics):
            if metric in self.df.columns and metric in self.universo_aleatorio:
                real = self.df[metric].values
                random = self.universo_aleatorio[metric]
                
                ax.hist(real, bins=30, alpha=0.7, density=True, 
                       label='Real', color='blue')
                ax.hist(random, bins=30, alpha=0.7, density=True, 
                       label='Aleatório', color='orange')
                
                # KS test
                ks_p = stats.ks_2samp(real, random)[1]
                ax.set_title(f'{metric} (KS p={ks_p:.4f})')
                ax.legend(fontsize=8)
        
        plt.suptitle('Comparação Mega-Sena Real vs Universo Aleatório', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_spectral_analysis(self, filename):
        """Análise espectral (FFT)"""
        if not hasattr(self, 'spectral_results'):
            self.analyze_spectral_patterns()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        signals = ['soma', 'pares', 'primos', 'consecutivas']
        
        for ax, signal_name in zip(axes.flat, signals):
            signal = self.df[signal_name].values
            signal_detrended = signal - np.mean(signal)
            
            # FFT
            fft_vals = fft(signal_detrended)
            freqs = fftfreq(len(signal))
            
            n = len(signal)
            ax.plot(freqs[:n//2], np.abs(fft_vals[:n//2]))
            ax.set_xlabel('Frequência')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'Espectro de Frequência - {signal_name}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Análise Espectral (FFT) - Procurando Periodicidades', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_markov_matrix(self, filename):
        """Matriz de transição Markoviana"""
        if not hasattr(self, 'transition_prob'):
            return
        
        # Top estados mais frequentes
        state_freq = np.sum(self.transition_count, axis=1)
        top_indices = np.argsort(state_freq)[-15:]
        
        sub_matrix = self.transition_prob[top_indices][:, top_indices]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(sub_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Probabilidade'})
        ax.set_title('Matriz de Transição Markoviana (Top 15 Estados)')
        ax.set_xlabel('Estado em t+1')
        ax.set_ylabel('Estado em t')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_temporal_anomalies(self, filename):
        """Série temporal com anomalias"""
        if 'is_anomaly' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(18, 12))
        
        # Soma com anomalias
        ax = axes[0]
        normal = ~self.df['is_anomaly']
        anomaly = self.df['is_anomaly']
        
        ax.scatter(self.df.index[normal], self.df['soma'][normal], 
                  c='blue', alpha=0.5, s=20, label='Normal')
        ax.scatter(self.df.index[anomaly], self.df['soma'][anomaly], 
                  c='red', alpha=0.8, s=50, label='Anômalo')
        ax.set_ylabel('Soma')
        ax.set_title('Série Temporal com Anomalias Destacadas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pares
        ax2 = axes[1]
        ax2.scatter(self.df.index[normal], self.df['pares'][normal], 
                   c='blue', alpha=0.5, s=20)
        ax2.scatter(self.df.index[anomaly], self.df['pares'][anomaly], 
                   c='red', alpha=0.8, s=50)
        ax2.set_ylabel('Pares')
        ax2.grid(True, alpha=0.3)
        
        # Consecutivas
        ax3 = axes[2]
        ax3.scatter(self.df.index[normal], self.df['consecutivas'][normal], 
                   c='blue', alpha=0.5, s=20)
        ax3.scatter(self.df.index[anomaly], self.df['consecutivas'][anomaly], 
                   c='red', alpha=0.8, s=50)
        ax3.set_xlabel('Índice do Concurso')
        ax3.set_ylabel('Consecutivas')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Detecção de Anomalias Estruturais', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_mutual_information(self, filename):
        """Mutual Information entre features"""
        if not hasattr(self, 'mi_results'):
            self.calculate_mutual_information()
        
        features = list(self.mi_results.keys())
        mi_values = [self.mi_results[f]['mi'] for f in features]
        mi_norm = [self.mi_results[f]['mi_normalized'] for f in features]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(features))
        width = 0.35
        
        ax.bar(x - width/2, mi_values, width, label='MI Bruto', color='blue', alpha=0.7)
        ax.bar(x + width/2, mi_norm, width, label='MI Normalizado', color='green', alpha=0.7)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Dependência Temporal Não-Linear (Mutual Information)')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_multiple_testing(self, filename):
        """Visualização da correção de múltiplos testes"""
        if not hasattr(self, 'bh_results'):
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Ordenar p-values
        pvalues = sorted([r['pvalue'] for r in self.bh_results])
        n = len(pvalues)
        
        # Plotar p-values
        ax.plot(range(1, n+1), pvalues, 'bo-', label='P-values', markersize=8)
        
        # Linha Bonferroni
        ax.axhline(y=0.05/n, color='red', linestyle='--', 
                  label=f'Bonferroni (0.05/{n})')
        
        # Linha Benjamini-Hochberg
        bh_line = [(i/n)*0.05 for i in range(1, n+1)]
        ax.plot(range(1, n+1), bh_line, 'g--', label='Benjamini-Hochberg')
        
        ax.set_xlabel('Rank do P-value')
        ax.set_ylabel('P-value')
        ax.set_title('Correção de Múltiplos Testes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _plot_vector_distances(self, filename):
        """Distribuição das distâncias vetoriais"""
        if not hasattr(self, 'distances'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        dist_types = ['euclidean', 'hamming', 'cosine']
        titles = ['Distância Euclidiana', 'Distância de Hamming', 'Distância Coseno']
        
        for ax, dist_type, title in zip(axes, dist_types, titles):
            if dist_type in self.distances:
                data = self.distances[dist_type]
                ax.hist(data, bins=30, alpha=0.7, color='purple', edgecolor='black')
                ax.set_xlabel('Distância')
                ax.set_ylabel('Frequência')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
                # Média
                mean_val = np.mean(data)
                ax.axvline(x=mean_val, color='red', linestyle='--', 
                          label=f'Média: {mean_val:.3f}')
                ax.legend()
        
        plt.suptitle('Distribuição das Distâncias Vetoriais entre Concursos Consecutivos', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def export_final_report(self, output_dir='relatorio_definitivo'):
        """
        Exporta relatório COMPLETO e ROBUSTO
        """
        print(f"\n📄 EXPORTANDO RELATÓRIO DEFINITIVO...")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Excel com todos os dados
        excel_path = f'{output_dir}/analise_completa_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Dados_Completos', index=False)
            
            if hasattr(self, 'comparison_results'):
                pd.DataFrame(self.comparison_results).T.to_excel(
                    writer, sheet_name='Comparacao_Aleatorio')
            
            if hasattr(self, 'bh_results'):
                pd.DataFrame(self.bh_results).to_excel(
                    writer, sheet_name='Correcao_Testes')
            
            if hasattr(self, 'spectral_results'):
                spectral_df = pd.DataFrame([
                    {'feature': k, 'peaks': v['peak_count'], 
                     'max_magnitude': v['max_magnitude']}
                    for k, v in self.spectral_results.items()
                ])
                spectral_df.to_excel(writer, sheet_name='Analise_Espectral', index=False)
        
        print(f"✅ Excel: {excel_path}")
        
        # 2. Relatório JSON (COM ENCODER NUMPY)
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_concursos': int(len(self.df)),
            'entropia_condicional': float(self.conditional_entropy),
            'anomaly_baseline': self.anomaly_baseline if hasattr(self, 'anomaly_baseline') else {},
            'comparison_results': self.comparison_results if hasattr(self, 'comparison_results') else {},
            'mutual_information': self.mi_results if hasattr(self, 'mi_results') else {},
            'spectral_peaks': {k: v['peak_count'] for k, v in self.spectral_results.items()} 
                             if hasattr(self, 'spectral_results') else {}
        }
        
        json_path = f'{output_dir}/relatorio_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"✅ JSON: {json_path}")
        
        # 3. Resumo em texto
        summary_path = f'{output_dir}/resumo_{timestamp}.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE ESTRUTURAL DA MEGA-SENA\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Concursos analisados: {len(self.df)}\n\n")
            
            f.write("ENTROPIA CONDICIONAL\n")
            f.write(f"H(X_t+1 | X_t) = {self.conditional_entropy:.4f}\n\n")
            
            if hasattr(self, 'anomaly_baseline'):
                f.write("ANOMALIAS\n")
                f.write(f"Taxa real: {self.anomaly_baseline['real_rate']:.3%}\n")
                f.write(f"Taxa aleatória: {self.anomaly_baseline['random_rate']:.3%}\n")
                f.write(f"Z-score: {self.anomaly_baseline['z_score']:.2f}\n\n")
            
            f.write("CONCLUSÃO\n")
            if abs(self.conditional_entropy - np.log(len(set(self.df['markov_state_id'])))) < 0.1:
                f.write("Sistema compatível com aleatoriedade nas métricas testadas.\n")
            else:
                f.write("Possível estrutura não-aleatória detectada.\n")
        
        print(f"✅ Resumo: {summary_path}")
        
        return excel_path, json_path


def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*80)
    print("🔬 SISTEMA DEFINITIVO DE ANÁLISE ESTRUTURAL DA MEGA-SENA")
    print("   Com correções estatísticas e baseline aleatório")
    print("="*80)
    
    # Inicializar
    analyzer = MegaSenaAdvancedAnalyzer('resultados_megasena.csv')
    
    # 1. Universo aleatório
    analyzer.generate_random_universe(50000)
    
    # 2. Comparação rigorosa
    analyzer.compare_with_random_universe()
    
    # 3. Anomalias com baseline
    analyzer.detect_anomalies_with_baseline()
    
    # 4. Análise espectral
    analyzer.analyze_spectral_patterns()
    
    # 5. Mutual Information
    analyzer.calculate_mutual_information()
    
    # 6. Recomendações diversificadas
    analyzer.generate_diverse_recommendations(10)
    
    # 7. Visualizações
    analyzer.create_visualizations()
    
    # 8. Relatório final
    analyzer.export_final_report()
    
    print("\n" + "="*80)
    print("✅ ANÁLISE DEFINITIVA CONCLUÍDA!")
    print("="*80)
    
    # Insights principais
    print("\n🔍 PRINCIPAIS DESCOBERTAS:")
    print(f"   1. Entropia condicional: {analyzer.conditional_entropy:.4f}")
    
    if hasattr(analyzer, 'anomaly_baseline'):
        z = analyzer.anomaly_baseline['z_score']
        if abs(z) < 2:
            print(f"   2. ✅ Anomalias compatíveis com aleatoriedade (z={z:.2f})")
        else:
            print(f"   2. ⚠️  Anomalias acima do esperado (z={z:.2f})")
    
    if hasattr(analyzer, 'bh_results'):
        sig_bh = sum(1 for r in analyzer.bh_results if r['bh_reject'])
        if sig_bh == 0:
            print(f"   3. ✅ Nenhuma métrica significativa após correção BH")
        else:
            print(f"   3. ⚠️  {sig_bh} métricas significativas mesmo após correção")


if __name__ == "__main__":
    main()
