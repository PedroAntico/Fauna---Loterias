#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRAMEWORK DE VALIDAÇÃO DE ALEATORIEDADE - MEGA-SENA vs RNG
==========================================================
Versão 5.0 - Análise Comparativa Internacional + Testes NIST-like + Permutation Entropy

NOVAS CAPACIDADES:
✅ FFT com baseline aleatório (z-score espectral)
✅ Permutation Entropy para estrutura ordinal
✅ Testes estatísticos NIST-like (runs, serial, cumulative sums)
✅ Comparação contra loterias internacionais
✅ Análise de compressibilidade
✅ Detecção de viés estrutural comparativo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from scipy.stats import pearsonr, entropy, chi2_contingency, norm
from scipy.spatial.distance import euclidean, hamming, cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, mutual_info_score
from collections import Counter, defaultdict
from itertools import combinations, product, permutations
from datetime import datetime
import warnings
import os
import json
import zlib
from tqdm import tqdm
import hashlib

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150

class NumpyEncoder(json.JSONEncoder):
    """Encoder NumPy para JSON"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

class AleatoriedadeValidator:
    """
    Framework científico para validação de aleatoriedade
    Compara Mega-Sena contra RNG puro e loterias internacionais
    """
    
    def __init__(self, csv_path='resultados_megasena.csv'):
        """Inicialização do framework de validação"""
        self.csv_path = csv_path
        self.df = None
        self.dezenas = None
        
        # Universos de comparação
        self.rng_baseline = None
        self.international_lotteries = {}
        
        # Resultados de testes
        self.fft_baseline = {}
        self.permutation_entropy = {}
        self.nist_tests = {}
        self.compressibility = {}
        
        # Conjuntos matemáticos
        self.fibonacci = self._gen_fibonacci(60)
        self.primes = self._gen_primes(60)
        
        self.load_data()
        
    def _gen_fibonacci(self, limit):
        fib = [0, 1]
        while fib[-1] <= limit:
            fib.append(fib[-1] + fib[-2])
        return set(fib[2:])
    
    def _gen_primes(self, limit):
        return {n for n in range(2, limit+1) 
                if all(n % i != 0 for i in range(2, int(n**0.5)+1))}
    
    def load_data(self):
        """Carrega e prepara dados da Mega-Sena"""
        print("📂 CARREGANDO DADOS DA MEGA-SENA...")
        
        try:
            self.df = pd.read_csv(self.csv_path, sep=';', encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
            except:
                self.df = pd.read_csv(self.csv_path, sep=';', encoding='latin-1')
        
        self.df.columns = ['concurso', 'data', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
        self.dezenas = self.df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6']].values
        
        # Criar features estruturais
        self._create_features()
        
        print(f"✅ {len(self.df)} concursos carregados")
    
    def _create_features(self):
        """Features estruturais completas"""
        self.df['soma'] = self.dezenas.sum(axis=1)
        self.df['pares'] = np.sum(self.dezenas % 2 == 0, axis=1)
        self.df['primos'] = np.sum(np.isin(self.dezenas, list(self.primes)), axis=1)
        self.df['fibonacci'] = np.sum(np.isin(self.dezenas, list(self.fibonacci)), axis=1)
        self.df['amplitude'] = self.dezenas.max(axis=1) - self.dezenas.min(axis=1)
        self.df['consecutivas'] = [self._count_consecutive(d) for d in self.dezenas]
        
        # Vetores binários (60 dimensões)
        self.binary_vectors = np.zeros((len(self.dezenas), 60))
        for i, d in enumerate(self.dezenas):
            self.binary_vectors[i, d-1] = 1
    
    def _count_consecutive(self, dezenas):
        """Método padronizado para consecutivas"""
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
    
    def generate_rng_baseline(self, n_simulations=100000):
        """
        GERA BASELINE RNG ROBUSTO
        Múltiplas características para comparação justa
        """
        print(f"\n🎲 GERANDO BASELINE RNG ({n_simulations:,} simulações)...")
        
        random_games = np.zeros((n_simulations, 6), dtype=int)
        for i in tqdm(range(n_simulations), desc="RNG puro"):
            random_games[i] = sorted(np.random.choice(range(1, 61), 6, replace=False))
        
        # Features do baseline
        self.rng_baseline = {
            'jogos': random_games,
            'soma': random_games.sum(axis=1),
            'pares': np.sum(random_games % 2 == 0, axis=1),
            'primos': np.sum(np.isin(random_games, list(self.primes)), axis=1),
            'amplitude': random_games.max(axis=1) - random_games.min(axis=1),
            'consecutivas': np.array([self._count_consecutive(g) for g in random_games]),
            'binary_vectors': np.array([
                np.eye(60)[g-1].sum(axis=0) for g in random_games[:10000]
            ])
        }
        
        print("✅ Baseline RNG gerado!")
        return self.rng_baseline
    
    def fft_with_baseline(self):
        """
        ANÁLISE ESPECTRAL COM BASELINE
        Compara FFT da Mega vs FFT do RNG
        Responde: "Os picos espectrais são anormais?"
        """
        print("\n🌊 ANÁLISE ESPECTRAL COM BASELINE COMPARATIVO...")
        
        if self.rng_baseline is None:
            self.generate_rng_baseline()
        
        features = ['soma', 'pares', 'primos', 'consecutivas']
        
        for feature in features:
            # Sinal real
            signal_real = self.df[feature].values
            signal_real = signal_real - np.mean(signal_real)
            
            # FFT real
            fft_real = np.abs(fft(signal_real)[:len(signal_real)//2])
            
            # Gerar distribuição de referência via Monte Carlo
            n_permutations = 1000
            fft_max_magnitudes = []
            fft_peak_counts = []
            
            for _ in tqdm(range(n_permutations), desc=f"FFT baseline {feature}", leave=False):
                # Amostra aleatória do baseline
                sample_size = len(signal_real)
                random_sample = np.random.choice(
                    self.rng_baseline[feature], 
                    size=sample_size, 
                    replace=False
                )
                random_sample = random_sample - np.mean(random_sample)
                
                # FFT do baseline
                fft_random = np.abs(fft(random_sample)[:sample_size//2])
                
                # Coletar estatísticas
                fft_max_magnitudes.append(np.max(fft_random))
                
                # Contar picos (threshold = média + 2*std)
                threshold = np.percentile(fft_random, 99)
            
            # Estatísticas do baseline
            max_mag_mean = np.mean(fft_max_magnitudes)
            max_mag_std = np.std(fft_max_magnitudes)
            peak_count_mean = np.mean(fft_peak_counts)
            peak_count_std = np.std(fft_peak_counts)
            
            # Valores observados
            observed_max = np.max(fft_real)
            threshold_obs = np.percentile(fft_real, 99)
            
            # Z-scores
            z_max = (observed_max - max_mag_mean) / max_mag_std if max_mag_std > 0 else 0
            
            self.fft_baseline[feature] = {
                'observed_max_magnitude': float(observed_max),
                'baseline_max_mean': float(max_mag_mean),
                'baseline_max_std': float(max_mag_std),
                'z_score_max': float(z_max),
                'is_anomalous': abs(z_max) > 2
            }
        
        # Resultados
        print(f"\n📊 RESULTADOS FFT COM BASELINE:")
        for feature, results in self.fft_baseline.items():
            print(f"   • {feature}:")
            print(f"     - Magnitude máxima: z={results['z_score_max']:.2f}")
            print(f"     - Anômalo: {'⚠️ SIM' if results['is_anomalous'] else '✅ NÃO'}")
            
        return self.fft_baseline
    
    def permutation_entropy_analysis(self, order=3, delay=1):
        """
        PERMUTATION ENTROPY
        Mede estrutura ordinal nos dados
        Mais sensível que Shannon para dependências temporais
        """
        print(f"\n🔢 ANALISANDO PERMUTATION ENTROPY (ordem={order})...")
        
        def calc_permutation_entropy(series, order, delay):
            """Calcula Permutation Entropy de Bandt-Pompe"""
            n = len(series)
            permutations_count = Counter()
            total_patterns = 0
            
            for i in range(n - (order - 1) * delay):
                # Extrair padrão
                pattern = tuple(series[i + j*delay] for j in range(order))
                
                # Encontrar permutação (ordem relativa)
                sorted_indices = tuple(np.argsort(pattern))
                permutations_count[sorted_indices] += 1
                total_patterns += 1
            
            # Calcular entropia
            if total_patterns == 0:
                return 0, 0
            
            probs = np.array([count / total_patterns for count in permutations_count.values()])
            pe = entropy(probs)
            pe_normalized = pe / np.log(math.factorial(order))  # Normalizar
            
            return pe, pe_normalized
        
        features = ['soma', 'pares', 'primos', 'consecutivas', 'amplitude']
        
        for feature in features:
            # Mega-Sena
            signal_real = self.df[feature].values
            pe_real, pe_norm_real = calc_permutation_entropy(signal_real, order, delay)
            
            # Baseline RNG
            if self.rng_baseline is None:
                self.generate_rng_baseline()
            
            # Distribuição de PE no baseline
            n_baseline = 1000
            pe_baseline = []
            
            for _ in range(n_baseline):
                sample = np.random.choice(self.rng_baseline[feature],  size=len(signal_real),  replace=True)
                _, pe_norm = calc_permutation_entropy(sample, order, delay)
                pe_baseline.append(pe_norm)
            
            pe_mean = np.mean(pe_baseline)
            pe_std = np.std(pe_baseline)
            
            self.permutation_entropy[feature] = {
                'pe_real': float(pe_real),
                'pe_normalized': float(pe_norm_real),
                'pe_baseline_mean': float(pe_mean),
                'pe_baseline_std': float(pe_std),
                'z_score': float((pe_real - pe_mean) / pe_std) if pe_std > 0 else 0,
                'p_value': float( 2 * (1 - norm.cdf(abs((pe_real - pe_mean) / pe_std)))) if pe_std > 0 else 1.0
            }
        
        print(f"\n📊 PERMUTATION ENTROPY:")
        for feature, results in self.permutation_entropy.items():
            print(f"   • {feature}: PE={results['pe_normalized']:.4f} "
                  f"(baseline: {results['pe_baseline_mean']:.4f}±{results['pe_baseline_std']:.4f}) "
                  f"{'⚠️' if results['p_value'] < 0.05 else '✅'}")
        
        return self.permutation_entropy
    
    def nist_statistical_tests(self):
        """
        TESTES ESTATÍSTICOS NIST-LIKE
        Transforma concursos em stream binário e aplica bateria de testes
        """
        print("\n🧪 APLICANDO TESTES ESTATÍSTICOS NIST-LIKE...")
        
        # Converter concursos para stream binário
        # Método: paridade das dezenas em ordem
        binary_stream = []
        for dezenas in self.dezenas:
            # Sequência de 60 bits (1=presente, 0=ausente)
            bits = np.zeros(60, dtype=int)
            bits[dezenas-1] = 1
            binary_stream.extend(bits)
        
        binary_stream = np.array(binary_stream)
        
        # Gerar baseline
        n_baseline_streams = 100
        baseline_results = defaultdict(list)
        
        print("   Gerando baseline para testes NIST...")
        for _ in tqdm(range(n_baseline_streams), desc="Baseline NIST"):
            # Gerar jogos aleatórios
            n_games = len(self.dezenas)
            random_stream = []
            for _ in range(n_games):
                game = sorted(np.random.choice(range(1, 61), 6, replace=False))
                bits = np.zeros(60, dtype=int)
                bits[np.array(game)-1] = 1
                random_stream.extend(bits)
            
            random_stream = np.array(random_stream)
            
            # Aplicar testes
            baseline_results['frequency'].append(self._frequency_test(random_stream))
            baseline_results['runs'].append(self._runs_test(random_stream))
            baseline_results['cumulative_sums_forward'].append(
                self._cumulative_sums_test(random_stream, mode=0))
            baseline_results['cumulative_sums_backward'].append(
                self._cumulative_sums_test(random_stream, mode=1))
            baseline_results['serial'].append(self._serial_test(random_stream))
            baseline_results['approximate_entropy'].append(
                self._approximate_entropy_test(random_stream))
        
        # Testes na Mega-Sena
        self.nist_tests = {
            'frequency': self._frequency_test(binary_stream),
            'runs': self._runs_test(binary_stream),
            'cumulative_sums_forward': self._cumulative_sums_test(binary_stream, mode=0),
            'cumulative_sums_backward': self._cumulative_sums_test(binary_stream, mode=1),
            'serial': self._serial_test(binary_stream),
            'approximate_entropy': self._approximate_entropy_test(binary_stream)
        }
        
        # Comparar com baseline
        for test_name, observed in self.nist_tests.items():
            baseline_vals = baseline_results[test_name]
            baseline_mean = np.mean(baseline_vals)
            baseline_std = np.std(baseline_vals)
            
            z_score = (observed - baseline_mean) / baseline_std if baseline_std > 0 else 0
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            
            self.nist_tests[test_name] = {
                'observed': float(observed),
                'baseline_mean': float(baseline_mean),
                'baseline_std': float(baseline_std),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'is_significant': p_value < 0.01
            }
        
        print(f"\n📊 RESULTADOS TESTES NIST-LIKE:")
        for test_name, results in self.nist_tests.items():
            sig = '⚠️ SIGNIFICATIVO' if results['is_significant'] else '✅ OK'
            print(f"   • {test_name}: p={results['p_value']:.4f} {sig}")
        
        return self.nist_tests
    
    def _frequency_test(self, bits):
        """Teste de frequência (proporção de 1s)"""
        n = len(bits)
        s = np.sum(2 * bits - 1)  # Soma de ±1
        s_obs = abs(s) / np.sqrt(n)
        p_value = 2 * (1 - norm.cdf(s_obs))
        return p_value
    
    def _runs_test(self, bits):
        """Teste de runs (oscilações)"""
        n = len(bits)
        pi = np.mean(bits)
        
        if abs(pi - 0.5) >= 2 / np.sqrt(n):
            return 0.0
        
        # Contar runs
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        num = abs(runs - 2 * n * pi * (1 - pi))
        den = 2 * np.sqrt(2 * n) * pi * (1 - pi)
        
        if den == 0:
            return 0.0
        
        p_value = 2 * (1 - norm.cdf(num / den))
        return p_value
    
    def _cumulative_sums_test(self, bits, mode=0):
        """Teste de somas cumulativas"""
        n = len(bits)
        
        if mode == 1:  # Backward
            bits = bits[::-1]
        
        # Converter para ±1
        x = 2 * bits - 1
        s = np.cumsum(x)
        z = np.max(np.abs(s))
        
        # Aproximação
        sum1 = 0.0
        for k in range(int(np.floor((-n/z + 1) / 4)), int(np.floor((n/z - 1) / 4)) + 1):
            sum1 += norm.cdf((4*k + 1) * z / np.sqrt(n)) - norm.cdf((4*k - 1) * z / np.sqrt(n))
        
        sum2 = 0.0
        for k in range(int(np.floor((-n/z - 3) / 4)), int(np.floor((n/z - 1) / 4)) + 1):
            sum2 += norm.cdf((4*k + 3) * z / np.sqrt(n)) - norm.cdf((4*k + 1) * z / np.sqrt(n))
        
        p_value = 1 - sum1 + sum2
        return max(0, min(1, p_value))
    
    def _serial_test(self, bits, m=2):
        """Teste serial (pares de bits)"""
        n = len(bits)
        
        # Extender sequência
        bits_extended = np.concatenate([bits, bits[:m-1]])
        
        # Contar padrões
        def count_patterns(seq, pattern_len):
            counts = defaultdict(int)
            for i in range(n):
                pattern = tuple(seq[i:i+pattern_len])
                counts[pattern] += 1
            return counts
        
        counts_m = count_patterns(bits_extended, m)
        counts_m1 = count_patterns(bits_extended, m-1)
        counts_m2 = count_patterns(bits_extended, m-2)
        
        # Calcular estatísticas
        psi2_m = sum(c**2 for c in counts_m.values()) * (2**m / n) - n
        psi2_m1 = sum(c**2 for c in counts_m1.values()) * (2**(m-1) / n) - n
        psi2_m2 = sum(c**2 for c in counts_m2.values()) * (2**(m-2) / n) - n
        
        del1 = psi2_m - psi2_m1
        del2 = psi2_m - 2*psi2_m1 + psi2_m2
        
        p_value1 = 1 - stats.chi2.cdf(del1, 2**(m-1))
        p_value2 = 1 - stats.chi2.cdf(del2, 2**(m-2))
        
        return (p_value1 + p_value2) / 2
    
    def _approximate_entropy_test(self, bits, m=2):
        """Teste de entropia aproximada"""
        n = len(bits)
        
        def phi(m):
            # Extender
            seq = np.concatenate([bits, bits[:m]])
            
            # Contar padrões
            counts = defaultdict(int)
            for i in range(n):
                pattern = tuple(seq[i:i+m])
                counts[pattern] += 1
            
            # Calcular phi
            return sum((c/n) * np.log(c/n) for c in counts.values() if c > 0)
        
        phi_m = phi(m)
        phi_m1 = phi(m+1)
        
        apen = phi_m - phi_m1
        chi2 = 2 * n * (np.log(2) - apen)
        
        p_value = 1 - stats.chi2.cdf(chi2, 2**(m-1))
        return p_value
    
    def analyze_compressibility(self):
        """
        ANÁLISE DE COMPRESSIBILIDADE
        Dados verdadeiramente aleatórios são incompressíveis
        """
        print("\n📦 ANALISANDO COMPRESSIBILIDADE...")
        
        # Converter concursos para diferentes representações
        representations = {
            'dezenas_raw': self.dezenas.tobytes(),
            'binary_60bits': self.binary_vectors.tobytes(),
            'soma_sequence': self.df['soma'].values.tobytes(),
            'pares_sequence': self.df['pares'].values.tobytes()
        }
        
        for name, data in representations.items():
            # Comprimir
            compressed = zlib.compress(data, level=9)
            
            # Métricas
            original_size = len(data)
            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size
            # Baseline RNG equivalente
            baseline_ratios = []
            
            for _ in range(100):
                random_games = np.zeros_like(self.dezenas)
            
                for i in range(len(random_games)):
                    random_games[i] = sorted(
                        np.random.choice(range(1, 61), 6, replace=False)
                    )
            
                if name == 'dezenas_raw':
                    baseline_data = random_games.tobytes()
            
                elif name == 'binary_60bits':
                    baseline_binary = np.zeros((len(random_games), 60))
                    for i, g in enumerate(random_games):
                        baseline_binary[i, g-1] = 1
                    baseline_data = baseline_binary.tobytes()
            
                elif name == 'soma_sequence':
                    baseline_data = random_games.sum(axis=1).tobytes()
            
                elif name == 'pares_sequence':
                    baseline_data = np.sum(random_games % 2 == 0, axis=1).tobytes()
            
                baseline_compressed = zlib.compress(baseline_data, level=9)
            
                baseline_ratios.append(len(baseline_compressed) / len(baseline_data))
            
            baseline_mean = np.mean(baseline_ratios)
            baseline_std = np.std(baseline_ratios)
            
            z_score = ((compression_ratio - baseline_mean) / baseline_std
                if baseline_std > 0 else 0)
            
            self.compressibility[name] = {
                'original_bytes': original_size,
                'compressed_bytes': compressed_size,
                'ratio': float(compression_ratio),
                'entropy_estimate': float(compression_ratio * 8),  # bits por byte
                'baseline_mean': float(baseline_mean),
                'baseline_std': float(baseline_std),
                'z_score': float(z_score),
                'is_anomalous': abs(z_score) > 2
                            }
        
        print(f"\n📊 COMPRESSIBILIDADE:")
        for name, results in self.compressibility.items():
            status = '⚠️ ANÔMALO' if results['is_anomalous'] else '✅ NORMAL'
            print(f"   • {name}: " f"ratio={results['ratio']:.3f} " f"(baseline={results['baseline_mean']:.3f}±{results['baseline_std']:.3f}) " f"z={results['z_score']:.2f} " f"{status}")
        
        return self.compressibility
    
    def compare_international_lotteries(self):
        """
        COMPARAÇÃO COM LOTERIAS INTERNACIONAIS (SIMULADAS)
        Em produção, carregaria dados reais de cada loteria
        """
        print("\n🌍 COMPARANDO COM LOTERIAS INTERNACIONAIS (simulação)...")
        
        # Simular diferentes loterias com parâmetros distintos
        lotteries_config = {
            'Mega-Sena': {'n_numbers': 6, 'max_number': 60, 'n_games': len(self.df)},
            'Powerball': {'n_numbers': 5, 'max_number': 69, 'n_games': len(self.df)},
            'EuroMillions': {'n_numbers': 5, 'max_number': 50, 'n_games': len(self.df)},
            'SuperEnalotto': {'n_numbers': 6, 'max_number': 90, 'n_games': len(self.df)},
            'El Gordo': {'n_numbers': 5, 'max_number': 54, 'n_games': len(self.df)}
        }
        
        for name, config in lotteries_config.items():
            if name == 'Mega-Sena':
                # Dados reais
                games = self.dezenas
            else:
                # Simular loteria
                games = np.zeros((config['n_games'], config['n_numbers']), dtype=int)
                for i in range(config['n_games']):
                    games[i] = sorted(np.random.choice(
                        range(1, config['max_number'] + 1), 
                        config['n_numbers'], 
                        replace=False
                    ))
            
            # Calcular métricas
            self.international_lotteries[name] = {
                'n_games': len(games),
                'n_numbers': config['n_numbers'],
                'max_number': config['max_number'],
                'avg_sum': float(np.mean(games.sum(axis=1))),
                'std_sum': float(np.std(games.sum(axis=1))),
                'avg_pares': float(np.mean(np.sum(games % 2 == 0, axis=1))),
                'entropy_dezenas': float(entropy(np.bincount(games.flatten(), minlength=config['max_number']+1)[1:]))
            }
        
        print(f"\n📊 COMPARAÇÃO INTERNACIONAL:")
        for name, metrics in self.international_lotteries.items():
            print(f"   • {name}: soma={metrics['avg_sum']:.1f}±{metrics['std_sum']:.1f}, "
                  f"pares={metrics['avg_pares']:.2f}, entropia={metrics['entropy_dezenas']:.4f}")
        
        return self.international_lotteries
    
    def run_full_validation_suite(self):
        """
        EXECUTA TODA A BATERIA DE VALIDAÇÃO
        """
        print("\n" + "="*80)
        print("🔬 INICIANDO BATERIA COMPLETA DE VALIDAÇÃO DE ALEATORIEDADE")
        print("="*80)
        
        results = {}
        
        # 1. Gerar baseline RNG
        self.generate_rng_baseline(50000)
        
        # 2. FFT com baseline
        print("\n" + "="*60)
        results['fft'] = self.fft_with_baseline()
        
        # 3. Permutation Entropy
        print("\n" + "="*60)
        results['permutation_entropy'] = self.permutation_entropy_analysis()
        
        # 4. Testes NIST-like
        print("\n" + "="*60)
        results['nist'] = self.nist_statistical_tests()
        
        # 5. Compressibilidade
        print("\n" + "="*60)
        results['compressibility'] = self.analyze_compressibility()
        self.apply_global_multiple_testing_correction(results)
        
        # 6. Comparação internacional
        print("\n" + "="*60)
        results['international'] = self.compare_international_lotteries()
        
        # Sumarizar
        self._generate_validation_summary(results)
        
        return results

    def apply_global_multiple_testing_correction(self, results):
        """
        Correção global de múltiplos testes
        """
        
        all_pvalues = []
    
        # FFT
        for feature, res in results['fft'].items():
            p = 2 * (1 - norm.cdf(abs(res['z_score_max'])))
            all_pvalues.append(p)
    
        # Permutation Entropy
        for feature, res in results['permutation_entropy'].items():
            z = abs(res['z_score'])
            p = 2 * (1 - norm.cdf(z))
            all_pvalues.append(p)
    
        # NIST
        for test, res in results['nist'].items():
            all_pvalues.append(res['p_value'])
    
        n_tests = len(all_pvalues)
    
        # Bonferroni
        bonf_threshold = 0.05 / n_tests
    
        significant = sum(p < bonf_threshold for p in all_pvalues)
    
        print("\n🔬 CORREÇÃO GLOBAL DE MÚLTIPLOS TESTES")
        print(f"   • Total de testes: {n_tests}")
        print(f"   • Threshold Bonferroni: {bonf_threshold:.6f}")
        print(f"   • Significativos após correção: {significant}")
    
        self.global_multiple_testing = {
            'n_tests': n_tests,
            'threshold': bonf_threshold,
            'significant_after_correction': significant}
    
    def _generate_validation_summary(self, results):
        """Gera sumário executivo da validação"""
        print("\n" + "="*80)
        print("📋 SUMÁRIO EXECUTIVO DE VALIDAÇÃO")
        print("="*80)
        
        # Contar testes significativos
        total_tests = 0
        significant_tests = 0
        
        # FFT
        for feature, res in results['fft'].items():
            total_tests += 2
            if res['is_anomalous']:
                significant_tests += 1
        
        # Permutation Entropy
        for feature, res in results['permutation_entropy'].items():
            total_tests += 1
            z = abs(res.get('z_score', 0))
            if z > 2:
                significant_tests += 1
        
        # NIST
        for test, res in results['nist'].items():
            total_tests += 1
            if res['is_significant']:
                significant_tests += 1
        
        # Compressibilidade
        total_tests += len(results['compressibility'])
        significant_tests += sum(  1 for r in results['compressibility'].values()
            if r['is_anomalous'])
        
        print(f"\n📊 TOTAL DE TESTES: {total_tests}")
        print(f"⚠️  TESTES SIGNIFICATIVOS: {significant_tests}")
        print(f"✅ TESTES NÃO-SIGNIFICATIVOS: {total_tests - significant_tests}")
        
        # Taxa de significância
        sig_rate = significant_tests / total_tests if total_tests > 0 else 0
        print(f"📈 TAXA DE SIGNIFICÂNCIA: {sig_rate:.1%}")
        
        # Interpretação
        print(f"\n🔍 INTERPRETAÇÃO:")
        
        if sig_rate <= 0.05:
            print("   ✅ Mega-Sena é COMPATÍVEL com processo aleatório")
            print("   ✅ Taxa de significância dentro do esperado pelo acaso")
            print("   ✅ Múltiplos testes não revelaram estrutura oculta")
        elif sig_rate <= 0.10:
            print("   ⚠️  Taxa de significância LIGEIRAMENTE elevada")
            print("   ⚠️  Possível viés residual detectado")
            print("   ⚠️  Recomendação: investigar features específicas")
        else:
            print("   🚨 Taxa de significância ANORMALMENTE alta")
            print("   🚨 Possível não-aleatoriedade detectada")
            print("   🚨 Recomendação: auditoria completa do processo")
        
        # Conclusão final
        print(f"\n🎯 CONCLUSÃO FINAL:")
        
        # Verificar consistência dos resultados
        nist_significant = sum(1 for r in results['nist'].values() if r['is_significant'])
        
        if nist_significant == 0 and sig_rate <= 0.05:
            print("   MEGA-SENA NÃO APRESENTA EVIDÊNCIA DE NÃO-ALEATORIEDADE")
            print("   Os dados são estatisticamente indistinguíveis de um RNG puro")
            print("   Este resultado fortalece a credibilidade do sistema de sorteios")
        else:
            print("   FORAM ENCONTRADAS POSSÍVEIS ANOMALIAS")
            print("   Recomenda-se auditoria independente dos sorteios")
        if hasattr(self, 'global_multiple_testing'):
            corrected_sig = self.global_multiple_testing['significant_after_correction']
        
            print(f"\n🛡️ APÓS CORREÇÃO GLOBAL:")
            print(f"   • Testes significativos restantes: {corrected_sig}")
        
            if corrected_sig == 0:
                print("   ✅ Nenhuma evidência robusta de não-aleatoriedade")
                
        return sig_rate
    
    def export_validation_report(self, output_dir='relatorio_validacao'):
        """Exporta relatório completo de validação"""
        print(f"\n📄 EXPORTANDO RELATÓRIO DE VALIDAÇÃO...")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Relatório JSON completo
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_analisada': {
                'concursos': int(len(self.df)),
                'periodo': f"{self.df['data'].min().strftime('%d/%m/%Y')} a {self.df['data'].max().strftime('%d/%m/%Y')}"
            },
            'resultados_fft': self.fft_baseline,
            'permutation_entropy': self.permutation_entropy,
            'testes_nist': self.nist_tests,
            'compressibilidade': self.compressibility,
            'comparacao_internacional': self.international_lotteries
        }
        
        json_path = f'{output_dir}/validacao_completa_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"✅ Relatório JSON: {json_path}")
        
        # Sumário executivo em texto
        summary_path = f'{output_dir}/sumario_executivo_{timestamp}.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("VALIDAÇÃO DE ALEATORIEDADE - MEGA-SENA\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Concursos: {len(self.df)}\n\n")
            
            f.write("RESULTADOS PRINCIPAIS:\n")
            f.write("-"*30 + "\n")
            
            # NIST tests
            nist_ok = all(not r['is_significant'] for r in self.nist_tests.values())
            f.write(f"Testes NIST-like: {'TODOS OK' if nist_ok else 'ANOMALIAS DETECTADAS'}\n")
            
            # FFT
            fft_ok = all(not r['is_anomalous'] for r in self.fft_baseline.values())
            f.write(f"Análise Espectral: {'OK' if fft_ok else 'ANOMALIAS DETECTADAS'}\n")
            
            # Permutation Entropy
            pe_ok = all(abs(r.get('z_score', 0)) <= 3 
            for r in self.permutation_entropy.values())
            f.write(f"Permutation Entropy: {'OK' if pe_ok else 'DIFERENÇAS DETECTADAS'}\n")
            
            f.write("\nCONCLUSÃO:\n")
            if nist_ok and fft_ok and pe_ok:
                f.write("Mega-Sena compatível com processo aleatório.\n")
                f.write("Nenhuma estrutura oculta detectada.\n")
            else:
                f.write("Possíveis anomalias requerem investigação.\n")
        
        print(f"✅ Sumário: {summary_path}")
        
        return json_path


def main():
    """EXECUÇÃO DO FRAMEWORK DE VALIDAÇÃO"""
    print("="*80)
    print("🔬 FRAMEWORK DE VALIDAÇÃO DE ALEATORIEDADE")
    print("   Mega-Sena vs RNG + Loterias Internacionais")
    print("="*80)
    
    # Inicializar
    validator = AleatoriedadeValidator('resultados_megasena.csv')
    
    # Executar bateria completa
    results = validator.run_full_validation_suite()
    
    # Exportar relatório
    validator.export_validation_report()
    
    print("\n" + "="*80)
    print("✅ VALIDAÇÃO CONCLUÍDA!")
    print("📁 Relatórios salvos em: relatorio_validacao/")
    print("="*80)


if __name__ == "__main__":
    main()
