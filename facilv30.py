#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v37
TESTE DE CAPACIDADE PREDITIVA DAS FEATURES

NOVO:
✅ Opção 11: Teste de capacidade preditiva (walk‑forward com modelo)
   - Para cada feature alvo, treina um modelo linear e compara com baseline
   - Métricas: RMSE, MAE, redução vs. baseline, significância estatística
✅ Mantém todas as funcionalidades do v36.2
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon, pearsonr
from scipy.spatial.distance import cdist
from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.multitest import multipletests
from collections import Counter
from itertools import combinations
import os, random, time, warnings
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Algumas funções estarão indisponíveis.")

# ============================================================
# CONSTANTES GERAIS
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude", "compressao",
]
N_FEATURES = len(FEATURE_NAMES)

TEMPORAL_FEATURES = {
    'repeticoes': lambda d, prev: len(set(d) & prev) if prev else 8,
    'soma': lambda d, prev: sum(d),
    'pares': lambda d, prev: sum(1 for x in d if x%2==0),
    'primos': lambda d, prev: sum(1 for x in d if x in PRIMES),
    'clusterizacao': lambda d, prev: sum(1 for i in range(len(d)-1) if d[i+1]-d[i]<=2)/14,
    'gaps': lambda d, prev: np.mean([d[i+1]-d[i] for i in range(len(d)-1)]),
}

# Features que tentaremos prever
PREDICTABLE_FEATURES = ['pares', 'primos', 'moldura', 'soma', 'repeticoes', 'consecutivos', 'amplitude']

# Estruturais
MAX_CONSECUTIVOS_RUN = 7
STRUCTURAL_TARGETS = {
    'pares': (7.5, 3.0, 0.5),
    'primos': (5.0, 3.0, 0.5),
    'moldura': (9.5, 3.0, 0.3),
    'repeticoes': (9.0, 4.0, 0.3),
    'soma': (195.0, 40.0, 0.1),
    'consecutivos': (5.5, 5.0, 0.1),
    'amplitude': (22.0, 6.0, 0.1),
}
SOFT_PENALTY_WEIGHT = 0.02

# Cobertura
DEFAULT_MAX_INTERSECTION = 8
DEFAULT_HAMMING_MIN_DIST = 5
MAX_PAIR_COVERAGE = 0.95

EXPONENTIAL_WEIGHTS = {
    11: 1.0,
    12: 5.0,
    13: 50.0,
    14: 500.0,
    15: 50000.0,
}

# ============================================================
# BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}
    def get_mask(self, game):
        key = tuple(game) if isinstance(game, list) else game
        if key not in self._cache:
            mask = 0
            for d in key:
                mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]

BITMASK_CACHE = BitmaskCache()
mask_intersection = lambda m1, m2: (m1 & m2).bit_count()
hamming_distance = lambda g1, g2: 15 - mask_intersection(BITMASK_CACHE.get_mask(g1), BITMASK_CACHE.get_mask(g2))

def draw_masks_to_array(draws):
    return np.array([BITMASK_CACHE.get_mask(d) for d in draws], dtype=np.uint32)

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        return None
    contests = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17: continue
            try:
                dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                if len(dezenas) != 15 or len(set(dezenas)) != 15: continue
                if any(x < 1 or x > 25 for x in dezenas): continue
                contests.append({'concurso': int(parts[0]), 'data': parts[1], 'dezenas': sorted(dezenas)})
            except: continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos válidos")
    return contests

# ============================================================
# EXTRATOR DE FEATURES (mantido)
# ============================================================
class FeatureExtractor:
    def __init__(self, contests):
        self.contests = contests
        self._repeat_history = []
        for i, c in enumerate(contests):
            if i > 0:
                self._repeat_history.append(len(set(contests[i-1]['dezenas']) & set(c['dezenas'])))
            else:
                self._repeat_history.append(0)
        raw_features = self._build_raw_feature_matrix()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        if self.scaler is not None and len(raw_features) > 10:
            self.scaler.fit(raw_features)
        if self.scaler is not None:
            self.standardized_features = self.scaler.transform(raw_features)
        else:
            self.feature_means = np.mean(raw_features, axis=0)
            self.feature_stds = np.std(raw_features, axis=0) + 1e-10
            self.standardized_features = (raw_features - self.feature_means) / self.feature_stds
        self._build_multivariate_model(self.standardized_features)

    def _build_raw_feature_matrix(self):
        features_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            features_list.append(self._extract_raw(c['dezenas'], last))
        return np.array(features_list, dtype=np.float64)

    def _extract_raw(self, dezenas, last_contest=None):
        d = sorted(dezenas); gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        rep = len(set(d) & set(last_contest)) if last_contest else 8
        ent_trans = 0.0
        if len(self._repeat_history) >= 5:
            trans = [self._repeat_history[i+1]-self._repeat_history[i] for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs>0, probs, 1e-10)))
        amplitude = max(d)-min(d); std_pos = np.std(d) if len(d)>1 else 0.0
        compressao = std_pos/amplitude if amplitude>0 else 0.5
        return np.array([
            float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps)),
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))), ent_trans,
            float(len(set((x-1)//5 for x in d))),
            float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),
            float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),
            float(np.mean(d)-np.median(d)),
            float(sum(1 for g in gaps if g<=2)/len(gaps)), float(rep),
            float(sum(1 for x in d if x%2==0)),
            float(sum(1 for x in d if x in PRIMES)),
            float(sum(1 for x in d if x in MOLDURA)),
            float(sum(d)), float(max(d)-min(d)), compressao,
        ], dtype=np.float64)

    def _build_multivariate_model(self, standardized_features):
        if SKLEARN_AVAILABLE and len(standardized_features) > N_FEATURES:
            try:
                lw = LedoitWolf().fit(standardized_features)
                self.precision_matrix = lw.precision_
                self.cov_matrix = lw.covariance_
            except:
                cov = np.cov(standardized_features.T) + np.eye(N_FEATURES) * 1e-6
                self.precision_matrix = np.linalg.inv(cov)
                self.cov_matrix = cov
        else:
            cov = np.cov(standardized_features.T) + np.eye(N_FEATURES) * 1e-6
            self.precision_matrix = np.linalg.inv(cov)
            self.cov_matrix = cov
        self._mean_vector = np.mean(standardized_features, axis=0)
        self.historical_mahalanobis = self.mahalanobis_batch(standardized_features)

    def mahalanobis_batch(self, fmatrix):
        diff = fmatrix - self._mean_vector
        temp = np.dot(diff, self.precision_matrix)
        return np.sqrt(np.maximum(0, np.sum(temp * diff, axis=1)))

    def compute_rarity_scores_batch(self, fmatrix):
        dists = self.mahalanobis_batch(fmatrix)
        percentiles = np.array([np.mean(self.historical_mahalanobis <= d) for d in dists])
        scores = np.ones_like(percentiles)
        mask_low = percentiles <= 0.8
        scores[mask_low] = percentiles[mask_low] / 0.8
        mask_high = percentiles > 0.99
        excess = (percentiles[mask_high] - 0.99) / 0.01
        scores[mask_high] = 1.0 - excess * 5.0
        return np.clip(scores, 0, 1), percentiles, dists

    def extract_features(self, game, last_contest=None):
        key = (tuple(sorted(game)), tuple(last_contest) if last_contest else None)
        if not hasattr(self, '_feature_cache'): self._feature_cache = {}
        if key not in self._feature_cache:
            raw = self._extract_raw(game, last_contest)
            if self.scaler is not None:
                scaled = self.scaler.transform(raw.reshape(1, -1)).flatten()
            else:
                scaled = (raw - self.feature_means) / self.feature_stds
            self._feature_cache[key] = scaled
        return self._feature_cache[key]

    def extract_features_batch(self, games, last_contest=None):
        return np.array([self.extract_features(g, last_contest) for g in games])

    def compute_structural_penalty(self, game):
        d = sorted(game)
        penalty = 0.0
        actuals = {
            'pares': sum(1 for x in d if x%2==0),
            'primos': sum(1 for x in d if x in PRIMES),
            'moldura': sum(1 for x in d if x in MOLDURA),
            'repeticoes': len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8,
            'soma': sum(d),
            'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),
            'amplitude': max(d)-min(d),
        }
        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name]-target)
                if dev > tol:
                    penalty += (dev - tol) * w
        max_run = run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run += 1; max_run = max(max_run, run)
            else: run = 1
        if max_run > MAX_CONSECUTIVOS_RUN:
            penalty += (max_run - MAX_CONSECUTIVOS_RUN) * 2.0
        return penalty

# ============================================================
# GERADOR COM CRITÉRIOS
# ============================================================
class LooseGenerator:
    def __init__(self, extractor=None):
        self.extractor = extractor

    def generate_one(self, max_penalty=30, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        for _ in range(500):
            game = self._generate_raw(allowed_pares, allowed_moldura, allowed_primos)
            if game is None:
                continue
            if self.extractor is not None:
                pen = self.extractor.compute_structural_penalty(game)
                if pen <= max_penalty:
                    return game
            else:
                return game
        raise RuntimeError(
            f"Não foi possível gerar jogo com os critérios: "
            f"pares={allowed_pares}, moldura={allowed_moldura}, primos={allowed_primos}"
        )

    def _generate_raw(self, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        if allowed_pares is None and allowed_moldura is None and allowed_primos is None:
            return self._generate_raw_old()
        for _ in range(200):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if allowed_pares is not None:
                pares = sum(1 for x in game if x % 2 == 0)
                if pares not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                mol = sum(1 for x in game if x in MOLDURA)
                if mol not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                prim = sum(1 for x in game if x in PRIMES)
                if prim not in allowed_primos:
                    continue
            return game
        return None

    def _generate_raw_old(self):
        game = set()
        available = set(range(1, 26))
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                test = sorted(game | {d})
                quad_score = len(set((x-1)//5 for x in test)) * 1.2
                if len(test) > 1:
                    gaps = [test[i+1]-test[i] for i in range(len(test)-1)]
                    cluster_penalty = sum(1 for g in gaps if g <= 2) * 1.8
                else: cluster_penalty = 0.0
                run = 1; max_run = 1
                for i in range(len(test)-1):
                    if test[i+1]-test[i]==1: run += 1; max_run = max(max_run, run)
                    else: run = 1
                consec_penalty = max(0, max_run - 4) * 5
                scores.append(quad_score - cluster_penalty - consec_penalty)
            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores -= np.max(scores)
                probs = np.exp(scores / 2.0)
                probs /= probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else: chosen = random.choice(candidates)
            game.add(chosen); available.remove(chosen)
        return sorted(game)[:15]

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# FUNÇÕES DE DEPENDÊNCIA TEMPORAL (inalteradas)
# ============================================================
def compute_temporal_features(contests):
    series = {name: [] for name in TEMPORAL_FEATURES}
    for i, c in enumerate(contests):
        prev = set(contests[i-1]['dezenas']) if i > 0 else None
        for name, func in TEMPORAL_FEATURES.items():
            val = func(c['dezenas'], prev)
            if name == 'gaps': val = round(val, 2)
            elif name == 'clusterizacao': val = round(val, 3)
            elif name == 'soma': val = val // 5 * 5
            series[name].append(val)
    return series

def mutual_information_feature(series_x, series_y):
    joint = Counter(zip(series_x, series_y))
    total = sum(joint.values())
    x_arr = np.asarray(series_x); y_arr = np.asarray(series_y)
    mi = 0.0
    for (x_val, y_val), count in joint.items():
        p_xy = count / total
        p_x = np.mean(x_arr == x_val)
        p_y = np.mean(y_arr == y_val)
        if p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi

def permutation_mi_test(series_t, lag=1, n_perm=500):
    x = series_t[lag:]; y = series_t[:-lag]
    mi_obs = mutual_information_feature(x, y)
    mi_null = np.zeros(n_perm)
    y_arr = np.asarray(y)
    for i in range(n_perm):
        y_perm = np.random.permutation(y_arr)
        mi_null[i] = mutual_information_feature(x, y_perm)
    adjusted_mi = mi_obs - np.mean(mi_null)
    p_value = np.mean(mi_null >= mi_obs)
    return mi_obs, adjusted_mi, p_value, mi_null

def autocorrelation_test(series_t, lag=1):
    x = np.array(series_t[lag:], dtype=float)
    y = np.array(series_t[:-lag], dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return 0.0, 1.0
    corr, p_val = pearsonr(x, y)
    return corr, p_val

def partial_autocorrelation(series_t, nlags=5):
    x = np.array(series_t, dtype=float)
    if len(x) < nlags + 1: return np.zeros(nlags)
    try: return pacf(x, nlags=nlags)
    except: return np.zeros(nlags)

def fft_analysis(series_t):
    x = np.array(series_t, dtype=float); x = x - np.mean(x)
    freqs, power = periodogram(x)
    idx = np.argmax(power[1:]) + 1
    return freqs[idx], power[idx], freqs, power

def generate_synthetic_contests(n):
    return [{'concurso': i, 'data': '', 'dezenas': sorted(np.random.choice(range(1,26), 15, replace=False))} for i in range(n)]

def compare_real_vs_synthetic(contests, n_synthetic=None):
    if n_synthetic is None: n_synthetic = len(contests)
    print(f"\n🔄 Gerando {n_synthetic} concursos sintéticos (i.i.d.)...")
    synthetic = generate_synthetic_contests(n_synthetic)
    print("\n📊 INFORMAÇÃO MÚTUA AJUSTADA (X_t; X_{{t-1}}) POR FEATURE:")
    print(f"{'Feature':<15} {'Real MI_adj':<12} {'Real p':<10} {'Sintético MI_adj':<15} {'Sintético p':<10}")
    print("-" * 65)
    real_series = compute_temporal_features(contests)
    synth_series = compute_temporal_features(synthetic)
    real_pvals = []
    for name in TEMPORAL_FEATURES:
        _, mi_adj_real, p_real, _ = permutation_mi_test(real_series[name], lag=1)
        _, mi_adj_synth, p_synth, _ = permutation_mi_test(synth_series[name], lag=1)
        real_pvals.append(p_real)
        print(f"{name:<15} {mi_adj_real:<12.4f} {p_real:<10.3f} {mi_adj_synth:<15.4f} {p_synth:<10.3f}")
    _, fdr_pvals, _, _ = multipletests(real_pvals, method='fdr_bh')
    print("\n🔍 Após correção FDR (Benjamini-Hochberg):")
    for i, name in enumerate(TEMPORAL_FEATURES):
        sig = "⚠️" if fdr_pvals[i] <= 0.05 else "não"
        print(f"   {name}: p_raw={real_pvals[i]:.3f}, p_fdr={fdr_pvals[i]:.3f} -> {sig}")
    if SKLEARN_AVAILABLE:
        ext_real = FeatureExtractor(contests)
        ext_synth = FeatureExtractor(synthetic)
        real_pca = pca_analysis(ext_real.standardized_features)
        synth_pca = pca_analysis(ext_synth.standardized_features)
        if real_pca is not None and synth_pca is not None:
            print("\n📊 PCA (variância explicada pelas 3 primeiras PCs):")
            print(f"   Real: PC1={real_pca[0]:.2%}, PC2={real_pca[1]:.2%}, PC3={real_pca[2]:.2%}, Soma={np.sum(real_pca):.2%}")
            print(f"   Sintético: PC1={synth_pca[0]:.2%}, PC2={synth_pca[1]:.2%}, PC3={synth_pca[2]:.2%}, Soma={np.sum(synth_pca):.2%}")
            if np.sum(real_pca) < 0.5: print("   ➡️ Variância explicada baixa: espaço aproximadamente isotrópico.")
    else: print("\n⚠️ PCA indisponível.")
    print("\n📊 DISTRIBUIÇÃO DE GAPS (média):")
    for label, data in [("Real", contests), ("Sintético", synthetic)]:
        gaps_all = []
        for c in data:
            d = sorted(c['dezenas'])
            gaps_all.extend([d[i+1]-d[i] for i in range(len(d)-1)])
        print(f"   {label}: média={np.mean(gaps_all):.2f}, desvio={np.std(gaps_all):.2f}")
    print("\n📊 DISTÂNCIA DE MAHALANOBIS MÉDIA:")
    try:
        ext = FeatureExtractor(contests)
        real_f = ext.standardized_features
        synth_f_raw = ext._build_raw_feature_matrix_from_contests(synthetic)
        if ext.scaler is not None: synth_f = ext.scaler.transform(synth_f_raw)
        else: synth_f = (synth_f_raw - ext.feature_means) / ext.feature_stds
        print(f"   Real: {np.mean(ext.mahalanobis_batch(real_f)):.2f}, Sintético: {np.mean(ext.mahalanobis_batch(synth_f)):.2f}")
    except Exception as e: print(f"   Não foi possível calcular: {e}")
    print("\n🔍 Se todos os valores forem muito próximos, reforça a hipótese de pseudoaleatoriedade efetiva.")

def _build_raw_feature_matrix_from_contests(self, contests_list):
    feats = []
    for i, c in enumerate(contests_list):
        last = set(contests_list[i-1]['dezenas']) if i > 0 else None
        feats.append(self._extract_raw(c['dezenas'], last))
    return np.array(feats, dtype=np.float64)

FeatureExtractor._build_raw_feature_matrix_from_contests = _build_raw_feature_matrix_from_contests

def pca_analysis(feature_matrix):
    if not SKLEARN_AVAILABLE: return None
    pca = PCA(n_components=min(5, feature_matrix.shape[1]))
    pca.fit(feature_matrix)
    return pca.explained_variance_ratio_[:3]

# ============================================================
# NOVAS FUNÇÕES DO v36 (mantidas)
# ============================================================
def count_universe(allowed_pares=None, allowed_moldura=None, allowed_primos=None):
    print(f"\n📊 ESTIMATIVA DO UNIVERSO COMBINATÓRIO")
    print(f"   Critérios: Pares={allowed_pares}, Moldura={allowed_moldura}, Primos={allowed_primos}")
    n_amostra = 500000
    count_valid = 0
    for _ in tqdm(range(n_amostra), desc="Estimando universo"):
        game = sorted(np.random.choice(range(1, 26), 15, replace=False))
        valid = True
        if allowed_pares is not None:
            if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                valid = False
        if allowed_moldura is not None and valid:
            if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                valid = False
        if allowed_primos is not None and valid:
            if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                valid = False
        if valid:
            count_valid += 1
    proporcao = count_valid / n_amostra
    total_universe = comb(25, 15)
    estimativa = int(total_universe * proporcao)
    print(f"   Proporção na amostra: {proporcao:.4%}")
    print(f"   Universo total (C(25,15)): {total_universe:,}")
    print(f"   Universo estimado com critérios: {estimativa:,}")
    print(f"   Redução: {(1 - proporcao)*100:.1f}%")
    if estimativa < 50000:
        print("   ⚠️ Universo pequeno – critérios muito restritivos podem limitar cobertura")
    elif estimativa > 500000:
        print("   ℹ️ Universo ainda grande – critérios pouco restritivos")
    return estimativa, proporcao

def analyze_historical_criteria(contests, n_recent=500, coverage_pct=0.80):
    print(f"\n📊 ANÁLISE DE CRITÉRIOS NOS ÚLTIMOS {n_recent} CONCURSOS")
    recent = contests[-n_recent:] if n_recent < len(contests) else contests
    pares_dist = Counter()
    moldura_dist = Counter()
    primos_dist = Counter()
    for c in recent:
        d = c['dezenas']
        pares_dist[sum(1 for x in d if x % 2 == 0)] += 1
        moldura_dist[sum(1 for x in d if x in MOLDURA)] += 1
        primos_dist[sum(1 for x in d if x in PRIMES)] += 1
    total = len(recent)
    print("\n📈 DISTRIBUIÇÃO DE PARES:")
    for k in sorted(pares_dist.keys()):
        pct = pares_dist[k] / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {k} pares: {pares_dist[k]:4d} ({pct:5.1f}%) {bar}")
    print("\n📈 DISTRIBUIÇÃO DE MOLDURA:")
    for k in sorted(moldura_dist.keys()):
        pct = moldura_dist[k] / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {k} moldura: {moldura_dist[k]:4d} ({pct:5.1f}%) {bar}")
    print("\n📈 DISTRIBUIÇÃO DE PRIMOS:")
    for k in sorted(primos_dist.keys()):
        pct = primos_dist[k] / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {k} primos: {primos_dist[k]:4d} ({pct:5.1f}%) {bar}")
    def suggest(values_dist, name):
        sorted_items = sorted(values_dist.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0
        suggested = []
        for val, count in sorted_items:
            cumulative += count
            suggested.append(val)
            if cumulative / total >= coverage_pct:
                break
        return sorted(suggested)
    suggested_pares = suggest(pares_dist, "pares")
    suggested_moldura = suggest(moldura_dist, "moldura")
    suggested_primos = suggest(primos_dist, "primos")
    print(f"\n💡 SUGESTÃO DE CRITÉRIOS (cobertura ≥ {coverage_pct*100:.0f}%):")
    print(f"   Pares: {suggested_pares}")
    print(f"   Moldura: {suggested_moldura}")
    print(f"   Primos: {suggested_primos}")
    count_universe(suggested_pares, suggested_moldura, suggested_primos)
    return suggested_pares, suggested_moldura, suggested_primos

def historical_exact_frequency(contests, allowed_pares, allowed_moldura, allowed_primos):
    single_pares = allowed_pares if (allowed_pares and len(allowed_pares)==1) else None
    single_mol = allowed_moldura if (allowed_moldura and len(allowed_moldura)==1) else None
    single_prim = allowed_primos if (allowed_primos and len(allowed_primos)==1) else None
    count = 0
    for c in contests:
        d = c['dezenas']
        p = sum(1 for x in d if x%2==0)
        m = sum(1 for x in d if x in MOLDURA)
        pr = sum(1 for x in d if x in PRIMES)
        ok = True
        if single_pares is not None and p != single_pares[0]:
            ok = False
        elif allowed_pares and p not in allowed_pares:
            ok = False
        if ok and single_mol is not None and m != single_mol[0]:
            ok = False
        elif allowed_moldura and m not in allowed_moldura:
            ok = False
        if ok and single_prim is not None and pr != single_prim[0]:
            ok = False
        elif allowed_primos and pr not in allowed_primos:
            ok = False
        if ok:
            count += 1
    prop = count / len(contests)
    return count, prop

# ============================================================
# TESTE DE CAPACIDADE PREDITIVA (NOVO v37)
# ============================================================
def predictive_test(contests, n_windows=20, train_size=300, test_size=1):
    """
    Para cada feature alvo em PREDICTABLE_FEATURES, faz walk‑forward:
    - Treina um modelo linear com features do concurso anterior
    - Compara com baseline (último valor)
    - Mede redução de erro e significância
    """
    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn necessário para este teste.")
        return

    print(f"\n🔮 TESTE DE CAPACIDADE PREDITIVA DAS FEATURES")
    print(f"   Walk‑forward: {n_windows} janelas, treino={train_size}, teste={test_size}")
    print(f"   Modelo: regressão linear com features do concurso anterior")
    print(f"   Baseline: repetir o último valor observado\n")

    # Extrair todas as features temporais
    series = compute_temporal_features(contests)
    
    # Para cada feature, construir matriz de features (t-1) e target (t)
    results = {}
    for target_name in PREDICTABLE_FEATURES:
        if target_name not in series:
            continue
        
        y_all = np.array(series[target_name], dtype=float)
        
        # Features preditoras: todas as outras features temporais no lag 1
        predictor_names = [n for n in PREDICTABLE_FEATURES if n != target_name]
        X_all = np.column_stack([np.array(series[n], dtype=float) for n in predictor_names])
        
        errors_model = []
        errors_baseline = []
        
        for w in range(n_windows):
            test_end = len(contests) - w * test_size
            test_start = test_end - test_size
            train_end = test_start
            train_start = max(0, train_end - train_size)
            
            if train_start + 1 >= train_end or test_start + 1 >= test_end:
                continue
            
            # Treino: X[train_start:train_end-1] -> y[train_start+1:train_end]
            X_train = X_all[train_start:train_end-1]
            y_train = y_all[train_start+1:train_end]
            
            # Teste: X[test_start:test_end-1] -> y[test_start+1:test_end]
            X_test = X_all[test_start:test_end-1]
            y_test = y_all[test_start+1:test_end]
            
            if len(X_train) < 10 or len(X_test) < 1:
                continue
            
            # Modelo linear
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Baseline: último valor conhecido
            baseline_preds = y_all[test_start:test_end-1]
            
            errors_model.extend(np.abs(preds - y_test).tolist())
            errors_baseline.extend(np.abs(baseline_preds - y_test).tolist())
        
        if len(errors_model) == 0:
            continue
        
        errors_model = np.array(errors_model)
        errors_baseline = np.array(errors_baseline)
        
        mae_model = np.mean(errors_model)
        mae_baseline = np.mean(errors_baseline)
        rmse_model = np.sqrt(np.mean(errors_model**2))
        rmse_baseline = np.sqrt(np.mean(errors_baseline**2))
        
        # Redução percentual do erro
        reduction = (mae_baseline - mae_model) / mae_baseline * 100
        
        # Teste estatístico nos erros absolutos
        diff = errors_baseline - errors_model
        try:
            _, p_value = wilcoxon(errors_model, errors_baseline)
        except:
            p_value = 1.0
        
        results[target_name] = {
            'mae_model': mae_model,
            'mae_baseline': mae_baseline,
            'rmse_model': rmse_model,
            'rmse_baseline': rmse_baseline,
            'reduction_pct': reduction,
            'p_value': p_value,
            'n_tests': len(errors_model)
        }
    
    # Exibir resultados
    print(f"{'Feature':<15} {'MAE model':<10} {'MAE base':<10} {'Redução':<10} {'p-valor':<10} {'Conclusão'}")
    print("-" * 70)
    for name, res in sorted(results.items(), key=lambda x: x[1]['reduction_pct'], reverse=True):
        if res['reduction_pct'] > 0 and res['p_value'] < 0.05:
            conclusao = "🔍 Promissor"
        elif res['reduction_pct'] > 0:
            conclusao = "📊 Leve (não sig.)"
        else:
            conclusao = "❌ Sem sinal"
        print(f"{name:<15} {res['mae_model']:<10.4f} {res['mae_baseline']:<10.4f} {res['reduction_pct']:<10.1f}% {res['p_value']:<10.4f} {conclusao}")
    
    # Verificar se alguma feature passou
    promising = [n for n, r in results.items() if r['reduction_pct'] > 0 and r['p_value'] < 0.05]
    print(f"\n🔍 Features com sinal preditivo significativo: {len(promising)}")
    if promising:
        print(f"   {promising}")
    else:
        print("   Nenhuma feature mostrou capacidade preditiva robusta fora da amostra.")
        print("   Isso reforça a hipótese de que a Lotofácil se comporta como um processo")
        print("   pseudoaleatório sem dependência temporal explorável.")
    
    return results

# ============================================================
# OTIMIZADOR v36.2 (mantido)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = LooseGenerator(self.extractor)
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos
        self.historical_masks = draw_masks_to_array([c['dezenas'] for c in self.contests])
        if len(self.historical_masks) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_masks))]
            self.historical_masks = np.concatenate([self.historical_masks, draw_masks_to_array(extra)])

    def _validate_game(self, game):
        d = sorted(game)
        pares = sum(1 for x in d if x % 2 == 0)
        mol = sum(1 for x in d if x in MOLDURA)
        prim = sum(1 for x in d if x in PRIMES)
        if self.allowed_pares is not None and pares not in self.allowed_pares:
            return False
        if self.allowed_moldura is not None and mol not in self.allowed_moldura:
            return False
        if self.allowed_primos is not None and prim not in self.allowed_primos:
            return False
        return True

    def _create_candidate(self, game, rarity_score=None, percentile=None, mahal_dist=None, validate=True):
        if validate and not self._validate_game(game):
            raise ValueError(f"Jogo inválido gerado: {game}")
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        if rarity_score is None:
            rarity_score, percentile, mahal_dist = self.extractor.compute_rarity_scores_batch(features.reshape(1, -1))
            rarity_score = rarity_score[0]; percentile = percentile[0]; mahal_dist = mahal_dist[0]
        pen = self.extractor.compute_structural_penalty(game)
        return GameCandidate(game, mask, features, rarity_score, rarity_score, mahal_dist, percentile, pen)

    def _decade_coverage(self, portfolio):
        freq = np.bincount([d for c in portfolio for d in c.game], minlength=26)[1:]
        target = len(portfolio) * 15 / 25
        return -np.std(freq - target)

    def _unique_quads(self, portfolio):
        quads = set()
        for c in portfolio: quads.update(combinations(sorted(c.game), 4))
        return len(quads)

    def _unique_triples(self, portfolio):
        all_triples = set()
        for c in portfolio: all_triples.update(combinations(sorted(c.game), 3))
        return len(all_triples)

    def _pair_redundancy(self, portfolio):
        pair_counts = Counter()
        for c in portfolio: pair_counts.update(combinations(sorted(c.game), 2))
        redundant = sum(max(0, cnt-1) for cnt in pair_counts.values())
        max_possible = (len(portfolio)-1) * comb(15, 2)
        return redundant / max_possible if max_possible > 0 else 0

    def _portfolio_entropy(self, portfolio):
        freq = np.bincount([d for c in portfolio for d in c.game], minlength=26)[1:]
        probs = freq / np.sum(freq); probs = np.where(probs > 0, probs, 1e-10)
        return entropy(probs) / np.log(25)

    def _max_overlap(self, portfolio):
        max_inter = 0
        masks = [c.mask for c in portfolio]
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                inter = mask_intersection(masks[i], masks[j])
                if inter > max_inter:
                    max_inter = inter
        return max_inter

    def _monte_carlo_score(self, portfolio, n_sim=500):
        portfolio_masks = np.array([c.mask for c in portfolio], dtype=np.uint32)
        if len(self.historical_masks) > n_sim:
            idx = np.random.choice(len(self.historical_masks), n_sim, replace=False)
        else: idx = np.arange(len(self.historical_masks))
        total = 0.0
        for dm in self.historical_masks[idx]:
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11: total += EXPONENTIAL_WEIGHTS.get(hits, 0)
        return total / n_sim

    def _portfolio_score(self, portfolio, max_inter=DEFAULT_MAX_INTERSECTION):
        pair_cov = len(set(p for c in portfolio for p in combinations(sorted(c.game), 2))) / comb(25,2)
        if pair_cov > MAX_PAIR_COVERAGE: return -1000.0
        triples = self._unique_triples(portfolio)
        triple_score = triples / (len(portfolio) * comb(15, 3))
        quads = self._unique_quads(portfolio)
        quad_score = quads / (len(portfolio) * comb(15, 4))
        ent_score = self._portfolio_entropy(portfolio)
        redundancy = self._pair_redundancy(portfolio)
        decade_cov = self._decade_coverage(portfolio)
        max_overlap = self._max_overlap(portfolio)
        avg_rarity = np.mean([c.rarity_score for c in portfolio])
        raw_mc = self._monte_carlo_score(portfolio)
        if not hasattr(self, '_mc_bounds'): self._mc_bounds = self._compute_mc_bounds()
        p5, p95 = self._mc_bounds
        mc_norm = max(0.0, min(1.0, (raw_mc - p5) / (p95 - p5 + 1e-10)))
        avg_penalty = np.mean([c.penalty for c in portfolio])
        overlap_penalty = max(0, max_overlap - max_inter) * 0.3
        return (triple_score * 0.20 + quad_score * 0.20 + ent_score * 0.15 +
                decade_cov * 0.10 + mc_norm * 0.25 + avg_rarity * 0.10 -
                redundancy * 0.5 - avg_penalty * SOFT_PENALTY_WEIGHT - overlap_penalty)

    def _compute_mc_bounds(self):
        scores = []
        for _ in range(30):
            rand_games = [self.generator.generate_pure_random() for _ in range(5)]
            rand_candidates = [self._create_candidate(g, validate=False) for g in rand_games]
            scores.append(self._monte_carlo_score(rand_candidates, 300))
        scores = np.array(scores)
        return np.percentile(scores, 5), np.percentile(scores, 95)

    def _select_max_diversity(self, candidates, n_select, max_inter, hamming_min):
        unique = {}
        for c in candidates:
            if c.mask not in unique:
                unique[c.mask] = c
        cand_unique = list(unique.values())
        if len(cand_unique) < n_select:
            cand_unique = candidates[:n_select]
        masks = np.array([c.mask for c in cand_unique], dtype=np.uint32)
        n = len(cand_unique)
        selected_idx = [0]
        for _ in range(n_select - 1):
            min_dists = np.full(n, np.inf, dtype=np.float64)
            for idx in selected_idx:
                intersect = np.array([mask_intersection(masks[i], masks[idx]) for i in range(n)])
                dist = 15.0 - intersect
                dist[intersect > max_inter] = -999.0
                min_dists = np.minimum(min_dists, dist)
            min_dists[selected_idx] = -1.0
            valid = np.where(min_dists >= 0)[0]
            if len(valid) == 0:
                break
            next_idx = valid[np.argmax(min_dists[valid])]
            selected_idx.append(next_idx)
        selected = [cand_unique[i] for i in selected_idx]
        if len(selected) < n_select:
            for c in candidates:
                if len(selected) >= n_select: break
                if c in selected: continue
                if any(mask_intersection(c.mask, s.mask) > max_inter for s in selected): continue
                selected.append(c)
        return selected[:n_select]

    def _select_diverse_portfolio(self, candidates, n_games, use_hamming=True, use_max_diversity=True,
                                  max_inter=DEFAULT_MAX_INTERSECTION, hamming_min=DEFAULT_HAMMING_MIN_DIST):
        if use_max_diversity and len(candidates) >= n_games:
            port = self._select_max_diversity(candidates, n_games, max_inter, hamming_min)
            if len(port) == n_games:
                return port
        selected, masks, games = [], [], []
        for c in candidates:
            if len(selected) >= n_games: break
            if masks and any(mask_intersection(c.mask, m) > max_inter for m in masks): continue
            if use_hamming and games:
                if any(hamming_distance(c.game, g) < hamming_min for g in games): continue
            selected.append(c); masks.append(c.mask); games.append(c.game)
        while len(selected) < n_games:
            for c in candidates:
                if c not in selected:
                    selected.append(c); break
        return selected

    def _build_portfolio_with_relaxation(self, candidates, n_games):
        portfolio = self._select_diverse_portfolio(candidates, n_games,
                                                   use_hamming=True, use_max_diversity=True,
                                                   max_inter=DEFAULT_MAX_INTERSECTION,
                                                   hamming_min=DEFAULT_HAMMING_MIN_DIST)
        if len(portfolio) == n_games:
            return portfolio
        for max_inter in range(DEFAULT_MAX_INTERSECTION+1, 12):
            for hamming_min in range(DEFAULT_HAMMING_MIN_DIST-1, 2, -1):
                portfolio = self._select_diverse_portfolio(candidates, n_games,
                                                           use_hamming=True, use_max_diversity=True,
                                                           max_inter=max_inter, hamming_min=hamming_min)
                if len(portfolio) == n_games:
                    print(f"   ⚠️ Restrições de diversidade relaxadas: MAX_INTERSECTION={max_inter}, HAMMING_MIN_DIST={hamming_min}")
                    return portfolio
        return candidates[:n_games]

    def optimize(self, n_games=5, n_candidates=30000):
        print(f"\n🧩 CARTEIRA COM CRITÉRIOS: {n_games} jogos")
        if self.allowed_pares: print(f"   Pares permitidos: {self.allowed_pares}")
        if self.allowed_moldura: print(f"   Moldura permitida: {self.allowed_moldura}")
        if self.allowed_primos: print(f"   Primos permitidos: {self.allowed_primos}")
        t0 = time.time()
        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    allowed_pares=self.allowed_pares,
                    allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key); raw_pool.append(g)
            except RuntimeError:
                break
        print(f"   Pool gerado: {len(raw_pool)} jogos")
        if len(raw_pool) < n_games * 10:
            print("⚠️ Pool pequeno. Gerando jogos extras (mantendo critérios)...")
            for _ in tqdm(range(n_candidates), desc="Gerando pool extra"):
                try:
                    g = self.generator.generate_one(
                        allowed_pares=self.allowed_pares,
                        allowed_moldura=self.allowed_moldura,
                        allowed_primos=self.allowed_primos
                    )
                    key = tuple(g)
                    if key not in seen:
                        seen.add(key); raw_pool.append(g)
                except RuntimeError:
                    break
        pool = []
        for g in raw_pool:
            if self._validate_game(g):
                pool.append(g)
            if len(pool) >= 5000:
                break
        if len(pool) < n_games:
            raise RuntimeError("Não foi possível gerar jogos suficientes com os critérios fornecidos.")
        print(f"   Pool filtrado (critérios OK): {len(pool)} jogos")
        fmat = self.extractor.extract_features_batch(pool, self.last)
        r_scores, pcts, m_dists = self.extractor.compute_rarity_scores_batch(fmat)
        candidates = []
        for i, game in enumerate(pool):
            mask = BITMASK_CACHE.get_mask(game)
            pen = self.extractor.compute_structural_penalty(game)
            candidates.append(GameCandidate(game, mask, fmat[i], r_scores[i], r_scores[i], m_dists[i], pcts[i], pen))
        candidates.sort(key=lambda c: c.central_score - c.penalty * 0.1, reverse=True)
        portfolio = self._build_portfolio_with_relaxation(candidates, n_games)
        for c in portfolio:
            if not self._validate_game(c.game):
                raise RuntimeError(f"Jogo inválido no portfólio final: {c.game}")
        best_score = self._portfolio_score(portfolio)
        print(f"✅ Otimizado em {time.time()-t0:.1f}s")
        print(f"   Máxima interseção entre jogos: {self._max_overlap(portfolio)}")
        return [c.game for c in portfolio], best_score

    def backtest(self, portfolio, test_draws):
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {'empirical': prob, 'theoretical': theo_prob,
                'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts}

    def diagnostic_distribution(self, portfolio, test_draws):
        n_jogos = len(portfolio)
        n_test = len(test_draws)
        total_sim = n_jogos * n_test
        hit_counts = {k:0 for k in range(11,16)}
        max_hits = []
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            max_hit = 0
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits > max_hit: max_hit = hits
                if hits >= 11: hit_counts[hits] += 1
            max_hits.append(max_hit)
        expected = {k: total_sim * HYPE_PROBS.get(k, 0) for k in range(11,16)}
        print(f"\n📊 DIAGNÓSTICO DE DISTRIBUIÇÃO (n={total_sim} tentativas):")
        print(f"{'Acertos':<8} {'Observado':<10} {'Esperado':<10} {'Razão O/E':<10}")
        print("-" * 40)
        for k in range(11,16):
            obs = hit_counts.get(k, 0)
            exp = expected[k]
            ratio = obs/exp if exp > 0 else float('inf')
            flag = " ⚠️" if ratio < 0.5 or ratio > 2.0 else ""
            print(f"{k:<8} {obs:<10} {exp:<10.1f} {ratio:<10.2f}{flag}")
        print(f"\n📊 MAX_HIT POR CONCURSO:")
        for k in range(11,16):
            count = sum(1 for h in max_hits if h == k)
            print(f"   {k} pontos: {count} concursos (máximo entre os {n_jogos} jogos)")
        return hit_counts, expected

    def backtest_criteria(self, allowed_pares, allowed_moldura, allowed_primos, test_draws=None):
        if test_draws is None:
            test_draws = self.contests[-200:]
        print(f"\n📊 BACKTEST DE CRITÉRIOS – TETO TEÓRICO (últimos {len(test_draws)} concursos)")
        print(f"   ⚠️ Interpretação: mostra o MÁXIMO que alguém conseguiria se soubesse")
        print(f"      exatamente qual jogo jogar DENTRO do subconjunto. Não é uma estratégia.")
        print(f"   Pares: {allowed_pares}")
        print(f"   Moldura: {allowed_moldura}")
        print(f"   Primos: {allowed_primos}")
        print(f"   Gerando 5000 jogos com esses critérios para cada concurso...")
        pool_jogos = []
        seen = set()
        for _ in range(10000):
            try:
                g = self.generator.generate_one(
                    allowed_pares=allowed_pares,
                    allowed_moldura=allowed_moldura,
                    allowed_primos=allowed_primos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool_jogos.append(g)
                if len(pool_jogos) >= 5000:
                    break
            except RuntimeError:
                break
        pool_masks = np.array([BITMASK_CACHE.get_mask(g) for g in pool_jogos], dtype=np.uint32)
        print(f"   Pool de {len(pool_jogos)} jogos únicos gerados")
        hit_counts = {k: 0 for k in range(11, 16)}
        max_hits = []
        total_premio = 0.0
        for draw in tqdm(test_draws, desc="Backtest"):
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            best_hit = 0
            for pm in pool_masks:
                hits = mask_intersection(pm, dm)
                if hits > best_hit:
                    best_hit = hits
                if hits >= 11:
                    hit_counts[hits] += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
            max_hits.append(best_hit)
        total_sim = len(pool_jogos) * len(test_draws)
        expected = {k: total_sim * HYPE_PROBS.get(k, 0) for k in range(11, 16)}
        total_custo = len(pool_jogos) * len(test_draws) * CUSTO_APOSTA
        print(f"\n📊 RESULTADOS (amostra de {len(pool_jogos)} jogos por concurso):")
        print(f"{'Acertos':<8} {'Observado':<10} {'Esperado':<10} {'Razão O/E':<10}")
        print("-" * 40)
        for k in range(11, 16):
            obs = hit_counts.get(k, 0)
            exp = expected[k]
            ratio = obs/exp if exp > 0 else float('inf')
            print(f"{k:<8} {obs:<10} {exp:<10.1f} {ratio:<10.2f}")
        print(f"\n📊 MAX_HIT POR CONCURSO (melhor acerto entre os {len(pool_jogos)} jogos):")
        for k in range(11, 16):
            count = sum(1 for h in max_hits if h == k)
            print(f"   {k} pontos: {count} concursos")
        print(f"\n💰 RESUMO FINANCEIRO (se jogasse TODOS os {len(pool_jogos)} jogos):")
        print(f"   Custo total: R$ {total_custo:,.2f}")
        print(f"   Prêmio total: R$ {total_premio:,.2f}")
        print(f"   ROI: {((total_premio - total_custo) / total_custo * 100):+.1f}%")
        print(f"\n⚠️ LEMBRE-SE: isto é um TETO TEÓRICO, não uma estratégia viável.")
        return hit_counts, max_hits

class GameCandidate:
    __slots__ = ('game','mask','features','rarity_score','central_score',
                 'mahalanobis_dist','rarity_percentile','penalty')
    def __init__(self, game, mask, features, rarity_score=0, central_score=0,
                 mahalanobis_dist=0, rarity_percentile=0, penalty=0):
        self.game = game; self.mask = mask; self.features = features
        self.rarity_score = rarity_score; self.central_score = central_score
        self.mahalanobis_dist = mahalanobis_dist; self.rarity_percentile = rarity_percentile
        self.penalty = penalty

# ============================================================
# WALK-FORWARD (mantido)
# ============================================================
def walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                            use_hybrid=False, n_random_benchmark=100,
                            allowed_pares=None, allowed_moldura=None, allowed_primos=None):
    label = "HÍBRIDO" if use_hybrid else "padrão"
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas) {label}")
    print(f"   Benchmark robusto: {n_random_benchmark} carteiras aleatórias por janela")
    if allowed_pares: print(f"   Critérios: Pares={allowed_pares}")
    if allowed_moldura: print(f"   Critérios: Moldura={allowed_moldura}")
    if allowed_primos: print(f"   Critérios: Primos={allowed_primos}")
    results = []
    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue
        opt = PortfolioOptimizer(train_data, allowed_pares, allowed_moldura, allowed_primos)
        if use_hybrid:
            portfolio = opt.hybrid_portfolio(n_central=1, n_extreme=0, n_balanced=4, n_candidates=10000, use_anti_central=True)
            assert len(portfolio) == 5, f"Híbrido retornou {len(portfolio)} jogos"
            bt = opt.backtest(portfolio, test_data)
            total_random_games = n_games
            strat_games = portfolio
        else:
            portfolio, _ = opt.optimize(n_games, n_candidates=10000)
            bt = opt.backtest(portfolio, test_data)
            total_random_games = n_games
            strat_games = portfolio
        rand_lifts = []
        rand_pairs_list = []
        rand_triples_list = []
        rand_quads_list = []
        for _ in range(n_random_benchmark):
            rand_port = [opt.generator.generate_pure_random() for _ in range(total_random_games)]
            bt_rand = opt.backtest(rand_port, test_data)
            rand_lifts.append(bt_rand['lift'])
            pairs = set(p for game in rand_port for p in combinations(sorted(game), 2))
            triples = set(t for game in rand_port for t in combinations(sorted(game), 3))
            quads = set(q for game in rand_port for q in combinations(sorted(game), 4))
            rand_pairs_list.append(len(pairs))
            rand_triples_list.append(len(triples))
            rand_quads_list.append(len(quads))
        rand_lifts = np.array(rand_lifts)
        mean_rand_lift = np.mean(rand_lifts)
        std_rand_lift = np.std(rand_lifts)
        z_score = (bt['lift'] - mean_rand_lift) / std_rand_lift if std_rand_lift > 0 else 0.0
        pct_rank = np.mean(rand_lifts <= bt['lift'])
        strat_pairs = len(set(p for game in strat_games for p in combinations(sorted(game), 2)))
        strat_triples = len(set(t for game in strat_games for t in combinations(sorted(game), 3)))
        strat_quads = len(set(q for game in strat_games for q in combinations(sorted(game), 4)))
        mean_rand_pairs = np.mean(rand_pairs_list)
        mean_rand_triples = np.mean(rand_triples_list)
        mean_rand_quads = np.mean(rand_quads_list)
        results.append({
            'window': w,
            'strat_lift': bt['lift'],
            'mean_rand_lift': mean_rand_lift,
            'std_rand_lift': std_rand_lift,
            'z_score': z_score,
            'pct_rank': pct_rank,
            'strat_14': bt['hit_distribution'].get(14,0),
            'strat_pairs': strat_pairs,
            'strat_triples': strat_triples,
            'strat_quads': strat_quads,
            'rand_pairs': mean_rand_pairs,
            'rand_triples': mean_rand_triples,
            'rand_quads': mean_rand_quads,
        })
        print(f"   Janela {w}: lift={bt['lift']:.3f} | z={z_score:+.2f} | rank={pct_rank:.2f} | 14pts={bt['hit_distribution'].get(14,0)}")
        print(f"      Cobertura: Pares {strat_pairs}/{mean_rand_pairs:.0f}, Triplas {strat_triples}/{mean_rand_triples:.0f}, Quadras {strat_quads}/{mean_rand_quads:.0f}")
    if results:
        print(f"\n📊 RESUMO FINAL:")
        print(f"   Média lift estratégia: {np.mean([r['strat_lift'] for r in results]):.3f}")
        print(f"   Média lift aleatório: {np.mean([r['mean_rand_lift'] for r in results]):.3f}")
        print(f"   Média z‑score: {np.mean([r['z_score'] for r in results]):+.2f}")
        print(f"   % janelas com rank > 0.5: {np.mean([r['pct_rank'] > 0.5 for r in results]):.1%}")
        print(f"   Cobertura Pares: {np.mean([r['strat_pairs'] for r in results]):.0f} vs {np.mean([r['rand_pairs'] for r in results]):.0f} (aleatório)")
        print(f"   Cobertura Triplas: {np.mean([r['strat_triples'] for r in results]):.0f} vs {np.mean([r['rand_triples'] for r in results]):.0f} (aleatório)")
        print(f"   Cobertura Quadras: {np.mean([r['strat_quads'] for r in results]):.0f} vs {np.mean([r['rand_quads'] for r in results]):.0f} (aleatório)")
        print(f"   14pts total: Estratégia={sum(r['strat_14'] for r in results)}")
        diffs = [r['strat_lift'] - r['mean_rand_lift'] for r in results]
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p (lifts): {p:.4f}")
        except:
            pass
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v37")
    print("   TESTE DE CAPACIDADE PREDITIVA")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira com critérios (pares, moldura, primos)")
        print("2. Backtest de critérios (teto teórico)")
        print("3. Walk‑forward com critérios")
        print("4. Diagnóstico de distribuição (backtest + max_hit)")
        print("5. Análise de dependência temporal (MI + FDR)")
        print("6. Comparar real vs. sintético (RNG)")
        print("7. Autocorrelação, PACF e espectro")
        print("8. Contar universo combinatório dos critérios")
        print("9. Análise automática de critérios (últimos concursos)")
        print("10. Frequência histórica exata dos critérios")
        print("11. Teste de capacidade preditiva das features")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            print("\n📝 CRITÉRIOS PARA GERAÇÃO DA CARTEIRA")
            print("   Digite os valores permitidos (separados por espaço) ou ENTER para sem restrição")
            pares_str = input("   Pares (ex: 6 8 9): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10): ").strip()
            primos_str = input("   Primos (ex: 4 5 6): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio, score = opt.optimize(5, 10000)
            last = contests[-1]['dezenas']
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for d in g if d%2==0); pr = sum(1 for d in g if d in PRIMES)
                m = sum(1 for d in g if d in MOLDURA); rep = len(set(g) & set(last))
                cons = sum(1 for j in range(len(g)-1) if g[j+1]-g[j]==1)
                feat = opt.extractor.extract_features(g, last).reshape(1, -1)
                r_score, r_pct, r_mahal = opt.extractor.compute_rarity_scores_batch(feat)
                pen = opt.extractor.compute_structural_penalty(g)
                print(f" {i:2d}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep} Cons:{cons} Rarity:{r_score[0]:.2f} Pct:{r_pct[0]:.2f} Mahal:{r_mahal[0]:.1f} Pen:{pen:.1f}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (últimos 200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
                print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} 13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
        
        elif op == '2':
            print("\n📝 CRITÉRIOS PARA BACKTEST (TETO TEÓRICO)")
            pares_str = input("   Pares (ex: 6 8 9): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10): ").strip()
            primos_str = input("   Primos (ex: 4 5 6): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests)
            opt.backtest_criteria(allowed_pares, allowed_moldura, allowed_primos, contests[-200:])
        
        elif op == '3':
            print("\n📝 CRITÉRIOS PARA WALK‑FORWARD (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                                    n_random_benchmark=100, allowed_pares=allowed_pares,
                                    allowed_moldura=allowed_moldura, allowed_primos=allowed_primos)
        
        elif op == '4':
            opt = PortfolioOptimizer(contests)
            portfolio, _ = opt.optimize(5, 10000)
            print("Carteira gerada. Executando diagnóstico...")
            opt.diagnostic_distribution(portfolio, contests[-200:])
        
        elif op == '5':
            print("\n📊 INFORMAÇÃO MÚTUA AJUSTADA (lag=1) + CORREÇÃO FDR")
            series = compute_temporal_features(contests)
            pvals = []
            print(f"{'Feature':<15} {'MI_obs':<8} {'MI_adj':<8} {'p_raw':<8}")
            print("-" * 45)
            for name in TEMPORAL_FEATURES:
                mi_obs, mi_adj, p_val, _ = permutation_mi_test(series[name], lag=1)
                pvals.append(p_val)
                print(f"{name:<15} {mi_obs:<8.4f} {mi_adj:<8.4f} {p_val:<8.3f}")
            _, fdr_pvals, _, _ = multipletests(pvals, method='fdr_bh')
            print("\nApós FDR (Benjamini-Hochberg):")
            for i, name in enumerate(TEMPORAL_FEATURES):
                sig = "⚠️" if fdr_pvals[i] <= 0.05 else "não"
                print(f"   {name}: p_fdr={fdr_pvals[i]:.3f} -> {sig}")
        
        elif op == '6':
            compare_real_vs_synthetic(contests)
        
        elif op == '7':
            print("\n📊 AUTOCORRELAÇÃO, PACF E ESPECTRO POR FEATURE")
            series = compute_temporal_features(contests)
            for name in TEMPORAL_FEATURES:
                print(f"\n--- {name} ---")
                s = series[name]
                print("Lag  Corr    p_corr  MI_adj  p_mi")
                for lag in range(1, 6):
                    corr, p_corr = autocorrelation_test(s, lag)
                    _, mi_adj, p_mi, _ = permutation_mi_test(s, lag)
                    print(f"{lag:3d}  {corr:+.3f}  {p_corr:.3f}   {mi_adj:.4f}  {p_mi:.3f}")
                pacf_vals = partial_autocorrelation(s, nlags=5)
                print(f"PACF (lags 1-5): {np.array2string(pacf_vals[1:], precision=3, separator=', ')}")
                freq_dom, pow_dom, freqs, power = fft_analysis(s)
                print(f"Espectro: freq dominante = {freq_dom:.4f} (período ≈ {1/freq_dom if freq_dom>0 else np.inf:.1f}), potência = {pow_dom:.4f}")
        
        elif op == '8':
            print("\n📝 CRITÉRIOS PARA CONTAGEM DO UNIVERSO")
            pares_str = input("   Pares (ex: 6 8 9): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10): ").strip()
            primos_str = input("   Primos (ex: 4 5 6): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            count_universe(allowed_pares, allowed_moldura, allowed_primos)
        
        elif op == '9':
            n_recent = input("   Quantos concursos analisar? [500]: ").strip()
            n_recent = int(n_recent) if n_recent else 500
            coverage = input("   Cobertura desejada (ex: 0.80 para 80%) [0.80]: ").strip()
            coverage = float(coverage) if coverage else 0.80
            analyze_historical_criteria(contests, n_recent, coverage)
        
        elif op == '10':
            print("\n📝 CRITÉRIOS PARA FREQUÊNCIA HISTÓRICA EXATA")
            pares_str = input("   Pares (ex: 8): ").strip()
            moldura_str = input("   Moldura (ex: 11): ").strip()
            primos_str = input("   Primos (ex: 4): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            count, prop = historical_exact_frequency(contests, allowed_pares, allowed_moldura, allowed_primos)
            univ_est, univ_prop = count_universe(allowed_pares, allowed_moldura, allowed_primos)
            print(f"\n📊 FREQUÊNCIA NOS {len(contests)} CONCURSOS:")
            print(f"   Concursos que atendem exatamente: {count} ({prop*100:.2f}%)")
            print(f"   Proporção do universo: {univ_prop*100:.2f}%")
            if prop > univ_prop * 1.2:
                print("   🔍 A região parece super-representada nos concursos reais (>20% acima do esperado).")
            elif prop < univ_prop * 0.8:
                print("   🔍 A região parece sub-representada nos concursos reais.")
            else:
                print("   ✅ A frequência é compatível com o tamanho do universo.")
        
        elif op == '11':
            if not SKLEARN_AVAILABLE:
                print("❌ Scikit-learn é necessário para este teste. Instale com: pip install scikit-learn")
            else:
                predictive_test(contests)
        
        elif op == '0':
            break
        
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
