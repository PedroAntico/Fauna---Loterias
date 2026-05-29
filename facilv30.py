#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v33

CONSOLIDADO FINAL COM TODAS AS MELHORIAS:

✅ Penalidade estrutural suave (soft penalty)
✅ Farthest‑point sampling para diversidade geométrica real
✅ Carteira híbrida (centrais + extremos + balanceados)
✅ Ensemble de cobertura (Hamming, pares, entropia)
✅ Diagnóstico de distribuição hipergeométrica (observado vs esperado)
✅ Walk‑forward honesto com Wilcoxon
✅ Análise de dependência temporal completa:
   - MI ajustada com permutation test
   - Autocorrelação linear (Pearson)
   - Autocorrelação parcial (PACF)
   - Correção FDR para múltiplos testes
   - Espectro de potência (FFT)
✅ Comparação real vs sintético (RNG)
✅ Análise de cobertura de triplas e pares
✅ Backtest com distribuição de acertos e ROI
✅ Validação estatística rigorosa em todas as etapas
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. PCA indisponível.")

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

# Features para análise de dependência temporal (espaço reduzido)
TEMPORAL_FEATURES = {
    'repeticoes': lambda d, prev: len(set(d) & prev) if prev else 8,
    'soma': lambda d, prev: sum(d),
    'pares': lambda d, prev: sum(1 for x in d if x%2==0),
    'primos': lambda d, prev: sum(1 for x in d if x in PRIMES),
    'clusterizacao': lambda d, prev: sum(1 for i in range(len(d)-1) if d[i+1]-d[i]<=2)/14,
    'gaps': lambda d, prev: np.mean([d[i+1]-d[i] for i in range(len(d)-1)]),
}

# Estruturais – relaxadas com penalidade suave
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
SOFT_PENALTY_WEIGHT = 0.05

# Cobertura
MAX_PAIR_COVERAGE = 0.95
MAX_INTERSECTION = 7
HAMMING_MIN_DIST = 4

# Pesos MC (atenuados)
EXPONENTIAL_WEIGHTS = {
    11: 1.0,
    12: 4.0,
    13: 25.0,
    14: 400.0,
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
# EXTRATOR DE FEATURES
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
# GERADOR (COM PENALIDADE SUAVE)
# ============================================================
class LooseGenerator:
    def __init__(self, extractor=None):
        self.extractor = extractor
    def generate_one(self, max_penalty=30):
        for _ in range(50):
            game = self._generate_raw()
            if self.extractor is not None:
                pen = self.extractor.compute_structural_penalty(game)
                if pen <= max_penalty:
                    return game
            else:
                return game
        return self._generate_raw()
    def _generate_raw(self):
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
# FUNÇÕES DE DEPENDÊNCIA TEMPORAL
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
# OTIMIZADOR DE CARTEIRA (PENALIDADE SUAVE + FARTHEST-POINT)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = LooseGenerator(self.extractor)
        self.historical_masks = draw_masks_to_array([c['dezenas'] for c in self.contests])
        if len(self.historical_masks) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_masks))]
            self.historical_masks = np.concatenate([self.historical_masks, draw_masks_to_array(extra)])

    def _create_candidate(self, game, rarity_score=None, percentile=None, mahal_dist=None):
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        if rarity_score is None:
            rarity_score, percentile, mahal_dist = self.extractor.compute_rarity_scores_batch(features.reshape(1, -1))
            rarity_score = rarity_score[0]; percentile = percentile[0]; mahal_dist = mahal_dist[0]
        pen = self.extractor.compute_structural_penalty(game)
        return GameCandidate(game, mask, features, rarity_score, rarity_score, mahal_dist, percentile, pen)

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

    def _portfolio_score(self, portfolio):
        pair_cov = len(set(p for c in portfolio for p in combinations(sorted(c.game), 2))) / comb(25,2)
        if pair_cov > MAX_PAIR_COVERAGE: return -1000.0
        triples = self._unique_triples(portfolio)
        triple_score = triples / (len(portfolio) * comb(15, 3))
        ent_score = self._portfolio_entropy(portfolio)
        redundancy = self._pair_redundancy(portfolio)
        avg_rarity = np.mean([c.rarity_score for c in portfolio])
        raw_mc = self._monte_carlo_score(portfolio)
        if not hasattr(self, '_mc_bounds'): self._mc_bounds = self._compute_mc_bounds()
        p5, p95 = self._mc_bounds
        mc_norm = max(0.0, min(1.0, (raw_mc - p5) / (p95 - p5 + 1e-10)))
        avg_penalty = np.mean([c.penalty for c in portfolio])
        return (triple_score * 0.35 + ent_score * 0.25 + avg_rarity * 0.2 +
                mc_norm * 0.1 - redundancy * 0.5 - avg_penalty * SOFT_PENALTY_WEIGHT)

    def _compute_mc_bounds(self):
        scores = [self._monte_carlo_score([self._create_candidate(self.generator.generate_pure_random()) for _ in range(5)], 300) for _ in range(30)]
        scores = np.array(scores)
        return np.percentile(scores, 5), np.percentile(scores, 95)

    def _farthest_point_sampling(self, candidates, n_select):
        features = np.array([c.features for c in candidates])
        selected_idx = [0]
        for _ in range(n_select - 1):
            dists = cdist(features, features[selected_idx], metric='euclidean')
            min_dists = np.min(dists, axis=1)
            min_dists[selected_idx] = -1
            next_idx = np.argmax(min_dists)
            selected_idx.append(next_idx)
        return [candidates[i] for i in selected_idx]

    def _select_diverse_portfolio(self, candidates, n_games, use_hamming=True, use_farthest=False):
        if use_farthest and len(candidates) >= n_games:
            return self._farthest_point_sampling(candidates, n_games)
        selected, masks, games = [], [], []
        for c in candidates:
            if len(selected) >= n_games: break
            if masks and any(mask_intersection(c.mask, m) > MAX_INTERSECTION for m in masks): continue
            if use_hamming and games:
                if any(hamming_distance(c.game, g) < HAMMING_MIN_DIST for g in games): continue
            selected.append(c); masks.append(c.mask); games.append(c.game)
        while len(selected) < n_games:
            for c in candidates:
                if c not in selected:
                    selected.append(c); break
        return selected

    def optimize(self, n_games=5, n_candidates=30000, use_hamming=True, use_farthest=False):
        method = "farthest-point" if use_farthest else ("Hamming + interseção" if use_hamming else "greedy básico")
        print(f"\n🧩 CARTEIRA DE COBERTURA: {n_games} jogos | método: {method}")
        t0 = time.time()
        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            g = self.generator.generate_one()
            key = tuple(g)
            if key not in seen:
                seen.add(key); raw_pool.append(g)
        pool = raw_pool[:5000]
        fmat = self.extractor.extract_features_batch(pool, self.last)
        r_scores, pcts, m_dists = self.extractor.compute_rarity_scores_batch(fmat)
        candidates = []
        for i, game in enumerate(pool):
            mask = BITMASK_CACHE.get_mask(game)
            pen = self.extractor.compute_structural_penalty(game)
            candidates.append(GameCandidate(game, mask, fmat[i], r_scores[i], r_scores[i], m_dists[i], pcts[i], pen))
        candidates.sort(key=lambda c: c.central_score - c.penalty * 0.1, reverse=True)
        portfolio = self._select_diverse_portfolio(candidates, n_games, use_hamming, use_farthest)
        best_score = self._portfolio_score(portfolio)
        top = candidates[:500]
        improved = True
        while improved:
            improved = False
            for i in range(len(portfolio)):
                for c in top:
                    if c in portfolio: continue
                    new_port = portfolio.copy(); new_port[i] = c
                    masks_new = [x.mask for x in new_port]
                    if any(mask_intersection(masks_new[a], masks_new[b]) > MAX_INTERSECTION for a in range(len(new_port)) for b in range(a+1, len(new_port))): continue
                    if use_hamming and any(hamming_distance(c.game, x.game) < HAMMING_MIN_DIST for x in new_port if x != c): continue
                    ns = self._portfolio_score(new_port)
                    if ns > best_score:
                        portfolio = new_port; best_score = ns
                        improved = True; break
                if improved: break
        print(f"✅ Otimizado em {time.time()-t0:.1f}s")
        return [c.game for c in portfolio], best_score

    def hybrid_portfolio(self, n_central=2, n_extreme=2, n_balanced=1, n_candidates=30000):
        print(f"\n🎯 CARTEIRA HÍBRIDA: {n_central} centrais + {n_extreme} extremos + {n_balanced} balanceado")
        t0 = time.time()
        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            g = self.generator.generate_one(max_penalty=40)
            key = tuple(g)
            if key not in seen:
                seen.add(key); raw_pool.append(g)
        pool = raw_pool[:5000]
        fmat = self.extractor.extract_features_batch(pool, self.last)
        r_scores, pcts, m_dists = self.extractor.compute_rarity_scores_batch(fmat)
        candidates = []
        for i, game in enumerate(pool):
            mask = BITMASK_CACHE.get_mask(game)
            pen = self.extractor.compute_structural_penalty(game)
            candidates.append(GameCandidate(game, mask, fmat[i], r_scores[i], r_scores[i], m_dists[i], pcts[i], pen))
        candidates.sort(key=lambda c: c.central_score, reverse=True)
        centrais = candidates[:n_central*100]
        candidates_by_mahal = sorted(candidates, key=lambda c: c.mahalanobis_dist, reverse=True)
        extremos = candidates_by_mahal[:n_extreme*100]
        selected = []
        sel_cent = self._select_diverse_portfolio(centrais, n_central, use_hamming=True)
        selected.extend(sel_cent)
        masks_sel = [c.mask for c in selected]
        sel_ext = []
        for c in extremos:
            if len(sel_ext) >= n_extreme: break
            if any(mask_intersection(c.mask, m) > MAX_INTERSECTION for m in masks_sel): continue
            if sel_ext and any(hamming_distance(c.game, x.game) < HAMMING_MIN_DIST for x in sel_ext): continue
            sel_ext.append(c); masks_sel.append(c.mask)
        selected.extend(sel_ext)
        mid_idx = len(candidates) // 2
        balanceados = candidates[mid_idx:mid_idx+200]
        sel_bal = self._select_diverse_portfolio(balanceados, n_balanced, use_hamming=True)
        for c in sel_bal:
            if len(selected) >= n_central + n_extreme + n_balanced: break
            if any(mask_intersection(c.mask, m) > MAX_INTERSECTION for m in masks_sel): continue
            selected.append(c)
        print(f"✅ Carteira híbrida gerada em {time.time()-t0:.1f}s")
        return [c.game for c in selected]

    def diagnostic_distribution(self, portfolio, test_draws):
        n_jogos = len(portfolio)
        n_test = len(test_draws)
        total_sim = n_jogos * n_test
        hit_counts = {k:0 for k in range(11,16)}
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11: hit_counts[hits] += 1
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
        ratio_13 = hit_counts.get(13,0)/expected[13] if expected[13] > 0 else 0
        if ratio_13 < 0.5:
            print("\n🔍 ALERTA: 13 pontos muito abaixo do esperado.")
            print("   Possível compressão de variância – o sistema está eliminando jogos extremos.")
            print("   Recomendação: relaxar filtros estruturais ou aumentar proporção de jogos extremos.")
        return hit_counts, expected

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

    def ensemble_optimize(self, n_carteiras=3, n_games_por_carteira=5, n_candidates=30000):
        print(f"\n🎯 GERANDO ENSEMBLE DE {n_carteiras} CARTEIRAS")
        ensembles = []
        print("   Carteira 1: cobertura padrão (farthest-point)")
        port1, _ = self.optimize(n_games_por_carteira, n_candidates, use_hamming=True, use_farthest=True)
        ensembles.append(port1)
        print("   Carteira 2: foco em pares")
        old_score = self._portfolio_score
        self._portfolio_score = lambda p: -self._pair_redundancy(p)
        port2, _ = self.optimize(n_games_por_carteira, n_candidates, use_hamming=False)
        self._portfolio_score = old_score
        ensembles.append(port2)
        print("   Carteira 3: máxima entropia")
        self._portfolio_score = lambda p: self._portfolio_entropy(p)
        port3, _ = self.optimize(n_games_por_carteira, n_candidates, use_hamming=False)
        self._portfolio_score = old_score
        ensembles.append(port3)
        print("✅ Ensemble gerado.")
        return ensembles

    def backtest_ensemble(self, ensemble, test_draws):
        total_success = 0; total_premio = 0.0
        total_custo = sum(len(port) for port in ensemble) * len(test_draws) * CUSTO_APOSTA
        hit_counts_total = {k:0 for k in range(11,16)}
        for portfolio in ensemble:
            bt = self.backtest(portfolio, test_draws)
            total_success += bt['n_success']; total_premio += bt['total_premio']
            for k in range(11,16): hit_counts_total[k] += bt['hit_distribution'][k]
        prob = total_success / (sum(len(p) for p in ensemble) * len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        total_jogos = sum(len(p) for p in ensemble)
        theo_prob = 1 - (1-p_single)**total_jogos
        return {'empirical': prob, 'theoretical': theo_prob,
                'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': total_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts_total}

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
# WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                            use_ensemble=False, use_hybrid=False):
    label = "ENSEMBLE" if use_ensemble else ("HÍBRIDO" if use_hybrid else "padrão")
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas) {label}")
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
        opt = PortfolioOptimizer(train_data)
        if use_ensemble:
            portfolio = opt.ensemble_optimize(n_carteiras=3, n_games_por_carteira=5, n_candidates=10000)
            bt = opt.backtest_ensemble(portfolio, test_data)
            total_random_games = sum(len(p) for p in portfolio)
            rand_port = [opt.generator.generate_pure_random() for _ in range(total_random_games)]
            bt_rand = opt.backtest(rand_port, test_data)
        elif use_hybrid:
            portfolio = opt.hybrid_portfolio(n_central=1, n_extreme=0, n_balanced=4, n_candidates=10000)
            bt = opt.backtest(portfolio, test_data)
            bt_rand = opt.backtest([opt.generator.generate_pure_random() for _ in range(n_games)], test_data)
        else:
            portfolio, _ = opt.optimize(n_games, n_candidates=10000)
            bt = opt.backtest(portfolio, test_data)
            bt_rand = opt.backtest([opt.generator.generate_pure_random() for _ in range(n_games)], test_data)
        results.append({
            'window': w, 'diff_lift': bt['lift']-bt_rand['lift'],
            'diff_roi': bt['roi']-bt_rand['roi'],
            'strat_14': bt['hit_distribution'].get(14,0),
            'rand_14': bt_rand['hit_distribution'].get(14,0),
        })
        print(f"   Janela {w}: diff_lift={bt['lift']-bt_rand['lift']:+.3f} 14pts: {bt['hit_distribution'].get(14,0)} vs {bt_rand['hit_distribution'].get(14,0)}")
    if results:
        diffs = [r['diff_lift'] for r in results]
        print(f"\n📊 RESUMO: Média diff lift: {np.mean(diffs):+.3f} | Janelas +: {sum(1 for d in diffs if d>0)}/{len(results)}")
        print(f"   14pts total: Estratégia={sum(r['strat_14'] for r in results)} vs Aleatório={sum(r['rand_14'] for r in results)}")
        try: _, p = wilcoxon(diffs); print(f"   Wilcoxon p: {p:.4f}")
        except: pass
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v33")
    print("   CONSOLIDADO FINAL")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira padrão (penalidade suave + farthest-point)")
        print("2. Gerar carteira híbrida (central + extrema)")
        print("3. Walk-forward padrão")
        print("4. Walk-forward híbrido")
        print("5. Walk-forward ensemble")
        print("6. Diagnóstico de distribuição (backtest + O/E)")
        print("7. Análise de dependência temporal (MI + FDR)")
        print("8. Comparar real vs. sintético (RNG)")
        print("9. Autocorrelação, PACF e espectro")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            opt = PortfolioOptimizer(contests)
            use_farthest = input("Usar farthest-point sampling? (s/n) [s]: ").strip().lower() != 'n'
            portfolio, score = opt.optimize(5, 10000, use_hamming=True, use_farthest=use_farthest)
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
            opt = PortfolioOptimizer(contests)
            portfolio = opt.hybrid_portfolio(n_central=1, n_extreme=0, n_balanced=4, n_candidates=30000)
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
                opt.diagnostic_distribution(portfolio, contests[-200:])
                print(f"\n   Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        elif op == '3':
            walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5)
        elif op == '4':
            walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5, use_hybrid=True)
        elif op == '5':
            walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5, use_ensemble=True)
        elif op == '6':
            opt = PortfolioOptimizer(contests)
            portfolio, _ = opt.optimize(5, 10000)
            print("Carteira gerada. Executando diagnóstico...")
            opt.diagnostic_distribution(portfolio, contests[-200:])
        elif op == '7':
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
        elif op == '8':
            compare_real_vs_synthetic(contests)
        elif op == '9':
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
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
