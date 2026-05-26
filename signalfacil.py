#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v32.3

Refinamentos estatísticos:
✅ Permutation test com permutações independentes (np.random.permutation)
✅ MI ajustada (mi_obs - média da distribuição nula)
✅ Autocorrelação linear de Pearson adicionada
✅ Suporte a múltiplos lags (1 a 5) para detectar dependência de ordem superior
✅ Opção "6" no menu para análise de autocorrelação e lags

Mantém todas as funcionalidades do v32.2:
- Carteira de cobertura otimizada
- Walk-forward honesto
- MI por features discretizadas + permutation test
- Comparação real vs. sintético (RNG)
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon, pearsonr
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
    print("⚠️ Scikit-learn não instalado. Algumas análises estarão indisponíveis.")

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.0

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

# Estruturais
MAX_CONSECUTIVOS_RUN = 5
MAX_CLUSTERIZACAO = 0.95
STRUCTURAL_TARGETS = {
    'pares': (7.5, 2.5, 1.0),
    'primos': (5.0, 2.5, 1.0),
    'moldura': (9.5, 2.5, 0.5),
    'repeticoes': (9.0, 3.0, 0.5),
    'soma': (195.0, 30.0, 0.3),
    'consecutivos': (5.5, 4.0, 0.2),
    'amplitude': (22.0, 5.0, 0.3),
}
STRUCTURAL_REJECT_THRESHOLD = 15

# Cobertura
MAX_PAIR_COVERAGE = 0.90
MAX_INTERSECTION = 7

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
# EXTRATOR DE FEATURES (mantido corrigido)
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
                if dev > tol: penalty += (dev - tol) * w
        max_run = run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run += 1; max_run = max(max_run, run)
            else: run = 1
        if max_run > MAX_CONSECUTIVOS_RUN: penalty += (max_run - MAX_CONSECUTIVOS_RUN) * 10.0
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        clusterizacao = sum(1 for g in gaps if g <= 2) / len(gaps)
        if clusterizacao > MAX_CLUSTERIZACAO: penalty += (clusterizacao - MAX_CLUSTERIZACAO) * 10.0
        return penalty

    def is_structurally_valid(self, game):
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

# ============================================================
# GERADOR ANTI-CLUSTER (v30.1)
# ============================================================
class LooseGenerator:
    def __init__(self, extractor=None):
        self.extractor = extractor
    def generate_one(self):
        for _ in range(50):
            game = self._generate_raw()
            if self.extractor is not None and self.extractor.is_structurally_valid(game):
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
# NOVAS FUNÇÕES DE DEPENDÊNCIA TEMPORAL (v32.3)
# ============================================================
def compute_temporal_features(contests):
    """Extrai features temporais para cada concurso (par chave:valor)."""
    series = {name: [] for name in TEMPORAL_FEATURES}
    for i, c in enumerate(contests):
        prev = set(contests[i-1]['dezenas']) if i > 0 else None
        for name, func in TEMPORAL_FEATURES.items():
            val = func(c['dezenas'], prev)
            # Discretização simples para features contínuas
            if name == 'gaps':
                val = round(val, 2)
            elif name == 'clusterizacao':
                val = round(val, 3)
            elif name == 'soma':
                val = val // 5 * 5  # agrupa somas em bins de 5
            series[name].append(val)
    return series

def mutual_information_feature(x, y):
    """MI discreta entre duas sequências (já discretizadas)."""
    joint = Counter(zip(x, y))
    total = sum(joint.values())
    mi = 0.0
    for (x_val, y_val), count in joint.items():
        p_xy = count / total
        p_x = x.count(x_val) / len(x)
        p_y = y.count(y_val) / len(y)
        if p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi

def permutation_mi_test(series_t, lag=1, n_perm=500):
    """
    Testa MI(series_t, series_{t-lag}) com permutações independentes.
    Retorna mi_obs, adjusted_mi, p_value, null_distribution.
    """
    x = series_t[lag:]          # X_t
    y = series_t[:-lag]         # X_{t-lag}
    mi_obs = mutual_information_feature(x, y)

    # Distribuição nula: permutações independentes de y
    mi_null = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = np.random.permutation(y)
        mi_null[i] = mutual_information_feature(x, y_perm)

    # MI ajustada (subtrai viés)
    mean_null = np.mean(mi_null)
    adjusted_mi = mi_obs - mean_null

    # p-valor bicaudal? Vamos usar unicaudal à direita: P(null >= obs)
    p_value = np.mean(mi_null >= mi_obs)
    return mi_obs, adjusted_mi, p_value, mi_null

def autocorrelation_test(series_t, lag=1):
    """Correlação de Pearson entre X_t e X_{t-lag}."""
    x = np.array(series_t[lag:], dtype=float)
    y = np.array(series_t[:-lag], dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    corr, p_val = pearsonr(x, y)
    return corr, p_val

# ============================================================
# COMPARAÇÃO REAL vs SINTÉTICO
# ============================================================
def generate_synthetic_contests(n):
    return [{'concurso': i, 'data': '', 'dezenas': sorted(np.random.choice(range(1,26), 15, replace=False))} for i in range(n)]

def compare_real_vs_synthetic(contests, n_synthetic=None):
    if n_synthetic is None:
        n_synthetic = len(contests)
    print(f"\n🔄 Gerando {n_synthetic} concursos sintéticos (i.i.d.)...")
    synthetic = generate_synthetic_contests(n_synthetic)

    # MI para cada feature (lag=1)
    print("\n📊 INFORMAÇÃO MÚTUA AJUSTADA (X_t; X_{{t-1}}) POR FEATURE:")
    print(f"{'Feature':<15} {'Real MI_adj':<12} {'Real p':<10} {'Sintético MI_adj':<15} {'Sintético p':<10}")
    print("-" * 65)

    real_series = compute_temporal_features(contests)
    synth_series = compute_temporal_features(synthetic)

    for name in TEMPORAL_FEATURES:
        _, mi_adj_real, p_real, _ = permutation_mi_test(real_series[name], lag=1)
        _, mi_adj_synth, p_synth, _ = permutation_mi_test(synth_series[name], lag=1)
        print(f"{name:<15} {mi_adj_real:<12.4f} {p_real:<10.3f} {mi_adj_synth:<15.4f} {p_synth:<10.3f}")

    print("\n🔍 Se MI_adj ≈ 0 e p > 0.05, não há dependência temporal significativa.")
    # Verificação geral
    all_insignificant = all(permutation_mi_test(real_series[name], lag=1)[2] > 0.05 for name in TEMPORAL_FEATURES)
    if all_insignificant:
        print("   ➡️ Nenhuma feature mostrou dependência temporal significativa nos dados reais.")

    # PCA
    if SKLEARN_AVAILABLE:
        ext_real = FeatureExtractor(contests)
        ext_synth = FeatureExtractor(synthetic)
        real_pca = pca_analysis(ext_real.standardized_features)
        synth_pca = pca_analysis(ext_synth.standardized_features)
        if real_pca is not None and synth_pca is not None:
            print("\n📊 PCA (variância explicada pelas 3 primeiras PCs):")
            print(f"   Real: PC1={real_pca[0]:.2%}, PC2={real_pca[1]:.2%}, PC3={real_pca[2]:.2%}, Soma={np.sum(real_pca):.2%}")
            print(f"   Sintético: PC1={synth_pca[0]:.2%}, PC2={synth_pca[1]:.2%}, PC3={synth_pca[2]:.2%}, Soma={np.sum(synth_pca):.2%}")
            if np.sum(real_pca) < 0.5:
                print("   ➡️ Variância explicada baixa: espaço aproximadamente isotrópico.")
    else:
        print("\n⚠️ PCA indisponível.")

    # Distribuição de gaps
    print("\n📊 DISTRIBUIÇÃO DE GAPS (média):")
    for label, data in [("Real", contests), ("Sintético", synthetic)]:
        gaps_all = []
        for c in data:
            d = sorted(c['dezenas'])
            gaps_all.extend([d[i+1]-d[i] for i in range(len(d)-1)])
        print(f"   {label}: média={np.mean(gaps_all):.2f}, desvio={np.std(gaps_all):.2f}")

    # Mahalanobis
    print("\n📊 DISTÂNCIA DE MAHALANOBIS MÉDIA:")
    try:
        ext = FeatureExtractor(contests)
        real_f = ext.standardized_features
        synth_f_raw = ext._build_raw_feature_matrix_from_contests(synthetic)
        if ext.scaler is not None:
            synth_f = ext.scaler.transform(synth_f_raw)
        else:
            synth_f = (synth_f_raw - ext.feature_means) / ext.feature_stds
        print(f"   Real: {np.mean(ext.mahalanobis_batch(real_f)):.2f}, Sintético: {np.mean(ext.mahalanobis_batch(synth_f)):.2f}")
    except Exception as e:
        print(f"   Não foi possível calcular: {e}")

    print("\n🔍 Se todos os valores forem muito próximos, reforça a hipótese de pseudoaleatoriedade efetiva.")

# Helper para FeatureExtractor
def _build_raw_feature_matrix_from_contests(self, contests_list):
    feats = []
    for i, c in enumerate(contests_list):
        last = set(contests_list[i-1]['dezenas']) if i > 0 else None
        feats.append(self._extract_raw(c['dezenas'], last))
    return np.array(feats, dtype=np.float64)

FeatureExtractor._build_raw_feature_matrix_from_contests = _build_raw_feature_matrix_from_contests

def pca_analysis(feature_matrix):
    if not SKLEARN_AVAILABLE:
        return None
    pca = PCA(n_components=min(5, feature_matrix.shape[1]))
    pca.fit(feature_matrix)
    return pca.explained_variance_ratio_[:3]

# ============================================================
# OTIMIZADOR DE CARTEIRA (mantido)
# ============================================================
class PortfolioOptimizerV32:
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
        return GameCandidate(game, mask, features, rarity_score, rarity_score, mahal_dist, percentile)

    def _unique_triples(self, portfolio):
        all_triples = set()
        for c in portfolio:
            all_triples.update(combinations(sorted(c.game), 3))
        return len(all_triples)

    def _pair_redundancy(self, portfolio):
        pair_counts = Counter()
        for c in portfolio:
            pair_counts.update(combinations(sorted(c.game), 2))
        redundant = sum(max(0, cnt-1) for cnt in pair_counts.values())
        max_possible = (len(portfolio)-1) * comb(15, 2)
        return redundant / max_possible if max_possible > 0 else 0

    def _portfolio_entropy(self, portfolio):
        freq = np.bincount([d for c in portfolio for d in c.game], minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        return entropy(probs) / np.log(25)

    def _monte_carlo_score(self, portfolio, n_sim=500):
        portfolio_masks = np.array([c.mask for c in portfolio], dtype=np.uint32)
        if len(self.historical_masks) > n_sim:
            idx = np.random.choice(len(self.historical_masks), n_sim, replace=False)
        else:
            idx = np.arange(len(self.historical_masks))
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
        if not hasattr(self, '_mc_bounds'):
            self._mc_bounds = self._compute_mc_bounds()
        p5, p95 = self._mc_bounds
        mc_norm = max(0.0, min(1.0, (raw_mc - p5) / (p95 - p5 + 1e-10)))
        return (triple_score * 0.4 + ent_score * 0.3 + avg_rarity * 0.2 + mc_norm * 0.1 - redundancy * 0.5)

    def _compute_mc_bounds(self):
        scores = [self._monte_carlo_score([self._create_candidate(self.generator.generate_pure_random()) for _ in range(5)], 300) for _ in range(30)]
        scores = np.array(scores)
        return np.percentile(scores, 5), np.percentile(scores, 95)

    def _select_diverse_portfolio(self, candidates, n_games):
        selected, masks = [], []
        for c in candidates:
            if len(selected) >= n_games: break
            if masks and any(mask_intersection(c.mask, m) > MAX_INTERSECTION for m in masks): continue
            selected.append(c); masks.append(c.mask)
        while len(selected) < n_games:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                    break
        return selected

    def optimize(self, n_games=5, n_candidates=10000):
        print(f"\n🧩 CARTEIRA DE COBERTURA: {n_games} jogos")
        t0 = time.time()
        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            g = self.generator.generate_one()
            key = tuple(g)
            if key not in seen and self.extractor.is_structurally_valid(g):
                seen.add(key); raw_pool.append(g)
        pool = raw_pool[:5000]
        fmat = self.extractor.extract_features_batch(pool, self.last)
        r_scores, pcts, m_dists = self.extractor.compute_rarity_scores_batch(fmat)
        candidates = [GameCandidate(pool[i], BITMASK_CACHE.get_mask(pool[i]), fmat[i], r_scores[i], r_scores[i], m_dists[i], pcts[i]) for i in range(len(pool))]
        candidates.sort(key=lambda c: c.central_score, reverse=True)
        portfolio = self._select_diverse_portfolio(candidates, n_games)
        best_score = self._portfolio_score(portfolio)
        top = candidates[:300]
        improved = True
        while improved:
            improved = False
            for i in range(len(portfolio)):
                for c in top:
                    if c in portfolio: continue
                    new_port = portfolio.copy(); new_port[i] = c
                    masks_new = [x.mask for x in new_port]
                    if any(mask_intersection(masks_new[a], masks_new[b]) > MAX_INTERSECTION for a in range(len(new_port)) for b in range(a+1, len(new_port))): continue
                    ns = self._portfolio_score(new_port)
                    if ns > best_score:
                        portfolio = new_port; best_score = ns
                        improved = True; break
                if improved: break
        print(f"✅ Otimizado em {time.time()-t0:.1f}s")
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
        return {'empirical': prob, 'theoretical': theo_prob, 'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success, 'total_premio': total_premio,
                'total_custo': total_custo, 'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts}

class GameCandidate:
    __slots__ = ('game','mask','features','rarity_score','central_score','mahalanobis_dist','rarity_percentile')
    def __init__(self, game, mask, features, rarity_score=0, central_score=0, mahalanobis_dist=0, rarity_percentile=0):
        self.game = game; self.mask = mask; self.features = features
        self.rarity_score = rarity_score; self.central_score = central_score
        self.mahalanobis_dist = mahalanobis_dist; self.rarity_percentile = rarity_percentile

# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)...")
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
        opt = PortfolioOptimizerV32(train_data)
        portfolio, _ = opt.optimize(n_games, n_candidates=8000)
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
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v32.3")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira de cobertura")
        print("2. Walk-forward (falsificação de hipóteses)")
        print("3. Análise de dependência temporal (MI + permutação, lag=1)")
        print("4. Comparar concursos reais vs. sintéticos (RNG)")
        print("5. Análise de autocorrelação e múltiplos lags (1 a 5)")
        print("6. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            opt = PortfolioOptimizerV32(contests)
            portfolio, score = opt.optimize(5, 10000)
            last = contests[-1]['dezenas']
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for d in g if d%2==0)
                pr = sum(1 for d in g if d in PRIMES)
                m = sum(1 for d in g if d in MOLDURA)
                rep = len(set(g) & set(last))
                cons = sum(1 for j in range(len(g)-1) if g[j+1]-g[j]==1)
                feat = opt.extractor.extract_features(g, last).reshape(1, -1)
                r_score, r_pct, r_mahal = opt.extractor.compute_rarity_scores_batch(feat)
                print(f" {i:2d}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep} Cons:{cons} Rarity:{r_score[0]:.2f} Pct:{r_pct[0]:.2f} Mahal:{r_mahal[0]:.1f}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (últimos 200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
                print(f"   Distribuição: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} 13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
        elif op == '2':
            walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5)
        elif op == '3':
            print("\n📊 INFORMAÇÃO MÚTUA AJUSTADA (lag=1) + TESTE DE PERMUTAÇÃO")
            series = compute_temporal_features(contests)
            print(f"{'Feature':<15} {'MI_obs':<8} {'MI_adj':<8} {'p_val':<8} {'Signif.':<10}")
            print("-" * 55)
            for name in TEMPORAL_FEATURES:
                mi_obs, mi_adj, p_val, _ = permutation_mi_test(series[name], lag=1)
                sig = "⚠️" if p_val <= 0.05 else "não"
                print(f"{name:<15} {mi_obs:<8.4f} {mi_adj:<8.4f} {p_val:<8.3f} {sig:<10}")
            print("\n🔍 Se MI_adj ≈ 0 e p > 0.05, não há dependência temporal significativa.")
        elif op == '4':
            compare_real_vs_synthetic(contests)
        elif op == '5':
            print("\n📊 AUTOCORRELAÇÃO E MI POR LAG (1 a 5)")
            series = compute_temporal_features(contests)
            for lag in range(1, 6):
                print(f"\n--- Lag {lag} ---")
                print(f"{'Feature':<15} {'Corr':<8} {'p_corr':<8} {'MI_adj':<8} {'p_mi':<8} {'Signif.':<10}")
                print("-" * 65)
                for name in TEMPORAL_FEATURES:
                    corr, p_corr = autocorrelation_test(series[name], lag)
                    _, mi_adj, p_mi, _ = permutation_mi_test(series[name], lag)
                    sig = "⚠️" if (p_corr <= 0.05 or p_mi <= 0.05) else "não"
                    print(f"{name:<15} {corr:<8.3f} {p_corr:<8.3f} {mi_adj:<8.4f} {p_mi:<8.3f} {sig:<10}")
            print("\n🔍 Correlação e MI ajustada próximas de zero + p > 0.05 = independência temporal.")
        elif op == '6':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
