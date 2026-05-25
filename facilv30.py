#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v27 (RARIDADE MULTIVARIADA)
==========================================================================
LIÇÕES DO v25/v26:
✅ Rarity score MULTIVARIADO (Mahalanobis) - não univariado
✅ Penalidade de colapso estrutural (consecutivos, clusterização)
✅ KL Divergence para sanity check (distribuição gerada vs histórica)
✅ PCA para embeddings temporais (detecção de estados raros)
✅ MC com pesos INVERTIDOS (favorece draws distantes)
✅ Gerador mais solto, mas COM limite de consecutivos (6)
✅ Carteira CONCENTRADA (5-6 jogos)
✅ Pesos ultra-agressivos: 13:0.2, 14:5000, 15:300000
✅ BitmaskCache + GameCandidate + vetorização mantidos
✅ Mistura de 3-4 concursos para sintéticos (50% MC)
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
from collections import Counter
from itertools import combinations
from datetime import datetime
import warnings
import os
from math import comb
from tqdm import tqdm
import random
import time

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Use: pip install scikit-learn")

# ============================================================
# CONJUNTOS E CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.0

FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
    "compressao",
]
N_FEATURES = len(FEATURE_NAMES)

# Constraints ESTRUTURAIS (com limites de colapso)
MAX_CONSECUTIVOS_RUN = 6  # Impede exploits de superclusterização
MAX_CLUSTERIZACAO = 0.95  # Limite superior para gaps pequenos

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

MAX_PAIR_COVERAGE = 0.75
MIN_GEO_DIVERSITY = 0.25
MAX_GEO_DIVERSITY = 0.85

# Pesos ULTRA-AGRESSIVOS
EXPONENTIAL_WEIGHTS = {
    11: 0.0,
    12: 0.0,
    13: 0.2,
    14: 5000.0,
    15: 300000.0,
}


# ============================================================
# UTILITÁRIOS BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}

    def get_mask(self, game):
        key = tuple(game) if isinstance(game, list) else game
        if key not in self._cache:
            mask = 0
            for d in key: mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]


BITMASK_CACHE = BitmaskCache()
def mask_intersection(m1, m2): return (m1 & m2).bit_count()
def draw_masks_to_array(draws): return np.array([BITMASK_CACHE.get_mask(d) for d in draws], dtype=np.uint32)


# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path): return None
    contests = []
    try:
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
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


# ============================================================
# EXTRATOR DE FEATURES + MODELO MULTIVARIADO
# ============================================================
class FeatureExtractor:
    def __init__(self, contests):
        self.contests = contests
        self._repeat_history = []
        for i, c in enumerate(contests):
            if i > 0: self._repeat_history.append(len(set(contests[i-1]['dezenas']) & set(c['dezenas'])))
            else: self._repeat_history.append(0)
        raw_features = self._build_raw_feature_matrix()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        if self.scaler is not None and len(raw_features) > 10: self.scaler.fit(raw_features)
        self.feature_means = np.mean(raw_features, axis=0)
        self.feature_stds = np.std(raw_features, axis=0) + 1e-10
        self._feature_cache = {}

        # MODELO MULTIVARIADO (Mahalanobis + PCA)
        self._build_multivariate_model(raw_features)

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
                freq = Counter(trans); probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs>0, probs, 1e-10)))
        amplitude = max(d)-min(d); std_pos = np.std(d) if len(d)>1 else 0.0
        compressao = std_pos/amplitude if amplitude>0 else 0.5
        return np.array([
            float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps)),
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))), ent_trans,
            float(len(set((x-1)//5 for x in d))),
            float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),
            float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),
            float(np.mean(d)-np.median(d)), float(sum(1 for g in gaps if g<=2)/len(gaps)),
            float(rep), float(sum(1 for x in d if x%2==0)), float(sum(1 for x in d if x in PRIMES)),
            float(sum(1 for x in d if x in MOLDURA)), float(sum(d)), float(max(d)-min(d)), compressao,
        ], dtype=np.float64)

    def _build_multivariate_model(self, raw_features):
        """Constrói modelo de Mahalanobis (LedoitWolf) e PCA."""
        # LedoitWolf para matriz de precisão robusta
        if SKLEARN_AVAILABLE and len(raw_features) > N_FEATURES:
            try:
                lw = LedoitWolf().fit(raw_features)
                self.precision_matrix = lw.precision_
                self.cov_matrix = lw.covariance_
                self._has_precision = True
            except:
                cov = np.cov(raw_features.T) + np.eye(N_FEATURES) * 1e-6
                self.precision_matrix = np.linalg.inv(cov)
                self.cov_matrix = cov
                self._has_precision = True
        else:
            cov = np.cov(raw_features.T) + np.eye(N_FEATURES) * 1e-6
            self.precision_matrix = np.linalg.inv(cov)
            self.cov_matrix = cov
            self._has_precision = True
        self._mean_vector = np.mean(raw_features, axis=0)

        # PCA para embeddings temporais
        if SKLEARN_AVAILABLE and len(raw_features) > 10:
            try:
                self.pca = PCA(n_components=min(5, N_FEATURES))
                self.pca.fit(raw_features)
                self._has_pca = True
            except:
                self._has_pca = False
        else:
            self._has_pca = False

        # Pré-computar Mahalanobis de todos os históricos
        self.historical_mahalanobis = np.array([
            self.mahalanobis_distance(f) for f in raw_features
        ])

    def mahalanobis_distance(self, features):
        """Distância de Mahalanobis multivariada (raridade REAL)."""
        diff = features - self._mean_vector
        try:
            dist_sq = np.dot(np.dot(diff.T, self.precision_matrix), diff)
            return float(np.sqrt(max(0, dist_sq)))
        except:
            return float(np.linalg.norm(diff / (np.std(self.historical_mahalanobis) + 1e-10)))

    def compute_rarity_score(self, game, all_features=None):
        """
        RARITY SCORE MULTIVARIADO: baseado na distância de Mahalanobis.
        Normalizado 0-1 usando percentis históricos.
        """
        features = self.extract_features(game, None)
        dist = self.mahalanobis_distance(features)
        # Percentil da distância em relação ao histórico
        percentile = np.mean(self.historical_mahalanobis <= dist)
        return float(percentile)

    def compute_pca_embedding(self, features):
        """Embedding PCA para detecção de estados raros."""
        if self._has_pca:
            return self.pca.transform(features.reshape(1, -1)).flatten()
        return features[:5] if len(features) >= 5 else features

    def extract_features(self, game, last_contest=None):
        key = (tuple(sorted(game)), tuple(last_contest) if last_contest else None)
        if key not in self._feature_cache:
            raw = self._extract_raw(game, last_contest)
            self._feature_cache[key] = (self.scaler.transform(raw.reshape(1, -1)).flatten()
                                        if self.scaler is not None
                                        else (raw - self.feature_means) / self.feature_stds)
        return self._feature_cache[key]

    def build_feature_matrix(self):
        raw = self._build_raw_feature_matrix()
        return self.scaler.transform(raw) if self.scaler is not None else (raw - self.feature_means) / self.feature_stds

    def compute_structural_penalty(self, game):
        d = sorted(game)
        penalty = 0.0
        actuals = {
            'pares': sum(1 for x in d if x%2==0), 'primos': sum(1 for x in d if x in PRIMES),
            'moldura': sum(1 for x in d if x in MOLDURA),
            'repeticoes': len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8,
            'soma': sum(d), 'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),
            'amplitude': max(d)-min(d),
        }
        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name]-target)
                if dev > tol: penalty += (dev - tol) * w
        # PENALIDADE DE COLAPSO ESTRUTURAL
        max_run = 1; run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1; max_run=max(max_run,run)
            else: run=1
        if max_run > MAX_CONSECUTIVOS_RUN:
            penalty += (max_run - MAX_CONSECUTIVOS_RUN) * 5.0
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        clusterizacao = sum(1 for g in gaps if g <= 2) / len(gaps)
        if clusterizacao > MAX_CLUSTERIZACAO:
            penalty += (clusterizacao - MAX_CLUSTERIZACAO) * 10.0
        return penalty

    def is_structurally_valid(self, game):
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

    def compute_kl_divergence(self, generated_features, n_bins=10):
        """
        KL Divergence entre distribuição gerada e histórica.
        Sanity check para detectar degeneração.
        """
        hist_features = self._build_raw_feature_matrix()
        kl_total = 0.0
        for i in range(min(N_FEATURES, generated_features.shape[1])):
            hist_vals = hist_features[:, i]
            gen_vals = generated_features[:, i]
            try:
                bins = np.linspace(min(hist_vals.min(), gen_vals.min()),
                                   max(hist_vals.max(), gen_vals.max()), n_bins+1)
                hist_hist, _ = np.histogram(hist_vals, bins=bins, density=True)
                gen_hist, _ = np.histogram(gen_vals, bins=bins, density=True)
                hist_hist = np.where(hist_hist>0, hist_hist, 1e-10)
                gen_hist = np.where(gen_hist>0, gen_hist, 1e-10)
                kl = np.sum(gen_hist * np.log(gen_hist / hist_hist))
                kl_total += kl
            except:
                pass
        return kl_total


# ============================================================
# GAMECANDIDATE
# ============================================================
class GameCandidate:
    __slots__ = ('game', 'mask', 'features', 'rarity_score', 'central_score', 'mahalanobis_dist')
    def __init__(self, game, mask, features, rarity_score=0, central_score=0, mahalanobis_dist=0):
        self.game = game; self.mask = mask; self.features = features
        self.rarity_score = rarity_score; self.central_score = central_score
        self.mahalanobis_dist = mahalanobis_dist


# ============================================================
# GERADOR (MAIS SOLTO, MAS COM LIMITE DE CONSECUTIVOS)
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
                test = game | {d}
                s = len(set((x-1)//5 for x in test)) * 2
                # Penalidade LEVE para runs excessivos
                st = sorted(test)
                run = 1; max_run = 1
                for i in range(len(st)-1):
                    if st[i+1]-st[i]==1: run+=1; max_run=max(max_run,run)
                    else: run=1
                if max_run > MAX_CONSECUTIVOS_RUN:
                    s -= (max_run - MAX_CONSECUTIVOS_RUN) * 3
                scores.append(s)
            if scores:
                scores = np.array(scores, dtype=np.float64); scores -= np.max(scores)
                probs = np.exp(scores / 2.0); probs /= probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else:
                chosen = random.choice(candidates)
            game.add(chosen); available.remove(chosen)
        return sorted(game)[:15]

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))


# ============================================================
# OTIMIZADOR v27 (RARIDADE MULTIVARIADA)
# ============================================================
class PortfolioOptimizerV27:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = LooseGenerator(self.extractor)
        self._mc_cache = {}
        self._mc_norm_params = None
        self.historical_draws = [c['dezenas'] for c in self.contests]
        self.historical_masks = draw_masks_to_array(self.historical_draws)
        if len(self.historical_draws) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draw_masks_to_array(self.historical_draws)
        self.historical_features = np.array([self.extractor.extract_features(list(d), None) for d in self.historical_draws])

    def _create_candidate(self, game):
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        rarity = self.extractor.compute_rarity_score(game)
        mahalanobis = self.extractor.mahalanobis_distance(features)
        return GameCandidate(game, mask, features, rarity, rarity, mahalanobis)

    def _pair_coverage(self, portfolio):
        covered = set()
        for c in portfolio:
            for pair in combinations(sorted(c.game), 2): covered.add(pair)
        return len(covered)/comb(25,2)

    def _portfolio_entropy(self, portfolio):
        freq = np.bincount([d for c in portfolio for d in c.game], minlength=26)[1:]
        probs = freq/np.sum(freq); probs = np.where(probs>0, probs, 1e-10)
        return float(entropy(probs)/np.log(25))

    def _portfolio_diversity(self, portfolio):
        if len(portfolio) < 2: return 1.0
        masks = [c.mask for c in portfolio]
        sims = [mask_intersection(masks[i], masks[j]) for i in range(len(masks)) for j in range(i+1, len(masks))]
        return 1.0 - np.mean(sims)/15.0 if sims else 1.0

    def _geometric_diversity(self, portfolio):
        if len(portfolio) < 2: return 0.5
        fvs = np.array([c.features for c in portfolio])
        dists = [np.linalg.norm(fvs[i]-fvs[j]) for i in range(len(fvs)) for j in range(i+1, len(fvs))]
        return np.mean(dists)/(2*np.sqrt(N_FEATURES)) if dists else 0

    def _generate_synthetic_draws(self, n_synthetic):
        synthetic = []
        for _ in range(n_synthetic):
            sources = random.sample(self.historical_draws, random.randint(3, 4))
            mix = set()
            for src in sources:
                mix.update(random.sample(src, random.randint(4, 5)))
            available = set(range(1,26)) - mix
            while len(mix) < 15 and available:
                mix.add(random.choice(list(available)))
                available = set(range(1,26)) - mix
            synthetic.append(sorted(mix)[:15])
        return synthetic

    def _monte_carlo_hybrid(self, portfolio_candidates, n_simulations=500):
        cache_key = tuple(tuple(sorted(c.game)) for c in portfolio_candidates)
        if cache_key in self._mc_cache: return self._mc_cache[cache_key]

        recent_f = self.feature_matrix[-20:] if len(self.feature_matrix)>=20 else self.feature_matrix
        dists = np.linalg.norm(self.historical_features[:, None, :] - recent_f[None, :, :], axis=2)
        avg_dists = np.mean(dists, axis=1)
        weights = avg_dists**2 + 1e-6; weights /= weights.sum()

        n_hist = int(n_simulations*0.5); n_synth = n_simulations - n_hist
        hist_indices = np.random.choice(len(self.historical_masks), size=n_hist, p=weights)
        hist_masks = self.historical_masks[hist_indices]
        synthetic_draws = self._generate_synthetic_draws(n_synth)
        synth_masks = draw_masks_to_array(synthetic_draws)
        all_masks = np.concatenate([hist_masks, synth_masks])
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)

        total_weighted_score = 0.0
        for j in range(len(all_masks)):
            dm = all_masks[j]
            for i, pm in enumerate(portfolio_masks):
                hits = mask_intersection(pm, dm)
                if hits >= 13:
                    total_weighted_score += EXPONENTIAL_WEIGHTS.get(hits, 0)

        avg_score = total_weighted_score / len(all_masks)
        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization(portfolio_size=len(portfolio_candidates))
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        normalized = max(0.0, min(1.0, (avg_score - p5) / (p95 - p5 + 1e-10)))
        self._mc_cache[cache_key] = normalized
        return normalized

    def _compute_mc_normalization(self, portfolio_size=5, n_samples=200):
        raw_scores = []
        for _ in range(n_samples):
            rand_port = [self._create_candidate(self.generator.generate_pure_random()) for _ in range(portfolio_size)]
            raw = self._monte_carlo_hybrid_raw(rand_port, 300)
            raw_scores.append(raw)
        raw_scores = np.array(raw_scores)
        return {'p5': float(np.percentile(raw_scores,5)), 'p95': float(np.percentile(raw_scores,95))}

    def _monte_carlo_hybrid_raw(self, portfolio_candidates, n_simulations=300):
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)
        indices = np.random.choice(len(self.historical_masks), size=n_simulations)
        total_score = 0.0
        for idx in indices:
            drawn_mask = self.historical_masks[idx]
            for pm in portfolio_masks:
                hits = mask_intersection(pm, drawn_mask)
                if hits >= 13: total_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
        return total_score/len(indices)

    def _portfolio_score(self, portfolio):
        if self._pair_coverage(portfolio) > MAX_PAIR_COVERAGE: return -1000.0
        if not (MIN_GEO_DIVERSITY <= self._geometric_diversity(portfolio) <= MAX_GEO_DIVERSITY): return -1000.0
        mc_score = self._monte_carlo_hybrid(portfolio)
        avg_rarity = np.mean([c.rarity_score for c in portfolio])
        return mc_score*0.55 + avg_rarity*0.25 + self._portfolio_diversity(portfolio)*0.10 + self._geometric_diversity(portfolio)*0.10

    def _mutate_candidate(self, candidate):
        for _ in range(20):
            mutated = list(candidate.game)
            for _ in range(random.randint(1, 4)):
                pos = random.randint(0,14); avail = [d for d in range(1,26) if d not in mutated]
                if avail: mutated[pos] = random.choice(avail)
            mutated = sorted(mutated)[:15]
            if self.extractor.is_structurally_valid(mutated): return self._create_candidate(mutated)
        return candidate

    def optimize(self, n_games=5, n_candidates=50000, iterations=100):
        print(f"   Carteira CONCENTRADA: {n_games} jogos")
        print(f"   Rarity MULTIVARIADO (Mahalanobis)")
        print(f"   MC com pesos INVERTIDOS | Limites anti-colapso")
        print(f"   Pesos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")

        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Fase 1"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen and self.extractor.is_structurally_valid(game):
                seen.add(key); raw_pool.append(game)

        top_pool = random.sample(raw_pool, min(5000, len(raw_pool)))
        candidates = [self._create_candidate(g) for g in tqdm(top_pool, desc="Fase 2")]
        candidates.sort(key=lambda c: c.central_score, reverse=True)

        portfolio, portfolio_masks = [], []
        for c in candidates:
            if len(portfolio) >= n_games: break
            if portfolio_masks and max(mask_intersection(c.mask, pm) for pm in portfolio_masks) > 10: continue
            portfolio.append(c); portfolio_masks.append(c.mask)

        best_portfolio, best_score = list(portfolio), self._portfolio_score(portfolio)

        elite_pool = candidates[:len(candidates)//4]
        for it in tqdm(range(iterations), desc="Annealing"):
            temp = 1.0 * (0.95**it)
            new_portfolio = list(portfolio)
            idx = random.randint(0, len(new_portfolio)-1)
            if random.random() < 0.4 and elite_pool:
                new_candidate = random.choice(elite_pool)
            elif random.random() < 0.7:
                new_candidate = self._mutate_candidate(new_portfolio[idx])
            else:
                new_candidate = self._create_candidate(self.generator.generate_one())
            if any(j != idx and mask_intersection(new_candidate.mask, c.mask) > 10 for j, c in enumerate(new_portfolio)):
                continue
            new_portfolio[idx] = new_candidate
            new_score = self._portfolio_score(new_portfolio)
            if new_score > best_score: best_portfolio, best_score = list(new_portfolio), new_score
            elif random.random() < np.exp((new_score - self._portfolio_score(portfolio))/max(0.01, temp)):
                portfolio = new_portfolio
        return [c.game for c in best_portfolio], best_score

    def backtest(self, portfolio, test_draws):
        n_success, total_premio = 0, 0.0
        total_custo = len(portfolio)*len(test_draws)*CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k: 0 for k in range(11, 16)}
        for draw in test_draws:
            draw_mask = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, draw_mask)
                if hits >= 11: n_success += 1; total_premio += PREMIO_VALORES.get(hits, 0); hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if len(test_draws)>0 else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {'empirical': prob, 'theoretical': theo_prob, 'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=5):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)...")
    results = []
    for w in range(n_windows):
        test_end = len(contests)-w*test_size; test_start = test_end-test_size
        train_end = test_start; train_start = max(0, train_end-train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data, test_data = contests[train_start:train_end], contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue
        opt = PortfolioOptimizerV27(train_data)
        portfolio, _ = opt.optimize(n_games, n_candidates=50000, iterations=50)
        bt = opt.backtest(portfolio, test_data)
        bt_rand = opt.backtest([opt.generator.generate_pure_random() for _ in range(n_games)], test_data)
        results.append({
            'window': w, 'diff_lift': bt['lift']-bt_rand['lift'], 'diff_roi': bt['roi']-bt_rand['roi'],
            'strat_14': bt['hit_distribution'].get(14,0), 'rand_14': bt_rand['hit_distribution'].get(14,0),
        })
        print(f" Janela {w}: diff_lift={bt['lift']-bt_rand['lift']:+.3f} "
              f"14pts: {bt['hit_distribution'].get(14,0)} vs {bt_rand['hit_distribution'].get(14,0)}")
    if results:
        diffs = [r['diff_lift'] for r in results]
        strat_14_total = sum(r['strat_14'] for r in results)
        rand_14_total = sum(r['rand_14'] for r in results)
        print(f"\n📊 RESUMO:")
        print(f"   Média diff lift: {np.mean(diffs):+.3f} | Janelas +: {sum(1 for d in diffs if d>0)}/{len(results)}")
        print(f"   14pts total: Estratégia={strat_14_total} vs Aleatório={rand_14_total}")
        try: _, p = wilcoxon(diffs); print(f"   Wilcoxon p: {p:.4f}")
        except: pass
    return results


# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR DE CARTEIRA v27 - RARIDADE MULTIVARIADA (MAHALANOBIS)")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado."); return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print(f"\n📊 Mahalanobis + LedoitWolf | MC INVERTIDO | Anti-colapso")
    print(f"   Pesos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")
    print("Opções: 1. Gerar carteira | 2. Walk-forward | 3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"
    if op in ("1", "3"):
        t0 = time.time(); opt = PortfolioOptimizerV27(contests)
        print(f"   ✅ Init {time.time()-t0:.1f}s")
        portfolio, _ = opt.optimize(5, 50000, 100)
        last = contests[-1]['dezenas']
        # Calcular KL Divergence
        gen_features = np.array([opt.extractor.extract_features(g, last) for g in portfolio])
        kl = opt.extractor.compute_kl_divergence(gen_features)
        print(f"\n📊 KL Divergence (gerado vs histórico): {kl:.3f} (0=idêntico)")
        for i, g in enumerate(portfolio, 1):
            p = sum(1 for d in g if d%2==0); pr = sum(1 for d in g if d in PRIMES)
            m = sum(1 for d in g if d in MOLDURA); rep = len(set(g)&set(last))
            cons = sum(1 for j in range(len(g)-1) if g[j+1]-g[j]==1)
            rarity = opt.extractor.compute_rarity_score(g)
            mahal = opt.extractor.mahalanobis_distance(opt.extractor.extract_features(g, last))
            print(f"   {i:2d}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep} Cons:{cons} Rarity:{rarity:.2f} Mahal:{mahal:.1f}")
        if len(contests) > 200:
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Distribuição: 11pts:{bt['hit_distribution'].get(11,0)} "
                  f"12pts:{bt['hit_distribution'].get(12,0)} 13pts:{bt['hit_distribution'].get(13,0)} "
                  f"14pts:{bt['hit_distribution'].get(14,0)} 15pts:{bt['hit_distribution'].get(15,0)}")
    if op in ("2", "3"): walk_forward_validation(contests, 10, 500, 50, 5)
    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
