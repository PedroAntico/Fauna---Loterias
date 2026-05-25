#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v31.1 (AJUSTES FINOS)

Ajustes em relação ao v31:
✅ KDE dos percentis: bandwidth reduzido para 0.03 (maior sensibilidade)
✅ Monte Carlo softmax: temperatura aumentada para 5.0 (distribuição mais plana)
✅ Penalidade de correlação intra-portfólio: threshold reduzido para 0.2
✅ Cache de características raw e escaladas otimizado
✅ Log detalhado de convergência (KL, correlação, dispersão)
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
from sklearn.neighbors import KernelDensity
from collections import Counter
from itertools import combinations
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Use: pip install scikit-learn")

# ============================================================
# CONJUNTOS E CONSTANTES
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
    "pares", "primos", "moldura", "soma", "amplitude",
    "compressao",
]
N_FEATURES = len(FEATURE_NAMES)

# Constraints estruturais (rigorosas)
MAX_CONSECUTIVOS_RUN = 5
MAX_TOTAL_CONSECUTIVOS = 7
MAX_CLUSTERIZACAO = 0.85

STRUCTURAL_TARGETS = {
    'pares': (7.5, 2.5, 1.0),
    'primos': (5.0, 2.5, 1.0),
    'moldura': (9.5, 2.5, 0.5),
    'repeticoes': (9.0, 3.0, 0.5),
    'soma': (195.0, 30.0, 0.3),
    'consecutivos': (5.5, 4.0, 0.2),
    'amplitude': (22.0, 5.0, 0.3),
}
STRUCTURAL_REJECT_THRESHOLD = 12

MAX_PAIR_COVERAGE = 0.75
MIN_GEO_DIVERSITY = 0.25
MAX_GEO_DIVERSITY = 0.85

EXPONENTIAL_WEIGHTS = {11: 0.0, 12: 0.0, 13: 0.2, 14: 5000.0, 15: 300000.0}

# Ajustes finos
KDE_BANDWIDTH = 0.03            # menor = mais sensível à forma real
SOFTMAX_TEMPERATURE = 5.0       # maior = menos pico, mais exploração
CORRELATION_THRESHOLD = 0.2     # menor = pune mais cedo

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
            for d in key:
                mask |= (1 << d)
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
    if not os.path.exists(csv_path):
        return None
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
# EXTRATOR DE FEATURES (COM KDE DOS PERCENTIS HISTÓRICOS)
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
        self.feature_means = np.mean(raw_features, axis=0)
        self.feature_stds = np.std(raw_features, axis=0) + 1e-10
        self._raw_cache = {}
        self._feature_cache = {}

        # Modelo multivariado Mahalanobis
        self._build_mahalanobis_model(raw_features)

        # Distribuição empírica dos percentis de Mahalanobis históricos
        self._build_percentile_kde()

    def _build_raw_feature_matrix(self):
        features_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            features_list.append(self._extract_raw(c['dezenas'], last))
        return np.array(features_list, dtype=np.float64)

    def _extract_raw(self, dezenas, last_contest=None):
        d = sorted(dezenas)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        rep = len(set(d) & set(last_contest)) if last_contest else 8
        ent_trans = 0.0
        if len(self._repeat_history) >= 5:
            trans = [self._repeat_history[i+1]-self._repeat_history[i] for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs>0, probs, 1e-10)))
        amplitude = max(d)-min(d)
        std_pos = np.std(d) if len(d)>1 else 0.0
        compressao = std_pos/amplitude if amplitude>0 else 0.5
        return np.array([
            float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps)),
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))),
            ent_trans,
            float(len(set((x-1)//5 for x in d))),
            float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),
            float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),
            float(np.mean(d)-np.median(d)),
            float(sum(1 for g in gaps if g<=2)/len(gaps)),
            float(rep),
            float(sum(1 for x in d if x%2==0)),
            float(sum(1 for x in d if x in PRIMES)),
            float(sum(1 for x in d if x in MOLDURA)),
            float(sum(d)),
            float(max(d)-min(d)),
            compressao,
        ], dtype=np.float64)

    def _build_mahalanobis_model(self, raw_features):
        self._raw_feature_matrix = raw_features
        if SKLEARN_AVAILABLE and len(raw_features) > N_FEATURES:
            try:
                lw = LedoitWolf().fit(raw_features)
                self.precision_matrix = lw.precision_
                self.cov_matrix = lw.covariance_
            except:
                cov = np.cov(raw_features.T) + np.eye(N_FEATURES) * 1e-6
                self.precision_matrix = np.linalg.inv(cov)
                self.cov_matrix = cov
        else:
            cov = np.cov(raw_features.T) + np.eye(N_FEATURES) * 1e-6
            self.precision_matrix = np.linalg.inv(cov)
            self.cov_matrix = cov

        self._mean_vector = np.mean(raw_features, axis=0)
        self.historical_mahalanobis = np.array([self.mahalanobis_distance(f) for f in raw_features])

    def _build_percentile_kde(self):
        """Ajusta um KDE à distribuição empírica dos percentis das distâncias de Mahalanobis históricas."""
        pcts = np.array([np.mean(self.historical_mahalanobis <= d) for d in self.historical_mahalanobis])
        # bandwidth reduzido para capturar melhor a forma
        self._pct_kde = KernelDensity(bandwidth=KDE_BANDWIDTH).fit(pcts.reshape(-1, 1))
        self._hist_pct_log_dens = self._pct_kde.score_samples(pcts.reshape(-1, 1))

    def mahalanobis_distance(self, raw_features):
        diff = raw_features - self._mean_vector
        try:
            return float(np.sqrt(max(0, np.dot(np.dot(diff.T, self.precision_matrix), diff))))
        except:
            return float(np.linalg.norm(diff / (np.std(self.historical_mahalanobis) + 1e-10)))

    def compute_naturalness_score(self, game):
        """
        Score baseado na densidade empírica do percentil no espaço histórico.
        Jogos com percentis historicamente frequentes recebem pontuação alta.
        """
        raw = self.extract_raw_features(game)
        dist = self.mahalanobis_distance(raw)
        pct = np.mean(self.historical_mahalanobis <= dist)
        log_dens = self._pct_kde.score_samples([[pct]])[0]
        # Sigmoide centrada na mediana das log-densidades históricas
        hist_log_median = np.median(self._hist_pct_log_dens)
        score = 1.0 / (1.0 + np.exp(-(log_dens - hist_log_median) * 2.0))
        return float(score), float(pct), float(dist)

    def extract_raw_features(self, game):
        key = tuple(sorted(game))
        if key not in self._raw_cache:
            self._raw_cache[key] = self._extract_raw(game, None)
        return self._raw_cache[key]

    def extract_features(self, game, last_contest=None):
        key = (tuple(sorted(game)), tuple(last_contest) if last_contest else None)
        if key not in self._feature_cache:
            raw = self._extract_raw(game, last_contest)
            if self.scaler is not None:
                self._feature_cache[key] = self.scaler.transform(raw.reshape(1, -1)).flatten()
            else:
                self._feature_cache[key] = (raw - self.feature_means) / self.feature_stds
        return self._feature_cache[key]

    def build_feature_matrix(self):
        raw = self._build_raw_feature_matrix()
        if self.scaler is not None:
            return self.scaler.transform(raw)
        return (raw - self.feature_means) / self.feature_stds

    def compute_structural_penalty(self, game):
        d = sorted(game)
        penalty = 0.0
        total_pares = sum(1 for x in d if x%2==0)
        total_primos = sum(1 for x in d if x in PRIMES)
        total_moldura = sum(1 for x in d if x in MOLDURA)
        total_rep = len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8
        total_soma = sum(d)
        total_amplitude = max(d)-min(d)
        total_consec = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)

        actuals = {
            'pares': total_pares,
            'primos': total_primos,
            'moldura': total_moldura,
            'repeticoes': total_rep,
            'soma': total_soma,
            'consecutivos': total_consec,
            'amplitude': total_amplitude,
        }
        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name]-target)
                if dev > tol:
                    penalty += (dev - tol) * w

        if total_consec > MAX_TOTAL_CONSECUTIVOS:
            penalty += (total_consec - MAX_TOTAL_CONSECUTIVOS) * 8.0

        max_run = 1
        run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run > MAX_CONSECUTIVOS_RUN:
            penalty += (max_run - MAX_CONSECUTIVOS_RUN) * 5.0

        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        clusterizacao = sum(1 for g in gaps if g <= 2) / len(gaps)
        if clusterizacao > MAX_CLUSTERIZACAO:
            penalty += (clusterizacao - MAX_CLUSTERIZACAO) * 12.0

        return penalty

    def is_structurally_valid(self, game):
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

    def compute_kl_divergence(self, generated_features, n_bins=10):
        hist_features = self._raw_feature_matrix
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
                kl_total += np.sum(gen_hist * np.log(gen_hist / hist_hist))
            except:
                pass
        return kl_total

# ============================================================
# GAME CANDIDATE
# ============================================================
class GameCandidate:
    __slots__ = ('game', 'mask', 'features', 'naturalness_score', 'naturalness_pct', 'mahalanobis_dist')
    def __init__(self, game, mask, features, naturalness_score=0, naturalness_pct=0, mahalanobis_dist=0):
        self.game = game
        self.mask = mask
        self.features = features
        self.naturalness_score = naturalness_score
        self.naturalness_pct = naturalness_pct
        self.mahalanobis_dist = mahalanobis_dist

# ============================================================
# GERADOR
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
                st = sorted(test)
                run = 1; max_run = 1
                for i in range(len(st)-1):
                    if st[i+1]-st[i]==1:
                        run += 1; max_run = max(max_run, run)
                    else:
                        run = 1
                if max_run > MAX_CONSECUTIVOS_RUN:
                    s -= (max_run - MAX_CONSECUTIVOS_RUN) * 3
                total_consec = sum(1 for i in range(len(st)-1) if st[i+1]-st[i]==1)
                if total_consec > MAX_TOTAL_CONSECUTIVOS:
                    s -= (total_consec - MAX_TOTAL_CONSECUTIVOS) * 2
                scores.append(s)
            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores -= np.max(scores)
                probs = np.exp(scores / 2.0)
                probs /= probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else:
                chosen = random.choice(candidates)
            game.add(chosen)
            available.remove(chosen)
        return sorted(game)[:15]

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# OTIMIZADOR v31.1 (AJUSTES FINOS)
# ============================================================
class PortfolioOptimizerV31:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = LooseGenerator(self.extractor)
        self._mc_cache = {}
        self._mc_norm_params = None
        self._score_cache = {}

        self.historical_draws = [c['dezenas'] for c in self.contests]
        self.historical_masks = draw_masks_to_array(self.historical_draws)
        if len(self.historical_draws) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draw_masks_to_array(self.historical_draws)

        self.historical_features = np.array([self.extractor.extract_features(list(d), None) for d in self.historical_draws])
        recent_f = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix
        self.recent_centroid = np.mean(recent_f, axis=0)

    def _create_candidate(self, game):
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        nat_score, nat_pct, mahal_dist = self.extractor.compute_naturalness_score(game)
        return GameCandidate(game, mask, features, nat_score, nat_pct, mahal_dist)

    def _pair_coverage(self, portfolio):
        covered = set()
        for c in portfolio:
            for pair in combinations(sorted(c.game), 2):
                covered.add(pair)
        return len(covered) / comb(25, 2)

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

    def _correlation_penalty(self, portfolio):
        """Penaliza portfólio cujos jogos têm features altamente correlacionadas."""
        if len(portfolio) < 2:
            return 0.0
        fvs = np.array([c.features for c in portfolio])
        corr = np.corrcoef(fvs)
        triu = np.triu_indices_from(corr, k=1)
        mean_abs_corr = np.mean(np.abs(corr[triu]))
        if mean_abs_corr > CORRELATION_THRESHOLD:
            return (mean_abs_corr - CORRELATION_THRESHOLD) * 0.2
        return 0.0

    def _monte_carlo_hybrid(self, portfolio_candidates, n_simulations=500):
        cache_key = tuple(tuple(sorted(c.game)) for c in portfolio_candidates)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        avg_dists = np.linalg.norm(self.historical_features - self.recent_centroid, axis=1)
        # Softmax com temperatura ajustada
        logits = avg_dists / SOFTMAX_TEMPERATURE
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()

        indices = np.random.choice(len(self.historical_masks), size=n_simulations, p=weights)
        drawn_masks = self.historical_masks[indices]
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)

        total_weighted_score = 0.0
        for dm in drawn_masks:
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 13:
                    total_weighted_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
        avg_score = total_weighted_score / len(drawn_masks)

        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization(portfolio_size=len(portfolio_candidates))
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        normalized = max(0.0, min(1.0, (avg_score - p5) / (p95 - p5 + 1e-10)))
        self._mc_cache[cache_key] = normalized
        return normalized

    def _monte_carlo_raw(self, portfolio_candidates, n_simulations=300):
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)
        indices = np.random.choice(len(self.historical_masks), size=n_simulations)
        total_score = 0.0
        for idx in indices:
            drawn_mask = self.historical_masks[idx]
            for pm in portfolio_masks:
                hits = mask_intersection(pm, drawn_mask)
                if hits >= 13:
                    total_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
        return total_score / len(indices)

    def _compute_mc_normalization(self, portfolio_size=5, n_samples=200):
        raw_scores = []
        for _ in range(n_samples):
            rand_port = [self._create_candidate(self.generator.generate_pure_random()) for _ in range(portfolio_size)]
            raw = self._monte_carlo_raw(rand_port, 300)
            raw_scores.append(raw)
        raw_scores = np.array(raw_scores)
        return {'p5': float(np.percentile(raw_scores,5)), 'p95': float(np.percentile(raw_scores,95))}

    def _portfolio_score(self, portfolio):
        key = tuple(c.mask for c in portfolio)
        if key in self._score_cache:
            return self._score_cache[key]

        if self._pair_coverage(portfolio) > MAX_PAIR_COVERAGE:
            score = -1000.0
        elif not (MIN_GEO_DIVERSITY <= self._geometric_diversity(portfolio) <= MAX_GEO_DIVERSITY):
            score = -1000.0
        else:
            mc_score = self._monte_carlo_hybrid(portfolio)
            avg_naturalness = np.mean([c.naturalness_score for c in portfolio])
            corr_penalty = self._correlation_penalty(portfolio)
            score = (mc_score * 0.7 + avg_naturalness * 0.1 +
                     self._portfolio_diversity(portfolio) * 0.1 +
                     self._geometric_diversity(portfolio) * 0.1 - corr_penalty)
        self._score_cache[key] = score
        return score

    def _mutate_candidate(self, candidate):
        for _ in range(20):
            mutated = list(candidate.game)
            for _ in range(random.randint(1,4)):
                pos = random.randint(0,14)
                avail = [d for d in range(1,26) if d not in mutated]
                if avail:
                    mutated[pos] = random.choice(avail)
            mutated = sorted(mutated)[:15]
            if self.extractor.is_structurally_valid(mutated):
                return self._create_candidate(mutated)
        return candidate

    def _weighted_sample(self, candidates, k):
        if len(candidates) <= k:
            return candidates
        scores = np.array([c.naturalness_score for c in candidates])
        probs = scores / scores.sum()
        indices = np.random.choice(len(candidates), size=k, replace=False, p=probs)
        return [candidates[i] for i in indices]

    def optimize(self, n_games=5, n_candidates=12000, iterations=100):
        print(f"🎯 Carteira CONCENTRADA: {n_games} jogos")
        print(f"📊 Naturalness score (KDE bandwidth={KDE_BANDWIDTH}) + anti‑correlação (th={CORRELATION_THRESHOLD})")
        print(f"🌡️ Softmax temperatura={SOFTMAX_TEMPERATURE}")
        print(f"⚖️ Pesos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")

        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Fase 1"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen and self.extractor.is_structurally_valid(game):
                seen.add(key)
                raw_pool.append(game)

        top_pool = random.sample(raw_pool, min(5000, len(raw_pool)))
        candidates = [self._create_candidate(g) for g in tqdm(top_pool, desc="Fase 2")]

        portfolio_candidates = self._weighted_sample(candidates, min(200, len(candidates)))
        portfolio_candidates.sort(key=lambda c: c.naturalness_score, reverse=True)
        portfolio, portfolio_masks = [], []
        for c in portfolio_candidates:
            if len(portfolio) >= n_games: break
            if portfolio_masks and max(mask_intersection(c.mask, pm) for pm in portfolio_masks) > 10:
                continue
            portfolio.append(c)
            portfolio_masks.append(c.mask)

        elite_pool = self._weighted_sample(candidates, min(400, len(candidates)))
        best_portfolio, best_score = list(portfolio), self._portfolio_score(portfolio)

        for it in tqdm(range(iterations), desc="Annealing"):
            temp = 1.0 * (0.95 ** it)
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
            if new_score > best_score:
                best_portfolio, best_score = list(new_portfolio), new_score
            elif random.random() < np.exp((new_score - self._portfolio_score(portfolio)) / max(0.01, temp)):
                portfolio = new_portfolio

        return [c.game for c in best_portfolio], best_score

    def backtest(self, portfolio, test_draws):
        n_success, total_premio = 0, 0.0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        for draw in test_draws:
            draw_mask = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, draw_mask)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits,0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if len(test_draws)>0 else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {
            'empirical': prob, 'theoretical': theo_prob,
            'lift': prob/theo_prob if theo_prob>0 else 1.0,
            'n_test': len(test_draws), 'n_success': n_success,
            'total_premio': total_premio, 'total_custo': total_custo,
            'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
            'hit_distribution': hit_counts
        }

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
        opt = PortfolioOptimizerV31(train_data)
        portfolio, _ = opt.optimize(n_games, n_candidates=12000, iterations=50)
        bt = opt.backtest(portfolio, test_data)
        bt_rand = opt.backtest([opt.generator.generate_pure_random() for _ in range(n_games)], test_data)
        results.append({
            'window': w, 'diff_lift': bt['lift']-bt_rand['lift'],
            'diff_roi': bt['roi']-bt_rand['roi'],
            'strat_14': bt['hit_distribution'].get(14,0),
            'rand_14': bt_rand['hit_distribution'].get(14,0),
        })
        print(f"   Janela {w}: diff_lift={bt['lift']-bt_rand['lift']:+.3f} "
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
    print("🧬 GERADOR DE CARTEIRA v31.1 - DISTRIBUIÇÃO HISTÓRICA + AJUSTES FINOS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado."); return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print(f"\n📊 Naturalness score (KDE bandwidth={KDE_BANDWIDTH})")
    print(f"🌡️ Softmax temperatura={SOFTMAX_TEMPERATURE}")
    print(f"🔗 Anti‑correlação threshold={CORRELATION_THRESHOLD}")
    print(f"💰 Pesos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")

    op = input("Opções: 1. Gerar carteira | 2. Walk-forward | 3. Ambos\nEscolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        t0 = time.time()
        opt = PortfolioOptimizerV31(contests)
        print(f"   ✅ Init {time.time()-t0:.1f}s")
        portfolio, _ = opt.optimize(5, 12000, 100)
        last = contests[-1]['dezenas']
        gen_features = np.array([opt.extractor.extract_features(g, last) for g in portfolio])
        kl = opt.extractor.compute_kl_divergence(gen_features)
        corr_pen = opt._correlation_penalty([opt._create_candidate(g) for g in portfolio])
        pcts = [opt.extractor.compute_naturalness_score(g)[1] for g in portfolio]
        print(f"\n📊 KL Divergence: {kl:.3f} (ideal < 40)")
        print(f"   Penalidade de correlação: {corr_pen:.3f}")
        print(f"   Percentis do portfólio: {[f'{p:.3f}' for p in pcts]}")
        for i, g in enumerate(portfolio, 1):
            p = sum(1 for d in g if d%2==0)
            pr = sum(1 for d in g if d in PRIMES)
            m = sum(1 for d in g if d in MOLDURA)
            rep = len(set(g) & set(last))
            cons = sum(1 for j in range(len(g)-1) if g[j+1]-g[j]==1)
            nat, pct, mahal = opt.extractor.compute_naturalness_score(g)
            print(f"   {i:2d}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep} Cons:{cons} Nat:{nat:.3f} Pct:{pct:.3f} Mahal:{mahal:.1f}")
        if len(contests) > 200:
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Distribuição: 11pts:{bt['hit_distribution'].get(11,0)} "
                  f"12pts:{bt['hit_distribution'].get(12,0)} 13pts:{bt['hit_distribution'].get(13,0)} "
                  f"14pts:{bt['hit_distribution'].get(14,0)} 15pts:{bt['hit_distribution'].get(15,0)}")

    if op in ("2", "3"):
        walk_forward_validation(contests, 10, 500, 50, 5)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
