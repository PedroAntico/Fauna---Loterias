#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v13
=================================================
CORREÇÕES BASEADAS EM EVIDÊNCIA EMPÍRICA (v12 → v13):
✅ Score exponencial RECALIBRADO (11:1, 12:3, 13:7, 14:20, 15:60)
✅ Geo diversity com META MÍNIMA (0.55) e MÁXIMA (0.80)
✅ Penalidade de similaridade estrutural (cosine entre features)
✅ Repair com similaridade MODERADA (faixa 0.3-0.7, não máxima)
✅ Normalização EMPÍRICA do Monte Carlo (percentis de carteiras aleatórias)
✅ Ensemble estrutural melhorado (pares + score estrutural)
✅ HARD CAPS mantidos (pair_cov ≤ 0.90, geo_div ≤ 0.80)
✅ Structural penalty, soft-hard gate, z-score, GMM, bootstrap preservados
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
from collections import Counter, defaultdict
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
    from sklearn.mixture import GaussianMixture
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
CENTRO = {7,8,9,12,13,14,17,18,19}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# Features topológicas v13 (17 dimensões)
FEATURE_NAMES_V13 = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
]
IDX = {name: i for i, name in enumerate(FEATURE_NAMES_V13)}

# Pesos para penalidade estrutural HARD-SOFT
STRUCTURAL_TARGETS = {
    'pares': (7.5, 1.5, 2.0),
    'primos': (5.0, 1.5, 2.5),
    'moldura': (9.5, 1.5, 1.5),
    'repeticoes': (9.0, 1.5, 1.5),
    'soma': (195.0, 20.0, 1.0),
    'consecutivos': (5.5, 2.0, 1.0),
    'amplitude': (22.0, 3.0, 1.0),
}

# HARD CAPS (v13: geo_div com meta MÍNIMA também)
MAX_PAIR_COVERAGE = 0.90
MIN_GEO_DIVERSITY = 0.55      # NOVO: penalizar abaixo disso
MAX_GEO_DIVERSITY = 0.80
TARGET_ENTROPY_MIN = 0.90
TARGET_ENTROPY_MAX = 0.96
STRUCTURAL_REJECT_THRESHOLD = 15.0

# Score EXPONENCIAL RECALIBRADO (v13: reduzido peso de 15)
EXPONENTIAL_WEIGHTS = {
    11: 1.0,
    12: 3.0,
    13: 7.0,
    14: 20.0,
    15: 60.0,
}

# Features com sinal temporal
TEMPORAL_FEATURES = {
    'moldura': 0.30,
    'amplitude': 0.25,
    'energia_jogo': 0.20,
    'densidade_local': 0.15,
    'clusterizacao': 0.10,
}

# Faixa de similaridade para repair moderado
REPAIR_SIMILARITY_MIN = 0.3
REPAIR_SIMILARITY_MAX = 0.7

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return None
    contests = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17:
                continue
            try:
                concurso = int(parts[0])
                data = parts[1]
                dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                if len(dezenas) != 15 or len(set(dezenas)) != 15:
                    continue
                if any(x < 1 or x > 25 for x in dezenas):
                    continue
                contests.append({
                    'concurso': concurso, 'data': data,
                    'dezenas': sorted(dezenas)
                })
            except (ValueError, IndexError):
                continue
        contests.sort(key=lambda x: x['concurso'])
        print(f"✅ {len(contests)} concursos válidos")
        return contests
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


# ============================================================
# EXTRATOR DE FEATURES v13 (com z-score)
# ============================================================
class TopologicalFeatureExtractorV13:
    def __init__(self, contests):
        self.contests = contests
        self._repeat_history = []
        for i, c in enumerate(contests):
            if i > 0:
                self._repeat_history.append(
                    len(set(contests[i-1]['dezenas']) & set(c['dezenas'])))
            else:
                self._repeat_history.append(0)
        self._recent_freq = self._compute_recent_freq()
        raw_features = self._build_raw_feature_matrix()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        if self.scaler is not None and len(raw_features) > 10:
            self.scaler.fit(raw_features)
        self.feature_means = np.mean(raw_features, axis=0)
        self.feature_stds = np.std(raw_features, axis=0) + 1e-10

    def _compute_recent_freq(self, window=50):
        freq = Counter()
        start = max(0, len(self.contests) - window)
        for c in self.contests[start:]:
            freq.update(c['dezenas'])
        total = len(self.contests[start:])
        return {d: freq.get(d, 0) / total for d in range(1, 26)}

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
            trans = [self._repeat_history[i+1]-self._repeat_history[i]
                     for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs>0, probs, 1e-10)))

        return np.array([
            float(np.mean(gaps)), float(np.var(gaps)),
            float(max(gaps)), float(min(gaps)),
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))),
            ent_trans,
            float(len(set((x-1)//5 for x in d))),
            float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),
            float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),
            float(np.mean(d) - np.median(d)),
            float(sum(1 for g in gaps if g <= 2) / len(gaps)),
            float(rep),
            float(sum(1 for x in d if x % 2 == 0)),
            float(sum(1 for x in d if x in PRIMES)),
            float(sum(1 for x in d if x in MOLDURA)),
            float(sum(d)),
            float(max(d) - min(d)),
        ], dtype=np.float64)

    def extract_features(self, game, last_contest=None):
        raw = self._extract_raw(game, last_contest)
        if self.scaler is not None:
            return self.scaler.transform(raw.reshape(1, -1)).flatten()
        return (raw - self.feature_means) / self.feature_stds

    def build_feature_matrix(self):
        raw = self._build_raw_feature_matrix()
        if self.scaler is not None:
            return self.scaler.transform(raw)
        return (raw - self.feature_means) / self.feature_stds

    def compute_structural_penalty(self, game):
        d = sorted(game)
        penalty = 0.0

        actuals = {
            'pares': sum(1 for x in d if x % 2 == 0),
            'primos': sum(1 for x in d if x in PRIMES),
            'moldura': sum(1 for x in d if x in MOLDURA),
            'repeticoes': len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8,
            'soma': sum(d),
            'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),
            'amplitude': max(d) - min(d),
        }

        for name, (target, tolerance, weight) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                deviation = abs(actuals[name] - target)
                if deviation > tolerance:
                    excess = deviation - tolerance
                    penalty += excess * weight

        return penalty

    def is_structurally_valid(self, game):
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

    def compute_structural_score(self, game):
        penalty = self.compute_structural_penalty(game)
        return np.exp(-penalty / 5.0)

    def get_recent_freq_bonus(self, game):
        return np.mean([self._recent_freq.get(d, 0) for d in game])

    def compute_temporal_score(self, game, all_features, recent_window=20):
        game_feats = self.extract_features(game, None)
        score = 0.0
        total_weight = 0.0

        for name, weight in TEMPORAL_FEATURES.items():
            if name in IDX:
                idx = IDX[name]
                game_val = game_feats[idx]

                if len(all_features) >= recent_window:
                    recent = all_features[-recent_window:, idx]
                else:
                    recent = all_features[:, idx]
                recent_mean = np.mean(recent)
                recent_std = np.std(recent) + 1e-10

                z_recent = (game_val - recent_mean) / recent_std
                base_score = np.exp(-0.5 * z_recent**2)
                score += weight * base_score
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.5


# ============================================================
# MODELO DE DISTRIBUIÇÃO (GMM com features padronizadas)
# ============================================================
class DistributionModelV13:
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        self._build_gmm()
        self._gmm_norm = self._compute_gmm_norm()

    def _build_gmm(self):
        if SKLEARN_AVAILABLE and self.feature_matrix.shape[0] > 100:
            try:
                n_comp = min(6, self.feature_matrix.shape[0] // 200)
                self.gmm = GaussianMixture(
                    n_components=max(3, n_comp), random_state=42)
                self.gmm.fit(self.feature_matrix)
                self._has_gmm = True
                return
            except Exception:
                pass
        self._has_gmm = False

    def _compute_gmm_norm(self):
        if self._has_gmm:
            scores = self.gmm.score_samples(self.feature_matrix)
            return {'min': float(np.min(scores)), 'max': float(np.max(scores))}
        return {'min': -100.0, 'max': 100.0}

    def score_samples_normalized(self, features):
        if self._has_gmm:
            raw = float(self.gmm.score_samples(features.reshape(1, -1))[0])
            rng = self._gmm_norm['max'] - self._gmm_norm['min']
            return (raw - self._gmm_norm['min']) / rng if rng > 0 else 0.5
        return 0.5

    def predict_cluster(self, features):
        if self._has_gmm:
            return int(self.gmm.predict(features.reshape(1, -1))[0])
        return 0

    @property
    def n_components(self):
        return self.gmm.n_components if self._has_gmm else 1


# ============================================================
# GERADOR LIVRE
# ============================================================
class FreeGeneratorV13:
    def __init__(self, last_contest=None, extractor=None):
        self.last = set(last_contest) if last_contest else None
        self.extractor = extractor

    def generate_one(self):
        max_attempts = 50
        for _ in range(max_attempts):
            game = self._generate_raw()
            if self.extractor is not None:
                if not self.extractor.is_structurally_valid(game):
                    continue
            return game
        return self._generate_raw()

    def _generate_raw(self):
        game = set()
        available = set(range(1, 26))
        if self.last and random.random() < 0.3:
            rep_pool = list(self.last & available)
            if rep_pool:
                n = random.randint(5, 10)
                game.update(random.sample(rep_pool, min(n, len(rep_pool))))
                available -= game
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                test = game | {d}
                s = len(set((x-1)//5 for x in test)) * 3
                st = sorted(test)
                cons = sum(1 for i in range(len(st)-1) if st[i+1]-st[i]==1)
                if cons > 6:
                    s -= (cons - 6) * 1.5
                scores.append(s)
            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores = scores - np.max(scores)
                probs = np.exp(scores / 3.0)
                probs = probs / probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else:
                chosen = random.choice(candidates)
            game.add(chosen)
            available.remove(chosen)
        return sorted(game)[:15]

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))


# ============================================================
# OTIMIZADOR DE CARTEIRA HÍBRIDA v13
# ============================================================
class HybridPortfolioOptimizerV13:
    """
    Otimizador v13: REPAIR MODERADO + SCORE RECALIBRADO + NORMALIZAÇÃO EMPÍRICA.
    """
    def __init__(self, contests):
        self.contests = contests
        self.extractor = TopologicalFeatureExtractorV13(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.dist_model = DistributionModelV13(self.feature_matrix)
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = FreeGeneratorV13(self.last, self.extractor)
        self._mc_cache = {}
        # Normalização empírica (pré-computada)
        self._mc_norm_params = None

    def _compute_mc_normalization(self, n_samples=200, portfolio_size=10):
        """Pré-computa parâmetros de normalização empírica do Monte Carlo."""
        if self._mc_norm_params is not None:
            return self._mc_norm_params

        raw_scores = []
        for _ in range(n_samples):
            random_portfolio = [self.generator.generate_pure_random()
                               for _ in range(portfolio_size)]
            raw = self._monte_carlo_weighted_sum_raw(random_portfolio, n_simulations=500)
            raw_scores.append(raw)

        raw_scores = np.array(raw_scores)
        self._mc_norm_params = {
            'p5': float(np.percentile(raw_scores, 5)),
            'p95': float(np.percentile(raw_scores, 95)),
            'median': float(np.median(raw_scores)),
            'mean': float(np.mean(raw_scores)),
            'std': float(np.std(raw_scores)),
        }
        return self._mc_norm_params

    def _score_game_central(self, game):
        features = self.extractor.extract_features(game, self.last)
        gmm_score = self.dist_model.score_samples_normalized(features)
        structural_score = self.extractor.compute_structural_score(game)
        combined = gmm_score * 0.6 + structural_score * 0.4
        return combined, features, self.dist_model.predict_cluster(features)

    def _score_game_temporal(self, game):
        temporal_score = self.extractor.compute_temporal_score(
            game, self.feature_matrix)
        structural_score = self.extractor.compute_structural_score(game)
        combined = temporal_score * 0.5 + structural_score * 0.5
        features = self.extractor.extract_features(game, self.last)
        return combined, features, self.dist_model.predict_cluster(features)

    def _feature_similarity(self, feats1, feats2):
        """Similaridade de cosseno entre vetores de features."""
        dot = np.dot(feats1, feats2)
        norm = np.linalg.norm(feats1) * np.linalg.norm(feats2)
        if norm < 1e-10:
            return 1.0
        return dot / norm

    def _find_most_similar_valid(self, game, pool, max_intersection=8):
        """
        REPAIR MODERADO: busca jogos com similaridade na faixa 0.3-0.7.
        Não o mais similar (evita colapso estrutural).
        """
        game_features = self.extractor.extract_features(game, self.last)
        valid_candidates = []

        sample = random.sample(pool, min(500, len(pool)))

        for candidate in sample:
            if candidate == game:
                continue
            if not self.extractor.is_structurally_valid(candidate):
                continue

            cand_features = self.extractor.extract_features(candidate, self.last)
            similarity = self._feature_similarity(game_features, cand_features)

            # Faixa moderada de similaridade
            if REPAIR_SIMILARITY_MIN <= similarity <= REPAIR_SIMILARITY_MAX:
                valid_candidates.append((similarity, candidate))

        if valid_candidates:
            # Escolher aleatoriamente entre os válidos (diversidade)
            return random.choice(valid_candidates)[1]

        # Fallback: similaridade mais próxima (qualquer)
        best_candidate = None
        best_similarity = -float('inf')
        for candidate in sample:
            if candidate == game:
                continue
            if not self.extractor.is_structurally_valid(candidate):
                continue
            cand_features = self.extractor.extract_features(candidate, self.last)
            similarity = self._feature_similarity(game_features, cand_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = candidate

        return best_candidate if best_candidate is not None else self.generator.generate_one()

    def _greedy_marginal_coverage_balanced(self, pool, existing_portfolio,
                                            n_select, max_intersection=7):
        covered_pairs = set()
        for g in existing_portfolio:
            for pair in combinations(sorted(g), 2):
                covered_pairs.add(pair)

        selected = list(existing_portfolio)
        remaining = [g for g in pool if g not in selected]

        for _ in range(n_select):
            if not remaining:
                break

            current_coverage = len(covered_pairs) / comb(25, 2)
            if current_coverage >= MAX_PAIR_COVERAGE:
                coverage_weight = 0.1
                structural_weight = 0.9
            else:
                coverage_weight = 0.5
                structural_weight = 0.5

            best_game = None
            best_score = -float('inf')

            sample = random.sample(remaining, min(300, len(remaining)))

            for game in sample:
                if selected:
                    max_inter = max(len(set(game) & set(sg))
                                    for sg in selected)
                    if max_inter > max_intersection:
                        continue

                penalty = self.extractor.compute_structural_penalty(game)
                if penalty > STRUCTURAL_REJECT_THRESHOLD:
                    continue

                new_pairs = set()
                for pair in combinations(sorted(game), 2):
                    if pair not in covered_pairs:
                        new_pairs.add(pair)

                coverage_score = len(new_pairs) / 105.0
                structural_score = self.extractor.compute_structural_score(game)
                combined = (coverage_score * coverage_weight +
                           structural_score * structural_weight)

                if combined > best_score:
                    best_score = combined
                    best_game = game

            if best_game:
                selected.append(best_game)
                remaining.remove(best_game)
                for pair in combinations(sorted(best_game), 2):
                    covered_pairs.add(pair)

        return selected

    def _pair_coverage(self, portfolio):
        covered = set()
        for g in portfolio:
            for pair in combinations(sorted(g), 2):
                covered.add(pair)
        return len(covered) / comb(25, 2)

    def _portfolio_entropy(self, portfolio):
        all_dezenas = [d for g in portfolio for d in g]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        return float(entropy(probs) / np.log(25))

    def _portfolio_diversity(self, portfolio):
        if len(portfolio) < 2:
            return 1.0
        sims = []
        for i in range(len(portfolio)):
            for j in range(i+1, len(portfolio)):
                sims.append(len(set(portfolio[i]) & set(portfolio[j])))
        return 1.0 - np.mean(sims) / 15.0

    def _geometric_diversity(self, portfolio):
        if len(portfolio) < 2:
            return 0.5

        feature_vectors = []
        for g in portfolio:
            feats = self.extractor.extract_features(g, self.last)
            feature_vectors.append(feats)
        feature_vectors = np.array(feature_vectors)

        distances = []
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                dist = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
                distances.append(dist)

        avg_dist = np.mean(distances) if distances else 0
        max_expected = 2 * np.sqrt(len(FEATURE_NAMES_V13))
        return avg_dist / max_expected

    def _structural_overlap_penalty(self, portfolio):
        """
        Penalidade de similaridade estrutural entre jogos.
        Penaliza carteiras onde os jogos têm features muito similares.
        """
        if len(portfolio) < 2:
            return 0.0

        feature_vectors = []
        for g in portfolio:
            feats = self.extractor.extract_features(g, self.last)
            feature_vectors.append(feats)
        feature_vectors = np.array(feature_vectors)

        similarities = []
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                sim = self._feature_similarity(feature_vectors[i], feature_vectors[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0
        # Penalizar similaridade alta (>0.85)
        if avg_similarity > 0.85:
            return (avg_similarity - 0.85) * 5.0
        return 0.0

    def _average_structural_score(self, portfolio):
        scores = [self.extractor.compute_structural_score(g) for g in portfolio]
        return np.mean(scores) if scores else 0.5

    def _monte_carlo_weighted_sum_raw(self, portfolio, n_simulations=2000):
        """Monte Carlo com soma ponderada (VALOR BRUTO, sem normalização)."""
        historical_draws = [set(c['dezenas']) for c in self.contests[-500:]]
        if len(historical_draws) < 100:
            historical_draws = [
                set(np.random.choice(range(1, 26), 15, replace=False))
                for _ in range(500)
            ]

        total_score = 0.0
        n_eval = min(n_simulations, len(historical_draws))

        for i in range(n_eval):
            drawn = historical_draws[random.randint(0, len(historical_draws) - 1)]
            draw_score = 0.0
            for g in portfolio:
                hits = len(set(g) & drawn)
                if hits >= 11:
                    draw_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
            total_score += draw_score

        return total_score / n_eval

    def _monte_carlo_weighted_sum(self, portfolio, n_simulations=2000):
        """
        Monte Carlo com NORMALIZAÇÃO EMPÍRICA.
        Usa percentis de carteiras aleatórias para calibrar a escala.
        """
        cache_key = tuple(tuple(sorted(g)) for g in portfolio)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        raw = self._monte_carlo_weighted_sum_raw(portfolio, n_simulations)

        # Normalização empírica
        params = self._compute_mc_normalization()
        p5, p95 = params['p5'], params['p95']

        if p95 - p5 > 1e-10:
            normalized = (raw - p5) / (p95 - p5)
        else:
            normalized = 0.5

        # Clampar para [0, 1]
        normalized = max(0.0, min(1.0, normalized))

        self._mc_cache[cache_key] = normalized
        if len(self._mc_cache) > 500:
            keys = list(self._mc_cache.keys())[:250]
            for k in keys:
                del self._mc_cache[k]
        return normalized

    def _repair_portfolio(self, portfolio, pool):
        """
        REPAIR MODERADO: Substitui jogos problemáticos por similares na faixa 0.3-0.7.
        """
        repaired = list(portfolio)
        changed = True
        max_iterations = 20
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            pair_cov = self._pair_coverage(repaired)
            geo_div = self._geometric_diversity(repaired)

            if pair_cov > MAX_PAIR_COVERAGE:
                best_idx = None
                best_contribution = -1
                for idx, game in enumerate(repaired):
                    other_pairs = set()
                    for j, g2 in enumerate(repaired):
                        if j != idx:
                            for pair in combinations(sorted(g2), 2):
                                other_pairs.add(pair)
                    game_pairs = set(combinations(sorted(game), 2))
                    contribution = len(game_pairs - other_pairs)
                    if contribution > best_contribution:
                        best_contribution = contribution
                        best_idx = idx

                if best_idx is not None:
                    old_game = repaired[best_idx]
                    new_game = self._find_most_similar_valid(old_game, pool)
                    repaired[best_idx] = new_game
                    changed = True

            elif geo_div > MAX_GEO_DIVERSITY or geo_div < MIN_GEO_DIVERSITY:
                feature_vectors = []
                for g in repaired:
                    feats = self.extractor.extract_features(g, self.last)
                    feature_vectors.append(feats)
                feature_vectors = np.array(feature_vectors)
                centroid = np.mean(feature_vectors, axis=0)
                distances = [np.linalg.norm(fv - centroid) for fv in feature_vectors]

                if geo_div > MAX_GEO_DIVERSITY:
                    best_idx = np.argmax(distances)
                else:
                    best_idx = np.argmin(distances)

                old_game = repaired[best_idx]
                new_game = self._find_most_similar_valid(old_game, pool)
                repaired[best_idx] = new_game
                changed = True

        return repaired

    def _portfolio_score(self, portfolio):
        """Score com tetos HARD + meta MÍNIMA de geo diversity."""
        pair_cov = self._pair_coverage(portfolio)
        geo_div = self._geometric_diversity(portfolio)

        # HARD CAPS
        if pair_cov > MAX_PAIR_COVERAGE:
            return -1000.0
        if geo_div > MAX_GEO_DIVERSITY:
            return -1000.0

        # Score principal (normalizado empiricamente)
        weighted_sum = self._monte_carlo_weighted_sum(portfolio, n_simulations=2000)

        # Componentes secundários
        structural = self._average_structural_score(portfolio)
        entropy_val = self._portfolio_entropy(portfolio)
        diversity = self._portfolio_diversity(portfolio)

        # Penalidades
        entropy_penalty = 0.0
        if entropy_val < TARGET_ENTROPY_MIN:
            entropy_penalty = (TARGET_ENTROPY_MIN - entropy_val) * 3.0
        elif entropy_val > TARGET_ENTROPY_MAX:
            entropy_penalty = (entropy_val - TARGET_ENTROPY_MAX) * 3.0

        # Penalidade de similaridade estrutural (NOVO)
        struct_overlap = self._structural_overlap_penalty(portfolio)

        # Penalidade de geo diversity mínima
        geo_min_penalty = 0.0
        if geo_div < MIN_GEO_DIVERSITY:
            geo_min_penalty = (MIN_GEO_DIVERSITY - geo_div) * 5.0

        return (weighted_sum * 0.35 + structural * 0.25 +
                diversity * 0.15 + geo_div * 0.10 -
                entropy_penalty * 0.10 - struct_overlap * 0.10 - geo_min_penalty * 0.05)

    def _mutate_game(self, game):
        max_attempts = 20
        for _ in range(max_attempts):
            mutated = list(game)
            n_changes = random.randint(1, 3)
            for _ in range(n_changes):
                pos = random.randint(0, 14)
                available = [d for d in range(1, 26) if d not in mutated]
                if available:
                    mutated[pos] = random.choice(available)
            mutated = sorted(mutated)[:15]
            if self.extractor.is_structurally_valid(mutated):
                return mutated
        return sorted(game)[:15]

    def optimize_hybrid(self, n_games=10, n_candidates=200000, iterations=100):
        print(f"\n🎯 OTIMIZANDO CARTEIRA v13 ({n_games} jogos)...")

        n_central = max(1, int(n_games * 0.40))
        n_coverage = max(1, int(n_games * 0.35))
        n_temporal = n_games - n_central - n_coverage

        print(f"   Composição: {n_central} centrais + {n_coverage} cobertura + {n_temporal} temporais")
        print(f"   HARD CAPS: pair_cov ≤ {MAX_PAIR_COVERAGE}, geo_div ∈ [{MIN_GEO_DIVERSITY}, {MAX_GEO_DIVERSITY}]")
        print(f"   Score EXPONENCIAL RECALIBRADO: {EXPONENTIAL_WEIGHTS}")
        print(f"   Normalização Monte Carlo: EMPÍRICA (percentis)")

        # Pré-computar normalização
        self._compute_mc_normalization(n_samples=200, portfolio_size=n_games)

        print(f"   Gerando {n_candidates:,} candidatos...")
        pool_central = []
        pool_temporal = []
        pool_all = []
        seen = set()
        n_rejected = 0
        for _ in tqdm(range(n_candidates), desc="Candidatos"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                if not self.extractor.is_structurally_valid(game):
                    n_rejected += 1
                    continue
                sc, feats, cluster = self._score_game_central(game)
                pool_central.append((sc, game, feats, cluster))
                st, feats2, cluster2 = self._score_game_temporal(game)
                pool_temporal.append((st, game, feats2, cluster2))
                pool_all.append(game)

        if n_rejected > 0:
            print(f"   🚫 {n_rejected} jogos rejeitados por penalidade estrutural")

        pool_central.sort(key=lambda x: x[0], reverse=True)
        pool_temporal.sort(key=lambda x: x[0], reverse=True)

        # Centrais
        central_games = []
        cluster_counts = defaultdict(int)
        max_per_cluster = max(2, n_central // self.dist_model.n_components + 1)
        for sc, game, feats, cluster in pool_central:
            if len(central_games) >= n_central:
                break
            if cluster_counts[cluster] >= max_per_cluster:
                continue
            if any(len(set(game) & set(sg)) > 8 for sg in central_games):
                continue
            central_games.append(game)
            cluster_counts[cluster] += 1

        # Cobertura via greedy EQUILIBRADO
        coverage_games = self._greedy_marginal_coverage_balanced(
            [g for g in pool_all if g not in central_games],
            central_games, n_coverage, max_intersection=7
        )
        coverage_games = coverage_games[len(central_games):]

        # Temporais
        temporal_games = []
        for st, game, feats, cluster in pool_temporal:
            if game in central_games or game in coverage_games:
                continue
            if len(temporal_games) >= n_temporal:
                break
            if any(len(set(game) & set(sg)) > 8 for sg in temporal_games):
                continue
            if any(len(set(game) & set(sg)) > 9
                   for sg in central_games + coverage_games):
                continue
            temporal_games.append(game)

        # Completar
        while len(central_games) < n_central:
            game = self.generator.generate_one()
            if game not in central_games:
                central_games.append(game)
        while len(coverage_games) < n_coverage:
            game = self.generator.generate_one()
            if game not in central_games and game not in coverage_games:
                coverage_games.append(game)
        while len(temporal_games) < n_temporal:
            game = self.generator.generate_one()
            if (game not in central_games and game not in coverage_games
                    and game not in temporal_games):
                temporal_games.append(game)

        portfolio = central_games + coverage_games + temporal_games

        # REPAIR MODERADO
        portfolio = self._repair_portfolio(portfolio, pool_all)
        print(f"   ✅ Carteira inicial: pair_cov={self._pair_coverage(portfolio):.3f}, geo_div={self._geometric_diversity(portfolio):.3f}")

        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)
        current_score = best_score

        # Simulated Annealing
        temp = 1.0
        elite_pool = [(s, g) for s, g, _, _ in pool_central[:len(pool_central)//4]]

        for it in tqdm(range(iterations), desc="Annealing"):
            temp *= 0.95
            new_portfolio = list(portfolio)
            idx = random.randint(0, len(new_portfolio) - 1)

            if random.random() < 0.4 and elite_pool:
                _, new_game = random.choice(elite_pool)
            elif random.random() < 0.7:
                new_game = self._mutate_game(new_portfolio[idx])
            else:
                new_game = self.generator.generate_one()

            too_similar = False
            for j, sg in enumerate(new_portfolio):
                if j != idx and len(set(new_game) & set(sg)) > 8:
                    too_similar = True
                    break
            if too_similar:
                continue

            new_portfolio[idx] = new_game
            new_score = self._portfolio_score(new_portfolio)

            if new_score < -100:
                continue

            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / max(0.01, temp)):
                portfolio = new_portfolio
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_portfolio = list(portfolio)

        return best_portfolio, best_score

    def generate_ensemble_structural(self, n_strategies=3, n_games_per_strategy=5):
        """
        ENSEMBLE ESTRUTURAL MELHORADO: vota em pares + score estrutural.
        """
        print(f"\n🤝 ENSEMBLE ESTRUTURAL...")
        strategy_results = []

        for s in range(n_strategies):
            portfolio, score = self.optimize_hybrid(
                n_games=n_games_per_strategy, n_candidates=50000, iterations=30)

            pair_counter = Counter()
            for g in portfolio:
                for pair in combinations(sorted(g), 2):
                    pair_counter[pair] += 1

            weighted_sum = self._monte_carlo_weighted_sum(portfolio, n_simulations=1000)
            structural_avg = self._average_structural_score(portfolio)

            strategy_results.append({
                'portfolio': portfolio,
                'weighted_sum': weighted_sum,
                'structural': structural_avg,
                'pairs': pair_counter,
                'dezenas': [d for g in portfolio for d in g],
            })

        # Votação por pares ponderada por score combinado
        pair_votes = Counter()
        for sr in strategy_results:
            weight = sr['weighted_sum'] * 0.6 + sr['structural'] * 0.4
            for pair, count in sr['pairs'].items():
                pair_votes[pair] += weight * count

        top_pairs = [pair for pair, _ in pair_votes.most_common(100)]

        # Gerar jogos com muitos pares do consenso + penalidade estrutural
        consensus_games = []
        seen = set()
        for _ in range(30000):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                if not self.extractor.is_structurally_valid(game):
                    continue
                game_pairs = set(combinations(sorted(game), 2))
                consensus_count = len(game_pairs & set(top_pairs[:100]))
                if consensus_count >= 20:
                    consensus_games.append(game)

        selected = []
        for game in consensus_games[:500]:
            if len(selected) >= 10:
                break
            if not any(len(set(game) & set(sg)) > 8 for sg in selected):
                selected.append(game)

        return selected if len(selected) >= 5 else consensus_games[:10]

    def generate_pure_random_portfolio(self, n_games=10):
        return [self.generator.generate_pure_random() for _ in range(n_games)]

    def generate_coverage_baseline(self, n_games=10):
        pool = []
        seen = set()
        for _ in range(50000):
            game = self.generator.generate_pure_random()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                pool.append(game)
        selected = self._greedy_marginal_coverage_balanced(
            pool, [], n_games, max_intersection=7)
        return selected[:n_games]

    def backtest(self, portfolio, test_draws):
        n_success = 0
        for draw in test_draws:
            actual = set(draw['dezenas'])
            if any(len(set(g) & actual) >= 11 for g in portfolio):
                n_success += 1
        prob = n_success / len(test_draws) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11, 16))
        p_none = (1 - p_single) ** len(portfolio)
        theo_prob = 1 - p_none
        return {
            'empirical': prob, 'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws), 'n_success': n_success,
        }


# ============================================================
# WALK-FORWARD COM STABILITY SCORE
# ============================================================
def walk_forward_validation(contests, n_windows=10, train_size=500,
                            test_size=50, n_games=10):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)...")
    results = []

    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end:
            continue

        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5:
            continue

        optimizer = HybridPortfolioOptimizerV13(train_data)
        portfolio, _ = optimizer.optimize_hybrid(
            n_games, n_candidates=50000, iterations=50)
        random_portfolio = optimizer.generate_pure_random_portfolio(n_games)
        coverage_portfolio = optimizer.generate_coverage_baseline(n_games)

        bt_strat = optimizer.backtest(portfolio, test_data)
        bt_rand = optimizer.backtest(random_portfolio, test_data)
        bt_cov = optimizer.backtest(coverage_portfolio, test_data)

        results.append({
            'window': w,
            'strat_lift': bt_strat['lift'],
            'rand_lift': bt_rand['lift'],
            'cov_lift': bt_cov['lift'],
            'diff_vs_rand': bt_strat['lift'] - bt_rand['lift'],
            'diff_vs_cov': bt_strat['lift'] - bt_cov['lift'],
        })
        print(f" Janela {w}: lift={bt_strat['lift']:.3f} "
              f"rand={bt_rand['lift']:.3f} cov={bt_cov['lift']:.3f}")

    if results:
        diffs_rand = [r['diff_vs_rand'] for r in results]
        diffs_cov = [r['diff_vs_cov'] for r in results]

        mean_diff_rand = np.mean(diffs_rand)
        std_diff_rand = np.std(diffs_rand)
        mean_diff_cov = np.mean(diffs_cov)

        cv_rand = abs(std_diff_rand / (mean_diff_rand + 1e-10))
        stability = 1.0 / (1.0 + cv_rand)

        print(f"\n📊 RESUMO:")
        print(f"   Média diff vs Aleatório: {mean_diff_rand:+.3f}")
        print(f"   Média diff vs Cobertura: {mean_diff_cov:+.3f}")
        print(f"   Stability score: {stability:.3f} (0=instável, 1=estável)")
        print(f"   Janelas + (vs Aleatório): "
              f"{sum(1 for d in diffs_rand if d > 0)}/{len(results)}")
        try:
            _, p_rand = wilcoxon(diffs_rand)
            print(f"   Wilcoxon p (vs Aleatório): {p_rand:.4f}")
        except Exception:
            pass
    return results


# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR DE CARTEIRA v13 - REPAIR MODERADO + SCORE RECALIBRADO")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    print("\n📊 REGIME ESTRUTURAL ALVO:")
    for name, (target, tol, weight) in STRUCTURAL_TARGETS.items():
        print(f"   {name}: alvo={target:.1f} ±{tol:.1f} (peso={weight:.1f})")
    print(f"\n📊 HARD CAPS + METAS:")
    print(f"   Pair coverage ≤ {MAX_PAIR_COVERAGE}")
    print(f"   Geo diversity ∈ [{MIN_GEO_DIVERSITY}, {MAX_GEO_DIVERSITY}]")
    print(f"   Entropia alvo: {TARGET_ENTROPY_MIN}-{TARGET_ENTROPY_MAX}")
    print(f"\n📊 SCORE EXPONENCIAL RECALIBRADO:")
    for k, w in EXPONENTIAL_WEIGHTS.items():
        print(f"   {k} pontos: peso={w:.0f}")
    print(f"\n📊 REPAIR MODERADO:")
    print(f"   Similaridade alvo: {REPAIR_SIMILARITY_MIN}-{REPAIR_SIMILARITY_MAX}")
    print(f"   Normalização MC: EMPÍRICA (percentis)")

    print("\nOpções:")
    print("1. Gerar carteira otimizada")
    print("2. Ensemble estrutural melhorado")
    print("3. Walk-forward validation (10 janelas)")
    print("4. TUDO")
    op = input("Escolha [4]: ").strip() or "4"

    if op in ("1", "4"):
        print(f"\n🔧 INICIALIZANDO...")
        t0 = time.time()
        optimizer = HybridPortfolioOptimizerV13(contests)
        print(f"   ✅ Inicializado em {time.time()-t0:.1f}s")
        print(f"   GMM: {optimizer.dist_model.n_components} componentes")

        portfolio, score = optimizer.optimize_hybrid(
            n_games=10, n_candidates=200000, iterations=100)

        if score < -100:
            print(f"\n❌ Não foi possível encontrar carteira dentro dos HARD CAPS.")
            print(f"   Tente relaxar os limites.")
        else:
            print(f"\n🏆 CARTEIRA (Score: {score:.3f})")
            last = contests[-1]['dezenas']
            for i, game in enumerate(portfolio, 1):
                p = sum(1 for d in game if d % 2 == 0)
                pr = sum(1 for d in game if d in PRIMES)
                m = sum(1 for d in game if d in MOLDURA)
                rep = len(set(game) & set(last))
                amp = max(game) - min(game)
                penalty = optimizer.extractor.compute_structural_penalty(game)
                print(f"   {i:2d}. {game} | P:{p} Pr:{pr} M:{m} Rep:{rep} Amp:{amp} Pen:{penalty:.1f}")

            all_d = set(d for g in portfolio for d in g)
            pair_cov = optimizer._pair_coverage(portfolio)
            structural_avg = optimizer._average_structural_score(portfolio)
            geo_div = optimizer._geometric_diversity(portfolio)
            entropy_val = optimizer._portfolio_entropy(portfolio)
            struct_overlap = optimizer._structural_overlap_penalty(portfolio)
            print(f"\n📊 Cobertura dezenas: {len(all_d)}/25")
            print(f"📊 Cobertura pares: {pair_cov:.3f} (cap: ≤{MAX_PAIR_COVERAGE})")
            print(f"📊 Score estrutural: {structural_avg:.3f}")
            print(f"📊 Diversidade geométrica: {geo_div:.3f} (meta: [{MIN_GEO_DIVERSITY}, {MAX_GEO_DIVERSITY}])")
            print(f"📊 Entropia: {entropy_val:.3f} (alvo: {TARGET_ENTROPY_MIN}-{TARGET_ENTROPY_MAX})")
            print(f"📊 Penalidade overlap estrutural: {struct_overlap:.3f}")

            weighted_sum = optimizer._monte_carlo_weighted_sum(portfolio, n_simulations=5000)
            print(f"📊 Score exponencial normalizado: {weighted_sum:.4f}")

            test_size = min(200, len(contests) // 3)
            if test_size > 10:
                test_data = contests[-test_size:]
                bt = optimizer.backtest(portfolio, test_data)
                print(f"\n🔬 BACKTEST ({bt['n_test']} concursos):")
                print(f"   Prob ≥1 acerto 11+: {bt['empirical']:.2%} "
                      f"(teórico: {bt['theoretical']:.2%})")
                print(f"   Lift: {bt['lift']:.2f}x")

    if op in ("2", "4"):
        optimizer = HybridPortfolioOptimizerV13(contests)
        consensus = optimizer.generate_ensemble_structural(
            n_strategies=3, n_games_per_strategy=5)
        print(f"\n🤝 CARTEIRA ENSEMBLE ESTRUTURAL ({len(consensus)} jogos):")
        last = contests[-1]['dezenas']
        for i, game in enumerate(consensus, 1):
            p = sum(1 for d in game if d % 2 == 0)
            pr = sum(1 for d in game if d in PRIMES)
            m = sum(1 for d in game if d in MOLDURA)
            rep = len(set(game) & set(last))
            penalty = optimizer.extractor.compute_structural_penalty(game)
            print(f"   {i:2d}. {game} | P:{p} Pr:{pr} M:{m} Rep:{rep} Pen:{penalty:.1f}")

    if op in ("3", "4"):
        walk_forward_validation(contests, n_windows=10, train_size=500,
                                test_size=50, n_games=10)

    print("\n✅ Concluído!")


if __name__ == "__main__":
    main()
```
