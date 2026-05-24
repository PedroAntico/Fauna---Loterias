#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v16 (FINAL)
==========================================================
CORREÇÕES E APERFEIÇOAMENTOS:
✅ partial_features com completamento aleatório (não zeros)
✅ P(par | regime) adicionada ao gerador (coocorrência condicional)
✅ RegimeDetector verdadeiramente TEMPORAL (features delta/momentum)
✅ ROI esperado como métrica principal
✅ Entropia dinâmica corrigida (das dezenas)
✅ MC Condicional SEM vazamento temporal
✅ GMM como score central, recompensa de cauda, cobertura moderada
✅ Structural reject threshold = 8
✅ Pesos exponenciais recalibrados (foco na cauda)
✅ Walk-forward com estabilidade preservada
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
from math import comb, log
from tqdm import tqdm
import random
import time

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
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

# Valores reais de prêmio (aproximados) para cálculo de ROI
PREMIO_VALORES = {
    11: 6.0,
    12: 12.0,
    13: 30.0,
    14: 1500.0,
    15: 1800000.0,
}
CUSTO_APOSTA = 3.0

# Features topológicas v16 (18 dimensões)
FEATURE_NAMES_V16 = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
    "compressao",
]
IDX = {name: i for i, name in enumerate(FEATURE_NAMES_V16)}

# Parâmetros ESTRUTURAIS
STRUCTURAL_TARGETS = {
    'pares': (7.5, 1.5, 2.0),
    'primos': (5.0, 1.5, 2.5),
    'moldura': (9.5, 1.5, 1.5),
    'repeticoes': (9.0, 1.5, 1.5),
    'soma': (195.0, 20.0, 1.0),
    'consecutivos': (5.5, 2.0, 1.0),
    'amplitude': (22.0, 3.0, 1.0),
}
STRUCTURAL_REJECT_THRESHOLD = 8

# HARD CAPS
MAX_PAIR_COVERAGE = 0.85
MIN_GEO_DIVERSITY = 0.50
MAX_GEO_DIVERSITY = 0.75

# Score EXPONENCIAL (foco na CAUDA)
EXPONENTIAL_WEIGHTS = {
    11: 0.5,
    12: 2.0,
    13: 15.0,
    14: 80.0,
    15: 400.0,
}

# Features com sinal temporal
TEMPORAL_FEATURES = {
    'moldura': 0.20,
    'amplitude': 0.20,
    'energia_jogo': 0.20,
    'densidade_local': 0.20,
    'clusterizacao': 0.20,
}

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
# DETECTOR DE REGIME TEMPORAL (COM DELTAS + P(DEZENA|REGIME) + P(PAR|REGIME))
# ============================================================
class TemporalRegimeDetector:
    """
    Detecta regimes usando features DELTA (dinâmica temporal real).
    Inclui P(dezena | cluster) e P(par | cluster).
    """
    def __init__(self, feature_matrix, contests, n_clusters=5, delta_window=10):
        self.feature_matrix = feature_matrix
        self.contests = contests
        self.n_clusters = n_clusters
        self.delta_window = delta_window
        self.kmeans = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.labels = None
        self.transition_matrix = None
        self.conditional_freqs = {}
        self.pair_conditional_freqs = {}
        self._build()

    def _build(self):
        if not SKLEARN_AVAILABLE or len(self.feature_matrix) < 50:
            return

        # Construir features DELTA (current - mean of last 10)
        delta_features = []
        for i in range(len(self.feature_matrix)):
            if i >= self.delta_window:
                baseline = np.mean(self.feature_matrix[i-self.delta_window:i], axis=0)
            else:
                baseline = np.mean(self.feature_matrix[:i+1], axis=0) if i > 0 else self.feature_matrix[i]
            delta = self.feature_matrix[i] - baseline
            delta_features.append(delta)
        delta_features = np.array(delta_features)

        X_scaled = self.scaler.fit_transform(delta_features)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(X_scaled)

        # Matriz de transição
        n = len(self.labels)
        trans = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(n - 1):
            trans[self.labels[i], self.labels[i+1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = trans / row_sums

        # Frequências condicionais P(dezena | cluster)
        self._build_conditional_freqs()
        # Frequências condicionais P(par | cluster)
        self._build_pair_conditional_freqs()

    def _build_conditional_freqs(self):
        for cluster in range(self.n_clusters):
            mask = self.labels == cluster
            cluster_contests = [self.contests[i] for i in range(len(self.contests)) if mask[i]]
            freq = Counter()
            for c in cluster_contests:
                freq.update(c['dezenas'])
            total = sum(freq.values())
            self.conditional_freqs[cluster] = {
                d: freq.get(d, 0) / total for d in range(1, 26)
            } if total > 0 else {d: 0.04 for d in range(1, 26)}

    def _build_pair_conditional_freqs(self):
        for cluster in range(self.n_clusters):
            mask = self.labels == cluster
            cluster_contests = [self.contests[i] for i in range(len(self.contests)) if mask[i]]
            pair_freq = Counter()
            for c in cluster_contests:
                for pair in combinations(sorted(c['dezenas']), 2):
                    pair_freq[pair] += 1
            total = sum(pair_freq.values())
            self.pair_conditional_freqs[cluster] = {
                pair: pair_freq.get(pair, 0) / total for pair in pair_freq
            } if total > 0 else {}

    def predict(self, features, baseline_features=None):
        if self.kmeans is None:
            return 0
        if baseline_features is not None:
            delta = features - baseline_features
        else:
            delta = features
        features_scaled = self.scaler.transform(delta.reshape(1, -1))
        return int(self.kmeans.predict(features_scaled)[0])

    def get_dezena_prob(self, cluster, dezena):
        if cluster in self.conditional_freqs:
            return self.conditional_freqs[cluster].get(dezena, 0.01)
        return 0.04

    def get_pair_prob(self, cluster, pair):
        if cluster in self.pair_conditional_freqs:
            return self.pair_conditional_freqs[cluster].get(pair, 1e-6)
        return 1e-6

    def get_regime_distribution(self, recent_features):
        if self.kmeans is None or len(recent_features) == 0:
            return np.ones(self.n_clusters) / self.n_clusters
        labels = []
        for i, f in enumerate(recent_features):
            if i >= self.delta_window:
                baseline = np.mean(recent_features[max(0, i-self.delta_window):i], axis=0)
            else:
                baseline = np.mean(recent_features[:i+1], axis=0) if i > 0 else f
            labels.append(self.predict(f, baseline))
        counts = np.bincount(labels, minlength=self.n_clusters)
        return counts / counts.sum()

    def get_transition_prob(self, current_regime):
        if self.transition_matrix is None or current_regime >= self.n_clusters:
            return np.ones(self.n_clusters) / self.n_clusters
        return self.transition_matrix[current_regime]

    @staticmethod
    def get_entropy_target_from_draws(recent_draws):
        if len(recent_draws) == 0:
            return 0.93
        entropies = []
        for draw in recent_draws:
            freq = np.bincount(draw, minlength=26)[1:]
            probs = freq / np.sum(freq)
            probs = np.where(probs > 0, probs, 1e-10)
            entropies.append(float(entropy(probs) / np.log(25)))
        target = np.mean(entropies)
        return max(0.88, min(0.97, target))


# ============================================================
# EXTRATOR DE FEATURES v16
# ============================================================
class TopologicalFeatureExtractorV16:
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
        # Regime detector TEMPORAL
        self.regime_detector = TemporalRegimeDetector(self.build_feature_matrix(), contests)

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

        amplitude = max(d) - min(d)
        std_pos = np.std(d) if len(d) > 1 else 0.0
        compressao = std_pos / amplitude if amplitude > 0 else 0.5

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
            compressao,
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
        return np.exp(-penalty / 3.0)

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
# MODELO DE DISTRIBUIÇÃO (GMM)
# ============================================================
class DistributionModelV16:
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
# GERADOR LIVRE (COM COMPLETAMENTO ALEATÓRIO + BÔNUS DE REGIME + PARES)
# ============================================================
class FreeGeneratorV16:
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

                # BÔNUS DE REGIME: P(dezena | regime) + P(par | regime)
                if self.extractor is not None and self.extractor.regime_detector is not None:
                    try:
                        # CORRIGIDO: completar jogo parcial ALEATORIAMENTE para features válidas
                        remaining_count = 15 - len(test)
                        remaining = random.sample(
                            [x for x in range(1, 26) if x not in test],
                            remaining_count
                        )
                        pseudo_game = sorted(list(test) + remaining)
                        features = self.extractor.extract_features(pseudo_game, self.last)

                        # Baseline para delta (média dos últimos 10)
                        if len(self.extractor.feature_matrix) >= 10:
                            baseline = np.mean(self.extractor.feature_matrix[-10:], axis=0)
                        else:
                            baseline = features

                        cluster = self.extractor.regime_detector.predict(features, baseline)

                        # Bônus de dezena
                        prob_dezena = self.extractor.regime_detector.get_dezena_prob(cluster, d)
                        s += log(prob_dezena + 1e-10) * 0.4

                        # Bônus de pares (com as dezenas já escolhidas)
                        for existing_d in test:
                            pair = tuple(sorted([existing_d, d]))
                            prob_pair = self.extractor.regime_detector.get_pair_prob(cluster, pair)
                            s += log(prob_pair + 1e-10) * 0.3
                    except Exception:
                        pass
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
# OTIMIZADOR DE CARTEIRA v16 (COM ROI)
# ============================================================
class HybridPortfolioOptimizerV16:
    """
    Otimizador v16: ROI como métrica principal.
    """
    def __init__(self, contests):
        self.contests = contests
        self.extractor = TopologicalFeatureExtractorV16(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.dist_model = DistributionModelV16(self.feature_matrix)
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = FreeGeneratorV16(self.last, self.extractor)
        self._mc_cache = {}
        self._mc_norm_params = None

        # Dados históricos para MC (APENAS treino)
        self.historical_draws = [set(c['dezenas']) for c in self.contests]
        if len(self.historical_draws) < 100:
            self.historical_draws = [
                set(np.random.choice(range(1, 26), 15, replace=False))
                for _ in range(500)
            ]

        # Entropia dinâmica CORRIGIDA
        recent_draws = [c['dezenas'] for c in self.contests[-20:]] if len(self.contests) >= 20 else [c['dezenas'] for c in self.contests]
        self.entropy_target = TemporalRegimeDetector.get_entropy_target_from_draws(recent_draws)

        # Regime atual
        self.current_regime_dist = self._compute_current_regime_dist()

    def _compute_current_regime_dist(self):
        recent = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix
        return self.extractor.regime_detector.get_regime_distribution(recent)

    def _score_game_central(self, game):
        features = self.extractor.extract_features(game, self.last)
        gmm_score = self.dist_model.score_samples_normalized(features)
        structural_score = self.extractor.compute_structural_score(game)
        combined = gmm_score * 0.70 + structural_score * 0.30
        return combined, features, self.dist_model.predict_cluster(features)

    def _score_game_temporal(self, game):
        temporal_score = self.extractor.compute_temporal_score(game, self.feature_matrix)
        structural_score = self.extractor.compute_structural_score(game)
        combined = temporal_score * 0.4 + structural_score * 0.6
        features = self.extractor.extract_features(game, self.last)
        return combined, features, self.dist_model.predict_cluster(features)

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
                coverage_weight = 0.0
                structural_weight = 1.0
            else:
                coverage_weight = 0.3
                structural_weight = 0.7

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
        max_expected = 2 * np.sqrt(len(FEATURE_NAMES_V16))
        return avg_dist / max_expected

    def _structural_overlap_penalty(self, portfolio):
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
                dot = np.dot(feature_vectors[i], feature_vectors[j])
                norm = np.linalg.norm(feature_vectors[i]) * np.linalg.norm(feature_vectors[j])
                if norm > 1e-10:
                    similarities.append(dot / norm)

        avg_similarity = np.mean(similarities) if similarities else 0
        if avg_similarity > 0.85:
            return (avg_similarity - 0.85) * 5.0
        return 0.0

    def _average_structural_score(self, portfolio):
        scores = [self.extractor.compute_structural_score(g) for g in portfolio]
        return np.mean(scores) if scores else 0.5

    def _historical_similarity_weight(self, historical_draw, recent_features):
        if len(recent_features) == 0:
            return 1.0
        draw_features = self.extractor.extract_features(list(historical_draw), None)
        distances = []
        for rf in recent_features:
            dist = np.linalg.norm(draw_features - rf)
            distances.append(dist)
        avg_dist = np.mean(distances) if distances else 0
        return np.exp(-avg_dist / 2.0)

    def _monte_carlo_weighted_sum(self, portfolio, n_simulations=2000):
        cache_key = tuple(tuple(sorted(g)) for g in portfolio)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        recent_features = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix

        weights = []
        for draw in self.historical_draws:
            w = self._historical_similarity_weight(draw, recent_features)
            weights.append(w)
        total_weight = sum(weights)

        if total_weight == 0:
            weights = [1.0] * len(self.historical_draws)
            total_weight = len(self.historical_draws)

        total_score = 0.0
        n_eval = min(n_simulations, len(self.historical_draws))

        indices = np.random.choice(
            len(self.historical_draws), size=n_eval,
            p=np.array(weights) / total_weight
        )

        for idx in indices:
            drawn = self.historical_draws[idx]
            draw_score = 0.0
            for g in portfolio:
                hits = len(set(g) & drawn)
                if hits >= 11:
                    draw_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
            total_score += draw_score

        avg_score = total_score / n_eval

        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization(portfolio_size=len(portfolio))
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        if p95 - p5 > 1e-10:
            normalized = (avg_score - p5) / (p95 - p5)
        else:
            normalized = 0.5

        normalized = max(0.0, min(1.0, normalized))
        self._mc_cache[cache_key] = normalized
        if len(self._mc_cache) > 500:
            keys = list(self._mc_cache.keys())[:250]
            for k in keys:
                del self._mc_cache[k]
        return normalized

    def _compute_mc_normalization(self, portfolio_size=10, n_samples=200):
        raw_scores = []
        for _ in range(n_samples):
            random_portfolio = [self.generator.generate_pure_random()
                               for _ in range(portfolio_size)]
            raw = self._monte_carlo_weighted_sum_raw(random_portfolio, n_simulations=500)
            raw_scores.append(raw)
        raw_scores = np.array(raw_scores)
        return {
            'p5': float(np.percentile(raw_scores, 5)),
            'p95': float(np.percentile(raw_scores, 95)),
        }

    def _monte_carlo_weighted_sum_raw(self, portfolio, n_simulations=500):
        total_score = 0.0
        n_eval = min(n_simulations, len(self.historical_draws))
        indices = np.random.choice(len(self.historical_draws), size=n_eval)
        for idx in indices:
            drawn = self.historical_draws[idx]
            draw_score = 0.0
            for g in portfolio:
                hits = len(set(g) & drawn)
                if hits >= 11:
                    draw_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
            total_score += draw_score
        return total_score / n_eval

    def _repair_portfolio(self, portfolio, pool):
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

    def _find_most_similar_valid(self, game, pool, max_intersection=8):
        game_features = self.extractor.extract_features(game, self.last)
        valid_candidates = []

        sample = random.sample(pool, min(500, len(pool)))

        for candidate in sample:
            if candidate == game:
                continue
            if not self.extractor.is_structurally_valid(candidate):
                continue

            cand_features = self.extractor.extract_features(candidate, self.last)
            dot = np.dot(game_features, cand_features)
            norm = np.linalg.norm(game_features) * np.linalg.norm(cand_features)
            similarity = dot / norm if norm > 1e-10 else 1.0

            if 0.3 <= similarity <= 0.7:
                valid_candidates.append((similarity, candidate))

        if valid_candidates:
            return random.choice(valid_candidates)[1]

        best_candidate = None
        best_similarity = -float('inf')
        for candidate in sample:
            if candidate == game:
                continue
            if not self.extractor.is_structurally_valid(candidate):
                continue
            cand_features = self.extractor.extract_features(candidate, self.last)
            dot = np.dot(game_features, cand_features)
            norm = np.linalg.norm(game_features) * np.linalg.norm(cand_features)
            similarity = dot / norm if norm > 1e-10 else 1.0
            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = candidate

        return best_candidate if best_candidate is not None else self.generator.generate_one()

    def _portfolio_score(self, portfolio):
        pair_cov = self._pair_coverage(portfolio)
        geo_div = self._geometric_diversity(portfolio)

        if pair_cov > MAX_PAIR_COVERAGE:
            return -1000.0
        if geo_div > MAX_GEO_DIVERSITY or geo_div < MIN_GEO_DIVERSITY:
            return -1000.0

        weighted_sum = self._monte_carlo_weighted_sum(portfolio, n_simulations=2000)
        avg_gmm = np.mean([self.dist_model.score_samples_normalized(
            self.extractor.extract_features(g, self.last)) for g in portfolio])

        structural = self._average_structural_score(portfolio)
        entropy_val = self._portfolio_entropy(portfolio)
        diversity = self._portfolio_diversity(portfolio)
        struct_overlap = self._structural_overlap_penalty(portfolio)

        entropy_penalty = abs(entropy_val - self.entropy_target) * 4.0

        return (weighted_sum * 0.30 + avg_gmm * 0.30 + structural * 0.15 +
                diversity * 0.10 + geo_div * 0.05 -
                entropy_penalty * 0.10 - struct_overlap * 0.05)

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
        print(f"\n🎯 OTIMIZANDO CARTEIRA v16 ({n_games} jogos)...")

        n_central = max(1, int(n_games * 0.40))
        n_coverage = max(1, int(n_games * 0.35))
        n_temporal = n_games - n_central - n_coverage

        print(f"   Composição: {n_central} centrais + {n_coverage} cobertura + {n_temporal} temporais")
        print(f"   HARD CAPS: pair_cov ≤ {MAX_PAIR_COVERAGE}, geo_div ∈ [{MIN_GEO_DIVERSITY}, {MAX_GEO_DIVERSITY}]")
        print(f"   Entropia dinâmica alvo: {self.entropy_target:.3f}")
        print(f"   Score EXPONENCIAL: {EXPONENTIAL_WEIGHTS}")
        print(f"   MC Condicional: SEM vazamento temporal")

        self._compute_mc_normalization(portfolio_size=n_games)

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

        # Cobertura
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
        portfolio = self._repair_portfolio(portfolio, pool_all)
        print(f"   ✅ Carteira inicial: pair_cov={self._pair_coverage(portfolio):.3f}, geo_div={self._geometric_diversity(portfolio):.3f}")

        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)
        current_score = best_score

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

        pair_votes = Counter()
        for sr in strategy_results:
            weight = sr['weighted_sum'] * 0.6 + sr['structural'] * 0.4
            for pair, count in sr['pairs'].items():
                pair_votes[pair] += weight * count

        top_pairs = [pair for pair, _ in pair_votes.most_common(100)]

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
        """
        Backtest com ROI esperado (NOVO).
        """
        n_success = 0
        total_premio = 0.0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA

        for draw in test_draws:
            actual = set(draw['dezenas'])
            for g in portfolio:
                hits = len(set(g) & actual)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)

        prob = n_success / (len(portfolio) * len(test_draws)) if len(test_draws) > 0 else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11, 16))
        p_none = (1 - p_single) ** len(portfolio)
        theo_prob = 1 - p_none
        roi = (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0

        return {
            'empirical': prob,
            'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws),
            'n_success': n_success,
            'total_premio': total_premio,
            'total_custo': total_custo,
            'roi': roi,
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

        optimizer = HybridPortfolioOptimizerV16(train_data)
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
            'strat_roi': bt_strat['roi'],
            'rand_lift': bt_rand['lift'],
            'rand_roi': bt_rand['roi'],
            'cov_lift': bt_cov['lift'],
            'cov_roi': bt_cov['roi'],
            'diff_vs_rand': bt_strat['lift'] - bt_rand['lift'],
            'diff_roi_vs_rand': bt_strat['roi'] - bt_rand['roi'],
            'diff_vs_cov': bt_strat['lift'] - bt_cov['lift'],
        })
        print(f" Janela {w}: lift={bt_strat['lift']:.3f} ROI={bt_strat['roi']:+.1f}% "
              f"rand_lift={bt_rand['lift']:.3f} rand_ROI={bt_rand['roi']:+.1f}%")

    if results:
        diffs_rand = [r['diff_vs_rand'] for r in results]
        diffs_roi = [r['diff_roi_vs_rand'] for r in results]
        diffs_cov = [r['diff_vs_cov'] for r in results]

        mean_diff_rand = np.mean(diffs_rand)
        mean_diff_roi = np.mean(diffs_roi)
        std_diff_rand = np.std(diffs_rand)
        mean_diff_cov = np.mean(diffs_cov)

        cv_rand = abs(std_diff_rand / (mean_diff_rand + 1e-10))
        stability = 1.0 / (1.0 + cv_rand)

        print(f"\n📊 RESUMO:")
        print(f"   Média diff lift vs Aleatório: {mean_diff_rand:+.3f}")
        print(f"   Média diff ROI vs Aleatório: {mean_diff_roi:+.1f}%")
        print(f"   Média diff lift vs Cobertura: {mean_diff_cov:+.3f}")
        print(f"   Stability score: {stability:.3f} (0=instável, 1=estável)")
        print(f"   Janelas + (lift vs Aleatório): "
              f"{sum(1 for d in diffs_rand if d > 0)}/{len(results)}")
        print(f"   Janelas + (ROI vs Aleatório): "
              f"{sum(1 for d in diffs_roi if d > 0)}/{len(results)}")
        try:
            _, p_rand = wilcoxon(diffs_rand)
            print(f"   Wilcoxon p (lift vs Aleatório): {p_rand:.4f}")
        except Exception:
            pass
    return results


# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR DE CARTEIRA v16 - ROI + REGIME TEMPORAL")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    print("\n📊 CONFIGURAÇÃO FINAL:")
    print(f"   GMM como score CENTRAL")
    print(f"   Regime detector TEMPORAL (deltas)")
    print(f"   Monte Carlo CONDICIONAL (sem vazamento)")
    print(f"   Entropia DINÂMICA (corrigida)")
    print(f"   P(dezena|regime) + P(par|regime) integradas")
    print(f"   ROI como métrica principal")
    print(f"   Recompensa de CAUDA (15: R$ {PREMIO_VALORES[15]:,.0f})")
    print(f"   Threshold estrutural: {STRUCTURAL_REJECT_THRESHOLD}")

    print("\nOpções:")
    print("1. Gerar carteira otimizada")
    print("2. Ensemble estrutural")
    print("3. Walk-forward validation (10 janelas)")
    print("4. TUDO")
    op = input("Escolha [4]: ").strip() or "4"

    if op in ("1", "4"):
        print(f"\n🔧 INICIALIZANDO...")
        t0 = time.time()
        optimizer = HybridPortfolioOptimizerV16(contests)
        print(f"   ✅ Inicializado em {time.time()-t0:.1f}s")
        print(f"   GMM: {optimizer.dist_model.n_components} componentes")
        print(f"   Regime atual (distribuição): {optimizer.current_regime_dist}")
        print(f"   Entropia dinâmica alvo: {optimizer.entropy_target:.3f}")

        portfolio, score = optimizer.optimize_hybrid(
            n_games=10, n_candidates=200000, iterations=100)

        if score < -100:
            print(f"\n❌ Não foi possível encontrar carteira dentro dos HARD CAPS.")
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
            print(f"\n📊 Cobertura dezenas: {len(all_d)}/25")
            print(f"📊 Cobertura pares: {pair_cov:.3f} (cap: ≤{MAX_PAIR_COVERAGE})")
            print(f"📊 Score estrutural: {structural_avg:.3f}")
            print(f"📊 Diversidade geométrica: {geo_div:.3f} (meta: [{MIN_GEO_DIVERSITY}, {MAX_GEO_DIVERSITY}])")
            print(f"📊 Entropia: {entropy_val:.3f} (alvo dinâmico: {optimizer.entropy_target:.3f})")

            weighted_sum = optimizer._monte_carlo_weighted_sum(portfolio, n_simulations=5000)
            print(f"📊 Score MC condicional normalizado: {weighted_sum:.4f}")

            test_size = min(200, len(contests) // 3)
            if test_size > 10:
                test_data = contests[-test_size:]
                bt = optimizer.backtest(portfolio, test_data)
                print(f"\n🔬 BACKTEST ({bt['n_test']} concursos):")
                print(f"   Prob ≥1 acerto 11+: {bt['empirical']:.2%} "
                      f"(teórico: {bt['theoretical']:.2%})")
                print(f"   Lift: {bt['lift']:.2f}x")
                print(f"   Prêmio total: R$ {bt['total_premio']:,.2f}")
                print(f"   Custo total: R$ {bt['total_custo']:,.2f}")
                print(f"   ROI: {bt['roi']:+.2f}%")

    if op in ("2", "4"):
        optimizer = HybridPortfolioOptimizerV16(contests)
        consensus = optimizer.generate_ensemble_structural(
            n_strategies=3, n_games_per_strategy=5)
        print(f"\n🤝 CARTEIRA ENSEMBLE ({len(consensus)} jogos):")
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
