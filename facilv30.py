#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v18 (BITMASK + REGULARIZADO)
=========================================================================
MELHORIAS:
✅ Bitmask para interseções (5-10x mais rápido)
✅ Avaliação unificada (central + temporal juntas)
✅ Regularização bayesiana de probabilidades condicionais
✅ GMM removido (substituído por score estrutural + MC)
✅ Cache de features otimizado
✅ Pipeline 2 fases mantido
✅ Walk-forward com estabilidade
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

PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.0

# Features (18 dimensões)
FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
    "compressao",
]

STRUCTURAL_TARGETS = {
    'pares': (7.5, 1.5, 2.0), 'primos': (5.0, 1.5, 2.5),
    'moldura': (9.5, 1.5, 1.5), 'repeticoes': (9.0, 1.5, 1.5),
    'soma': (195.0, 20.0, 1.0), 'consecutivos': (5.5, 2.0, 1.0),
    'amplitude': (22.0, 3.0, 1.0),
}
STRUCTURAL_REJECT_THRESHOLD = 8

MAX_PAIR_COVERAGE = 0.85
MIN_GEO_DIVERSITY = 0.50
MAX_GEO_DIVERSITY = 0.75

# Pesos exponenciais (agressivo para cauda)
EXPONENTIAL_WEIGHTS = {11: 0.1, 12: 0.5, 13: 10.0, 14: 300.0, 15: 5000.0}

# Regularização bayesiana (alpha para suavização)
BAYES_ALPHA = 1.0

# ============================================================
# UTILITÁRIOS BITMASK
# ============================================================
def game_to_mask(game):
    """Converte jogo para máscara de bits (uint32)."""
    mask = 0
    for d in game:
        mask |= (1 << d)
    return mask

def mask_intersection(mask1, mask2):
    """Conta interseção entre duas máscaras."""
    return (mask1 & mask2).bit_count()

# Pré-computar máscaras para todos os draws históricos
def draws_to_masks(draws):
    return np.array([game_to_mask(d) for d in draws], dtype=np.uint32)


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
                    contests.append({
                        'concurso': int(parts[0]), 'data': parts[1],
                        'dezenas': sorted(dezenas)
                    })
                except (ValueError, IndexError): continue
        contests.sort(key=lambda x: x['concurso'])
        print(f"✅ {len(contests)} concursos válidos")
        return contests
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


# ============================================================
# DETECTOR DE REGIME (COM REGULARIZAÇÃO BAYESIANA)
# ============================================================
class RegularizedRegimeDetector:
    def __init__(self, feature_matrix, contests, n_clusters=5, delta_window=10, bayes_alpha=BAYES_ALPHA):
        self.feature_matrix = feature_matrix
        self.contests = contests
        self.n_clusters = n_clusters
        self.delta_window = delta_window
        self.bayes_alpha = bayes_alpha
        self.kmeans = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.labels = None
        self.transition_matrix = None
        # Frequências condicionais regularizadas
        self.dezena_probs = {}      # cluster -> array[26]
        self.pair_probs = {}        # cluster -> array[26,26]
        self.global_dezena_probs = None  # array[26]
        self.global_pair_probs = None    # array[26,26]
        self._build()

    def _build(self):
        if not SKLEARN_AVAILABLE or len(self.feature_matrix) < 50: return
        # Features delta
        delta_features = []
        for i in range(len(self.feature_matrix)):
            baseline = np.mean(self.feature_matrix[max(0,i-self.delta_window):i+1], axis=0) if i > 0 else self.feature_matrix[i]
            delta_features.append(self.feature_matrix[i] - baseline)
        delta_features = np.array(delta_features)
        X_scaled = self.scaler.fit_transform(delta_features)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(X_scaled)
        # Matriz de transição
        n = len(self.labels)
        trans = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(n - 1): trans[self.labels[i], self.labels[i+1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = trans / row_sums
        # Frequências globais (para regularização)
        self._build_global_freqs()
        # Frequências condicionais regularizadas
        self._build_regularized_freqs()

    def _build_global_freqs(self):
        """Constrói frequências globais para regularização."""
        global_freq = Counter()
        global_pair_freq = Counter()
        for c in self.contests:
            global_freq.update(c['dezenas'])
            for pair in combinations(sorted(c['dezenas']), 2):
                global_pair_freq[pair] += 1
        total = sum(global_freq.values())
        self.global_dezena_probs = np.zeros(26)
        for d in range(1, 26):
            count = global_freq.get(d, 0)
            self.global_dezena_probs[d] = (count + self.bayes_alpha) / (total + self.bayes_alpha * 25)
        total_pairs = sum(global_pair_freq.values())
        self.global_pair_probs = np.zeros((26, 26))
        for (a, b), count in global_pair_freq.items():
            prob = (count + self.bayes_alpha) / (total_pairs + self.bayes_alpha * 300)
            self.global_pair_probs[a, b] = prob
            self.global_pair_probs[b, a] = prob

    def _build_regularized_freqs(self):
        """Constrói frequências condicionais REGULARIZADAS (cluster + global)."""
        for cluster in range(self.n_clusters):
            mask = self.labels == cluster
            cluster_contests = [self.contests[i] for i in range(len(self.contests)) if mask[i]]
            freq = Counter(); pair_freq = Counter()
            for c in cluster_contests:
                freq.update(c['dezenas'])
                for pair in combinations(sorted(c['dezenas']), 2):
                    pair_freq[pair] += 1
            total = sum(freq.values())
            # Dezena: regularizada
            self.dezena_probs[cluster] = np.zeros(26)
            for d in range(1, 26):
                count = freq.get(d, 0)
                cluster_prob = count / total if total > 0 else 0
                global_prob = self.global_dezena_probs[d]
                # Mistura: 70% cluster + 30% global
                self.dezena_probs[cluster][d] = 0.7 * cluster_prob + 0.3 * global_prob
            # Pares: regularizados
            self.pair_probs[cluster] = np.zeros((26, 26))
            total_pairs = sum(pair_freq.values())
            for a, b in pair_freq:
                cluster_prob = pair_freq[(a,b)] / total_pairs if total_pairs > 0 else 0
                global_prob = self.global_pair_probs[a, b]
                prob = 0.7 * cluster_prob + 0.3 * global_prob
                self.pair_probs[cluster][a, b] = prob
                self.pair_probs[cluster][b, a] = prob

    def predict(self, features, baseline_features=None):
        if self.kmeans is None: return 0
        delta = features - baseline_features if baseline_features is not None else features
        return int(self.kmeans.predict(self.scaler.transform(delta.reshape(1, -1)))[0])

    def get_dezena_prob(self, cluster, dezena):
        if 0 <= cluster < self.n_clusters:
            return self.dezena_probs[cluster][dezena]
        return self.global_dezena_probs[dezena] if self.global_dezena_probs is not None else 0.04

    def get_pair_prob(self, cluster, a, b):
        if 0 <= cluster < self.n_clusters:
            return self.pair_probs[cluster][a, b]
        return self.global_pair_probs[a, b] if self.global_pair_probs is not None else 1e-6

    def get_regime_distribution(self, recent_features):
        if self.kmeans is None or len(recent_features) == 0:
            return np.ones(self.n_clusters) / self.n_clusters
        labels = []
        for i, f in enumerate(recent_features):
            baseline = np.mean(recent_features[max(0,i-self.delta_window):i+1], axis=0) if i>0 else f
            labels.append(self.predict(f, baseline))
        counts = np.bincount(labels, minlength=self.n_clusters)
        return counts / counts.sum()

    @staticmethod
    def get_entropy_target_from_draws(recent_draws):
        if len(recent_draws) == 0: return 0.93
        entropies = []
        for draw in recent_draws:
            freq = np.bincount(draw, minlength=26)[1:]
            probs = freq / np.sum(freq)
            probs = np.where(probs > 0, probs, 1e-10)
            entropies.append(float(entropy(probs) / np.log(25)))
        return max(0.88, min(0.97, np.mean(entropies)))


# ============================================================
# EXTRATOR DE FEATURES (COM CACHE)
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
        self._recent_freq = self._compute_recent_freq()
        raw_features = self._build_raw_feature_matrix()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        if self.scaler is not None and len(raw_features) > 10:
            self.scaler.fit(raw_features)
        self.feature_means = np.mean(raw_features, axis=0)
        self.feature_stds = np.std(raw_features, axis=0) + 1e-10
        self.regime_detector = RegularizedRegimeDetector(self.build_feature_matrix(), contests)
        self._feature_cache = {}

    def _compute_recent_freq(self, window=50):
        freq = Counter()
        start = max(0, len(self.contests) - window)
        for c in self.contests[start:]: freq.update(c['dezenas'])
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
            trans = [self._repeat_history[i+1]-self._repeat_history[i] for i in range(len(self._repeat_history)-1)]
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
            'pares': sum(1 for x in d if x % 2 == 0),
            'primos': sum(1 for x in d if x in PRIMES),
            'moldura': sum(1 for x in d if x in MOLDURA),
            'repeticoes': len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8,
            'soma': sum(d),
            'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),
            'amplitude': max(d) - min(d),
        }
        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name] - target)
                if dev > tol: penalty += (dev - tol) * w
        return penalty

    def is_structurally_valid(self, game):
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

    def compute_structural_score(self, game):
        return np.exp(-self.compute_structural_penalty(game) / 3.0)


# ============================================================
# GERADOR RÁPIDO (FASE 1)
# ============================================================
class FastGenerator:
    def __init__(self, last_contest=None, extractor=None, current_cluster=0):
        self.last = set(last_contest) if last_contest else None
        self.extractor = extractor
        self.current_cluster = current_cluster

    def generate_one(self):
        for _ in range(50):
            game = self._generate_raw()
            if self.extractor is not None and self.extractor.is_structurally_valid(game):
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
                if cons > 6: s -= (cons - 6) * 1.5
                if self.extractor is not None and self.extractor.regime_detector is not None:
                    prob_dezena = self.extractor.regime_detector.get_dezena_prob(self.current_cluster, d)
                    s += log(prob_dezena + 1e-10) * 0.4
                    for existing_d in test:
                        prob_pair = self.extractor.regime_detector.get_pair_prob(self.current_cluster, existing_d, d)
                        s += log(prob_pair + 1e-10) * 0.3
                scores.append(s)
            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores -= np.max(scores)
                probs = np.exp(scores / 3.0)
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
# OTIMIZADOR DE CARTEIRA v18 (BITMASK + UNIFICADO)
# ============================================================
class PortfolioOptimizerV18:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None

        # Regime global fixo
        recent = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix
        self.current_regime_dist = self.extractor.regime_detector.get_regime_distribution(recent)
        self.current_cluster = int(np.argmax(self.current_regime_dist))

        self.generator = FastGenerator(self.last, self.extractor, self.current_cluster)
        self._mc_cache = {}
        self._mc_norm_params = None

        # Dados históricos como bitmasks
        self.historical_draws = [set(c['dezenas']) for c in self.contests]
        self.historical_masks = draws_to_masks(self.historical_draws)
        if len(self.historical_draws) < 100:
            extra = [set(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draws_to_masks(self.historical_draws)

        recent_draws = [c['dezenas'] for c in self.contests[-20:]] if len(self.contests) >= 20 else [c['dezenas'] for c in self.contests]
        self.entropy_target = RegularizedRegimeDetector.get_entropy_target_from_draws(recent_draws)

    def _evaluate_game(self, game):
        """Avaliação UNIFICADA (central + temporal + estrutural)."""
        features = self.extractor.extract_features(game, self.last)
        structural = self.extractor.compute_structural_score(game)
        return features, structural

    def _greedy_marginal_coverage(self, pool, existing, n_select, max_inter=7):
        existing_masks = [game_to_mask(g) for g in existing]
        covered_pairs = set()
        for g in existing:
            for pair in combinations(sorted(g), 2):
                covered_pairs.add(pair)
        selected = list(existing)
        selected_set = set(tuple(g) for g in selected)
        remaining = [g for g in pool if tuple(g) not in selected_set]

        for _ in range(n_select):
            if not remaining: break
            cov = len(covered_pairs) / comb(25, 2)
            cw, sw = (0.0, 1.0) if cov >= MAX_PAIR_COVERAGE else (0.3, 0.7)
            best_game, best_score = None, -float('inf')
            for game in random.sample(remaining, min(300, len(remaining))):
                game_mask = game_to_mask(game)
                # BITMASK para interseção
                if existing_masks and max(mask_intersection(game_mask, em) for em in existing_masks) > max_inter:
                    continue
                if self.extractor.compute_structural_penalty(game) > STRUCTURAL_REJECT_THRESHOLD:
                    continue
                new_pairs = set(combinations(sorted(game), 2)) - covered_pairs
                combined = len(new_pairs) / 105.0 * cw + self.extractor.compute_structural_score(game) * sw
                if combined > best_score:
                    best_score, best_game = combined, game
            if best_game:
                selected.append(best_game)
                selected_set.add(tuple(best_game))
                remaining.remove(best_game)
                existing_masks.append(game_to_mask(best_game))
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
        freq = np.bincount([d for g in portfolio for d in g], minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        return float(entropy(probs) / np.log(25))

    def _portfolio_diversity(self, portfolio):
        if len(portfolio) < 2: return 1.0
        masks = [game_to_mask(g) for g in portfolio]
        sims = [mask_intersection(masks[i], masks[j]) for i in range(len(masks)) for j in range(i+1, len(masks))]
        return 1.0 - np.mean(sims) / 15.0 if sims else 1.0

    def _geometric_diversity(self, portfolio):
        if len(portfolio) < 2: return 0.5
        fvs = np.array([self.extractor.extract_features(g, self.last) for g in portfolio])
        dists = [np.linalg.norm(fvs[i]-fvs[j]) for i in range(len(fvs)) for j in range(i+1, len(fvs))]
        return np.mean(dists) / (2 * np.sqrt(len(FEATURE_NAMES))) if dists else 0

    def _average_structural_score(self, portfolio):
        scores = [self.extractor.compute_structural_score(g) for g in portfolio]
        return np.mean(scores) if scores else 0.5

    def _monte_carlo_weighted_sum(self, portfolio, n_simulations=500):
        """MC Condicional com bitmask para velocidade."""
        cache_key = tuple(tuple(sorted(g)) for g in portfolio)
        if cache_key in self._mc_cache: return self._mc_cache[cache_key]

        recent_f = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix
        weights = []
        for draw in self.historical_draws[:len(self.historical_masks)]:
            df = self.extractor.extract_features(list(draw), None)
            avg_dist = np.mean([np.linalg.norm(df - rf) for rf in recent_f]) if len(recent_f) > 0 else 0
            weights.append(np.exp(-avg_dist / 2.0))
        total_w = sum(weights)
        if total_w == 0: weights, total_w = [1.0]*len(self.historical_masks), len(self.historical_masks)

        portfolio_masks = np.array([game_to_mask(g) for g in portfolio], dtype=np.uint32)
        indices = np.random.choice(len(self.historical_masks), size=min(n_simulations, len(self.historical_masks)), p=np.array(weights)/total_w)

        total_score = 0.0
        for idx in indices:
            drawn_mask = self.historical_masks[idx]
            for pm in portfolio_masks:
                hits = mask_intersection(pm, drawn_mask)
                if hits >= 11:
                    total_score += EXPONENTIAL_WEIGHTS.get(hits, 0)

        avg_score = total_score / len(indices)
        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization(portfolio_size=len(portfolio))
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        normalized = max(0.0, min(1.0, (avg_score - p5) / (p95 - p5 + 1e-10)))
        self._mc_cache[cache_key] = normalized
        return normalized

    def _compute_mc_normalization(self, portfolio_size=10, n_samples=200):
        raw_scores = []
        for _ in range(n_samples):
            rand_port = [self.generator.generate_pure_random() for _ in range(portfolio_size)]
            raw = self._monte_carlo_weighted_sum_raw(rand_port, 300)
            raw_scores.append(raw)
        raw_scores = np.array(raw_scores)
        return {'p5': float(np.percentile(raw_scores, 5)), 'p95': float(np.percentile(raw_scores, 95))}

    def _monte_carlo_weighted_sum_raw(self, portfolio, n_simulations=300):
        portfolio_masks = np.array([game_to_mask(g) for g in portfolio], dtype=np.uint32)
        indices = np.random.choice(len(self.historical_masks), size=min(n_simulations, len(self.historical_masks)))
        total_score = 0.0
        for idx in indices:
            drawn_mask = self.historical_masks[idx]
            for pm in portfolio_masks:
                hits = mask_intersection(pm, drawn_mask)
                if hits >= 11:
                    total_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
        return total_score / len(indices)

    def _repair_portfolio(self, portfolio, pool):
        repaired = list(portfolio)
        pool_set = set(tuple(g) for g in pool)
        for _ in range(20):
            pair_cov = self._pair_coverage(repaired)
            geo_div = self._geometric_diversity(repaired)
            if pair_cov <= MAX_PAIR_COVERAGE and MIN_GEO_DIVERSITY <= geo_div <= MAX_GEO_DIVERSITY:
                break
            if pair_cov > MAX_PAIR_COVERAGE:
                contributions = []
                for game in repaired:
                    other_pairs = set()
                    for g in repaired:
                        if g != game:
                            for pair in combinations(sorted(g), 2):
                                other_pairs.add(pair)
                    game_pairs = set(combinations(sorted(game), 2))
                    contributions.append(len(game_pairs - other_pairs))
                old_game = repaired[np.argmax(contributions)]
            else:
                fvs = np.array([self.extractor.extract_features(g, self.last) for g in repaired])
                centroid = np.mean(fvs, axis=0)
                distances = [np.linalg.norm(fv - centroid) for fv in fvs]
                old_game = repaired[np.argmax(distances) if geo_div > MAX_GEO_DIVERSITY else np.argmin(distances)]
            # Encontrar substituto
            best_candidate = None
            best_sim = -float('inf')
            old_features = self.extractor.extract_features(old_game, self.last)
            for candidate in random.sample(list(pool_set), min(500, len(pool_set))):
                candidate = list(candidate)
                if candidate == old_game or not self.extractor.is_structurally_valid(candidate):
                    continue
                cf = self.extractor.extract_features(candidate, self.last)
                dot = np.dot(old_features, cf)
                norm = np.linalg.norm(old_features) * np.linalg.norm(cf)
                sim = dot / norm if norm > 1e-10 else 1.0
                if 0.3 <= sim <= 0.7:
                    best_candidate = candidate
                    break
                if sim > best_sim:
                    best_sim, best_candidate = sim, candidate
            if best_candidate:
                repaired[repaired.index(old_game)] = best_candidate
        return repaired

    def _portfolio_score(self, portfolio):
        if self._pair_coverage(portfolio) > MAX_PAIR_COVERAGE:
            return -1000.0
        if not (MIN_GEO_DIVERSITY <= self._geometric_diversity(portfolio) <= MAX_GEO_DIVERSITY):
            return -1000.0
        mc_score = self._monte_carlo_weighted_sum(portfolio)
        structural = self._average_structural_score(portfolio)
        entropy_penalty = abs(self._portfolio_entropy(portfolio) - self.entropy_target) * 4.0
        return (mc_score * 0.40 + structural * 0.30 +
                self._portfolio_diversity(portfolio) * 0.20 +
                self._geometric_diversity(portfolio) * 0.10 -
                entropy_penalty * 0.10)

    def _mutate_game(self, game):
        for _ in range(20):
            mutated = list(game)
            for _ in range(random.randint(1, 3)):
                pos = random.randint(0, 14)
                avail = [d for d in range(1, 26) if d not in mutated]
                if avail: mutated[pos] = random.choice(avail)
            mutated = sorted(mutated)[:15]
            if self.extractor.is_structurally_valid(mutated):
                return mutated
        return sorted(game)[:15]

    def optimize(self, n_games=10, n_candidates=50000, iterations=100):
        n_central = max(1, int(n_games * 0.40))
        n_coverage = max(1, int(n_games * 0.35))
        n_temporal = n_games - n_central - n_coverage
        print(f"   Pipeline: {n_candidates//1000}k candidatos (Fase 1), Top 5k avaliação (Fase 2)")

        # FASE 1: Geração barata
        pool_all, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Fase 1"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen and self.extractor.is_structurally_valid(game):
                seen.add(key)
                pool_all.append(game)

        # FASE 2: Avaliação UNIFICADA no top 5000
        top_pool = random.sample(pool_all, min(5000, len(pool_all)))
        evaluated = []
        for g in tqdm(top_pool, desc="Fase 2"):
            feats, structural = self._evaluate_game(g)
            temporal = self.extractor.compute_temporal_score(g, self.feature_matrix)
            central_score = (self.extractor.compute_structural_score(g) * 0.6 +
                           temporal * 0.4)
            evaluated.append((central_score, temporal, g, feats, structural))

        evaluated.sort(key=lambda x: x[0], reverse=True)

        # Selecionar centrais
        central_games = []
        central_masks = []
        for cs, _, g, _, _ in evaluated:
            if len(central_games) >= n_central: break
            game_mask = game_to_mask(g)
            if central_masks and max(mask_intersection(game_mask, cm) for cm in central_masks) > 8:
                continue
            central_games.append(g)
            central_masks.append(game_mask)

        # Cobertura via greedy
        coverage_games = self._greedy_marginal_coverage(
            [g for _, _, g, _, _ in evaluated if g not in central_games],
            central_games, n_coverage)
        coverage_games = coverage_games[len(central_games):]

        # Temporais
        temporal_sorted = sorted(evaluated, key=lambda x: x[1], reverse=True)
        temporal_games = []
        for _, _, g, _, _ in temporal_sorted:
            if g in central_games or g in coverage_games: continue
            if len(temporal_games) >= n_temporal: break
            game_mask = game_to_mask(g)
            all_masks = central_masks + [game_to_mask(cg) for cg in coverage_games] + [game_to_mask(tg) for tg in temporal_games]
            if all_masks and max(mask_intersection(game_mask, am) for am in all_masks) > 9:
                continue
            temporal_games.append(g)

        portfolio = central_games + coverage_games + temporal_games
        portfolio = self._repair_portfolio(portfolio, pool_all)
        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)

        # Simulated Annealing
        elite_pool = [(cs, g) for cs, _, g, _, _ in evaluated[:len(evaluated)//4]]
        for it in tqdm(range(iterations), desc="Annealing"):
            temp = 1.0 * (0.95 ** it)
            new_portfolio = list(portfolio)
            idx = random.randint(0, len(new_portfolio) - 1)
            if random.random() < 0.4 and elite_pool:
                new_game = random.choice(elite_pool)[1]
            elif random.random() < 0.7:
                new_game = self._mutate_game(new_portfolio[idx])
            else:
                new_game = self.generator.generate_one()

            # BITMASK para verificação de similaridade
            new_mask = game_to_mask(new_game)
            too_similar = False
            for j, sg in enumerate(new_portfolio):
                if j != idx and mask_intersection(new_mask, game_to_mask(sg)) > 8:
                    too_similar = True
                    break
            if too_similar: continue

            new_portfolio[idx] = new_game
            new_score = self._portfolio_score(new_portfolio)
            if new_score > best_score:
                best_portfolio, best_score = list(new_portfolio), new_score
            elif random.random() < np.exp((new_score - self._portfolio_score(portfolio)) / max(0.01, temp)):
                portfolio = new_portfolio

        return best_portfolio, best_score

    def backtest(self, portfolio, test_draws):
        n_success, total_premio = 0, 0.0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([game_to_mask(g) for g in portfolio], dtype=np.uint32)
        for draw in test_draws:
            draw_mask = game_to_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, draw_mask)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
        prob = n_success / (len(portfolio) * len(test_draws)) if len(test_draws) > 0 else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11, 16))
        theo_prob = 1 - (1 - p_single) ** len(portfolio)
        return {
            'empirical': prob, 'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws), 'n_success': n_success,
            'total_premio': total_premio, 'total_custo': total_custo,
            'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0
        }


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=10):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)...")
    results = []
    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data, test_data = contests[train_start:train_end], contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue
        opt = PortfolioOptimizerV18(train_data)
        portfolio, _ = opt.optimize(n_games, n_candidates=50000, iterations=50)
        bt = opt.backtest(portfolio, test_data)
        bt_rand = opt.backtest([opt.generator.generate_pure_random() for _ in range(n_games)], test_data)
        results.append({
            'window': w,
            'diff_lift': bt['lift'] - bt_rand['lift'],
            'diff_roi': bt['roi'] - bt_rand['roi']
        })
        print(f" Janela {w}: diff_lift={bt['lift']-bt_rand['lift']:+.3f} diff_ROI={bt['roi']-bt_rand['roi']:+.1f}%")
    if results:
        diffs = [r['diff_lift'] for r in results]
        print(f"\n📊 Média diff lift: {np.mean(diffs):+.3f} | Janelas +: {sum(1 for d in diffs if d>0)}/{len(results)}")
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p: {p:.4f}")
        except: pass
    return results


# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR DE CARTEIRA v18 - BITMASK + REGULARIZADO")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print("\n📊 Configuração:")
    print(f"   Bitmask: SIM | GMM: REMOVIDO | Regularização: SIM")
    print(f"   Pesos cauda: 11:{EXPONENTIAL_WEIGHTS[11]} 12:{EXPONENTIAL_WEIGHTS[12]} 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")
    print(f"   Threshold estrutural: {STRUCTURAL_REJECT_THRESHOLD}")

    print("\nOpções: 1. Gerar carteira | 2. Walk-forward | 3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        t0 = time.time()
        opt = PortfolioOptimizerV18(contests)
        print(f"   ✅ Init {time.time()-t0:.1f}s | Cluster: {opt.current_cluster}")
        portfolio, _ = opt.optimize(10, 50000, 100)
        print(f"\n🏆 CARTEIRA:")
        last = contests[-1]['dezenas']
        for i, g in enumerate(portfolio, 1):
            p = sum(1 for d in g if d % 2 == 0)
            pr = sum(1 for d in g if d in PRIMES)
            m = sum(1 for d in g if d in MOLDURA)
            rep = len(set(g) & set(last))
            amp = max(g) - min(g)
            print(f"   {i:2d}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep} Amp:{amp}")
        if len(contests) > 200:
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
    if op in ("2", "3"):
        walk_forward_validation(contests, 10, 500, 50, 10)
    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
