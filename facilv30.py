#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v10
=================================================
CORREÇÕES BASEADAS EM EVIDÊNCIA EMPÍRICA:
✅ Teto de pair coverage (alvo 0.86, penalidade se >0.90)
✅ Limite de diversidade geométrica (alvo 0.45-0.70)
✅ Soft-hard gate: rejeitar jogos com penalidade > threshold
✅ Ensemble como estratégia PRINCIPAL
✅ Cobertura moderada + estrutura preservada
✅ Scores normalizados (z-score) mantidos
✅ Bootstrap histórico mantido
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

# Features topológicas v10 (17 dimensões)
FEATURE_NAMES_V10 = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
]
IDX = {name: i for i, name in enumerate(FEATURE_NAMES_V10)}

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

# Threshold para rejeitar jogos estruturalmente ruins
STRUCTURAL_REJECT_THRESHOLD = 15.0

# Alvos de cobertura moderada
TARGET_PAIR_COVERAGE = 0.86
TARGET_GEO_DIVERSITY_MIN = 0.45
TARGET_GEO_DIVERSITY_MAX = 0.70

# Features com sinal temporal (mantidas para 15% temporal)
TEMPORAL_FEATURES = {
    'moldura': 0.30,
    'amplitude': 0.25,
    'energia_jogo': 0.20,
    'densidade_local': 0.15,
    'clusterizacao': 0.10,
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
# EXTRATOR DE FEATURES v10 (com z-score)
# ============================================================
class TopologicalFeatureExtractorV10:
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
        """Penalidade HARD-SOFT: penaliza jogos fora do regime histórico médio."""
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
        """Soft-hard gate: rejeita jogos com penalidade acima do threshold."""
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

    def compute_structural_score(self, game):
        """Score estrutural (0-1, maior = mais típico)"""
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
class DistributionModelV10:
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
class FreeGeneratorV10:
    def __init__(self, last_contest=None, extractor=None):
        self.last = set(last_contest) if last_contest else None
        self.extractor = extractor

    def generate_one(self):
        max_attempts = 50
        for _ in range(max_attempts):
            game = self._generate_raw()
            # Soft-hard gate: rejeitar jogos com penalidade estrutural alta
            if self.extractor is not None:
                if not self.extractor.is_structurally_valid(game):
                    continue
            return game
        # Fallback: retornar o último gerado mesmo que inválido
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
# OTIMIZADOR DE CARTEIRA HÍBRIDA v10
# ============================================================
class HybridPortfolioOptimizerV10:
    """
    Otimizador de carteira v10: COBERTURA MODERADA + TETOS.
    
    Mudanças:
    - Teto de pair coverage (penalidade se >0.90)
    - Limite de diversidade geométrica (alvo 0.45-0.70)
    - Soft-hard gate (rejeitar jogos com penalidade > threshold)
    - Ensemble como estratégia PRINCIPAL
    """
    def __init__(self, contests):
        self.contests = contests
        self.extractor = TopologicalFeatureExtractorV10(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.dist_model = DistributionModelV10(self.feature_matrix)
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = FreeGeneratorV10(self.last, self.extractor)
        self._mc_cache = {}

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

    def _greedy_marginal_coverage_balanced(self, pool, existing_portfolio,
                                            n_select, max_intersection=7):
        """Greedy coverage EQUILIBRADO com teto de pair coverage."""
        covered_pairs = set()
        for g in existing_portfolio:
            for pair in combinations(sorted(g), 2):
                covered_pairs.add(pair)

        selected = list(existing_portfolio)
        remaining = [g for g in pool if g not in selected]

        for _ in range(n_select):
            if not remaining:
                break

            # Verificar se já atingiu o teto de cobertura
            current_coverage = len(covered_pairs) / comb(25, 2)
            if current_coverage >= TARGET_PAIR_COVERAGE:
                # Já atingiu cobertura moderada, focar apenas em estrutura
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

                # Penalidade estrutural forte (soft-hard gate)
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
        """
        Diversidade geométrica MODERADA.
        Penaliza tanto excesso (>0.70) quanto falta (<0.45).
        """
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
        max_expected = 2 * np.sqrt(len(FEATURE_NAMES_V10))
        raw = avg_dist / max_expected

        # Penalizar se estiver fora da faixa alvo
        if raw < TARGET_GEO_DIVERSITY_MIN:
            return raw / TARGET_GEO_DIVERSITY_MIN
        elif raw > TARGET_GEO_DIVERSITY_MAX:
            excess = raw - TARGET_GEO_DIVERSITY_MAX
            return max(0.1, 1.0 - excess * 3.0)
        else:
            return 1.0

    def _average_structural_score(self, portfolio):
        scores = [self.extractor.compute_structural_score(g) for g in portfolio]
        return np.mean(scores) if scores else 0.5

    def _monte_carlo_p11(self, portfolio, n_simulations=2000):
        cache_key = tuple(tuple(sorted(g)) for g in portfolio)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        historical_draws = [set(c['dezenas']) for c in self.contests[-500:]]
        if len(historical_draws) < 100:
            historical_draws = [
                set(np.random.choice(range(1, 26), 15, replace=False))
                for _ in range(500)
            ]

        successes = 0
        n_eval = min(n_simulations, len(historical_draws))

        for i in range(n_eval):
            drawn = historical_draws[random.randint(0, len(historical_draws) - 1)]
            if any(len(set(g) & drawn) >= 11 for g in portfolio):
                successes += 1

        prob = successes / n_eval if n_eval > 0 else 0
        self._mc_cache[cache_key] = prob
        if len(self._mc_cache) > 500:
            keys = list(self._mc_cache.keys())[:250]
            for k in keys:
                del self._mc_cache[k]
        return prob

    def _portfolio_score(self, portfolio):
        """Score equilibrado com tetos de cobertura e diversidade"""
        p11 = self._monte_carlo_p11(portfolio, n_simulations=1000)
        pair_cov = self._pair_coverage(portfolio)

        # TETO: penalizar cobertura excessiva
        if pair_cov > 0.90:
            pair_cov = 0.90 - (pair_cov - 0.90) * 3.0
        elif pair_cov > TARGET_PAIR_COVERAGE:
            pair_cov = TARGET_PAIR_COVERAGE + (pair_cov - TARGET_PAIR_COVERAGE) * 0.3

        structural = self._average_structural_score(portfolio)
        entropy_val = self._portfolio_entropy(portfolio)
        diversity = self._portfolio_diversity(portfolio)
        geo_div = self._geometric_diversity(portfolio)

        return (p11 * 0.25 + pair_cov * 0.20 + structural * 0.25 +
                entropy_val * 0.10 + diversity * 0.10 + geo_div * 0.10)

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
            # Soft-hard gate
            if self.extractor.is_structurally_valid(mutated):
                return mutated
        return sorted(game)[:15]

    def optimize_hybrid(self, n_games=10, n_candidates=200000, iterations=100):
        print(f"\n🎯 OTIMIZANDO CARTEIRA v10 ({n_games} jogos)...")

        n_central = max(1, int(n_games * 0.40))
        n_coverage = max(1, int(n_games * 0.35))
        n_temporal = n_games - n_central - n_coverage

        print(f"   Composição: {n_central} centrais + {n_coverage} cobertura + {n_temporal} temporais")
        print(f"   Teto pair coverage: {TARGET_PAIR_COVERAGE}")
        print(f"   Alvo geo diversidade: {TARGET_GEO_DIVERSITY_MIN}-{TARGET_GEO_DIVERSITY_MAX}")
        print(f"   Threshold estrutural: {STRUCTURAL_REJECT_THRESHOLD}")

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
                # Soft-hard gate
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

        # Cobertura via greedy EQUILIBRADO com teto
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

            # Verificar interseção
            too_similar = False
            for j, sg in enumerate(new_portfolio):
                if j != idx and len(set(new_game) & set(sg)) > 8:
                    too_similar = True
                    break
            if too_similar:
                continue

            new_portfolio[idx] = new_game
            new_score = self._portfolio_score(new_portfolio)

            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / max(0.01, temp)):
                portfolio = new_portfolio
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_portfolio = list(portfolio)

        return best_portfolio, best_score

    def generate_ensemble_consensus(self, n_strategies=3, n_games_per_strategy=5):
        """
        Ensemble PONDERADO - estratégia PRINCIPAL da v10.
        """
        print(f"\n🤝 ENSEMBLE PONDERADO (estratégia principal)...")
        strategy_results = []

        for s in range(n_strategies):
            portfolio, score = self.optimize_hybrid(
                n_games=n_games_per_strategy, n_candidates=50000, iterations=30)
            p11 = self._monte_carlo_p11(portfolio, n_simulations=1000)
            p_single = sum(HYPE_PROBS[k] for k in range(11, 16))
            p_none = (1 - p_single) ** len(portfolio)
            theo_prob = 1 - p_none
            lift = p11 / theo_prob if theo_prob > 0 else 1.0

            avg_structural = self._average_structural_score(portfolio)
            pair_cov = self._pair_coverage(portfolio)

            strategy_results.append({
                'portfolio': portfolio,
                'lift': lift,
                'structural': avg_structural,
                'pair_cov': pair_cov,
                'dezenas': [d for g in portfolio for d in g],
            })

        # Votação ponderada por lift E structural score
        weighted_freq = Counter()
        for sr in strategy_results:
            # Penalizar cobertura excessiva no peso
            cov_penalty = max(0, sr['pair_cov'] - TARGET_PAIR_COVERAGE) * 2
            weight = sr['lift'] * 0.5 + sr['structural'] * 0.5 - cov_penalty
            weight = max(0.1, weight)
            for d in sr['dezenas']:
                weighted_freq[d] += weight

        top_dezenas = [d for d, _ in weighted_freq.most_common(20)]

        # Gerar jogos com bônus de consenso + penalidade estrutural
        consensus_games = []
        seen = set()
        for _ in range(30000):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                consensus_count = len(set(game) & set(top_dezenas[:15]))
                if consensus_count >= 9:
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

        optimizer = HybridPortfolioOptimizerV10(train_data)
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
    print("🧬 GERADOR DE CARTEIRA v10 - COBERTURA MODERADA + TETOS")
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
    print(f"\n📊 TETOS:")
    print(f"   Pair coverage alvo: {TARGET_PAIR_COVERAGE}")
    print(f"   Geo diversidade: {TARGET_GEO_DIVERSITY_MIN}-{TARGET_GEO_DIVERSITY_MAX}")
    print(f"   Threshold estrutural: {STRUCTURAL_REJECT_THRESHOLD}")

    print("\nOpções:")
    print("1. Gerar carteira otimizada (cobertura moderada)")
    print("2. Ensemble ponderado (estratégia principal)")
    print("3. Walk-forward validation (10 janelas)")
    print("4. TUDO")
    op = input("Escolha [4]: ").strip() or "4"

    if op in ("1", "4"):
        print(f"\n🔧 INICIALIZANDO...")
        t0 = time.time()
        optimizer = HybridPortfolioOptimizerV10(contests)
        print(f"   ✅ Inicializado em {time.time()-t0:.1f}s")
        print(f"   GMM: {optimizer.dist_model.n_components} componentes")

        portfolio, score = optimizer.optimize_hybrid(
            n_games=10, n_candidates=200000, iterations=100)

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
        print(f"\n📊 Cobertura dezenas: {len(all_d)}/25")
        print(f"📊 Cobertura pares: {pair_cov:.3f} (alvo: {TARGET_PAIR_COVERAGE})")
        print(f"📊 Score estrutural: {structural_avg:.3f}")
        print(f"📊 Diversidade geométrica: {geo_div:.3f} (alvo: {TARGET_GEO_DIVERSITY_MIN}-{TARGET_GEO_DIVERSITY_MAX})")
        print(f"📊 Entropia: {optimizer._portfolio_entropy(portfolio):.3f}")
        p11_est = optimizer._monte_carlo_p11(portfolio, n_simulations=5000)
        print(f"📊 P(≥1 acerto 11+) estimada: {p11_est:.3f}")

        test_size = min(200, len(contests) // 3)
        if test_size > 10:
            test_data = contests[-test_size:]
            bt = optimizer.backtest(portfolio, test_data)
            print(f"\n🔬 BACKTEST ({bt['n_test']} concursos):")
            print(f"   Prob ≥1 acerto 11+: {bt['empirical']:.2%} "
                  f"(teórico: {bt['theoretical']:.2%})")
            print(f"   Lift: {bt['lift']:.2f}x")

    if op in ("2", "4"):
        optimizer = HybridPortfolioOptimizerV10(contests)
        consensus = optimizer.generate_ensemble_consensus(
            n_strategies=3, n_games_per_strategy=5)
        print(f"\n🤝 CARTEIRA CONSENSO ({len(consensus)} jogos):")
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
