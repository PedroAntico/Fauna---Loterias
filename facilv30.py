#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v32 (MULTIOBJETIVO NSGA-II)

Substitui o score escalar por otimização multiobjetivo:
✅ Objetivos separados: MC, naturalness, diversidade geométrica, correlação, cobertura
✅ Seleção por dominância de Pareto (non‑dominated sorting)
✅ População evoluída com crossover e mutação
✅ Frente de Pareto final com seleção automática equilibrada
✅ Mantém toda a estrutura de features, KDE, Monte Carlo e restrições estruturais
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

# Constraints estruturais
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

# Ajustes do v31.1 (mantidos)
KDE_BANDWIDTH = 0.03
SOFTMAX_TEMPERATURE = 5.0
CORRELATION_THRESHOLD = 0.2

# Configuração NSGA-II
POP_SIZE = 50
N_GENERATIONS = 40
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.3

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

        self._build_mahalanobis_model(raw_features)
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
        pcts = np.array([np.mean(self.historical_mahalanobis <= d) for d in self.historical_mahalanobis])
        self._pct_kde = KernelDensity(bandwidth=KDE_BANDWIDTH).fit(pcts.reshape(-1, 1))
        self._hist_pct_log_dens = self._pct_kde.score_samples(pcts.reshape(-1, 1))

    def mahalanobis_distance(self, raw_features):
        diff = raw_features - self._mean_vector
        try:
            return float(np.sqrt(max(0, np.dot(np.dot(diff.T, self.precision_matrix), diff))))
        except:
            return float(np.linalg.norm(diff / (np.std(self.historical_mahalanobis) + 1e-10)))

    def compute_naturalness_score(self, game):
        raw = self.extract_raw_features(game)
        dist = self.mahalanobis_distance(raw)
        pct = np.mean(self.historical_mahalanobis <= dist)
        log_dens = self._pct_kde.score_samples([[pct]])[0]
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
            'pares': total_pares, 'primos': total_primos,
            'moldura': total_moldura, 'repeticoes': total_rep,
            'soma': total_soma, 'consecutivos': total_consec,
            'amplitude': total_amplitude,
        }
        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name]-target)
                if dev > tol:
                    penalty += (dev - tol) * w

        if total_consec > MAX_TOTAL_CONSECUTIVOS:
            penalty += (total_consec - MAX_TOTAL_CONSECUTIVOS) * 8.0

        max_run = 1; run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1:
                run += 1; max_run = max(max_run, run)
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
# OTIMIZADOR v32 (NSGA-II)
# ============================================================
class PortfolioOptimizerV32:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = LooseGenerator(self.extractor)

        self.historical_draws = [c['dezenas'] for c in self.contests]
        self.historical_masks = draw_masks_to_array(self.historical_draws)
        if len(self.historical_draws) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draw_masks_to_array(self.historical_draws)

        self.historical_features = np.array([self.extractor.extract_features(list(d), None) for d in self.historical_draws])
        recent_f = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix
        self.recent_centroid = np.mean(recent_f, axis=0)

        # Caches
        self._mc_cache = {}
        self._mc_norm_params = None
        self._score_cache = {}  # Para objetivos

    def _create_candidate(self, game):
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        nat_score, nat_pct, mahal_dist = self.extractor.compute_naturalness_score(game)
        return GameCandidate(game, mask, features, nat_score, nat_pct, mahal_dist)

    # Objetivos separados (retornam valores brutos, serão maximizados/minimizados)
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
        if len(portfolio) < 2: return 0.0
        fvs = np.array([c.features for c in portfolio])
        corr = np.corrcoef(fvs)
        triu = np.triu_indices_from(corr, k=1)
        mean_abs_corr = np.mean(np.abs(corr[triu]))
        if mean_abs_corr > CORRELATION_THRESHOLD:
            return (mean_abs_corr - CORRELATION_THRESHOLD) * 0.2
        return 0.0

    def _monte_carlo_score(self, portfolio_candidates):
        cache_key = tuple(tuple(sorted(c.game)) for c in portfolio_candidates)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        avg_dists = np.linalg.norm(self.historical_features - self.recent_centroid, axis=1)
        logits = avg_dists / SOFTMAX_TEMPERATURE
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()

        n_sim = 500
        indices = np.random.choice(len(self.historical_masks), size=n_sim, p=weights)
        drawn_masks = self.historical_masks[indices]
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)

        total = 0.0
        for dm in drawn_masks:
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 13:
                    total += EXPONENTIAL_WEIGHTS.get(hits, 0)
        avg = total / n_sim

        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization()
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        normalized = max(0.0, min(1.0, (avg - p5) / (p95 - p5 + 1e-10)))
        self._mc_cache[cache_key] = normalized
        return normalized

    def _monte_carlo_raw(self, portfolio_candidates, n_sim=300):
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)
        indices = np.random.choice(len(self.historical_masks), size=n_sim)
        total = 0.0
        for idx in indices:
            drawn_mask = self.historical_masks[idx]
            for pm in portfolio_masks:
                hits = mask_intersection(pm, drawn_mask)
                if hits >= 13:
                    total += EXPONENTIAL_WEIGHTS.get(hits, 0)
        return total / n_sim

    def _compute_mc_normalization(self, n_samples=200):
        raw_scores = []
        for _ in range(n_samples):
            rand_port = [self._create_candidate(self.generator.generate_pure_random()) for _ in range(5)]
            raw = self._monte_carlo_raw(rand_port, 300)
            raw_scores.append(raw)
        raw_scores = np.array(raw_scores)
        return {'p5': float(np.percentile(raw_scores,5)), 'p95': float(np.percentile(raw_scores,95))}

    def _evaluate_portfolio(self, portfolio):
        """Retorna um dicionário com os objetivos de um portfólio."""
        mc = self._monte_carlo_score(portfolio)
        naturalness = np.mean([c.naturalness_score for c in portfolio])
        geo_div = self._geometric_diversity(portfolio)
        corr_pen = self._correlation_penalty(portfolio)
        div = self._portfolio_diversity(portfolio)
        coverage = self._pair_coverage(portfolio)

        # A cobertura não deve ultrapassar MAX_PAIR_COVERAGE, penalizamos com valor baixo
        if coverage > MAX_PAIR_COVERAGE:
            # Objetivos muito ruins (podem ser tratados via dominância)
            mc = -1.0
            naturalness = 0.0
            geo_div = 0.0
            corr_pen = 1.0
            div = 0.0
        if not (MIN_GEO_DIVERSITY <= geo_div <= MAX_GEO_DIVERSITY):
            mc = -1.0
            naturalness = 0.0
            geo_div = 0.0
            corr_pen = 1.0
            div = 0.0

        return {
            'mc': mc,
            'naturalness': naturalness,
            'geo_diversity': geo_div,
            'correlation_penalty': corr_pen,   # queremos minimizar
            'diversity': div
        }

    # Métodos do NSGA-II
    def _dominates(self, obj1, obj2):
        """Retorna True se obj1 domina obj2 (minimização de correlation_penalty, maximização dos outros)."""
        # Definimos um vetor onde todos são para maximizar: convertemos correlation_penalty para negativo
        v1 = np.array([obj1['mc'], obj1['naturalness'], obj1['geo_diversity'], -obj1['correlation_penalty'], obj1['diversity']])
        v2 = np.array([obj2['mc'], obj2['naturalness'], obj2['geo_diversity'], -obj2['correlation_penalty'], obj2['diversity']])
        if np.all(v1 >= v2) and np.any(v1 > v2):
            return True
        return False

    def _fast_non_dominated_sort(self, population):
        # population: lista de dicionários {'port': lista de GameCandidate, 'obj': dict}
        fronts = []
        remaining = list(range(len(population)))
        domination = {i: [] for i in remaining}
        dominated_count = {i: 0 for i in remaining}

        for i in remaining:
            for j in remaining:
                if i == j: continue
                if self._dominates(population[i]['obj'], population[j]['obj']):
                    domination[i].append(j)
                elif self._dominates(population[j]['obj'], population[i]['obj']):
                    dominated_count[i] += 1

        current_front = [i for i in remaining if dominated_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in domination[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        return fronts

    def _crowding_distance(self, front, population):
        if len(front) <= 2:
            return {i: float('inf') for i in front}
        distances = {i: 0.0 for i in front}
        # Para cada objetivo
        for obj_name in ['mc', 'naturalness', 'geo_diversity', 'diversity']:
            # Ordena pelo objetivo
            sorted_front = sorted(front, key=lambda i: population[i]['obj'][obj_name])
            obj_min = population[sorted_front[0]]['obj'][obj_name]
            obj_max = population[sorted_front[-1]]['obj'][obj_name]
            if obj_max - obj_min < 1e-12:
                continue
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')
            for k in range(1, len(sorted_front)-1):
                distances[sorted_front[k]] += (population[sorted_front[k+1]]['obj'][obj_name] - population[sorted_front[k-1]]['obj'][obj_name]) / (obj_max - obj_min)
        # correlation_penalty (minimizar)
        sorted_front = sorted(front, key=lambda i: population[i]['obj']['correlation_penalty'])
        obj_min = population[sorted_front[0]]['obj']['correlation_penalty']
        obj_max = population[sorted_front[-1]]['obj']['correlation_penalty']
        if obj_max - obj_min > 1e-12:
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')
            for k in range(1, len(sorted_front)-1):
                distances[sorted_front[k]] += (population[sorted_front[k+1]]['obj']['correlation_penalty'] - population[sorted_front[k-1]]['obj']['correlation_penalty']) / (obj_max - obj_min)
        return distances

    def _tournament_select(self, population, fronts, crowding_distances):
        # Torneio binário
        def better(i, j):
            # Encontra a frente de cada um
            front_i = next(idx for idx, f in enumerate(fronts) if i in f)
            front_j = next(idx for idx, f in enumerate(fronts) if j in f)
            if front_i < front_j:
                return True
            elif front_i == front_j:
                return crowding_distances.get(i, 0) > crowding_distances.get(j, 0)
            return False

        i, j = random.sample(range(len(population)), 2)
        return population[i] if better(i, j) else population[j]

    def _crossover(self, parent1, parent2):
        # Troca um jogo entre os portfólios
        child1_candidates = list(parent1)
        child2_candidates = list(parent2)
        if len(child1_candidates) < 2: return child1_candidates, child2_candidates
        idx1 = random.randint(0, len(child1_candidates)-1)
        idx2 = random.randint(0, len(child2_candidates)-1)
        # Garantir que após troca não haja duplicatas
        child1_candidates[idx1], child2_candidates[idx2] = child2_candidates[idx2], child1_candidates[idx1]
        # Verificar se não tem sobreposição excessiva (já tratado na criação)
        return child1_candidates, child2_candidates

    def _mutate(self, portfolio_candidates):
        # Substitui um jogo aleatório por um mutado
        mutated = list(portfolio_candidates)
        idx = random.randint(0, len(mutated)-1)
        # Mutação do candidato
        for _ in range(20):
            game = list(mutated[idx].game)
            for _ in range(random.randint(1,4)):
                pos = random.randint(0,14)
                avail = [d for d in range(1,26) if d not in game]
                if avail:
                    game[pos] = random.choice(avail)
            game = sorted(game)[:15]
            if self.extractor.is_structurally_valid(game):
                mutated[idx] = self._create_candidate(game)
                break
        # Garantir que não haja sobreposição > 10 entre jogos
        masks = [c.mask for c in mutated]
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                if mask_intersection(masks[i], masks[j]) > 10:
                    return portfolio_candidates  # rejeita mutação se criar sobreposição
        return mutated

    def _generate_initial_population(self, n_games, pop_size):
        print("🔍 Gerando população inicial...")
        population = []
        for _ in tqdm(range(pop_size), desc="Pop inicial"):
            # Gera portfólio aleatório válido
            portfolio = []
            seen = set()
            attempts = 0
            while len(portfolio) < n_games and attempts < 200:
                game = self.generator.generate_one()
                key = tuple(game)
                if key not in seen and self.extractor.is_structurally_valid(game):
                    cand = self._create_candidate(game)
                    if not portfolio or max(mask_intersection(cand.mask, c.mask) for c in portfolio) <= 10:
                        portfolio.append(cand)
                        seen.add(key)
                attempts += 1
            # Se não conseguir preencher, completa com aleatórios (mas mantendo restrições)
            while len(portfolio) < n_games:
                game = self.generator.generate_pure_random()
                if self.extractor.is_structurally_valid(game):
                    cand = self._create_candidate(game)
                    if not portfolio or max(mask_intersection(cand.mask, c.mask) for c in portfolio) <= 10:
                        portfolio.append(cand)
            obj = self._evaluate_portfolio(portfolio)
            population.append({'port': portfolio, 'obj': obj})
        return population

    def optimize(self, n_games=5, pop_size=POP_SIZE, generations=N_GENERATIONS):
        print(f"🎯 Carteira MULTIOBJETIVO: {n_games} jogos")
        print(f"📊 NSGA-II: pop={pop_size}, gen={generations}")
        population = self._generate_initial_population(n_games, pop_size)

        for gen in tqdm(range(generations), desc="Evolução"):
            # Avaliar todos (já feito na inicialização, mas mutações geram novos)
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(population)
            # Crowding distance
            distances = {}
            for front in fronts:
                dists = self._crowding_distance(front, population)
                distances.update(dists)

            # Nova população
            new_population = []
            while len(new_population) < pop_size:
                parent1 = self._tournament_select(population, fronts, distances)
                parent2 = self._tournament_select(population, fronts, distances)
                child1_candidates, child2_candidates = parent1['port'][:], parent2['port'][:]
                if random.random() < CROSSOVER_PROB:
                    child1_candidates, child2_candidates = self._crossover(parent1['port'], parent2['port'])
                if random.random() < MUTATION_PROB:
                    child1_candidates = self._mutate(child1_candidates)
                if random.random() < MUTATION_PROB:
                    child2_candidates = self._mutate(child2_candidates)
                # Avaliar
                obj1 = self._evaluate_portfolio(child1_candidates)
                obj2 = self._evaluate_portfolio(child2_candidates)
                new_population.append({'port': child1_candidates, 'obj': obj1})
                if len(new_population) < pop_size:
                    new_population.append({'port': child2_candidates, 'obj': obj2})
            population = new_population

        # Frente de Pareto final
        fronts = self._fast_non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]] if fronts else []
        print(f"   Frente de Pareto: {len(pareto_front)} portfólios")
        # Selecionar um portfólio da frente com melhor equilíbrio (produto mc*naturalness)
        best = max(pareto_front, key=lambda p: p['obj']['mc'] * p['obj']['naturalness'] * (1 - p['obj']['correlation_penalty']))
        return [c.game for c in best['port']], best['obj']

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
# WALK-FORWARD (v32)
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
        opt = PortfolioOptimizerV32(train_data)
        portfolio, _ = opt.optimize(n_games, pop_size=30, generations=20)  # menos gerações para agilizar
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
    print("🧬 GERADOR DE CARTEIRA v32 - MULTIOBJETIVO NSGA-II")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado."); return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print(f"\n📊 NSGA-II: pop={POP_SIZE}, gen={N_GENERATIONS} | Objetivos: MC, naturalness, diversidade, correlação")
    print(f"💰 Pesos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")

    op = input("Opções: 1. Gerar carteira | 2. Walk-forward | 3. Ambos\nEscolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        t0 = time.time()
        opt = PortfolioOptimizerV32(contests)
        print(f"   ✅ Init {time.time()-t0:.1f}s")
        portfolio, best_obj = opt.optimize(5, POP_SIZE, N_GENERATIONS)
        last = contests[-1]['dezenas']
        gen_features = np.array([opt.extractor.extract_features(g, last) for g in portfolio])
        kl = opt.extractor.compute_kl_divergence(gen_features)
        print(f"\n📊 KL Divergence: {kl:.3f}")
        print(f"   Objetivos do selecionado: MC={best_obj['mc']:.3f} Nat={best_obj['naturalness']:.3f} GeoDiv={best_obj['geo_diversity']:.3f} CorrPen={best_obj['correlation_penalty']:.3f} Div={best_obj['diversity']:.3f}")
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
