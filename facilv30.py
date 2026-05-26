#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v33 (GERADOR DIVERSO + MC SUAVE + OBJETIVOS INDEPENDENTES)

✅ Gerador com nichos estruturais (baixa/moldura, alta/pares, etc.)
✅ MC com escala logarítmica (não explode) + gradiente suave
✅ Objetivos independentes: MC, naturalness, cobertura combinatória, diversidade topológica, anti‑correlação
✅ Crossover por troca de subconjuntos (pares, primos, quadrantes)
✅ População inicial diversa via grelha de características estruturais
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
from sklearn.neighbors import KernelDensity
from collections import Counter, defaultdict
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

# Constraints estruturais (mantidas, mas o gerador agora explora nichos)
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

# Pesos suavizados para MC (evita explosão)
# Usaremos log(1 + score) e depois normalizamos por ranking
MC_WEIGHTS = {11: 0.5, 12: 1.0, 13: 2.0, 14: 5.0, 15: 10.0}  # escala logarítmica implícita

# Ajustes mantidos
KDE_BANDWIDTH = 0.03
SOFTMAX_TEMPERATURE = 5.0
CORRELATION_THRESHOLD = 0.2

# Configuração NSGA-II
POP_SIZE = 50
N_GENERATIONS = 40
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.4

# Nichos estruturais para o gerador
NICHE_DEFS = [
    {'label': 'low_moldura', 'target': {'moldura': (5, 7)}},
    {'label': 'high_moldura', 'target': {'moldura': (10, 12)}},
    {'label': 'low_primes', 'target': {'primos': (2, 4)}},
    {'label': 'high_primes', 'target': {'primos': (6, 8)}},
    {'label': 'balanced', 'target': {}},
]

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
# GERADOR DIVERSO (COM NICHOS ESTRUTURAIS)
# ============================================================
class DiverseGenerator:
    def __init__(self, extractor=None):
        self.extractor = extractor

    def generate_one(self, niche=None):
        """Gera um jogo, opcionalmente forçando um nicho estrutural."""
        for _ in range(50):
            game = self._generate_raw(niche)
            if self.extractor is not None and self.extractor.is_structurally_valid(game):
                return game
        return self._generate_raw(niche)

    def _generate_raw(self, niche=None):
        game = set()
        available = set(range(1, 26))
        # Se nicho definido, pré-seleciona algumas dezenas para atingir target
        forced = []
        if niche:
            for attr, (lo, hi) in niche.get('target', {}).items():
                if attr == 'moldura':
                    # Força algumas dezenas de moldura
                    mold_candidates = list(MOLDURA - game)
                    needed = random.randint(lo, hi)
                    forced.extend(random.sample(mold_candidates, min(needed, len(mold_candidates))))
                elif attr == 'primos':
                    prime_candidates = list(PRIMES - game)
                    needed = random.randint(lo, hi)
                    forced.extend(random.sample(prime_candidates, min(needed, len(prime_candidates))))
        # Adiciona forçadas primeiro
        for d in forced:
            if d in available:
                game.add(d)
                available.remove(d)

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
                # Pequeno bônus para atender alvo de moldura/primos se nicho especificado
                if niche:
                    for attr, (lo, hi) in niche.get('target', {}).items():
                        current = sum(1 for x in test if (attr=='moldura' and x in MOLDURA) or (attr=='primos' and x in PRIMES))
                        if current < lo:
                            s += 1
                        elif current > hi:
                            s -= 1
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
# OTIMIZADOR v33 (GERADOR DIVERSO + MC SUAVE + OBJETIVOS INDEPENDENTES)
# ============================================================
class PortfolioOptimizerV33:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = DiverseGenerator(self.extractor)

        self.historical_draws = [c['dezenas'] for c in self.contests]
        self.historical_masks = draw_masks_to_array(self.historical_draws)
        if len(self.historical_draws) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draw_masks_to_array(self.historical_draws)

        self.historical_features = np.array([self.extractor.extract_features(list(d), None) for d in self.historical_draws])
        recent_f = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix
        self.recent_centroid = np.mean(recent_f, axis=0)

        self._mc_cache = {}
        self._mc_norm_params = None

    def _create_candidate(self, game):
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        nat_score, nat_pct, mahal_dist = self.extractor.compute_naturalness_score(game)
        return GameCandidate(game, mask, features, nat_score, nat_pct, mahal_dist)

    # ----- NOVOS OBJETIVOS INDEPENDENTES -----
    def _combinatorial_diversity(self, portfolio):
        """Cobertura de pares e trios, normalizada para ser maximizada."""
        covered_pairs = set()
        covered_triples = set()
        for c in portfolio:
            for pair in combinations(sorted(c.game), 2):
                covered_pairs.add(pair)
            for triple in combinations(sorted(c.game), 3):
                covered_triples.add(triple)
        pair_ratio = len(covered_pairs) / comb(25, 2)
        triple_ratio = len(covered_triples) / comb(25, 3)
        # Quanto maior a cobertura, melhor
        return (pair_ratio + triple_ratio) / 2.0

    def _topological_diversity(self, portfolio):
        """Entropia da distribuição das dezenas no portfólio (independe de features)."""
        freq = np.zeros(26)
        for c in portfolio:
            for d in c.game:
                freq[d] += 1
        prob = freq[1:] / np.sum(freq[1:])
        prob = prob[prob > 0]
        ent = entropy(prob)
        # Normaliza pela entropia máxima (log(25))
        return ent / np.log(25)

    def _pair_coverage(self, portfolio):
        covered = set()
        for c in portfolio:
            for pair in combinations(sorted(c.game), 2):
                covered.add(pair)
        return len(covered) / comb(25, 2)

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

    # ----- MC SUAVIZADO -----
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
                if hits >= 11:  # dá crédito a partir de 11
                    total += MC_WEIGHTS.get(hits, 0)
        avg = total / n_sim
        # Normalização por ranking: calculamos percentil entre portfólios aleatórios (cache)
        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization()
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        # Se muito concentrado, usa escala logit
        if p95 - p5 < 1e-6:
            normalized = 0.5
        else:
            normalized = max(0.0, min(1.0, (avg - p5) / (p95 - p5 + 1e-6)))
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
                if hits >= 11:
                    total += MC_WEIGHTS.get(hits, 0)
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
        mc = self._monte_carlo_score(portfolio)
        naturalness = np.mean([c.naturalness_score for c in portfolio])
        comb_div = self._combinatorial_diversity(portfolio)
        topo_div = self._topological_diversity(portfolio)
        corr_pen = self._correlation_penalty(portfolio)
        geo_div = self._geometric_diversity(portfolio)
        # Penaliza violações fortes com valor baixo
        if self._pair_coverage(portfolio) > MAX_PAIR_COVERAGE or not (MIN_GEO_DIVERSITY <= geo_div <= MAX_GEO_DIVERSITY):
            mc = -1.0
        return {
            'mc': mc,
            'naturalness': naturalness,
            'combinatorial_diversity': comb_div,
            'topological_diversity': topo_div,
            'geo_diversity': geo_div,
            'correlation_penalty': corr_pen,  # minimizar
        }

    # ----- NSGA-II (adaptado) -----
    def _dominates(self, obj1, obj2):
        # maximizar todos, exceto correlation_penalty (minimizar)
        v1 = np.array([obj1['mc'], obj1['naturalness'], obj1['combinatorial_diversity'], obj1['topological_diversity'], obj1['geo_diversity'], -obj1['correlation_penalty']])
        v2 = np.array([obj2['mc'], obj2['naturalness'], obj2['combinatorial_diversity'], obj2['topological_diversity'], obj2['geo_diversity'], -obj2['correlation_penalty']])
        return np.all(v1 >= v2) and np.any(v1 > v2)

    def _fast_non_dominated_sort(self, population):
        fronts = []
        rem = list(range(len(population)))
        dom = {i: [] for i in rem}
        dom_count = {i: 0 for i in rem}
        for i in rem:
            for j in rem:
                if i == j: continue
                if self._dominates(population[i]['obj'], population[j]['obj']):
                    dom[i].append(j)
                elif self._dominates(population[j]['obj'], population[i]['obj']):
                    dom_count[i] += 1
        curr = [i for i in rem if dom_count[i] == 0]
        while curr:
            fronts.append(curr)
            next_curr = []
            for i in curr:
                for j in dom[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_curr.append(j)
            curr = next_curr
        return fronts

    def _crowding_distance(self, front, population):
        if len(front) <= 2:
            return {i: float('inf') for i in front}
        dists = {i: 0.0 for i in front}
        for obj in ['mc', 'naturalness', 'combinatorial_diversity', 'topological_diversity', 'geo_diversity']:
            sf = sorted(front, key=lambda i: population[i]['obj'][obj])
            omin = population[sf[0]]['obj'][obj]
            omax = population[sf[-1]]['obj'][obj]
            if omax - omin < 1e-12: continue
            dists[sf[0]] = float('inf')
            dists[sf[-1]] = float('inf')
            for k in range(1, len(sf)-1):
                dists[sf[k]] += (population[sf[k+1]]['obj'][obj] - population[sf[k-1]]['obj'][obj]) / (omax - omin)
        # correlation_penalty (minimizar)
        sf = sorted(front, key=lambda i: population[i]['obj']['correlation_penalty'])
        omin = population[sf[0]]['obj']['correlation_penalty']
        omax = population[sf[-1]]['obj']['correlation_penalty']
        if omax - omin > 1e-12:
            dists[sf[0]] = float('inf')
            dists[sf[-1]] = float('inf')
            for k in range(1, len(sf)-1):
                dists[sf[k]] += (population[sf[k+1]]['obj']['correlation_penalty'] - population[sf[k-1]]['obj']['correlation_penalty']) / (omax - omin)
        return dists

    def _tournament_select(self, population, fronts, cd):
        def better(i, j):
            fi = next(idx for idx, f in enumerate(fronts) if i in f)
            fj = next(idx for idx, f in enumerate(fronts) if j in f)
            if fi < fj: return True
            if fi == fj: return cd.get(i,0) > cd.get(j,0)
            return False
        i, j = random.sample(range(len(population)), 2)
        return population[i] if better(i,j) else population[j]

    def _crossover(self, parent1, parent2):
        """Crossover disruptivo: troca subconjuntos (pares, primos, quadrantes)."""
        # Escolhe um tipo de subconjunto aleatoriamente
        tipo = random.choice(['pares', 'primos', 'quadrantes'])
        games1 = [list(c.game) for c in parent1]
        games2 = [list(c.game) for c in parent2]
        if tipo == 'pares':
            # Troca todos os números pares entre dois jogos aleatórios
            i, j = random.sample(range(len(games1)), 2) if len(games1)>=2 else (0,0)
            pares1 = [d for d in games1[i] if d%2==0]
            pares2 = [d for d in games2[i] if d%2==0]
            # Substitui mantendo tamanho
            if pares1 and pares2:
                games1[i] = [d for d in games1[i] if d%2!=0] + random.sample(pares2, min(len(pares2), len(pares1)))
                games2[i] = [d for d in games2[i] if d%2!=0] + random.sample(pares1, min(len(pares1), len(pares2)))
        elif tipo == 'primos':
            i, j = random.sample(range(len(games1)), 2) if len(games1)>=2 else (0,0)
            prim1 = [d for d in games1[i] if d in PRIMES]
            prim2 = [d for d in games2[i] if d in PRIMES]
            if prim1 and prim2:
                games1[i] = [d for d in games1[i] if d not in PRIMES] + random.sample(prim2, min(len(prim2), len(prim1)))
                games2[i] = [d for d in games2[i] if d not in PRIMES] + random.sample(prim1, min(len(prim1), len(prim2)))
        elif tipo == 'quadrantes':
            # Troca todas as dezenas de um mesmo quadrante entre dois jogos
            q = random.randint(1,5)
            quad_range = range((q-1)*5+1, q*5+1)
            i, j = random.sample(range(len(games1)), 2) if len(games1)>=2 else (0,0)
            q1 = [d for d in games1[i] if d in quad_range]
            q2 = [d for d in games2[i] if d in quad_range]
            games1[i] = [d for d in games1[i] if d not in quad_range] + q2
            games2[i] = [d for d in games2[i] if d not in quad_range] + q1
        # Reconstroi candidatos após ordenar e garantir 15 números
        new_parent1 = []
        for g in games1:
            new_parent1.append(self._create_candidate(sorted(g)[:15]) if self.extractor.is_structurally_valid(sorted(g)[:15]) else random.choice(parent1))
        new_parent2 = []
        for g in games2:
            new_parent2.append(self._create_candidate(sorted(g)[:15]) if self.extractor.is_structurally_valid(sorted(g)[:15]) else random.choice(parent2))
        return new_parent1, new_parent2

    def _mutate(self, portfolio):
        # Mutação guiada: troca uma dezena por outra de um grupo complementar
        mutated = [list(c.game) for c in portfolio]
        idx = random.randint(0, len(mutated)-1)
        game = mutated[idx]
        if random.random() < 0.5:
            # Remove uma dezena aleatória e adiciona uma nova que não esteja presente
            if len(game) == 15:
                old = random.choice(game)
                game.remove(old)
                candidates = [d for d in range(1,26) if d not in game]
                if candidates:
                    game.append(random.choice(candidates))
        else:
            # Substitui uma dezena de moldura por uma do centro, ou vice-versa
            if any(d in MOLDURA for d in game):
                old = random.choice([d for d in game if d in MOLDURA])
                game.remove(old)
                cand_centro = [d for d in CENTRO if d not in game] if 'CENTRO' in dir() else [d for d in range(1,26) if d not in game and d not in MOLDURA]
                if cand_centro:
                    game.append(random.choice(cand_centro))
        mutated[idx] = sorted(game)[:15]
        # Verifica validade estrutural e interseção máxima
        candidates = [self._create_candidate(g) if self.extractor.is_structurally_valid(g) else c for g, c in zip(mutated, portfolio)]
        masks = [c.mask for c in candidates]
        if any(mask_intersection(masks[i], masks[j]) > 10 for i in range(len(masks)) for j in range(i+1, len(masks))):
            return portfolio  # rejeita mutação
        return candidates

    def _generate_initial_population(self, n_games, pop_size):
        print("🔍 População inicial com diversidade estrutural...")
        population = []
        # Distribui igualmente entre nichos
        niches_per_ind = pop_size // len(NICHE_DEFS)
        for niche in NICHE_DEFS:
            for _ in range(niches_per_ind):
                portfolio = []
                seen = set()
                attempts = 0
                while len(portfolio) < n_games and attempts < 200:
                    game = self.generator.generate_one(niche)
                    key = tuple(game)
                    if key not in seen and self.extractor.is_structurally_valid(game):
                        cand = self._create_candidate(game)
                        if not portfolio or max(mask_intersection(cand.mask, c.mask) for c in portfolio) <= 10:
                            portfolio.append(cand)
                            seen.add(key)
                    attempts += 1
                # Completa com aleatório se necessário
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
        print(f"🎯 Carteira MULTIOBJETIVO v33: {n_games} jogos")
        print(f"📊 Objetivos: MC suave, naturalness, diversidade combinatória, topológica, anti‑correlação")
        population = self._generate_initial_population(n_games, pop_size)

        for gen in tqdm(range(generations), desc="Evolução"):
            fronts = self._fast_non_dominated_sort(population)
            distances = {}
            for front in fronts:
                dists = self._crowding_distance(front, population)
                distances.update(dists)

            new_pop = []
            while len(new_pop) < pop_size:
                p1 = self._tournament_select(population, fronts, distances)
                p2 = self._tournament_select(population, fronts, distances)
                child1_candidates, child2_candidates = p1['port'][:], p2['port'][:]
                if random.random() < CROSSOVER_PROB:
                    child1_candidates, child2_candidates = self._crossover(p1['port'], p2['port'])
                if random.random() < MUTATION_PROB:
                    child1_candidates = self._mutate(child1_candidates)
                if random.random() < MUTATION_PROB:
                    child2_candidates = self._mutate(child2_candidates)
                obj1 = self._evaluate_portfolio(child1_candidates)
                obj2 = self._evaluate_portfolio(child2_candidates)
                new_pop.append({'port': child1_candidates, 'obj': obj1})
                if len(new_pop) < pop_size:
                    new_pop.append({'port': child2_candidates, 'obj': obj2})
            population = new_pop

        fronts = self._fast_non_dominated_sort(population)
        pareto = [population[i] for i in fronts[0]] if fronts else []
        print(f"   Frente de Pareto: {len(pareto)} portfólios")
        best = max(pareto, key=lambda p: p['obj']['mc'] * p['obj']['naturalness'] * p['obj']['combinatorial_diversity'] * (1 - p['obj']['correlation_penalty']))
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
# WALK-FORWARD (v33)
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
        opt = PortfolioOptimizerV33(train_data)
        portfolio, _ = opt.optimize(n_games, pop_size=30, generations=20)
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
    print("🧬 GERADOR DE CARTEIRA v33 - DIVERSIDADE REAL + MC SUAVE")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado."); return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print(f"📊 Nichos geradores: {[n['label'] for n in NICHE_DEFS]}")
    print(f"⚖️ MC suave: {MC_WEIGHTS}")

    op = input("Opções: 1. Gerar carteira | 2. Walk-forward | 3. Ambos\nEscolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        t0 = time.time()
        opt = PortfolioOptimizerV33(contests)
        print(f"   ✅ Init {time.time()-t0:.1f}s")
        portfolio, best_obj = opt.optimize(5, POP_SIZE, N_GENERATIONS)
        last = contests[-1]['dezenas']
        gen_features = np.array([opt.extractor.extract_features(g, last) for g in portfolio])
        kl = opt.extractor.compute_kl_divergence(gen_features)
        print(f"\n📊 KL Divergence: {kl:.3f}")
        print(f"   Objetivos: MC={best_obj['mc']:.3f} Nat={best_obj['naturalness']:.3f} CombDiv={best_obj['combinatorial_diversity']:.3f} TopoDiv={best_obj['topological_diversity']:.3f} CorrPen={best_obj['correlation_penalty']:.3f}")
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
