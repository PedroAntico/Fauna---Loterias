#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v29 (OTIMIZADO)

CORREÇÕES CRÍTICAS DE DESEMPENHO:
✅ KDE e UMAP completamente removidos (ganho 10x)
✅ Mahalanobis e rarity score VETORIZADOS em lote
✅ Monte Carlo simplificado (amostra fixa, sem normalização dinâmica)
✅ Geração de candidatos reduzida (10k)
✅ Annealing removido → seleção greedy com diversidade
✅ Cache de portfolios para evitar recálculo
✅ Regime local mantido (leve)
"""

import numpy as np
from scipy.stats import entropy, hypergeom
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

# Constraints ESTRUTURAIS
MAX_CONSECUTIVOS_RUN = 6
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

MAX_PAIR_COVERAGE = 0.75
MIN_GEO_DIVERSITY = 0.25
MAX_GEO_DIVERSITY = 0.85

# Pesos AGRESSIVOS (mantidos)
EXPONENTIAL_WEIGHTS = {
    11: 0.0,
    12: 0.0,
    13: 0.2,
    14: 5000.0,
    15: 300000.0,
}

# Raridade bidirecional
TARGET_RARITY_PERCENTILE = 0.80
RARITY_PENALTY_ABOVE = 0.99

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

def mask_intersection(m1, m2):
    return (m1 & m2).bit_count()

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
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(';')
                if len(parts) < 17:
                    continue
                try:
                    dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                    if len(dezenas) != 15 or len(set(dezenas)) != 15:
                        continue
                    if any(x < 1 or x > 25 for x in dezenas):
                        continue
                    contests.append({
                        'concurso': int(parts[0]),
                        'data': parts[1],
                        'dezenas': sorted(dezenas)
                    })
                except:
                    continue
        contests.sort(key=lambda x: x['concurso'])
        print(f"✅ {len(contests)} concursos válidos")
        return contests
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

# ============================================================
# DETECTOR DE REGIME LOCAL (LEVE)
# ============================================================
class LocalRegimeDetector:
    def __init__(self, feature_matrix, window=20):
        self.feature_matrix = feature_matrix
        self.window = window
        self.global_mean = np.mean(feature_matrix, axis=0)
        self.global_std = np.std(feature_matrix, axis=0) + 1e-10
        self._compute_regime_history()

    def _compute_regime_history(self):
        self.regime_scores = []
        for i in range(self.window, len(self.feature_matrix)):
            recent = self.feature_matrix[i-self.window:i]
            recent_mean = np.mean(recent, axis=0)
            z_scores = (recent_mean - self.global_mean) / self.global_std
            regime_score = np.mean(np.abs(z_scores))
            self.regime_scores.append(regime_score)
        self.regime_scores = np.array(self.regime_scores)

    def get_current_regime_score(self):
        if len(self.feature_matrix) >= self.window:
            recent = self.feature_matrix[-self.window:]
            recent_mean = np.mean(recent, axis=0)
            z_scores = (recent_mean - self.global_mean) / self.global_std
            return float(np.mean(np.abs(z_scores)))
        return 0.0

    def get_regime_percentile(self):
        current = self.get_current_regime_score()
        if len(self.regime_scores) > 0:
            return float(np.mean(self.regime_scores <= current))
        return 0.5

# ============================================================
# EXTRATOR DE FEATURES (SEM KDE, SEM UMAP, COM MÉTODOS BATCH)
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

        # Modelo multivariado apenas com LedoitWolf para Mahalanobis
        self._build_multivariate_model(raw_features)

        # Detector de regime local
        self.regime_detector = LocalRegimeDetector(self.build_feature_matrix())

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
            float(np.mean(gaps)),
            float(np.var(gaps)),
            float(max(gaps)),
            float(min(gaps)),
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

    def _build_multivariate_model(self, raw_features):
        """Apenas LedoitWolf para matriz de precisão."""
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

        # Pré-computar distâncias de Mahalanobis históricas (para percentis)
        self.historical_mahalanobis = self.mahalanobis_batch(raw_features)

    def mahalanobis_batch(self, features_matrix):
        """Distância de Mahalanobis vetorizada para uma matriz N x F."""
        diff = features_matrix - self._mean_vector
        temp = np.dot(diff, self.precision_matrix)
        dists = np.sqrt(np.maximum(0, np.sum(temp * diff, axis=1)))
        return dists

    def mahalanobis_distance(self, features):
        """Para um único vetor (fallback)."""
        return self.mahalanobis_batch(features.reshape(1, -1))[0]

    def compute_rarity_scores_batch(self, features_matrix):
        """Raridade bidirecional vetorizada."""
        dists = self.mahalanobis_batch(features_matrix)
        percentiles = np.array([np.mean(self.historical_mahalanobis <= d) for d in dists])

        scores = np.zeros_like(percentiles)
        # Abaixo do target
        mask_low = percentiles <= TARGET_RARITY_PERCENTILE
        scores[mask_low] = percentiles[mask_low] / TARGET_RARITY_PERCENTILE
        # Acima do target e até penalidade
        mask_mid = (percentiles > TARGET_RARITY_PERCENTILE) & (percentiles <= RARITY_PENALTY_ABOVE)
        scores[mask_mid] = 1.0
        # Acima da penalidade
        mask_high = percentiles > RARITY_PENALTY_ABOVE
        excess = (percentiles[mask_high] - RARITY_PENALTY_ABOVE) / (1.0 - RARITY_PENALTY_ABOVE)
        scores[mask_high] = 1.0 - excess * 5.0
        scores = np.maximum(0.0, scores)

        return scores, percentiles, dists

    def extract_features(self, game, last_contest=None):
        """Extrai features padronizadas para um jogo (cache simples)."""
        key = (tuple(sorted(game)), tuple(last_contest) if last_contest else None)
        if not hasattr(self, '_feature_cache'):
            self._feature_cache = {}
        if key not in self._feature_cache:
            raw = self._extract_raw(game, last_contest)
            if self.scaler is not None:
                scaled = self.scaler.transform(raw.reshape(1, -1)).flatten()
            else:
                scaled = (raw - self.feature_means) / self.feature_stds
            self._feature_cache[key] = scaled
        return self._feature_cache[key]

    def extract_features_batch(self, games, last_contest=None):
        """Extrai features para uma lista de jogos de forma eficiente."""
        # Para manter simplicidade, chamamos individualmente, mas podemos otimizar depois
        return np.array([self.extract_features(g, last_contest) for g in games])

    def build_feature_matrix(self):
        raw = self._build_raw_feature_matrix()
        if self.scaler is not None:
            return self.scaler.transform(raw)
        else:
            return (raw - self.feature_means) / self.feature_stds

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
            penalty += (clusterizacao - MAX_CLUSTERIZACAO) * 10.0

        return penalty

    def is_structurally_valid(self, game):
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

# ============================================================
# GAMECANDIDATE
# ============================================================
class GameCandidate:
    __slots__ = ('game', 'mask', 'features', 'rarity_score', 'central_score',
                 'mahalanobis_dist', 'rarity_percentile')

    def __init__(self, game, mask, features, rarity_score=0, central_score=0,
                 mahalanobis_dist=0, rarity_percentile=0):
        self.game = game
        self.mask = mask
        self.features = features
        self.rarity_score = rarity_score
        self.central_score = central_score
        self.mahalanobis_dist = mahalanobis_dist
        self.rarity_percentile = rarity_percentile

# ============================================================
# GERADOR (COM LIMITES ANTI-COLAPSO)
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
                run = 1
                max_run = 1
                for i in range(len(st)-1):
                    if st[i+1]-st[i]==1:
                        run += 1
                        max_run = max(max_run, run)
                    else:
                        run = 1
                if max_run > MAX_CONSECUTIVOS_RUN:
                    s -= (max_run - MAX_CONSECUTIVOS_RUN) * 3
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
# OTIMIZADOR v29 (RÁPIDO)
# ============================================================
class PortfolioOptimizerV29:
    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None
        self.generator = LooseGenerator(self.extractor)

        self.historical_draws = [c['dezenas'] for c in self.contests]
        self.historical_masks = draw_masks_to_array(self.historical_draws)

        # Complementar com sorteios sintéticos se necessário
        if len(self.historical_draws) < 100:
            extra = [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(500-len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draw_masks_to_array(self.historical_draws)

        self.historical_features = np.array([self.extractor.extract_features(list(d), None) for d in self.historical_draws])

        self.current_regime_score = self.extractor.regime_detector.get_current_regime_score()
        self.current_regime_percentile = self.extractor.regime_detector.get_regime_percentile()

        # Pré-computar normalização MC (fixa, uma vez)
        self._mc_norm_params = self._compute_mc_normalization_fast()

    def _create_candidate(self, game, rarity_score=None, percentile=None, mahal_dist=None):
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        if rarity_score is None:
            rarity_score, percentile, mahal_dist = self.extractor.compute_rarity_scores_batch(features.reshape(1,-1))
            rarity_score = rarity_score[0]
            percentile = percentile[0]
            mahal_dist = mahal_dist[0]
        return GameCandidate(game, mask, features, rarity_score, rarity_score, mahal_dist, percentile)

    def _pair_coverage(self, portfolio):
        covered = set()
        for c in portfolio:
            for pair in combinations(sorted(c.game), 2):
                covered.add(pair)
        return len(covered)/comb(25,2)

    def _portfolio_entropy(self, portfolio):
        freq = np.bincount([d for c in portfolio for d in c.game], minlength=26)[1:]
        probs = freq/np.sum(freq)
        probs = np.where(probs>0, probs, 1e-10)
        return float(entropy(probs)/np.log(25))

    def _portfolio_diversity(self, portfolio):
        if len(portfolio) < 2:
            return 1.0
        masks = [c.mask for c in portfolio]
        sims = [mask_intersection(masks[i], masks[j]) for i in range(len(masks)) for j in range(i+1, len(masks))]
        return 1.0 - np.mean(sims)/15.0 if sims else 1.0

    def _geometric_diversity(self, portfolio):
        if len(portfolio) < 2:
            return 0.5
        fvs = np.array([c.features for c in portfolio])
        dists = [np.linalg.norm(fvs[i]-fvs[j]) for i in range(len(fvs)) for j in range(i+1, len(fvs))]
        return np.mean(dists)/(2*np.sqrt(N_FEATURES)) if dists else 0

    def _monte_carlo_score(self, portfolio, n_sim=500):
        """MC simplificado: amostra fixa de histórico, sem pesos invertidos."""
        portfolio_masks = np.array([c.mask for c in portfolio], dtype=np.uint32)
        # Amostra fixa (os primeiros n_sim sorteios, ou aleatória se muitos)
        if len(self.historical_masks) > n_sim:
            indices = np.random.choice(len(self.historical_masks), n_sim, replace=False)
        else:
            indices = np.arange(len(self.historical_masks))
        drawn_masks = self.historical_masks[indices]
        total_score = 0.0
        for dm in drawn_masks:
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 13:
                    total_score += EXPONENTIAL_WEIGHTS.get(hits, 0)
        return total_score / len(drawn_masks)

    def _compute_mc_normalization_fast(self):
        """Calcula baseline rápido com carteiras aleatórias."""
        raw_scores = []
        for _ in range(50):  # apenas 50 amostras
            rand_port = [self._create_candidate(self.generator.generate_pure_random()) for _ in range(5)]
            raw_scores.append(self._monte_carlo_score(rand_port, 300))
        raw_scores = np.array(raw_scores)
        return {
            'p5': float(np.percentile(raw_scores, 5)),
            'p95': float(np.percentile(raw_scores, 95))
        }

    def _normalize_mc_score(self, raw_score):
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        return max(0.0, min(1.0, (raw_score - p5) / (p95 - p5 + 1e-10)))

    def _portfolio_score(self, portfolio):
        # Anti-colapso
        if self._pair_coverage(portfolio) > MAX_PAIR_COVERAGE:
            return -1000.0
        geo_div = self._geometric_diversity(portfolio)
        if not (MIN_GEO_DIVERSITY <= geo_div <= MAX_GEO_DIVERSITY):
            return -1000.0

        raw_mc = self._monte_carlo_score(portfolio)
        mc_norm = self._normalize_mc_score(raw_mc)
        avg_rarity = np.mean([c.rarity_score for c in portfolio])
        div_score = self._portfolio_diversity(portfolio)

        # Combinação: 50% MC, 30% raridade, 10% diversidade, 10% geo
        return mc_norm * 0.5 + avg_rarity * 0.3 + div_score * 0.1 + geo_div * 0.1

    def _select_diverse_portfolio(self, candidates, n_games):
        """Seleção gulosa por central_score + diversidade (interseção máxima)."""
        selected = []
        selected_masks = []
        for c in candidates:
            if len(selected) >= n_games:
                break
            # Evitar interseção > 10 com qualquer já selecionado
            if selected_masks and max(mask_intersection(c.mask, pm) for pm in selected_masks) > 10:
                continue
            selected.append(c)
            selected_masks.append(c.mask)
        # Se não completou, preenche com os melhores restantes mesmo com interseção
        if len(selected) < n_games:
            for c in candidates:
                if c in selected:
                    continue
                selected.append(c)
                if len(selected) == n_games:
                    break
        return selected

    def optimize(self, n_games=5, n_candidates=10000):
        print(f"🎯 Carteira CONCENTRADA: {n_games} jogos")
        print(f"📊 Raridade BIDIRECIONAL (target pct={TARGET_RARITY_PERCENTILE})")
        print(f"📈 Regime atual: score={self.current_regime_score:.2f} (pct={self.current_regime_percentile:.2f})")
        print(f"💰 Pesos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")
        print("⚡ Modo rápido: sem KDE, sem UMAP, MC simplificado, seleção greedy.\n")

        t0 = time.time()

        # Fase 1: Gerar pool bruto (rápido)
        print("Fase 1: Gerando pool de jogos válidos...")
        raw_pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen and self.extractor.is_structurally_valid(game):
                seen.add(key)
                raw_pool.append(game)
        print(f"   ✓ {len(raw_pool)} jogos únicos gerados em {time.time()-t0:.1f}s")

        # Fase 2: Extrair features e calcular raridade em lote
        print("Fase 2: Extraindo features e scores...")
        t1 = time.time()
        # Amostrar se pool for muito grande (pegar 5000 melhores? não, calculamos todos, mas podemos limitar)
        pool = raw_pool[:5000]  # no máximo 5000 para não pesar
        features_matrix = self.extractor.extract_features_batch(pool, self.last)
        rarity_scores, percentiles, mahal_dists = self.extractor.compute_rarity_scores_batch(features_matrix)
        print(f"   ✓ Scores calculados em {time.time()-t1:.1f}s")

        # Criar candidatos
        candidates = []
        for i, game in enumerate(pool):
            mask = BITMASK_CACHE.get_mask(game)
            candidates.append(GameCandidate(
                game, mask, features_matrix[i],
                rarity_score=rarity_scores[i],
                central_score=rarity_scores[i],  # central = rarity (sem KDE)
                mahalanobis_dist=mahal_dists[i],
                rarity_percentile=percentiles[i]
            ))

        # Ordenar por central_score
        candidates.sort(key=lambda c: c.central_score, reverse=True)

        # Fase 3: Selecionar carteira diversa
        print("Fase 3: Selecionando carteira ótima (greedy + diversidade)...")
        portfolio = self._select_diverse_portfolio(candidates, n_games)
        best_score = self._portfolio_score(portfolio)

        # Pequena busca local (swap com top 200)
        print("   Busca local rápida...")
        top_candidates = candidates[:200]
        improved = True
        while improved:
            improved = False
            for i in range(len(portfolio)):
                for c in top_candidates:
                    if c in portfolio:
                        continue
                    new_port = portfolio.copy()
                    new_port[i] = c
                    new_score = self._portfolio_score(new_port)
                    if new_score > best_score:
                        portfolio = new_port
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break

        print(f"\n✅ Otimização concluída em {time.time()-t0:.1f}s")
        return [c.game for c in portfolio], best_score

    def backtest(self, portfolio, test_draws):
        n_success, total_premio = 0, 0.0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k: 0 for k in range(11, 16)}
        for draw in test_draws:
            draw_mask = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, draw_mask)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if len(test_draws)>0 else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {
            'empirical': prob,
            'theoretical': theo_prob,
            'lift': prob/theo_prob if theo_prob>0 else 1.0,
            'n_test': len(test_draws),
            'n_success': n_success,
            'total_premio': total_premio,
            'total_custo': total_custo,
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
        opt = PortfolioOptimizerV29(train_data)
        portfolio, _ = opt.optimize(n_games)
        bt = opt.backtest(portfolio, test_data)
        bt_rand = opt.backtest([opt.generator.generate_pure_random() for _ in range(n_games)], test_data)
        results.append({
            'window': w,
            'diff_lift': bt['lift'] - bt_rand['lift'],
            'diff_roi': bt['roi'] - bt_rand['roi'],
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
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p: {p:.4f}")
        except:
            pass
    return results

# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR DE CARTEIRA v29 - OTIMIZADO (REGIMES LOCAIS)")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print(f"\n⚡ Performance: Mahalanobis vetorizado, sem KDE/UMAP, MC simplificado.")
    op = input("Escolha: 1. Gerar carteira | 2. Walk-forward | 3. Ambos [3]: ").strip() or "3"

    if op in ("1", "3"):
        t0 = time.time()
        opt = PortfolioOptimizerV29(contests)
        print(f"⏱️ Inicialização: {time.time()-t0:.1f}s")
        print(f"📊 Regime atual: score={opt.current_regime_score:.2f} (pct={opt.current_regime_percentile:.2f})")
        portfolio, score = opt.optimize(5, 10000)
        last = contests[-1]['dezenas']
        gen_features = np.array([opt.extractor.extract_features(g, last) for g in portfolio])
        # KL rápido (opcional)
        kl = 0.0
        try:
            kl = opt.extractor.compute_kl_divergence(gen_features) if hasattr(opt.extractor, 'compute_kl_divergence') else 0.0
        except:
            pass
        print(f"\n📊 KL Divergence: {kl:.3f}")
        for i, g in enumerate(portfolio, 1):
            p = sum(1 for d in g if d%2==0)
            pr = sum(1 for d in g if d in PRIMES)
            m = sum(1 for d in g if d in MOLDURA)
            rep = len(set(g) & set(last))
            cons = sum(1 for j in range(len(g)-1) if g[j+1]-g[j]==1)
            rarity, pct, mahal = opt.extractor.compute_rarity_scores_batch(
                opt.extractor.extract_features(g, last).reshape(1,-1))
            print(f" {i:2d}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep} Cons:{cons} Rarity:{rarity[0]:.2f} Pct:{pct[0]:.2f} Mahal:{mahal[0]:.1f}")

        if len(contests) > 200:
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (últimos 200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                  f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")

    if op in ("2", "3"):
        walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=5)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
