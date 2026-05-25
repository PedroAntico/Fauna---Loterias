#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v24 (v20.5 CONSOLIDADO)
======================================================================
FILOSOFIA: Agressividade do v20 + Infraestrutura do v23

✅ Pesos agressivos restaurados (13:5, 14:3000, 15:150000)
✅ Centralidade simples: structural*0.60 + temporal*0.40
✅ Diversidade funcional: 40% centrais + 35% cobertura + 25% temporais
✅ MAX_PAIR_COVERAGE = 0.85 (mantido)
✅ BitmaskCache + GameCandidate (pré-computação)
✅ Percentile rank (robustez não-paramétrica)
✅ MC híbrido com mistura de concursos
✅ Vetorização de distâncias no MC
✅ Score Bayesiano LEVE (apenas estabilidade numérica)

REMOVIDO do v23:
❌ Anti-frequência (ANTI_FREQ_WEIGHT = 0)
❌ DPP (volta greedy coverage)
❌ Markov temporal (removido)
❌ Portfólio adversarial 60/40 (volta 40/35/25)
❌ Grafo de coocorrência (removido)
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

FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
    "compressao",
]

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

MAX_PAIR_COVERAGE = 0.85
MIN_GEO_DIVERSITY = 0.50
MAX_GEO_DIVERSITY = 0.75

# Pesos AGRESSIVOS restaurados do v20
EXPONENTIAL_WEIGHTS = {
    11: 0.0,
    12: 0.0,
    13: 5.0,
    14: 3000.0,
    15: 150000.0,
}

# Score Bayesiano LEVE (apenas estabilidade)
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 50.0

# Features temporais com percentil rank
TEMPORAL_FEATURES = ['moldura', 'amplitude', 'energia_jogo', 'densidade_local', 'clusterizacao']
TEMPORAL_INDICES = {
    'moldura': 14, 'amplitude': 16, 'energia_jogo': 4,
    'densidade_local': 8, 'clusterizacao': 10
}
TEMPORAL_WEIGHTS = {k: 0.20 for k in TEMPORAL_FEATURES}


# ============================================================
# UTILITÁRIOS BITMASK (COM CACHE)
# ============================================================
class BitmaskCache:
    """Cache global de bitmasks para evitar recálculo."""
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

    def intersection(self, game1, game2):
        m1 = self.get_mask(game1)
        m2 = self.get_mask(game2)
        return (m1 & m2).bit_count()


BITMASK_CACHE = BitmaskCache()


def mask_intersection(m1, m2):
    """Interseção rápida entre duas máscaras."""
    return (m1 & m2).bit_count()


def draw_masks_to_array(draws):
    """Converte lista de draws para array numpy de máscaras."""
    return np.array([BITMASK_CACHE.get_mask(d) for d in draws], dtype=np.uint32)


# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    """Carrega todos os concursos do arquivo CSV."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo não encontrado: {csv_path}")
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
                except (ValueError, IndexError):
                    continue
        contests.sort(key=lambda x: x['concurso'])
        print(f"✅ {len(contests)} concursos válidos")
        return contests
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


# ============================================================
# EXTRATOR DE FEATURES (SIMPLIFICADO, SEM ANTI-FREQ)
# ============================================================
class FeatureExtractor:
    """Extrai features topológicas com cache e percentil rank."""

    def __init__(self, contests):
        self.contests = contests
        self._repeat_history = []
        for i, c in enumerate(contests):
            if i > 0:
                prev = set(contests[i-1]['dezenas'])
                self._repeat_history.append(len(prev & set(c['dezenas'])))
            else:
                self._repeat_history.append(0)

        self._recent_freq = self._compute_recent_freq()
        raw_features = self._build_raw_feature_matrix()

        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        if self.scaler is not None and len(raw_features) > 10:
            self.scaler.fit(raw_features)
        self.feature_means = np.mean(raw_features, axis=0)
        self.feature_stds = np.std(raw_features, axis=0) + 1e-10

        self._feature_cache = {}
        self._cache_max_size = 100000

        # Referências para percentil rank
        self._percentile_refs = {}
        for idx in TEMPORAL_INDICES.values():
            self._percentile_refs[idx] = np.sort(raw_features[:, idx])

    def _compute_recent_freq(self, window=50):
        """Frequência das dezenas nos últimos N concursos."""
        freq = Counter()
        start = max(0, len(self.contests) - window)
        for c in self.contests[start:]:
            freq.update(c['dezenas'])
        total = len(self.contests[start:])
        return {d: freq.get(d, 0) / total for d in range(1, 26)}

    def _build_raw_feature_matrix(self):
        """Constrói matriz de features brutas."""
        features_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            features_list.append(self._extract_raw(c['dezenas'], last))
        return np.array(features_list, dtype=np.float64)

    def _extract_raw(self, dezenas, last_contest=None):
        """Extrai features brutas de um conjunto de dezenas."""
        d = sorted(dezenas)
        gaps = [d[i+1] - d[i] for i in range(len(d)-1)]
        rep = len(set(d) & set(last_contest)) if last_contest else 8

        ent_trans = 0.0
        if len(self._repeat_history) >= 5:
            trans = [self._repeat_history[i+1] - self._repeat_history[i]
                     for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v, 0) / len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs > 0, probs, 1e-10)))

        amplitude = max(d) - min(d)
        std_pos = np.std(d) if len(d) > 1 else 0.0
        compressao = std_pos / amplitude if amplitude > 0 else 0.5

        return np.array([
            float(np.mean(gaps)),
            float(np.var(gaps)),
            float(max(gaps)),
            float(min(gaps)),
            float(sum(abs(d[i] - d[i-1]) for i in range(1, len(d)))),
            ent_trans,
            float(len(set((x-1)//5 for x in d))),
            float(sum(1 for i in range(len(d)-1) if d[i+1] - d[i] == 1)),
            float(np.mean([sum(1 for y in d if abs(x-y) <= 2) for x in d]) / 15),
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
        """Extrai features com cache."""
        key = (tuple(sorted(game)), tuple(last_contest) if last_contest else None)
        if key not in self._feature_cache:
            raw = self._extract_raw(game, last_contest)
            if self.scaler is not None:
                self._feature_cache[key] = self.scaler.transform(raw.reshape(1, -1)).flatten()
            else:
                self._feature_cache[key] = (raw - self.feature_means) / self.feature_stds
            # Limitar cache
            if len(self._feature_cache) > self._cache_max_size:
                self._feature_cache.clear()
        return self._feature_cache[key]

    def build_feature_matrix(self):
        """Constrói matriz de features padronizadas."""
        raw = self._build_raw_feature_matrix()
        if self.scaler is not None:
            return self.scaler.transform(raw)
        return (raw - self.feature_means) / self.feature_stds

    def compute_structural_penalty(self, game):
        """Calcula penalidade estrutural (SEM anti-frequência)."""
        d = sorted(game)
        penalty = 0.0

        actuals = {
            'pares': sum(1 for x in d if x % 2 == 0),
            'primos': sum(1 for x in d if x in PRIMES),
            'moldura': sum(1 for x in d if x in MOLDURA),
            'repeticoes': len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8,
            'soma': sum(d),
            'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1] - d[i] == 1),
            'amplitude': max(d) - min(d),
        }

        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name] - target)
                if dev > tol:
                    penalty += (dev - tol) * w

        return penalty

    def is_structurally_valid(self, game):
        """Verifica se o jogo é estruturalmente válido."""
        return self.compute_structural_penalty(game) < STRUCTURAL_REJECT_THRESHOLD

    def compute_structural_score(self, game):
        """Score estrutural (0-1, maior = mais típico)."""
        penalty = self.compute_structural_penalty(game)
        return np.exp(-penalty / 3.0)

    def compute_temporal_score_percentile(self, game, all_features, recent_window=20):
        """Score temporal usando PERCENTILE RANK (robusto para features não-gaussianas)."""
        game_feats = self.extract_features(game, None)
        score = 0.0
        total_w = 0.0

        for name, idx in TEMPORAL_INDICES.items():
            val = game_feats[idx]
            recent = (all_features[-recent_window:, idx]
                      if len(all_features) >= recent_window
                      else all_features[:, idx])
            percentile = np.mean(recent <= val)
            score += TEMPORAL_WEIGHTS[name] * (1.0 - abs(percentile - 0.5) * 2)
            total_w += TEMPORAL_WEIGHTS[name]

        return score / total_w if total_w > 0 else 0.5


# ============================================================
# GAMECANDIDATE (OBJETO PRÉ-COMPUTADO)
# ============================================================
class GameCandidate:
    """Objeto que encapsula todas as pré-computações de um jogo."""
    __slots__ = ('game', 'mask', 'features', 'structural_score',
                 'central_score', 'temporal_score')

    def __init__(self, game, mask, features, structural_score,
                 central_score=0, temporal_score=0):
        self.game = game
        self.mask = mask
        self.features = features
        self.structural_score = structural_score
        self.central_score = central_score
        self.temporal_score = temporal_score


# ============================================================
# GERADOR RÁPIDO (FASE 1)
# ============================================================
class FastGenerator:
    """Gerador rápido com viés de repetição e distribuição de dezenas."""

    def __init__(self, last_contest=None, extractor=None):
        self.last = set(last_contest) if last_contest else None
        self.extractor = extractor

    def generate_one(self):
        """Gera um jogo válido."""
        for _ in range(50):
            game = self._generate_raw()
            if self.extractor is not None and self.extractor.is_structurally_valid(game):
                return game
        return self._generate_raw()

    def _generate_raw(self):
        """Geração bruta com heurísticas leves."""
        game = set()
        available = set(range(1, 26))

        # Viés de repetição leve
        if self.last and random.random() < 0.3:
            rep_pool = list(self.last & available)
            if rep_pool:
                n = random.randint(5, 10)
                game.update(random.sample(rep_pool, min(n, len(rep_pool))))
                available -= game

        # Completar com diversidade espacial
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                test = game | {d}
                s = len(set((x-1)//5 for x in test)) * 3
                st = sorted(test)
                cons = sum(1 for i in range(len(st)-1) if st[i+1] - st[i] == 1)
                if cons > 6:
                    s -= (cons - 6) * 1.5
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
        """Gera jogo puramente aleatório (baseline)."""
        return sorted(np.random.choice(range(1, 26), 15, replace=False))


# ============================================================
# OTIMIZADOR DE CARTEIRA v24 (AGRESSIVO + OTIMIZADO)
# ============================================================
class PortfolioOptimizerV24:
    """
    Otimizador v24: Agressividade do v20 + Infraestrutura do v23.

    Pipeline 2 fases:
    Fase 1: Geração barata (50k candidatos)
    Fase 2: Avaliação profunda (top 5k) + Simulated Annealing
    """

    def __init__(self, contests):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.last = contests[-1]['dezenas'] if contests else None

        self.generator = FastGenerator(self.last, self.extractor)

        # Cache e normalização
        self._mc_cache = {}
        self._mc_norm_params = None

        # Dados históricos como bitmasks e features
        self.historical_draws = [c['dezenas'] for c in self.contests]
        self.historical_masks = draw_masks_to_array(self.historical_draws)
        if len(self.historical_draws) < 100:
            extra = [sorted(np.random.choice(range(1, 26), 15, replace=False))
                     for _ in range(500 - len(self.historical_draws))]
            self.historical_draws.extend(extra)
            self.historical_masks = draw_masks_to_array(self.historical_draws)

        # Pré-computar features históricas (MC otimizado)
        self.historical_features = np.array([
            self.extractor.extract_features(list(d), None)
            for d in self.historical_draws
        ])

    # ============================================================
    # CRIAÇÃO DE CANDIDATOS
    # ============================================================
    def _create_candidate(self, game):
        """Cria GameCandidate pré-computado."""
        mask = BITMASK_CACHE.get_mask(game)
        features = self.extractor.extract_features(game, self.last)
        structural = self.extractor.compute_structural_score(game)
        temporal = self.extractor.compute_temporal_score_percentile(game, self.feature_matrix)
        # Centralidade SIMPLES (v20): structural*0.60 + temporal*0.40
        central = structural * 0.60 + temporal * 0.40
        return GameCandidate(game, mask, features, structural, central, temporal)

    # ============================================================
    # MÉTRICAS DE CARTEIRA
    # ============================================================
    def _pair_coverage(self, portfolio):
        """Cobertura de pares de dezenas."""
        covered = set()
        for c in portfolio:
            for pair in combinations(sorted(c.game), 2):
                covered.add(pair)
        return len(covered) / comb(25, 2)

    def _portfolio_entropy(self, portfolio):
        """Entropia da distribuição de dezenas na carteira."""
        freq = np.bincount([d for c in portfolio for d in c.game], minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        return float(entropy(probs) / np.log(25))

    def _portfolio_diversity(self, portfolio):
        """Diversidade baseada em interseção de dezenas."""
        if len(portfolio) < 2:
            return 1.0
        masks = [c.mask for c in portfolio]
        sims = [mask_intersection(masks[i], masks[j])
                for i in range(len(masks)) for j in range(i+1, len(masks))]
        return 1.0 - np.mean(sims) / 15.0 if sims else 1.0

    def _geometric_diversity(self, portfolio):
        """Diversidade geométrica (distância entre features)."""
        if len(portfolio) < 2:
            return 0.5
        fvs = np.array([c.features for c in portfolio])
        dists = [np.linalg.norm(fvs[i] - fvs[j])
                 for i in range(len(fvs)) for j in range(i+1, len(fvs))]
        return np.mean(dists) / (2 * np.sqrt(len(FEATURE_NAMES))) if dists else 0

    def _average_structural_score(self, portfolio):
        """Score estrutural médio da carteira."""
        return np.mean([c.structural_score for c in portfolio]) if portfolio else 0.5

    # ============================================================
    # MONTE CARLO HÍBRIDO
    # ============================================================
    def _generate_synthetic_draws(self, n_synthetic):
        """Gera draws sintéticos por mistura de 2 concursos reais."""
        synthetic = []
        for _ in range(n_synthetic):
            d1 = random.choice(self.historical_draws)
            d2 = random.choice(self.historical_draws)
            mix = set(random.sample(d1, random.randint(7, 8)))
            mix |= set(random.sample(d2, random.randint(7, 8)))
            available = set(range(1, 26)) - mix
            while len(mix) < 15 and available:
                mix.add(random.choice(list(available)))
                available = set(range(1, 26)) - mix
            synthetic.append(sorted(mix)[:15])
        return synthetic

    def _bayesian_score(self, successes, trials):
        """Score Bayesiano Beta-Binomial LEVE."""
        alpha_post = BETA_PRIOR_ALPHA + successes
        beta_post = BETA_PRIOR_BETA + (trials - successes)
        return alpha_post / (alpha_post + beta_post)

    def _monte_carlo_hybrid(self, portfolio_candidates, n_simulations=500):
        """
        Monte Carlo Híbrido: 70% histórico + 30% sintético.
        Vetorizado com numpy para performance.
        """
        cache_key = tuple(tuple(sorted(c.game)) for c in portfolio_candidates)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        recent_f = self.feature_matrix[-20:] if len(self.feature_matrix) >= 20 else self.feature_matrix

        # Vetorização de distâncias
        dists = np.linalg.norm(
            self.historical_features[:, None, :] - recent_f[None, :, :],
            axis=2
        )
        avg_dists = np.mean(dists, axis=1)
        weights = np.exp(-avg_dists / 2.0)
        weights /= weights.sum()

        n_hist = int(n_simulations * 0.7)
        n_synth = n_simulations - n_hist

        hist_indices = np.random.choice(len(self.historical_masks), size=n_hist, p=weights)
        hist_masks = self.historical_masks[hist_indices]
        synthetic_draws = self._generate_synthetic_draws(n_synth)
        synth_masks = draw_masks_to_array(synthetic_draws)

        all_masks = np.concatenate([hist_masks, synth_masks])
        portfolio_masks = np.array([c.mask for c in portfolio_candidates], dtype=np.uint32)

        # Vetorização dos hits
        total_weighted_score = 0.0
        for j in range(len(all_masks)):
            dm = all_masks[j]
            for i, pm in enumerate(portfolio_masks):
                hits = mask_intersection(pm, dm)
                if hits >= 13:  # Apenas 13+ têm peso positivo
                    total_weighted_score += EXPONENTIAL_WEIGHTS.get(hits, 0)

        avg_score = total_weighted_score / len(all_masks)

        # Score Bayesiano para estabilidade numérica
        max_possible = EXPONENTIAL_WEIGHTS[15] * len(portfolio_candidates)
        success_rate = avg_score / max_possible if max_possible > 0 else 0
        bayesian_score = self._bayesian_score(
            successes=success_rate * len(portfolio_candidates) * n_simulations,
            trials=len(portfolio_candidates) * n_simulations
        )

        # Normalização empírica
        if self._mc_norm_params is None:
            self._mc_norm_params = self._compute_mc_normalization(
                portfolio_size=len(portfolio_candidates)
            )
        p5, p95 = self._mc_norm_params['p5'], self._mc_norm_params['p95']
        normalized = max(0.0, min(1.0, (bayesian_score - p5) / (p95 - p5 + 1e-10)))

        self._mc_cache[cache_key] = normalized
        return normalized

    def _compute_mc_normalization(self, portfolio_size=10, n_samples=200):
        """Pré-computa parâmetros de normalização empírica."""
        raw_scores = []
        for _ in range(n_samples):
            rand_port = [self._create_candidate(self.generator.generate_pure_random())
                         for _ in range(portfolio_size)]
            raw = self._monte_carlo_hybrid_raw(rand_port, 300)
            raw_scores.append(raw)
        raw_scores = np.array(raw_scores)
        return {
            'p5': float(np.percentile(raw_scores, 5)),
            'p95': float(np.percentile(raw_scores, 95))
        }

    def _monte_carlo_hybrid_raw(self, portfolio_candidates, n_simulations=300):
        """MC bruto para normalização."""
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

    # ============================================================
    # SELEÇÃO DE CARTEIRA
    # ============================================================
    def _greedy_marginal_coverage(self, pool, existing, n_select, max_inter=7):
        """Greedy coverage: seleciona jogos que adicionam mais pares inéditos."""
        existing_masks = [c.mask for c in existing] if existing else []
        covered_pairs = set()
        for c in existing:
            for pair in combinations(sorted(c.game), 2):
                covered_pairs.add(pair)

        selected = list(existing)
        selected_set = set(tuple(c.game) for c in selected)
        remaining = [c for c in pool if tuple(c.game) not in selected_set]

        for _ in range(n_select):
            if not remaining:
                break

            cov = len(covered_pairs) / comb(25, 2)
            cw, sw = (0.0, 1.0) if cov >= MAX_PAIR_COVERAGE else (0.3, 0.7)

            best_c, best_score = None, -float('inf')
            for c in random.sample(remaining, min(300, len(remaining))):
                if existing_masks and max(mask_intersection(c.mask, em) for em in existing_masks) > max_inter:
                    continue
                if self.extractor.compute_structural_penalty(c.game) > STRUCTURAL_REJECT_THRESHOLD:
                    continue

                new_pairs = set(combinations(sorted(c.game), 2)) - covered_pairs
                combined = len(new_pairs) / 105.0 * cw + c.structural_score * sw
                if combined > best_score:
                    best_score, best_c = combined, c

            if best_c:
                selected.append(best_c)
                selected_set.add(tuple(best_c.game))
                remaining.remove(best_c)
                existing_masks.append(best_c.mask)
                for pair in combinations(sorted(best_c.game), 2):
                    covered_pairs.add(pair)

        return selected

    def _repair_portfolio(self, portfolio, pool):
        """Repair: substitui jogos problemáticos por similares válidos."""
        repaired = list(portfolio)
        for _ in range(20):
            pair_cov = self._pair_coverage(repaired)
            geo_div = self._geometric_diversity(repaired)

            if pair_cov <= MAX_PAIR_COVERAGE and MIN_GEO_DIVERSITY <= geo_div <= MAX_GEO_DIVERSITY:
                break

            if pair_cov > MAX_PAIR_COVERAGE:
                contributions = []
                for c in repaired:
                    other_pairs = set()
                    for c2 in repaired:
                        if c2 != c:
                            for pair in combinations(sorted(c2.game), 2):
                                other_pairs.add(pair)
                    game_pairs = set(combinations(sorted(c.game), 2))
                    contributions.append(len(game_pairs - other_pairs))
                old = repaired[np.argmax(contributions)]
            else:
                fvs = np.array([c.features for c in repaired])
                centroid = np.mean(fvs, axis=0)
                distances = [np.linalg.norm(fv - centroid) for fv in fvs]
                old = repaired[np.argmax(distances) if geo_div > MAX_GEO_DIVERSITY else np.argmin(distances)]

            # Encontrar substituto similar
            best_c, best_sim = None, -float('inf')
            old_features = old.features
            for c in random.sample(pool, min(500, len(pool))):
                if c == old:
                    continue
                dot = np.dot(old_features, c.features)
                norm = np.linalg.norm(old_features) * np.linalg.norm(c.features)
                sim = dot / norm if norm > 1e-10 else 1.0
                if 0.3 <= sim <= 0.7:
                    best_c = c
                    break
                if sim > best_sim:
                    best_sim, best_c = sim, c

            if best_c:
                repaired[repaired.index(old)] = best_c

        return repaired

    # ============================================================
    # SCORE DA CARTEIRA
    # ============================================================
    def _portfolio_score(self, portfolio):
        """Score multiobjetivo da carteira."""
        if self._pair_coverage(portfolio) > MAX_PAIR_COVERAGE:
            return -1000.0
        if not (MIN_GEO_DIVERSITY <= self._geometric_diversity(portfolio) <= MAX_GEO_DIVERSITY):
            return -1000.0

        mc_score = self._monte_carlo_hybrid(portfolio)
        structural = self._average_structural_score(portfolio)

        return (mc_score * 0.50 + structural * 0.20 +
                self._portfolio_diversity(portfolio) * 0.20 +
                self._geometric_diversity(portfolio) * 0.10)

    # ============================================================
    # MUTAÇÃO
    # ============================================================
    def _mutate_candidate(self, candidate):
        """Mutação local: troca 1-3 dezenas."""
        for _ in range(20):
            mutated = list(candidate.game)
            for _ in range(random.randint(1, 3)):
                pos = random.randint(0, 14)
                avail = [d for d in range(1, 26) if d not in mutated]
                if avail:
                    mutated[pos] = random.choice(avail)
            mutated = sorted(mutated)[:15]
            if self.extractor.is_structurally_valid(mutated):
                return self._create_candidate(mutated)
        return candidate

    # ============================================================
    # OTIMIZAÇÃO PRINCIPAL
    # ============================================================
    def optimize(self, n_games=10, n_candidates=50000, iterations=100):
        """
        Pipeline completo de otimização.

        Diversidade funcional: 40% centrais + 35% cobertura + 25% temporais
        """
        n_central = max(1, int(n_games * 0.40))
        n_coverage = max(1, int(n_games * 0.35))
        n_temporal = n_games - n_central - n_coverage

        print(f"   Pipeline: {n_candidates//1000}k candidatos")
        print(f"   Diversidade funcional: {n_central} centrais + {n_coverage} cobertura + {n_temporal} temporais")
        print(f"   Pesos agressivos: 13:{EXPONENTIAL_WEIGHTS[13]} 14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")
        print(f"   Centralidade: structural*0.60 + temporal*0.40")

        # FASE 1: Geração barata
        raw_pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Fase 1"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen and self.extractor.is_structurally_valid(game):
                seen.add(key)
                raw_pool.append(game)

        # FASE 2: Pré-computação completa no top 5000
        top_pool = random.sample(raw_pool, min(5000, len(raw_pool)))
        candidates = []
        for g in tqdm(top_pool, desc="Fase 2"):
            candidates.append(self._create_candidate(g))

        # Ordenar por score central
        candidates.sort(key=lambda c: c.central_score, reverse=True)

        # Selecionar CENTRAIS (alta densidade, diversos)
        central_candidates = []
        central_masks = []
        for c in candidates:
            if len(central_candidates) >= n_central:
                break
            if central_masks and max(mask_intersection(c.mask, cm) for cm in central_masks) > 8:
                continue
            central_candidates.append(c)
            central_masks.append(c.mask)

        # Selecionar COBERTURA via greedy marginal
        coverage_candidates = self._greedy_marginal_coverage(
            [c for c in candidates if c not in central_candidates],
            central_candidates,
            n_coverage
        )
        coverage_candidates = coverage_candidates[len(central_candidates):]

        # Selecionar TEMPORAIS (ordenados por score temporal)
        candidates.sort(key=lambda c: c.temporal_score, reverse=True)
        temporal_candidates = []
        all_existing = central_candidates + coverage_candidates
        for c in candidates:
            if c in all_existing:
                continue
            if len(temporal_candidates) >= n_temporal:
                break
            if any(mask_intersection(c.mask, ec.mask) > 9 for ec in all_existing + temporal_candidates):
                continue
            temporal_candidates.append(c)

        # Combinar carteira inicial
        portfolio = central_candidates + coverage_candidates + temporal_candidates
        portfolio = self._repair_portfolio(portfolio, candidates)

        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)

        # FASE 3: Simulated Annealing
        elite_pool = candidates[:len(candidates)//4]
        for it in tqdm(range(iterations), desc="Annealing"):
            temp = 1.0 * (0.95 ** it)
            new_portfolio = list(portfolio)
            idx = random.randint(0, len(new_portfolio) - 1)

            if random.random() < 0.4 and elite_pool:
                new_candidate = random.choice(elite_pool)
            elif random.random() < 0.7:
                new_candidate = self._mutate_candidate(new_portfolio[idx])
            else:
                new_candidate = self._create_candidate(self.generator.generate_one())

            # Verificar interseção
            too_similar = any(
                j != idx and mask_intersection(new_candidate.mask, c.mask) > 8
                for j, c in enumerate(new_portfolio)
            )
            if too_similar:
                continue

            new_portfolio[idx] = new_candidate
            new_score = self._portfolio_score(new_portfolio)

            if new_score > best_score:
                best_portfolio, best_score = list(new_portfolio), new_score
            elif random.random() < np.exp((new_score - self._portfolio_score(portfolio)) / max(0.01, temp)):
                portfolio = new_portfolio

        return [c.game for c in best_portfolio], best_score

    # ============================================================
    # BACKTEST
    # ============================================================
    def backtest(self, portfolio, test_draws):
        """Backtest: contabiliza >=11 para o usuário final."""
        n_success, total_premio = 0, 0.0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)

        for draw in test_draws:
            draw_mask = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, draw_mask)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)

        prob = n_success / (len(portfolio) * len(test_draws)) if len(test_draws) > 0 else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11, 16))
        theo_prob = 1 - (1 - p_single) ** len(portfolio)

        return {
            'empirical': prob,
            'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws),
            'n_success': n_success,
            'total_premio': total_premio,
            'total_custo': total_custo,
            'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0
        }


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=10):
    """Validação walk-forward com múltiplas janelas."""
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

        opt = PortfolioOptimizerV24(train_data)
        portfolio, _ = opt.optimize(n_games, n_candidates=50000, iterations=50)

        bt = opt.backtest(portfolio, test_data)
        bt_rand = opt.backtest(
            [opt.generator.generate_pure_random() for _ in range(n_games)],
            test_data
        )

        results.append({
            'window': w,
            'diff_lift': bt['lift'] - bt_rand['lift'],
            'diff_roi': bt['roi'] - bt_rand['roi']
        })
        print(f" Janela {w}: diff_lift={bt['lift']-bt_rand['lift']:+.3f} "
              f"diff_ROI={bt['roi']-bt_rand['roi']:+.1f}%")

    if results:
        diffs = [r['diff_lift'] for r in results]
        print(f"\n📊 Média diff lift: {np.mean(diffs):+.3f} | "
              f"Janelas +: {sum(1 for d in diffs if d > 0)}/{len(results)}")
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p: {p:.4f}")
        except Exception:
            pass

    return results


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("=" * 70)
    print("🧬 GERADOR DE CARTEIRA v24 - AGRESSIVO + OTIMIZADO")
    print("=" * 70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")
    print(f"\n📊 Configuração:")
    print(f"   Pesos agressivos: 13:{EXPONENTIAL_WEIGHTS[13]} "
          f"14:{EXPONENTIAL_WEIGHTS[14]} 15:{EXPONENTIAL_WEIGHTS[15]}")
    print(f"   Centralidade: structural*0.60 + temporal*0.40")
    print(f"   Diversidade funcional: 40% centrais + 35% cobertura + 25% temporais")
    print(f"   MAX_PAIR_COVERAGE: {MAX_PAIR_COVERAGE}")
    print(f"   MC Híbrido: 70% real + 30% misto | Score Bayesiano LEVE")

    print("\nOpções: 1. Gerar carteira | 2. Walk-forward | 3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        t0 = time.time()
        opt = PortfolioOptimizerV24(contests)
        print(f"   ✅ Init {time.time()-t0:.1f}s")

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
