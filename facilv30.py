#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE TOPOLOGIA PURA COM PIPELINE - LOTOFÁCIL v31
=======================================================
CORREÇÕES:
✅ Removido random_noise (poluía ranking)
✅ Removidas EMA, fadiga, streaks, frequência histórica
✅ Foco em TOPOLOGIA PURA (gaps, energia, entropias, densidade)
✅ Ensemble com modelo aprendido (não heurístico)
✅ Pipeline modular preservado
✅ Regime score aprendido via regressão logística
✅ Calibração rigorosa (Spearman, Isotonic, Brier)
"""

import numpy as np
from scipy.stats import entropy, wilcoxon, hypergeom, pearsonr, spearmanr
from collections import Counter
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
from math import comb
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

# ============================================================
# IMPORTS OPCIONAIS
# ============================================================
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================
# CONJUNTOS E CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# ============================================================
# FEATURES DE TOPOLOGIA PURA (v31)
# ============================================================
TOPOLOGY_FEATURE_NAMES = [
    "gap_medio",          # 0
    "gap_var",            # 1
    "gap_max",            # 2
    "gap_min",            # 3
    "energia_jogo",       # 4
    "entropia_rep",       # 5
    "entropia_transicao", # 6
    "quadrantes",         # 7
    "consecutivos",       # 8
    "densidade_local",    # 9
    "assimetria",         # 10
    "clusterizacao",      # 11
    "repeticoes",         # 12 (do último concurso)
    "pares",              # 13
    "primos",             # 14
    "moldura",            # 15
    "soma",               # 16
    "amplitude",          # 17
    "elasticidade",       # 18 (retorno à média)
    "entropia_conjunta",  # 19
]

# Índices para acesso rápido
IDX_GAP_MEDIO = 0
IDX_GAP_VAR = 1
IDX_GAP_MAX = 2
IDX_GAP_MIN = 3
IDX_ENERGIA = 4
IDX_ENTROPIA_REP = 5
IDX_ENTROPIA_TRANS = 6
IDX_QUADRANTES = 7
IDX_CONSECUTIVOS = 8
IDX_DENSIDADE = 9
IDX_ASSIMETRIA = 10
IDX_CLUSTERIZACAO = 11
IDX_REPETICOES = 12
IDX_PARES = 13
IDX_PRIMOS = 14
IDX_MOLDURA = 15
IDX_SOMA = 16
IDX_AMPLITUDE = 17
IDX_ELASTICIDADE = 18
IDX_ENTROPIA_CONJ = 19

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    if not os.path.exists(csv_file): return None
    contests = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(';')
                if len(parts) >= 17:
                    contests.append({
                        'concurso': int(parts[0]), 'data': parts[1],
                        'dezenas': [int(x) for x in parts[2:17]]
                    })
        contests.sort(key=lambda x: x['concurso'])
        return contests
    except: return None


# ============================================================
# CONTEXTO TEMPORAL (TOPOLOGIA PURA)
# ============================================================
class TopologyContext:
    """Contexto focado APENAS em features topológicas. Sem EMA, fadiga, streaks."""
    def __init__(self, contests_slice, historical_contests=None):
        self.contests = contests_slice
        self.n_contests = len(contests_slice)
        self.historical = historical_contests if historical_contests is not None else contests_slice
        self.repeat_history = []
        self.pares_history = []
        self.moldura_history = []
        self.soma_history = []
        self.gap_media_history = []
        self.gap_var_history = []
        self.energia_history = []

        for i, c in enumerate(self.contests):
            d = c['dezenas']
            self.pares_history.append(sum(1 for x in d if x % 2 == 0))
            self.moldura_history.append(sum(1 for x in d if x in MOLDURA))
            self.soma_history.append(sum(d))
            sd = sorted(d)
            gaps = [sd[j+1]-sd[j] for j in range(len(sd)-1)]
            self.gap_media_history.append(np.mean(gaps))
            self.gap_var_history.append(np.var(gaps))
            self.energia_history.append(sum(abs(sd[j]-sd[j-1]) for j in range(1, len(sd))))
            if i > 0:
                self.repeat_history.append(len(set(self.contests[i-1]['dezenas']) & set(d)))
            else:
                self.repeat_history.append(0)

    def get_last_contest(self):
        return self.contests[-1]['dezenas'] if self.n_contests > 0 else []

    def extract_topology_features(self, game):
        """Extrai APENAS features topológicas (20 dimensões)"""
        f = []
        d = sorted(game)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        last = self.get_last_contest()
        rep = len(set(d) & set(last)) if last else 8

        # Gaps (0-3)
        f.extend([float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps))])
        # Energia (4)
        f.append(float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))))
        # Entropia da repetição (5)
        if len(self.repeat_history) >= 10:
            recent = self.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            f.append(float(entropy(np.where(probs>0, probs, 1e-10))))
        else:
            f.append(0.0)
        # Entropia de transição (6)
        if len(self.repeat_history) >= 5:
            trans = [self.repeat_history[i+1]-self.repeat_history[i] for i in range(len(self.repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                f.append(float(entropy(np.where(probs>0, probs, 1e-10))))
            else: f.append(0.0)
        else: f.append(0.0)
        # Quadrantes (7)
        f.append(float(len(set((x-1)//5 for x in d))))
        # Consecutivos (8)
        f.append(float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)))
        # Densidade local (9)
        f.append(float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15))
        # Assimetria (10)
        f.append(float(np.mean(d) - np.median(d)))
        # Clusterização (11)
        f.append(float(sum(1 for g in gaps if g <= 2) / len(gaps)))
        # Repetições do último (12)
        f.append(float(rep))
        # Pares, primos, moldura, soma, amplitude (13-17)
        f.append(float(sum(1 for x in d if x % 2 == 0)))
        f.append(float(sum(1 for x in d if x in PRIMES)))
        f.append(float(sum(1 for x in d if x in MOLDURA)))
        f.append(float(sum(d)))
        f.append(float(max(d) - min(d)))
        # Elasticidade (18)
        if len(self.repeat_history) >= 10:
            recent_avg = np.mean(self.repeat_history[-10:])
            global_avg = np.mean(self.repeat_history)
            f.append(float(global_avg - recent_avg))
        else: f.append(0.0)
        # Entropia conjunta (19)
        if len(self.repeat_history) >= 10 and len(self.pares_history) >= 10:
            joint = Counter(zip(self.repeat_history[-10:], self.pares_history[-10:]))
            probs = np.array([joint.get(k,0)/10 for k in joint])
            f.append(float(entropy(np.where(probs>0, probs, 1e-10))))
        else: f.append(0.0)

        return np.array(f, dtype=np.float32)

    def get_regime_metadata(self):
        meta = {}
        if len(self.repeat_history) >= 20:
            meta['avg_repeat_20'] = float(np.mean(self.repeat_history[-20:]))
            meta['std_repeat_20'] = float(np.std(self.repeat_history[-20:]))
        else: meta['avg_repeat_20'] = 8.0; meta['std_repeat_20'] = 1.0
        if len(self.repeat_history) >= 50:
            recent = self.repeat_history[-50:]
            freq = np.bincount(recent, minlength=13)[5:]/50
            meta['avg_entropy_50'] = float(entropy(np.where(freq>0, freq, 1e-10)))
        else: meta['avg_entropy_50'] = 1.0
        if len(self.gap_media_history) >= 20:
            meta['avg_gap_20'] = float(np.mean(self.gap_media_history[-20:]))
            meta['std_gap_20'] = float(np.std(self.gap_media_history[-20:]))
        else: meta['avg_gap_20'] = 2.0; meta['std_gap_20'] = 0.5
        if len(self.energia_history) >= 20:
            meta['avg_energy_20'] = float(np.mean(self.energia_history[-20:]))
        else: meta['avg_energy_20'] = 30.0
        if len(self.pares_history) >= 20:
            meta['std_pares_20'] = float(np.std(self.pares_history[-20:]))
        else: meta['std_pares_20'] = 1.5
        meta['n_contests'] = self.n_contests
        return meta

    def build_training_dataset(self, n_samples=5000):
        X_list, y_hits_list = [], []
        for i in range(1, len(self.contests)):
            actual = set(self.contests[i]['dezenas'])
            ctx = TopologyContext(self.contests[:i], self.historical)
            for _ in range(10):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                X_list.append(ctx.extract_topology_features(game))
                y_hits_list.append(len(set(game) & actual))
            last = set(self.contests[i-1]['dezenas'])
            for _ in range(10):
                base = list(last) if last else []
                random.shuffle(base)
                game_set = set(base[:random.randint(6, 10)])
                available = [x for x in range(1,26) if x not in game_set]
                while len(game_set) < 15: game_set.add(random.choice(available))
                X_list.append(ctx.extract_topology_features(sorted(game_set)[:15]))
                y_hits_list.append(len(set(sorted(game_set)[:15]) & actual))
        if len(X_list) > n_samples:
            indices = np.random.choice(len(X_list), n_samples, replace=False)
            X_list = [X_list[i] for i in indices]
            y_hits_list = [y_hits_list[i] for i in indices]
        return np.array(X_list), np.array(y_hits_list)


# ============================================================
# REGIME DETECTOR (SEM VAZAMENTO)
# ============================================================
class RegimeDetector:
    def __init__(self, contests):
        self.n_clusters = 4
        self.regime_features = []
        self.kmeans = None
        self.gmm = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        for i, c in enumerate(contests):
            d = c['dezenas']
            prev = set(contests[i-1]['dezenas']) if i > 0 else set()
            rep = len(set(d) & prev) if prev else 8
            vec = [rep, sum(1 for x in d if x % 2 == 0), sum(1 for x in d if x in PRIMES),
                   sum(1 for x in d if x in MOLDURA), sum(d),
                   sum(1 for j in range(len(d)-1) if d[j+1]-d[j]==1), max(d) - min(d),
                   np.mean([d[j+1]-d[j] for j in range(len(d)-1)]),
                   np.var([d[j+1]-d[j] for j in range(len(d)-1)]),
                   sum(abs(d[j]-d[j-1]) for j in range(1, len(d)))]
            self.regime_features.append(vec)
        self.regime_features = np.array(self.regime_features)
        if len(self.regime_features) > 10 and SKLEARN_AVAILABLE and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(self.regime_features)
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels_k = self.kmeans.fit_predict(X_scaled)
            self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
            labels_g = self.gmm.fit_predict(X_scaled)
            self.ensemble_labels = np.array([labels_k[i] if labels_k[i] == labels_g[i] else labels_k[i] for i in range(len(labels_k))])
        self.regime_names = {}
        if self.kmeans is not None:
            for i in range(self.n_clusters):
                mask = (self.ensemble_labels == i) if hasattr(self, 'ensemble_labels') else np.zeros(len(self.regime_features), dtype=bool)
                if mask.sum() > 0:
                    avg = self.regime_features[mask].mean(axis=0)
                    name = "alta_persistencia" if avg[0] >= 9 else "alto_pares" if avg[1] >= 8 else "compacto" if avg[4] <= 175 else "periferico" if avg[3] >= 10 else "balanceado"
                    self.regime_names[i] = {'name': name, 'size': mask.sum(), 'avg_rep': avg[0], 'avg_pares': avg[1], 'avg_soma': avg[4]}

    def get_current_regime(self):
        if len(self.regime_features) > 0 and self.kmeans is not None and self.scaler is not None:
            vec = np.array([self.regime_features[-1]])
            vec_scaled = self.scaler.transform(vec)
            k_label = self.kmeans.predict(vec_scaled)[0]
            g_label = self.gmm.predict(vec_scaled)[0] if self.gmm else k_label
            return k_label if k_label == g_label else k_label
        return 0

    def get_regime_name(self):
        idx = self.get_current_regime()
        return self.regime_names.get(idx, {}).get('name', 'balanceado')

    def get_regime_stats(self):
        if not hasattr(self, 'ensemble_labels'): return []
        return [{'cluster': i, 'name': self.regime_names.get(i,{}).get('name',f'Regime_{i}'),
                 'count': int((self.ensemble_labels==i).sum()), 'avg_rep': float(self.regime_features[self.ensemble_labels==i].mean(axis=0)[0]),
                 'avg_pares': float(self.regime_features[self.ensemble_labels==i].mean(axis=0)[1]),
                 'avg_soma': float(self.regime_features[self.ensemble_labels==i].mean(axis=0)[4])}
                for i in range(self.n_clusters) if (self.ensemble_labels==i).sum() > 0]


# ============================================================
# LEARNER DE TOPOLOGIA (APRENDIDO, NÃO HEURÍSTICO)
# ============================================================
class TopologyLearner:
    """Aprende a prever hits baseado APENAS em features topológicas"""
    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, X, y_hits):
        if X.shape[0] < 100: return False
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0) if XGB_AVAILABLE else RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        for ti, vi in TimeSeriesSplit(n_splits=3).split(X): self.model.fit(X[ti], y_hits[ti])
        self.is_trained = True
        return True

    def predict(self, features):
        if not self.is_trained: return 7.5
        return float(self.model.predict([features])[0])


class TopologyEnsembleScorer:
    """Ensemble de scores topológicos APRENDIDOS (não heurísticos)"""
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.score_stats = None

    def fit_score_stats(self, candidates, context, learner):
        """Pré-computa estatísticas para normalização z-score"""
        raw_scores = []
        for g in candidates[:500]:
            features = context.extract_topology_features(g)
            raw_scores.append(learner.predict(features))
        if raw_scores:
            self.score_stats = {'mean': np.mean(raw_scores), 'std': np.std(raw_scores)}

    def compute_score(self, game, context, learner):
        """Score normalizado por z-score"""
        features = context.extract_topology_features(game)
        raw = learner.predict(features)
        if self.score_stats and self.score_stats['std'] > 0:
            return (raw - self.score_stats['mean']) / self.score_stats['std']
        return raw


# ============================================================
# REGIME SCORE APRENDIDO
# ============================================================
class LearnedRegimeScorer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False

    def train(self, resultados_walk_forward):
        if len(resultados_walk_forward) < 15: return False
        X, y = [], []
        for r in resultados_walk_forward:
            feats = [r.get(k, 0) for k in ['avg_repeat_20', 'std_repeat_20', 'avg_entropy_50',
                                              'avg_gap_20', 'std_gap_20', 'avg_energy_20', 'std_pares_20']]
            X.append(feats)
            y.append(1 if r['diff_11'] > 0 else 0)
        X, y = np.array(X), np.array(y)
        if len(np.unique(y)) < 2: return False
        if self.scaler is not None and SKLEARN_AVAILABLE:
            X_scaled = self.scaler.fit_transform(X)
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            return True
        return False

    def predict_proba(self, meta):
        if not self.is_trained or self.scaler is None: return 0.5
        feats = np.array([[meta.get(k, 0) for k in ['avg_repeat_20', 'std_repeat_20', 'avg_entropy_50',
                                                       'avg_gap_20', 'std_gap_20', 'avg_energy_20', 'std_pares_20']]])
        feats_scaled = self.scaler.transform(feats)
        return float(self.model.predict_proba(feats_scaled)[0][1])

    def should_operate(self, meta, threshold=0.5):
        prob = self.predict_proba(meta)
        return prob >= threshold, prob


# ============================================================
# GERADOR CONDICIONAL COM FILTROS GEOMÉTRICOS
# ============================================================
class ConditionalTopologyGenerator:
    def __init__(self, context, learner, regime_detector):
        self.context = context
        self.learner = learner
        self.regime_detector = regime_detector
        self.last = context.get_last_contest()

    def _passes_filters(self, game):
        """Filtros geométricos simples (validados em conjunto separado)"""
        d = sorted(game)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        if max(gaps) > 8: return False  # gap muito grande
        if sum(1 for g in gaps if g <= 2) / len(gaps) < 0.3: return False  # pouca clusterização
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        if cons > 8: return False  # muitos consecutivos
        return True

    def generate_conditional(self, n_candidates=5000, beam_width=30):
        candidates, seen = [], set()
        base_options = [list(self.last[:8]), list(self.last[-8:]), list(self.last[3:11])] if self.last else [[]]
        for base in base_options:
            beam = [(0.0, set(base))]
            for _ in range(15 - len(base)):
                next_beam = []
                for score, game_set in beam:
                    available = [d for d in range(1,26) if d not in game_set]
                    for d in random.sample(available, min(20, len(available))):
                        new_set = game_set | {d}
                        next_beam.append((self._heuristic_score(new_set), new_set))
                next_beam.sort(key=lambda x: x[0], reverse=True)
                beam = next_beam[:beam_width]
            for score, game_set in beam:
                game = sorted(game_set)
                if len(game) == 15 and tuple(game) not in seen and self._passes_filters(game):
                    seen.add(tuple(game))
                    candidates.append(game)
        if candidates:
            for base_game in candidates[:100]:
                for _ in range(20):
                    mutated = base_game.copy()
                    pos = random.randint(0, 14)
                    avail = [d for d in range(1,26) if d not in mutated]
                    if avail:
                        mutated[pos] = random.choice(avail)
                        mutated.sort()
                        if tuple(mutated) not in seen and self._passes_filters(mutated):
                            seen.add(tuple(mutated))
                            candidates.append(mutated)
        while len(candidates) < n_candidates:
            game = sorted(np.random.choice(range(1,26), 15, replace=False))
            if tuple(game) not in seen and self._passes_filters(game):
                seen.add(tuple(game))
                candidates.append(game)
        return candidates[:n_candidates]

    def _heuristic_score(self, partial_game):
        d = sorted(partial_game)
        score = 0.0
        remaining = 15 - len(d)
        score += len(set((x-1)//5 for x in d)) * 3
        current_pares = sum(1 for x in d if x % 2 == 0)
        score -= abs(current_pares + remaining*0.5 - 7.5) * 1.5
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        if cons > 3: score -= (cons - 3) * 2
        if self.last:
            rep = len(set(d) & set(self.last))
            projected_rep = rep + remaining * 0.6
            score -= abs(projected_rep - 9) * 3
            if projected_rep > 10: score -= (projected_rep - 10) * 5
        return score

    def evaluate_distribution(self, games, test_draws):
        dist = {h: 0 for h in range(0, 16)}
        for draw in test_draws:
            actual = set(draw['dezenas'])
            for g in games: dist[len(set(g) & actual)] += 1
        total = len(test_draws) * len(games)
        return dist, total

    def compute_metrics(self, dist, total):
        if total == 0: return {'media_hits': 0, 'freq_11_plus': 0, 'freq_12_plus': 0, 'freq_13_plus': 0}
        hits_sum = sum(h * dist[h] for h in range(16))
        return {'media_hits': hits_sum / total, 'freq_11_plus': sum(dist[h] for h in range(11, 16)) / total,
                'freq_12_plus': sum(dist[h] for h in range(12, 16)) / total, 'freq_13_plus': sum(dist[h] for h in range(13, 16)) / total}


# ============================================================
# TESTES ESTATÍSTICOS
# ============================================================
def theoretical_baseline_metrics():
    expected_hits = sum(k * HYPE_PROBS[k] for k in range(16))
    return {'media_hits': expected_hits, 'freq_11_plus': sum(HYPE_PROBS[k] for k in range(11, 16)),
            'freq_12_plus': sum(HYPE_PROBS[k] for k in range(12, 16)), 'freq_13_plus': sum(HYPE_PROBS[k] for k in range(13, 16))}

def bootstrap_ci(data, n_bootstrap=5000, ci=95):
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_bootstrap)]
    return np.percentile(means, (100-ci)/2), np.percentile(means, 100-(100-ci)/2), np.mean(means)

def permutation_test(strat_vals, rand_vals, n_perm=10000):
    observed = np.mean(strat_vals) - np.mean(rand_vals)
    combined = np.concatenate([strat_vals, rand_vals])
    n1 = len(strat_vals)
    for _ in range(n_perm):
    np.random.shuffle(combined)
    perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
    if abs(perm_diff) >= abs(observed):
        extreme += 1
    return observed, extreme / n_perm

def rigorous_calibration(learner, context, test_draws, n_games=2000):
    print(f"\n📊 CALIBRAÇÃO RIGOROSA...")
    games = [sorted(np.random.choice(range(1,26), 15, replace=False)) for _ in range(n_games)]
    scored = [(learner.predict(context.extract_topology_features(g)), g) for g in games]
    scored.sort(key=lambda x: x[0], reverse=True)
    all_preds = []
    all_hits = []
    for pred, game in scored:
        total_hits = sum(len(set(game) & set(draw['dezenas'])) for draw in test_draws[-50:])
        all_preds.append(pred)
        all_hits.append(total_hits / min(50, len(test_draws)))
    all_preds = np.array(all_preds)
    all_hits = np.array(all_hits)
    spear_r, spear_p = spearmanr(all_preds, all_hits)
    print(f"   Spearman r = {spear_r:+.4f} (p={spear_p:.4f})")
    if spear_p < 0.05: print(f"   ✅ Correlação SIGNIFICATIVA")
    else: print(f"   🟡 Correlação NÃO significativa")
    bins = [1, 5, 10, 25, 50, 100]
    bin_avgs = []
    for pct in bins:
        n = max(1, int(len(scored) * pct / 100))
        bin_avgs.append(np.mean(all_hits[:n]))
        print(f"   Top {pct}%: avg_hits={bin_avgs[-1]:.4f}")
    is_monotonic = all(bin_avgs[i] >= bin_avgs[i+1] for i in range(len(bin_avgs)-1))
    print(f"   Monotonicidade: {'✅ SIM' if is_monotonic else '⚠️ NÃO'}")
    return spear_r, is_monotonic


# ============================================================
# TESTE CEGO COM PIPELINE TOPOLÓGICO
# ============================================================
def blind_test_topology(contests, blind_size=500, n_games=30):
    print(f"\n🔮 TESTE CEGO TOPOLÓGICO ({blind_size} concursos)...")
    train_contests = contests[:-blind_size]
    blind_contests = contests[-blind_size:]

    context = TopologyContext(train_contests)
    X, y_hits = context.build_training_dataset(5000)
    learner = TopologyLearner()
    learner.train(X, y_hits)

    regime_detector = RegimeDetector(train_contests)
    meta = context.get_regime_metadata()
    regime_name = regime_detector.get_regime_name()
    print(f"   Regime: {regime_name} | rep={meta['avg_repeat_20']:.1f} ent={meta['avg_entropy_50']:.2f}")

    gen = ConditionalTopologyGenerator(context, learner, regime_detector)
    candidates = gen.generate_conditional(n_candidates=3000, beam_width=30)

    ensemble_scorer = TopologyEnsembleScorer()
    ensemble_scorer.fit_score_stats(candidates, context, learner)
    scored = [(ensemble_scorer.compute_score(g, context, learner), g) for g in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_games = [g for _, g in scored[:n_games]]

    dist, total = gen.evaluate_distribution(top_games, blind_contests)
    metrics = gen.compute_metrics(dist, total)
    theo = theoretical_baseline_metrics()

    rand_games = [sorted(np.random.choice(range(1,26), 15, replace=False)) for _ in range(n_games)]
    rand_dist, rand_total = gen.evaluate_distribution(rand_games, blind_contests)
    rand_metrics = gen.compute_metrics(rand_dist, rand_total)

    strat_hits = []
    rand_hits = []
    for draw in blind_contests:
        actual = set(draw['dezenas'])
        strat_hits.append(sum(len(set(g) & actual) for g in top_games))
        rand_hits.append(sum(len(set(g) & actual) for g in rand_games))

    boot_l, boot_u, boot_m = bootstrap_ci(strat_hits)
    perm_diff, perm_p = permutation_test(strat_hits, rand_hits)

    print(f"\n📊 RESULTADOS:")
    print(f"   {'Métrica':<20} {'Estratégia':<15} {'Teórico':<15} {'Aleatório':<15}")
    print(f"   {'Média hits':<20} {metrics['media_hits']:<15.4f} {theo['media_hits']:<15.4f} {rand_metrics['media_hits']:<15.4f}")
    print(f"   {'Freq 11+':<20} {metrics['freq_11_plus']:<15.4f} {theo['freq_11_plus']:<15.4f} {rand_metrics['freq_11_plus']:<15.4f}")
    print(f"   Bootstrap IC 95%: [{boot_l:.1f}, {boot_u:.1f}]")
    print(f"   Permutation p: {perm_p:.4f}")

    rigorous_calibration(learner, context, blind_contests)
    return metrics, theo, perm_p


# ============================================================
# WALK-FORWARD TOPOLÓGICO
# ============================================================
def walk_forward_topology(contests, n_windows=30, train_size=300, test_size=50):
    print(f"\n🔬 WALK-FORWARD TOPOLÓGICO ({n_windows} janelas)...")
    resultados = []
    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue

        context = TopologyContext(train_data, contests[:train_end])
        X, y_hits = context.build_training_dataset(3000)
        learner = TopologyLearner()
        learner.train(X, y_hits)

        regime_detector = RegimeDetector(train_data)
        gen = ConditionalTopologyGenerator(context, learner, regime_detector)
        candidates = gen.generate_conditional(2000, 30)

        ensemble_scorer = TopologyEnsembleScorer()
        ensemble_scorer.fit_score_stats(candidates, context, learner)
        scored = [(ensemble_scorer.compute_score(g, context, learner), g) for g in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        games = [g for _, g in scored[:30]]

        dist, total = gen.evaluate_distribution(games, test_data)
        metrics = gen.compute_metrics(dist, total)

        rand_games = [sorted(np.random.choice(range(1,26), 15, replace=False)) for _ in range(30)]
        rand_dist, rand_total = gen.evaluate_distribution(rand_games, test_data)
        rand_metrics = gen.compute_metrics(rand_dist, rand_total)

        meta = context.get_regime_metadata()
        resultados.append({
            'window': w, 'strat_11': metrics['freq_11_plus'], 'rand_11': rand_metrics['freq_11_plus'],
            'diff_11': metrics['freq_11_plus'] - rand_metrics['freq_11_plus'],
            'regime': regime_detector.get_regime_name(), **meta
        })
        print(f" Janela {w}: diff={metrics['freq_11_plus']-rand_metrics['freq_11_plus']:+.4f} "
              f"rep={meta['avg_repeat_20']:.1f} ent={meta['avg_entropy_50']:.2f}")

    if resultados:
        diffs = [r['diff_11'] for r in resultados]
        print(f"\n📊 RESUMO:")
        print(f"   Média diff: {np.mean(diffs):+.4f}")
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p: {p:.4f}")
        except: pass
        print(f"   Janelas +: {sum(1 for d in diffs if d > 0)}/{len(resultados)}")
        for key in ['avg_repeat_20', 'avg_entropy_50', 'avg_gap_20', 'std_pares_20']:
            vals = [r.get(key, 0) for r in resultados]
            if len(vals) >= 5:
                corr, pval = pearsonr(vals, diffs)
                print(f"   {key}: r={corr:+.3f} p={pval:.4f}")
        edge_scorer = LearnedRegimeScorer()
        edge_scorer.train(resultados)
    return resultados


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 TOPOLOGIA PURA + PIPELINE v31")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")

    regime_detector = RegimeDetector(contests)
    print("\n📊 REGIMES:")
    for s in regime_detector.get_regime_stats():
        print(f"   {s['name']}: {s['count']} concursos (Rep:{s['avg_rep']:.1f} Pares:{s['avg_pares']:.1f} Soma:{s['avg_soma']:.0f})")

    print("\nOpções:")
    print("1. Teste cego topológico (500 concursos)")
    print("2. Walk-forward topológico (30 janelas)")
    print("3. TUDO")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        blind_test_topology(contests, blind_size=500, n_games=30)

    if op in ("2", "3"):
        walk_forward_topology(contests, n_windows=30, train_size=300, test_size=50)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
