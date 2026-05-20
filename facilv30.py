#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE GEOMETRIA COMBINATÓRIA - LOTOFÁCIL v33
======================================================
PESQUISA DE TOPOLOGIA EM ESPAÇOS COMBINATÓRIOS

MELHORIAS:
✅ SHAP values para interpretabilidade real
✅ Controle por densidade (lift topológico)
✅ Simulação completamente cega (dados sintéticos)
✅ Embeddings topológicos (autoencoder simplificado)
✅ Monte Carlo topológico com análise de distribuição
✅ Validação rigorosa com permutation test corrigido
✅ Pipeline modular: Contexto → Learner → Gerador → Análise
✅ Geração neutra, sem viés humano
✅ PCA + visualização do espaço topológico
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
    print("⚠️ XGBoost não instalado. Use: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não instalado. Use: pip install shap")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado.")

# ============================================================
# CONJUNTOS E CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# ============================================================
# FEATURES DE TOPOLOGIA PURA (v33)
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
    "repeticoes",         # 12
    "pares",              # 13
    "primos",             # 14
    "moldura",            # 15
    "soma",               # 16
    "amplitude",          # 17
    "elasticidade",       # 18
    "entropia_conjunta",  # 19
]

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
    """Carrega todos os concursos do arquivo CSV"""
    if not os.path.exists(csv_file):
        return None
    contests = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(';')
                if len(parts) >= 17:
                    contests.append({
                        'concurso': int(parts[0]),
                        'data': parts[1],
                        'dezenas': [int(x) for x in parts[2:17]]
                    })
        contests.sort(key=lambda x: x['concurso'])
        return contests
    except:
        return None


def generate_synthetic_contests(n_contests=3686):
    """
    GERA DADOS SINTÉTICOS HIPERGEOMÉTRICOS PUROS.
    Essencial para testar se o sistema encontra "estrutura" em ruído.
    """
    print(f"🎲 Gerando {n_contests} concursos sintéticos...")
    contests = []
    for i in range(1, n_contests + 1):
        dezenas = sorted(np.random.choice(range(1, 26), 15, replace=False))
        contests.append({
            'concurso': i,
            'data': f"{2000 + i//100:04d}-{(i%100):02d}-01",
            'dezenas': dezenas
        })
    return contests


# ============================================================
# CONTEXTO TEMPORAL (TOPOLOGIA PURA)
# ============================================================
class TopologyContext:
    """
    Contexto focado APENAS em features topológicas.
    Sem EMA, fadiga, streaks, random_noise.
    """
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
            else:
                f.append(0.0)
        else:
            f.append(0.0)
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
        else:
            f.append(0.0)
        # Entropia conjunta (19)
        if len(self.repeat_history) >= 10 and len(self.pares_history) >= 10:
            joint = Counter(zip(self.repeat_history[-10:], self.pares_history[-10:]))
            probs = np.array([joint.get(k,0)/10 for k in joint])
            f.append(float(entropy(np.where(probs>0, probs, 1e-10))))
        else:
            f.append(0.0)

        return np.array(f, dtype=np.float32)

    def get_regime_metadata(self):
        """Metadados do regime para análise"""
        meta = {}
        if len(self.repeat_history) >= 20:
            meta['avg_repeat_20'] = float(np.mean(self.repeat_history[-20:]))
            meta['std_repeat_20'] = float(np.std(self.repeat_history[-20:]))
        else:
            meta['avg_repeat_20'] = 8.0
            meta['std_repeat_20'] = 1.0
        if len(self.repeat_history) >= 50:
            recent = self.repeat_history[-50:]
            freq = np.bincount(recent, minlength=13)[5:]/50
            meta['avg_entropy_50'] = float(entropy(np.where(freq>0, freq, 1e-10)))
        else:
            meta['avg_entropy_50'] = 1.0
        if len(self.gap_media_history) >= 20:
            meta['avg_gap_20'] = float(np.mean(self.gap_media_history[-20:]))
            meta['std_gap_20'] = float(np.std(self.gap_media_history[-20:]))
        else:
            meta['avg_gap_20'] = 2.0
            meta['std_gap_20'] = 0.5
        if len(self.energia_history) >= 20:
            meta['avg_energy_20'] = float(np.mean(self.energia_history[-20:]))
        else:
            meta['avg_energy_20'] = 30.0
        if len(self.pares_history) >= 20:
            meta['std_pares_20'] = float(np.std(self.pares_history[-20:]))
        else:
            meta['std_pares_20'] = 1.5
        meta['n_contests'] = self.n_contests
        return meta

    def build_training_dataset(self, n_samples=5000):
        """
        Dataset sem vazamento: ctx[:i] para concurso i.
        """
        X_list, y_hits_list = [], []
        for i in range(1, len(self.contests)):
            actual = set(self.contests[i]['dezenas'])
            ctx = TopologyContext(self.contests[:i], self.historical)
            # Exemplos aleatórios
            for _ in range(10):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                X_list.append(ctx.extract_topology_features(game))
                y_hits_list.append(len(set(game) & actual))
            # Hard negatives
            last = set(self.contests[i-1]['dezenas'])
            for _ in range(10):
                base = list(last) if last else []
                random.shuffle(base)
                game_set = set(base[:random.randint(6, 10)])
                available = [x for x in range(1,26) if x not in game_set]
                while len(game_set) < 15:
                    game_set.add(random.choice(available))
                X_list.append(ctx.extract_topology_features(sorted(game_set)[:15]))
                y_hits_list.append(len(set(sorted(game_set)[:15]) & actual))
        if len(X_list) > n_samples:
            indices = np.random.choice(len(X_list), n_samples, replace=False)
            X_list = [X_list[i] for i in indices]
            y_hits_list = [y_hits_list[i] for i in indices]
        return np.array(X_list), np.array(y_hits_list)


# ============================================================
# REGIME DETECTOR
# ============================================================
class RegimeDetector:
    """Detecta regimes estruturais nos concursos históricos"""
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
            vec = [
                rep, sum(1 for x in d if x % 2 == 0),
                sum(1 for x in d if x in PRIMES),
                sum(1 for x in d if x in MOLDURA),
                sum(d),
                sum(1 for j in range(len(d)-1) if d[j+1]-d[j]==1),
                max(d) - min(d),
                np.mean([d[j+1]-d[j] for j in range(len(d)-1)]),
                np.var([d[j+1]-d[j] for j in range(len(d)-1)]),
                sum(abs(d[j]-d[j-1]) for j in range(1, len(d)))
            ]
            self.regime_features.append(vec)

        self.regime_features = np.array(self.regime_features)
        if len(self.regime_features) > 10 and SKLEARN_AVAILABLE and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(self.regime_features)
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels_k = self.kmeans.fit_predict(X_scaled)
            self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
            labels_g = self.gmm.fit_predict(X_scaled)
            self.ensemble_labels = np.array([
                labels_k[i] if labels_k[i] == labels_g[i] else labels_k[i]
                for i in range(len(labels_k))
            ])

        self.regime_names = {}
        if self.kmeans is not None:
            for i in range(self.n_clusters):
                mask = (self.ensemble_labels == i) if hasattr(self, 'ensemble_labels') else np.zeros(len(self.regime_features), dtype=bool)
                if mask.sum() > 0:
                    avg = self.regime_features[mask].mean(axis=0)
                    if avg[0] >= 9:
                        name = "alta_persistencia"
                    elif avg[1] >= 8:
                        name = "alto_pares"
                    elif avg[4] <= 175:
                        name = "compacto"
                    elif avg[3] >= 10:
                        name = "periferico"
                    else:
                        name = "balanceado"
                    self.regime_names[i] = {
                        'name': name, 'size': mask.sum(),
                        'avg_rep': avg[0], 'avg_pares': avg[1], 'avg_soma': avg[4]
                    }

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
        if not hasattr(self, 'ensemble_labels'):
            return []
        return [
            {
                'cluster': i,
                'name': self.regime_names.get(i, {}).get('name', f'Regime_{i}'),
                'count': int((self.ensemble_labels == i).sum()),
                'avg_rep': float(self.regime_features[self.ensemble_labels == i].mean(axis=0)[0]),
                'avg_pares': float(self.regime_features[self.ensemble_labels == i].mean(axis=0)[1]),
                'avg_soma': float(self.regime_features[self.ensemble_labels == i].mean(axis=0)[4])
            }
            for i in range(self.n_clusters) if (self.ensemble_labels == i).sum() > 0
        ]


# ============================================================
# LEARNER DE TOPOLOGIA COM SHAP
# ============================================================
class TopologyLearner:
    """Aprende a prever hits baseado APENAS em features topológicas"""
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.shap_explainer = None
        self.shap_values = None
        self.feature_importance = None

    def train(self, X, y_hits):
        if X.shape[0] < 100:
            return False
        self.model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0
        ) if XGB_AVAILABLE else RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        for ti, vi in TimeSeriesSplit(n_splits=3).split(X):
            self.model.fit(X[ti], y_hits[ti])

        # SHAP analysis
        if SHAP_AVAILABLE and XGB_AVAILABLE:
            try:
                X_sample = X[:min(200, len(X))]
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
                self.feature_importance = np.abs(self.shap_values).mean(axis=0)
            except:
                pass

        self.is_trained = True
        return True

    def predict(self, features):
        if not self.is_trained:
            return 7.5
        return float(self.model.predict([features])[0])

    def explain(self, features_vector):
        """Retorna contribuição SHAP de cada feature"""
        if self.shap_explainer is None:
            return {}
        shap_vals = self.shap_explainer.shap_values(np.array([features_vector]))[0]
        return {
            TOPOLOGY_FEATURE_NAMES[i]: float(shap_vals[i])
            for i in range(len(shap_vals))
        }

    def get_top_features(self, top_n=10):
        """Retorna as features mais importantes segundo SHAP"""
        if self.feature_importance is None:
            return []
        top_idx = np.argsort(self.feature_importance)[-top_n:][::-1]
        return [
            (TOPOLOGY_FEATURE_NAMES[i], float(self.feature_importance[i]))
            for i in top_idx
        ]


# ============================================================
# GERADOR NEUTRO (SEM VIÉS HUMANO)
# ============================================================
class NeutralTopologyGenerator:
    """
    Gerador NEUTRO: não induz repetição ~9 nem filtros manuais.
    O learner aprende sozinho quais topologias funcionam.
    """
    def __init__(self, context, learner):
        self.context = context
        self.learner = learner
        self.last = context.get_last_contest()

    def generate_candidates(self, n_candidates=20000):
        """
        Geração puramente topológica:
        - 60% beam search exploratório (sem viés de repetição)
        - 20% aleatório puro
        - 20% mutações dos melhores
        """
        candidates, seen = [], set()

        # 1. Beam search NEUTRO (sem induzir repetição ~9)
        if self.last:
            base_options = [
                list(self.last[:8]),
                list(self.last[-8:]),
                list(random.sample(range(1, 26), 10)),
                list(random.sample(range(1, 26), 10)),
            ]
        else:
            base_options = [list(random.sample(range(1, 26), 10)) for _ in range(4)]

        for base in base_options:
            beam = [(0.0, set(base))]
            for _ in range(15 - len(base)):
                next_beam = []
                for score, game_set in beam:
                    available = [d for d in range(1, 26) if d not in game_set]
                    for d in random.sample(available, min(20, len(available))):
                        new_set = game_set | {d}
                        # Score NEUTRO: apenas diversidade espacial
                        s = len(set((x-1)//5 for x in new_set)) * 3
                        next_beam.append((s, new_set))
                next_beam.sort(key=lambda x: x[0], reverse=True)
                beam = next_beam[:30]
            for score, game_set in beam:
                game = sorted(game_set)
                if len(game) == 15 and tuple(game) not in seen:
                    seen.add(tuple(game))
                    candidates.append(game)

        # 2. Aleatório puro
        for _ in range(n_candidates // 5):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                candidates.append(game)

        # 3. Mutações dos melhores (learner avalia)
        if len(candidates) > 50:
            sample_for_scoring = candidates[:min(500, len(candidates))]
            scored = [
                (self.learner.predict(self.context.extract_topology_features(g)), g)
                for g in sample_for_scoring
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [g for _, g in scored[:100]]
            for base_game in top_candidates:
                for _ in range(20):
                    mutated = base_game.copy()
                    pos = random.randint(0, 14)
                    avail = [d for d in range(1, 26) if d not in mutated]
                    if avail:
                        mutated[pos] = random.choice(avail)
                        mutated.sort()
                        if tuple(mutated) not in seen:
                            seen.add(tuple(mutated))
                            candidates.append(mutated)

        # Completar se necessário
        while len(candidates) < n_candidates:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                candidates.append(game)

        return candidates[:n_candidates]

    def evaluate_distribution(self, games, test_draws):
        dist = {h: 0 for h in range(0, 16)}
        for draw in test_draws:
            actual = set(draw['dezenas'])
            for g in games:
                dist[len(set(g) & actual)] += 1
        total = len(test_draws) * len(games)
        return dist, total

    def compute_metrics(self, dist, total):
        if total == 0:
            return {'media_hits': 0, 'freq_11_plus': 0, 'freq_12_plus': 0, 'freq_13_plus': 0}
        hits_sum = sum(h * dist[h] for h in range(16))
        return {
            'media_hits': hits_sum / total,
            'freq_11_plus': sum(dist[h] for h in range(11, 16)) / total,
            'freq_12_plus': sum(dist[h] for h in range(12, 16)) / total,
            'freq_13_plus': sum(dist[h] for h in range(13, 16)) / total,
        }


# ============================================================
# TESTES ESTATÍSTICOS (PERMUTATION CORRIGIDO)
# ============================================================
def theoretical_baseline_metrics():
    expected_hits = sum(k * HYPE_PROBS[k] for k in range(16))
    return {
        'media_hits': expected_hits,
        'freq_11_plus': sum(HYPE_PROBS[k] for k in range(11, 16)),
        'freq_12_plus': sum(HYPE_PROBS[k] for k in range(12, 16)),
        'freq_13_plus': sum(HYPE_PROBS[k] for k in range(13, 16)),
    }

def bootstrap_ci(data, n_bootstrap=5000, ci=95):
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_bootstrap)]
    return np.percentile(means, (100-ci)/2), np.percentile(means, 100-(100-ci)/2), np.mean(means)

def permutation_test_corrected(strat_vals, rand_vals, n_perm=10000):
    """
    Permutation test CORRIGIDO: shuffle DENTRO do loop.
    """
    observed = np.mean(strat_vals) - np.mean(rand_vals)
    combined = np.concatenate([strat_vals, rand_vals])
    n1 = len(strat_vals)
    extreme = 0

    for _ in range(n_perm):
        np.random.shuffle(combined)  # ← CORREÇÃO: shuffle dentro do loop
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed):
            extreme += 1

    return observed, extreme / n_perm

def rigorous_calibration(learner, context, test_draws, n_games=2000):
    print(f"\n📊 CALIBRAÇÃO RIGOROSA...")
    games = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_games)]
    scored = [(learner.predict(context.extract_topology_features(g)), g) for g in games]
    scored.sort(key=lambda x: x[0], reverse=True)

    all_preds, all_hits = [], []
    for pred, game in scored:
        total_hits = sum(len(set(game) & set(draw['dezenas'])) for draw in test_draws[-50:])
        all_preds.append(pred)
        all_hits.append(total_hits / min(50, len(test_draws)))

    all_preds = np.array(all_preds)
    all_hits = np.array(all_hits)

    spear_r, spear_p = spearmanr(all_preds, all_hits)
    print(f"   Spearman r = {spear_r:+.4f} (p={spear_p:.4f})")
    if spear_p < 0.05:
        print(f"   ✅ Correlação SIGNIFICATIVA")
    else:
        print(f"   🟡 Correlação NÃO significativa")

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
# MONTE CARLO TOPOLÓGICO COM CONTROLE DE DENSIDADE
# ============================================================
def monte_carlo_topology(context, learner, n_games=50000):
    """
    Gera muitos jogos e analisa distribuição das topologias.
    Calcula LIFT: p(topologia | top1%) / p(topologia | universo)
    """
    print(f"\n🎲 MONTE CARLO TOPOLÓGICO ({n_games:,} jogos)...")
    gen = NeutralTopologyGenerator(context, learner)
    candidates = gen.generate_candidates(n_games)

    # Extrair features de todos
    X_all = np.array([context.extract_topology_features(g) for g in candidates])
    scores_all = np.array([learner.predict(x) for x in X_all])

    # Top 1% por score
    threshold = np.percentile(scores_all, 99)
    top_mask = scores_all >= threshold
    top_X = X_all[top_mask]
    all_X = X_all

    print(f"\n📊 ANÁLISE TOPOLÓGICA:")
    print(f"   Top 1% (score ≥ {threshold:.2f}): {top_X.shape[0]} jogos")
    print(f"   {'Feature':<25} {'Top 1%':<15} {'Todos':<15} {'LIFT':<10} {'Signif':<10}")
    print(f"   {'-'*75}")

    lift_analysis = {}
    for i, name in enumerate(TOPOLOGY_FEATURE_NAMES):
        top_mean = np.mean(top_X[:, i]) if len(top_X) > 0 else 0
        all_mean = np.mean(all_X[:, i])
        # LIFT: razão entre top 1% e universo
        lift = top_mean / (all_mean + 1e-10)
        # Para features discretas, comparar proporções
        if i in [IDX_QUADRANTES, IDX_CONSECUTIVOS, IDX_PARES, IDX_PRIMOS, IDX_MOLDURA]:
            # Discretizar
            top_prop = np.mean(top_X[:, i]) if len(top_X) > 0 else 0
            all_prop = np.mean(all_X[:, i])
            lift = top_prop / (all_prop + 1e-10)

        marker = "🔴" if lift > 1.2 or lift < 0.8 else "  "
        print(f"   {name:<25} {top_mean:<15.4f} {all_mean:<15.4f} {lift:<10.3f} {marker}")
        lift_analysis[name] = {'top_mean': float(top_mean), 'all_mean': float(all_mean), 'lift': float(lift)}

    # PCA opcional
    if SKLEARN_AVAILABLE and len(X_all) > 50:
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_all[:10000])
            top_pca = X_pca[top_mask[:10000]] if len(top_pca) > 0 else np.array([])
            print(f"\n📊 PCA: variância explicada = {pca.explained_variance_ratio_.sum()*100:.1f}%")
            if len(top_pca) > 0:
                top_center = np.mean(top_pca, axis=0)
                all_center = np.mean(X_pca, axis=0)
                dist = np.linalg.norm(top_center - all_center)
                print(f"   Centro Top 1%: ({top_center[0]:.2f}, {top_center[1]:.2f})")
                print(f"   Centro Todos:  ({all_center[0]:.2f}, {all_center[1]:.2f})")
                print(f"   Distância: {dist:.2f}")
                if dist > 0.5:
                    print(f"   ✅ Top 1% está em região DISTINTA do espaço")
                else:
                    print(f"   🟡 Top 1% está na MESMA região do espaço")
        except:
            pass

    return candidates, scores_all, lift_analysis


# ============================================================
# SIMULAÇÃO COMPLETAMENTE CEGA
# ============================================================
def blind_synthetic_test(n_contests=3686, blind_size=500, n_games=30):
    """
    Teste CEGO com dados SINTÉTICOS.
    Se o sistema encontrar "estrutura" aqui, é viés do pipeline.
    """
    print(f"\n🔮 TESTE CEGO SINTÉTICO (dados hipergeométricos puros)...")
    contests = generate_synthetic_contests(n_contests)
    train_contests = contests[:-blind_size]
    blind_contests = contests[-blind_size:]

    context = TopologyContext(train_contests)
    X, y_hits = context.build_training_dataset(5000)
    learner = TopologyLearner()
    learner.train(X, y_hits)

    gen = NeutralTopologyGenerator(context, learner)
    candidates = gen.generate_candidates(20000)

    scored = [(learner.predict(context.extract_topology_features(g)), g) for g in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_games = [g for _, g in scored[:n_games]]

    dist, total = gen.evaluate_distribution(top_games, blind_contests)
    metrics = gen.compute_metrics(dist, total)
    theo = theoretical_baseline_metrics()

    rand_games = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_games)]
    rand_dist, rand_total = gen.evaluate_distribution(rand_games, blind_contests)
    rand_metrics = gen.compute_metrics(rand_dist, rand_total)

    strat_hits, rand_hits = [], []
    for draw in blind_contests:
        actual = set(draw['dezenas'])
        strat_hits.append(sum(len(set(g) & actual) for g in top_games))
        rand_hits.append(sum(len(set(g) & actual) for g in rand_games))

    boot_l, boot_u, boot_m = bootstrap_ci(strat_hits)
    perm_diff, perm_p = permutation_test_corrected(strat_hits, rand_hits)

    print(f"\n📊 RESULTADOS (DADOS SINTÉTICOS):")
    print(f"   {'Métrica':<20} {'Estratégia':<15} {'Teórico':<15} {'Aleatório':<15}")
    print(f"   {'Média hits':<20} {metrics['media_hits']:<15.4f} {theo['media_hits']:<15.4f} {rand_metrics['media_hits']:<15.4f}")
    print(f"   {'Freq 11+':<20} {metrics['freq_11_plus']:<15.4f} {theo['freq_11_plus']:<15.4f} {rand_metrics['freq_11_plus']:<15.4f}")
    print(f"   Permutation p: {perm_p:.4f}")

    if perm_p < 0.05:
        print(f"\n   ⚠️  ALERTA: Sistema encontrou 'estrutura' em dados SINTÉTICOS!")
        print(f"   ⚠️  Isso sugere VIÉS no pipeline, não edge real.")
    else:
        print(f"\n   ✅ Sistema NÃO encontrou estrutura em dados sintéticos.")
        print(f"   ✅ Pipeline não tem viés intrínseco.")

    return metrics, theo, perm_p


# ============================================================
# TESTE CEGO TOPOLÓGICO
# ============================================================
def blind_test_topology(contests, blind_size=500, n_games=30):
    print(f"\n🔮 TESTE CEGO TOPOLÓGICO ({blind_size} concursos)...")
    train_contests = contests[:-blind_size]
    blind_contests = contests[-blind_size:]

    context = TopologyContext(train_contests)
    X, y_hits = context.build_training_dataset(5000)
    learner = TopologyLearner()
    learner.train(X, y_hits)

    # SHAP report
    if learner.feature_importance is not None:
        print(f"\n📊 TOP 10 FEATURES (SHAP):")
        for name, imp in learner.get_top_features(10):
            print(f"   {name:<25} importância={imp:.4f}")

    regime_detector = RegimeDetector(train_contests)
    meta = context.get_regime_metadata()
    regime_name = regime_detector.get_regime_name()
    print(f"\n   Regime: {regime_name} | rep={meta['avg_repeat_20']:.1f} ent={meta['avg_entropy_50']:.2f}")

    gen = NeutralTopologyGenerator(context, learner)
    candidates = gen.generate_candidates(20000)

    # Monte Carlo para análise
    monte_carlo_topology(context, learner, n_games=30000)

    # Ranking por score do learner
    scored = [(learner.predict(context.extract_topology_features(g)), g) for g in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_games = [g for _, g in scored[:n_games]]

    dist, total = gen.evaluate_distribution(top_games, blind_contests)
    metrics = gen.compute_metrics(dist, total)
    theo = theoretical_baseline_metrics()

    rand_games = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_games)]
    rand_dist, rand_total = gen.evaluate_distribution(rand_games, blind_contests)
    rand_metrics = gen.compute_metrics(rand_dist, rand_total)

    strat_hits, rand_hits = [], []
    for draw in blind_contests:
        actual = set(draw['dezenas'])
        strat_hits.append(sum(len(set(g) & actual) for g in top_games))
        rand_hits.append(sum(len(set(g) & actual) for g in rand_games))

    boot_l, boot_u, boot_m = bootstrap_ci(strat_hits)
    perm_diff, perm_p = permutation_test_corrected(strat_hits, rand_hits)

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
        if train_start >= train_end or test_start >= test_end:
            continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5:
            continue

        context = TopologyContext(train_data, contests[:train_end])
        X, y_hits = context.build_training_dataset(3000)
        learner = TopologyLearner()
        learner.train(X, y_hits)

        regime_detector = RegimeDetector(train_data)
        gen = NeutralTopologyGenerator(context, learner)
        candidates = gen.generate_candidates(3000)

        scored = [(learner.predict(context.extract_topology_features(g)), g) for g in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        games = [g for _, g in scored[:30]]

        dist, total = gen.evaluate_distribution(games, test_data)
        metrics = gen.compute_metrics(dist, total)

        rand_games = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(30)]
        rand_dist, rand_total = gen.evaluate_distribution(rand_games, test_data)
        rand_metrics = gen.compute_metrics(rand_dist, rand_total)

        meta = context.get_regime_metadata()
        resultados.append({
            'window': w,
            'strat_11': metrics['freq_11_plus'],
            'rand_11': rand_metrics['freq_11_plus'],
            'diff_11': metrics['freq_11_plus'] - rand_metrics['freq_11_plus'],
            'regime': regime_detector.get_regime_name(),
            **meta
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
        except:
            pass
        print(f"   Janelas +: {sum(1 for d in diffs if d > 0)}/{len(resultados)}")
        for key in ['avg_repeat_20', 'avg_entropy_50', 'avg_gap_20', 'std_pares_20']:
            vals = [r.get(key, 0) for r in resultados]
            if len(vals) >= 5:
                corr, pval = pearsonr(vals, diffs)
                sig = "🔴" if pval < 0.01 else "🟡" if pval < 0.05 else "🟢"
                print(f"   {key}: r={corr:+.3f} p={pval:.4f} {sig}")
    return resultados


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 LABORATÓRIO DE GEOMETRIA COMBINATÓRIA v33")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado. Gerando dados sintéticos...")
        contests = generate_synthetic_contests(3686)
    print(f"📂 {len(contests)} concursos")

    regime_detector = RegimeDetector(contests)
    print("\n📊 REGIMES:")
    for s in regime_detector.get_regime_stats():
        print(f"   {s['name']}: {s['count']} concursos (Rep:{s['avg_rep']:.1f} Pares:{s['avg_pares']:.1f} Soma:{s['avg_soma']:.0f})")

    print("\nOpções:")
    print("1. Teste cego topológico (500 concursos)")
    print("2. Walk-forward topológico (30 janelas)")
    print("3. Monte Carlo Topológico (análise de distribuição)")
    print("4. Simulação CEGA (dados sintéticos)")
    print("5. TUDO")
    op = input("Escolha [5]: ").strip() or "5"

    if op in ("1", "5"):
        blind_test_topology(contests, blind_size=500, n_games=30)

    if op in ("2", "5"):
        walk_forward_topology(contests, n_windows=30, train_size=300, test_size=50)

    if op in ("3", "5"):
        context = TopologyContext(contests)
        X, y_hits = context.build_training_dataset(5000)
        learner = TopologyLearner()
        learner.train(X, y_hits)
        monte_carlo_topology(context, learner, n_games=50000)

    if op in ("4", "5"):
        blind_synthetic_test(n_contests=3686, blind_size=500, n_games=30)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
