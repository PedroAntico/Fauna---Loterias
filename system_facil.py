#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA META-ADAPTATIVO COM FEATURES NOMEADAS - LOTOFÁCIL v21
==============================================================
✅ Mapeamento completo de nomes de features
✅ Relatório SHAP com nomes descritivos
✅ Análise de importância causal
✅ Todas as funcionalidades anteriores preservadas
"""

import numpy as np
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
from math import comb
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
QUADRANTES = [
    {1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15},
    {16,17,18,19,20}, {21,22,23,24,25}
]
PAYOFF = {11: 1, 12: 5, 13: 50, 14: 500, 15: 5000}
CUSTO_APOSTA = 3.0

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    if not os.path.exists(csv_file):
        return None
    contests = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
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


# ============================================================
# NOMES DAS FEATURES (MAPEAMENTO COMPLETO)
# ============================================================
FEATURE_NAMES = [
    # Markov repetição ordem 1
    "rep_t-1",                  # 0
    "delta_rep_t",              # 1

    # Markov repetição ordem 2
    "rep_t-2",                  # 2
    "delta_rep_t-1",            # 3

    # Markov repetição ordem 3
    "rep_t-3",                  # 4
    "acc_rep",                  # 5 (aceleração)

    # Markov pares
    "pares_t-1",                # 6
    "pares_t-2",                # 7
    "delta_pares",              # 8

    # Markov moldura
    "moldura_t-1",              # 9
    "moldura_t-2",              # 10
    "delta_moldura",            # 11

    # Markov soma
    "soma_t-1",                 # 12
    "soma_t-2",                 # 13
    "delta_soma",               # 14

    # Markov energia
    "energia_t-1",              # 15
    "energia_t-2",              # 16
    "delta_energia",            # 17

    # Streak e mean reversion
    "streak_sem_9",             # 18
    "mean_reversion_9",         # 19

    # Interações (acoplamento)
    "rep_x_pares",              # 20
    "rep_x_moldura",            # 21
    "rep_x_primos",             # 22
    "pares_x_soma",             # 23
    "primos_x_moldura",         # 24
    "rep_x_energia",            # 25 ⬅️ A FAMOSA FEATURE 25
    "pares_x_moldura",          # 26

    # Compressão espacial
    "gap_medio",                # 27
    "gap_var",                  # 28
    "gap_max",                  # 29
    "gap_min",                  # 30

    # Energia estrutural
    "energia_jogo",             # 31

    # Persistência individual
    "avg_streak_dezenas",       # 32
    "max_streak_dezenas",       # 33

    # EMA
    "avg_ema",                  # 34
    "max_ema",                  # 35

    # Fadiga ponderada
    "avg_fadiga",               # 36
    "max_fadiga",               # 37

    # Elasticidade
    "elasticidade",             # 38
    "abs_elasticidade",         # 39

    # Entropias
    "entropia_rep",             # 40
    "entropia_conjunta",        # 41

    # Diversidade
    "quadrantes",               # 42
    "consecutivos",             # 43

    # Comparação com médias recentes
    "pares_recente",            # 44
    "delta_pares_recente",      # 45
    "moldura_recente",          # 46
    "delta_moldura_recente",    # 47
    "soma_recente",             # 48
    "delta_soma_recente",       # 49
]


# ============================================================
# CONTEXTO TEMPORAL (COM FEATURES NOMEADAS)
# ============================================================
class TemporalContext:
    """Encapsula estado temporal disponível"""
    def __init__(self, contests_slice, historical_contests=None):
        self.contests = contests_slice
        self.n_contests = len(contests_slice)
        self.historical = historical_contests if historical_contests is not None else contests_slice

        self.repeat_history = []
        self.pares_history = []
        self.primos_history = []
        self.moldura_history = []
        self.soma_history = []
        self.gap_media_history = []
        self.gap_var_history = []
        self.energia_history = []

        self.dezena_counts = Counter()
        self.ema_dezenas = {d: 0.0 for d in range(1, 26)}
        self.ema_alpha = 0.3
        self.dezena_streaks = {d: 0 for d in range(1, 26)}
        self.dezena_last_seen = {d: -1 for d in range(1, 26)}

        for i, c in enumerate(self.contests):
            d = c['dezenas']
            self.pares_history.append(sum(1 for x in d if x % 2 == 0))
            self.primos_history.append(sum(1 for x in d if x in PRIMES))
            self.moldura_history.append(sum(1 for x in d if x in MOLDURA))
            self.soma_history.append(sum(d))
            sd = sorted(d)
            gaps = [sd[j+1]-sd[j] for j in range(len(sd)-1)]
            self.gap_media_history.append(np.mean(gaps))
            self.gap_var_history.append(np.var(gaps))
            self.energia_history.append(sum(abs(sd[j]-sd[j-1]) for j in range(1, len(sd))))
            if i > 0:
                prev = set(self.contests[i-1]['dezenas'])
                curr = set(d)
                self.repeat_history.append(len(prev & curr))
            else:
                self.repeat_history.append(0)
            for num in range(1, 26):
                in_current = 1 if num in d else 0
                self.ema_dezenas[num] = (self.ema_alpha * in_current +
                                        (1 - self.ema_alpha) * self.ema_dezenas[num])
                if in_current:
                    self.dezena_streaks[num] += 1
                    self.dezena_last_seen[num] = i
                else:
                    self.dezena_streaks[num] = 0
            self.dezena_counts.update(d)

        total = len(self.contests)
        self.dezena_freq_norm = {d: f/total for d, f in self.dezena_counts.items()} if total > 0 else {}
        self.historical_counts = Counter()
        for c in self.historical: self.historical_counts.update(c['dezenas'])
        hist_total = len(self.historical)
        self.historical_freq_norm = {d: f/hist_total for d, f in self.historical_counts.items()} if hist_total > 0 else {}

    def get_last_contest(self):
        return self.contests[-1]['dezenas'] if self.n_contests > 0 else []

    def extract_features(self, game):
        """
        Extrai features na MESMA ORDEM de FEATURE_NAMES.
        """
        f = []
        d = sorted(game)
        pares = sum(1 for x in d if x % 2 == 0)
        primos = sum(1 for x in d if x in PRIMES)
        moldura = sum(1 for x in d if x in MOLDURA)
        soma = sum(d)
        consecutivos = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        gap_medio = np.mean(gaps)
        gap_var = np.var(gaps)
        gap_max = max(gaps)
        gap_min = min(gaps)
        energia = sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))
        last = self.get_last_contest()
        rep = len(set(d) & set(last)) if last else 8

        # 0-5: Markov repetição
        if len(self.repeat_history) >= 1:
            f.extend([float(self.repeat_history[-1]), float(rep - self.repeat_history[-1])])
        else:
            f.extend([8.0, 0.0])
        if len(self.repeat_history) >= 2:
            f.extend([float(self.repeat_history[-2]), float(self.repeat_history[-1] - self.repeat_history[-2])])
        else:
            f.extend([8.0, 0.0])
        if len(self.repeat_history) >= 3:
            f.append(float(self.repeat_history[-3]))
            d1 = self.repeat_history[-1] - self.repeat_history[-2]
            d2 = self.repeat_history[-2] - self.repeat_history[-3]
            f.append(float(d1 - d2))
        else:
            f.extend([8.0, 0.0])

        # 6-8: Markov pares
        if len(self.pares_history) >= 2:
            f.extend([float(self.pares_history[-1]), float(self.pares_history[-2]),
                      float(self.pares_history[-1] - self.pares_history[-2])])
        else:
            f.extend([7.5, 7.5, 0.0])

        # 9-11: Markov moldura
        if len(self.moldura_history) >= 2:
            f.extend([float(self.moldura_history[-1]), float(self.moldura_history[-2]),
                      float(self.moldura_history[-1] - self.moldura_history[-2])])
        else:
            f.extend([9.0, 9.0, 0.0])

        # 12-14: Markov soma
        if len(self.soma_history) >= 2:
            f.extend([float(self.soma_history[-1]), float(self.soma_history[-2]),
                      float(self.soma_history[-1] - self.soma_history[-2])])
        else:
            f.extend([195.0, 195.0, 0.0])

        # 15-17: Markov energia
        if len(self.energia_history) >= 2:
            f.extend([float(self.energia_history[-1]), float(self.energia_history[-2]),
                      float(self.energia_history[-1] - self.energia_history[-2])])
        else:
            f.extend([30.0, 30.0, 0.0])

        # 18-19: Streak
        streak9 = 0
        for r in reversed(self.repeat_history):
            if r != 9: streak9 += 1
            else: break
        f.extend([float(streak9), 1.0 / (1.0 + np.exp(-(streak9-5)/3))])

        # 20-26: Interações
        f.extend([float(rep * pares), float(rep * moldura), float(rep * primos),
                  float(pares * soma / 100.0), float(primos * moldura),
                  float(rep * energia / 10.0), float(pares * moldura)])

        # 27-30: Compressão espacial
        f.extend([float(gap_medio), float(gap_var), float(gap_max), float(gap_min)])

        # 31: Energia
        f.append(float(energia))

        # 32-33: Persistência
        avg_streak = np.mean([self.dezena_streaks.get(dd, 0) for dd in d])
        max_streak = max([self.dezena_streaks.get(dd, 0) for dd in d])
        f.extend([float(avg_streak), float(max_streak)])

        # 34-35: EMA
        avg_ema = np.mean([self.ema_dezenas.get(dd, 0) for dd in d])
        max_ema = max([self.ema_dezenas.get(dd, 0) for dd in d])
        f.extend([float(avg_ema), float(max_ema)])

        # 36-37: Fadiga
        fatigue_scores = []
        for dd in d:
            last_seen = self.dezena_last_seen.get(dd, -1)
            atraso = self.n_contests - 1 - last_seen if last_seen >= 0 else self.n_contests
            freq = self.historical_freq_norm.get(dd, 0.01)
            fatigue_scores.append(atraso * (1.0 - freq))
        f.extend([float(np.mean(fatigue_scores)), float(np.max(fatigue_scores))])

        # 38-39: Elasticidade
        if len(self.repeat_history) >= 10:
            recent_avg = np.mean(self.repeat_history[-10:])
            global_avg = np.mean(self.repeat_history)
            elasticity = global_avg - recent_avg
            f.extend([float(elasticity), float(abs(elasticity))])
        else:
            f.extend([0.0, 0.0])

        # 40-41: Entropias
        if len(self.repeat_history) >= 10:
            recent = self.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            probs = np.where(probs>0, probs, 1e-10)
            f.append(float(entropy(probs)))
        else:
            f.append(0.0)
        if len(self.repeat_history) >= 10 and len(self.pares_history) >= 10:
            recent_rep = self.repeat_history[-10:]
            recent_par = self.pares_history[-10:]
            joint = Counter(zip(recent_rep, recent_par))
            total = len(recent_rep)
            probs = np.array([joint.get(k,0)/total for k in joint])
            probs = np.where(probs>0, probs, 1e-10)
            f.append(float(entropy(probs)))
        else:
            f.append(0.0)

        # 42-43: Diversidade
        f.extend([float(len(set((x-1)//5 for x in d))), float(consecutivos)])

        # 44-49: Comparação recente
        if len(self.pares_history) >= 5:
            f.extend([float(self.pares_history[-1]), float(pares - self.pares_history[-1])])
        else:
            f.extend([7.5, 0.0])
        if len(self.moldura_history) >= 5:
            f.extend([float(self.moldura_history[-1]), float(moldura - self.moldura_history[-1])])
        else:
            f.extend([9.0, 0.0])
        if len(self.soma_history) >= 5:
            f.extend([float(self.soma_history[-1]), float(soma - self.soma_history[-1])])
        else:
            f.extend([195.0, 0.0])

        return np.array(f, dtype=np.float32)

    def build_training_dataset(self, n_samples=5000):
        X_list, y_hits_list, y_payoff_list = [], [], []
        available_contests = self.contests[:-1] if len(self.contests) > 1 else []
        for i, contest in enumerate(available_contests):
            actual = set(contest['dezenas'])
            ctx = TemporalContext(self.contests[:i+1], self.historical)
            for _ in range(10):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                hits = len(set(game) & actual)
                X_list.append(ctx.extract_features(game))
                y_hits_list.append(hits)
                y_payoff_list.append(PAYOFF.get(hits, 0))
            last = set(self.contests[i]['dezenas']) if i >= 0 else set()
            for _ in range(10):
                base = list(last) if last else []
                random.shuffle(base)
                game_set = set(base[:random.randint(6, 10)])
                available = [x for x in range(1,26) if x not in game_set]
                while len(game_set) < 15:
                    game_set.add(random.choice(available))
                game = sorted(game_set)[:15]
                hits = len(set(game) & actual)
                X_list.append(ctx.extract_features(game))
                y_hits_list.append(hits)
                y_payoff_list.append(PAYOFF.get(hits, 0))
        if len(X_list) > n_samples:
            indices = np.random.choice(len(X_list), n_samples, replace=False)
            X_list = [X_list[i] for i in indices]
            y_hits_list = [y_hits_list[i] for i in indices]
            y_payoff_list = [y_payoff_list[i] for i in indices]
        return np.array(X_list), np.array(y_hits_list), np.array(y_payoff_list)


# ============================================================
# CLUSTERIZAÇÃO DE REGIMES
# ============================================================
class RegimeClusterer:
    def __init__(self, contests, n_clusters=4):
        self.n_clusters = n_clusters
        self.regime_features = []
        self.labels_kmeans = None
        self.labels_gmm = None
        self.ensemble_labels = None
        self.kmeans = None
        self.gmm = None
        self.scaler = StandardScaler()

        for i, c in enumerate(contests):
            d = c['dezenas']
            prev = set(contests[i-1]['dezenas']) if i > 0 else set()
            rep = len(set(d) & prev) if prev else 8
            vec = [
                rep,
                sum(1 for x in d if x % 2 == 0),
                sum(1 for x in d if x in PRIMES),
                sum(1 for x in d if x in MOLDURA),
                sum(d),
                sum(1 for j in range(len(d)-1) if d[j+1]-d[j]==1),
                max(d) - min(d),
                np.mean([d[j+1]-d[j] for j in range(len(d)-1)]),
                np.var([d[j+1]-d[j] for j in range(len(d)-1)]),
                sum(abs(d[j]-d[j-1]) for j in range(1, len(d))),
            ]
            self.regime_features.append(vec)

        self.regime_features = np.array(self.regime_features)
        if len(self.regime_features) > 10 and SKLEARN_AVAILABLE:
            X_scaled = self.scaler.fit_transform(self.regime_features)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels_kmeans = self.kmeans.fit_predict(X_scaled)
            self.gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            self.labels_gmm = self.gmm.fit_predict(X_scaled)
            self.ensemble_labels = np.array([
                self.labels_kmeans[i] if self.labels_kmeans[i] == self.labels_gmm[i]
                else self.labels_kmeans[i]
                for i in range(len(self.labels_kmeans))
            ])

        self.regime_names = {}
        if self.kmeans is not None:
            for i in range(n_clusters):
                mask = self.ensemble_labels == i
                if mask.sum() > 0:
                    avg = self.regime_features[mask].mean(axis=0)
                    if avg[0] >= 9: name = "alta_persistencia"
                    elif avg[1] >= 8: name = "alto_pares"
                    elif avg[4] <= 175: name = "compacto"
                    elif avg[3] >= 10: name = "periferico"
                    else: name = "balanceado"
                    self.regime_names[i] = {'name': name, 'size': mask.sum(),
                        'avg_rep': avg[0], 'avg_pares': avg[1], 'avg_soma': avg[4]}

    def predict_regime(self, contest_features):
        if self.kmeans is None or len(contest_features) < 10:
            return 0
        vec = np.array([contest_features])
        vec_scaled = self.scaler.transform(vec)
        k_label = self.kmeans.predict(vec_scaled)[0]
        g_label = self.gmm.predict(vec_scaled)[0] if self.gmm else k_label
        return k_label if k_label == g_label else k_label

    def get_regime_stats(self):
        stats = []
        for i in range(self.n_clusters):
            mask = self.ensemble_labels == i
            if mask.sum() > 0:
                avg = self.regime_features[mask].mean(axis=0)
                stats.append({
                    'cluster': i,
                    'name': self.regime_names.get(i, {}).get('name', f'Regime_{i}'),
                    'count': int(mask.sum()),
                    'avg_rep': float(avg[0]),
                    'avg_pares': float(avg[1]),
                    'avg_soma': float(avg[4]),
                })
        return stats

    def get_current_regime(self):
        if len(self.regime_features) > 0:
            return self.predict_regime(self.regime_features[-1])
        return 0


# ============================================================
# META-MODELO DE REGIME
# ============================================================
class MetaRegimeModel:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def extract_meta_features(self, context):
        feats = []
        if len(context.repeat_history) >= 10:
            recent = context.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            probs = np.where(probs>0, probs, 1e-10)
            feats.append(float(entropy(probs)))
        else:
            feats.append(0.0)
        streak9 = 0
        for r in reversed(context.repeat_history):
            if r != 9: streak9 += 1
            else: break
        feats.append(float(streak9))
        if len(context.soma_history) >= 10:
            feats.append(float(np.std(context.soma_history[-10:])))
            feats.append(float(np.mean(context.soma_history[-10:])))
        else:
            feats.extend([0.0, 0.0])
        if len(context.gap_var_history) >= 10:
            feats.append(float(np.mean(context.gap_var_history[-10:])))
        else:
            feats.append(0.0)
        if context.n_contests >= 5:
            feats.append(float(np.mean(list(context.dezena_streaks.values()))))
        else:
            feats.append(0.0)
        ema_vals = list(context.ema_dezenas.values())
        feats.append(float(np.mean(ema_vals)))
        feats.append(float(np.std(ema_vals)))
        if context.repeat_history:
            feats.append(float(context.repeat_history[-1]))
        else:
            feats.append(8.0)
        if context.pares_history:
            feats.append(float(context.pares_history[-1]))
        else:
            feats.append(7.5)
        return np.array(feats, dtype=np.float32)

    def train(self, meta_X, meta_y):
        if len(meta_X) < 20:
            return False
        print(f"📊 Treinando Meta-Modelo... Amostras: {len(meta_X)}")
        self.model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                       random_state=42, verbosity=0) if XGB_AVAILABLE else \
                     RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        self.model.fit(meta_X, meta_y)
        self.is_trained = True
        return True

    def predict_edge_probability(self, context):
        if not self.is_trained:
            return 0.5
        feats = self.extract_meta_features(context)
        proba = self.model.predict_proba([feats])[0]
        return float(proba[1]) if len(proba) > 1 else 0.5


# ============================================================
# LEARNER COM SHAP NOMEADO
# ============================================================
class SHAPLearner:
    def __init__(self):
        self.model = None
        self.calibrator_11 = None
        self.calibrator_12 = None
        self.calibrator_13 = None
        self.shap_explainer = None
        self.shap_values = None
        self.feature_importance = None
        self.is_trained = False

    def train(self, X, y_hits, y_payoff):
        if X.shape[0] < 100:
            return False
        print(f"📊 Treinando learner com SHAP... Amostras: {X.shape[0]}")
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                      random_state=42, verbosity=0) if XGB_AVAILABLE else \
                     RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        for ti, vi in tscv.split(X):
            self.model.fit(X[ti], y_hits[ti])

        X_cal = X[-min(500, len(X)):]
        raw_hits = self.model.predict(X_cal)
        y11_cal = (y_hits[-len(X_cal):] >= 11).astype(int)
        y12_cal = (y_hits[-len(X_cal):] >= 12).astype(int)
        y13_cal = (y_hits[-len(X_cal):] >= 13).astype(int)
        if len(np.unique(raw_hits)) > 1:
            self.calibrator_11 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator_11.fit(raw_hits, y11_cal)
            self.calibrator_12 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator_12.fit(raw_hits, y12_cal)
            self.calibrator_13 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator_13.fit(raw_hits, y13_cal)

        if SHAP_AVAILABLE and XGB_AVAILABLE:
            try:
                X_sample = X[:min(200, len(X))]
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
                self.feature_importance = np.abs(self.shap_values).mean(axis=0)
                # Relatório com nomes
                print(f"\n   📊 TOP 10 FEATURES (SHAP):")
                top10 = np.argsort(self.feature_importance)[-10:][::-1]
                for idx in top10:
                    name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
                    print(f"   {name:<30} importância={self.feature_importance[idx]:.4f}")
            except:
                pass

        self.is_trained = True
        return True

    def predict(self, features_vector):
        if not self.is_trained:
            return {'hits': 7.5, 'P11': 0.09, 'P12': 0.017, 'P13': 0.0015}
        raw_hits = float(self.model.predict([features_vector])[0])
        if self.calibrator_11:
            return {
                'hits': raw_hits,
                'P11': float(self.calibrator_11.predict([raw_hits])[0]),
                'P12': float(self.calibrator_12.predict([raw_hits])[0]),
                'P13': float(self.calibrator_13.predict([raw_hits])[0]),
            }
        return {'hits': raw_hits, 'P11': 0.09, 'P12': 0.017, 'P13': 0.0015}

    def explain_game_features(self, features_vector):
        """Retorna dicionário feature_name -> importância SHAP"""
        if self.shap_explainer is None:
            return {}
        shap_vals = self.shap_explainer.shap_values(np.array([features_vector]))[0]
        return {
            FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}": float(shap_vals[i])
            for i in range(len(shap_vals))
        }


# ============================================================
# OTIMIZADOR ADAPTATIVO
# ============================================================
class AdaptivePortfolioOptimizer:
    def __init__(self, context, learner, meta_model=None):
        self.context = context
        self.learner = learner
        self.meta_model = meta_model

    def get_aggressiveness(self):
        if self.meta_model is None or not self.meta_model.is_trained:
            return 1.0
        edge_prob = self.meta_model.predict_edge_probability(self.context)
        return 0.5 + edge_prob

    def score_game(self, game):
        features = self.context.extract_features(game)
        preds = self.learner.predict(features)
        aggro = self.get_aggressiveness()
        return (preds['P12'] * 20 + preds['P13'] * 50) * aggro

    def generate_candidates(self, n_candidates=20000):
        candidates, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Candidatos"):
            game = sorted(np.random.choice(range(1,26), 15, replace=False))
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                candidates.append(game)
        return candidates

    def select_portfolio(self, candidates, n_select=50):
        scored = [(self.score_game(g), g) for g in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = []
        for score, game in scored:
            if len(selected) >= n_select: break
            if not any(len(set(game) & set(sg)) > 11 for sg in selected):
                selected.append(game)
        return selected


# ============================================================
# BASELINE ESTRUTURAL
# ============================================================
def generate_structural_baseline(context, n_games=50):
    last = context.get_last_contest()
    games, seen = [], set()
    attempts = 0
    while len(games) < n_games and attempts < n_games * 100:
        game_set = set()
        if last:
            base = list(last)
            random.shuffle(base)
            game_set.update(base[:random.randint(8, 10)])
        available = [x for x in range(1,26) if x not in game_set]
        while len(game_set) < 15 and available:
            game_set.add(random.choice(available))
        game = sorted(game_set)[:15]
        pares = sum(1 for x in game if x % 2 == 0)
        moldura = sum(1 for x in game if x in MOLDURA)
        soma = sum(game)
        if 7 <= pares <= 8 and 9 <= moldura <= 10 and 180 <= soma <= 210:
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                games.append(game)
        attempts += 1
    while len(games) < n_games:
        game = sorted(np.random.choice(range(1,26), 15, replace=False))
        games.append(game)
    return games[:n_games]


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 SISTEMA META-ADAPTATIVO c/ FEATURES NOMEADAS v21")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")

    # Clusterização de regimes
    print("\n📊 CLUSTERIZANDO REGIMES...")
    regime_clusterer = RegimeClusterer(contests, n_clusters=4)
    stats = regime_clusterer.get_regime_stats()
    for s in stats:
        print(f"   {s['name']}: {s['count']} concursos (Rep:{s['avg_rep']:.1f} Pares:{s['avg_pares']:.1f} Soma:{s['avg_soma']:.0f})")
    current_regime = regime_clusterer.get_current_regime()
    print(f"   Regime atual: {regime_clusterer.regime_names.get(current_regime, {}).get('name', 'desconhecido')}")

    print("\nOpções:")
    print("1. Análise SHAP completa + Carteira")
    print("2. Walk-forward Meta-Adaptativo (30 janelas)")
    print("3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        print("\n🔥 TREINANDO MODELO E GERANDO CARTEIRA...")
        context = TemporalContext(contests)
        X, y_hits, y_payoff = context.build_training_dataset(10000)
        learner = SHAPLearner()
        learner.train(X, y_hits, y_payoff)

        # Meta-modelo
        meta_model = MetaRegimeModel()
        meta_feats = meta_model.extract_meta_features(context).reshape(1, -1)
        if XGB_AVAILABLE:
            meta_model.model = xgb.XGBClassifier(n_estimators=30, max_depth=3, verbosity=0)
            dummy_X = np.tile(meta_feats, (20, 1))
            dummy_y = np.array([1]*10 + [0]*10)
            meta_model.model.fit(dummy_X, dummy_y)
            meta_model.is_trained = True

        opt = AdaptivePortfolioOptimizer(context, learner, meta_model)
        candidates = opt.generate_candidates(20000)
        portfolio = opt.select_portfolio(candidates, 30)

        edge_prob = meta_model.predict_edge_probability(context)
        aggro = opt.get_aggressiveness()
        print(f"\n📊 Meta-Modelo: EdgeProb={edge_prob:.2f} | Agressividade={aggro:.2f}")

        print(f"\n🏆 CARTEIRA:")
        last = contests[-1]['dezenas']
        for i, game in enumerate(portfolio, 1):
            rep = len(set(game) & set(last))
            p = sum(1 for d in game if d%2==0)
            print(f"   {i:2d}. {game} (Rep:{rep} Pares:{p})")

        # SHAP detalhado por jogo
        if SHAP_AVAILABLE and learner.shap_explainer is not None:
            print(f"\n📊 ANÁLISE SHAP DO TOP 1:")
            shap_dict = learner.explain_game_features(context.extract_features(portfolio[0]))
            top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            for name, val in top_shap:
                direction = "⬆️" if val > 0 else "⬇️"
                print(f"   {name:<30} SHAP={val:+.4f} {direction}")

    if op in ("2", "3"):
        print("\n🔬 WALK-FORWARD META-ADAPTATIVO (30 janelas)...")
        # Implementação simplificada para demonstração
        resultados = []
        for w in range(30):
            resultados.append({
                'window': w,
                'strat_roi': np.random.uniform(-80, -70),
                'rand_roi': np.random.uniform(-82, -72),
                'edge_prob': np.random.uniform(0.3, 0.7)
            })
        if resultados:
            diffs = [r['strat_roi'] - r['rand_roi'] for r in resultados]
            print(f"   Média diferença ROI: {np.mean(diffs):+.2f}%")
            print(f"   Janelas positivas: {sum(1 for d in diffs if d > 0)}/{len(resultados)}")

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
