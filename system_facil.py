#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE OTIMIZAÇÃO COMBINATÓRIA DE CARTEIRA - LOTOFÁCIL v25
===============================================================
CORREÇÕES E APERFEIÇOAMENTOS:
✅ Limitador de repetição (função com ótimo em 9, penalidade >10)
✅ Diversidade estrutural real (vetor de forma + distância euclidiana)
✅ Penalidade de centro de massa (anti-aglomeração)
✅ Triple coverage com amostragem (eficiência)
✅ Ablação completa do grupo "repeticao_completa"
✅ Correção np.int64 (conversão explícita para int)
✅ Beam search com heurísticas leves; learner.predict() só em completos
✅ Fitness de portfólio global (cobertura, entropia, diversidade, payoff)
✅ Estabilidade SHAP, walk-forward real, exportação de relatórios
"""

import numpy as np
from scipy.stats import entropy, wilcoxon
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

# ============================================================
# IMPORTS OPCIONAIS
# ============================================================
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
# NOMES DAS FEATURES
# ============================================================
FEATURE_NAMES = [
    "rep_t-1", "delta_rep_t", "rep_t-2", "delta_rep_t-1",
    "rep_t-3", "acc_rep",
    "pares_t-1", "pares_t-2", "delta_pares",
    "moldura_t-1", "moldura_t-2", "delta_moldura",
    "soma_t-1", "soma_t-2", "delta_soma",
    "energia_t-1", "energia_t-2", "delta_energia",
    "streak_sem_9", "mean_reversion_9",
    "rep_x_pares", "rep_x_moldura", "rep_x_primos",
    "pares_x_soma", "primos_x_moldura", "rep_x_energia", "pares_x_moldura",
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo",
    "avg_streak_dezenas", "max_streak_dezenas",
    "avg_ema", "max_ema",
    "avg_fadiga", "max_fadiga",
    "elasticidade", "abs_elasticidade",
    "entropia_rep", "entropia_conjunta",
    "quadrantes", "consecutivos",
    "pares_recente", "delta_pares_recente",
    "moldura_recente", "delta_moldura_recente",
    "soma_recente", "delta_soma_recente",
]

FEATURE_GROUPS = {
    "repeticao": list(range(0, 6)),
    "pares": [6, 7, 8, 44, 45],
    "moldura": [9, 10, 11, 46, 47],
    "soma": [12, 13, 14, 48, 49],
    "energia": [15, 16, 17, 31],
    "interacoes": list(range(20, 27)),
    "compressao": list(range(27, 31)),
    "persistencia": [32, 33, 34, 35],
    "fadiga": [36, 37],
    "elasticidade": [38, 39],
    "entropia": [40, 41],
    "diversidade": [42, 43],
    "streak": [18, 19],
    # NOVO: grupo completo para ablação total da repetição
    "repeticao_completa": [0,1,2,3,4,5,18,19,20,21,22,25],
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
        print(f"📂 Tentando abrir: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines[1:]):  # pula cabeçalho

            parts = line.strip().split(';')

            # DEBUG
            if len(parts) < 17:
                print(f"⚠️ Linha {i+2} inválida: {parts}")
                continue

            try:
                concurso = int(parts[0])
                data = parts[1]

                dezenas = []

                # pega EXATAMENTE as 15 bolas
                for x in parts[2:17]:
                    dezenas.append(int(x))

                contests.append({
                    'concurso': concurso,
                    'data': data,
                    'dezenas': dezenas
                })

            except Exception as e:
                print(f"❌ Erro na linha {i+2}: {e}")
                print(parts)
                continue

        contests.sort(key=lambda x: x['concurso'])

        print(f"✅ {len(contests)} concursos carregados")

        return contests

    except Exception as e:
        print(f"❌ Erro lendo CSV: {e}")
        return None

# ============================================================
# CONTEXTO TEMPORAL
# ============================================================
class TemporalContext:
    def __init__(self, contests_slice, historical_contests=None):
        self.contests = contests_slice
        self.n_contests = len(contests_slice)
        self.historical = historical_contests if historical_contests is not None else contests_slice
        self.repeat_history, self.pares_history, self.primos_history = [], [], []
        self.moldura_history, self.soma_history = [], []
        self.gap_media_history, self.gap_var_history, self.energia_history = [], [], []
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
                self.ema_dezenas[num] = (self.ema_alpha * in_current + (1 - self.ema_alpha) * self.ema_dezenas[num])
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
            d1, d2 = self.repeat_history[-1] - self.repeat_history[-2], self.repeat_history[-2] - self.repeat_history[-3]
            f.append(float(d1 - d2))
        else:
            f.extend([8.0, 0.0])
        # 6-8: Markov pares
        if len(self.pares_history) >= 2:
            f.extend([float(self.pares_history[-1]), float(self.pares_history[-2]), float(self.pares_history[-1] - self.pares_history[-2])])
        else:
            f.extend([7.5, 7.5, 0.0])
        # 9-11: Markov moldura
        if len(self.moldura_history) >= 2:
            f.extend([float(self.moldura_history[-1]), float(self.moldura_history[-2]), float(self.moldura_history[-1] - self.moldura_history[-2])])
        else:
            f.extend([9.0, 9.0, 0.0])
        # 12-14: Markov soma
        if len(self.soma_history) >= 2:
            f.extend([float(self.soma_history[-1]), float(self.soma_history[-2]), float(self.soma_history[-1] - self.soma_history[-2])])
        else:
            f.extend([195.0, 195.0, 0.0])
        # 15-17: Markov energia
        if len(self.energia_history) >= 2:
            f.extend([float(self.energia_history[-1]), float(self.energia_history[-2]), float(self.energia_history[-1] - self.energia_history[-2])])
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
        # 27-30: Compressão
        f.extend([float(gap_medio), float(gap_var), float(gap_max), float(gap_min)])
        # 31: Energia
        f.append(float(energia))
        # 32-33: Persistência
        f.extend([float(np.mean([self.dezena_streaks.get(dd,0) for dd in d])), float(max([self.dezena_streaks.get(dd,0) for dd in d]))])
        # 34-35: EMA
        f.extend([float(np.mean([self.ema_dezenas.get(dd,0) for dd in d])), float(max([self.ema_dezenas.get(dd,0) for dd in d]))])
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
            joint = Counter(zip(self.repeat_history[-10:], self.pares_history[-10:]))
            total = 10
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
# LEARNER (mantido da v24)
# ============================================================
class StableSHAPLearner:
    def __init__(self):
        self.model = None
        self.calibrator_11 = None
        self.calibrator_12 = None
        self.calibrator_13 = None
        self.shap_explainer = None
        self.feature_importance = None
        self.shap_stability = []
        self.is_trained = False

    def _train_core(self, X, y_hits):
        if X.shape[0] < 100: return False
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0) if XGB_AVAILABLE else RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        for ti, vi in tscv.split(X): self.model.fit(X[ti], y_hits[ti])
        X_cal = X[-min(500, len(X)):]
        raw_hits = self.model.predict(X_cal)
        if len(np.unique(raw_hits)) > 1:
            self.calibrator_11 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator_11.fit(raw_hits, (y_hits[-len(X_cal):] >= 11).astype(int))
            self.calibrator_12 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator_12.fit(raw_hits, (y_hits[-len(X_cal):] >= 12).astype(int))
            self.calibrator_13 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator_13.fit(raw_hits, (y_hits[-len(X_cal):] >= 13).astype(int))
        if SHAP_AVAILABLE and XGB_AVAILABLE:
            try:
                X_sample = X[:min(200, len(X))]
                self.shap_explainer = shap.TreeExplainer(self.model)
                shap_vals = self.shap_explainer.shap_values(X_sample)
                self.feature_importance = np.abs(shap_vals).mean(axis=0)
            except: pass
        self.is_trained = True
        return True

    def train(self, X, y_hits, y_payoff):
        return self._train_core(X, y_hits)

    def train_multiple_rounds(self, X, y_hits, y_payoff, n_rounds=10):
        print(f"\n📊 ESTABILIDADE SHAP ({n_rounds} rodadas)...")
        self.shap_stability = []
        for r in range(n_rounds):
            np.random.seed(42 + r)
            idx = np.random.choice(len(X), min(len(X), 3000), replace=False)
            self._train_core(X[idx], y_hits[idx])
            if self.feature_importance is not None:
                self.shap_stability.append(self.feature_importance.copy())
        if len(self.shap_stability) >= 2:
            importances = np.array(self.shap_stability)
            mean_imp = importances.mean(axis=0)
            std_imp = importances.std(axis=0)
            cv_imp = std_imp / (mean_imp + 1e-10)
            print(f"\n   📊 TOP 10 FEATURES ESTÁVEIS (menor CV):")
            stable_idx = np.argsort(cv_imp)[:10]
            for idx in stable_idx:
                name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"f{idx}"
                print(f"   {name:<30} imp={mean_imp[idx]:.3f}±{std_imp[idx]:.3f} CV={cv_imp[idx]:.2f}")
            top_idx = np.argsort(mean_imp)[-10:][::-1]
            print(f"\n   📊 TOP 10 POR IMPORTÂNCIA MÉDIA:")
            for idx in top_idx:
                name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"f{idx}"
                print(f"   {name:<30} imp={mean_imp[idx]:.3f}±{std_imp[idx]:.3f}")
            self.feature_importance = mean_imp
        return True

    def predict(self, features_vector):
        if not self.is_trained: return {'hits': 7.5, 'P11': 0.09, 'P12': 0.017, 'P13': 0.0015}
        raw_hits = float(self.model.predict([features_vector])[0])
        if self.calibrator_11:
            return {'hits': raw_hits, 'P11': float(self.calibrator_11.predict([raw_hits])[0]),
                    'P12': float(self.calibrator_12.predict([raw_hits])[0]),
                    'P13': float(self.calibrator_13.predict([raw_hits])[0])}
        return {'hits': raw_hits, 'P11': 0.09, 'P12': 0.017, 'P13': 0.0015}


# ============================================================
# REGIME CLUSTERER
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
            vec = [rep, sum(1 for x in d if x % 2 == 0), sum(1 for x in d if x in PRIMES),
                   sum(1 for x in d if x in MOLDURA), sum(d),
                   sum(1 for j in range(len(d)-1) if d[j+1]-d[j]==1), max(d) - min(d),
                   np.mean([d[j+1]-d[j] for j in range(len(d)-1)]),
                   np.var([d[j+1]-d[j] for j in range(len(d)-1)]),
                   sum(abs(d[j]-d[j-1]) for j in range(1, len(d)))]
            self.regime_features.append(vec)
        self.regime_features = np.array(self.regime_features)
        if len(self.regime_features) > 10 and SKLEARN_AVAILABLE:
            X_scaled = self.scaler.fit_transform(self.regime_features)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels_kmeans = self.kmeans.fit_predict(X_scaled)
            self.gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            self.labels_gmm = self.gmm.fit_predict(X_scaled)
            self.ensemble_labels = np.array([self.labels_kmeans[i] if self.labels_kmeans[i] == self.labels_gmm[i] else self.labels_kmeans[i] for i in range(len(self.labels_kmeans))])
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
                    self.regime_names[i] = {'name': name, 'size': mask.sum(), 'avg_rep': avg[0], 'avg_pares': avg[1], 'avg_soma': avg[4]}

    def get_regime_stats(self):
        return [{'cluster': i, 'name': self.regime_names.get(i,{}).get('name',f'Regime_{i}'),
                 'count': int((self.ensemble_labels==i).sum()), 'avg_rep': float(self.regime_features[self.ensemble_labels==i].mean(axis=0)[0]),
                 'avg_pares': float(self.regime_features[self.ensemble_labels==i].mean(axis=0)[1]),
                 'avg_soma': float(self.regime_features[self.ensemble_labels==i].mean(axis=0)[4])}
                for i in range(self.n_clusters) if (self.ensemble_labels==i).sum() > 0]

    def get_current_regime(self):
        if len(self.regime_features) > 0 and self.kmeans is not None:
            vec = np.array([self.regime_features[-1]])
            vec_scaled = self.scaler.transform(vec)
            k_label = self.kmeans.predict(vec_scaled)[0]
            g_label = self.gmm.predict(vec_scaled)[0] if self.gmm else k_label
            return k_label if k_label == g_label else k_label
        return 0


# ============================================================
# GERADOR GUIADO v25 (COM CORREÇÕES)
# ============================================================
class GuidedGeneratorV25:
    """
    Geração guiada com:
    - Limitador de repetição (função com ótimo em 9)
    - Diversidade estrutural real (vetor de forma)
    - Penalidade de centro de massa
    - Triple coverage com amostragem
    """
    def __init__(self, context, learner, regime_clusterer=None):
        self.context = context
        self.learner = learner
        self.regime = regime_clusterer
        self.last = context.get_last_contest()

    def _shape_vector(self, game):
        """Vetor de forma estrutural do jogo"""
        d = sorted(game)
        return np.array([
            sum(1 for x in d if x % 2 == 0),           # pares
            sum(1 for x in d if x in PRIMES),          # primos
            sum(1 for x in d if x in MOLDURA),         # moldura
            sum(d),                                     # soma
            np.mean([d[i+1]-d[i] for i in range(14)]), # gap médio
            max(d) - min(d),                            # amplitude
        ], dtype=np.float32)

    def _heuristic_score(self, partial_game):
        """
        Score heurístico CORRIGIDO para jogos incompletos.
        - Função com ótimo em 9 repetidas (não linear crescente)
        - Penalidade forte >10
        """
        d = sorted(partial_game)
        score = 0.0
        remaining = 15 - len(d)

        # Diversidade de quadrantes
        score += len(set((x-1)//5 for x in d)) * 3

        # Balanceamento par/ímpar
        current_pares = sum(1 for x in d if x % 2 == 0)
        projected_pares = current_pares + (remaining * 0.5)
        score -= abs(projected_pares - 7.5) * 1.5

        # Penalidade de consecutivos
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        if cons > 3: score -= (cons - 3) * 2

        # REPETIÇÃO CORRIGIDA: função com ótimo em 9
        if self.last:
            rep = len(set(d) & set(self.last))
            projected_rep = rep + (remaining * 0.6)
            # Função triangular: máximo em 9, penaliza desvios
            rep_score = -abs(projected_rep - 9) * 3
            # Penalidade EXTRA para repetição excessiva
            if projected_rep > 10:
                rep_score -= (projected_rep - 10) * 5
            score += rep_score

        # Bônus por momentum
        if hasattr(self.context, 'ema_dezenas'):
            ema_vals = [self.context.ema_dezenas.get(x, 0) for x in d]
            score += np.mean(ema_vals) * 3

        return score

    def _learner_score(self, game):
        """Score do learner APENAS para jogos completos"""
        feats = self.context.extract_features(game)
        preds = self.learner.predict(feats)
        return preds['P12'] * 20 + preds['P13'] * 50

    def generate_beam_search(self, n_candidates=5000, beam_width=30):
        """Beam search com heurísticas leves (learner só no final)"""
        candidates = []
        seen = set()

        if self.last:
            base_options = [list(self.last[:8]), list(self.last[-8:]), list(self.last[3:11])]
        else:
            base_options = [[]]

        for base in base_options:
            beam = [(0.0, set(base))]
            for _ in range(15 - len(base)):
                next_beam = []
                for score, game_set in beam:
                    available = [d for d in range(1, 26) if d not in game_set]
                    sampled = random.sample(available, min(20, len(available)))
                    for d in sampled:
                        new_set = game_set | {d}
                        s = self._heuristic_score(new_set)
                        next_beam.append((s, new_set))
                next_beam.sort(key=lambda x: x[0], reverse=True)
                beam = next_beam[:beam_width]
            for score, game_set in beam:
                game = sorted(game_set)
                if len(game) == 15:
                    key = tuple(game)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(game)

        # Mutação dos melhores (learner em completos)
        if candidates:
            scored = [(self._learner_score(g), g) for g in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [g for _, g in scored[:100]]
            for base_game in top_candidates:
                for _ in range(20):
                    mutated = base_game.copy()
                    pos = random.randint(0, 14)
                    available = [d for d in range(1, 26) if d not in mutated]
                    if available:
                        mutated[pos] = random.choice(available)
                    mutated.sort()
                    key = tuple(mutated)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(mutated)

        while len(candidates) < n_candidates:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                candidates.append(game)

        return candidates[:n_candidates]

    def portfolio_fitness(self, portfolio):
        """
        Fitness de portfólio GLOBAL (v25).
        Adiciona: diversidade estrutural real, penalidade de centro de massa,
        triple coverage com amostragem.
        """
        # Payoff esperado
        scores = [self._learner_score(g) for g in portfolio]
        avg_score = np.mean(scores)

        # Cobertura de pares
        covered_pairs = set()
        for g in portfolio:
            for pair in combinations(sorted(g), 2):
                covered_pairs.add(pair)
        pair_coverage = len(covered_pairs) / comb(25, 2)

        # Cobertura de trincas COM AMOSTRAGEM
        all_triples = list(combinations(range(1, 26), 3))
        sample_size = min(500, len(all_triples))
        sampled_triples = random.sample(all_triples, sample_size)
        covered_sample = 0
        for triple in sampled_triples:
            for g in portfolio:
                if set(triple).issubset(set(g)):
                    covered_sample += 1
                    break
        triple_coverage = covered_sample / sample_size

        # Entropia do portfólio
        all_dezenas = [d for g in portfolio for d in g]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        portfolio_entropy = entropy(probs) / np.log(25)

        # DIVERSIDADE ESTRUTURAL REAL (vetor de forma)
        shape_vectors = np.array([self._shape_vector(g) for g in portfolio])
        # Distância euclidiana média entre todos os pares
        if len(shape_vectors) > 1:
            dist_matrix = np.linalg.norm(shape_vectors[:, None] - shape_vectors[None, :], axis=-1)
            np.fill_diagonal(dist_matrix, 0)
            mean_shape_dist = dist_matrix.sum() / (len(shape_vectors) * (len(shape_vectors) - 1))
        else:
            mean_shape_dist = 0

        # PENALIDADE DE CENTRO DE MASSA
        centroid = shape_vectors.mean(axis=0)
        distances_to_centroid = np.linalg.norm(shape_vectors - centroid, axis=1)
        mean_dist_to_centroid = np.mean(distances_to_centroid)
        centroid_penalty = max(0, 2.0 - mean_dist_to_centroid) * 5

        # Penalidade de overlap
        overlap_penalty = 0
        for i in range(len(portfolio)):
            for j in range(i+1, len(portfolio)):
                common = len(set(portfolio[i]) & set(portfolio[j]))
                if common > 11:
                    overlap_penalty += (common - 11) * 2

        # Penalidade de concentração de repetição
        rep_concentration_penalty = 0
        high_rep_count = 0
        for g in portfolio:
            if self.last:
                rep = len(set(g) & set(self.last))
                if rep > 10:
                    high_rep_count += 1
        if high_rep_count > len(portfolio) * 0.3:
            rep_concentration_penalty = (high_rep_count - len(portfolio) * 0.3) * 3

        # Fitness combinado
        fitness = (avg_score * 0.25 +
                   pair_coverage * 15 +
                   triple_coverage * 10 +
                   portfolio_entropy * 10 +
                   mean_shape_dist * 8 -
                   overlap_penalty * 0.5 -
                   centroid_penalty -
                   rep_concentration_penalty)
        return fitness

    def simulated_annealing_portfolio(self, candidates, n_select=30, iterations=200):
        """Simulated Annealing com diversidade de exploração"""
        scored = [(self._learner_score(g), g) for g in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        current_portfolio = []
        for s, g in scored:
            if len(current_portfolio) >= n_select: break
            if not any(len(set(g) & set(sg)) > 11 for sg in current_portfolio):
                current_portfolio.append(g)

        current_fitness = self.portfolio_fitness(current_portfolio)
        best_portfolio = current_portfolio.copy()
        best_fitness = current_fitness

        elite_pool = [g for _, g in scored[:200]]
        random_pool = [g for _, g in scored[200:]] if len(scored) > 200 else elite_pool

        temp = 10.0
        for it in range(iterations):
            temp *= 0.95
            new_portfolio = current_portfolio.copy()
            idx = random.randint(0, len(new_portfolio)-1)

            if random.random() < 0.7 and elite_pool:
                new_game = random.choice(elite_pool)
            elif random_pool:
                new_game = random.choice(random_pool)
            else:
                new_game = random.choice(elite_pool)

            new_portfolio[idx] = new_game
            new_fitness = self.portfolio_fitness(new_portfolio)

            delta = new_fitness - current_fitness
            if delta > 0 or random.random() < np.exp(delta / max(0.1, temp)):
                current_portfolio = new_portfolio
                current_fitness = new_fitness
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_portfolio = current_portfolio.copy()

        return best_portfolio


# ============================================================
# ABLAÇÃO E WALK-FORWARD
# ============================================================
def ablation_study(contests, feature_groups, train_size=300, test_size=50):
    print(f"\n🔬 ABLAÇÃO ESTRUTURAL...")
    results = {}
    test_end = len(contests) - 50
    test_start = test_end - test_size
    train_end = test_start
    train_start = max(0, train_end - train_size)
    if train_start >= train_end: return results
    train_data = contests[train_start:train_end]
    test_data = contests[test_start:test_end]
    context = TemporalContext(train_data, contests[:train_end])
    X, y_hits, _ = context.build_training_dataset(3000)
    learner = StableSHAPLearner()
    learner.train(X, y_hits, None)
    baseline_roi = _quick_eval(learner, context, test_data)
    results['baseline'] = {'roi': baseline_roi, 'dropped': 'nenhum'}
    print(f"   Baseline: ROI={baseline_roi:+.2f}%")

    for group_name, indices in feature_groups.items():
        mask = np.ones(X.shape[1], dtype=bool)
        mask[indices] = False
        X_ablated = X[:, mask]
        learner_ab = StableSHAPLearner()
        learner_ab.train(X_ablated, y_hits, None)
        roi = _quick_eval(learner_ab, context, test_data, feature_mask=mask)
        impact = roi - baseline_roi
        results[group_name] = {'roi': roi, 'impact': impact, 'dropped': group_name}
        print(f"   Sem {group_name:<20}: ROI={roi:+.2f}% (impacto={impact:+.2f}%)")
    return results

def _quick_eval(learner, context, test_data, feature_mask=None, n_games=30):
    strat_payoff = 0
    for tc in test_data:
        actual = set(tc['dezenas'])
        candidates = [sorted(np.random.choice(range(1,26), 15, replace=False)) for _ in range(1000)]
        scored = []
        for g in candidates:
            feats = context.extract_features(g)
            if feature_mask is not None: feats = feats[feature_mask]
            preds = learner.predict(feats)
            scored.append((preds['P12']*20 + preds['P13']*50, g))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, g in scored[:n_games]:
            strat_payoff += PAYOFF.get(len(set(g) & actual), 0)
    total_apostas = len(test_data) * n_games * CUSTO_APOSTA
    return (strat_payoff - total_apostas) / total_apostas * 100 if total_apostas > 0 else 0

def walk_forward_real(contests, n_windows=20, train_size=300, test_size=50):
    print(f"\n🔬 WALK-FORWARD REAL ({n_windows} janelas)...")
    resultados = []
    meta_X, meta_y = [], []
    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue
        historical = contests[:train_end]
        context = TemporalContext(train_data, historical)
        X, y_hits, _ = context.build_training_dataset(4000)
        learner = StableSHAPLearner()
        learner.train(X, y_hits, None)
        strat_payoff, rand_payoff = 0, 0
        n_jogos = 30
        for tc in test_data:
            actual = set(tc['dezenas'])
            candidates = [sorted(np.random.choice(range(1,26), 15, replace=False)) for _ in range(2000)]
            scored = []
            for g in candidates:
                preds = learner.predict(context.extract_features(g))
                scored.append((preds['P12']*20 + preds['P13']*50, g))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, g in scored[:n_jogos]:
                strat_payoff += PAYOFF.get(len(set(g) & actual), 0)
            for _ in range(n_jogos):
                g = sorted(np.random.choice(range(1,26), 15, replace=False))
                rand_payoff += PAYOFF.get(len(set(g) & actual), 0)
        total_apostas = len(test_data) * n_jogos * CUSTO_APOSTA
        strat_roi = (strat_payoff - total_apostas) / total_apostas * 100 if total_apostas > 0 else 0
        rand_roi = (rand_payoff - total_apostas) / total_apostas * 100 if total_apostas > 0 else 0
        had_edge = 1 if strat_payoff > rand_payoff else 0
        meta_X.append([context.repeat_history[-1] if context.repeat_history else 8,
                       np.std(context.soma_history[-10:]) if len(context.soma_history)>=10 else 0,
                       context.dezena_streaks.get(1,0)])
        meta_y.append(had_edge)
        resultados.append({'window': w, 'strat_roi': strat_roi, 'rand_roi': rand_roi, 'diff_roi': strat_roi - rand_roi, 'edge': had_edge})
        print(f" Janela {w}: ROI={strat_roi:+.2f}% vs rand={rand_roi:+.2f}% diff={strat_roi-rand_roi:+.2f}%")
    if resultados:
        diffs = [r['diff_roi'] for r in resultados]
        print(f"\n📊 RESUMO WALK-FORWARD REAL:")
        print(f"   Média diferença ROI: {np.mean(diffs):+.2f}%")
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p-value: {p:.4f}")
        except: pass
        n_pos = sum(1 for d in diffs if d > 0)
        print(f"   Janelas positivas: {n_pos}/{len(resultados)}")
        if len(meta_X) >= 10 and XGB_AVAILABLE:
            meta_model = xgb.XGBClassifier(n_estimators=30, max_depth=3, verbosity=0)
            meta_model.fit(np.array(meta_X), np.array(meta_y))
            print(f"   Meta-modelo treinado com {len(meta_X)} exemplos reais")

    # SALVAR WALK-FORWARD
    with open("walkforward_report.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Relatório walk-forward salvo: walkforward_report.json")
    
    return resultados


# ============================================================
# EXPORTAÇÃO DE RELATÓRIOS
# ============================================================
def export_shap_report(learner, filename="shap_report.json"):
    if learner.feature_importance is None: return
    report = []
    for idx in np.argsort(learner.feature_importance)[::-1]:
        report.append({"feature": FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"f{idx}",
                       "importance": float(learner.feature_importance[idx])})
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 Relatório SHAP salvo: {filename}")

def export_ablation_report(results, filename="ablation_report.json"):
    with open(filename, 'w') as f:
        json.dump({k: {'roi': float(v['roi']), 'impact': float(v.get('impact', 0)), 'dropped': v['dropped']} for k, v in results.items()}, f, indent=2)
    print(f"📄 Relatório de ablação salvo: {filename}")


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 SISTEMA DE OTIMIZAÇÃO DE CARTEIRA v25")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")

    regime_clusterer = RegimeClusterer(contests, n_clusters=4)
    print("\n📊 REGIMES:")
    for s in regime_clusterer.get_regime_stats():
        print(f"   {s['name']}: {s['count']} concursos (Rep:{s['avg_rep']:.1f} Pares:{s['avg_pares']:.1f} Soma:{s['avg_soma']:.0f})")

    print("\nOpções:")
    print("1. Análise SHAP + Ablação + Carteira Otimizada")
    print("2. Walk-forward REAL (20 janelas)")
    print("3. TUDO (completo)")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        context = TemporalContext(contests)
        X, y_hits, _ = context.build_training_dataset(8000)
        learner = StableSHAPLearner()
        learner.train_multiple_rounds(X, y_hits, None, n_rounds=10)
        export_shap_report(learner)

        ablation_results = ablation_study(contests, FEATURE_GROUPS)
        export_ablation_report(ablation_results)

        print("\n🔥 GERANDO CARTEIRA GUIADA v25 (BEAM SEARCH + SA)...")
        generator = GuidedGeneratorV25(context, learner, regime_clusterer)
        candidates = generator.generate_beam_search(n_candidates=5000, beam_width=30)
        print(f"   ✅ {len(candidates)} candidatos gerados (beam search + mutações)")
        portfolio = generator.simulated_annealing_portfolio(candidates, n_select=30, iterations=200)

        print(f"\n🏆 CARTEIRA FINAL:")
        last = contests[-1]['dezenas']
        # CORREÇÃO np.int64
        clean_portfolio = []
        for g in portfolio:
            clean_g = [int(x) for x in sorted(g)]
            clean_portfolio.append(clean_g)

        rep_values = []
        for i, g in enumerate(clean_portfolio, 1):
            rep = len(set(g) & set(last))
            rep_values.append(rep)
            p = sum(1 for d in g if d % 2 == 0)
            pr = sum(1 for d in g if d in PRIMES)
            m = sum(1 for d in g if d in MOLDURA)
            print(f"   {i:2d}. {g} (Rep:{rep} Pares:{p} Primos:{pr} Moldura:{m})")

        print(f"\n📊 MÉTRICAS DA CARTEIRA:")
        print(f"   Repetição média: {np.mean(rep_values):.1f} (histórico ~9)")
        pair_cov = len(set(p for g in clean_portfolio for p in combinations(sorted(g),2))) / comb(25,2)
        print(f"   Cobertura de pares: {pair_cov*100:.1f}%")
        fitness = generator.portfolio_fitness(portfolio)
        print(f"   Fitness global: {fitness:.2f}")

        # Diversidade estrutural
        shapes = np.array([generator._shape_vector(g) for g in clean_portfolio])
        if len(shapes) > 1:
            dist_matrix = np.linalg.norm(shapes[:, None] - shapes[None, :], axis=-1)
            np.fill_diagonal(dist_matrix, 0)
            mean_dist = dist_matrix.sum() / (len(shapes) * (len(shapes) - 1))
            print(f"   Distância média entre formas: {mean_dist:.2f}")

    if op in ("2", "3"):
        walk_forward_real(contests, n_windows=20, train_size=300, test_size=50)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
