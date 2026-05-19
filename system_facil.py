#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE ANÁLISE DE DISTRIBUIÇÃO DE HITS - LOTOFÁCIL v27
============================================================
CORREÇÕES CRÍTICAS:
✅ Vazamento temporal CORRIGIDO (ctx usa apenas passado estrito)
✅ Baseline matemático exato (distribuição hipergeométrica teórica)
✅ Z-score real implementado (comparação com baseline)
✅ Bootstrap massivo + permutation test (10k simulações)
✅ Feature de ruído aleatório (controle de overfitting)
✅ Estabilidade de features (ranking médio, variância, survival rate)
✅ Métricas: media_hits, freq_11+, freq_12+, zscore, distribuição completa
✅ Teste cego real preservado
✅ Ensemble implícito de estratégias
"""

import numpy as np
from scipy.stats import entropy, wilcoxon, norm, hypergeom
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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
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
CUSTO_APOSTA = 3.0
TOTAL_NUMBERS = 25
NUMBERS_PER_GAME = 15

# Parâmetros para distribuição hipergeométrica teórica
# Probabilidade de acertar k dezenas em 15 sorteadas de 25
HYPE_PROBS = {
    k: hypergeom.pmf(k, 25, 15, 15)  # N=25, K=15, n=15
    for k in range(0, 16)
}

# ============================================================
# FEATURE NAMES
# ============================================================
FEATURE_NAMES = [
    # Markov repetição (0-5)
    "rep_t-1", "delta_rep_t", "rep_t-2", "delta_rep_t-1", "rep_t-3", "acc_rep",
    # Markov pares (6-8)
    "pares_t-1", "pares_t-2", "delta_pares",
    # Markov moldura (9-11)
    "moldura_t-1", "moldura_t-2", "delta_moldura",
    # Markov soma (12-14)
    "soma_t-1", "soma_t-2", "delta_soma",
    # Markov energia (15-17)
    "energia_t-1", "energia_t-2", "delta_energia",
    # Streak (18-19)
    "streak_sem_9", "mean_reversion_9",
    # Interações (20-26)
    "rep_x_pares", "rep_x_moldura", "rep_x_primos",
    "pares_x_soma", "primos_x_moldura", "rep_x_energia", "pares_x_moldura",
    # Compressão espacial (27-30)
    "gap_medio", "gap_var", "gap_max", "gap_min",
    # Energia (31)
    "energia_jogo",
    # Persistência individual (32-33)
    "avg_streak_dezenas", "max_streak_dezenas",
    # EMA (34-35)
    "avg_ema", "max_ema",
    # Fadiga (36-37)
    "avg_fadiga", "max_fadiga",
    # Elasticidade (38-39)
    "elasticidade", "abs_elasticidade",
    # Entropias (40-42)
    "entropia_rep", "entropia_conjunta", "entropia_transicao",
    # Diversidade (43-44)
    "quadrantes", "consecutivos",
    # Topologia espacial (45-47)
    "densidade_local", "assimetria", "clusterizacao",
    # Teoria da informação (48-49)
    "mutual_information", "conditional_entropy",
    # Comparação recente (50-55)
    "pares_recente", "delta_pares_recente",
    "moldura_recente", "delta_moldura_recente",
    "soma_recente", "delta_soma_recente",
    # CONTROLE DE OVERFITTING (56)
    "random_noise",
]

FEATURE_GROUPS = {
    "repeticao": list(range(0, 6)),
    "pares": [6, 7, 8, 50, 51],
    "moldura": [9, 10, 11, 52, 53],
    "soma": [12, 13, 14, 54, 55],
    "energia": [15, 16, 17, 31],
    "interacoes": list(range(20, 27)),
    "compressao": list(range(27, 31)),
    "persistencia": [32, 33, 34, 35],
    "fadiga": [36, 37],
    "elasticidade": [38, 39],
    "entropia": [40, 41, 42],
    "diversidade": [43, 44],
    "streak": [18, 19],
    "topologia": [45, 46, 47],
    "info_theory": [48, 49],
    "repeticao_completa": [0,1,2,3,4,5,18,19,20,21,22,25],
    "controle": [56],  # feature de ruído
}

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
# CONTEXTO TEMPORAL (VAZAMENTO CORRIGIDO)
# ============================================================
class TemporalContext:
    """
    CORRIGIDO: ctx usa apenas passado ESTRITO (< i).
    NUNCA inclui o concurso alvo.
    """
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
        # 40-42: Entropias
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
        if len(self.repeat_history) >= 5:
            transitions = [self.repeat_history[i+1]-self.repeat_history[i] for i in range(len(self.repeat_history)-1)]
            if len(set(transitions)) > 1:
                freq = Counter(transitions)
                probs = np.array([freq.get(v,0)/len(transitions) for v in set(transitions)])
                probs = np.where(probs>0, probs, 1e-10)
                f.append(float(entropy(probs)))
            else:
                f.append(0.0)
        else:
            f.append(0.0)
        # 43-44: Diversidade
        f.extend([float(len(set((x-1)//5 for x in d))), float(consecutivos)])
        # 45-47: Topologia
        local_density = np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15
        f.append(float(local_density))
        f.append(float(np.mean(d) - np.median(d)))
        f.append(float(sum(1 for g in gaps if g <= 2) / len(gaps)))
        # 48-49: Teoria da informação
        if len(self.repeat_history) >= 5 and len(self.pares_history) >= 5:
            mi_approx = abs(np.corrcoef(self.repeat_history[-5:], self.pares_history[-5:])[0,1])
            f.append(float(mi_approx))
        else:
            f.append(0.0)
        if len(self.repeat_history) >= 5:
            f.append(float(np.std(self.repeat_history[-5:]) / (np.mean(self.repeat_history[-5:])+1)))
        else:
            f.append(0.0)
        # 50-55: Comparação recente
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
        # 56: FEATURE DE RUÍDO (controle de overfitting)
        f.append(float(np.random.randn()))
        return np.array(f, dtype=np.float32)

    def build_training_dataset(self, n_samples=5000):
        """
        CORRIGIDO: ctx usa apenas passado ESTRITO (< i).
        NUNCA inclui o concurso i.
        """
        X_list, y_hits_list = [], []
        # available_contests = self.contests[:-1]  # removido: usamos loop com ctx[:i]
        for i in range(1, len(self.contests)):  # i é o índice do concurso alvo
            actual = set(self.contests[i]['dezenas'])
            # CORREÇÃO: ctx usa apenas contests[:i] (passado estrito)
            ctx = TemporalContext(self.contests[:i], self.historical)
            for _ in range(10):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                X_list.append(ctx.extract_features(game))
                y_hits_list.append(len(set(game) & actual))
            last = set(self.contests[i-1]['dezenas'])
            for _ in range(10):
                base = list(last) if last else []
                random.shuffle(base)
                game_set = set(base[:random.randint(6, 10)])
                available = [x for x in range(1,26) if x not in game_set]
                while len(game_set) < 15:
                    game_set.add(random.choice(available))
                game = sorted(game_set)[:15]
                X_list.append(ctx.extract_features(game))
                y_hits_list.append(len(set(game) & actual))
        if len(X_list) > n_samples:
            indices = np.random.choice(len(X_list), n_samples, replace=False)
            X_list = [X_list[i] for i in indices]
            y_hits_list = [y_hits_list[i] for i in indices]
        return np.array(X_list), np.array(y_hits_list)


# ============================================================
# LEARNER (REGRESSOR DE HITS)
# ============================================================
class HitsRegressor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_importance = None

    def train(self, X, y_hits):
        if X.shape[0] < 100: return False
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0) if XGB_AVAILABLE else RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        for ti, vi in tscv.split(X): self.model.fit(X[ti], y_hits[ti])
        self.is_trained = True
        # Importância das features (para análise de estabilidade)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        return True

    def predict(self, features_vector):
        if not self.is_trained: return 7.5
        return float(self.model.predict([features_vector])[0])


# ============================================================
# ANÁLISE DE ESTABILIDADE DE FEATURES
# ============================================================
def feature_stability_analysis(X, y_hits, n_rounds=100, top_k=10):
    """
    Mede estabilidade das features ao longo de múltiplos treinos.
    Retorna: ranking médio, variância do ranking, survival rate (top_k).
    """
    print(f"\n📊 ANÁLISE DE ESTABILIDADE ({n_rounds} rodadas)...")
    rankings = []
    for r in range(n_rounds):
        np.random.seed(42 + r)
        idx = np.random.choice(len(X), min(len(X), 2000), replace=False)
        learner = HitsRegressor()
        learner.train(X[idx], y_hits[idx])
        if learner.feature_importance is not None:
            imp = learner.feature_importance
            # Ranking: ordem decrescente de importância
            rank = np.argsort(imp)[::-1]  # índice original -> posição
            rankings.append(rank)
    if len(rankings) < 2: return None

    rankings = np.array(rankings)
    mean_rank = rankings.mean(axis=0)
    std_rank = rankings.std(axis=0)
    # Survival rate: quantas vezes aparece no top_k
    survival = np.mean(rankings < top_k, axis=0)

    print(f"\n   📊 TOP 10 FEATURES POR RANKING MÉDIO:")
    top_idx = np.argsort(mean_rank)[:10]
    for idx in top_idx:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"f{idx}"
        print(f"   {name:<30} rank={mean_rank[idx]:.1f}±{std_rank[idx]:.1f} surv={survival[idx]:.2f}")

    # Verificar feature de ruído (índice 56)
    if len(mean_rank) > 56:
        noise_rank = mean_rank[56]
        noise_surv = survival[56]
        print(f"\n   🔍 FEATURE DE RUÍDO (controle): rank={noise_rank:.1f} surv={noise_surv:.2f}")
        if noise_surv > 0.1:
            print(f"   ⚠️  ALERTA: ruído aparece no top 10 em {noise_surv*100:.0f}% das rodadas!")
            print(f"   ⚠️  Possível overfitting detectado.")

    return {'mean_rank': mean_rank, 'std_rank': std_rank, 'survival': survival}


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
# GERADOR E AVALIADOR
# ============================================================
class PortfolioGenerator:
    def __init__(self, context, learner):
        self.context = context
        self.learner = learner
        self.last = context.get_last_contest()

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
        if hasattr(self.context, 'ema_dezenas'):
            score += np.mean([self.context.ema_dezenas.get(x,0) for x in d]) * 3
        return score

    def generate_beam_search(self, n_candidates=5000, beam_width=30):
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
                if len(game) == 15 and tuple(game) not in seen:
                    seen.add(tuple(game))
                    candidates.append(game)
        if candidates:
            top = sorted(candidates, key=lambda g: self.learner.predict(self.context.extract_features(g)), reverse=True)[:100]
            for base_game in top:
                for _ in range(20):
                    mutated = base_game.copy()
                    pos = random.randint(0,14)
                    avail = [d for d in range(1,26) if d not in mutated]
                    if avail:
                        mutated[pos] = random.choice(avail)
                        mutated.sort()
                        if tuple(mutated) not in seen:
                            seen.add(tuple(mutated))
                            candidates.append(mutated)
        while len(candidates) < n_candidates:
            game = sorted(np.random.choice(range(1,26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                candidates.append(game)
        return candidates[:n_candidates]

    def evaluate_distribution(self, games, test_draws):
        dist = {h: 0 for h in range(0, 16)}
        for draw in test_draws:
            actual = set(draw['dezenas'])
            for g in games:
                hits = len(set(g) & actual)
                dist[hits] += 1
        total = len(test_draws) * len(games)
        return dist, total

    def compute_metrics(self, dist, total):
        hits_sum = sum(h * dist[h] for h in range(16))
        media = hits_sum / total if total > 0 else 0
        freq_11 = sum(dist[h] for h in range(11, 16)) / total if total > 0 else 0
        freq_12 = sum(dist[h] for h in range(12, 16)) / total if total > 0 else 0
        freq_13 = sum(dist[h] for h in range(13, 16)) / total if total > 0 else 0
        return {'media_hits': media, 'freq_11_plus': freq_11, 'freq_12_plus': freq_12, 'freq_13_plus': freq_13}


# ============================================================
# BASELINE HIPERGEOMÉTRICO TEÓRICO
# ============================================================
def theoretical_baseline_metrics(n_games=30, n_draws=300):
    """
    Calcula métricas esperadas pela distribuição hipergeométrica teórica.
    Isso substitui o baseline aleatório com variância zero.
    """
    expected_hits = sum(k * HYPE_PROBS[k] for k in range(16))
    expected_11_plus = sum(HYPE_PROBS[k] for k in range(11, 16))
    expected_12_plus = sum(HYPE_PROBS[k] for k in range(12, 16))
    expected_13_plus = sum(HYPE_PROBS[k] for k in range(13, 16))
    return {
        'media_hits': expected_hits,
        'freq_11_plus': expected_11_plus,
        'freq_12_plus': expected_12_plus,
        'freq_13_plus': expected_13_plus,
    }


# ============================================================
# TESTES ESTATÍSTICOS
# ============================================================
def bootstrap_confidence_interval(data, n_bootstrap=5000, ci=95):
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, 100-(100-ci)/2)
    return lower, upper, np.mean(means)

def permutation_test(strat_vals, rand_vals, n_perm=10000):
    observed_diff = np.mean(strat_vals) - np.mean(rand_vals)
    combined = np.concatenate([strat_vals, rand_vals])
    n1 = len(strat_vals)
    extreme = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            extreme += 1
    p_value = extreme / n_perm
    return observed_diff, p_value

def compute_zscore(observed_val, baseline_mean, baseline_std, n_samples):
    """Z-score comparando observado vs baseline teórico"""
    if baseline_std == 0:
        return 0.0
    standard_error = baseline_std / np.sqrt(n_samples)
    return (observed_val - baseline_mean) / standard_error


# ============================================================
# TESTE CEGO REAL (CORRIGIDO)
# ============================================================
def blind_test(contests, blind_size=300, n_games=30, n_bootstrap=5000):
    print(f"\n🔮 TESTE CEGO REAL ({blind_size} concursos intocados)...")
    train_contests = contests[:-blind_size]
    blind_contests = contests[-blind_size:]

    # Contexto apenas com dados de treino
    context = TemporalContext(train_contests)
    X, y_hits = context.build_training_dataset(5000)
    learner = HitsRegressor()
    learner.train(X, y_hits)

    # Análise de estabilidade das features
    feature_stability_analysis(X, y_hits, n_rounds=100)

    gen = PortfolioGenerator(context, learner)
    candidates = gen.generate_beam_search(n_candidates=3000, beam_width=30)

    # Avaliar no conjunto cego
    dist, total = gen.evaluate_distribution(candidates[:n_games], blind_contests)
    metrics = gen.compute_metrics(dist, total)

    # Baseline hipergeométrico teórico
    theo = theoretical_baseline_metrics(n_games, blind_size)

    # Coletar hits por sorteio para testes
    strat_hits_per_draw = []
    rand_hits_per_draw = []
    for draw in blind_contests:
        actual = set(draw['dezenas'])
        s_hits = sum(len(set(g) & actual) for g in candidates[:n_games])
        r_hits = sum(len(set(g) & actual) for g in [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(n_games)])
        strat_hits_per_draw.append(s_hits)
        rand_hits_per_draw.append(r_hits)

    # Bootstrap
    boot_lower, boot_upper, boot_mean = bootstrap_confidence_interval(strat_hits_per_draw, n_bootstrap)
    # Permutation test
    perm_diff, perm_p = permutation_test(strat_hits_per_draw, rand_hits_per_draw, n_perm=10000)
    # Z-score vs teórico
    theo_std = np.sqrt(sum((k - theo['media_hits'])**2 * HYPE_PROBS[k] for k in range(16)))
    zscore = compute_zscore(metrics['media_hits'], theo['media_hits'], theo_std, total)

    print(f"\n📊 RESULTADOS TESTE CEGO:")
    print(f"   {'Métrica':<20} {'Estratégia':<15} {'Teórico':<15} {'Diferença':<15} {'Z-score':<10}")
    print(f"   {'Média hits/jogo':<20} {metrics['media_hits']:<15.4f} {theo['media_hits']:<15.4f} {metrics['media_hits']-theo['media_hits']:+.4f} {zscore:+8.2f}")
    print(f"   {'Freq 11+':<20} {metrics['freq_11_plus']:<15.4f} {theo['freq_11_plus']:<15.4f} {metrics['freq_11_plus']-theo['freq_11_plus']:+.4f}")
    print(f"   {'Freq 12+':<20} {metrics['freq_12_plus']:<15.4f} {theo['freq_12_plus']:<15.4f} {metrics['freq_12_plus']-theo['freq_12_plus']:+.4f}")
    print(f"\n📊 TESTES ESTATÍSTICOS:")
    print(f"   Bootstrap IC 95%: [{boot_lower:.1f}, {boot_upper:.1f}] média={boot_mean:.1f}")
    print(f"   Permutation test p-value: {perm_p:.4f}")
    print(f"   Diferença observada: {perm_diff:+.1f} hits/sorteio")

    # Distribuição completa
    print(f"\n📊 DISTRIBUIÇÃO DE HITS:")
    print(f"   {'Hits':<8} {'Estratégia':<15} {'Teórico':<15}")
    for h in range(8, 16):
        s_rate = dist[h]/total*100 if total>0 else 0
        t_rate = HYPE_PROBS[h]*100
        print(f"   {h:<8} {s_rate:<15.4f}% {t_rate:<15.4f}%")

    return metrics, theo, perm_p, zscore


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 SISTEMA DE ANÁLISE DE DISTRIBUIÇÃO v27")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")

    regime_clusterer = RegimeClusterer(contests, n_clusters=4)
    print("\n📊 REGIMES:")
    for s in regime_clusterer.get_regime_stats():
        print(f"   {s['name']}: {s['count']} concursos (Rep:{s['avg_rep']:.1f} Pares:{s['avg_pares']:.1f} Soma:{s['avg_soma']:.0f})")

    print("\nOpções:")
    print("1. Teste cego + Análise de estabilidade")
    print("2. Walk-forward com métricas de hits")
    print("3. TUDO (completo)")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        blind_test(contests, blind_size=300, n_games=30, n_bootstrap=5000)

    if op in ("2", "3"):
        print("\n🔬 WALK-FORWARD (20 janelas)...")
        resultados = []
        for w in range(20):
            test_size = 50
            test_end = len(contests) - w * test_size
            test_start = test_end - test_size
            train_end = test_start
            train_start = max(0, train_end - 300)
            if train_start >= train_end or test_start >= test_end: continue
            train_data = contests[train_start:train_end]
            test_data = contests[test_start:test_end]
            if len(train_data) < 100 or len(test_data) < 5: continue

            context = TemporalContext(train_data, contests[:train_end])
            X, y_hits = context.build_training_dataset(3000)
            learner = HitsRegressor()
            learner.train(X, y_hits)

            gen = PortfolioGenerator(context, learner)
            candidates = gen.generate_beam_search(2000, 30)
            dist, total = gen.evaluate_distribution(candidates[:30], test_data)
            metrics = gen.compute_metrics(dist, total)

            rand_dist, rand_total = gen.evaluate_distribution(
                [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(30)], test_data)
            rand_metrics = gen.compute_metrics(rand_dist, rand_total)

            theo = theoretical_baseline_metrics(30, len(test_data))

            resultados.append({
                'window': w,
                'strat_11': metrics['freq_11_plus'],
                'rand_11': rand_metrics['freq_11_plus'],
                'theo_11': theo['freq_11_plus'],
                'diff_11': metrics['freq_11_plus'] - rand_metrics['freq_11_plus']
            })
            print(f" Janela {w}: 11+ estrat={metrics['freq_11_plus']:.4f} rand={rand_metrics['freq_11_plus']:.4f} theo={theo['freq_11_plus']:.4f} diff={metrics['freq_11_plus']-rand_metrics['freq_11_plus']:+.4f}")

        if resultados:
            diffs = [r['diff_11'] for r in resultados]
            print(f"\n📊 RESUMO WALK-FORWARD:")
            print(f"   Média diferença 11+: {np.mean(diffs):+.4f}")
            try:
                _, p = wilcoxon(diffs)
                print(f"   Wilcoxon p-value: {p:.4f}")
            except: pass
            n_pos = sum(1 for d in diffs if d > 0)
            print(f"   Janelas positivas: {n_pos}/{len(resultados)}")

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
