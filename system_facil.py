#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE CIÊNCIA DE REGIME - LOTOFÁCIL v28
=============================================
MUDANÇA DE PARADIGMA:
✅ Metadados por janela (condições de regime vs desempenho)
✅ Modelos específicos por regime (um learner por cluster)
✅ Confidence gating (threshold extremo: top 1%, top 0.5%)
✅ Meta-modelo de janela (prevê quando o modelo funciona)
✅ Teste de topologia pura (sem repetição, EMA, fadiga)
✅ Teste cego ampliado (500-1000 concursos)
✅ Correlação condições de regime ↔ edge
✅ Feature de ruído para controle de overfitting
✅ Vazamento temporal corrigido (ctx[:i])
✅ Baseline hipergeométrico teórico
"""

import numpy as np
from scipy.stats import entropy, wilcoxon, norm, hypergeom, pearsonr
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
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
# ============================================================
# UTILITÁRIOS
# ============================================================
def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)

def save_json(data, filename):
    ensure_reports_dir()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"📄 Relatório salvo: {filename}")
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

# Distribuição hipergeométrica teórica
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# ============================================================
# FEATURE NAMES (VERSÃO REDUZIDA PARA TOPOLOGIA)
# ============================================================
FEATURE_NAMES_FULL = [
    "rep_t-1", "delta_rep_t", "rep_t-2", "delta_rep_t-1", "rep_t-3", "acc_rep",
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
    "entropia_rep", "entropia_conjunta", "entropia_transicao",
    "quadrantes", "consecutivos",
    "densidade_local", "assimetria", "clusterizacao",
    "mutual_information", "conditional_entropy",
    "pares_recente", "delta_pares_recente",
    "moldura_recente", "delta_moldura_recente",
    "soma_recente", "delta_soma_recente",
    "random_noise",
]

# Índices para features de TOPOLOGIA apenas
TOPOLOGY_INDICES = list(range(27, 31)) + [31] + list(range(40, 48))  # gaps + energia + entropias + topologia

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
    def __init__(self, contests_slice, historical_contests=None, use_topology_only=False):
        self.contests = contests_slice
        self.n_contests = len(contests_slice)
        self.historical = historical_contests if historical_contests is not None else contests_slice
        self.use_topology_only = use_topology_only
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
        """Extrai features. Se use_topology_only=True, retorna apenas features de topologia."""
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

        if not self.use_topology_only:
            # Todas as features (versão completa)
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
            if len(self.pares_history) >= 2:
                f.extend([float(self.pares_history[-1]), float(self.pares_history[-2]), float(self.pares_history[-1] - self.pares_history[-2])])
            else:
                f.extend([7.5, 7.5, 0.0])
            if len(self.moldura_history) >= 2:
                f.extend([float(self.moldura_history[-1]), float(self.moldura_history[-2]), float(self.moldura_history[-1] - self.moldura_history[-2])])
            else:
                f.extend([9.0, 9.0, 0.0])
            if len(self.soma_history) >= 2:
                f.extend([float(self.soma_history[-1]), float(self.soma_history[-2]), float(self.soma_history[-1] - self.soma_history[-2])])
            else:
                f.extend([195.0, 195.0, 0.0])
            if len(self.energia_history) >= 2:
                f.extend([float(self.energia_history[-1]), float(self.energia_history[-2]), float(self.energia_history[-1] - self.energia_history[-2])])
            else:
                f.extend([30.0, 30.0, 0.0])
            streak9 = 0
            for r in reversed(self.repeat_history):
                if r != 9: streak9 += 1
                else: break
            f.extend([float(streak9), 1.0 / (1.0 + np.exp(-(streak9-5)/3))])
            f.extend([float(rep * pares), float(rep * moldura), float(rep * primos),
                      float(pares * soma / 100.0), float(primos * moldura),
                      float(rep * energia / 10.0), float(pares * moldura)])

        # Features de TOPOLOGIA (sempre incluídas)
        f.extend([float(gap_medio), float(gap_var), float(gap_max), float(gap_min)])
        f.append(float(energia))

        if not self.use_topology_only:
            f.extend([float(np.mean([self.dezena_streaks.get(dd,0) for dd in d])), float(max([self.dezena_streaks.get(dd,0) for dd in d]))])
            f.extend([float(np.mean([self.ema_dezenas.get(dd,0) for dd in d])), float(max([self.ema_dezenas.get(dd,0) for dd in d]))])
            fatigue_scores = []
            for dd in d:
                last_seen = self.dezena_last_seen.get(dd, -1)
                atraso = self.n_contests - 1 - last_seen if last_seen >= 0 else self.n_contests
                freq = self.historical_freq_norm.get(dd, 0.01)
                fatigue_scores.append(atraso * (1.0 - freq))
            f.extend([float(np.mean(fatigue_scores)), float(np.max(fatigue_scores))])
            if len(self.repeat_history) >= 10:
                recent_avg = np.mean(self.repeat_history[-10:])
                global_avg = np.mean(self.repeat_history)
                elasticity = global_avg - recent_avg
                f.extend([float(elasticity), float(abs(elasticity))])
            else:
                f.extend([0.0, 0.0])

        # Entropias (parte da topologia)
        if len(self.repeat_history) >= 10:
            recent = self.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            probs = np.where(probs>0, probs, 1e-10)
            f.append(float(entropy(probs)))
        else:
            f.append(0.0)
        if not self.use_topology_only:
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

        # Diversidade e topologia
        f.extend([float(len(set((x-1)//5 for x in d))), float(consecutivos)])
        local_density = np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15
        f.append(float(local_density))
        f.append(float(np.mean(d) - np.median(d)))
        f.append(float(sum(1 for g in gaps if g <= 2) / len(gaps)))

        if not self.use_topology_only:
            if len(self.repeat_history) >= 5 and len(self.pares_history) >= 5:
                mi_approx = abs(np.corrcoef(self.repeat_history[-5:], self.pares_history[-5:])[0,1])
                f.append(float(mi_approx))
            else:
                f.append(0.0)
            if len(self.repeat_history) >= 5:
                f.append(float(np.std(self.repeat_history[-5:]) / (np.mean(self.repeat_history[-5:])+1)))
            else:
                f.append(0.0)
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
            f.append(float(np.random.randn()))  # ruído

        return np.array(f, dtype=np.float32)

    def get_regime_metadata(self):
        """Extrai metadados do regime atual para correlação com desempenho."""
        meta = {}
        if len(self.repeat_history) >= 20:
            meta['avg_repeat_20'] = float(np.mean(self.repeat_history[-20:]))
            meta['std_repeat_20'] = float(np.std(self.repeat_history[-20:]))
        else:
            meta['avg_repeat_20'] = 8.0
            meta['std_repeat_20'] = 1.0
        if len(self.repeat_history) >= 50:
            meta['avg_entropy_50'] = float(entropy(np.bincount(self.repeat_history[-50:], minlength=13)[5:]/50+1e-10) if len(set(self.repeat_history[-50:]))>1 else 0)
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
        X_list, y_hits_list = [], []
        for i in range(1, len(self.contests)):
            actual = set(self.contests[i]['dezenas'])
            ctx = TemporalContext(self.contests[:i], self.historical, self.use_topology_only)
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
# LEARNER
# ============================================================
class HitsRegressor:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, X, y_hits):
        if X.shape[0] < 100: return False
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0) if XGB_AVAILABLE else RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        for ti, vi in tscv.split(X): self.model.fit(X[ti], y_hits[ti])
        self.is_trained = True
        return True

    def predict(self, features_vector):
        if not self.is_trained: return 7.5
        return float(self.model.predict([features_vector])[0])


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

    def get_regime_labels(self):
        return self.ensemble_labels.tolist() if self.ensemble_labels is not None else []


# ============================================================
# GERADOR COM CONFIDENCE GATING
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

    def apply_confidence_gating(self, candidates, top_percent=1.0):
        """
        Filtra candidatos por confidence gating.
        top_percent: 1.0 = top 1%, 0.5 = top 0.5%
        """
        scored = [(self.learner.predict(self.context.extract_features(g)), g) for g in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        n_select = max(1, int(len(candidates) * top_percent / 100))
        return [g for _, g in scored[:n_select]]

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
        if total == 0: return {'media_hits': 0, 'freq_11_plus': 0, 'freq_12_plus': 0, 'freq_13_plus': 0}
        hits_sum = sum(h * dist[h] for h in range(16))
        return {
            'media_hits': hits_sum / total,
            'freq_11_plus': sum(dist[h] for h in range(11, 16)) / total,
            'freq_12_plus': sum(dist[h] for h in range(12, 16)) / total,
            'freq_13_plus': sum(dist[h] for h in range(13, 16)) / total,
        }


# ============================================================
# BASELINE TEÓRICO E TESTES
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

def permutation_test(strat_vals, rand_vals, n_perm=10000):
    observed = np.mean(strat_vals) - np.mean(rand_vals)
    combined = np.concatenate([strat_vals, rand_vals])
    n1 = len(strat_vals)
    extreme = 0

    for _ in range(n_perm):
        np.random.shuffle(combined)

    perm_diff = (
        np.mean(combined[:n1]) -
        np.mean(combined[n1:]))

    if abs(perm_diff) >= abs(observed):
        extreme += 1
        
    return observed, extreme / n_perm


# ============================================================
# TESTE CEGO COM METADADOS
# ============================================================
def blind_test_with_metadata(contests, blind_size=500, n_games=30, use_topology=False, use_regime_models=False, confidence_gate=100.0):
    """
    Teste cego completo com:
    - Metadados por janela
    - Modelos específicos por regime (se use_regime_models=True)
    - Confidence gating (confidence_gate < 100)
    - Teste de topologia pura (se use_topology=True)
    """
    print(f"\n🔮 TESTE CEGO ({blind_size} concursos)")
    print(f"   Topologia: {use_topology} | Regime: {use_regime_models} | Gate: {confidence_gate}%")
    train_contests = contests[:-blind_size]
    blind_contests = contests[-blind_size:]

    context = TemporalContext(train_contests, use_topology_only=use_topology)
    X, y_hits = context.build_training_dataset(5000)

    regime = RegimeClusterer(train_contests, n_clusters=4)
    current_regime = regime.get_current_regime()

    if use_regime_models:
        # Treinar modelos específicos por regime
        learners_by_regime = {}
        regime_labels = regime.get_regime_labels()
        for cluster_id in range(4):
            mask = np.array(regime_labels) == cluster_id
            if mask.sum() > 50:
                # Filtrar X e y para este regime
                mask = regime_labels[:len(y_hits)] == cluster_id
                X_reg = X[mask]
                y_reg = y_hits[mask]
                learner = HitsRegressor()
                learner.train(X, y_hits)
                learners_by_regime[cluster_id] = learner
        learner = learners_by_regime.get(current_regime, HitsRegressor())
        if not learner.is_trained:
            learner.train(X, y_hits)
    else:
        learner = HitsRegressor()
        learner.train(X, y_hits)

    gen = PortfolioGenerator(context, learner)
    candidates = gen.generate_beam_search(n_candidates=3000, beam_width=30)

    # Confidence gating
    if confidence_gate < 100:
        candidates = gen.apply_confidence_gating(candidates, top_percent=confidence_gate)
        n_games_actual = len(candidates)
    else:
        n_games_actual = n_games
        candidates = candidates[:n_games]

    # Avaliar
    dist, total = gen.evaluate_distribution(candidates[:n_games_actual], blind_contests)
    metrics = gen.compute_metrics(dist, total)
    theo = theoretical_baseline_metrics()

    # Hits por sorteio
    strat_hits = []
    rand_hits = []
    for draw in blind_contests:
        actual = set(draw['dezenas'])
        s = sum(len(set(g) & actual) for g in candidates[:n_games_actual])
        r = sum(len(set(g) & actual) for g in [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(n_games_actual)])
        strat_hits.append(s)
        rand_hits.append(r)

    boot_l, boot_u, boot_m = bootstrap_ci(strat_hits)
    perm_diff, perm_p = permutation_test(strat_hits, rand_hits)

    # Metadados do regime atual
    meta = context.get_regime_metadata()

    print(f"\n📊 RESULTADOS:")
    print(f"   Regime atual: {regime.regime_names.get(current_regime, {}).get('name', '?')}")
    print(f"   {'Métrica':<20} {'Estratégia':<15} {'Teórico':<15} {'Diferença':<15}")
    print(f"   {'Média hits/jogo':<20} {metrics['media_hits']:<15.4f} {theo['media_hits']:<15.4f} {metrics['media_hits']-theo['media_hits']:+.4f}")
    print(f"   {'Freq 11+':<20} {metrics['freq_11_plus']:<15.4f} {theo['freq_11_plus']:<15.4f} {metrics['freq_11_plus']-theo['freq_11_plus']:+.4f}")
    print(f"   Bootstrap IC 95%: [{boot_l:.1f}, {boot_u:.1f}]")
    print(f"   Permutation p: {perm_p:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "metrics": metrics,
        "theoretical": theo,
        "permutation_p": perm_p,
        "metadata": meta,
        "current_regime": int(current_regime),
        "use_topology": use_topology,
        "use_regime_models": use_regime_models,
        "confidence_gate": confidence_gate,
        "bootstrap_ci": [boot_l, boot_u, boot_m],
        "distribution": dist
    }

    save_json(report,
        f"reports/blind_test_v28_{timestamp}.json")
    
    return metrics, theo, perm_p, meta, current_regime


# ============================================================
# WALK-FORWARD COM METADADOS
# ============================================================
def walk_forward_with_metadata(contests, n_windows=30, train_size=300, test_size=50, use_topology=False, use_regime_models=False, confidence_gate=100.0):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas) com metadados...")
    resultados = []
    window_metas = []

    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue

        context = TemporalContext(train_data, contests[:train_end], use_topology_only=use_topology)
        X, y_hits = context.build_training_dataset(3000)
        learner = HitsRegressor()
        learner.train(X, y_hits)

        gen = PortfolioGenerator(context, learner)
        candidates = gen.generate_beam_search(2000, 30)

        if confidence_gate < 100:
            candidates = gen.apply_confidence_gating(candidates, top_percent=confidence_gate)
            n_actual = len(candidates)
        else:
            n_actual = 30
            candidates = candidates[:30]

        dist, total = gen.evaluate_distribution(candidates[:n_actual], test_data)
        metrics = gen.compute_metrics(dist, total)

        rand_dist, rand_total = gen.evaluate_distribution(
            [sorted(np.random.choice(range(1,26),15,replace=False)) for _ in range(n_actual)], test_data)
        rand_metrics = gen.compute_metrics(rand_dist, rand_total)
        theo = theoretical_baseline_metrics()

        meta = context.get_regime_metadata()
        window_metas.append(meta)

        resultados.append({
            'window': w,
            'strat_11': metrics['freq_11_plus'],
            'rand_11': rand_metrics['freq_11_plus'],
            'theo_11': theo['freq_11_plus'],
            'diff_11': metrics['freq_11_plus'] - rand_metrics['freq_11_plus'],
            'diff_theo': metrics['freq_11_plus'] - theo['freq_11_plus'],
            **meta
        })
        print(f" Janela {w}: diff={metrics['freq_11_plus']-rand_metrics['freq_11_plus']:+.4f} "
              f"rep={meta.get('avg_repeat_20',0):.1f} ent={meta.get('avg_entropy_50',0):.2f}")

    if resultados:
        diffs = [r['diff_11'] for r in resultados]
        print(f"\n📊 RESUMO:")
        print(f"   Média diff: {np.mean(diffs):+.4f}")
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p: {p:.4f}")
        except: pass
        n_pos = sum(1 for d in diffs if d > 0)
        print(f"   Janelas +: {n_pos}/{len(resultados)}")

        # CORRELAÇÃO: condições de regime vs desempenho
        print(f"\n📊 CORRELAÇÃO REGIME vs DESEMPENHO:")
        for key in ['avg_repeat_20', 'std_repeat_20', 'avg_entropy_50', 'avg_gap_20', 'std_gap_20', 'avg_energy_20', 'std_pares_20']:
            values = [r.get(key, 0) for r in resultados if key in r]
            if len(values) >= 5 and len(diffs) >= len(values):
                corr, pval = pearsonr(values[:len(diffs)], diffs[:len(values)])
                sig = "🔴" if pval < 0.01 else "🟡" if pval < 0.05 else "🟢"
                print(f"   {key:<25} r={corr:+.3f} p={pval:.4f} {sig}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "results": resultados,
        "mean_diff": float(np.mean(diffs)),
        "positive_windows": int(n_pos),
        "total_windows": len(resultados),
    }

    if 'p' in locals():
        summary["wilcoxon_p"] = float(p)

    save_json( summary,
        f"reports/walkforward_v28_{timestamp}.json" )
    
    return resultados, window_metas


# ============================================================
# META-MODELO DE JANELA
# ============================================================
def train_meta_model(resultados):
    """Treina meta-modelo para prever quando o modelo funciona."""
    if len(resultados) < 20: return None
    meta_X = []
    meta_y = []
    for r in resultados:
        feats = [
            r.get('avg_repeat_20', 8),
            r.get('std_repeat_20', 1),
            r.get('avg_entropy_50', 1),
            r.get('avg_gap_20', 2),
            r.get('std_gap_20', 0.5),
            r.get('avg_energy_20', 30),
            r.get('std_pares_20', 1.5),
        ]
        meta_X.append(feats)
        meta_y.append(1 if r['diff_11'] > 0 else 0)

    meta_X = np.array(meta_X)
    meta_y = np.array(meta_y)

    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(meta_X, meta_y)

    # Importância das features do meta-modelo
    meta_names = ['avg_repeat_20', 'std_repeat_20', 'avg_entropy_50', 'avg_gap_20', 'std_gap_20', 'avg_energy_20', 'std_pares_20']
    print(f"\n📊 META-MODELO TREINADO:")
    if hasattr(model, 'feature_importances_'):
        for name, imp in zip(meta_names, model.feature_importances_):
            print(f"   {name}: {imp:.3f}")
    if hasattr(model, 'feature_importances_'):
        meta_importance = {
            name: float(imp)
            for name, imp in zip(meta_names, model.feature_importances_)
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_json(
            meta_importance,
            f"reports/meta_model_importance_{timestamp}.json"
        )
    return model


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 SISTEMA DE CIÊNCIA DE REGIME v28")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")
    ensure_reports_dir()
    random.seed(42)
    np.random.seed(42)
    regime_clusterer = RegimeClusterer(contests, n_clusters=4)
    print("\n📊 REGIMES:")
    for s in regime_clusterer.get_regime_stats():
        print(f"   {s['name']}: {s['count']} concursos (Rep:{s['avg_rep']:.1f} Pares:{s['avg_pares']:.1f} Soma:{s['avg_soma']:.0f})")

    print("\nOpções:")
    print("1. Teste cego (500 concursos)")
    print("2. Walk-forward com metadados (30 janelas)")
    print("3. Teste de TOPOLOGIA pura (sem repetição/EMA/fadiga)")
    print("4. Confidence gating (top 1%)")
    print("5. TUDO")
    op = input("Escolha [5]: ").strip() or "5"

    if op in ("1", "5"):
        blind_test_with_metadata(contests, blind_size=500, n_games=30)

    if op in ("2", "5"):
        resultados, metas = walk_forward_with_metadata(contests, n_windows=30, train_size=300, test_size=50)
        if resultados:
            train_meta_model(resultados)

    if op in ("3", "5"):
        print("\n🔬 TESTE DE TOPOLOGIA PURA...")
        blind_test_with_metadata(contests, blind_size=500, n_games=30, use_topology=True)
        resultados_topo, _ = walk_forward_with_metadata(contests, n_windows=20, train_size=300, test_size=50, use_topology=True)

    if op in ("4", "5"):
        print("\n🔬 CONFIDENCE GATING (top 1%)...")
        blind_test_with_metadata(contests, blind_size=500, n_games=30, confidence_gate=1.0)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
