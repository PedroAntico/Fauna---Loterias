#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA COMPLETO DE ANÁLISE E OTIMIZAÇÃO - LOTOFÁCIL v18
==========================================================
MELHORIAS INCORPORADAS:
✅ Features de interação (rep * pares, rep * moldura, pares * soma)
✅ Compressão espacial (gaps: médio, variância, máximo, mínimo)
✅ Energia estrutural (soma das diferenças absolutas consecutivas)
✅ Persistência individual de dezenas (streaks)
✅ Frequência exponencial (EMA por dezena)
✅ Múltiplos targets: P(>=11), P(>=12), P(>=13), payoff esperado
✅ Hard negative mining (exemplos difíceis)
✅ Walk-forward validation obrigatória com ROI e estabilidade
✅ Separação clara: Feature Engine → Learner → Otimizador de Carteira
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, wilcoxon
from scipy.spatial.distance import jaccard
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
from math import comb
from tqdm import tqdm
import random
import struct
import zlib

warnings.filterwarnings('ignore')

# ============================================================
# IMPORTS OPCIONAIS (com fallback)
# ============================================================
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost não instalado. Use: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
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
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None

# ============================================================
# FEATURE ENGINE AVANÇADO (v18)
# ============================================================
class AdvancedFeatureEngine:
    """
    Extrai features temporais e estruturais para aprendizagem.
    """
    def __init__(self, contests):
        self.contests = contests
        self.n_contests = len(contests)

        # Históricos
        self.repeat_history = []
        self.pares_history = []
        self.primos_history = []
        self.moldura_history = []
        self.soma_history = []
        self.gap_media_history = []
        self.gap_var_history = []
        self.energia_history = []

        # Frequências para EMA e fadiga
        self.dezena_counts = Counter()
        self.ema_dezenas = {d: 0.0 for d in range(1, 26)}
        self.ema_alpha = 0.3  # fator de suavização

        # Persistência individual
        self.dezena_streaks = {d: 0 for d in range(1, 26)}

        # Preencher históricos
        for i in range(len(contests)):
            d = contests[i]['dezenas']
            self.pares_history.append(sum(1 for x in d if x % 2 == 0))
            self.primos_history.append(sum(1 for x in d if x in PRIMES))
            self.moldura_history.append(sum(1 for x in d if x in MOLDURA))
            self.soma_history.append(sum(d))

            # Gaps e energia
            sd = sorted(d)
            gaps = [sd[j+1]-sd[j] for j in range(len(sd)-1)]
            self.gap_media_history.append(np.mean(gaps))
            self.gap_var_history.append(np.var(gaps))
            self.energia_history.append(sum(abs(sd[j]-sd[j-1]) for j in range(1, len(sd))))

            if i > 0:
                prev = set(contests[i-1]['dezenas'])
                curr = set(d)
                self.repeat_history.append(len(prev & curr))
            else:
                self.repeat_history.append(0)

            # Atualizar EMA e streaks
            for num in range(1, 26):
                in_current = 1 if num in d else 0
                self.ema_dezenas[num] = (self.ema_alpha * in_current +
                                        (1 - self.ema_alpha) * self.ema_dezenas[num])
                if in_current:
                    self.dezena_streaks[num] += 1
                else:
                    self.dezena_streaks[num] = 0

            self.dezena_counts.update(d)

        # Frequências normalizadas
        total = len(contests)
        self.dezena_freq_norm = {d: f/total for d, f in self.dezena_counts.items()}

    def extract_features(self, game, idx):
        """
        Retorna vetor de features para um jogo candidato no contexto do concurso idx.
        idx é o índice do concurso de referência (passado mais recente).
        """
        features = []
        d = sorted(game)

        # --- Básicos do jogo ---
        pares = sum(1 for x in d if x % 2 == 0)
        primos = sum(1 for x in d if x in PRIMES)
        moldura = sum(1 for x in d if x in MOLDURA)
        centro = sum(1 for x in d if x in CENTRO)
        soma = sum(d)
        amplitude = max(d) - min(d)
        consecutivos = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)

        # Gaps do jogo
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        gap_medio = np.mean(gaps)
        gap_var = np.var(gaps)
        gap_max = max(gaps)
        gap_min = min(gaps)

        # Energia do jogo
        energia = sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))

        # Repetição (em relação ao último concurso)
        if idx > 0 and idx < len(self.contests):
            last = set(self.contests[idx]['dezenas'])
        else:
            last = set()
        rep = len(set(d) & last) if last else 8

        # --- Features temporais (baseadas nos históricos) ---
        # Repetição: markov 1,2,3, velocidade, aceleração
        if len(self.repeat_history) >= 1:
            prev_rep = self.repeat_history[-1]
            features.append(float(prev_rep))
            features.append(float(rep - prev_rep))
        else:
            features.extend([8.0, 0.0])

        if len(self.repeat_history) >= 2:
            prev2_rep = self.repeat_history[-2]
            features.append(float(prev2_rep))
            features.append(float(self.repeat_history[-1] - prev2_rep))
        else:
            features.extend([8.0, 0.0])

        if len(self.repeat_history) >= 3:
            features.append(float(self.repeat_history[-3]))
            # Aceleração
            d1 = self.repeat_history[-1] - self.repeat_history[-2]
            d2 = self.repeat_history[-2] - self.repeat_history[-3]
            features.append(float(d1 - d2))
        else:
            features.extend([8.0, 0.0])

        # Streak sem 9 repetidas e mean reversion
        streak9 = 0
        for r in reversed(self.repeat_history):
            if r != 9: streak9 += 1
            else: break
        features.append(float(streak9))
        features.append(1.0 / (1.0 + np.exp(-(streak9-5)/3)))

        # --- Interações (rep * pares, etc.) ---
        features.append(float(rep * pares))
        features.append(float(rep * moldura))
        features.append(float(pares * soma / 100.0))
        features.append(float(primos * moldura))

        # --- Compressão espacial (gaps) ---
        features.append(float(gap_medio))
        features.append(float(gap_var))
        features.append(float(gap_max))
        features.append(float(gap_min))

        # --- Energia estrutural ---
        features.append(float(energia))

        # --- Persistência individual de dezenas (streaks) ---
        avg_streak = np.mean([self.dezena_streaks.get(dd, 0) for dd in d])
        max_streak = max([self.dezena_streaks.get(dd, 0) for dd in d])
        features.append(float(avg_streak))
        features.append(float(max_streak))

        # --- Frequência exponencial (EMA) ---
        avg_ema = np.mean([self.ema_dezenas.get(dd, 0) for dd in d])
        max_ema = max([self.ema_dezenas.get(dd, 0) for dd in d])
        features.append(float(avg_ema))
        features.append(float(max_ema))

        # --- Fadiga ponderada ---
        fatigue_scores = []
        for dd in d:
            atraso = 50
            for lookback in range(1, min(50, idx)):
                if idx - lookback >= 0 and dd in self.contests[idx - lookback]['dezenas']:
                    atraso = lookback
                    break
            freq = self.dezena_freq_norm.get(dd, 0.01)
            fatigue_scores.append(atraso * (1.0 - freq))
        features.append(float(np.mean(fatigue_scores)))
        features.append(float(np.max(fatigue_scores)))

        # --- Elasticidade estrutural ---
        if len(self.repeat_history) >= 10:
            recent_avg = np.mean(self.repeat_history[-10:])
            global_avg = np.mean(self.repeat_history)
            elasticity = global_avg - recent_avg
            features.append(float(elasticity))
            features.append(float(abs(elasticity)))
        else:
            features.extend([0.0, 0.0])

        # --- Entropia recente da repetição ---
        if len(self.repeat_history) >= 10:
            recent = self.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            probs = np.where(probs>0, probs, 1e-10)
            features.append(float(entropy(probs)))
        else:
            features.append(0.0)

        # --- Diversidade estrutural ---
        qtd_quadrantes = len(set((x-1)//5 for x in d))
        features.append(float(qtd_quadrantes))
        features.append(float(consecutivos))

        # --- Comparação com médias históricas recentes ---
        if len(self.pares_history) >= 5:
            features.append(float(self.pares_history[-1]))
            features.append(float(pares - self.pares_history[-1]))
        else:
            features.extend([7.5, 0.0])
        if len(self.moldura_history) >= 5:
            features.append(float(self.moldura_history[-1]))
            features.append(float(moldura - self.moldura_history[-1]))
        else:
            features.extend([9.0, 0.0])
        if len(self.soma_history) >= 5:
            features.append(float(self.soma_history[-1]))
            features.append(float(soma - self.soma_history[-1]))
        else:
            features.extend([195.0, 0.0])

        return np.array(features, dtype=np.float32)

    def build_training_dataset(self, start_idx=None, end_idx=None, n_samples=10000):
        """
        Constrói dataset de treino com hard negative mining.
        """
        if start_idx is None:
            start_idx = max(50, self.n_contests - 500)
        if end_idx is None:
            end_idx = self.n_contests

        X_list = []
        y11_list = []
        y12_list = []
        y13_list = []
        payoff_list = []

        for idx in tqdm(range(start_idx, end_idx), desc="Construindo dataset"):
            actual = set(self.contests[idx]['dezenas'])
            # Gerar jogos candidatos variados
            for _ in range(30):  # número de exemplos por concurso
                # Estratégia de geração variada
                r = random.random()
                if r < 0.3:
                    # Aleatório puro
                    game = sorted(np.random.choice(range(1,26), 15, replace=False))
                elif r < 0.6:
                    # Baseado em repetição (hard negative potencial)
                    last = set(self.contests[idx-1]['dezenas']) if idx>0 else set()
                    base = list(last) if last else []
                    random.shuffle(base)
                    game_set = set(base[:random.randint(5,10)])
                    available = [x for x in range(1,26) if x not in game_set]
                    while len(game_set) < 15:
                        game_set.add(random.choice(available))
                    game = sorted(game_set)
                elif r < 0.8:
                    # Estruturalmente plausível
                    game_set = set()
                    for q in QUADRANTES:
                        if random.random() < 0.7:
                            game_set.update(random.sample(list(q), random.randint(2,4)))
                    available = [x for x in range(1,26) if x not in game_set]
                    while len(game_set) < 15:
                        game_set.add(random.choice(available))
                    game = sorted(game_set)[:15]
                else:
                    # Extremo
                    game = sorted(np.random.choice(range(1,26), 15, replace=False))

                hits = len(set(game) & actual)

                features = self.extract_features(game, idx-1)  # idx-1 = passado disponível
                X_list.append(features)
                y11_list.append(1 if hits >= 11 else 0)
                y12_list.append(1 if hits >= 12 else 0)
                y13_list.append(1 if hits >= 13 else 0)
                payoff_list.append(PAYOFF.get(hits, 0))

        X = np.array(X_list)
        y11 = np.array(y11_list)
        y12 = np.array(y12_list)
        y13 = np.array(y13_list)
        payoff = np.array(payoff_list)

        return X, y11, y12, y13, payoff

# ============================================================
# APRENDIZ DE MÚLTIPLOS TARGETS
# ============================================================
class MultiTargetLearner:
    def __init__(self):
        self.models = {}
        self.is_trained = False

    def train(self, X, y11, y12, y13, payoff):
        if X.shape[0] < 100:
            return False
        print(f"📊 Treinando learners... Amostras: {X.shape[0]}")
        tscv = TimeSeriesSplit(n_splits=3)

        # Target 11+
        pos_weight11 = (len(y11)-y11.sum()) / max(1, y11.sum())
        model11 = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                    learning_rate=0.05, scale_pos_weight=pos_weight11,
                                    random_state=42, verbosity=0) if XGB_AVAILABLE else \
                   RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        for ti, vi in tscv.split(X):
            model11.fit(X[ti], y11[ti])
        self.models['11'] = model11

        # Target 12+
        pos_weight12 = (len(y12)-y12.sum()) / max(1, y12.sum())
        model12 = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                    learning_rate=0.05, scale_pos_weight=pos_weight12,
                                    random_state=42, verbosity=0) if XGB_AVAILABLE else \
                   RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        for ti, vi in tscv.split(X):
            model12.fit(X[ti], y12[ti])
        self.models['12'] = model12

        # Target 13+
        pos_weight13 = (len(y13)-y13.sum()) / max(1, y13.sum())
        model13 = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                    learning_rate=0.05, scale_pos_weight=pos_weight13,
                                    random_state=42, verbosity=0) if XGB_AVAILABLE else \
                   RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        for ti, vi in tscv.split(X):
            model13.fit(X[ti], y13[ti])
        self.models['13'] = model13

        # Payoff (regressão)
        model_payoff = xgb.XGBRegressor(n_estimators=100, max_depth=5,
                                        learning_rate=0.05, random_state=42,
                                        verbosity=0) if XGB_AVAILABLE else \
                       RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        for ti, vi in tscv.split(X):
            model_payoff.fit(X[ti], payoff[ti])
        self.models['payoff'] = model_payoff

        self.is_trained = True
        print("✅ Modelos treinados")
        return True

    def predict(self, features_vector):
        if not self.is_trained:
            return {'11': 0.1, '12': 0.02, '13': 0.005, 'payoff': 0.0}
        results = {}
        results['11'] = float(self.models['11'].predict_proba([features_vector])[0][1])
        results['12'] = float(self.models['12'].predict_proba([features_vector])[0][1])
        results['13'] = float(self.models['13'].predict_proba([features_vector])[0][1])
        results['payoff'] = float(self.models['payoff'].predict([features_vector])[0])
        return results

# ============================================================
# OTIMIZADOR DE CARTEIRA (GENÉTICO)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, feature_engine, learner, contests):
        self.engine = feature_engine
        self.learner = learner
        self.contests = contests

    def score_game(self, game, idx):
        features = self.engine.extract_features(game, idx)
        preds = self.learner.predict(features)
        # Score combinado: peso maior para targets altos e payoff
        score = (preds['11'] * 0.1 + preds['12'] * 0.3 +
                 preds['13'] * 0.4 + preds['payoff'] * 0.2 / 100)
        return score

    def generate_candidates(self, n_candidates=20000, idx=None):
        if idx is None:
            idx = len(self.contests) - 1
        candidates = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando candidatos"):
            game = sorted(np.random.choice(range(1,26), 15, replace=False))
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                candidates.append(game)
        return candidates

    def select_portfolio(self, candidates, n_select=50, idx=None):
        if idx is None:
            idx = len(self.contests) - 1
        scored = []
        for game in candidates:
            s = self.score_game(game, idx-1)  # usar passado
            scored.append((s, game))
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        used_dezenas = Counter()
        for score, game in scored:
            if len(selected) >= n_select:
                break
            # Diversidade: penalizar similaridade
            too_similar = False
            for sg in selected:
                if len(set(game) & set(sg)) > 11:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(game)
        return selected

# ============================================================
# VALIDAÇÃO WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=5, train_size=300, test_size=50):
    print(f"\n🔬 WALK-FORWARD VALIDATION ({n_windows} janelas)")
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

        engine = AdvancedFeatureEngine(train_data)
        X, y11, y12, y13, payoff = engine.build_training_dataset(n_samples=5000)
        learner = MultiTargetLearner()
        learner.train(X, y11, y12, y13, payoff)
        opt = PortfolioOptimizer(engine, learner, train_data)

        strat_payoff = 0
        rand_payoff = 0
        n_jogos = 30
        for i, tc in enumerate(test_data):
            actual = set(tc['dezenas'])
            idx = len(train_data) + i
            candidates = opt.generate_candidates(3000, idx)
            portfolio = opt.select_portfolio(candidates, n_jogos, idx)
            for g in portfolio:
                hits = len(set(g) & actual)
                strat_payoff += PAYOFF.get(hits, 0)
            for _ in range(n_jogos):
                g = sorted(np.random.choice(range(1,26), 15, replace=False))
                hits = len(set(g) & actual)
                rand_payoff += PAYOFF.get(hits, 0)
        total_apostas = len(test_data) * n_jogos * CUSTO_APOSTA
        strat_roi = (strat_payoff - total_apostas) / total_apostas * 100
        rand_roi = (rand_payoff - total_apostas) / total_apostas * 100
        resultados.append({
            'window': w,
            'train_ini': train_data[0]['concurso'],
            'train_fim': train_data[-1]['concurso'],
            'test_ini': test_data[0]['concurso'],
            'test_fim': test_data[-1]['concurso'],
            'strat_roi': strat_roi,
            'rand_roi': rand_roi,
            'diff_roi': strat_roi - rand_roi
        })
        print(f" Janela {w}: ROI estrat={strat_roi:+.2f}% | rand={rand_roi:+.2f}% | diff={strat_roi-rand_roi:+.2f}%")

    if not resultados:
        print("⚠️ Dados insuficientes para walk-forward")
        return None
    diffs = [r['diff_roi'] for r in resultados]
    print(f"\n📊 RESUMO WALK-FORWARD:")
    print(f"   Média diferença ROI: {np.mean(diffs):+.2f}%")
    try:
        _, p = wilcoxon(diffs)
        print(f"   Wilcoxon p-value: {p:.4f}")
    except:
        pass
    n_pos = sum(1 for d in diffs if d > 0)
    print(f"   Janelas positivas: {n_pos}/{len(resultados)}")
    if n_pos > len(resultados)*0.7:
        print("   ✅ Vantagem consistente")
    elif n_pos > len(resultados)*0.5:
        print("   🟡 Marginal")
    else:
        print("   🟢 Sem vantagem")
    return resultados

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*60)
    print("🧬 SISTEMA COMPLETO LOTOFÁCIL v18")
    print("="*60)
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    print(f"📂 {len(contests)} concursos carregados")

    print("\nOpções:")
    print("1. Gerar carteira otimizada")
    print("2. Walk-forward validation")
    print("3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        print("\n🔥 Treinando modelo e gerando carteira...")
        engine = AdvancedFeatureEngine(contests)
        X, y11, y12, y13, payoff = engine.build_training_dataset(n_samples=10000)
        learner = MultiTargetLearner()
        learner.train(X, y11, y12, y13, payoff)
        opt = PortfolioOptimizer(engine, learner, contests)
        candidates = opt.generate_candidates(20000)
        portfolio = opt.select_portfolio(candidates, 30)
        print("\n🏆 CARTEIRA FINAL:")
        last = contests[-1]['dezenas']
        for i, game in enumerate(portfolio, 1):
            rep = len(set(game) & set(last))
            p = sum(1 for d in game if d%2==0)
            pr = sum(1 for d in game if d in PRIMES)
            m = sum(1 for d in game if d in MOLDURA)
            features = engine.extract_features(game, len(contests)-1)
            preds = learner.predict(features)
            print(f"{i:2d}. {game}")
            print(f"    P11:{preds['11']:.3f} P12:{preds['12']:.3f} P13:{preds['13']:.3f} "
                  f"Payoff:{preds['payoff']:.1f} Rep:{rep} Pares:{p} Primos:{pr} Moldura:{m}")

    if op in ("2", "3"):
        walk_forward_validation(contests, n_windows=5, train_size=300, test_size=50)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
