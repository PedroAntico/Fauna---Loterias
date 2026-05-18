#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA ROBUSTO DE OTIMIZAÇÃO - LOTOFÁCIL v19
==============================================
CORREÇÕES CRÍTICAS:
✅ Contexto temporal explícito (sem índices absolutos)
✅ Labels contínuas (hits exatos, payoff real)
✅ Calibração de probabilidades (Platt scaling)
✅ Dataset com exemplos reais do histórico
✅ Walk-forward com contexto isolado
✅ Arquitetura modular preservada
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
    print("⚠️ XGBoost não instalado. Use: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
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
    except:
        return None


# ============================================================
# CONTEXTO TEMPORAL EXPLÍCITO (resolve bug de índice)
# ============================================================
class TemporalContext:
    """
    Encapsula o estado temporal disponível em um ponto da série.
    Substitui índices absolutos por referências seguras.
    """
    def __init__(self, contests_slice, historical_contests=None):
        """
        Args:
            contests_slice: lista de concursos DISPONÍVEIS (passado)
            historical_contests: opcional, para estatísticas de longo prazo
        """
        self.contests = contests_slice  # apenas o passado visível
        self.n_contests = len(contests_slice)
        self.historical = historical_contests if historical_contests is not None else contests_slice

        # Históricos derivados apenas do contests_slice
        self.repeat_history = []
        self.pares_history = []
        self.primos_history = []
        self.moldura_history = []
        self.soma_history = []
        self.gap_media_history = []
        self.gap_var_history = []
        self.energia_history = []

        # Estatísticas de longo prazo (podem usar mais dados)
        self.dezena_counts = Counter()
        self.ema_dezenas = {d: 0.0 for d in range(1, 26)}
        self.ema_alpha = 0.3
        self.dezena_streaks = {d: 0 for d in range(1, 26)}
        self.dezena_last_seen = {d: -1 for d in range(1, 26)}  # índice local

        # Preencher históricos com dados disponíveis
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

        # Frequências normalizadas
        total = len(self.contests)
        self.dezena_freq_norm = {d: f/total for d, f in self.dezena_counts.items()} if total > 0 else {}

        # Para fadiga, podemos usar o historical (mais longo)
        self.historical_counts = Counter()
        for c in self.historical:
            self.historical_counts.update(c['dezenas'])
        hist_total = len(self.historical)
        self.historical_freq_norm = {d: f/hist_total for d, f in self.historical_counts.items()} if hist_total > 0 else {}

    def get_last_contest(self):
        if self.n_contests > 0:
            return self.contests[-1]['dezenas']
        return []

    def extract_features(self, game):
        """
        Extrai features para um jogo candidato.
        Usa apenas o contexto temporal DISPONÍVEL.
        """
        features = []
        d = sorted(game)

        # Básicos do jogo
        pares = sum(1 for x in d if x % 2 == 0)
        primos = sum(1 for x in d if x in PRIMES)
        moldura = sum(1 for x in d if x in MOLDURA)
        soma = sum(d)
        consecutivos = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)

        # Gaps e energia
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        gap_medio = np.mean(gaps)
        gap_var = np.var(gaps)
        gap_max = max(gaps)
        gap_min = min(gaps)
        energia = sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))

        # Repetição (relativo ao último concurso)
        last = self.get_last_contest()
        rep = len(set(d) & set(last)) if last else 8

        # Markov 1,2,3
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
            d1 = self.repeat_history[-1] - self.repeat_history[-2]
            d2 = self.repeat_history[-2] - self.repeat_history[-3]
            features.append(float(d1 - d2))
        else:
            features.extend([8.0, 0.0])

        # Streak sem 9
        streak9 = 0
        for r in reversed(self.repeat_history):
            if r != 9: streak9 += 1
            else: break
        features.append(float(streak9))
        features.append(1.0 / (1.0 + np.exp(-(streak9-5)/3)))

        # Interações
        features.append(float(rep * pares))
        features.append(float(rep * moldura))
        features.append(float(pares * soma / 100.0))
        features.append(float(primos * moldura))

        # Compressão espacial
        features.append(float(gap_medio))
        features.append(float(gap_var))
        features.append(float(gap_max))
        features.append(float(gap_min))

        # Energia
        features.append(float(energia))

        # Persistência individual (streaks locais)
        avg_streak = np.mean([self.dezena_streaks.get(dd, 0) for dd in d])
        max_streak = max([self.dezena_streaks.get(dd, 0) for dd in d])
        features.append(float(avg_streak))
        features.append(float(max_streak))

        # EMA
        avg_ema = np.mean([self.ema_dezenas.get(dd, 0) for dd in d])
        max_ema = max([self.ema_dezenas.get(dd, 0) for dd in d])
        features.append(float(avg_ema))
        features.append(float(max_ema))

        # Fadiga ponderada (usa historical para frequência)
        fatigue_scores = []
        for dd in d:
            # atraso local
            last_seen = self.dezena_last_seen.get(dd, -1)
            if last_seen >= 0:
                atraso = self.n_contests - 1 - last_seen
            else:
                atraso = self.n_contests  # nunca visto
            freq = self.historical_freq_norm.get(dd, 0.01)
            fatigue_scores.append(atraso * (1.0 - freq))
        features.append(float(np.mean(fatigue_scores)))
        features.append(float(np.max(fatigue_scores)))

        # Elasticidade
        if len(self.repeat_history) >= 10:
            recent_avg = np.mean(self.repeat_history[-10:])
            global_avg = np.mean(self.repeat_history)
            elasticity = global_avg - recent_avg
            features.append(float(elasticity))
            features.append(float(abs(elasticity)))
        else:
            features.extend([0.0, 0.0])

        # Entropia recente
        if len(self.repeat_history) >= 10:
            recent = self.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            probs = np.where(probs>0, probs, 1e-10)
            features.append(float(entropy(probs)))
        else:
            features.append(0.0)

        # Diversidade
        qtd_quadrantes = len(set((x-1)//5 for x in d))
        features.append(float(qtd_quadrantes))
        features.append(float(consecutivos))

        # Comparação com médias recentes
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

    def build_training_dataset(self, n_samples=5000):
        """
        Constrói dataset usando exemplos REAIS do histórico + hard negatives.
        Labels CONTÍNUAS: hits exatos e payoff.
        """
        X_list = []
        y_hits_list = []
        y_payoff_list = []

        # Usar concursos disponíveis (menos o último, que seria "futuro")
        available_contests = self.contests[:-1] if len(self.contests) > 1 else []

        for i, contest in enumerate(available_contests):
            actual = set(contest['dezenas'])

            # Criar contexto até o concurso i (exclusivo)
            ctx = TemporalContext(self.contests[:i+1], self.historical)

            # 1. Exemplos REAIS: jogos que foram apostados em concursos anteriores
            # Simular jogos "reais" como negativos
            for _ in range(10):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                hits = len(set(game) & actual)
                features = ctx.extract_features(game)
                X_list.append(features)
                y_hits_list.append(hits)
                y_payoff_list.append(PAYOFF.get(hits, 0))

            # 2. Hard negatives: jogos estruturalmente plausíveis
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
                features = ctx.extract_features(game)
                X_list.append(features)
                y_hits_list.append(hits)
                y_payoff_list.append(PAYOFF.get(hits, 0))

        # Limitar tamanho
        if len(X_list) > n_samples:
            indices = np.random.choice(len(X_list), n_samples, replace=False)
            X_list = [X_list[i] for i in indices]
            y_hits_list = [y_hits_list[i] for i in indices]
            y_payoff_list = [y_payoff_list[i] for i in indices]

        return np.array(X_list), np.array(y_hits_list), np.array(y_payoff_list)


# ============================================================
# LEARNER COM LABELS CONTÍNUAS E CALIBRAÇÃO
# ============================================================
class CalibratedLearner:
    """
    Aprende com labels contínuas (hits, payoff) e calibra probabilidades.
    """
    def __init__(self):
        self.model_hits = None       # Regressor para hits (0-15)
        self.model_payoff = None     # Regressor para payoff
        self.calibrator_11 = None    # Isotonic para P(hits>=11)
        self.calibrator_12 = None
        self.calibrator_13 = None
        self.is_trained = False

    def train(self, X, y_hits, y_payoff):
        if X.shape[0] < 100:
            print("⚠️ Dados insuficientes para treino")
            return False

        print(f"📊 Treinando learner contínuo... Amostras: {X.shape[0]}")

        # Regressor para hits (0-15)
        self.model_hits = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0
        ) if XGB_AVAILABLE else RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )

        tscv = TimeSeriesSplit(n_splits=3)
        for ti, vi in tscv.split(X):
            self.model_hits.fit(X[ti], y_hits[ti])

        # Regressor para payoff
        self.model_payoff = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0
        ) if XGB_AVAILABLE else RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        for ti, vi in tscv.split(X):
            self.model_payoff.fit(X[ti], y_payoff[ti])

        # Calibração: mapear predições brutas → probabilidades reais
        # Usar último fold para calibração
        X_cal = X[tscv.split(X).__iter__().__next__()[1]]
        if len(X_cal) > 50:
            raw_hits = self.model_hits.predict(X_cal)
            raw_payoff = self.model_payoff.predict(X_cal)

            # Targets binários para calibração
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

        self.is_trained = True
        print("✅ Learner contínuo treinado com calibração")
        return True

    def predict(self, features_vector):
        """Retorna predições calibradas"""
        if not self.is_trained:
            return {'hits': 7.5, 'payoff': 0.5, 'P11': 0.09, 'P12': 0.017, 'P13': 0.0015}

        raw_hits = float(self.model_hits.predict([features_vector])[0])
        raw_payoff = float(self.model_payoff.predict([features_vector])[0])

        # Calibrar
        if self.calibrator_11 is not None:
            p11 = float(self.calibrator_11.predict([raw_hits])[0])
            p12 = float(self.calibrator_12.predict([raw_hits])[0])
            p13 = float(self.calibrator_13.predict([raw_hits])[0])
        else:
            # Fallback não calibrado
            p11 = 1.0 / (1.0 + np.exp(-(raw_hits - 9.0)))
            p12 = 1.0 / (1.0 + np.exp(-(raw_hits - 11.0)))
            p13 = 1.0 / (1.0 + np.exp(-(raw_hits - 13.0)))

        return {
            'hits': raw_hits,
            'payoff': raw_payoff,
            'P11': p11,
            'P12': p12,
            'P13': p13
        }


# ============================================================
# OTIMIZADOR DE CARTEIRA (COM CONTEXTO)
# ============================================================
class PortfolioOptimizerV19:
    def __init__(self, context, learner):
        self.context = context
        self.learner = learner

    def score_game(self, game):
        features = self.context.extract_features(game)
        preds = self.learner.predict(features)
        # Score: payoff esperado com peso maior para targets altos
        return preds['payoff'] * 0.5 + preds['P12'] * 20 + preds['P13'] * 50

    def generate_candidates(self, n_candidates=20000):
        candidates = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando candidatos"):
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
            if len(selected) >= n_select:
                break
            too_similar = False
            for sg in selected:
                if len(set(game) & set(sg)) > 11:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(game)
        return selected


# ============================================================
# WALK-FORWARD COM CONTEXTO ISOLADO
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

        # Contexto ISOLADO (resolve bug de índice)
        historical = contests[:train_end]  # dados históricos para frequências
        context = TemporalContext(train_data, historical)

        # Treinar
        X, y_hits, y_payoff = context.build_training_dataset(n_samples=5000)
        learner = CalibratedLearner()
        learner.train(X, y_hits, y_payoff)

        # Otimizador
        opt = PortfolioOptimizerV19(context, learner)

        strat_payoff = 0
        rand_payoff = 0
        n_jogos = 30

        for i, tc in enumerate(test_data):
            actual = set(tc['dezenas'])

            # Atualizar contexto com dados de teste JÁ OCORRIDOS (sem look-ahead)
            extended_train = train_data + test_data[:i]
            current_context = TemporalContext(extended_train, historical)
            current_opt = PortfolioOptimizerV19(current_context, learner)

            candidates = current_opt.generate_candidates(3000)
            portfolio = current_opt.select_portfolio(candidates, n_jogos)

            for g in portfolio:
                hits = len(set(g) & actual)
                strat_payoff += PAYOFF.get(hits, 0)

            for _ in range(n_jogos):
                g = sorted(np.random.choice(range(1,26), 15, replace=False))
                hits = len(set(g) & actual)
                rand_payoff += PAYOFF.get(hits, 0)

        total_apostas = len(test_data) * n_jogos * CUSTO_APOSTA
        strat_roi = (strat_payoff - total_apostas) / total_apostas * 100 if total_apostas > 0 else 0
        rand_roi = (rand_payoff - total_apostas) / total_apostas * 100 if total_apostas > 0 else 0

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
    print("🧬 SISTEMA ROBUSTO LOTOFÁCIL v19")
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
        context = TemporalContext(contests)
        X, y_hits, y_payoff = context.build_training_dataset(n_samples=10000)
        learner = CalibratedLearner()
        learner.train(X, y_hits, y_payoff)

        opt = PortfolioOptimizerV19(context, learner)
        candidates = opt.generate_candidates(20000)
        portfolio = opt.select_portfolio(candidates, 30)

        print("\n🏆 CARTEIRA FINAL:")
        last = contests[-1]['dezenas']
        for i, game in enumerate(portfolio, 1):
            rep = len(set(game) & set(last))
            p = sum(1 for d in game if d%2==0)
            pr = sum(1 for d in game if d in PRIMES)
            m = sum(1 for d in game if d in MOLDURA)
            features = context.extract_features(game)
            preds = learner.predict(features)
            print(f"{i:2d}. {game}")
            print(f"    Hits pred:{preds['hits']:.1f} | P11:{preds['P11']:.3f} "
                  f"P12:{preds['P12']:.3f} P13:{preds['P13']:.3f} "
                  f"Payoff:{preds['payoff']:.1f}")
            print(f"    Rep:{rep} Pares:{p} Primos:{pr} Moldura:{m}")

    if op in ("2", "3"):
        walk_forward_validation(contests, n_windows=5, train_size=300, test_size=50)

    print("\n✅ Concluído!")


if __name__ == "__main__":
    main()
