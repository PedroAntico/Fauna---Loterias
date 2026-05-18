#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE APRENDIZADO AUTOMÁTICO DE PESOS - VERSÃO 17.0
==========================================================
MELHORIAS:
✅ Pesos aprendidos via XGBoost (não heurísticos)
✅ Markov de ordem 2 e 3 (memória multi-concurso)
✅ Features de velocidade e aceleração estrutural
✅ Fatigue score ponderado por frequência histórica
✅ Elasticidade estrutural (retorno à média)
✅ Treino: features → P(acertos >= 11)
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
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================
# CONJUNTOS
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
PAYOFF = {11: 1, 12: 5, 13: 50, 14: 500, 15: 5000}

# ============================================================
# CARREGAMENTO
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
# FEATURE ENGINE AVANÇADO
# ============================================================
class AdvancedFeatureEngine:
    """
    Extrai features de repetição com:
    - Markov ordem 1, 2 e 3
    - Velocidade e aceleração
    - Fadiga ponderada
    - Elasticidade estrutural
    """
    
    def __init__(self, contests):
        self.contests = contests
        self.repeat_history = []
        self.pares_history = []
        self.moldura_history = []
        self.soma_history = []
        
        # Extrair históricos
        for i in range(1, len(contests)):
            prev = set(contests[i-1]['dezenas'])
            curr = set(contests[i]['dezenas'])
            self.repeat_history.append(len(prev & curr))
            
            d = contests[i]['dezenas']
            self.pares_history.append(sum(1 for x in d if x % 2 == 0))
            self.moldura_history.append(sum(1 for x in d if x in MOLDURA))
            self.soma_history.append(sum(d))
        
        # Frequências históricas para fadiga ponderada
        self.dezena_freq = Counter()
        for c in contests:
            self.dezena_freq.update(c['dezenas'])
        total = len(contests)
        self.dezena_freq_norm = {d: f/total for d, f in self.dezena_freq.items()}
    
    def extract_features(self, game, last_contests, idx):
        """
        Extrai vetor de features para um jogo candidato.
        
        Features incluem:
        - Markov ordem 1, 2, 3
        - Velocidade e aceleração da repetição
        - Fadiga ponderada
        - Elasticidade estrutural
        """
        features = []
        
        # ---- REPETIÇÃO ----
        last = last_contests[-1] if last_contests else []
        rep = len(set(game) & set(last)) if last else 8
        
        # Markov ordem 1
        if len(self.repeat_history) >= 1:
            prev_rep = self.repeat_history[-1]
            features.append(float(prev_rep))  # repetição anterior
            features.append(float(rep - prev_rep))  # delta
        
        # Markov ordem 2
        if len(self.repeat_history) >= 2:
            prev2_rep = self.repeat_history[-2]
            features.append(float(prev2_rep))
            features.append(float(prev_rep - prev2_rep))  # delta anterior
        
        # Markov ordem 3
        if len(self.repeat_history) >= 3:
            prev3_rep = self.repeat_history[-3]
            features.append(float(prev3_rep))
        
        # Velocidade (delta) e aceleração
        if len(self.repeat_history) >= 3:
            delta_t = self.repeat_history[-1] - self.repeat_history[-2]
            delta_t1 = self.repeat_history[-2] - self.repeat_history[-3]
            features.append(float(delta_t))  # velocidade
            features.append(float(delta_t - delta_t1))  # aceleração
        
        # Streak sem 9 repetidas
        streak = 0
        for r in reversed(self.repeat_history):
            if r != 9: streak += 1
            else: break
        features.append(float(streak))
        features.append(1.0 / (1.0 + np.exp(-(streak - 5) / 3)))  # mean reversion sigmoide
        
        # ---- PARES ----
        pares = sum(1 for x in game if x % 2 == 0)
        if self.pares_history:
            features.append(float(self.pares_history[-1]))  # pares anterior
            features.append(float(pares - self.pares_history[-1]))  # delta pares
        if len(self.pares_history) >= 2:
            features.append(float(self.pares_history[-2]))  # pares t-2
        
        # ---- MOLDURA ----
        moldura = sum(1 for x in game if x in MOLDURA)
        if self.moldura_history:
            features.append(float(self.moldura_history[-1]))
            features.append(float(moldura - self.moldura_history[-1]))
        
        # ---- SOMA ----
        soma = sum(game)
        if self.soma_history:
            features.append(float(self.soma_history[-1]))
            features.append(float(soma - self.soma_history[-1]))
        
        # ---- FADIGA PONDERADA ----
        fatigue_scores = []
        for d in game:
            # Atraso desde última aparição
            atraso = 50
            for lookback in range(1, min(50, idx)):
                if idx - lookback >= 0 and d in self.contests[idx - lookback]['dezenas']:
                    atraso = lookback
                    break
            # Ponderar por frequência histórica
            freq = self.dezena_freq_norm.get(d, 0.01)
            fatigue_scores.append(atraso * (1.0 - freq))  # +peso se rara
        features.append(float(np.mean(fatigue_scores)))
        features.append(float(np.max(fatigue_scores)))
        
        # ---- ELASTICIDADE ESTRUTURAL ----
        # Tendência de retorno à média após extremos
        if len(self.repeat_history) >= 10:
            recent_avg = np.mean(self.repeat_history[-10:])
            global_avg = np.mean(self.repeat_history)
            elasticity = global_avg - recent_avg  # +positivo = abaixo da média
            features.append(float(elasticity))
            features.append(float(abs(elasticity)))  # magnitude do desvio
        else:
            features.extend([0.0, 0.0])
        
        # ---- ENTROPIA RECENTE ----
        if len(self.repeat_history) >= 10:
            recent = self.repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r, 0)/10 for r in range(5, 13)])
            probs = np.where(probs > 0, probs, 1e-10)
            features.append(float(entropy(probs)))
        else:
            features.append(0.0)
        
        # ---- DIVERSIDADE ESTRUTURAL ----
        features.append(float(len(set((x-1)//5 for x in game))))  # quadrantes
        features.append(float(sum(1 for i in range(len(game)-1) if game[i+1]-game[i]==1)))  # consecutivos
        
        return np.array(features, dtype=np.float32)
    
    def build_training_dataset(self, n_samples=5000):
        """
        Constrói dataset de treino:
        X = features do jogo
        y = 1 se acertou >= 11, 0 caso contrário
        """
        X_list = []
        y_list = []
        
        # Usar últimos concursos para gerar exemplos
        start_idx = max(50, len(self.contests) - 500)
        
        for idx in tqdm(range(start_idx, len(self.contests)), desc="Construindo dataset"):
            actual = set(self.contests[idx]['dezenas'])
            last_contests = [c['dezenas'] for c in self.contests[:idx]]
            
            # Gerar jogos candidatos
            for _ in range(min(20, n_samples // (len(self.contests) - start_idx))):
                game = sorted(np.random.choice(range(1, 26), 15, replace=False))
                hits = len(set(game) & actual)
                
                features = self.extract_features(game, last_contests, idx)
                X_list.append(features)
                y_list.append(1 if hits >= 11 else 0)
        
        return np.array(X_list), np.array(y_list)


# ============================================================
# APRENDIZ AUTOMÁTICO DE PESOS
# ============================================================
class WeightLearner:
    """
    Aprende pesos automaticamente via XGBoost.
    Entrada: features do jogo
    Saída: P(acertos >= 11)
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.is_trained = False
    
    def train(self, X, y):
        """Treina modelo para prever probabilidade de sucesso"""
        if len(X) < 100 or not XGB_AVAILABLE:
            print("   ⚠️ Dados insuficientes ou XGBoost indisponível")
            return False
        
        print(f"\n📊 TREINANDO APRENDIZ DE PESOS...")
        print(f"   Amostras: {X.shape[0]} | Features: {X.shape[1]}")
        
        # Balancear classes
        pos_weight = (len(y) - y.sum()) / max(1, y.sum())
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbosity=0
        )
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
        
        # Importância das features
        self.feature_importance = dict(zip(
            range(X.shape[1]),
            self.model.feature_importances_
        ))
        
        self.is_trained = True
        print(f"   ✅ Modelo treinado")
        print(f"   Top 5 features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
        return True
    
    def predict_proba(self, features_vector):
        """Retorna P(sucesso) para um jogo"""
        if not self.is_trained:
            return 0.1  # Probabilidade base ~1/11
        
        proba = self.model.predict_proba(np.array([features_vector]))[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])


# ============================================================
# OTIMIZADOR COM PESOS APRENDIDOS
# ============================================================
class LearnedWeightOptimizer:
    """
    Otimizador que usa pesos APRENDIDOS, não heurísticos.
    """
    
    def __init__(self, feature_engine, weight_learner, contests):
        self.engine = feature_engine
        self.learner = weight_learner
        self.contests = contests
        
        self.dezena_usage = Counter()
        self.generated_pool = []
    
    def _score_game(self, game, idx):
        """Score baseado em probabilidade aprendida"""
        last_contests = [c['dezenas'] for c in self.contests[:idx]]
        features = self.engine.extract_features(game, last_contests, idx)
        
        # Probabilidade aprendida (principal)
        learned_prob = self.learner.predict_proba(features)
        
        # Cobertura (secundário)
        coverage_bonus = sum(1.0 / (1.0 + self.dezena_usage.get(d, 0)) for d in game) * 0.5
        
        # Penalidade de similaridade
        similarity_penalty = 0
        for existing in self.generated_pool[-20:]:
            common = len(set(game) & set(existing))
            if common > 11:
                similarity_penalty += (common - 11) * 2
        
        return learned_prob * 0.7 + coverage_bonus * 0.2 - similarity_penalty * 0.1
    
    def generate_and_select(self, n_candidates=20000, n_select=30):
        """Gera candidatos e seleciona os melhores"""
        idx = len(self.contests) - 1
        
        candidates = []
        seen = set()
        
        for _ in tqdm(range(n_candidates), desc="Gerando candidatos"):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            key = tuple(game)
            if key not in seen and len(game) == 15:
                seen.add(key)
                candidates.append(game)
        
        # Pontuar com pesos aprendidos
        scored = [(self._score_game(g, idx), g) for g in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar com diversidade
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
                for d in game:
                    self.dezena_usage[d] += 1
                self.generated_pool.append(game)
        
        return selected


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*60)
    print("🧬 APRENDIZ AUTOMÁTICO DE PESOS v17")
    print("="*60)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    
    print(f"📂 {len(contests)} concursos")
    
    # Feature engine
    engine = AdvancedFeatureEngine(contests)
    
    # Construir dataset de treino
    print(f"\n📊 CONSTRUINDO DATASET DE TREINO...")
    X, y = engine.build_training_dataset(n_samples=10000)
    print(f"   ✅ {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"   Balanceamento: {y.mean()*100:.1f}% positivos")
    
    # Treinar aprendiz de pesos
    learner = WeightLearner()
    learner.train(X, y)
    
    # Otimizar com pesos aprendidos
    opt = LearnedWeightOptimizer(engine, learner, contests)
    
    print(f"\n🎲 GERANDO CARTEIRA COM PESOS APRENDIDOS...")
    portfolio = opt.generate_and_select(n_candidates=20000, n_select=30)
    
    print(f"\n🏆 CARTEIRA OTIMIZADA:")
    last = contests[-1]['dezenas']
    for i, game in enumerate(portfolio[:10], 1):
        rep = len(set(game) & set(last))
        p = sum(1 for d in game if d%2==0)
        pr = sum(1 for d in game if d in PRIMES)
        m = sum(1 for d in game if d in MOLDURA)
        features = engine.extract_features(game, [last], len(contests)-1)
        prob = learner.predict_proba(features)
        print(f"   {i:2d}. {game}")
        print(f"       P(sucesso):{prob:.3f} | Rep:{rep} | Pares:{p} | Primos:{pr} | Moldura:{m}")
    
    print(f"\n✅ CONCLUÍDO!")
    print(f"💡 Pesos agora são APRENDIDOS, não arbitrários!")


if __name__ == "__main__":
    main()
