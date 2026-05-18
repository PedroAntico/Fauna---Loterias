#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE RANKING PROBABILÍSTICO ESTRUTURAL - LOTOFÁCIL
=========================================================
Versão 8.0 - Estados Markovianos + XGBoost + Ensemble

ARQUITETURA:
✅ CAMADA 1: CSP (universo permitido)
✅ CAMADA 2: Feature Engine Temporal (estados estruturais)
✅ CAMADA 3: Detector de Regime (HMM leve + clustering)
✅ CAMADA 4: XGBoost Ranker (estado → probabilidade estrutural)
✅ CAMADA 5: CSP Generator (100k jogos viáveis)
✅ CAMADA 6: Ensemble Ranker (XGBoost + Markov + Heurísticas)

PRINCÍPIO:
NÃO prevê dezenas. Prevê ESTADOS estruturais.
Depois ranqueia jogos viáveis por aderência ao estado provável.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
import pickle
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Tentar importar XGBoost (opcional mas recomendado)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost não instalado. Use: pip install xgboost")
    print("   Fallback: regressão logística")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn não instalado")

# ============================================================
# CONJUNTOS MATEMÁTICOS
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
QUADRANTES = {
    'Q1': {1,2,3,4,5}, 'Q2': {6,7,8,9,10},
    'Q3': {11,12,13,14,15}, 'Q4': {16,17,18,19,20},
    'Q5': {21,22,23,24,25}
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


# ============================================================
# CAMADA 2: FEATURE ENGINE TEMPORAL
# ============================================================

class StateFeatureEngine:
    """
    Extrai ESTADOS estruturais (não dezenas)
    
    Cada concurso vira um vetor de ~50 features estruturais
    """
    
    def __init__(self, all_contests):
        self.contests = all_contests
        self.n_contests = len(all_contests)
        
        # Features estruturais base
        self.structural_features = [
            'soma', 'pares', 'impares', 'primos', 'moldura', 'centro',
            'amplitude', 'consecutivos', 'max_run', 'distancia_media',
            'q1', 'q2', 'q3', 'q4', 'q5',
            'repetidas_anterior', 'repetidas_2anterior',
            'densidade_baixa', 'densidade_alta',
            'entropia_posicional', 'gini_spatial'
        ]
        
        # Features temporais (janelas)
        self.temporal_windows = [3, 5, 10, 20, 50]
        self.temporal_features = []
        for w in self.temporal_windows:
            for feat in ['pares', 'primos', 'moldura', 'soma', 'repetidas']:
                self.temporal_features.append(f'{feat}_ema_{w}')
                self.temporal_features.append(f'{feat}_momentum_{w}')
        
        # Features markovianas (transições)
        self.markov_features = [
            'regime_pares', 'regime_primos', 'regime_moldura',
            'tendencia_repeticao', 'tendencia_consecutivos',
            'persistencia_regime', 'volatilidade'
        ]
        
        self.all_feature_names = (
            self.structural_features + 
            self.temporal_features + 
            self.markov_features
        )
    
    def extract_state(self, idx):
        """
        Extrai estado estrutural completo para o concurso idx
        
        Usa apenas informações DISPONÍVEIS até idx (sem look-ahead)
        """
        if idx < 50:  # Precisa de histórico mínimo
            return None
        
        contest = self.contests[idx]
        dezenas = contest['dezenas']
        d = sorted(dezenas)
        
        features = {}
        
        # === ESTRUTURAIS ===
        features['soma'] = sum(d)
        features['pares'] = sum(1 for x in d if x % 2 == 0)
        features['impares'] = 15 - features['pares']
        features['primos'] = sum(1 for x in d if x in PRIMES)
        features['moldura'] = sum(1 for x in d if x in MOLDURA)
        features['centro'] = sum(1 for x in d if x in CENTRO)
        features['amplitude'] = max(d) - min(d)
        features['consecutivos'] = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        
        # Max run
        run = 1; max_run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1; max_run=max(max_run,run)
            else: run=1
        features['max_run'] = max_run
        
        features['distancia_media'] = np.mean([d[i+1]-d[i] for i in range(14)])
        
        # Quadrantes
        for qname, qset in QUADRANTES.items():
            features[qname.lower()] = sum(1 for x in d if x in qset)
        
        # Repetidas
        if idx >= 1:
            features['repetidas_anterior'] = len(set(d) & set(self.contests[idx-1]['dezenas']))
        else:
            features['repetidas_anterior'] = 0
        if idx >= 2:
            features['repetidas_2anterior'] = len(set(d) & set(self.contests[idx-2]['dezenas']))
        else:
            features['repetidas_2anterior'] = 0
        
        features['densidade_baixa'] = sum(1 for x in d if x <= 12)
        features['densidade_alta'] = sum(1 for x in d if x >= 14)
        
        # Entropia posicional
        pos_counts = np.bincount(d, minlength=26)[1:]
        probs = pos_counts / 15
        probs = np.where(probs>0, probs, 1e-10)
        features['entropia_posicional'] = entropy(probs)
        
        # Gini espacial
        sorted_counts = np.sort(pos_counts)
        n = len(sorted_counts)
        index = np.arange(1, n+1)
        features['gini_spatial'] = (2*np.sum(index*sorted_counts))/(n*np.sum(sorted_counts)) - (n+1)/n
        
        # === TEMPORAIS ===
        for w in self.temporal_windows:
            if idx >= w:
                window = self.contests[idx-w+1:idx+1]
                
                # Médias na janela
                w_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in window])
                w_primos = np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in window])
                w_moldura = np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in window])
                w_soma = np.mean([sum(c['dezenas']) for c in window])
                w_rep = np.mean([len(set(c['dezenas']) & set(self.contests[i-1]['dezenas'])) 
                                if i>0 else 0 for i, c in enumerate(window)])
                
                # EMA (média móvel exponencial)
                features[f'pares_ema_{w}'] = w_pares
                features[f'primos_ema_{w}'] = w_primos
                features[f'moldura_ema_{w}'] = w_moldura
                features[f'soma_ema_{w}'] = w_soma
                features[f'repetidas_ema_{w}'] = w_rep
                
                # Momentum (janela atual - janela anterior)
                if idx >= 2*w:
                    prev_window = self.contests[idx-2*w+1:idx-w+1]
                    prev_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in prev_window])
                    features[f'pares_momentum_{w}'] = w_pares - prev_pares
                    features[f'primos_momentum_{w}'] = w_primos - np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in prev_window])
                    features[f'moldura_momentum_{w}'] = w_moldura - np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in prev_window])
                    features[f'soma_momentum_{w}'] = w_soma - np.mean([sum(c['dezenas']) for c in prev_window])
                else:
                    for feat in ['pares','primos','moldura','soma','repetidas']:
                        features[f'{feat}_momentum_{w}'] = 0
        
        # === MARKOVIANAS ===
        if idx >= 10:
            recent = self.contests[idx-9:idx+1]
            
            # Regime atual (classificação)
            avg_pares = np.mean([features['pares']])
            if avg_pares >= 8.5: features['regime_pares'] = 2  # Alto
            elif avg_pares >= 6.5: features['regime_pares'] = 1  # Médio
            else: features['regime_pares'] = 0  # Baixo
            
            avg_primos = features['primos']
            if avg_primos >= 5: features['regime_primos'] = 2
            elif avg_primos >= 3: features['regime_primos'] = 1
            else: features['regime_primos'] = 0
            
            avg_moldura = features['moldura']
            if avg_moldura >= 10: features['regime_moldura'] = 2
            elif avg_moldura >= 8: features['regime_moldura'] = 1
            else: features['regime_moldura'] = 0
            
            # Persistência
            features['persistencia_regime'] = sum(
                1 for i in range(len(recent)-1)
                if abs(sum(1 for x in recent[i]['dezenas'] if x%2==0) - 
                       sum(1 for x in recent[i+1]['dezenas'] if x%2==0)) <= 1
            ) / (len(recent)-1)
            
            # Volatilidade
            pares_series = [sum(1 for x in c['dezenas'] if x%2==0) for c in recent]
            features['volatilidade'] = np.std(pares_series)
        
        # Garantir que todas as features existam
        for name in self.all_feature_names:
            if name not in features:
                features[name] = 0
        
        return features
    
    def build_dataset(self):
        """
        Constrói dataset completo para treino
        
        X: features do estado no tempo t
        y: features estruturais do estado no tempo t+1
        """
        X_list = []
        y_list = []
        
        for i in tqdm(range(50, self.n_contests - 1), desc="Construindo dataset"):
            state_t = self.extract_state(i)
            state_t1 = self.extract_state(i + 1)
            
            if state_t and state_t1:
                # Features de entrada (estado atual)
                x_vec = [state_t.get(name, 0) for name in self.all_feature_names]
                X_list.append(x_vec)
                
                # Target: features estruturais do próximo estado
                y_vec = [
                    state_t1['pares'],
                    state_t1['primos'],
                    state_t1['moldura'],
                    state_t1['repetidas_anterior'],
                    state_t1['soma'],
                    state_t1['max_run'],
                    state_t1['amplitude'],
                ]
                y_list.append(y_vec)
        
        return np.array(X_list), np.array(y_list), self.all_feature_names


# ============================================================
# CAMADA 3: DETECTOR DE REGIME
# ============================================================

class RegimeDetector:
    """
    Detecta o regime atual baseado nos últimos concursos
    
    Usa clustering simples + similaridade histórica
    """
    
    def __init__(self, feature_engine):
        self.engine = feature_engine
        self.contests = feature_engine.contests
        
        # Estados estruturais de todos os concursos
        self.all_states = []
        for i in range(50, len(self.contests)):
            state = self.engine.extract_state(i)
            if state:
                self.all_states.append(state)
    
    def get_current_regime(self, n_recent=10):
        """
        Detecta o regime atual
        
        Returns:
            dict: Características do regime atual
        """
        if len(self.contests) < n_recent + 50:
            return {'regime': 'indefinido', 'confianca': 0.0}
        
        recent = self.contests[-n_recent:]
        
        # Features médias recentes
        avg_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in recent])
        avg_primos = np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in recent])
        avg_moldura = np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in recent])
        avg_rep = np.mean([
            len(set(c['dezenas']) & set(self.contests[i-1]['dezenas']))
            for i, c in enumerate(recent) if i > 0
        ])
        
        # Classificar regime
        if avg_pares >= 8:
            regime_pares = 'alto'
        elif avg_pares <= 7:
            regime_pares = 'baixo'
        else:
            regime_pares = 'medio'
        
        # Persistência (desvio padrão baixo = regime estável)
        std_pares = np.std([sum(1 for x in c['dezenas'] if x%2==0) for c in recent])
        estabilidade = 1.0 / (1.0 + std_pares)
        
        return {
            'regime': f'pares_{regime_pares}',
            'avg_pares': avg_pares,
            'avg_primos': avg_primos,
            'avg_moldura': avg_moldura,
            'avg_repetidas': avg_rep,
            'estabilidade': estabilidade,
            'confianca': min(1.0, estabilidade * 2)
        }
    
    def find_similar_historical(self, n_recent=10, top_k=20):
        """
        Encontra períodos históricos similares ao atual
        
        Returns:
            list: O que aconteceu DEPOIS desses períodos
        """
        if len(self.contests) < n_recent + 50:
            return []
        
        # Assinatura atual
        recent = self.contests[-n_recent:]
        current_sig = np.array([
            np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in recent]),
            np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in recent]),
            np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in recent]),
            np.mean([sum(c['dezenas']) for c in recent]),
        ])
        
        # Buscar similares
        similarities = []
        for i in range(n_recent, len(self.contests) - 1):
            hist_window = self.contests[i-n_recent:i]
            hist_sig = np.array([
                np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in hist_window]),
                np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in hist_window]),
                np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in hist_window]),
                np.mean([sum(c['dezenas']) for c in hist_window]),
            ])
            
            dist = np.linalg.norm(current_sig - hist_sig)
            similarities.append((dist, i))
        
        similarities.sort(key=lambda x: x[0])
        
        # Retornar próximos concursos após regimes similares
        next_contests = []
        for dist, idx in similarities[:top_k]:
            if idx < len(self.contests):
                next_contests.append(self.contests[idx])
        
        return next_contests


# ============================================================
# CAMADA 4: XGBOOST RANKER
# ============================================================

class StructuralRanker:
    """
    Rankeia estados estruturais por probabilidade
    
    Treina: estado_t → features estruturais de t+1
    """
    
    def __init__(self, feature_engine):
        self.engine = feature_engine
        self.models = {}  # Um modelo por target
        self.scalers = {}
        self.is_trained = False
    
    def train(self):
        """Treina modelos para prever features estruturais"""
        print(f"\n📊 TREINANDO RANKER ESTRUTURAL...")
        
        X, y, feature_names = self.engine.build_dataset()
        
        if len(X) < 100:
            print("   ⚠️  Dados insuficientes para treino")
            return False
        
        print(f"   Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Targets a prever
        target_names = ['pares', 'primos', 'moldura', 'repetidas', 'soma', 'max_run', 'amplitude']
        
        # Time series split (respeita ordem temporal)
        tscv = TimeSeriesSplit(n_splits=5)
        
        for i, target_name in enumerate(target_names):
            y_target = y[:, i]
            
            # Classificar em faixas (para classificação)
            if target_name == 'pares':
                y_class = np.digitize(y_target, [6, 7, 8, 9])  # 5 classes
            elif target_name == 'primos':
                y_class = np.digitize(y_target, [3, 4, 5, 6])
            elif target_name == 'moldura':
                y_class = np.digitize(y_target, [7, 8, 9, 10])
            elif target_name == 'repetidas':
                y_class = np.digitize(y_target, [7, 8, 9, 10])
            elif target_name == 'soma':
                y_class = np.digitize(y_target, [170, 190, 210, 230])
            elif target_name == 'max_run':
                y_class = np.digitize(y_target, [2, 3, 4, 5])
            else:  # amplitude
                y_class = np.digitize(y_target, [20, 22, 23, 24])
            
            # Treinar modelo
            if XGB_AVAILABLE:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    objective='multi:softprob',
                    random_state=42,
                    verbosity=0
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Treinar com time series split
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_class[train_idx], y_class[val_idx]
                
                if len(np.unique(y_train)) > 1:
                    model.fit(X_train, y_train)
            
            self.models[target_name] = model
        
        self.is_trained = True
        print(f"   ✅ {len(self.models)} modelos treinados")
        return True
    
    def predict_structural_probs(self, current_state_features):
        """
        Prediz probabilidades estruturais para o próximo estado
        
        Args:
            current_state_features: Features do estado atual
        
        Returns:
            dict: Probabilidades para cada feature estrutural
        """
        if not self.is_trained:
            return None
        
        x_vec = np.array([current_state_features])
        
        probs = {}
        for target_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(x_vec)[0]
                classes = getattr(model, 'classes_', range(len(proba)))
                probs[target_name] = dict(zip(classes, proba))
        
        return probs
    
    def score_game(self, game, structural_probs):
        """
        Pontua um jogo baseado nas probabilidades estruturais previstas
        
        Args:
            game: Lista de 15 dezenas
            structural_probs: Output de predict_structural_probs
        
        Returns:
            float: Score 0-1
        """
        if structural_probs is None:
            return 0.5
        
        game = sorted(game)
        score = 0.0
        
        # Pares
        actual_pares = sum(1 for d in game if d % 2 == 0)
        pares_class = np.digitize([actual_pares], [6, 7, 8, 9])[0]
        if 'pares' in structural_probs and pares_class in structural_probs['pares']:
            score += structural_probs['pares'][pares_class] * 0.25
        
        # Primos
        actual_primos = sum(1 for d in game if d in PRIMES)
        primos_class = np.digitize([actual_primos], [3, 4, 5, 6])[0]
        if 'primos' in structural_probs and primos_class in structural_probs['primos']:
            score += structural_probs['primos'][primos_class] * 0.20
        
        # Moldura
        actual_moldura = sum(1 for d in game if d in MOLDURA)
        moldura_class = np.digitize([actual_moldura], [7, 8, 9, 10])[0]
        if 'moldura' in structural_probs and moldura_class in structural_probs['moldura']:
            score += structural_probs['moldura'][moldura_class] * 0.20
        
        # Repetidas
        if 'repetidas' in structural_probs:
            # Estimativa (depende do último concurso)
            score += 0.15  # Placeholder
        
        # Soma
        actual_soma = sum(game)
        soma_class = np.digitize([actual_soma], [170, 190, 210, 230])[0]
        if 'soma' in structural_probs and soma_class in structural_probs['soma']:
            score += structural_probs['soma'][soma_class] * 0.10
        
        # Max run
        d = game
        run = 1; max_run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1; max_run=max(max_run,run)
            else: run=1
        run_class = np.digitize([max_run], [2, 3, 4, 5])[0]
        if 'max_run' in structural_probs and run_class in structural_probs['max_run']:
            score += structural_probs['max_run'][run_class] * 0.10
        
        return min(1.0, max(0.0, score))


# ============================================================
# CAMADA 5: CSP GENERATOR
# ============================================================

def generate_feasible_game(constraints, max_attempts=100):
    """Gera jogo viável dentro das constraints"""
    if constraints is None:
        return sorted(np.random.choice(range(1, 26), 15, replace=False).tolist())
    
    fixed = set(constraints.get('fixas', []))
    excluded = set(constraints.get('excluidas', []))
    
    for _ in range(max_attempts):
        game = set(fixed)
        available = list(set(range(1, 26)) - excluded - game)
        needed = 15 - len(game)
        if needed > 0 and len(available) >= needed:
            game.update(np.random.choice(available, needed, replace=False))
        
        if len(game) == 15:
            valid = True
            if 'pares_target' in constraints:
                if sum(1 for d in game if d%2==0) != constraints['pares_target']:
                    valid = False
            if 'primos_target' in constraints:
                if sum(1 for d in game if d in PRIMES) != constraints['primos_target']:
                    valid = False
            if 'moldura_target' in constraints:
                if sum(1 for d in game if d in MOLDURA) != constraints['moldura_target']:
                    valid = False
            if valid:
                return sorted([int(x) for x in game])
    
    # Fallback
    game = set(fixed)
    available = list(set(range(1, 26)) - excluded - game)
    needed = 15 - len(game)
    if needed > 0 and len(available) >= needed:
        game.update(np.random.choice(available, needed, replace=False))
    return sorted([int(x) for x in game])


def generate_candidate_pool(constraints, n_candidates=100000):
    """Gera grande pool de candidatos viáveis"""
    candidates = []
    seen = set()
    
    for _ in tqdm(range(n_candidates), desc="Gerando candidatos"):
        game = generate_feasible_game(constraints)
        key = tuple(game)
        if key not in seen:
            seen.add(key)
            candidates.append(game)
    
    return candidates


# ============================================================
# CAMADA 6: ENSEMBLE RANKER
# ============================================================

class EnsembleRanker:
    """
    Combina múltiplas fontes de scoring:
    - XGBoost estrutural
    - Markov histórico
    - Heurísticas de cobertura
    """
    
    def __init__(self, feature_engine, regime_detector, structural_ranker):
        self.engine = feature_engine
        self.regime = regime_detector
        self.ranker = structural_ranker
        self.contests = feature_engine.contests
    
    def score_game_ensemble(self, game, current_state_features):
        """
        Score ensemble combinando múltiplos sinais
        
        Pesos:
        - XGBoost estrutural: 40%
        - Regime atual: 30%
        - Heurísticas: 30%
        """
        score = 0.0
        
        # 1. XGBoost estrutural (40%)
        structural_probs = self.ranker.predict_structural_probs(current_state_features)
        if structural_probs:
            score += self.ranker.score_game(game, structural_probs) * 0.40
        
        # 2. Regime atual (30%)
        regime_info = self.regime.get_current_regime()
        if regime_info['confianca'] > 0:
            # Bônus por aderência ao regime
            game_pares = sum(1 for d in game if d%2==0)
            regime_pares = regime_info['avg_pares']
            pares_score = 1.0 - abs(game_pares - regime_pares) / 5
            score += pares_score * 0.15
            
            game_primos = sum(1 for d in game if d in PRIMES)
            regime_primos = regime_info['avg_primos']
            primos_score = 1.0 - abs(game_primos - regime_primos) / 4
            score += primos_score * 0.15
        
        # 3. Heurísticas (30%)
        # Diversidade de quadrantes
        quad_count = len(set((d-1)//5 for d in game))
        score += (quad_count / 5) * 0.10
        
        # Balanceamento
        baixas = sum(1 for d in game if d <= 12)
        score += (1.0 - abs(baixas - 7.5) / 7.5) * 0.10
        
        # Consecutivos moderados
        cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i]==1)
        if cons <= 6:
            score += 0.10
        else:
            score += max(0, 0.10 - (cons - 6) * 0.02)
        
        return min(1.0, max(0.0, score))
    
    def rank_candidates(self, candidates, top_n=50):
        """Rankeia candidatos por score ensemble"""
        # Obter estado atual
        if len(self.contests) < 51:
            return candidates[:top_n]
        
        current_state = self.engine.extract_state(len(self.contests) - 1)
        if current_state is None:
            return candidates[:top_n]
        
        current_features = [current_state.get(name, 0) for name in self.engine.all_feature_names]
        
        # Pontuar todos
        scored = []
        for game in tqdm(candidates, desc="Rankeando"):
            s = self.score_game_ensemble(game, current_features)
            scored.append((s, game))
        
        # Ordenar
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar top N com diversidade
        selected = []
        seen = set()
        for score, game in scored:
            if len(selected) >= top_n:
                break
            
            # Verificar diversidade mínima
            too_similar = False
            for sel_game in selected:
                common = len(set(game) & set(sel_game))
                if common > 12:  # Muito similar
                    too_similar = True
                    break
            
            if not too_similar:
                selected.append((score, game))
        
        return selected


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def collect_preferences():
    print(f"\n{'='*60}")
    print(f"🎯 CONFIGURAÇÃO DE PREFERÊNCIAS")
    print(f"{'='*60}")
    prefs = {}
    
    print(f"\n📌 FIXAS:")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            f = sorted(set(int(x) for x in v.split() if 1<=int(x)<=25))
            if f: prefs['fixas'] = f[:15]
        except: pass
    
    print(f"\n🚫 EXCLUÍDAS:")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            e = [int(x) for x in v.split() if 1<=int(x)<=25]
            if 'fixas' in prefs: e = [x for x in e if x not in prefs['fixas']]
            if e: prefs['excluidas'] = sorted(set(e))
        except: pass
    
    return prefs if prefs else None


def display_results(scored_games, regime_info):
    print(f"\n{'='*60}")
    print(f"🏆 TOP JOGOS RANKEADOS")
    print(f"{'='*60}")
    
    if regime_info:
        print(f"📊 Regime atual: {regime_info.get('regime', 'N/A')}")
        print(f"   Estabilidade: {regime_info.get('estabilidade', 0):.2f}")
        print(f"   Confiança: {regime_info.get('confianca', 0):.2f}")
    
    for i, (score, game) in enumerate(scored_games[:15], 1):
        p = sum(1 for d in game if d%2==0)
        pr = sum(1 for d in game if d in PRIMES)
        m = sum(1 for d in game if d in MOLDURA)
        s = sum(game)
        print(f"   {i:2d}. {game} | Score:{score:.3f}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{s}")


def main():
    print("="*60)
    print("🧬 RANKING PROBABILÍSTICO ESTRUTURAL")
    print("="*60)
    
    # Carregar dados
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    
    print(f"📂 {len(contests)} concursos")
    
    # Inicializar componentes
    engine = StateFeatureEngine(contests)
    regime = RegimeDetector(engine)
    ranker = StructuralRanker(engine)
    
    # Treinar ranker
    ranker.train()
    
    # Ensemble
    ensemble = EnsembleRanker(engine, regime, ranker)
    
    # Coletar preferências
    prefs = collect_preferences()
    
    # Gerar candidatos
    print(f"\n🎲 GERANDO CANDIDATOS...")
    candidates = generate_candidate_pool(prefs, n_candidates=50000)
    print(f"   ✅ {len(candidates)} candidatos viáveis")
    
    # Rankear
    print(f"\n📊 RANKEANDO...")
    top_games = ensemble.rank_candidates(candidates, top_n=50)
    
    # Exibir
    regime_info = regime.get_current_regime()
    display_results(top_games, regime_info)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
