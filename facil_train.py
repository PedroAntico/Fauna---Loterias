#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA PROBABILÍSTICO ESTRUTURAL - VERSÃO CORRIGIDA
=====================================================
Versão 9.0 - Markov States + Guided Generation + Beam Search

CORREÇÕES:
✅ np.digitize() - 1 em TODOS os lugares
✅ Médias calculadas corretamente (não média de 1 valor)
✅ Repetidas com índices REAIS dos concursos
✅ Substituição de 50k aleatórios por geração guiada
✅ Markov States (state_id discreto)
✅ Beam Search para construção incremental
✅ MultiOutputClassifier para features correlacionadas
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Tentar importar XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost não instalado. Use: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
# MARKOV STATE ENCODER
# ============================================================

class MarkovStateEncoder:
    """
    Codifica estado estrutural em ID discreto
    
    Estado = (faixa_pares, faixa_primos, faixa_moldura, faixa_repetidas)
    
    Exemplo: STATE_8_5_10_8 → ID único
    """
    
    def __init__(self):
        self.state_to_id = {}
        self.id_to_state = {}
        self.transition_matrix = None
        self.state_list = []
    
    def encode(self, pares, primos, moldura, repetidas):
        """Codifica features em ID de estado"""
        # Discretizar
        faixa_pares = min(4, max(0, pares - 5))      # 5-9 → 0-4
        faixa_primos = min(3, max(0, primos - 2))     # 2-6 → 0-3
        faixa_moldura = min(4, max(0, moldura - 6))   # 6-11 → 0-4
        faixa_rep = min(4, max(0, repetidas - 6))     # 6-11 → 0-4
        
        state_key = f"{faixa_pares}_{faixa_primos}_{faixa_moldura}_{faixa_rep}"
        
        if state_key not in self.state_to_id:
            state_id = len(self.state_to_id)
            self.state_to_id[state_key] = state_id
            self.id_to_state[state_id] = {
                'faixa_pares': faixa_pares,
                'faixa_primos': faixa_primos,
                'faixa_moldura': faixa_moldura,
                'faixa_rep': faixa_rep,
                'pares_range': f"{faixa_pares+5}-{faixa_pares+5}",
                'primos_range': f"{faixa_primos+2}-{faixa_primos+2}",
                'moldura_range': f"{faixa_moldura+6}-{faixa_moldura+6}",
            }
        
        return self.state_to_id[state_key]
    
    def build_transition_matrix(self, contests):
        """Constrói matriz de transição Markoviana"""
        n_states = len(self.state_to_id)
        if n_states == 0:
            return
        
        trans_count = np.zeros((n_states, n_states))
        self.state_list = []
        
        # Extrair sequência de estados
        prev_state = None
        for i in range(1, len(contests)):
            c = contests[i]
            d = c['dezenas']
            prev = contests[i-1]['dezenas']
            
            pares = sum(1 for x in d if x % 2 == 0)
            primos = sum(1 for x in d if x in PRIMES)
            moldura = sum(1 for x in d if x in MOLDURA)
            repetidas = len(set(d) & set(prev))
            
            state_id = self.encode(pares, primos, moldura, repetidas)
            self.state_list.append(state_id)
            
            if prev_state is not None:
                trans_count[prev_state, state_id] += 1
            
            prev_state = state_id
        
        # Normalizar
        row_sums = trans_count.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = trans_count / row_sums
    
    def get_transition_probs(self, current_state_id):
        """Probabilidades de transição do estado atual"""
        if self.transition_matrix is None or current_state_id >= len(self.transition_matrix):
            return None
        return self.transition_matrix[current_state_id]
    
    def get_most_likely_next_states(self, current_state_id, top_k=3):
        """Estados mais prováveis após o atual"""
        probs = self.get_transition_probs(current_state_id)
        if probs is None:
            return []
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [(idx, probs[idx]) for idx in top_indices if probs[idx] > 0]


# ============================================================
# FEATURE ENGINE (CORRIGIDO)
# ============================================================

class StateFeatureEngine:
    """Extrai features estruturais CORRIGIDAS"""
    
    def __init__(self, all_contests):
        self.contests = all_contests
        self.n_contests = len(all_contests)
        
        self.feature_names = [
            'soma', 'pares', 'primos', 'moldura', 'centro',
            'amplitude', 'consecutivos', 'max_run', 'distancia_media',
            'q1', 'q2', 'q3', 'q4', 'q5',
            'repetidas', 'repetidas_2',
            'densidade_baixa', 'densidade_alta',
            'entropia_posicional',
            'pares_ema_5', 'pares_ema_10', 'pares_ema_20',
            'primos_ema_5', 'primos_ema_10',
            'moldura_ema_5', 'moldura_ema_10',
            'repetidas_ema_5', 'repetidas_ema_10',
            'pares_momentum_5', 'primos_momentum_5',
            'volatilidade'
        ]
    
    def extract_state(self, idx):
        """Extrai estado CORRIGIDO para o concurso idx"""
        if idx < 50:
            return None
        
        contest = self.contests[idx]
        dezenas = contest['dezenas']
        d = sorted(dezenas)
        
        features = {}
        
        # Estruturais básicos
        features['soma'] = sum(d)
        features['pares'] = sum(1 for x in d if x % 2 == 0)
        features['primos'] = sum(1 for x in d if x in PRIMES)
        features['moldura'] = sum(1 for x in d if x in MOLDURA)
        features['centro'] = sum(1 for x in d if x in CENTRO)
        features['amplitude'] = max(d) - min(d)
        
        # Consecutivos
        cons_count = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        features['consecutivos'] = cons_count
        
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
        
        # Repetidas (ÍNDICES CORRETOS)
        if idx >= 1:
            features['repetidas'] = len(set(d) & set(self.contests[idx-1]['dezenas']))
        else:
            features['repetidas'] = 0
        
        if idx >= 2:
            features['repetidas_2'] = len(set(d) & set(self.contests[idx-2]['dezenas']))
        else:
            features['repetidas_2'] = 0
        
        features['densidade_baixa'] = sum(1 for x in d if x <= 12)
        features['densidade_alta'] = sum(1 for x in d if x >= 14)
        
        # Entropia posicional
        pos_counts = np.bincount(d, minlength=26)[1:]
        probs = pos_counts / 15
        probs = np.where(probs>0, probs, 1e-10)
        features['entropia_posicional'] = entropy(probs)
        
        # EMAs e Momentums (CORRIGIDOS)
        for w in [5, 10, 20]:
            if idx >= w:
                # Janela real
                window = self.contests[idx-w+1:idx+1]
                
                # Médias CORRETAS (não média de 1 valor)
                features[f'pares_ema_{w}'] = np.mean([
                    sum(1 for x in c['dezenas'] if x%2==0) for c in window
                ])
                
                if w <= 10:  # Só calcular para janelas menores
                    features[f'primos_ema_{w}'] = np.mean([
                        sum(1 for x in c['dezenas'] if x in PRIMES) for c in window
                    ])
                    features[f'moldura_ema_{w}'] = np.mean([
                        sum(1 for x in c['dezenas'] if x in MOLDURA) for c in window
                    ])
                    
                    # Repetidas com índices REAIS
                    reps = []
                    for j in range(idx-w+1, idx+1):
                        if j > 0:
                            rep = len(set(self.contests[j]['dezenas']) & 
                                    set(self.contests[j-1]['dezenas']))
                            reps.append(rep)
                    features[f'repetidas_ema_{w}'] = np.mean(reps) if reps else 0
        
        # Momentums (CORRIGIDOS)
        for w in [5]:
            if idx >= 2*w:
                window_curr = self.contests[idx-w+1:idx+1]
                window_prev = self.contests[idx-2*w+1:idx-w+1]
                
                curr_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in window_curr])
                prev_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in window_prev])
                features[f'pares_momentum_{w}'] = curr_pares - prev_pares
                
                curr_primos = np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in window_curr])
                prev_primos = np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in window_prev])
                features[f'primos_momentum_{w}'] = curr_primos - prev_primos
            else:
                features[f'pares_momentum_{w}'] = 0
                features[f'primos_momentum_{w}'] = 0
        
        # Volatilidade
        if idx >= 10:
            recent = self.contests[idx-9:idx+1]
            pares_series = [sum(1 for x in c['dezenas'] if x%2==0) for c in recent]
            features['volatilidade'] = float(np.std(pares_series))
        else:
            features['volatilidade'] = 0.0
        
        # Garantir todas as features
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
        
        return features
    
    def build_dataset(self):
        """Constrói dataset para treino"""
        X_list, y_list = [], []
        
        for i in tqdm(range(50, self.n_contests - 1), desc="Dataset"):
            state_t = self.extract_state(i)
            state_t1 = self.extract_state(i + 1)
            
            if state_t and state_t1:
                x_vec = [float(state_t.get(name, 0)) for name in self.feature_names]
                X_list.append(x_vec)
                
                # Target: classes discretizadas (CORRIGIDO: -1)
                y_vec = [
                    np.digitize([state_t1['pares']], [6, 7, 8, 9])[0] - 1,
                    np.digitize([state_t1['primos']], [3, 4, 5, 6])[0] - 1,
                    np.digitize([state_t1['moldura']], [7, 8, 9, 10])[0] - 1,
                    np.digitize([state_t1['repetidas']], [7, 8, 9, 10])[0] - 1,
                    np.digitize([state_t1['soma']], [170, 190, 210, 230])[0] - 1,
                    np.digitize([state_t1['max_run']], [2, 3, 4, 5])[0] - 1,
                ]
                y_list.append(y_vec)
        
        return np.array(X_list), np.array(y_list)


# ============================================================
# GUIDED CSP GENERATOR (BEAM SEARCH)
# ============================================================

class GuidedGenerator:
    """
    Geração GUIADA por scores (não aleatória)
    
    Usa beam search para construir jogos incrementalmente
    Maximizando aderência estrutural
    """
    
    def __init__(self, constraints=None, beam_width=20):
        self.constraints = constraints or {}
        self.beam_width = beam_width
        
        self.fixed = set(self.constraints.get('fixas', []))
        self.excluded = set(self.constraints.get('excluidas', []))
        
        # Targets
        self.target_pares = self.constraints.get('pares_target')
        self.target_primos = self.constraints.get('primos_target')
        self.target_moldura = self.constraints.get('moldura_target')
    
    def _score_partial_game(self, partial):
        """Pontua jogo parcial durante construção"""
        score = 0.0
        remaining = 15 - len(partial)
        
        # Pares
        if self.target_pares is not None:
            current = sum(1 for d in partial if d % 2 == 0)
            max_possible = current + remaining
            min_possible = current + max(0, remaining - sum(1 for d in range(1,26) if d not in partial and d % 2 != 0))
            if self.target_pares < min_possible or self.target_pares > max_possible:
                return -1000  # Inviável
            score -= abs(current + remaining//2 - self.target_pares) * 0.5
        
        # Primos
        if self.target_primos is not None:
            current = sum(1 for d in partial if d in PRIMES)
            score -= abs(current + remaining//3 - self.target_primos) * 0.3
        
        # Diversidade espacial
        quadrants_used = len(set((d-1)//5 for d in partial))
        score += quadrants_used * 2
        
        return score
    
    def generate_beam_search(self, n_games=100, momentum=None):
        """
        Gera jogos usando BEAM SEARCH
        
        Constrói incrementalmente, mantendo top-k candidatos
        """
        candidates = []
        seen = set()
        
        for _ in tqdm(range(n_games), desc="Beam Search"):
            # Inicializar com fixas
            current_beam = [(0.0, list(self.fixed))]
            
            # Construir incrementalmente
            while current_beam and len(current_beam[0][1]) < 15:
                next_beam = []
                
                for score, partial in current_beam:
                    available = [d for d in range(1, 26) 
                               if d not in partial and d not in self.excluded]
                    
                    # Pontuar cada candidato
                    for d in available:
                        new_partial = partial + [d]
                        new_score = self._score_partial_game(new_partial)
                        
                        # Bônus por momentum
                        if momentum and d in momentum:
                            new_score += momentum[d] * 5
                        
                        next_beam.append((new_score, new_partial))
                
                # Ordenar e manter top-k
                next_beam.sort(key=lambda x: x[0], reverse=True)
                current_beam = next_beam[:self.beam_width]
            
            # Coletar jogos completos
            for score, game in current_beam:
                if len(game) == 15:
                    game_sorted = tuple(sorted(game))
                    if game_sorted not in seen:
                        seen.add(game_sorted)
                        candidates.append((score, list(game_sorted)))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in candidates]


# ============================================================
# RANKER CORRIGIDO
# ============================================================

class StructuralRanker:
    """Ranker com MultiOutputClassifier"""
    
    def __init__(self, feature_engine):
        self.engine = feature_engine
        self.model = None
        self.is_trained = False
    
    def train(self):
        print(f"\n📊 TREINANDO RANKER...")
        
        X, y = self.engine.build_dataset()
        
        if len(X) < 100:
            print("   ⚠️  Dados insuficientes")
            return False
        
        print(f"   Dataset: {X.shape[0]} amostras, {X.shape[1]} features, {y.shape[1]} targets")
        
        # Modelo base
        if XGB_AVAILABLE:
            base = xgb.XGBClassifier(n_estimators=80, max_depth=4, 
                                     learning_rate=0.05, random_state=42, verbosity=0)
        else:
            base = RandomForestClassifier(n_estimators=80, max_depth=6, 
                                         random_state=42, n_jobs=-1)
        
        # MultiOutput para features correlacionadas
        self.model = MultiOutputClassifier(base)
        
        # Treinar
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
        
        self.is_trained = True
        print(f"   ✅ Modelo treinado")
        return True
    
    def predict_proba(self, features_vector):
        """Prediz probabilidades para cada target"""
        if not self.is_trained:
            return None
        
        x = np.array([features_vector])
        probas = self.model.predict_proba(x)
        
        return probas
    
    def score_game(self, game, features_vector):
        """Pontua jogo usando predições do modelo"""
        probas = self.predict_proba(features_vector)
        if probas is None:
            return 0.5
        
        score = 0.0
        d = sorted(game)
        
        # Pares (target 0)
        actual = sum(1 for x in d if x % 2 == 0)
        cls = min(4, max(0, np.digitize([actual], [6, 7, 8, 9])[0] - 1))
        if cls < len(probas[0]):
            score += probas[0][cls] * 0.25
        
        # Primos (target 1)
        actual = sum(1 for x in d if x in PRIMES)
        cls = min(3, max(0, np.digitize([actual], [3, 4, 5, 6])[0] - 1))
        if cls < len(probas[1]):
            score += probas[1][cls] * 0.20
        
        # Moldura (target 2)
        actual = sum(1 for x in d if x in MOLDURA)
        cls = min(4, max(0, np.digitize([actual], [7, 8, 9, 10])[0] - 1))
        if cls < len(probas[2]):
            score += probas[2][cls] * 0.20
        
        # Repetidas (target 3)
        actual = sum(1 for x in d if x in MOLDURA)  # Aproximação
        cls = min(4, max(0, np.digitize([actual], [7, 8, 9, 10])[0] - 1))
        if cls < len(probas[3]):
            score += probas[3][cls] * 0.15
        
        # Soma (target 4)
        actual = sum(d)
        cls = min(4, max(0, np.digitize([actual], [170, 190, 210, 230])[0] - 1))
        if cls < len(probas[4]):
            score += probas[4][cls] * 0.10
        
        # Max run (target 5)
        run = 1; max_run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1; max_run=max(max_run,run)
            else: run=1
        cls = min(4, max(0, np.digitize([max_run], [2, 3, 4, 5])[0] - 1))
        if cls < len(probas[5]):
            score += probas[5][cls] * 0.10
        
        return min(1.0, max(0.0, score))


# ============================================================
# ENSEMBLE RANKER
# ============================================================

class EnsembleRanker:
    """Combina Markov + XGBoost + Heurísticas"""
    
    def __init__(self, feature_engine, markov_encoder, structural_ranker):
        self.engine = feature_engine
        self.markov = markov_encoder
        self.ranker = structural_ranker
        self.contests = feature_engine.contests
    
    def score_game(self, game, features_vector, current_markov_state=None):
        """Score ensemble"""
        score = 0.0
        
        # 1. XGBoost (40%)
        score += self.ranker.score_game(game, features_vector) * 0.40
        
        # 2. Markov (30%)
        if current_markov_state is not None:
            d = sorted(game)
            pares = sum(1 for x in d if x % 2 == 0)
            primos = sum(1 for x in d if x in PRIMES)
            moldura = sum(1 for x in d if x in MOLDURA)
            rep = 8  # Estimativa
            
            game_state = self.markov.encode(pares, primos, moldura, rep)
            probs = self.markov.get_transition_probs(current_markov_state)
            
            if probs is not None and game_state < len(probs):
                score += probs[game_state] * 0.30
        
        # 3. Heurísticas (30%)
        quad_count = len(set((d-1)//5 for d in game))
        score += (quad_count / 5) * 0.10
        
        baixas = sum(1 for d in game if d <= 12)
        score += (1.0 - abs(baixas - 7.5) / 7.5) * 0.10
        
        cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i]==1)
        score += max(0, 0.10 - max(0, cons - 5) * 0.02)
        
        return min(1.0, max(0.0, score))
    
    def rank_candidates(self, candidates, top_n=50):
        """Rankeia candidatos"""
        if len(self.contests) < 51:
            return [(0.5, g) for g in candidates[:top_n]]
        
        # Estado atual
        current_state = self.engine.extract_state(len(self.contests) - 1)
        if current_state is None:
            return [(0.5, g) for g in candidates[:top_n]]
        
        features_vec = [float(current_state.get(n, 0)) for n in self.engine.feature_names]
        
        # Estado Markov atual
        last = self.contests[-1]['dezenas']
        prev = self.contests[-2]['dezenas'] if len(self.contests) > 1 else last
        current_markov = self.markov.encode(
            sum(1 for x in last if x%2==0),
            sum(1 for x in last if x in PRIMES),
            sum(1 for x in last if x in MOLDURA),
            len(set(last) & set(prev))
        )
        
        # Pontuar
        scored = []
        for game in tqdm(candidates, desc="Rankeando"):
            s = self.score_game(game, features_vec, current_markov)
            scored.append((s, game))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar com diversidade
        selected = []
        for score, game in scored:
            if len(selected) >= top_n: break
            too_similar = any(len(set(game) & set(sg)) > 12 for _, sg in selected)
            if not too_similar:
                selected.append((score, game))
        
        return selected


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def collect_preferences():
    print(f"\n{'='*60}")
    print(f"🎯 PREFERÊNCIAS")
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


def display_results(scored_games):
    print(f"\n{'='*60}")
    print(f"🏆 TOP JOGOS")
    print(f"{'='*60}")
    
    for i, (score, game) in enumerate(scored_games[:15], 1):
        p = sum(1 for d in game if d%2==0)
        pr = sum(1 for d in game if d in PRIMES)
        m = sum(1 for d in game if d in MOLDURA)
        s = sum(game)
        print(f"   {i:2d}. {game} | {score:.3f}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{s}")


def main():
    print("="*60)
    print("🧬 SISTEMA PROBABILÍSTICO ESTRUTURAL v9.0")
    print("="*60)
    
    # Carregar dados
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    
    print(f"📂 {len(contests)} concursos")
    
    # Markov Encoder
    markov = MarkovStateEncoder()
    markov.build_transition_matrix(contests)
    print(f"🔢 {len(markov.state_to_id)} estados Markovianos")
    
    # Feature Engine
    engine = StateFeatureEngine(contests)
    
    # Ranker
    ranker = StructuralRanker(engine)
    ranker.train()
    
    # Ensemble
    ensemble = EnsembleRanker(engine, markov, ranker)
    
    # Preferências
    prefs = collect_preferences()
    
    # Calcular momentum para guided generation
    if len(contests) >= 50:
        recent50 = contests[-50:]
        freq_recent = Counter()
        for c in recent50:
            freq_recent.update(c['dezenas'])
        recent100 = contests[-100:]
        freq_long = Counter()
        for c in recent100:
            freq_long.update(c['dezenas'])
        momentum = {}
        for d in range(1, 26):
            momentum[d] = (freq_recent.get(d,0)/50) - (freq_long.get(d,0)/100)
    else:
        momentum = None
    
    # Geração GUIADA (beam search)
    print(f"\n🎲 BEAM SEARCH...")
    generator = GuidedGenerator(prefs, beam_width=20)
    candidates = generator.generate_beam_search(n_games=5000, momentum=momentum)
    print(f"   ✅ {len(candidates)} candidatos gerados")
    
    # Rankear
    print(f"\n📊 RANKEANDO...")
    top_games = ensemble.rank_candidates(candidates, top_n=50)
    
    # Exibir
    display_results(top_games)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
