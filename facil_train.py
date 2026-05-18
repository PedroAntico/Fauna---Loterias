#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA PROBABILÍSTICO ESTRUTURAL - VERSÃO CORRIGIDA FINAL
===========================================================
Versão 11.0 - MultiOutput Corrigido + Transições Raras

CORREÇÕES:
✅ predict_proba: float(probas[i][0][cls]) - indexing correto
✅ Repetidas: compara com último concurso REAL
✅ Detecção de transições estruturais raras
✅ Reversão extrema, persistência anormal
✅ Score com último concurso real
"""

import numpy as np
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
from math import comb, sqrt, log
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

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
# SAFE DIGITIZE
# ============================================================

def safe_digitize(value, bins, max_class):
    """Digitize seguro que nunca retorna -1"""
    cls = np.digitize([value], bins)[0] - 1
    return max(0, min(max_class, cls))


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
    """Codifica estado estrutural em ID discreto"""
    
    def __init__(self):
        self.state_to_id = {}
        self.id_to_state = {}
        self.transition_matrix = None
        self.state_list = []
    
    def encode(self, pares, primos, moldura, repetidas):
        faixa_pares = min(4, max(0, pares - 5))
        faixa_primos = min(3, max(0, primos - 2))
        faixa_moldura = min(4, max(0, moldura - 6))
        faixa_rep = min(4, max(0, repetidas - 6))
        
        state_key = f"{faixa_pares}_{faixa_primos}_{faixa_moldura}_{faixa_rep}"
        
        if state_key not in self.state_to_id:
            state_id = len(self.state_to_id)
            self.state_to_id[state_key] = state_id
            self.id_to_state[state_id] = {
                'faixa_pares': faixa_pares,
                'faixa_primos': faixa_primos,
                'faixa_moldura': faixa_moldura,
                'faixa_rep': faixa_rep,
            }
        
        return self.state_to_id[state_key]
    
    def build_transition_matrix(self, contests):
        """Constrói matriz de transição Markoviana"""
        print(f"📊 Construindo matriz Markoviana...")
        
        # PASSO 1: Gerar todos os estados
        self.state_list = []
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
        
        # PASSO 2: Criar matriz
        n_states = len(self.state_to_id)
        print(f"   ✅ {n_states} estados Markovianos")
        
        if n_states == 0:
            return
        
        trans_count = np.zeros((n_states, n_states))
        
        # PASSO 3: Preencher transições
        for i in range(len(self.state_list) - 1):
            s1 = self.state_list[i]
            s2 = self.state_list[i + 1]
            trans_count[s1, s2] += 1
        
        # Normalizar
        row_sums = trans_count.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = trans_count / row_sums
        
        print(f"   ✅ Matriz {n_states}x{n_states} construída")
    
    def get_transition_probs(self, current_state_id):
        if self.transition_matrix is None:
            return None
        if current_state_id >= len(self.transition_matrix):
            return None
        return self.transition_matrix[current_state_id]
    
    def get_top_next_states(self, current_state_id, top_k=3):
        probs = self.get_transition_probs(current_state_id)
        if probs is None:
            return []
        indices = np.argsort(probs)[-top_k:][::-1]
        return [(int(idx), float(probs[idx])) for idx in indices if probs[idx] > 0]
    
    def detect_rare_transitions(self, current_state_id, threshold=0.05):
        """
        Detecta transições RARAS mas POSSÍVEIS
        
        Estados com probabilidade baixa (<5%) mas não zero
        Podem indicar reversões extremas
        """
        probs = self.get_transition_probs(current_state_id)
        if probs is None:
            return []
        
        rare = []
        for state_id, prob in enumerate(probs):
            if 0 < prob < threshold:
                info = self.id_to_state.get(state_id, {})
                rare.append({
                    'state_id': state_id,
                    'probability': float(prob),
                    'pares': info.get('faixa_pares', 0) + 5,
                    'primos': info.get('faixa_primos', 0) + 2,
                    'moldura': info.get('faixa_moldura', 0) + 6,
                })
        
        rare.sort(key=lambda x: x['probability'], reverse=True)
        return rare


# ============================================================
# FEATURE ENGINE
# ============================================================

class StateFeatureEngine:
    """Extrai features estruturais"""
    
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
            'pares_ema_5', 'pares_ema_10',
            'primos_ema_5', 'moldura_ema_5',
            'repetidas_ema_5',
            'pares_momentum_5', 'primos_momentum_5',
            'volatilidade'
        ]
    
    def extract_state(self, idx):
        if idx < 50:
            return None
        
        contest = self.contests[idx]
        dezenas = contest['dezenas']
        d = sorted(dezenas)
        
        features = {}
        
        features['soma'] = sum(d)
        features['pares'] = sum(1 for x in d if x % 2 == 0)
        features['primos'] = sum(1 for x in d if x in PRIMES)
        features['moldura'] = sum(1 for x in d if x in MOLDURA)
        features['centro'] = sum(1 for x in d if x in CENTRO)
        features['amplitude'] = max(d) - min(d)
        
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        features['consecutivos'] = cons
        
        run = 1; max_run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1; max_run=max(max_run,run)
            else: run=1
        features['max_run'] = max_run
        
        features['distancia_media'] = np.mean([d[i+1]-d[i] for i in range(14)])
        
        for qname, qset in QUADRANTES.items():
            features[qname.lower()] = sum(1 for x in d if x in qset)
        
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
        
        pos_counts = np.bincount(d, minlength=26)[1:]
        probs = pos_counts / 15
        probs = np.where(probs>0, probs, 1e-10)
        features['entropia_posicional'] = entropy(probs)
        
        for w in [5, 10]:
            if idx >= w:
                window = self.contests[idx-w+1:idx+1]
                features[f'pares_ema_{w}'] = np.mean([
                    sum(1 for x in c['dezenas'] if x%2==0) for c in window
                ])
        
        for w in [5]:
            if idx >= w:
                window = self.contests[idx-w+1:idx+1]
                features[f'primos_ema_{w}'] = np.mean([
                    sum(1 for x in c['dezenas'] if x in PRIMES) for c in window
                ])
                features[f'moldura_ema_{w}'] = np.mean([
                    sum(1 for x in c['dezenas'] if x in MOLDURA) for c in window
                ])
                
                reps = []
                for j in range(idx-w+1, idx+1):
                    if j > 0:
                        rep = len(set(self.contests[j]['dezenas']) & 
                                set(self.contests[j-1]['dezenas']))
                        reps.append(rep)
                features[f'repetidas_ema_{w}'] = np.mean(reps) if reps else 0
        
        if idx >= 10:
            curr = self.contests[idx-4:idx+1]
            prev = self.contests[idx-9:idx-4]
            curr_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in curr])
            prev_pares = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in prev])
            features['pares_momentum_5'] = curr_pares - prev_pares
        else:
            features['pares_momentum_5'] = 0
        
        if idx >= 10:
            recent = self.contests[idx-9:idx+1]
            pares_series = [sum(1 for x in c['dezenas'] if x%2==0) for c in recent]
            features['volatilidade'] = float(np.std(pares_series))
        else:
            features['volatilidade'] = 0.0
        
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
        
        return features
    
    def build_dataset(self):
        X_list, y_list = [], []
        
        for i in tqdm(range(50, self.n_contests - 1), desc="Dataset"):
            state_t = self.extract_state(i)
            state_t1 = self.extract_state(i + 1)
            
            if state_t and state_t1:
                x_vec = [float(state_t.get(name, 0)) for name in self.feature_names]
                X_list.append(x_vec)
                
                y_vec = [
                    safe_digitize(state_t1['pares'], [6, 7, 8, 9], 4),
                    safe_digitize(state_t1['primos'], [3, 4, 5, 6], 3),
                    safe_digitize(state_t1['moldura'], [7, 8, 9, 10], 4),
                    safe_digitize(state_t1['repetidas'], [7, 8, 9, 10], 4),
                    safe_digitize(state_t1['soma'], [170, 190, 210, 230], 4),
                    safe_digitize(state_t1['max_run'], [2, 3, 4, 5], 4),
                ]
                y_list.append(y_vec)
        
        return np.array(X_list), np.array(y_list)


# ============================================================
# RANKER CORRIGIDO (MultiOutput indexing)
# ============================================================

class StructuralRanker:
    """Ranker com MultiOutputClassifier - indexing CORRIGIDO"""
    
    def __init__(self, feature_engine):
        self.engine = feature_engine
        self.model = None
        self.is_trained = False
        self.last_contest_dezenas = None
        
        # Guardar último concurso para calcular repetidas
        if feature_engine.n_contests > 0:
            self.last_contest_dezenas = set(feature_engine.contests[-1]['dezenas'])
    
    def train(self):
        print(f"\n📊 TREINANDO RANKER...")
        
        X, y = self.engine.build_dataset()
        
        if len(X) < 100:
            print("   ⚠️  Dados insuficientes")
            return False
        
        print(f"   Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
        
        if XGB_AVAILABLE:
            base = xgb.XGBClassifier(n_estimators=80, max_depth=4,
                                     learning_rate=0.05, random_state=42, verbosity=0)
        else:
            base = RandomForestClassifier(n_estimators=80, max_depth=6,
                                         random_state=42, n_jobs=-1)
        
        self.model = MultiOutputClassifier(base)
        
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
        
        self.is_trained = True
        print(f"   ✅ Treinado")
        return True
    
    def predict_proba(self, features_vector):
        """
        Retorna probabilidades para cada target
        
        MultiOutputClassifier.predict_proba() retorna:
        [array([[p0, p1, p2, ...]]), array([[p0, p1, ...]]), ...]
        
        Onde cada elemento é um array 2D (1 amostra × n classes)
        """
        if not self.is_trained:
            return None
        return self.model.predict_proba(np.array([features_vector]))
    
    def score_game(self, game, features_vector):
        """
        Score CORRIGIDO:
        - Indexing: float(probas[i][0][cls])
        - Repetidas: compara com último concurso REAL
        """
        probas = self.predict_proba(features_vector)
        if probas is None:
            return 0.5
        
        d = sorted(game)
        score = 0.0
        
        # 1. Pares (target 0)
        cls = safe_digitize(sum(1 for x in d if x % 2 == 0), [6, 7, 8, 9], 4)
        if probas[0].shape[1] > cls:
            score += float(probas[0][0][cls]) * 0.25
        
        # 2. Primos (target 1)
        cls = safe_digitize(sum(1 for x in d if x in PRIMES), [3, 4, 5, 6], 3)
        if probas[1].shape[1] > cls:
            score += float(probas[1][0][cls]) * 0.20
        
        # 3. Moldura (target 2)
        cls = safe_digitize(sum(1 for x in d if x in MOLDURA), [7, 8, 9, 10], 4)
        if probas[2].shape[1] > cls:
            score += float(probas[2][0][cls]) * 0.20
        
        # 4. Repetidas (target 3) - CORRIGIDO: usa último concurso REAL
        if self.last_contest_dezenas:
            rep = len(set(d) & self.last_contest_dezenas)
        else:
            rep = 8  # Estimativa
        cls = safe_digitize(rep, [7, 8, 9, 10], 4)
        if probas[3].shape[1] > cls:
            score += float(probas[3][0][cls]) * 0.15
        
        # 5. Soma (target 4)
        cls = safe_digitize(sum(d), [170, 190, 210, 230], 4)
        if probas[4].shape[1] > cls:
            score += float(probas[4][0][cls]) * 0.10
        
        # 6. Max run (target 5)
        run = 1; mr = 1
        for i in range(len(d)-1):
            if d[i+1] - d[i] == 1:
                run += 1
                mr = max(mr, run)
            else:
                run = 1
        cls = safe_digitize(mr, [2, 3, 4, 5], 4)
        if probas[5].shape[1] > cls:
            score += float(probas[5][0][cls]) * 0.10
        
        return float(min(1.0, max(0.0, score)))


# ============================================================
# MCTS GENERATOR
# ============================================================

class MCTSNode:
    def __init__(self, game=None, parent=None):
        self.game = game or []
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_score = 0.0
        self.untried_actions = None
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0 if self.untried_actions is not None else False
    
    def is_terminal(self):
        return len(self.game) >= 15
    
    def ucb_score(self, exploration_weight=1.4):
        if self.visits == 0:
            return float('inf')
        return self.total_score / self.visits + exploration_weight * sqrt(
            log(self.parent.visits + 1) / self.visits
        )


class MCTSGenerator:
    """Monte Carlo Tree Search para geração de jogos"""
    
    def __init__(self, constraints=None, momentum=None, markov=None,
                 current_markov_state=None, ranker=None, features_vec=None,
                 rare_transitions=None, n_simulations=50, exploration_weight=1.4):
        self.constraints = constraints or {}
        self.momentum = momentum or {}
        self.markov = markov
        self.current_markov_state = current_markov_state
        self.ranker = ranker
        self.features_vec = features_vec
        self.rare_transitions = rare_transitions or []
        
        self.n_simulations = n_simulations
        self.exploration_weight = exploration_weight
        
        self.fixed = set(self.constraints.get('fixas', []))
        self.excluded = set(self.constraints.get('excluidas', []))
        
        self._score_cache = {}
    
    def _get_available_actions(self, game):
        return [d for d in range(1, 26) if d not in game and d not in self.excluded]
    
    def _score_game(self, game):
        key = tuple(sorted(game))
        if key in self._score_cache:
            return self._score_cache[key]
        
        score = 0.0
        remaining = 15 - len(game)
        
        # 1. Momentum (Z-score)
        if self.momentum:
            for d in game:
                score += self.momentum.get(d, 0) * 3
        
        # 2. Diversidade espacial
        quad_used = len(set((d-1)//5 for d in game))
        score += quad_used * 5
        
        # 3. Proximidade com targets
        if self.constraints.get('pares_target'):
            current = sum(1 for d in game if d % 2 == 0)
            target = self.constraints['pares_target']
            if remaining > 0:
                max_possible = current + remaining
                min_possible = current
                if target < min_possible or target > max_possible:
                    score -= 50
            score -= abs(current - target) * 2
        
        # 4. Penalidade de consecutivos
        sg = sorted(game)
        cons = sum(1 for i in range(len(sg)-1) if sg[i+1]-sg[i]==1)
        if cons > 5:
            score -= (cons - 5) * 3
        
        # 5. Score do ranker
        if self.ranker and self.features_vec and len(game) == 15:
            score += self.ranker.score_game(game, self.features_vec) * 10
        
        # 6. Score Markov
        if self.markov and self.current_markov_state is not None and len(game) == 15:
            pares = sum(1 for d in game if d % 2 == 0)
            primos = sum(1 for d in game if d in PRIMES)
            moldura = sum(1 for d in game if d in MOLDURA)
            rep = len(set(game) & set(self.constraints.get('ultimo_concurso', [])))
            game_state = self.markov.encode(pares, primos, moldura, rep)
            probs = self.markov.get_transition_probs(self.current_markov_state)
            if probs is not None and game_state < len(probs):
                score += probs[game_state] * 15
        
        # 7. Bônus por transições raras (reversões extremas)
        if self.rare_transitions and len(game) == 15:
            pares = sum(1 for d in game if d % 2 == 0)
            primos = sum(1 for d in game if d in PRIMES)
            moldura = sum(1 for d in game if d in MOLDURA)
            rep = len(set(game) & set(self.constraints.get('ultimo_concurso', [])))
            game_state = self.markov.encode(pares, primos, moldura, rep)
            
            for rare in self.rare_transitions:
                if rare['state_id'] == game_state:
                    score += rare['probability'] * 8  # Bônus proporcional
                    break
        
        self._score_cache[key] = score
        return score
    
    def _select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            node = max(node.children, key=lambda c: c.ucb_score(self.exploration_weight))
        return node
    
    def _expand(self, node):
        if node.untried_actions is None:
            node.untried_actions = self._get_available_actions(node.game)
        
        if not node.untried_actions:
            return node
        
        action_scores = []
        for action in node.untried_actions[:20]:
            new_game = node.game + [action]
            score = self._score_game(new_game)
            action_scores.append((score, action))
        
        action_scores.sort(key=lambda x: x[0], reverse=True)
        best_action = action_scores[0][1]
        
        node.untried_actions.remove(best_action)
        
        child = MCTSNode(node.game + [best_action], parent=node)
        node.children.append(child)
        
        return child
    
    def _simulate(self, node):
        game = list(node.game)
        available = self._get_available_actions(game)
        
        while len(game) < 15 and available:
            scores = [(self._score_game(game + [d]), d) for d in available[:30]]
            scores.sort(key=lambda x: x[0], reverse=True)
            top_n = min(3, len(scores))
            chosen = scores[random.randint(0, top_n-1)][1]
            game.append(chosen)
            available = [d for d in available if d != chosen]
        
        return self._score_game(game)
    
    def _backpropagate(self, node, score):
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent
    
    def generate_one(self):
        root = MCTSNode(list(self.fixed))
        
        for _ in range(self.n_simulations):
            leaf = self._select(root)
            score = self._score_game(leaf.game) if leaf.is_terminal() else self._simulate(leaf)
            self._backpropagate(leaf, score)
        
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            game = best_child.game
            
            if len(game) < 15:
                available = self._get_available_actions(game)
                while len(game) < 15 and available:
                    scores = [(self._score_game(game + [d]), d) for d in available]
                    scores.sort(key=lambda x: x[0], reverse=True)
                    game.append(scores[0][1])
                    available = [d for d in available if d != game[-1]]
            
            return sorted(game)[:15]
        
        game = list(self.fixed)
        available = self._get_available_actions(game)
        while len(game) < 15 and available:
            game.append(random.choice(available))
            available = self._get_available_actions(game)
        return sorted(game)[:15]
    
    def generate_many(self, n_games=100):
        games = []
        seen = set()
        for _ in tqdm(range(n_games), desc="MCTS"):
            game = self.generate_one()
            key = tuple(game)
            if key not in seen and len(game) == 15:
                seen.add(key)
                games.append(game)
        return games


# ============================================================
# INTERFACE
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


def display_results(games, markov=None, current_state=None, rare_transitions=None):
    print(f"\n{'='*60}")
    print(f"🏆 TOP JOGOS (MCTS + MARKOV)")
    print(f"{'='*60}")
    
    if markov and current_state is not None:
        top_states = markov.get_top_next_states(current_state, 3)
        if top_states:
            print(f"📊 Estados Markov mais prováveis:")
            for sid, prob in top_states:
                info = markov.id_to_state.get(sid, {})
                print(f"   P:{info.get('faixa_pares',0)+5} "
                      f"Pr:{info.get('faixa_primos',0)+2} "
                      f"M:{info.get('faixa_moldura',0)+6} "
                      f"({prob:.1%})")
        
        if rare_transitions:
            print(f"\n⚠️  Transições RARAS detectadas:")
            for r in rare_transitions[:3]:
                print(f"   P:{r['pares']} Pr:{r['primos']} "
                      f"M:{r['moldura']} ({r['probability']:.1%})")
    
    for i, game in enumerate(games[:15], 1):
        p = sum(1 for d in game if d%2==0)
        pr = sum(1 for d in game if d in PRIMES)
        m = sum(1 for d in game if d in MOLDURA)
        s = sum(game)
        print(f"   {i:2d}. {game}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{s}")


def main():
    print("="*60)
    print("🧬 MCTS + MARKOV + RARE TRANSITIONS")
    print("="*60)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    
    print(f"📂 {len(contests)} concursos")
    
    # Markov
    markov = MarkovStateEncoder()
    markov.build_transition_matrix(contests)
    
    # Feature Engine
    engine = StateFeatureEngine(contests)
    
    # Ranker
    ranker = StructuralRanker(engine)
    ranker.train()
    
    # Preferências
    prefs = collect_preferences()
    prefs['ultimo_concurso'] = contests[-1]['dezenas'] if contests else []
    
    # Momentum (Z-score)
    if len(contests) >= 100:
        recent = contests[-50:]
        older = contests[-100:-50]
        
        freq_recent = Counter()
        for c in recent: freq_recent.update(c['dezenas'])
        
        freq_older = Counter()
        for c in older: freq_older.update(c['dezenas'])
        
        all_freqs = [freq_older.get(d, 0) / len(older) for d in range(1, 26)]
        mean_hist = np.mean(all_freqs)
        std_hist = np.std(all_freqs) + 1e-10
        
        momentum = {}
        for d in range(1, 26):
            recent_rate = freq_recent.get(d, 0) / len(recent)
            momentum[d] = (recent_rate - mean_hist) / std_hist
    else:
        momentum = {}
    
    # Estado Markov atual
    last = contests[-1]['dezenas']
    prev = contests[-2]['dezenas'] if len(contests) > 1 else last
    current_markov = markov.encode(
        sum(1 for x in last if x%2==0),
        sum(1 for x in last if x in PRIMES),
        sum(1 for x in last if x in MOLDURA),
        len(set(last) & set(prev))
    )
    
    # Transições raras
    rare_transitions = markov.detect_rare_transitions(current_markov, threshold=0.05)
    
    # Features atuais
    current_state = engine.extract_state(len(contests) - 1)
    features_vec = [float(current_state.get(n, 0)) for n in engine.feature_names] if current_state else None
    
    # MCTS
    print(f"\n🎲 MCTS GERANDO JOGOS...")
    mcts = MCTSGenerator(
        constraints=prefs,
        momentum=momentum,
        markov=markov,
        current_markov_state=current_markov,
        ranker=ranker,
        features_vec=features_vec,
        rare_transitions=rare_transitions,
        n_simulations=50,
        exploration_weight=1.4
    )
    
    games = mcts.generate_many(n_games=200)
    print(f"   ✅ {len(games)} jogos gerados")
    
    display_results(games, markov, current_markov, rare_transitions)
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
