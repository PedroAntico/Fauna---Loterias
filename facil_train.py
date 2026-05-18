#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE OTIMIZAÇÃO DE CARTEIRA - VERSÃO FINAL
=================================================
Versão 12.0 - Cobertura + Diversidade + Ensemble de Regimes

PRINCÍPIOS:
✅ NÃO tenta prever dezenas - otimiza CARTEIRA
✅ Ensemble de regimes (40% principal, 30% secundário, 20% raro, 10% caos)
✅ Penalidade forte de similaridade (anti-colapso)
✅ Score de cobertura (dezenas pouco usadas)
✅ Temperatura no MCTS (softmax probabilístico)
✅ Diversidade forçada entre jogos gerados
"""

import numpy as np
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
from math import comb, sqrt, log, exp
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
    cls = np.digitize([value], bins)[0] - 1
    return max(0, min(max_class, cls))


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
# MARKOV STATE ENCODER
# ============================================================

class MarkovStateEncoder:
    def __init__(self):
        self.state_to_id = {}
        self.id_to_state = {}
        self.transition_matrix = None
        self.state_list = []
    
    def encode(self, pares, primos, moldura, repetidas):
        fp = min(4, max(0, pares - 5))
        fpr = min(3, max(0, primos - 2))
        fm = min(4, max(0, moldura - 6))
        fr = min(4, max(0, repetidas - 6))
        key = f"{fp}_{fpr}_{fm}_{fr}"
        if key not in self.state_to_id:
            self.state_to_id[key] = len(self.state_to_id)
            self.id_to_state[self.state_to_id[key]] = {
                'faixa_pares': fp, 'faixa_primos': fpr,
                'faixa_moldura': fm, 'faixa_rep': fr
            }
        return self.state_to_id[key]
    
    def build_transition_matrix(self, contests):
        self.state_list = []
        for i in range(1, len(contests)):
            c = contests[i]; d = c['dezenas']; prev = contests[i-1]['dezenas']
            sid = self.encode(
                sum(1 for x in d if x%2==0),
                sum(1 for x in d if x in PRIMES),
                sum(1 for x in d if x in MOLDURA),
                len(set(d) & set(prev))
            )
            self.state_list.append(sid)
        
        n = len(self.state_to_id)
        if n == 0: return
        tc = np.zeros((n, n))
        for i in range(len(self.state_list)-1):
            tc[self.state_list[i], self.state_list[i+1]] += 1
        rs = tc.sum(axis=1, keepdims=True)
        rs[rs==0] = 1
        self.transition_matrix = tc / rs
        print(f"   ✅ {n} estados Markov")
    
    def get_probs(self, sid):
        if self.transition_matrix is None or sid >= len(self.transition_matrix):
            return None
        return self.transition_matrix[sid]
    
    def get_all_next_states(self, sid):
        """Retorna TODOS os próximos estados com probabilidades"""
        probs = self.get_probs(sid)
        if probs is None: return []
        return [(int(i), float(probs[i])) for i in range(len(probs)) if probs[i] > 0]


# ============================================================
# FEATURE ENGINE
# ============================================================

class StateFeatureEngine:
    def __init__(self, contests):
        self.contests = contests
        self.n_contests = len(contests)
        self.feature_names = [
            'soma','pares','primos','moldura','centro','amplitude',
            'consecutivos','max_run','distancia_media',
            'q1','q2','q3','q4','q5','repetidas','repetidas_2',
            'densidade_baixa','densidade_alta','entropia_posicional',
            'pares_ema_5','pares_ema_10','primos_ema_5','moldura_ema_5',
            'repetidas_ema_5','pares_momentum_5','volatilidade'
        ]
    
    def extract_state(self, idx):
        if idx < 50: return None
        c = self.contests[idx]; d = sorted(c['dezenas'])
        f = {}
        f['soma'] = sum(d)
        f['pares'] = sum(1 for x in d if x%2==0)
        f['primos'] = sum(1 for x in d if x in PRIMES)
        f['moldura'] = sum(1 for x in d if x in MOLDURA)
        f['centro'] = sum(1 for x in d if x in CENTRO)
        f['amplitude'] = max(d)-min(d)
        f['consecutivos'] = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        run=1; mr=1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1; mr=max(mr,run)
            else: run=1
        f['max_run'] = mr
        f['distancia_media'] = np.mean([d[i+1]-d[i] for i in range(14)])
        for qn, qs in QUADRANTES.items(): f[qn.lower()] = sum(1 for x in d if x in qs)
        f['repetidas'] = len(set(d)&set(self.contests[idx-1]['dezenas'])) if idx>=1 else 0
        f['repetidas_2'] = len(set(d)&set(self.contests[idx-2]['dezenas'])) if idx>=2 else 0
        f['densidade_baixa'] = sum(1 for x in d if x<=12)
        f['densidade_alta'] = sum(1 for x in d if x>=14)
        pc = np.bincount(d, minlength=26)[1:]; pr = pc/15; pr=np.where(pr>0,pr,1e-10)
        f['entropia_posicional'] = entropy(pr)
        for w in [5,10]:
            if idx>=w:
                ww = self.contests[idx-w+1:idx+1]
                f[f'pares_ema_{w}'] = np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in ww])
        if idx>=5:
            ww=self.contests[idx-4:idx+1]
            f['primos_ema_5']=np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in ww])
            f['moldura_ema_5']=np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in ww])
            reps=[len(set(self.contests[j]['dezenas'])&set(self.contests[j-1]['dezenas'])) for j in range(idx-4,idx+1) if j>0]
            f['repetidas_ema_5']=np.mean(reps) if reps else 0
        if idx>=10:
            curr=self.contests[idx-4:idx+1]; prev=self.contests[idx-9:idx-4]
            f['pares_momentum_5']=np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in curr])-np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in prev])
            f['volatilidade']=float(np.std([sum(1 for x in c['dezenas'] if x%2==0) for c in self.contests[idx-9:idx+1]]))
        else: f['pares_momentum_5']=0; f['volatilidade']=0.0
        for n in self.feature_names:
            if n not in f: f[n]=0.0
        return f
    
    def build_dataset(self):
        X, y = [], []
        for i in tqdm(range(50, self.n_contests-1), desc="Dataset"):
            st=self.extract_state(i); st1=self.extract_state(i+1)
            if st and st1:
                X.append([float(st.get(n,0)) for n in self.feature_names])
                y.append([
                    safe_digitize(st1['pares'],[6,7,8,9],4),
                    safe_digitize(st1['primos'],[3,4,5,6],3),
                    safe_digitize(st1['moldura'],[7,8,9,10],4),
                    safe_digitize(st1['repetidas'],[7,8,9,10],4),
                    safe_digitize(st1['soma'],[170,190,210,230],4),
                    safe_digitize(st1['max_run'],[2,3,4,5],4),
                ])
        return np.array(X), np.array(y)


# ============================================================
# RANKER
# ============================================================

class StructuralRanker:
    def __init__(self, engine):
        self.engine = engine
        self.model = None
        self.is_trained = False
        self.last_contest_dezenas = set(engine.contests[-1]['dezenas']) if engine.n_contests>0 else None
    
    def train(self):
        X, y = self.engine.build_dataset()
        if len(X)<100: return False
        base = xgb.XGBClassifier(n_estimators=80,max_depth=4,learning_rate=0.05,random_state=42,verbosity=0) if XGB_AVAILABLE else RandomForestClassifier(n_estimators=80,max_depth=6,random_state=42,n_jobs=-1)
        self.model = MultiOutputClassifier(base)
        for ti,vi in TimeSeriesSplit(3).split(X): self.model.fit(X[ti],y[ti])
        self.is_trained=True
        return True
    
    def score_game(self, game, features_vec):
        if not self.is_trained: return 0.5
        probas = self.model.predict_proba(np.array([features_vec]))
        if probas is None: return 0.5
        d=sorted(game); s=0.0
        cls=safe_digitize(sum(1 for x in d if x%2==0),[6,7,8,9],4)
        if probas[0].shape[1]>cls: s+=float(probas[0][0][cls])*0.25
        cls=safe_digitize(sum(1 for x in d if x in PRIMES),[3,4,5,6],3)
        if probas[1].shape[1]>cls: s+=float(probas[1][0][cls])*0.20
        cls=safe_digitize(sum(1 for x in d if x in MOLDURA),[7,8,9,10],4)
        if probas[2].shape[1]>cls: s+=float(probas[2][0][cls])*0.20
        rep=len(set(d)&self.last_contest_dezenas) if self.last_contest_dezenas else 8
        cls=safe_digitize(rep,[7,8,9,10],4)
        if probas[3].shape[1]>cls: s+=float(probas[3][0][cls])*0.15
        cls=safe_digitize(sum(d),[170,190,210,230],4)
        if probas[4].shape[1]>cls: s+=float(probas[4][0][cls])*0.10
        run=1;mr=1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1;mr=max(mr,run)
            else: run=1
        cls=safe_digitize(mr,[2,3,4,5],4)
        if probas[5].shape[1]>cls: s+=float(probas[5][0][cls])*0.10
        return float(min(1.0,max(0.0,s)))


# ============================================================
# GERADOR COM COBERTURA + DIVERSIDADE + TEMPERATURA
# ============================================================

class PortfolioOptimizer:
    """
    Otimizador de CARTEIRA (não prevê dezenas)
    
    Princípios:
    - Ensemble de regimes (40/30/20/10)
    - Temperatura softmax no MCTS
    - Penalidade forte de similaridade
    - Score de cobertura (dezenas pouco usadas)
    - Diversidade forçada
    """
    
    def __init__(self, constraints=None, momentum=None, markov=None,
                 current_markov_state=None, ranker=None, features_vec=None,
                 temperature=0.7, similarity_threshold=11):
        self.constraints = constraints or {}
        self.momentum = momentum or {}
        self.markov = markov
        self.current_state = current_markov_state
        self.ranker = ranker
        self.features_vec = features_vec
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        
        self.fixed = set(self.constraints.get('fixas', []))
        self.excluded = set(self.constraints.get('excluidas', []))
        
        # Controle de cobertura
        self.dezena_usage = Counter()
        self.generated_pool = []
        
        # Ensemble de regimes
        self.regime_targets = self._build_regime_ensemble()
    
    def _build_regime_ensemble(self):
        """Constrói ensemble de regimes-alvo"""
        targets = []
        
        if self.markov and self.current_state is not None:
            all_states = self.markov.get_all_next_states(self.current_state)
            all_states.sort(key=lambda x: x[1], reverse=True)
            
            # 40% regime principal (top 1)
            if all_states:
                targets.append({'weight': 0.40, 'state_id': all_states[0][0], 
                               'info': self.markov.id_to_state.get(all_states[0][0],{})})
            
            # 30% regimes secundários (top 2-3)
            for i in range(1, min(4, len(all_states))):
                targets.append({'weight': 0.10, 'state_id': all_states[i][0],
                               'info': self.markov.id_to_state.get(all_states[i][0],{})})
            
            # 20% regimes raros
            rare = [s for s in all_states if 0 < s[1] < 0.05]
            for s in rare[:4]:
                targets.append({'weight': 0.05, 'state_id': s[0],
                               'info': self.markov.id_to_state.get(s[0],{})})
        
        # 10% caos controlado (qualquer estado)
        targets.append({'weight': 0.10, 'state_id': None, 'info': {}})
        
        return targets
    
    def _pick_regime_target(self):
        """Seleciona regime-alvo baseado no ensemble"""
        r = random.random()
        cumulative = 0
        for t in self.regime_targets:
            cumulative += t['weight']
            if r <= cumulative:
                return t
        return self.regime_targets[-1]
    
    def _score_game(self, game, regime_target=None):
        """Score com cobertura, diversidade e temperatura"""
        score = 0.0
        d = sorted(game)
        
        # 1. Momentum
        if self.momentum:
            score += sum(self.momentum.get(x, 0) for x in game) * 3
        
        # 2. Diversidade espacial
        quad_used = len(set((x-1)//5 for x in game))
        score += quad_used * 5
        
        # 3. Adesão ao regime-alvo
        if regime_target and regime_target.get('info'):
            info = regime_target['info']
            target_pares = info.get('faixa_pares', 0) + 5
            target_primos = info.get('faixa_primos', 0) + 2
            target_moldura = info.get('faixa_moldura', 0) + 6
            
            actual_pares = sum(1 for x in game if x % 2 == 0)
            actual_primos = sum(1 for x in game if x in PRIMES)
            actual_moldura = sum(1 for x in game if x in MOLDURA)
            
            score -= abs(actual_pares - target_pares) * 2
            score -= abs(actual_primos - target_primos) * 1.5
            score -= abs(actual_moldura - target_moldura) * 1.5
        
        # 4. Score do ranker
        if self.ranker and self.features_vec and len(game) == 15:
            score += self.ranker.score_game(game, self.features_vec) * 10
        
        # 5. COBERTURA (dezenas pouco usadas)
        if len(game) == 15:
            coverage_bonus = 0
            for x in game:
                usage = self.dezena_usage.get(x, 0)
                coverage_bonus += 1.0 / (1.0 + usage)
            score += coverage_bonus * 3
        
        # 6. Penalidade de consecutivos
        sg = sorted(game)
        cons = sum(1 for i in range(len(sg)-1) if sg[i+1]-sg[i]==1)
        if cons > 5:
            score -= (cons - 5) * 3
        
        # 7. Penalidade de SIMILARIDADE com jogos já gerados
        for existing in self.generated_pool[-20:]:  # Últimos 20
            common = len(set(game) & set(existing))
            if common > self.similarity_threshold:
                score -= (common - self.similarity_threshold) * 5
        
        return score
    
    def _softmax_choice(self, actions, scores, temperature=None):
        """Escolha probabilística com temperatura"""
        if temperature is None:
            temperature = self.temperature
        
        if not actions:
            return None
        
        vals = np.array(scores)
        # Subtrair máximo para estabilidade numérica
        vals = vals - np.max(vals)
        probs = np.exp(vals / max(0.1, temperature))
        probs = probs / probs.sum()
        
        return np.random.choice(actions, p=probs)
    
    def generate_one(self, regime_target=None):
        """Gera UM jogo com MCTS + temperatura"""
        if regime_target is None:
            regime_target = self._pick_regime_target()
        
        root = {'game': list(self.fixed), 'children': [], 'visits': 0, 'score': 0.0}
        
        # MCTS simplificado com temperatura
        for _ in range(30):
            game = list(self.fixed)
            available = [d for d in range(1, 26) if d not in game and d not in self.excluded]
            
            # Construir jogo com escolhas softmax
            while len(game) < 15 and available:
                # Avaliar opções
                scores = []
                for d in available[:30]:
                    test = game + [d]
                    scores.append(self._score_game(test, regime_target))
                
                # Softmax
                if scores:
                    chosen_idx = self._softmax_choice_idx(scores)
                    if chosen_idx < len(available):
                        game.append(available[chosen_idx])
                        available = [d for d in available if d != available[chosen_idx]]
                    else:
                        break
                else:
                    break
            
            # Completar
            while len(game) < 15 and available:
                game.append(available[0])
                available = available[1:]
            
            if len(game) == 15:
                final_score = self._score_game(game, regime_target)
                # Backpropagation simples
                root['visits'] += 1
                root['score'] += final_score
        
        # Gerar jogo final com temperatura
        game = list(self.fixed)
        available = [d for d in range(1, 26) if d not in game and d not in self.excluded]
        
        while len(game) < 15 and available:
            scores = []
            for d in available[:30]:
                test = game + [d]
                scores.append(self._score_game(test, regime_target))
            
            if scores:
                chosen_idx = self._softmax_choice_idx(scores)
                if chosen_idx < len(available):
                    game.append(available[chosen_idx])
                    available = [d for d in available if d != available[chosen_idx]]
                    continue
            break
        
        while len(game) < 15 and available:
            game.append(available[0])
            available = available[1:]
        
        result = sorted(game)[:15]
        
        # Atualizar cobertura
        for d in result:
            self.dezena_usage[d] += 1
        self.generated_pool.append(result)
        
        return result
    
    def _softmax_choice_idx(self, scores):
        """Retorna índice escolhido por softmax"""
        if not scores:
            return 0
        vals = np.array(scores)
        vals = vals - np.max(vals)
        probs = np.exp(vals / max(0.1, self.temperature))
        probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)
    
    def generate_many(self, n_games=100):
        """Gera múltiplos jogos com ensemble de regimes"""
        games = []
        seen = set()
        
        # Distribuir por regime
        regime_counts = defaultdict(int)
        for t in self.regime_targets:
            regime_counts[t['state_id']] = int(n_games * t['weight'])
        
        # Garantir total
        total_allocated = sum(regime_counts.values())
        if total_allocated < n_games:
            regime_counts[None] = n_games - total_allocated
        
        for state_id, count in regime_counts.items():
            regime_target = None
            for t in self.regime_targets:
                if t['state_id'] == state_id:
                    regime_target = t
                    break
            
            for _ in range(count):
                game = self.generate_one(regime_target)
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
    print(f"\n📌 FIXAS:"); v=input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            f=sorted(set(int(x) for x in v.split() if 1<=int(x)<=25))
            if f: prefs['fixas']=f[:15]
        except: pass
    print(f"\n🚫 EXCLUÍDAS:"); v=input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            e=[int(x) for x in v.split() if 1<=int(x)<=25]
            if 'fixas' in prefs: e=[x for x in e if x not in prefs['fixas']]
            if e: prefs['excluidas']=sorted(set(e))
        except: pass
    return prefs if prefs else None


def display_results(games):
    print(f"\n{'='*60}")
    print(f"🏆 CARTEIRA OTIMIZADA")
    print(f"{'='*60}")
    
    # Diversidade da carteira
    all_d = [d for g in games for d in g]
    unique = len(set(all_d))
    print(f"📊 Cobertura: {unique}/25 dezenas")
    
    # Distribuição de similaridade
    sims = []
    for i in range(min(30, len(games))):
        for j in range(i+1, min(30, len(games))):
            sims.append(len(set(games[i]) & set(games[j])))
    if sims:
        print(f"📊 Similaridade média: {np.mean(sims):.1f} (máx 15)")
    
    for i, game in enumerate(games[:15], 1):
        p=sum(1 for d in game if d%2==0)
        pr=sum(1 for d in game if d in PRIMES)
        m=sum(1 for d in game if d in MOLDURA)
        s=sum(game)
        print(f"   {i:2d}. {game}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{s}")


def main():
    print("="*60)
    print("🧬 OTIMIZADOR DE CARTEIRA")
    print("="*60)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")
    
    markov = MarkovStateEncoder()
    markov.build_transition_matrix(contests)
    
    engine = StateFeatureEngine(contests)
    ranker = StructuralRanker(engine)
    ranker.train()
    
    prefs = collect_preferences()
    prefs['ultimo_concurso'] = contests[-1]['dezenas'] if contests else []
    
    # Momentum
    momentum = {}
    if len(contests) >= 100:
        recent = contests[-50:]; older = contests[-100:-50]
        fr = Counter(); [fr.update(c['dezenas']) for c in recent]
        fo = Counter(); [fo.update(c['dezenas']) for c in older]
        all_f = [fo.get(d,0)/len(older) for d in range(1,26)]
        mh, sh = np.mean(all_f), np.std(all_f)+1e-10
        for d in range(1,26): momentum[d] = (fr.get(d,0)/len(recent) - mh) / sh
    
    last = contests[-1]['dezenas']
    prev = contests[-2]['dezenas'] if len(contests)>1 else last
    current_markov = markov.encode(
        sum(1 for x in last if x%2==0), sum(1 for x in last if x in PRIMES),
        sum(1 for x in last if x in MOLDURA), len(set(last)&set(prev))
    )
    
    current_state = engine.extract_state(len(contests)-1)
    features_vec = [float(current_state.get(n,0)) for n in engine.feature_names] if current_state else None
    
    # Otimizador de Carteira
    print(f"\n🎲 GERANDO CARTEIRA OTIMIZADA...")
    opt = PortfolioOptimizer(
        constraints=prefs, momentum=momentum, markov=markov,
        current_markov_state=current_markov, ranker=ranker,
        features_vec=features_vec, temperature=0.7, similarity_threshold=11
    )
    
    games = opt.generate_many(n_games=200)
    print(f"   ✅ {len(games)} jogos na carteira")
    
    display_results(games)
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
    
