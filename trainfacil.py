#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE VALIDAÇÃO CIENTÍFICA - OTIMIZADOR DE CARTEIRA
==========================================================
Versão 13.0 - Backtest + Anti-Recência + Diversidade Estrutural

MELHORIAS CRÍTICAS:
✅ Penalidade de assinatura estrutural (não apenas dezenas)
✅ Distribuição forçada de moldura (20/40/30/10)
✅ Desacoplamento momentum (só expansão, não score)
✅ Anti-recência (reversão à média)
✅ Backtest real contra baseline aleatória
✅ Métricas de validação (taxa de 11+, 12+)
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

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================
# CONJUNTOS
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}

# ============================================================
# SAFE DIGITIZE
# ============================================================

def safe_digitize(value, bins, max_class):
    return max(0, min(max_class, np.digitize([value], bins)[0] - 1))


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
# MARKOV
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
                'pares': fp+5, 'primos': fpr+2, 'moldura': fm+6, 'rep': fr+6
            }
        return self.state_to_id[key]
    
    def build_transition_matrix(self, contests):
        self.state_list = []
        for i in range(1, len(contests)):
            c = contests[i]; d = c['dezenas']; prev = contests[i-1]['dezenas']
            self.state_list.append(self.encode(
                sum(1 for x in d if x%2==0), sum(1 for x in d if x in PRIMES),
                sum(1 for x in d if x in MOLDURA), len(set(d)&set(prev))))
        n = len(self.state_to_id)
        if n == 0: return
        tc = np.zeros((n, n))
        for i in range(len(self.state_list)-1):
            tc[self.state_list[i], self.state_list[i+1]] += 1
        rs = tc.sum(axis=1, keepdims=True); rs[rs==0]=1
        self.transition_matrix = tc / rs
    
    def get_all_next(self, sid):
        if self.transition_matrix is None or sid >= len(self.transition_matrix):
            return []
        return [(int(i), float(self.transition_matrix[sid][i])) 
                for i in range(len(self.transition_matrix[sid])) 
                if self.transition_matrix[sid][i] > 0]


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
            'q1','q2','q3','q4','q5','repetidas',
            'densidade_baixa','densidade_alta','entropia_posicional',
            'pares_ema_5','primos_ema_5','moldura_ema_5',
            'pares_momentum_5','volatilidade'
        ]
    
    def extract_state(self, idx):
        if idx < 50: return None
        c = self.contests[idx]; d = sorted(c['dezenas'])
        f = {}
        f['soma']=sum(d); f['pares']=sum(1 for x in d if x%2==0)
        f['primos']=sum(1 for x in d if x in PRIMES)
        f['moldura']=sum(1 for x in d if x in MOLDURA)
        f['centro']=sum(1 for x in d if x in CENTRO)
        f['amplitude']=max(d)-min(d)
        f['consecutivos']=sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        run=1;mr=1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run+=1;mr=max(mr,run)
            else: run=1
        f['max_run']=mr
        f['distancia_media']=np.mean([d[i+1]-d[i] for i in range(14)])
        for qn,qs in [('q1',{1,2,3,4,5}),('q2',{6,7,8,9,10}),('q3',{11,12,13,14,15}),('q4',{16,17,18,19,20}),('q5',{21,22,23,24,25})]:
            f[qn]=sum(1 for x in d if x in qs)
        f['repetidas']=len(set(d)&set(self.contests[idx-1]['dezenas'])) if idx>=1 else 0
        f['densidade_baixa']=sum(1 for x in d if x<=12)
        f['densidade_alta']=sum(1 for x in d if x>=14)
        pc=np.bincount(d,minlength=26)[1:];pr=pc/15;pr=np.where(pr>0,pr,1e-10)
        f['entropia_posicional']=entropy(pr)
        if idx>=5:
            ww=self.contests[idx-4:idx+1]
            f['pares_ema_5']=np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in ww])
            f['primos_ema_5']=np.mean([sum(1 for x in c['dezenas'] if x in PRIMES) for c in ww])
            f['moldura_ema_5']=np.mean([sum(1 for x in c['dezenas'] if x in MOLDURA) for c in ww])
        else: f['pares_ema_5']=f['primos_ema_5']=f['moldura_ema_5']=0
        if idx>=10:
            curr=self.contests[idx-4:idx+1];prev=self.contests[idx-9:idx-4]
            f['pares_momentum_5']=np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in curr])-np.mean([sum(1 for x in c['dezenas'] if x%2==0) for c in prev])
            f['volatilidade']=float(np.std([sum(1 for x in c['dezenas'] if x%2==0) for c in self.contests[idx-9:idx+1]]))
        else: f['pares_momentum_5']=0;f['volatilidade']=0.0
        for n in self.feature_names:
            if n not in f: f[n]=0.0
        return f
    
    def build_dataset(self):
        X,y=[],[]
        for i in tqdm(range(50,self.n_contests-1),desc="Dataset"):
            st=self.extract_state(i);st1=self.extract_state(i+1)
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
        return np.array(X),np.array(y)


# ============================================================
# RANKER
# ============================================================

class StructuralRanker:
    def __init__(self,engine): self.engine=engine; self.model=None; self.is_trained=False
    
    def train(self):
        X,y=self.engine.build_dataset()
        if len(X)<100: return False
        base=xgb.XGBClassifier(n_estimators=80,max_depth=4,learning_rate=0.05,random_state=42,verbosity=0) if XGB_AVAILABLE else RandomForestClassifier(n_estimators=80,max_depth=6,random_state=42,n_jobs=-1)
        self.model=MultiOutputClassifier(base)
        for ti,vi in TimeSeriesSplit(3).split(X): self.model.fit(X[ti],y[ti])
        self.is_trained=True; return True
    
    def score_game(self,game,features_vec,last_contest_dezenas):
        if not self.is_trained: return 0.5
        probas=self.model.predict_proba(np.array([features_vec]))
        if probas is None: return 0.5
        d=sorted(game);s=0.0
        cls=safe_digitize(sum(1 for x in d if x%2==0),[6,7,8,9],4)
        if probas[0].shape[1]>cls: s+=float(probas[0][0][cls])*0.25
        cls=safe_digitize(sum(1 for x in d if x in PRIMES),[3,4,5,6],3)
        if probas[1].shape[1]>cls: s+=float(probas[1][0][cls])*0.20
        cls=safe_digitize(sum(1 for x in d if x in MOLDURA),[7,8,9,10],4)
        if probas[2].shape[1]>cls: s+=float(probas[2][0][cls])*0.20
        rep=len(set(d)&last_contest_dezenas) if last_contest_dezenas else 8
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
# OTIMIZADOR DE CARTEIRA (v13)
# ============================================================

class PortfolioOptimizerV13:
    """Otimizador com diversidade estrutural forçada + anti-recência"""
    
    def __init__(self, constraints=None, momentum=None, anti_momentum=None,
                 markov=None, current_state=None, ranker=None, features_vec=None,
                 last_contest=None, temperature=0.7):
        self.constraints = constraints or {}
        self.momentum = momentum or {}
        self.anti_momentum = anti_momentum or {}
        self.markov = markov
        self.current_state = current_state
        self.ranker = ranker
        self.features_vec = features_vec
        self.last_contest = last_contest or []
        self.temperature = temperature
        
        self.fixed = set(self.constraints.get('fixas', []))
        self.excluded = set(self.constraints.get('excluidas', []))
        
        # Controle de diversidade estrutural
        self.dezena_usage = Counter()
        self.structure_signatures = Counter()  # Assinaturas estruturais
        self.generated_pool = []
        
        # Distribuição forçada de moldura
        self.moldura_targets = [8, 8, 9, 9, 9, 9, 10, 10, 10, 11]  # 20/40/30/10
        
        # Ensemble de regimes
        self.regime_targets = self._build_regime_ensemble()
    
    def _build_regime_ensemble(self):
        targets = []
        if self.markov and self.current_state is not None:
            all_states = self.markov.get_all_next(self.current_state)
            all_states.sort(key=lambda x: x[1], reverse=True)
            if all_states:
                targets.append({'weight':0.40,'state_id':all_states[0][0],
                               'info':self.markov.id_to_state.get(all_states[0][0],{})})
            for i in range(1,min(4,len(all_states))):
                targets.append({'weight':0.10,'state_id':all_states[i][0],
                               'info':self.markov.id_to_state.get(all_states[i][0],{})})
            rare=[s for s in all_states if 0<s[1]<0.05]
            for s in rare[:4]:
                targets.append({'weight':0.05,'state_id':s[0],
                               'info':self.markov.id_to_state.get(s[0],{})})
        targets.append({'weight':0.10,'state_id':None,'info':{}})
        return targets
    
    def _pick_regime_target(self):
        r=random.random();cum=0
        for t in self.regime_targets:
            cum+=t['weight']
            if r<=cum: return t
        return self.regime_targets[-1]
    
    def _structure_signature(self, game):
        """Assinatura estrutural (não dezenas)"""
    
        d = sorted(game)
    
        # max_run
        run = 1
        max_run = 1
    
        for i in range(len(d) - 1):
            if d[i + 1] - d[i] == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
    
        return (
            sum(1 for x in d if x % 2 == 0),                 # pares
            sum(1 for x in d if x in PRIMES),                # primos
            sum(1 for x in d if x in MOLDURA),               # moldura
            len(set(d) & set(self.last_contest)) if self.last_contest else 8,
            sum(d) // 10,                                    # soma bucket
            max_run                                           # max_run
        )
    
    def _score_game(self, game, regime_target=None):
        score=0.0; d=sorted(game)
        
        # 1. Momentum (APENAS para expansão, peso reduzido no score)
        # NÃO contamina score estrutural
        
        # 2. Anti-recência (reversão à média)
        if self.anti_momentum:
            score += sum(self.anti_momentum.get(x,0) for x in game) * 4
        
        # 3. Diversidade espacial
        score += len(set((x-1)//5 for x in game)) * 5
        
        # 4. Adesão ao regime
        if regime_target and regime_target.get('info'):
            info=regime_target['info']
            score -= abs(sum(1 for x in game if x%2==0)-info.get('pares',7))*2
            score -= abs(sum(1 for x in game if x in PRIMES)-info.get('primos',5))*1.5
            score -= abs(sum(1 for x in game if x in MOLDURA)-info.get('moldura',9))*1.5
        
        # 5. Ranker
        if self.ranker and self.features_vec and len(game)==15:
            score += self.ranker.score_game(game, self.features_vec, 
                                           set(self.last_contest) if self.last_contest else None)*10
        
        # 6. COBERTURA DE DEZENAS
        if len(game)==15:
            for x in game:
                score += 1.0/(1.0+self.dezena_usage.get(x,0))*3
        
        # 7. PENALIDADE DE ASSINATURA ESTRUTURAL (NOVO!)
        if len(game)==15:
            sig = self._structure_signature(game)
            sig_count = self.structure_signatures.get(sig, 0)
            score -= sig_count * 1.5  # Penaliza assinaturas repetidas
        
        # 8. Penalidade de similaridade
        for existing in self.generated_pool[-20:]:
            common = len(set(game)&set(existing))
            if common > 12: score -= (common-11)*5
        
        # 9. Penalidade de consecutivos
        sg=sorted(game); cons=sum(1 for i in range(len(sg)-1) if sg[i+1]-sg[i]==1)
        if cons>5: score-=(cons-5)*3
        
        return score
    
    def _softmax_choice(self, actions, scores):
        if not actions: return None
        vals=np.array(scores); vals=vals-np.max(vals)
        probs=np.exp(vals/max(0.1,self.temperature)); probs/=probs.sum()
        return np.random.choice(actions,p=probs)
    
    def generate_one(self, regime_target=None):
        if regime_target is None: regime_target=self._pick_regime_target()
        
        # Escolher target de moldura forçada
        moldura_target = random.choice(self.moldura_targets)
        
        game=list(self.fixed)
        available=[d for d in range(1,26) if d not in game and d not in self.excluded]
        
        # Expansão com MOMENTUM (só aqui!)
        while len(game)<15 and available:
            scores=[]
            for d in available[:30]:
                test=game+[d]
                s=self._score_game(test, regime_target)
                # Momentum influencia EXPANSÃO (não score final)
                if self.momentum:
                    s+=self.momentum.get(d,0)*2
                # Anti-momentum
                if self.anti_momentum:
                    s+=self.anti_momentum.get(d,0)*1
                # Direcionar moldura
                current_mold=sum(1 for x in test if x in MOLDURA)
                remaining=15-len(test)
                if current_mold>moldura_target:
                    s-=5  # Penalidade se já excedeu
                elif current_mold+remaining<moldura_target:
                    if d not in MOLDURA and any(x in MOLDURA for x in available if x!=d):
                        s-=2  # Penalidade se não vai atingir target
                scores.append(s)
            
            chosen=self._softmax_choice(available[:len(scores)],scores)
            if chosen:
                game.append(chosen)
                available=[d for d in available if d!=chosen]
            else: break
        
        # Completar
        while len(game)<15 and available:
            game.append(available[0]); available=available[1:]
        
        result=sorted(game)[:15]
        
        # Atualizar estatísticas
        for d in result: self.dezena_usage[d]+=1
        if len(result)==15:
            sig=self._structure_signature(result)
            self.structure_signatures[sig]+=1
        self.generated_pool.append(result)
        
        return result
    
    def generate_many(self, n_games=200):
        games=[]; seen=set()
        regime_counts=defaultdict(int)
        for t in self.regime_targets:
            regime_counts[t['state_id']]+=int(n_games*t['weight'])
        total=sum(regime_counts.values())
        if total<n_games: regime_counts[None]=n_games-total
        
        for state_id,count in regime_counts.items():
            rt=None
            for t in self.regime_targets:
                if t['state_id']==state_id: rt=t; break
            for _ in range(count):
                game=self.generate_one(rt)
                key=tuple(game)
                if key not in seen and len(game)==15:
                    seen.add(key); games.append(game)
        return games


# ============================================================
# BACKTEST REAL
# ============================================================

def run_backtest(contests, n_test=500, n_games=50):
    """
    Backtest REAL:
    - Para cada concurso nos últimos N
    - Treina apenas com dados ANTERIORES
    - Gera jogos e verifica acertos
    - Compara com baseline aleatória
    """
    print(f"\n{'='*60}")
    print(f"🔬 BACKTEST REAL ({n_test} concursos)")
    print(f"{'='*60}")
    
    results = {'estrategia': {11:0,12:0,13:0,14:0,15:0},
               'aleatorio': {11:0,12:0,13:0,14:0,15:0}}
    
    start_idx = max(100, len(contests) - n_test)
    
    for i in tqdm(range(start_idx, len(contests)), desc="Backtest"):
        # Dados disponíveis ATÉ o concurso i (sem look-ahead)
        train_contests = contests[:i]
        actual_draw = set(contests[i]['dezenas'])
        
        # Treinar modelo com dados passados
        engine = StateFeatureEngine(train_contests)
        ranker = StructuralRanker(engine)
        ranker.train()
        
        markov = MarkovStateEncoder()
        markov.build_transition_matrix(train_contests)
        
        # Estado atual
        last = train_contests[-1]['dezenas']
        prev = train_contests[-2]['dezenas'] if len(train_contests)>1 else last
        current_markov = markov.encode(
            sum(1 for x in last if x%2==0), sum(1 for x in last if x in PRIMES),
            sum(1 for x in last if x in MOLDURA), len(set(last)&set(prev)))
        
        current_state = engine.extract_state(len(train_contests)-1)
        features_vec = [float(current_state.get(n,0)) for n in engine.feature_names] if current_state else None
        
        # Momentum
        momentum={}; anti_momentum={}
        if len(train_contests)>=100:
            recent=train_contests[-50:]; older=train_contests[-100:-50]
            fr=Counter(); [fr.update(c['dezenas']) for c in recent]
            fo=Counter(); [fo.update(c['dezenas']) for c in older]
            all_f=[fo.get(d,0)/len(older) for d in range(1,26)]
            mh,sh=np.mean(all_f),np.std(all_f)+1e-10
            for d in range(1,26):
                momentum[d]=(fr.get(d,0)/len(recent)-mh)/sh
                anti_momentum[d]=-momentum[d]*0.3
        
        # Gerar jogos com estratégia
        opt = PortfolioOptimizerV13(
            constraints={}, momentum=momentum, anti_momentum=anti_momentum,
            markov=markov, current_state=current_markov, ranker=ranker,
            features_vec=features_vec, last_contest=last, temperature=0.7)
        
        strategy_games = opt.generate_many(n_games)
        
        # Gerar jogos aleatórios (baseline)
        random_games = []
        for _ in range(n_games):
            g = sorted(np.random.choice(range(1,26), 15, replace=False))
            random_games.append(g)
        
        # Verificar acertos
        for g in strategy_games:
            hits = len(set(g) & actual_draw)
            if hits >= 11: results['estrategia'][hits] += 1
        
        for g in random_games:
            hits = len(set(g) & actual_draw)
            if hits >= 11: results['aleatorio'][hits] += 1
    
    # Exibir resultados
    print(f"\n📊 RESULTADOS DO BACKTEST:")
    print(f"   Concursos testados: {n_test}")
    print(f"   Jogos por concurso: {n_games}")
    print(f"   Total de jogos: {n_test * n_games:,}")
    
    print(f"\n{'Estratégia':<15} {'11pts':<10} {'12pts':<10} {'13pts':<10} {'14pts':<10} {'Total':<10}")
    print("-"*55)
    for label in ['estrategia','aleatorio']:
        name = "Estratégia" if label=='estrategia' else "Aleatório"
        total = sum(results[label].values())
        print(f"{name:<15} {results[label][11]:<10} {results[label][12]:<10} "
              f"{results[label][13]:<10} {results[label][14]:<10} {total:<10}")
    
    # Taxa de acerto
    total_jogos = n_test * n_games
    print(f"\n📊 TAXA DE ACERTO:")
    for label in ['estrategia','aleatorio']:
        name = "Estratégia" if label=='estrategia' else "Aleatório"
        total = sum(results[label].values())
        rate = total / total_jogos * 100
        print(f"   {name}: {total} prêmios ({rate:.3f}%)")
    
    # Diferença
    strat_total = sum(results['estrategia'].values())
    rand_total = sum(results['aleatorio'].values())
    diff = strat_total - rand_total
    diff_pct = (strat_total/rand_total - 1)*100 if rand_total>0 else 0
    print(f"\n📊 DIFERENÇA: {diff:+d} prêmios ({diff_pct:+.1f}%)")
    
    if diff > 0:
        print(f"   ✅ Estratégia SUPEROU o aleatório")
    else:
        print(f"   🟡 Estratégia NÃO superou o aleatório")
    
    return results


# ============================================================
# INTERFACE
# ============================================================

def display_results(games):
    print(f"\n{'='*60}")
    print(f"🏆 CARTEIRA OTIMIZADA")
    print(f"{'='*60}")
    
    all_d=[d for g in games for d in g]
    print(f"📊 Cobertura: {len(set(all_d))}/25 dezenas")
    
    sims=[]
    for i in range(min(30,len(games))):
        for j in range(i+1,min(30,len(games))):
            sims.append(len(set(games[i])&set(games[j])))
    if sims: print(f"📊 Similaridade média: {np.mean(sims):.1f}")
    
    # Distribuição de moldura
    molds=Counter(sum(1 for d in g if d in MOLDURA) for g in games)
    print(f"📊 Distribuição de moldura: {dict(sorted(molds.items()))}")
    
    for i,g in enumerate(games[:15],1):
        p=sum(1 for d in g if d%2==0); pr=sum(1 for d in g if d in PRIMES)
        m=sum(1 for d in g if d in MOLDURA); s=sum(g)
        print(f"   {i:2d}. {g}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{s}")


def main():
    print("="*60)
    print("🧬 OTIMIZADOR DE CARTEIRA v13 + BACKTEST")
    print("="*60)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")
    
    # Perguntar se quer backtest
    print(f"\n▶️  Opções:")
    print(f"   1. Gerar carteira otimizada")
    print(f"   2. Executar BACKTEST (validação científica)")
    choice = input(f"   Escolha [1]: ").strip() or "1"
    
    if choice == "2":
        n_test = int(input(f"   Concursos a testar [300]: ").strip() or "300")
        run_backtest(contests, n_test, n_games=50)
        return
    
    # Modo normal: gerar carteira
    markov = MarkovStateEncoder()
    markov.build_transition_matrix(contests)
    
    engine = StateFeatureEngine(contests)
    ranker = StructuralRanker(engine)
    ranker.train()
    
    last = contests[-1]['dezenas']
    prev = contests[-2]['dezenas'] if len(contests)>1 else last
    current_markov = markov.encode(
        sum(1 for x in last if x%2==0), sum(1 for x in last if x in PRIMES),
        sum(1 for x in last if x in MOLDURA), len(set(last)&set(prev)))
    
    current_state = engine.extract_state(len(contests)-1)
    features_vec = [float(current_state.get(n,0)) for n in engine.feature_names] if current_state else None
    
    momentum={}; anti_momentum={}
    if len(contests)>=100:
        recent=contests[-50:]; older=contests[-100:-50]
        fr=Counter(); [fr.update(c['dezenas']) for c in recent]
        fo=Counter(); [fo.update(c['dezenas']) for c in older]
        all_f=[fo.get(d,0)/len(older) for d in range(1,26)]
        mh,sh=np.mean(all_f),np.std(all_f)+1e-10
        for d in range(1,26):
            momentum[d]=(fr.get(d,0)/len(recent)-mh)/sh
            anti_momentum[d]=-momentum[d]*0.5
    
    opt = PortfolioOptimizerV13(
        constraints={}, momentum=momentum, anti_momentum=anti_momentum,
        markov=markov, current_state=current_markov, ranker=ranker,
        features_vec=features_vec, last_contest=last, temperature=0.7)
    
    print(f"\n🎲 GERANDO CARTEIRA...")
    games = opt.generate_many(n_games=200)
    print(f"   ✅ {len(games)} jogos")
    
    display_results(games)
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
