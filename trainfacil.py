#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE EXPLORAÇÃO COMBINATÓRIA INTELIGENTE
===============================================
Versão 14.0 - Regimes + Clusters + Monte Carlo + Validação Financeira

NOVO PARADIGMA:
✅ NÃO prevê dezenas - aprende REGIMES estruturais
✅ Clusterização histórica (KMeans + Gaussian Mixture)
✅ Monte Carlo massivo (50k candidatos)
✅ Score estrutural DESACOPLADO de cobertura
✅ Entropia de transição entre concursos
✅ Validação financeira (valor esperado)
✅ Aprende FALHAS (quais padrões tendem a falhar)
✅ Exploração combinatória superior a humanos
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
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================
# CONJUNTOS
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
QUADRANTES = {
    'Q1': {1,2,3,4,5}, 'Q2': {6,7,8,9,10},
    'Q3': {11,12,13,14,15}, 'Q4': {16,17,18,19,20},
    'Q5': {21,22,23,24,25}
}

# Pesos financeiros
PAYOFF = {11:1, 12:5, 13:50, 14:500, 15:5000}

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
# EXTRATOR DE REGIMES (substitui previsão de dezenas)
# ============================================================

class RegimeExtractor:
    """
    Extrai REGIMES estruturais (não dezenas)
    
    Um regime é um estado global do concurso:
    (pares, primos, moldura, soma, consecutivos, repetidas, amplitude)
    """
    
    def __init__(self, contests):
        self.contests = contests
        self.regime_vectors = []
        self.regime_labels = None
        self.cluster_model = None
        self.n_clusters = 5  # Tipos de concurso
        
        self._extract_all_regimes()
        self._cluster_regimes()
    
    def extract_regime(self, dezenas):
        """Extrai vetor de regime de um conjunto de dezenas"""
        d = sorted(dezenas)
        return np.array([
            sum(1 for x in d if x % 2 == 0),           # pares
            sum(1 for x in d if x in PRIMES),          # primos
            sum(1 for x in d if x in MOLDURA),         # moldura
            sum(1 for x in d if x in CENTRO),          # centro
            sum(d),                                     # soma
            sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),  # consecutivos
            max(d) - min(d),                            # amplitude
            len(set((x-1)//5 for x in d)),             # quadrantes
        ])
    
    def _extract_all_regimes(self):
        """Extrai vetores de regime de todos os concursos"""
        self.regime_vectors = []
        for c in self.contests:
            self.regime_vectors.append(self.extract_regime(c['dezenas']))
        self.regime_vectors = np.array(self.regime_vectors)
    
    def _cluster_regimes(self):
        """Clusteriza regimes históricos para descobrir TIPOS de concurso"""
        if len(self.regime_vectors) < 10 or not SKLEARN_AVAILABLE:
            return
        
        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(self.regime_vectors)
        
        # KMeans
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.regime_labels = self.cluster_model.fit_predict(X)
        
        # Nomear clusters baseado em características
        self.cluster_names = {}
        for i in range(self.n_clusters):
            mask = self.regime_labels == i
            cluster_data = self.regime_vectors[mask]
            avg = cluster_data.mean(axis=0)
            
            if avg[0] >= 8:  # pares altos
                name = "explosivo_pares"
            elif avg[1] >= 5:  # primos altos
                name = "denso_primos"
            elif avg[2] >= 10:  # moldura alta
                name = "periferico"
            elif avg[4] <= 170:  # soma baixa
                name = "compacto"
            else:
                name = "balanceado"
            
            self.cluster_names[i] = {
                'name': name,
                'size': mask.sum(),
                'avg_pares': avg[0],
                'avg_primos': avg[1],
                'avg_moldura': avg[2],
                'avg_soma': avg[3],
            }
        
        print(f"   ✅ {self.n_clusters} tipos de concurso descobertos:")
        for i, info in self.cluster_names.items():
            print(f"      {info['name']}: {info['size']} ocorrências "
                  f"(Pares:{info['avg_pares']:.1f} Soma:{info['avg_soma']:.0f})")
    
    def get_current_regime(self):
        """Retorna o regime do último concurso"""
        if len(self.regime_vectors) > 0:
            return self.regime_vectors[-1]
        return None
    
    def get_regime_transition_prob(self, from_regime, to_regime):
        """Probabilidade de transição entre regimes"""
        if self.regime_labels is None:
            return 1.0 / self.n_clusters
        
        # Encontrar clusters
        from_idx = self._find_nearest_cluster(from_regime)
        to_idx = self._find_nearest_cluster(to_regime)
        
        # Contar transições
        transitions = defaultdict(int)
        for i in range(len(self.regime_labels)-1):
            if self.regime_labels[i] == from_idx:
                transitions[self.regime_labels[i+1]] += 1
        
        total = sum(transitions.values())
        if total > 0:
            return transitions.get(to_idx, 0) / total
        return 1.0 / self.n_clusters
    
    def _find_nearest_cluster(self, regime_vector):
        """Encontra cluster mais próximo de um vetor de regime"""
        if self.cluster_model is None:
            return 0
        # Usar o mesmo scaler
        scaler = StandardScaler()
        scaler.fit(self.regime_vectors)
        X = scaler.transform([regime_vector])
        return self.cluster_model.predict(X)[0]
    
    def get_regime_diversity_score(self, regime_vector):
        """
        Score de diversidade: quão diferente é este regime do atual?
        Alta diversidade = exploração de regiões pouco visitadas
        """
        if len(self.regime_vectors) == 0:
            return 0.5
        
        # Distância para o regime atual
        current = self.get_current_regime()
        if current is not None:
            dist = np.linalg.norm(regime_vector - current)
            # Normalizar
            max_dist = np.linalg.norm(np.ones(8) * 20)
            return min(1.0, dist / max_dist)
        return 0.5


# ============================================================
# ENTROPIA DE TRANSIÇÃO
# ============================================================

class TransitionEntropy:
    """
    Mede entropia de transições entre concursos
    
    Alta entropia = aleatoriedade, baixa = persistência
    """
    
    def __init__(self, contests):
        self.contests = contests
        self.transition_entropies = []
        self._compute()
    
    def _compute(self):
        """Computa entropia de transição para cada par de concursos"""
        for i in range(1, len(self.contests)):
            prev = set(self.contests[i-1]['dezenas'])
            curr = set(self.contests[i]['dezenas'])
            
            # Features de transição
            repetidas = len(prev & curr)
            novas = 15 - repetidas
            pares_mudanca = abs(
                sum(1 for x in prev if x%2==0) - sum(1 for x in curr if x%2==0)
            )
            
            # Entropia da transição
            probs = np.array([repetidas/15, novas/15, pares_mudanca/8])
            probs = np.where(probs > 0, probs, 1e-10)
            ent = entropy(probs)
            self.transition_entropies.append(ent)
    
    def get_current_entropy(self, window=10):
        """Entropia de transição atual (média dos últimos N)"""
        if len(self.transition_entropies) >= window:
            return np.mean(self.transition_entropies[-window:])
        return 1.0
    
    def is_high_entropy_regime(self, threshold=0.8):
        """Estamos em regime de alta entropia (caótico)?"""
        return self.get_current_entropy() > threshold


# ============================================================
# APRENDIZ DE FALHAS
# ============================================================

class FailureLearner:
    """
    Aprende quais padrões estruturais tendem a FALHAR
    
    Mais útil que prever vencedores: evita armadilhas
    """
    
    def __init__(self):
        self.failure_patterns = Counter()
        self.success_patterns = Counter()
    
    def train(self, contests, n_backtest=200):
        """Treina com backtest simples"""
        for i in tqdm(range(max(50, len(contests)-n_backtest), len(contests)), desc="Aprendendo falhas"):
            train = contests[:i]
            actual = set(contests[i]['dezenas'])
            
            # Gerar jogos simples (baseline)
            for _ in range(20):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                hits = len(set(game) & actual)
                
                # Assinatura do jogo
                sig = (
                    sum(1 for x in game if x%2==0),
                    sum(1 for x in game if x in PRIMES),
                    sum(1 for x in game if x in MOLDURA),
                    sum(game) // 10,
                )
                
                if hits >= 11:
                    self.success_patterns[sig] += 1
                else:
                    self.failure_patterns[sig] += 1
    
    def failure_score(self, game):
        """Score de falha: quão provável é este jogo FALHAR?"""
        sig = (
            sum(1 for x in game if x%2==0),
            sum(1 for x in game if x in PRIMES),
            sum(1 for x in game if x in MOLDURA),
            sum(game) // 10,
        )
        
        fails = self.failure_patterns.get(sig, 0)
        successes = self.success_patterns.get(sig, 0)
        total = fails + successes
        
        if total == 0:
            return 0.5
        
        return fails / total  # 1.0 = sempre falha, 0.0 = nunca falha


# ============================================================
# OTIMIZADOR DE CARTEIRA (v14)
# ============================================================

class PortfolioOptimizerV14:
    """
    Otimizador focado em EXPLORAÇÃO COMBINATÓRIA
    
    Princípios:
    - NÃO prevê dezenas
    - Aprende REGIMES e clusters
    - Monte Carlo massivo
    - Score estrutural + cobertura desacoplados
    - Penaliza padrões de falha
    - Validação financeira
    """
    
    def __init__(self, regime_extractor, transition_entropy, failure_learner,
                 constraints=None, n_candidates=50000, temperature=0.7):
        self.regime = regime_extractor
        self.entropy = transition_entropy
        self.failures = failure_learner
        self.constraints = constraints or {}
        self.n_candidates = n_candidates
        self.temperature = temperature
        
        self.fixed = set(self.constraints.get('fixas', []))
        self.excluded = set(self.constraints.get('excluidas', []))
        
        # Controle
        self.dezena_usage = Counter()
        self.structure_sigs = Counter()
        self.generated_pool = []
    
    def _structural_score(self, game):
        """Score ESTRUTURAL puro (qualidade intrínseca)"""
        d = sorted(game)
        score = 0.0
        
        # Diversidade espacial
        score += len(set((x-1)//5 for x in d)) * 5
        
        # Balanceamento
        pares = sum(1 for x in d if x%2==0)
        score -= abs(pares - 7.5) * 1.5
        
        # Consecutivos moderados
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
        if cons <= 6: score += 3
        else: score -= (cons - 6) * 2
        
        # Entropia de transição (se alta, penalizar padrões rígidos)
        if self.entropy.is_high_entropy_regime():
            # Em regime caótico, evitar repetição excessiva
            sig = (pares, sum(d)//10)
            sig_count = self.structure_sigs.get(sig, 0)
            score -= sig_count * 2
        
        return score
    
    def _coverage_score(self, game):
        """Score de COBERTURA (diversidade na carteira)"""
        score = 0.0
        
        # Dezenas pouco usadas
        for x in game:
            score += 1.0 / (1.0 + self.dezena_usage.get(x, 0)) * 5
        
        # Penalidade de similaridade
        for existing in self.generated_pool[-30:]:
            common = len(set(game) & set(existing))
            if common > 11:
                score -= (common - 11) * 3
        
        # Diversidade de assinatura
        sig = (sum(1 for x in game if x%2==0), sum(game)//10)
        sig_count = self.structure_sigs.get(sig, 0)
        score -= sig_count * 3
        
        return score
    
    def _regime_score(self, game):
        """Score de ADERÊNCIA AO REGIME"""
        regime_vec = self.regime.extract_regime(game)
        current_regime = self.regime.get_current_regime()
        
        if current_regime is None:
            return 0.0
        
        # Distância ao regime atual (queremos proximidade)
        dist = np.linalg.norm(regime_vec - current_regime)
        max_dist = np.linalg.norm(np.ones(8) * 20)
        
        return max(0, 10 - (dist / max_dist) * 10)
    
    def _score_game(self, game):
        """Score combinado: 40% estrutural + 30% cobertura + 20% regime + 10% anti-falha"""
        structural = self._structural_score(game)
        coverage = self._coverage_score(game)
        regime = self._regime_score(game)
        
        # Anti-falha (inverter: baixa falha = alto score)
        failure_prob = self.failures.failure_score(game)
        anti_failure = (1.0 - failure_prob) * 10
        
        # Pesos
        score = structural * 0.4 + coverage * 0.3 + regime * 0.2 + anti_failure * 0.1
        
        return score
    
    def generate_candidates(self):
        """Monte Carlo massivo: gera 50k candidatos"""
        candidates = []
        seen = set()
        
        for _ in tqdm(range(self.n_candidates), desc="Monte Carlo"):
            game = list(self.fixed)
            available = [d for d in range(1,26) if d not in game and d not in self.excluded]
            
            # Completar com escolhas ponderadas
            while len(game) < 15 and available:
                # Pequena chance de escolha aleatória (exploração)
                if random.random() < 0.3:
                    game.append(random.choice(available))
                else:
                    # Escolha gulosa com temperatura
                    scores = []
                    for d in available[:20]:
                        test = game + [d]
                        scores.append(self._structural_score(test))
                    
                    if scores:
                        vals = np.array(scores)
                        vals = vals - np.max(vals)
                        probs = np.exp(vals / self.temperature)
                        probs = probs / probs.sum()
                        chosen_idx = np.random.choice(len(probs), p=probs)
                        game.append(available[chosen_idx])
                
                available = [d for d in available if d != game[-1]]
            
            game = sorted(game)[:15]
            key = tuple(game)
            
            if key not in seen and len(game) == 15:
                seen.add(key)
                candidates.append(game)
        
        return candidates
    
    def select_portfolio(self, candidates, n_select=50):
        """Seleciona carteira final dos candidatos"""
        # Pontuar todos
        scored = []
        for game in candidates:
            s = self._score_game(game)
            scored.append((s, game))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar com diversidade
        selected = []
        for score, game in scored:
            if len(selected) >= n_select:
                break
            
            # Verificar diversidade mínima
            too_similar = False
            for sg in selected:
                if len(set(game) & set(sg)) > 11:
                    too_similar = True
                    break
            
            if not too_similar:
                selected.append(game)
                # Atualizar contadores
                for d in game:
                    self.dezena_usage[d] += 1
                sig = (sum(1 for x in game if x%2==0), sum(game)//10)
                self.structure_sigs[sig] += 1
                self.generated_pool.append(game)
        
        return selected


# ============================================================
# BACKTEST COM VALIDAÇÃO FINANCEIRA
# ============================================================

def run_backtest_v14(contests, n_test=300, n_games=50):
    """Backtest com validação financeira"""
    print(f"\n{'='*60}")
    print(f"🔬 BACKTEST COM VALIDAÇÃO FINANCEIRA")
    print(f"{'='*60}")
    
    results = {
        'estrategia': {'premios': 0, 'payoff': 0, 'por_faixa': {11:0,12:0,13:0,14:0,15:0}},
        'aleatorio': {'premios': 0, 'payoff': 0, 'por_faixa': {11:0,12:0,13:0,14:0,15:0}}
    }
    
    start_idx = max(100, len(contests) - n_test)
    
    for i in tqdm(range(start_idx, len(contests)), desc="Backtest"):
        train = contests[:i]
        actual = set(contests[i]['dezenas'])
        
        # Treinar modelos
        regime_ext = RegimeExtractor(train)
        trans_ent = TransitionEntropy(train)
        failure_learner = FailureLearner()
        failure_learner.train(train, n_backtest=min(100, len(train)-50))
        
        # Gerar carteira
        opt = PortfolioOptimizerV14(
            regime_ext, trans_ent, failure_learner,
            n_candidates=5000, temperature=0.7)  # Reduzido para backtest
        
        candidates = opt.generate_candidates()
        strategy_games = opt.select_portfolio(candidates, n_games)
        
        # Baseline aleatória
        random_games = []
        for _ in range(n_games):
            random_games.append(sorted(np.random.choice(range(1,26), 15, replace=False)))
        
        # Avaliar
        for g in strategy_games:
            hits = len(set(g) & actual)
            if hits >= 11:
                results['estrategia']['premios'] += 1
                results['estrategia']['payoff'] += PAYOFF.get(hits, 0)
                results['estrategia']['por_faixa'][hits] += 1
        
        for g in random_games:
            hits = len(set(g) & actual)
            if hits >= 11:
                results['aleatorio']['premios'] += 1
                results['aleatorio']['payoff'] += PAYOFF.get(hits, 0)
                results['aleatorio']['por_faixa'][hits] += 1
    
    # Resultados
    total_jogos = n_test * n_games
    print(f"\n📊 RESULTADOS:")
    print(f"   Testes: {n_test} | Jogos/teste: {n_games} | Total: {total_jogos:,}")
    
    for label in ['estrategia', 'aleatorio']:
        name = "Estratégia" if label == 'estrategia' else "Aleatório"
        r = results[label]
        roi = (r['payoff'] - total_jogos * 3) / (total_jogos * 3) * 100  # Custo R$3
        
        print(f"\n   {name}:")
        print(f"      Prêmios: {r['premios']} ({r['premios']/total_jogos*100:.3f}%)")
        print(f"      Payoff: {r['payoff']} unidades")
        print(f"      ROI: {roi:+.2f}%")
        for hits in [11,12,13,14,15]:
            print(f"      {hits}pts: {r['por_faixa'][hits]}")
    
    strat_roi = (results['estrategia']['payoff'] - total_jogos*3) / (total_jogos*3) * 100
    rand_roi = (results['aleatorio']['payoff'] - total_jogos*3) / (total_jogos*3) * 100
    
    print(f"\n📊 DIFERENÇA DE ROI: {strat_roi - rand_roi:+.2f}%")
    
    if strat_roi > rand_roi:
        print(f"   ✅ Estratégia tem melhor retorno financeiro")
    else:
        print(f"   🟡 Estratégia NÃO supera aleatório em retorno")
    
    return results


# ============================================================
# INTERFACE
# ============================================================

def display_portfolio(games):
    print(f"\n{'='*60}")
    print(f"🏆 CARTEIRA OTIMIZADA")
    print(f"{'='*60}")
    
    all_d = [d for g in games for d in g]
    print(f"📊 Cobertura: {len(set(all_d))}/25")
    
    sims = []
    for i in range(min(30, len(games))):
        for j in range(i+1, min(30, len(games))):
            sims.append(len(set(games[i]) & set(games[j])))
    if sims: print(f"📊 Similaridade: {np.mean(sims):.1f}")
    
    for i, g in enumerate(games[:15], 1):
        p = sum(1 for d in g if d%2==0)
        pr = sum(1 for d in g if d in PRIMES)
        m = sum(1 for d in g if d in MOLDURA)
        print(f"   {i:2d}. {g}")
        print(f"       P:{p} Pr:{pr} M:{m} S:{sum(g)}")


def main():
    print("="*60)
    print("🧬 EXPLORADOR COMBINATÓRIO INTELIGENTE v14")
    print("="*60)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None: print("❌ Arquivo não encontrado"); return
    print(f"📂 {len(contests)} concursos")
    
    print(f"\n▶️  Opções:")
    print(f"   1. Gerar carteira otimizada")
    print(f"   2. Backtest com validação financeira")
    choice = input(f"   [1]: ").strip() or "1"
    
    if choice == "2":
        n = int(input(f"   Concursos [300]: ").strip() or "300")
        run_backtest_v14(contests, n, n_games=50)
        return
    
    # Modo normal
    regime_ext = RegimeExtractor(contests)
    trans_ent = TransitionEntropy(contests)
    failure_learner = FailureLearner()
    failure_learner.train(contests)
    
    opt = PortfolioOptimizerV14(
        regime_ext, trans_ent, failure_learner,
        n_candidates=30000, temperature=0.7)
    
    print(f"\n🎲 Monte Carlo: gerando 30k candidatos...")
    candidates = opt.generate_candidates()
    print(f"   ✅ {len(candidates)} candidatos")
    
    print(f"\n📊 Selecionando carteira...")
    portfolio = opt.select_portfolio(candidates, n_select=50)
    
    display_portfolio(portfolio)
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
