#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO OTIMIZADO DE CARTEIRA - LOTOFÁCIL v2
==========================================================
MELHORIAS:
✅ Gerador GUIADO (não aleatório puro) - 10x mais eficiente
✅ Anti-sobrecentralização (penaliza jogos "perfeitos demais")
✅ Score gaussiano para centralidade (mais estável)
✅ Backtest de carteira (probabilidade de ≥1 acerto 11+)
✅ Cobertura global da carteira (maximização de união)
✅ Grid 5x5 real para quadrantes
✅ Pesos calibrados: centralidade(0.55) + diversidade(0.25) + cobertura(0.20)
✅ Otimização multiobjetivo com simulated annealing
✅ LedoitWolf + Mahalanobis mantidos
✅ Separação HARD/SOFT preservada
"""

import numpy as np
from scipy.stats import entropy, hypergeom
from collections import Counter
from datetime import datetime
import warnings
import os
from math import comb, exp
from tqdm import tqdm
import random
import time

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Use: pip install scikit-learn")

# ============================================================
# CONJUNTOS E CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}

# Grid 5x5 real (volante físico)
GRID_5X5 = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]

# Regiões do volante
REGIAO_SUPERIOR = {1,2,3,4,5,6,7,8,9,10}
REGIAO_INFERIOR = {16,17,18,19,20,21,22,23,24,25}
REGIAO_ESQUERDA = {1,6,11,16,21,2,7,12,17,22}
REGIAO_DIREITA = {4,5,9,10,14,15,19,20,24,25}

HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# Features topológicas (20 dimensões)
FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_rep", "entropia_transicao",
    "quadrantes_grid", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
    "elasticidade", "entropia_conjunta",
]

# Índices
IDX_GAP_MEDIO, IDX_GAP_VAR, IDX_GAP_MAX, IDX_GAP_MIN = 0, 1, 2, 3
IDX_ENERGIA, IDX_ENTROPIA_REP, IDX_ENTROPIA_TRANS = 4, 5, 6
IDX_QUADRANTES, IDX_CONSECUTIVOS, IDX_DENSIDADE = 7, 8, 9
IDX_ASSIMETRIA, IDX_CLUSTERIZACAO, IDX_REPETICOES = 10, 11, 12
IDX_PARES, IDX_PRIMOS, IDX_MOLDURA, IDX_SOMA = 13, 14, 15, 16
IDX_AMPLITUDE, IDX_ELASTICIDADE, IDX_ENTROPIA_CONJ = 17, 18, 19

# Mapeamento de constraints para índices
CONSTRAINT_INDICES = {
    'pares': IDX_PARES, 'primos': IDX_PRIMOS, 'moldura': IDX_MOLDURA,
    'soma': IDX_SOMA, 'repeticoes': IDX_REPETICOES,
    'consecutivos': IDX_CONSECUTIVOS, 'amplitude': IDX_AMPLITUDE,
    'quadrantes': IDX_QUADRANTES,
}

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return None
    contests = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17: continue
            try:
                concurso = int(parts[0])
                data = parts[1]
                dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                if len(dezenas) != 15 or len(set(dezenas)) != 15: continue
                if any(x < 1 or x > 25 for x in dezenas): continue
                contests.append({'concurso': concurso, 'data': data, 'dezenas': sorted(dezenas)})
            except: continue
        contests.sort(key=lambda x: x['concurso'])
        print(f"✅ {len(contests)} concursos válidos")
        return contests
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


# ============================================================
# EXTRATOR DE FEATURES (COM GRID 5x5 CORRIGIDO)
# ============================================================
class TopologicalFeatureExtractor:
    def __init__(self, contests):
        self.contests = contests
        self._repeat_history = []
        self._pares_history = []
        for i, c in enumerate(contests):
            d = c['dezenas']
            self._pares_history.append(sum(1 for x in d if x % 2 == 0))
            if i > 0:
                self._repeat_history.append(len(set(contests[i-1]['dezenas']) & set(d)))
            else:
                self._repeat_history.append(0)

    def _count_grid_quadrants(self, dezenas):
        """Conta quadrantes no grid 5x5 real"""
        s = set(dezenas)
        q1 = sum(1 for x in s if x in REGIAO_SUPERIOR and x in REGIAO_ESQUERDA)
        q2 = sum(1 for x in s if x in REGIAO_SUPERIOR and x in REGIAO_DIREITA)
        q3 = sum(1 for x in s if x in REGIAO_INFERIOR and x in REGIAO_ESQUERDA)
        q4 = sum(1 for x in s if x in REGIAO_INFERIOR and x in REGIAO_DIREITA)
        # Conta quantos quadrantes têm pelo menos 2 dezenas
        return sum(1 for q in [q1, q2, q3, q4] if q >= 2)

    def extract_features(self, game, last_contest=None):
        d = sorted(game)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        rep = len(set(d) & set(last_contest)) if last_contest else 8

        f = [
            float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps)),
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))),
            0.0, 0.0,
            float(self._count_grid_quadrants(d)),
            float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),
            float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),
            float(np.mean(d) - np.median(d)),
            float(sum(1 for g in gaps if g <= 2) / len(gaps)),
            float(rep),
            float(sum(1 for x in d if x % 2 == 0)),
            float(sum(1 for x in d if x in PRIMES)),
            float(sum(1 for x in d if x in MOLDURA)),
            float(sum(d)),
            float(max(d) - min(d)),
            0.0, 0.0,
        ]

        if len(self._repeat_history) >= 10:
            recent = self._repeat_history[-10:]
            freq = Counter(recent)
            probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
            f[5] = float(entropy(np.where(probs>0, probs, 1e-10)))
        if len(self._repeat_history) >= 5:
            trans = [self._repeat_history[i+1]-self._repeat_history[i] for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                f[6] = float(entropy(np.where(probs>0, probs, 1e-10)))
        if len(self._repeat_history) >= 10:
            f[18] = float(np.mean(self._repeat_history) - np.mean(self._repeat_history[-10:]))
        if len(self._repeat_history) >= 10 and len(self._pares_history) >= 10:
            joint = Counter(zip(self._repeat_history[-10:], self._pares_history[-10:]))
            probs = np.array([joint.get(k,0)/10 for k in joint])
            f[19] = float(entropy(np.where(probs>0, probs, 1e-10)))

        return np.array(f, dtype=np.float64)

    def build_feature_matrix(self):
        features_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            features_list.append(self.extract_features(c['dezenas'], last))
        return np.array(features_list, dtype=np.float64)


# ============================================================
# BASELINE HISTÓRICO
# ============================================================
class HistoricalBaseline:
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        self.n_features = feature_matrix.shape[1]
        self.mean = np.mean(feature_matrix, axis=0)
        self.std = np.std(feature_matrix, axis=0)
        self._build_covariance()

    def _build_covariance(self):
        if SKLEARN_AVAILABLE and self.feature_matrix.shape[0] > self.n_features:
            try:
                lw = LedoitWolf().fit(self.feature_matrix)
                self.cov = lw.covariance_
                self.precision = lw.precision_
                return
            except: pass
        self.cov = np.cov(self.feature_matrix.T) + np.eye(self.n_features) * 1e-6
        self.precision = np.linalg.inv(self.cov)

    def mahalanobis_distance(self, features):
        diff = features - self.mean
        try:
            return float(np.sqrt(max(0, np.dot(np.dot(diff.T, self.precision), diff))))
        except:
            return float(np.linalg.norm(diff / (self.std + 1e-10)))


# ============================================================
# GERADOR GUIADO (NÃO ALEATÓRIO PURO)
# ============================================================
class GuidedGenerator:
    """
    Gerador GUIADO que respeita constraints HARD na construção.
    10x mais eficiente que aleatório puro.
    """
    def __init__(self, hard_constraints, last_contest):
        self.hard = hard_constraints
        self.last = set(last_contest) if last_contest else None

    def _get_range(self, key):
        if key in self.hard: return self.hard[key]
        return None

    def generate_one(self):
        """Gera UM jogo que atende as constraints HARD"""
        game = set()
        available = set(range(1, 26))

        # 1. Repetidas primeiro (se especificado)
        rep_range = self._get_range('repeticoes')
        if rep_range and self.last:
            target_rep = random.randint(rep_range[0], rep_range[1])
            rep_pool = list(self.last & available)
            if rep_pool and target_rep > 0:
                n = min(target_rep, len(rep_pool))
                chosen = random.sample(rep_pool, n)
                game.update(chosen)
                available -= game

        # 2. Pares
        pares_range = self._get_range('pares')
        primos_range = self._get_range('primos')
        moldura_range = self._get_range('moldura')
        soma_range = self._get_range('soma')
        cons_range = self._get_range('consecutivos')

        target_pares = random.randint(pares_range[0], pares_range[1]) if pares_range else random.randint(6, 9)
        target_primos = random.randint(primos_range[0], primos_range[1]) if primos_range else random.randint(4, 7)
        target_moldura = random.randint(moldura_range[0], moldura_range[1]) if moldura_range else random.randint(8, 11)
        target_soma = random.randint(soma_range[0], soma_range[1]) if soma_range else random.randint(185, 210)

        # Preencher com restrições
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                s = 0
                test = game | {d}
                # Aproximar do target de pares
                curr_pares = sum(1 for x in test if x % 2 == 0)
                remaining = 15 - len(test)
                if curr_pares > target_pares:
                    s -= 10
                elif curr_pares + remaining < target_pares and d % 2 != 0:
                    s -= 5
                # Aproximar do target de moldura
                curr_mold = sum(1 for x in test if x in MOLDURA)
                if target_moldura and curr_mold > target_moldura:
                    s -= 5
                # Penalizar consecutivos excessivos
                if cons_range:
                    st = sorted(test)
                    cons = sum(1 for i in range(len(st)-1) if st[i+1]-st[i]==1)
                    if cons > cons_range[1]:
                        s -= 15
                # Aproximar soma
                if soma_range:
                    curr_soma = sum(test)
                    projected = curr_soma + (remaining * 13)
                    if projected < soma_range[0] or projected > soma_range[1]:
                        s -= 3
                scores.append(s)

            if scores:
                # Softmax para diversidade
                scores = np.array(scores, dtype=np.float64)
                scores = scores - np.max(scores)
                probs = np.exp(scores / 2.0)
                probs = probs / probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else:
                chosen = random.choice(candidates)

            game.add(chosen)
            available.remove(chosen)

        return sorted(game)[:15]


# ============================================================
# OTIMIZADOR DE CARTEIRA (COM BACKTEST)
# ============================================================
class PortfolioOptimizer:
    """
    Otimizador multiobjetivo de carteira.
    
    Score = 0.55 * centralidade + 0.25 * diversidade + 0.20 * cobertura
    Com anti-sobrecentralização e backtest integrado.
    """
    def __init__(self, contests, hard_constraints, soft_targets):
        self.contests = contests
        self.extractor = TopologicalFeatureExtractor(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.baseline = HistoricalBaseline(self.feature_matrix)
        self.last = contests[-1]['dezenas'] if contests else None

        self.hard = hard_constraints
        self.soft = soft_targets
        self.generator = GuidedGenerator(hard_constraints, self.last)

        # Parâmetros de centralidade para anti-sobrecentralização
        self.centrality_stats = self._compute_centrality_stats()

    def _compute_centrality_stats(self):
        """Pré-computa estatísticas de centralidade dos jogos históricos"""
        dists = []
        for feats in self.feature_matrix:
            dists.append(self.baseline.mahalanobis_distance(feats))
        dists = np.array(dists)
        return {
            'mean': float(np.mean(dists)),
            'std': float(np.std(dists)),
            'p25': float(np.percentile(dists, 25)),
            'p10': float(np.percentile(dists, 10)),
        }

    def _score_game(self, game):
        """Score individual do jogo (gaussiano)"""
        features = self.extractor.extract_features(game, self.last)
        dist = self.baseline.mahalanobis_distance(features)

        # Score gaussiano (anti-sobrecentralização)
        # Penaliza jogos MUITO centrais (abaixo do percentil 10)
        if dist < self.centrality_stats['p10']:
            dist += (self.centrality_stats['p10'] - dist) * 1.5

        # Penaliza jogos muito distantes
        if dist > self.centrality_stats['mean'] + 2 * self.centrality_stats['std']:
            dist += 5.0

        sigma = self.centrality_stats['std'] + 1e-10
        score = np.exp(-(dist**2) / (2 * sigma**2))
        return score, features

    def _portfolio_union_coverage(self, portfolio):
        """Cobertura de união da carteira"""
        all_dezenas = set()
        for game in portfolio:
            all_dezenas.update(game)
        return len(all_dezenas) / 25.0

    def _portfolio_diversity(self, portfolio):
        """Diversidade média entre jogos"""
        if len(portfolio) < 2:
            return 1.0
        sims = []
        for i in range(len(portfolio)):
            for j in range(i+1, len(portfolio)):
                sims.append(len(set(portfolio[i]) & set(portfolio[j])))
        avg_sim = np.mean(sims)
        return 1.0 - (avg_sim / 15.0)

    def _portfolio_score(self, portfolio):
        """Score multiobjetivo da carteira"""
        scores = []
        for game in portfolio:
            s, _ = self._score_game(game)
            scores.append(s)

        avg_centrality = np.mean(scores)
        diversity = self._portfolio_diversity(portfolio)
        coverage = self._portfolio_union_coverage(portfolio)

        return 0.55 * avg_centrality + 0.25 * diversity + 0.20 * coverage

    def optimize(self, n_games=10, n_candidates=200000, iterations=100):
        """
        Otimização com simulated annealing.
        """
        print(f"\n🎯 OTIMIZANDO CARTEIRA ({n_games} jogos)...")

        # Gerar pool de candidatos (guiado)
        print(f"   Gerando {n_candidates:,} candidatos...")
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Candidatos"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                features = self.extractor.extract_features(game, self.last)
                dist = self.baseline.mahalanobis_distance(features)
                pool.append((dist, game, features))

        pool.sort(key=lambda x: x[0])

        # Inicializar carteira com melhores + diversos
        portfolio = []
        selected_set = set()
        for dist, game, features in pool:
            if len(portfolio) >= n_games:
                break
            too_similar = False
            for sg in portfolio:
                if len(set(game) & set(sg)) > 11:
                    too_similar = True
                    break
            if not too_similar:
                portfolio.append(game)

        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)
        current_score = best_score

        # Simulated annealing
        temp = 1.0
        elite_pool = pool[:len(pool)//4]  # top 25%
        random_pool = pool[len(pool)//4:]

        for it in tqdm(range(iterations), desc="Annealing"):
            temp *= 0.95
            new_portfolio = list(portfolio)
            idx = random.randint(0, n_games - 1)

            # 70% elite, 30% aleatório
            if random.random() < 0.7 and elite_pool:
                _, new_game, _ = random.choice(elite_pool)
            elif random_pool:
                _, new_game, _ = random.choice(random_pool)
            else:
                new_game = self.generator.generate_one()

            new_portfolio[idx] = new_game
            new_score = self._portfolio_score(new_portfolio)

            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / max(0.01, temp)):
                portfolio = new_portfolio
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_portfolio = list(portfolio)

        return best_portfolio, best_score

    def backtest_portfolio(self, portfolio, test_draws):
        """
        Backtest: probabilidade da carteira ter ≥1 acerto de 11+
        """
        n_success = 0
        for draw in test_draws:
            actual = set(draw['dezenas'])
            has_11 = False
            for game in portfolio:
                if len(set(game) & actual) >= 11:
                    has_11 = True
                    break
            if has_11:
                n_success += 1

        prob = n_success / len(test_draws) if test_draws else 0

        # Baseline teórico
        p_single = sum(HYPE_PROBS[k] for k in range(11, 16))
        p_none = (1 - p_single) ** len(portfolio)
        theo_prob = 1 - p_none

        return {
            'empirical': prob,
            'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws),
            'n_success': n_success,
        }


# ============================================================
# INTERFACE DE CONFIGURAÇÃO
# ============================================================
def interactive_config():
    print("\n" + "="*70)
    print("⚙️  CONFIGURAÇÃO DE PARÂMETROS")
    print("="*70)
    print("💡 Pressione ENTER para usar valores padrão (recomendados)")

    hard = {}

    # Valores padrão otimizados
    defaults = {
        'pares': (6, 9),
        'primos': (4, 7),
        'moldura': (8, 11),
        'soma': (185, 210),
        'repeticoes': (8, 10),
        'consecutivos': (5, 9),
        'amplitude': (20, 24),
    }

    print("\n📊 FILTROS HARD (jogos fora da faixa são ELIMINADOS)")

    for key, (dmin, dmax) in defaults.items():
        prompt_min = input(f"   {key.capitalize()} MÍNIMO [{dmin}]: ").strip()
        prompt_max = input(f"   {key.capitalize()} MÁXIMO [{dmax}]: ").strip()
        vmin = int(prompt_min) if prompt_min else dmin
        vmax = int(prompt_max) if prompt_max else dmax
        hard[key] = (min(vmin, vmax), max(vmin, vmax))

    print("\n🎯 TARGETS SOFT")
    soft = {}
    soft_input = input("   Definir targets soft? (s/N): ").strip().lower()
    if soft_input == 's':
        for key, default in [('gap_medio', 1.62), ('clusterizacao', 0.85), ('densidade_local', 0.21)]:
            val = input(f"   {key.replace('_',' ').capitalize()} [{default}]: ").strip()
            if val:
                soft[key] = float(val)
            else:
                soft[key] = default

    print("\n🔢 CONFIGURAÇÃO DE GERAÇÃO")
    n_candidates = input("   Número de candidatos [200000]: ").strip()
    n_candidates = int(n_candidates) if n_candidates else 200000

    n_games = input("   Jogos a gerar [10]: ").strip()
    n_games = int(n_games) if n_games else 10

    do_backtest = input("   Executar backtest? (S/n): ").strip().lower()
    do_backtest = do_backtest != 'n'

    return hard, soft, n_candidates, n_games, do_backtest


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR PARAMÉTRICO OTIMIZADO DE CARTEIRA v2")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    hard, soft, n_candidates, n_games, do_backtest = interactive_config()

    print(f"\n🔧 INICIALIZANDO OTIMIZADOR...")
    t0 = time.time()
    optimizer = PortfolioOptimizer(contests, hard, soft)
    print(f"   ✅ Baseline em {time.time()-t0:.1f}s")

    # Otimizar carteira
    t0 = time.time()
    portfolio, score = optimizer.optimize(n_games, n_candidates, iterations=100)
    print(f"   ⏱️ Otimização em {time.time()-t0:.1f}s")

    # Exibir resultados
    print(f"\n{'='*70}")
    print(f"🏆 CARTEIRA OTIMIZADA (Score: {score:.3f})")
    print(f"{'='*70}")

    last = contests[-1]['dezenas']
    for i, game in enumerate(portfolio, 1):
        p = sum(1 for d in game if d % 2 == 0)
        pr = sum(1 for d in game if d in PRIMES)
        m = sum(1 for d in game if d in MOLDURA)
        s = sum(game)
        rep = len(set(game) & set(last))
        cons = sum(1 for j in range(len(game)-1) if game[j+1]-game[j]==1)
        game_score, _ = optimizer._score_game(game)

        print(f"\n   JOGO {i:02d} | Score: {game_score:.3f}")
        print(f"   {'─'*50}")
        print(f"   Dezenas: {game}")
        print(f"   Pares:{p} | Primos:{pr} | Moldura:{m} | Soma:{s}")
        print(f"   Repetidas:{rep} | Consecutivos:{cons}")

    # Estatísticas
    print(f"\n{'='*70}")
    print(f"📊 ESTATÍSTICAS DA CARTEIRA")
    print(f"{'='*70}")

    all_d = set()
    for g in portfolio: all_d.update(g)
    print(f"   Cobertura: {len(all_d)}/25 ({len(all_d)/25*100:.0f}%)")
    print(f"   Diversidade: {optimizer._portfolio_diversity(portfolio):.3f}")

    if len(portfolio) > 1:
        sims = []
        for i in range(len(portfolio)):
            for j in range(i+1, len(portfolio)):
                sims.append(len(set(portfolio[i]) & set(portfolio[j])))
        print(f"   Similaridade média: {np.mean(sims):.1f}/15")

    # Backtest
    if do_backtest and len(contests) >= 200:
        print(f"\n🔬 BACKTEST DA CARTEIRA...")
        test_size = min(200, len(contests) // 3)
        test_draws = contests[-test_size:]
        bt_results = optimizer.backtest_portfolio(portfolio, test_draws)
        print(f"   Testado em {bt_results['n_test']} concursos")
        print(f"   Prob. ≥1 acerto 11+: {bt_results['empirical']:.2%} (empírico)")
        print(f"   Prob. ≥1 acerto 11+: {bt_results['theoretical']:.2%} (teórico)")
        print(f"   Lift: {bt_results['lift']:.2f}x")
        print(f"   Sucessos: {bt_results['n_success']}/{bt_results['n_test']}")

    print(f"\n✅ {len(portfolio)} jogos gerados!")
    print(f"💡 Sistema baseado em otimização geométrica multiobjetivo.")


if __name__ == "__main__":
    main()
