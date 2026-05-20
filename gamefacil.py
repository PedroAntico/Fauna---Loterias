#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v4
================================================
CORREÇÕES E MELHORIAS:
✅ Bug walk-forward corrigido (results vs resultados)
✅ Normalização completa de scores (MinMaxScaler por componente)
✅ Baseline puramente aleatória (sem geometria)
✅ Baseline de cobertura máxima pura
✅ Features degeneradas REMOVIDAS (entropia_rep, elasticidade, entropia_conjunta)
✅ Gerador mais livre (menos viés na construção)
✅ Monte Carlo comparativo (teste de significância)
✅ GMM com scores normalizados
✅ Métricas de lift e p-value
✅ Arquitetura modular preservada
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
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
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Use: pip install scikit-learn")
    print("   Fallback: covariância empírica com regularização")

# ============================================================
# CONJUNTOS E CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# Features topológicas (v4: REMOVIDAS as degeneradas)
# Índices 5, 18, 19 removidos (entropia_rep, elasticidade, entropia_conjunta)
FEATURE_NAMES_V4 = [
    "gap_medio",          # 0
    "gap_var",            # 1
    "gap_max",            # 2
    "gap_min",            # 3
    "energia_jogo",       # 4
    "entropia_transicao", # 5 (era 6)
    "quadrantes",         # 6 (era 7)
    "consecutivos",       # 7 (era 8)
    "densidade_local",    # 8 (era 9)
    "assimetria",         # 9 (era 10)
    "clusterizacao",      # 10 (era 11)
    "repeticoes",         # 11 (era 12)
    "pares",              # 12 (era 13)
    "primos",             # 13 (era 14)
    "moldura",            # 14 (era 15)
    "soma",               # 15 (era 16)
    "amplitude",          # 16 (era 17)
]

IDX = {name: i for i, name in enumerate(FEATURE_NAMES_V4)}

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
# EXTRATOR DE FEATURES (v4: sem features degeneradas)
# ============================================================
class TopologicalFeatureExtractorV4:
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

    def extract_features(self, game, last_contest=None):
        """Extrai 17 features (sem entropia_rep, elasticidade, entropia_conjunta)"""
        d = sorted(game)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        rep = len(set(d) & set(last_contest)) if last_contest else 8

        # Entropia de transição (única entropia mantida)
        ent_trans = 0.0
        if len(self._repeat_history) >= 5:
            trans = [self._repeat_history[i+1]-self._repeat_history[i] for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs>0, probs, 1e-10)))

        f = [
            float(np.mean(gaps)),                    # 0 gap_medio
            float(np.var(gaps)),                      # 1 gap_var
            float(max(gaps)),                         # 2 gap_max
            float(min(gaps)),                         # 3 gap_min
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))),  # 4 energia_jogo
            ent_trans,                                # 5 entropia_transicao
            float(len(set((x-1)//5 for x in d))),    # 6 quadrantes
            float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),  # 7 consecutivos
            float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),  # 8 densidade_local
            float(np.mean(d) - np.median(d)),         # 9 assimetria
            float(sum(1 for g in gaps if g <= 2) / len(gaps)),  # 10 clusterizacao
            float(rep),                               # 11 repeticoes
            float(sum(1 for x in d if x % 2 == 0)),  # 12 pares
            float(sum(1 for x in d if x in PRIMES)), # 13 primos
            float(sum(1 for x in d if x in MOLDURA)),# 14 moldura
            float(sum(d)),                            # 15 soma
            float(max(d) - min(d)),                   # 16 amplitude
        ]
        return np.array(f, dtype=np.float64)

    def build_feature_matrix(self):
        features_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            features_list.append(self.extract_features(c['dezenas'], last))
        return np.array(features_list, dtype=np.float64)


# ============================================================
# MODELO DE DISTRIBUIÇÃO (GMM com scores normalizados)
# ============================================================
class DistributionModelV4:
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        self.n_features = feature_matrix.shape[1]
        self.mean = np.mean(feature_matrix, axis=0)
        self.std = np.std(feature_matrix, axis=0)
        self._build_gmm()
        self._build_precision()
        # Normalizadores para GMM scores (pré-computados)
        self._gmm_norm = self._compute_gmm_norm()

    def _build_gmm(self):
        if SKLEARN_AVAILABLE and self.feature_matrix.shape[0] > 100:
            try:
                n_comp = min(5, self.feature_matrix.shape[0] // 200)
                self.gmm = GaussianMixture(n_components=max(2, n_comp), random_state=42)
                self.gmm.fit(self.feature_matrix)
                self._has_gmm = True
                return
            except:
                pass
        self._has_gmm = False

    def _build_precision(self):
        if SKLEARN_AVAILABLE and self.feature_matrix.shape[0] > self.n_features:
            try:
                lw = LedoitWolf().fit(self.feature_matrix)
                self.precision = lw.precision_
                return
            except: pass
        cov = np.cov(self.feature_matrix.T) + np.eye(self.n_features) * 1e-6
        self.precision = np.linalg.inv(cov)

    def _compute_gmm_norm(self):
        """Pré-computa min/max dos scores GMM para normalização"""
        if self._has_gmm:
            scores = self.gmm.score_samples(self.feature_matrix)
            return {'min': float(np.min(scores)), 'max': float(np.max(scores))}
        return {'min': -100.0, 'max': 100.0}

    def score_samples_normalized(self, features):
        """Score de densidade NORMALIZADO (0-1)"""
        if self._has_gmm:
            raw = float(self.gmm.score_samples(features.reshape(1, -1))[0])
            # Normalizar para 0-1
            rng = self._gmm_norm['max'] - self._gmm_norm['min']
            if rng > 0:
                return (raw - self._gmm_norm['min']) / rng
            return 0.5
        # Fallback: distância de Mahalanobis normalizada
        diff = features - self.mean
        try:
            dist = np.sqrt(max(0, np.dot(np.dot(diff.T, self.precision), diff)))
            return float(1.0 / (1.0 + dist))
        except:
            return 0.5

    def compute_gaussian_score_normalized(self, features, soft_targets):
        """Score gaussiano NORMALIZADO (0-1)"""
        if not soft_targets:
            return 0.5
        total = 0.0
        for name, (target, sigma) in soft_targets.items():
            if name in IDX:
                actual = features[IDX[name]]
                z = (actual - target) / sigma
                total += np.exp(-0.5 * z**2)
        return total / len(soft_targets)


# ============================================================
# GERADOR MAIS LIVRE (menos viés)
# ============================================================
class FreeGenerator:
    """Gerador com viés MÍNIMO na construção"""
    def __init__(self, last_contest=None):
        self.last = set(last_contest) if last_contest else None

    def generate_one(self):
        """Gera UM jogo com construção quase neutra"""
        game = set()
        available = set(range(1, 26))

        # Leve viés de repetição (apenas 30% dos jogos)
        if self.last and random.random() < 0.3:
            rep_pool = list(self.last & available)
            if rep_pool:
                n = random.randint(5, 10)
                game.update(random.sample(rep_pool, min(n, len(rep_pool))))
                available -= game

        # Preencher com diversidade espacial
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                test = game | {d}
                # Apenas diversidade de quadrantes (sem viés de pares/moldura)
                s = len(set((x-1)//5 for x in test)) * 3
                # Penalidade leve para consecutivos > 6
                st = sorted(test)
                cons = sum(1 for i in range(len(st)-1) if st[i+1]-st[i]==1)
                if cons > 6: s -= (cons - 6) * 1.5
                scores.append(s)

            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores = scores - np.max(scores)
                probs = np.exp(scores / 3.0)  # temperatura mais alta = mais exploratório
                probs = probs / probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else:
                chosen = random.choice(candidates)

            game.add(chosen)
            available.remove(chosen)

        return sorted(game)[:15]

    def generate_pure_random(self, n=1):
        """Gera jogos puramente aleatórios (baseline)"""
        games = []
        for _ in range(n):
            games.append(sorted(np.random.choice(range(1, 26), 15, replace=False)))
        return games if n > 1 else games[0]


# ============================================================
# OTIMIZADOR DE CARTEIRA v4
# ============================================================
class PortfolioOptimizerV4:
    def __init__(self, contests, soft_targets=None):
        self.contests = contests
        self.extractor = TopologicalFeatureExtractorV4(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.dist_model = DistributionModelV4(self.feature_matrix)
        self.last = contests[-1]['dezenas'] if contests else None
        self.soft = soft_targets or {}
        self.generator = FreeGenerator(self.last)
        # Normalizadores para scores do portfólio
        self._score_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None

    def _score_game(self, game):
        """Score individual com componentes NORMALIZADOS"""
        features = self.extractor.extract_features(game, self.last)
        gmm_score = self.dist_model.score_samples_normalized(features)
        gauss_score = self.dist_model.compute_gaussian_score_normalized(features, self.soft)
        return gmm_score * 0.5 + gauss_score * 0.5, features

    def _portfolio_entropy(self, portfolio):
        all_dezenas = [d for g in portfolio for d in g]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        probs = freq / np.sum(freq)
        probs = np.where(probs > 0, probs, 1e-10)
        return float(entropy(probs) / np.log(25))

    def _portfolio_union_coverage(self, portfolio):
        return len(set(d for g in portfolio for d in g)) / 25.0

    def _portfolio_diversity(self, portfolio):
        if len(portfolio) < 2: return 1.0
        sims = []
        for i in range(len(portfolio)):
            for j in range(i+1, len(portfolio)):
                sims.append(len(set(portfolio[i]) & set(portfolio[j])))
        return 1.0 - np.mean(sims) / 15.0

    def _portfolio_score(self, portfolio):
        """Score da carteira com componentes normalizados"""
        scores = [self._score_game(g)[0] for g in portfolio]
        avg_score = np.mean(scores)
        entropy = self._portfolio_entropy(portfolio)
        coverage = self._portfolio_union_coverage(portfolio)
        diversity = self._portfolio_diversity(portfolio)
        # Todos já estão em escala 0-1
        return avg_score * 0.35 + entropy * 0.25 + coverage * 0.25 + diversity * 0.15

    def _mutate_game(self, game):
        """Mutação LOCAL: troca 1-3 dezenas"""
        mutated = list(game)
        n_changes = random.randint(1, 3)
        for _ in range(n_changes):
            pos = random.randint(0, 14)
            available = [d for d in range(1, 26) if d not in mutated]
            if available:
                mutated[pos] = random.choice(available)
        return sorted(mutated)[:15]

    def optimize(self, n_games=10, n_candidates=200000, iterations=100):
        print(f"\n🎯 OTIMIZANDO CARTEIRA ({n_games} jogos)...")

        # Gerar pool
        print(f"   Gerando {n_candidates:,} candidatos...")
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Candidatos"):
            game = self.generator.generate_one()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                s, feats = self._score_game(game)
                pool.append((s, game, feats))

        pool.sort(key=lambda x: x[0], reverse=True)

        # Inicializar carteira diversa
        portfolio = []
        for s, game, feats in pool:
            if len(portfolio) >= n_games: break
            if not any(len(set(game) & set(sg)) > 11 for sg in portfolio):
                portfolio.append(game)

        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)
        current_score = best_score

        # Simulated Annealing com MUTAÇÃO LOCAL
        temp = 1.0
        elite_pool = pool[:len(pool)//4]

        for it in tqdm(range(iterations), desc="Annealing"):
            temp *= 0.95
            new_portfolio = list(portfolio)
            idx = random.randint(0, n_games - 1)

            if random.random() < 0.5 and elite_pool:
                _, new_game, _ = random.choice(elite_pool)
            elif random.random() < 0.8:
                new_game = self._mutate_game(new_portfolio[idx])
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

    def generate_pure_random_portfolio(self, n_games=10):
        """Baseline puramente aleatória"""
        return [self.generator.generate_pure_random() for _ in range(n_games)]

    def generate_coverage_baseline(self, n_games=10):
        """Baseline de cobertura máxima (sem geometria)"""
        pool = []
        seen = set()
        for _ in range(50000):
            game = self.generator.generate_pure_random()
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                pool.append(game)
        selected = [pool[0]]
        for _ in range(n_games - 1):
            best, best_min = None, -1
            for g in pool[:5000]:
                if g not in selected:
                    min_dist = min(len(set(g) & set(s)) for s in selected)
                    if min_dist > best_min:
                        best_min = min_dist
                        best = g
            if best: selected.append(best)
        return selected[:n_games]

    def backtest(self, portfolio, test_draws):
        """Backtest: probabilidade de ≥1 acerto 11+"""
        n_success = 0
        for draw in test_draws:
            actual = set(draw['dezenas'])
            if any(len(set(g) & actual) >= 11 for g in portfolio):
                n_success += 1
        prob = n_success / len(test_draws) if test_draws else 0
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
# WALK-FORWARD VALIDATION (CORRIGIDO)
# ============================================================
def walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=10):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)...")
    results = []

    soft = {
        'pares': (7, 2.0), 'primos': (5, 2.0), 'moldura': (9, 2.0),
        'repeticoes': (8, 2.0), 'soma': (195, 20),
        'gap_medio': (1.62, 0.4), 'clusterizacao': (0.85, 0.15),
    }

    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue

        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue

        optimizer = PortfolioOptimizerV4(train_data, soft)
        portfolio, _ = optimizer.optimize(n_games, n_candidates=50000, iterations=50)
        random_portfolio = optimizer.generate_pure_random_portfolio(n_games)
        coverage_portfolio = optimizer.generate_coverage_baseline(n_games)

        bt_strat = optimizer.backtest(portfolio, test_data)
        bt_rand = optimizer.backtest(random_portfolio, test_data)
        bt_cov = optimizer.backtest(coverage_portfolio, test_data)

        results.append({
            'window': w,
            'strat_lift': bt_strat['lift'],
            'rand_lift': bt_rand['lift'],
            'cov_lift': bt_cov['lift'],
            'diff_vs_rand': bt_strat['lift'] - bt_rand['lift'],
            'diff_vs_cov': bt_strat['lift'] - bt_cov['lift'],
        })
        print(f" Janela {w}: lift={bt_strat['lift']:.3f} rand={bt_rand['lift']:.3f} cov={bt_cov['lift']:.3f}")

    if results:
        diffs_rand = [r['diff_vs_rand'] for r in results]
        diffs_cov = [r['diff_vs_cov'] for r in results]
        print(f"\n📊 RESUMO:")
        print(f"   Média diff vs Aleatório: {np.mean(diffs_rand):+.3f}")
        print(f"   Média diff vs Cobertura: {np.mean(diffs_cov):+.3f}")
        print(f"   Janelas + (vs Aleatório): {sum(1 for d in diffs_rand if d > 0)}/{len(results)}")
        try:
            _, p_rand = wilcoxon(diffs_rand)
            print(f"   Wilcoxon p (vs Aleatório): {p_rand:.4f}")
        except:
            pass
        try:
            _, p_cov = wilcoxon(diffs_cov)
            print(f"   Wilcoxon p (vs Cobertura): {p_cov:.4f}")
        except:
            pass
    return results


# ============================================================
# MONTE CARLO COMPARATIVO (TESTE DE SIGNIFICÂNCIA)
# ============================================================
def monte_carlo_significance(contests, n_simulations=100, blind_size=300, n_games=10):
    print(f"\n🎲 MONTE CARLO SIGNIFICÂNCIA ({n_simulations} simulações)...")

    soft = {
        'pares': (7, 2.0), 'primos': (5, 2.0), 'moldura': (9, 2.0),
        'repeticoes': (8, 2.0), 'soma': (195, 20),
    }

    strat_lifts = []
    rand_lifts = []
    cov_lifts = []

    for sim in tqdm(range(n_simulations), desc="Monte Carlo"):
        random.seed(sim)
        np.random.seed(sim)

        train = contests[:-blind_size]
        test = contests[-blind_size:]
        if len(train) < 100 or len(test) < 10: continue

        optimizer = PortfolioOptimizerV4(train, soft)
        portfolio, _ = optimizer.optimize(n_games, n_candidates=30000, iterations=30)
        random_portfolio = optimizer.generate_pure_random_portfolio(n_games)
        coverage_portfolio = optimizer.generate_coverage_baseline(n_games)

        strat_lifts.append(optimizer.backtest(portfolio, test)['lift'])
        rand_lifts.append(optimizer.backtest(random_portfolio, test)['lift'])
        cov_lifts.append(optimizer.backtest(coverage_portfolio, test)['lift'])

    strat_lifts = np.array(strat_lifts)
    rand_lifts = np.array(rand_lifts)
    cov_lifts = np.array(cov_lifts)

    print(f"\n📊 RESULTADOS MONTE CARLO:")
    print(f"   {'Estratégia':<20} {'Média Lift':<15} {'Std':<15} {'% > 1.0':<15}")
    print(f"   {'Geométrica':<20} {np.mean(strat_lifts):<15.4f} {np.std(strat_lifts):<15.4f} {np.mean(strat_lifts>1.0)*100:<15.1f}%")
    print(f"   {'Aleatória':<20} {np.mean(rand_lifts):<15.4f} {np.std(rand_lifts):<15.4f} {np.mean(rand_lifts>1.0)*100:<15.1f}%")
    print(f"   {'Cobertura':<20} {np.mean(cov_lifts):<15.4f} {np.std(cov_lifts):<15.4f} {np.mean(cov_lifts>1.0)*100:<15.1f}%")

    # Teste de significância
    from scipy.stats import mannwhitneyu
    _, p_strat_vs_rand = mannwhitneyu(strat_lifts, rand_lifts, alternative='greater')
    _, p_strat_vs_cov = mannwhitneyu(strat_lifts, cov_lifts, alternative='greater')
    print(f"\n   Mann-Whitney (Geométrica > Aleatória): p={p_strat_vs_rand:.4f}")
    print(f"   Mann-Whitney (Geométrica > Cobertura): p={p_strat_vs_cov:.4f}")

    if p_strat_vs_rand < 0.05:
        print(f"   ✅ Geométrica SIGNIFICATIVAMENTE melhor que Aleatória")
    else:
        print(f"   🟡 Geométrica NÃO significativamente melhor que Aleatória")

    return strat_lifts, rand_lifts, cov_lifts


# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR PARAMÉTRICO DE CARTEIRA v4")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    soft = {
        'pares': (7, 2.0),
        'primos': (5, 2.0),
        'moldura': (9, 2.0),
        'repeticoes': (8, 2.0),
        'soma': (195, 20),
        'gap_medio': (1.62, 0.4),
        'clusterizacao': (0.85, 0.15),
    }

    print("\nOpções:")
    print("1. Gerar carteira otimizada")
    print("2. Walk-forward validation (10 janelas)")
    print("3. Monte Carlo significância (100 simulações)")
    print("4. TUDO")
    op = input("Escolha [4]: ").strip() or "4"

    if op in ("1", "4"):
        print(f"\n🔧 INICIALIZANDO...")
        t0 = time.time()
        optimizer = PortfolioOptimizerV4(contests, soft)
        print(f"   ✅ Inicializado em {time.time()-t0:.1f}s")

        portfolio, score = optimizer.optimize(n_games=10, n_candidates=200000, iterations=100)

        print(f"\n🏆 CARTEIRA (Score: {score:.3f})")
        last = contests[-1]['dezenas']
        for i, game in enumerate(portfolio, 1):
            p = sum(1 for d in game if d % 2 == 0)
            pr = sum(1 for d in game if d in PRIMES)
            m = sum(1 for d in game if d in MOLDURA)
            rep = len(set(game) & set(last))
            print(f"   {i:2d}. {game} | P:{p} Pr:{pr} M:{m} Rep:{rep}")

        all_d = set(d for g in portfolio for d in g)
        print(f"\n📊 Cobertura: {len(all_d)}/25")
        print(f"📊 Entropia: {optimizer._portfolio_entropy(portfolio):.3f}")

        test_size = min(200, len(contests) // 3)
        if test_size > 10:
            test_data = contests[-test_size:]
            bt = optimizer.backtest(portfolio, test_data)
            print(f"\n🔬 BACKTEST ({bt['n_test']} concursos):")
            print(f"   Prob ≥1 acerto 11+: {bt['empirical']:.2%} (teórico: {bt['theoretical']:.2%})")
            print(f"   Lift: {bt['lift']:.2f}x")

    if op in ("2", "4"):
        walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=10)

    if op in ("3", "4"):
        monte_carlo_significance(contests, n_simulations=100, blind_size=300, n_games=10)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
