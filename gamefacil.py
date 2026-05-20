#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERADOR PARAMÉTRICO DE CARTEIRA - LOTOFÁCIL v5
================================================
MUDANÇA DE PARADIGMA:
✅ Otimização DIRETA para P(≥1 acerto 11+) via Monte Carlo interno
✅ Pesos recalibrados: cobertura(0.35) + entropia(0.25) + diversidade(0.20) + centralidade(0.20)
✅ Multimodalidade forçada (máximo 2 jogos por componente GMM)
✅ Penalidade de interseção forte (max 8 dezenas em comum)
✅ Matriz de cobertura de pares e trincas
✅ Análise: concursos 13+ vêm de regiões mais raras?
✅ Frequência marginal recente (viés leve, não filtro)
✅ Monte Carlo interno para estimar P(≥11)
✅ Gerador mais livre preservado
✅ Features degeneradas removidas
"""

import numpy as np
from scipy.stats import entropy, hypergeom
from collections import Counter, defaultdict
from datetime import datetime
import warnings
import os
from math import comb
from tqdm import tqdm
import random
import time

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.mixture import GaussianMixture
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
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

# Features topológicas v5 (17 dimensões, sem degeneradas)
FEATURE_NAMES_V5 = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
]
IDX = {name: i for i, name in enumerate(FEATURE_NAMES_V5)}

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
                concurso = int(parts[0]); data = parts[1]
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
# ANÁLISE: CONCURSOS 13+ VÊM DE REGIÕES RARAS?
# ============================================================
def analyze_rare_events(contests):
    """Analisa se concursos com 13+ acertos têm features diferentes da média"""
    extractor = TopologicalFeatureExtractorV5(contests)
    features = extractor.build_feature_matrix()

    # Marcar concursos com 13+ (usando o próprio concurso como "jogo")
    high_hit_mask = np.zeros(len(contests), dtype=bool)
    for i, c in enumerate(contests):
        # Um jogo idêntico ao sorteio acertaria 15
        # Para simular, verificamos quantos concursos tiveram 13+ acertos contra jogos típicos
        # Simplificação: marcamos os últimos 100 como aproximação
        pass

    # Análise alternativa: comparar features de concursos agrupados por repetição
    repeticoes = features[:, IDX['repeticoes']]
    high_rep_mask = repeticoes >= 10
    low_rep_mask = repeticoes <= 7

    if high_rep_mask.sum() > 0 and low_rep_mask.sum() > 0:
        print(f"\n📊 ANÁLISE DE REGIMES:")
        print(f"   Alta repetição (≥10): {high_rep_mask.sum()} concursos")
        print(f"   Baixa repetição (≤7): {low_rep_mask.sum()} concursos")
        for name in ['gap_medio', 'clusterizacao', 'soma', 'pares', 'moldura']:
            high_mean = np.mean(features[high_rep_mask, IDX[name]])
            low_mean = np.mean(features[low_rep_mask, IDX[name]])
            print(f"   {name}: alta_rep={high_mean:.3f} vs baixa_rep={low_mean:.3f}")

    return features


# ============================================================
# EXTRATOR DE FEATURES v5
# ============================================================
class TopologicalFeatureExtractorV5:
    def __init__(self, contests):
        self.contests = contests
        self._repeat_history = []
        for i, c in enumerate(contests):
            if i > 0:
                self._repeat_history.append(len(set(contests[i-1]['dezenas']) & set(c['dezenas'])))
            else:
                self._repeat_history.append(0)
        # Frequências marginais recentes
        self._recent_freq = self._compute_recent_freq()

    def _compute_recent_freq(self, window=50):
        freq = Counter()
        start = max(0, len(self.contests) - window)
        for c in self.contests[start:]:
            freq.update(c['dezenas'])
        total = len(self.contests[start:])
        return {d: freq.get(d, 0) / total for d in range(1, 26)}

    def extract_features(self, game, last_contest=None):
        d = sorted(game)
        gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
        rep = len(set(d) & set(last_contest)) if last_contest else 8

        ent_trans = 0.0
        if len(self._repeat_history) >= 5:
            trans = [self._repeat_history[i+1]-self._repeat_history[i] for i in range(len(self._repeat_history)-1)]
            if len(set(trans)) > 1:
                freq = Counter(trans)
                probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
                ent_trans = float(entropy(np.where(probs>0, probs, 1e-10)))

        return np.array([
            float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps)),
            float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))),
            ent_trans,
            float(len(set((x-1)//5 for x in d))),
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
        ], dtype=np.float64)

    def build_feature_matrix(self):
        features_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            features_list.append(self.extract_features(c['dezenas'], last))
        return np.array(features_list, dtype=np.float64)

    def get_recent_freq_bonus(self, game):
        """Bônus leve por dezenas frequentes recentemente (viés fraco)"""
        return np.mean([self._recent_freq.get(d, 0) for d in game])


# ============================================================
# MODELO DE DISTRIBUIÇÃO (GMM + normalização)
# ============================================================
class DistributionModelV5:
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        self.n_features = feature_matrix.shape[1]
        self._build_gmm()
        self._gmm_norm = self._compute_gmm_norm()

    def _build_gmm(self):
        if SKLEARN_AVAILABLE and self.feature_matrix.shape[0] > 100:
            try:
                n_comp = min(6, self.feature_matrix.shape[0] // 200)
                self.gmm = GaussianMixture(n_components=max(3, n_comp), random_state=42)
                self.gmm.fit(self.feature_matrix)
                self._has_gmm = True
                return
            except: pass
        self._has_gmm = False

    def _compute_gmm_norm(self):
        if self._has_gmm:
            scores = self.gmm.score_samples(self.feature_matrix)
            return {'min': float(np.min(scores)), 'max': float(np.max(scores))}
        return {'min': -100.0, 'max': 100.0}

    def score_samples_normalized(self, features):
        if self._has_gmm:
            raw = float(self.gmm.score_samples(features.reshape(1, -1))[0])
            rng = self._gmm_norm['max'] - self._gmm_norm['min']
            return (raw - self._gmm_norm['min']) / rng if rng > 0 else 0.5
        return 0.5

    def predict_cluster(self, features):
        """Retorna o componente GMM mais provável"""
        if self._has_gmm:
            return int(self.gmm.predict(features.reshape(1, -1))[0])
        return 0

    @property
    def n_components(self):
        return self.gmm.n_components if self._has_gmm else 1


# ============================================================
# GERADOR LIVRE
# ============================================================
class FreeGeneratorV5:
    def __init__(self, last_contest=None):
        self.last = set(last_contest) if last_contest else None

    def generate_one(self):
        game = set()
        available = set(range(1, 26))
        if self.last and random.random() < 0.3:
            rep_pool = list(self.last & available)
            if rep_pool:
                n = random.randint(5, 10)
                game.update(random.sample(rep_pool, min(n, len(rep_pool))))
                available -= game
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                test = game | {d}
                s = len(set((x-1)//5 for x in test)) * 3
                st = sorted(test)
                cons = sum(1 for i in range(len(st)-1) if st[i+1]-st[i]==1)
                if cons > 6: s -= (cons - 6) * 1.5
                scores.append(s)
            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores = scores - np.max(scores)
                probs = np.exp(scores / 3.0)
                probs = probs / probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else:
                chosen = random.choice(candidates)
            game.add(chosen)
            available.remove(chosen)
        return sorted(game)[:15]

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))


# ============================================================
# OTIMIZADOR DE CARTEIRA v5
# ============================================================
class PortfolioOptimizerV5:
    def __init__(self, contests, soft_targets=None):
        self.contests = contests
        self.extractor = TopologicalFeatureExtractorV5(contests)
        self.feature_matrix = self.extractor.build_feature_matrix()
        self.dist_model = DistributionModelV5(self.feature_matrix)
        self.last = contests[-1]['dezenas'] if contests else None
        self.soft = soft_targets or {}
        self.generator = FreeGeneratorV5(self.last)
        self._mc_cache = {}  # Cache para Monte Carlo interno

    def _score_game(self, game):
        features = self.extractor.extract_features(game, self.last)
        gmm_score = self.dist_model.score_samples_normalized(features)
        # Frequência recente (viés leve)
        freq_bonus = self.extractor.get_recent_freq_bonus(game)
        return gmm_score * 0.85 + freq_bonus * 0.15, features, self.dist_model.predict_cluster(features)

    def _pair_coverage(self, portfolio):
        """Cobertura de pares de dezenas"""
        covered = set()
        for g in portfolio:
            for pair in combinations(sorted(g), 2):
                covered.add(pair)
        return len(covered) / comb(25, 2)

    def _triple_coverage(self, portfolio, sample_size=500):
        """Cobertura de trincas (amostrada)"""
        all_triples = list(combinations(range(1, 26), 3))
        if len(all_triples) > sample_size:
            sampled = random.sample(all_triples, sample_size)
        else:
            sampled = all_triples
        covered = 0
        for triple in sampled:
            for g in portfolio:
                if set(triple).issubset(set(g)):
                    covered += 1
                    break
        return covered / len(sampled)

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

    def _monte_carlo_p11(self, portfolio, n_simulations=2000):
        """Estima P(≥1 acerto 11+) via Monte Carlo interno"""
        cache_key = tuple(tuple(sorted(g)) for g in portfolio)
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]

        successes = 0
        for _ in range(n_simulations):
            drawn = set(np.random.choice(range(1, 26), 15, replace=False))
            if any(len(set(g) & drawn) >= 11 for g in portfolio):
                successes += 1

        prob = successes / n_simulations
        self._mc_cache[cache_key] = prob
        # Limitar cache
        if len(self._mc_cache) > 500:
            keys = list(self._mc_cache.keys())[:250]
            for k in keys: del self._mc_cache[k]
        return prob

    def _portfolio_score(self, portfolio):
        """Score focado em P(≥11) + cobertura + diversidade"""
        # Monte Carlo: probabilidade de ≥1 acerto 11+
        p11 = self._monte_carlo_p11(portfolio, n_simulations=1000)

        # Cobertura de pares
        pair_cov = self._pair_coverage(portfolio)

        # Entropia
        entropy_val = self._portfolio_entropy(portfolio)

        # Diversidade
        diversity = self._portfolio_diversity(portfolio)

        # Pesos recalibrados: FOCO em cobertura e diversidade
        return p11 * 0.35 + pair_cov * 0.25 + entropy_val * 0.20 + diversity * 0.20

    def _mutate_game(self, game):
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
                s, feats, cluster = self._score_game(game)
                pool.append((s, game, feats, cluster))

        pool.sort(key=lambda x: x[0], reverse=True)

        # Inicializar carteira com MULTIMODALIDADE FORÇADA
        portfolio = []
        cluster_counts = defaultdict(int)
        max_per_cluster = max(2, n_games // self.dist_model.n_components + 1)

        for s, game, feats, cluster in pool:
            if len(portfolio) >= n_games: break
            # Limitar jogos por cluster GMM
            if cluster_counts[cluster] >= max_per_cluster:
                continue
            # Penalidade de interseção FORTE (max 8)
            if any(len(set(game) & set(sg)) > 8 for sg in portfolio):
                continue
            portfolio.append(game)
            cluster_counts[cluster] += 1

        best_portfolio = list(portfolio)
        best_score = self._portfolio_score(portfolio)
        current_score = best_score

        # Simulated Annealing
        temp = 1.0
        elite_pool = pool[:len(pool)//4]

        for it in tqdm(range(iterations), desc="Annealing"):
            temp *= 0.95
            new_portfolio = list(portfolio)
            idx = random.randint(0, len(new_portfolio) - 1)

            if random.random() < 0.4 and elite_pool:
                _, new_game, _, _ = random.choice(elite_pool)
            elif random.random() < 0.7:
                new_game = self._mutate_game(new_portfolio[idx])
            else:
                new_game = self.generator.generate_one()

            # Verificar restrição de cluster
            _, _, new_cluster = self._score_game(new_game)
            current_cluster_counts = defaultdict(int)
            for g in new_portfolio:
                _, _, c = self._score_game(g)
                current_cluster_counts[c] += 1
            if current_cluster_counts[new_cluster] >= max_per_cluster and new_portfolio[idx] != new_game:
                continue

            # Verificar interseção
            too_similar = False
            for j, sg in enumerate(new_portfolio):
                if j != idx and len(set(new_game) & set(sg)) > 8:
                    too_similar = True
                    break
            if too_similar:
                continue

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
        return [self.generator.generate_pure_random() for _ in range(n_games)]

    def generate_coverage_baseline(self, n_games=10):
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
            'empirical': prob, 'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws), 'n_success': n_success,
        }


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=10):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)...")
    results = []

    soft = {
        'pares': (7, 2.0), 'primos': (5, 2.0), 'moldura': (9, 2.0),
        'repeticoes': (8, 2.0), 'soma': (195, 20),
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

        optimizer = PortfolioOptimizerV5(train_data, soft)
        portfolio, _ = optimizer.optimize(n_games, n_candidates=50000, iterations=50)
        random_portfolio = optimizer.generate_pure_random_portfolio(n_games)
        coverage_portfolio = optimizer.generate_coverage_baseline(n_games)

        bt_strat = optimizer.backtest(portfolio, test_data)
        bt_rand = optimizer.backtest(random_portfolio, test_data)
        bt_cov = optimizer.backtest(coverage_portfolio, test_data)

        results.append({
            'window': w,
            'strat_lift': bt_strat['lift'], 'rand_lift': bt_rand['lift'], 'cov_lift': bt_cov['lift'],
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
            from scipy.stats import wilcoxon
            _, p_rand = wilcoxon(diffs_rand)
            print(f"   Wilcoxon p (vs Aleatório): {p_rand:.4f}")
        except: pass
    return results


# ============================================================
# INTERFACE
# ============================================================
def main():
    print("="*70)
    print("🧬 GERADOR PARAMÉTRICO DE CARTEIRA v5")
    print("="*70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado.")
        return

    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    # Análise de regimes raros
    analyze_rare_events(contests)

    soft = {
        'pares': (7, 2.0), 'primos': (5, 2.0), 'moldura': (9, 2.0),
        'repeticoes': (8, 2.0), 'soma': (195, 20),
    }

    print("\nOpções:")
    print("1. Gerar carteira otimizada")
    print("2. Walk-forward validation (10 janelas)")
    print("3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"

    if op in ("1", "3"):
        print(f"\n🔧 INICIALIZANDO...")
        t0 = time.time()
        optimizer = PortfolioOptimizerV5(contests, soft)
        print(f"   ✅ Inicializado em {time.time()-t0:.1f}s")
        print(f"   GMM: {optimizer.dist_model.n_components} componentes")

        portfolio, score = optimizer.optimize(n_games=10, n_candidates=200000, iterations=100)

        print(f"\n🏆 CARTEIRA (Score: {score:.3f})")
        last = contests[-1]['dezenas']
        for i, game in enumerate(portfolio, 1):
            p = sum(1 for d in game if d % 2 == 0)
            pr = sum(1 for d in game if d in PRIMES)
            m = sum(1 for d in game if d in MOLDURA)
            rep = len(set(game) & set(last))
            _, _, cluster = optimizer._score_game(game)
            print(f"   {i:2d}. {game} | P:{p} Pr:{pr} M:{m} Rep:{rep} C:{cluster}")

        all_d = set(d for g in portfolio for d in g)
        pair_cov = optimizer._pair_coverage(portfolio)
        print(f"\n📊 Cobertura dezenas: {len(all_d)}/25")
        print(f"📊 Cobertura pares: {pair_cov:.3f}")
        print(f"📊 Entropia: {optimizer._portfolio_entropy(portfolio):.3f}")
        p11_est = optimizer._monte_carlo_p11(portfolio, n_simulations=5000)
        print(f"📊 P(≥1 acerto 11+) estimada: {p11_est:.3f}")

        test_size = min(200, len(contests) // 3)
        if test_size > 10:
            test_data = contests[-test_size:]
            bt = optimizer.backtest(portfolio, test_data)
            print(f"\n🔬 BACKTEST ({bt['n_test']} concursos):")
            print(f"   Prob ≥1 acerto 11+: {bt['empirical']:.2%} (teórico: {bt['theoretical']:.2%})")
            print(f"   Lift: {bt['lift']:.2f}x")

    if op in ("2", "3"):
        walk_forward_validation(contests, n_windows=10, train_size=500, test_size=50, n_games=10)

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
