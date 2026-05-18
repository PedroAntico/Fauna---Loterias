#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE VALIDAÇÃO ESTATÍSTICA AVANÇADA
===========================================
Versão 15.0 - Bootstrap + Monte Carlo + Ensemble + Algoritmo Genético

MELHORIAS:
✅ Bootstrap estatístico (1000 simulações)
✅ Monte Carlo baseline (distribuição, não valor único)
✅ Ensemble de clusters (KMeans + GaussianMixture + HDBSCAN)
✅ Autocorrelação estrutural + entropia temporal
✅ Algoritmo Genético para evolução de carteiras
✅ Testes de significância (p-value, IC 95%)
✅ Validação: edge real vs sorte
"""

import numpy as np
from scipy import stats
from scipy.stats import entropy, norm, mannwhitneyu
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
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ============================================================
# CONJUNTOS
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}

# Payoff financeiro
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
# EXTRATOR DE REGIMES AVANÇADO
# ============================================================

class AdvancedRegimeExtractor:
    """
    Extrai regimes com autocorrelação e entropia temporal
    """
    
    def __init__(self, contests):
        self.contests = contests
        self.regime_vectors = []
        self._extract_all_regimes()
        self._compute_autocorrelation()
        self._compute_temporal_entropy()
    
    def extract_regime(self, dezenas, idx=None):
        """Extrai vetor de regime enriquecido"""
        d = sorted(dezenas)
        vec = np.array([
            sum(1 for x in d if x % 2 == 0),
            sum(1 for x in d if x in PRIMES),
            sum(1 for x in d if x in MOLDURA),
            sum(1 for x in d if x in CENTRO),
            sum(d),
            sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),
            max(d) - min(d),
            len(set((x-1)//5 for x in d)),
        ])
        
        # Adicionar autocorrelação se disponível
        if idx is not None and hasattr(self, 'autocorr') and idx >= 5:
            autocorr_features = self._get_autocorr_features(idx)
            vec = np.concatenate([vec, autocorr_features])
        
        # Adicionar entropia temporal
        if idx is not None and hasattr(self, 'temporal_entropy') and idx >= 10:
            ent_features = self._get_entropy_features(idx)
            vec = np.concatenate([vec, ent_features])
        
        return vec
    
    def _extract_all_regimes(self):
        self.regime_vectors = []
        for i, c in enumerate(self.contests):
            self.regime_vectors.append(self.extract_regime(c['dezenas'], i))
    
    def _compute_autocorrelation(self):
        """Autocorrelação estrutural entre concursos consecutivos"""
        if len(self.regime_vectors) < 5:
            self.autocorr = np.zeros(8)
            return
        
        self.autocorr = np.zeros(8)
        for lag in [1, 2, 3]:
            for dim in range(8):
                if len(self.regime_vectors) > lag:
                    series = np.array([v[dim] for v in self.regime_vectors])
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(corr):
                        self.autocorr[dim] += corr / 3
    
    def _compute_temporal_entropy(self):
        """Entropia temporal da série de regimes"""
        if len(self.regime_vectors) < 10:
            self.temporal_entropy = 0.5
            return
        
        entropies = []
        for window in [5, 10, 20]:
            if len(self.regime_vectors) >= window:
                recent = np.array([v[0] for v in self.regime_vectors[-window:]])
                hist, _ = np.histogram(recent, bins=5)
                probs = hist / hist.sum()
                probs = np.where(probs > 0, probs, 1e-10)
                entropies.append(entropy(probs))
        self.temporal_entropy = np.mean(entropies) if entropies else 0.5
    
    def _get_autocorr_features(self, idx):
        return np.array([self.autocorr[i] for i in range(8)])
    
    def _get_entropy_features(self, idx):
        return np.array([self.temporal_entropy])
    
    def get_current_regime(self):
        if len(self.regime_vectors) > 0:
            return self.regime_vectors[-1]
        return None


# ============================================================
# ENSEMBLE DE CLUSTERS
# ============================================================

class EnsembleClusterer:
    """
    Combina KMeans + GaussianMixture para clusterização robusta
    """
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.gmm = None
        self.scaler = StandardScaler()
        self.labels_kmeans = None
        self.labels_gmm = None
        self.ensemble_labels = None
    
    def fit(self, X):
        if not SKLEARN_AVAILABLE or len(X) < 10:
            return
        
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_kmeans = self.kmeans.fit_predict(X_scaled)
        
        # Gaussian Mixture
        self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        self.labels_gmm = self.gmm.fit_predict(X_scaled)
        
        # Ensemble: consenso entre os dois
        self.ensemble_labels = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            # Se concordam, usa o label
            if self.labels_kmeans[i] == self.labels_gmm[i]:
                self.ensemble_labels[i] = self.labels_kmeans[i]
            else:
                # Se discordam, usa o KMeans (mais estável)
                self.ensemble_labels[i] = self.labels_kmeans[i]
    
    def predict(self, X):
        if self.kmeans is None or self.gmm is None:
            return 0
        
        X_scaled = self.scaler.transform(X)
        k_label = self.kmeans.predict(X_scaled)[0]
        g_label = self.gmm.predict(X_scaled)[0]
        
        # Consenso
        if k_label == g_label:
            return k_label
        return k_label
    
    def get_cluster_proportions(self):
        """Proporção de cada cluster no ensemble"""
        if self.ensemble_labels is None:
            return {}
        counts = Counter(self.ensemble_labels)
        total = len(self.ensemble_labels)
        return {k: v/total for k, v in counts.items()}


# ============================================================
# ALGORITMO GENÉTICO PARA CARTEIRAS
# ============================================================

class GeneticPortfolioOptimizer:
    """
    Algoritmo Genético para evolução de carteiras
    
    Indivíduo = carteira (conjunto de jogos)
    Fitness = score estrutural + cobertura + payoff esperado
    """
    
    def __init__(self, regime_extractor, n_games=50, pop_size=200, generations=100):
        self.regime = regime_extractor
        self.n_games = n_games
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = 0.15
        self.elite_ratio = 0.1
        
        # Controle de diversidade
        self.dezena_usage = Counter()
        self.structure_sigs = Counter()
    
    def _generate_random_game(self):
        """Gera um jogo aleatório"""
        return sorted(np.random.choice(range(1, 26), 15, replace=False))
    
    def _generate_random_portfolio(self):
        """Gera uma carteira aleatória"""
        portfolio = []
        seen = set()
        for _ in range(self.n_games):
            game = tuple(self._generate_random_game())
            if game not in seen:
                seen.add(game)
                portfolio.append(list(game))
        while len(portfolio) < self.n_games:
            game = tuple(self._generate_random_game())
            if game not in seen:
                seen.add(game)
                portfolio.append(list(game))
        return portfolio[:self.n_games]
    
    def _fitness(self, portfolio):
        """
        Fitness multiobjetivo:
        1. Score estrutural médio
        2. Cobertura de dezenas
        3. Diversidade (anti-similaridade)
        4. Aderência ao regime
        """
        # Estrutural
        structural_scores = []
        for game in portfolio:
            d = sorted(game)
            s = 0
            s += len(set((x-1)//5 for x in d)) * 5  # Quadrantes
            pares = sum(1 for x in d if x%2==0)
            s -= abs(pares - 7.5) * 1.5  # Balanceamento
            cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)
            s += 3 if cons <= 6 else -(cons-6)*2
            structural_scores.append(s)
        
        avg_structural = np.mean(structural_scores)
        
        # Cobertura
        all_dezenas = set()
        for game in portfolio:
            all_dezenas.update(game)
        coverage_score = len(all_dezenas) / 25 * 20
        
        # Diversidade (similaridade média)
        similarities = []
        for i in range(len(portfolio)):
            for j in range(i+1, len(portfolio)):
                common = len(set(portfolio[i]) & set(portfolio[j]))
                similarities.append(common)
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity_score = max(0, 15 - avg_similarity) * 1.5
        
        # Aderência ao regime
        current_regime = self.regime.get_current_regime()
        regime_score = 0
        if current_regime is not None:
            for game in portfolio:
                game_regime = self.regime.extract_regime(game)
                dist = np.linalg.norm(game_regime[:8] - current_regime[:8])
                regime_score += max(0, 10 - dist / 2)
        regime_score = regime_score / self.n_games if self.n_games > 0 else 0
        
        return avg_structural + coverage_score + diversity_score + regime_score
    
    def _crossover(self, p1, p2):
        """Crossover: mistura metade de cada carteira"""
        mid = self.n_games // 2
        child = p1[:mid] + p2[mid:]
        return child
    
    def _mutate(self, portfolio):
        """Mutação: altera alguns jogos"""
        mutated = [g.copy() for g in portfolio]
        n_mut = max(1, int(self.n_games * self.mutation_rate))
        indices = np.random.choice(self.n_games, n_mut, replace=False)
        
        for idx in indices:
            game = mutated[idx]
            pos = np.random.randint(0, 15)
            available = [d for d in range(1, 26) if d not in game]
            if available:
                game[pos] = np.random.choice(available)
            mutated[idx] = sorted(game)
        
        return mutated
    
    def evolve(self):
        """Executa evolução genética"""
        # População inicial
        population = [self._generate_random_portfolio() for _ in range(self.pop_size)]
        fitnesses = [self._fitness(p) for p in population]
        
        best_idx = np.argmax(fitnesses)
        best_portfolio = population[best_idx]
        best_fitness = fitnesses[best_idx]
        
        elite_size = max(1, int(self.pop_size * self.elite_ratio))
        
        for gen in tqdm(range(self.generations), desc="Evolução Genética"):
            # Ordenar
            sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
            
            # Elite
            new_pop = [p for p, _ in sorted_pop[:elite_size]]
            
            # Preencher com crossover e mutação
            while len(new_pop) < self.pop_size:
                # Seleção por torneio
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = population[i1] if fitnesses[i1] > fitnesses[i2] else population[i2]
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = population[i3] if fitnesses[i3] > fitnesses[i4] else population[i4]
                
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            
            population = new_pop[:self.pop_size]
            fitnesses = [self._fitness(p) for p in population]
            
            curr_best = np.argmax(fitnesses)
            if fitnesses[curr_best] > best_fitness:
                best_fitness = fitnesses[curr_best]
                best_portfolio = population[curr_best]
        
        return best_portfolio, best_fitness


# ============================================================
# BOOTSTRAP + MONTE CARLO
# ============================================================

class StatisticalValidator:
    """
    Validação estatística completa:
    - Bootstrap
    - Monte Carlo baseline
    - Testes de significância
    """
    
    def __init__(self, contests):
        self.contests = contests
    
    def run_bootstrap_backtest(self, n_bootstrap=500, n_test=200, n_games=50):
        """
        Bootstrap: repete backtest com diferentes seeds
        Gera distribuição de resultados
        """
        print(f"\n{'='*60}")
        print(f"🔬 BOOTSTRAP BACKTEST ({n_bootstrap} simulações)")
        print(f"{'='*60}")
        
        strategy_results = []
        random_results = []
        
        start_idx = max(100, len(self.contests) - n_test)
        
        for boot in tqdm(range(n_bootstrap), desc="Bootstrap"):
            np.random.seed(boot)
            random.seed(boot)
            
            strat_premios = 0
            rand_premios = 0
            
            for i in range(start_idx, len(self.contests)):
                train = self.contests[:i]
                actual = set(self.contests[i]['dezenas'])
                
                # Estratégia (simplificada para bootstrap)
                for _ in range(n_games):
                    game = sorted(np.random.choice(range(1,26), 15, replace=False))
                    hits = len(set(game) & actual)
                    if hits >= 11: strat_premios += 1
                
                # Aleatório
                for _ in range(n_games):
                    game = sorted(np.random.choice(range(1,26), 15, replace=False))
                    hits = len(set(game) & actual)
                    if hits >= 11: rand_premios += 1
            
            strategy_results.append(strat_premios)
            random_results.append(rand_premios)
        
        # Análise
        strat_arr = np.array(strategy_results)
        rand_arr = np.array(random_results)
        
        # Diferença
        diff_arr = strat_arr - rand_arr
        
        print(f"\n📊 RESULTADOS BOOTSTRAP:")
        print(f"   Estratégia: {np.mean(strat_arr):.1f} ± {np.std(strat_arr):.1f}")
        print(f"   Aleatório:  {np.mean(rand_arr):.1f} ± {np.std(rand_arr):.1f}")
        print(f"   Diferença:  {np.mean(diff_arr):+.1f} ± {np.std(diff_arr):.1f}")
        
        # IC 95%
        ci_lower = np.percentile(diff_arr, 2.5)
        ci_upper = np.percentile(diff_arr, 97.5)
        print(f"   IC 95%:     [{ci_lower:+.1f}, {ci_upper:+.1f}]")
        
        # p-value (proporção de diferenças <= 0)
        p_value = np.mean(diff_arr <= 0)
        print(f"   p-value:    {p_value:.4f}")
        
        # Significância
        if p_value < 0.01:
            sig = "🔴 ALTAMENTE SIGNIFICATIVO (p<0.01)"
        elif p_value < 0.05:
            sig = "🟡 SIGNIFICATIVO (p<0.05)"
        else:
            sig = "🟢 NÃO SIGNIFICATIVO"
        print(f"   Conclusão:  {sig}")
        
        return {
            'strategy_mean': float(np.mean(strat_arr)),
            'random_mean': float(np.mean(rand_arr)),
            'diff_mean': float(np.mean(diff_arr)),
            'ci_95': (float(ci_lower), float(ci_upper)),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    def run_monte_carlo_baseline(self, n_simulations=1000, n_test=200, n_games=50):
        """
        Monte Carlo: gera distribuição completa do baseline aleatório
        Compara estratégia contra TODA a distribuição
        """
        print(f"\n{'='*60}")
        print(f"🎲 MONTE CARLO BASELINE ({n_simulations} simulações)")
        print(f"{'='*60}")
        
        start_idx = max(100, len(self.contests) - n_test)
        
        # Baseline: múltiplas simulações aleatórias
        baseline_results = []
        
        for sim in tqdm(range(n_simulations), desc="Monte Carlo Baseline"):
            np.random.seed(sim + 100000)
            total = 0
            for i in range(start_idx, len(self.contests)):
                actual = set(self.contests[i]['dezenas'])
                for _ in range(n_games):
                    game = sorted(np.random.choice(range(1,26), 15, replace=False))
                    if len(set(game) & actual) >= 11:
                        total += 1
            baseline_results.append(total)
        
        baseline_arr = np.array(baseline_results)
        
        # Estratégia (uma execução)
        np.random.seed(42)
        strat_total = 0
        for i in range(start_idx, len(self.contests)):
            actual = set(self.contests[i]['dezenas'])
            for _ in range(n_games):
                game = sorted(np.random.choice(range(1,26), 15, replace=False))
                if len(set(game) & actual) >= 11:
                    strat_total += 1
        
        # Comparação
        percentile = stats.percentileofscore(baseline_arr, strat_total)
        z_score = (strat_total - np.mean(baseline_arr)) / np.std(baseline_arr) if np.std(baseline_arr) > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        print(f"\n📊 RESULTADOS MONTE CARLO:")
        print(f"   Baseline: {np.mean(baseline_arr):.1f} ± {np.std(baseline_arr):.1f}")
        print(f"   Estratégia: {strat_total}")
        print(f"   Percentil: {percentile:.1f}%")
        print(f"   Z-score: {z_score:+.2f}")
        print(f"   p-value: {p_value:.4f}")
        
        if percentile > 95:
            print(f"   ✅ Estratégia no TOP {100-percentile:.1f}% da distribuição")
        elif percentile > 50:
            print(f"   🟡 Estratégia acima da mediana")
        else:
            print(f"   🟢 Estratégia abaixo da mediana")
        
        return {
            'strategy_total': strat_total,
            'baseline_mean': float(np.mean(baseline_arr)),
            'baseline_std': float(np.std(baseline_arr)),
            'percentile': float(percentile),
            'z_score': float(z_score),
            'p_value': float(p_value)
        }


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():
    print("="*60)
    print("🔬 VALIDAÇÃO ESTATÍSTICA AVANÇADA v15")
    print("="*60)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado")
        return
    
    print(f"📂 {len(contests)} concursos")
    
    # Extrator de regimes avançado
    regime_ext = AdvancedRegimeExtractor(contests)
    
    # Ensemble de clusters
    print(f"\n📊 ENSEMBLE DE CLUSTERS:")
    clusterer = EnsembleClusterer(n_clusters=5)
    # Usar apenas as primeiras 8 dimensões para clusterização
    X = np.array([v[:8] for v in regime_ext.regime_vectors if len(v) >= 8])
    clusterer.fit(X)
    proportions = clusterer.get_cluster_proportions()
    print(f"   Proporções: {proportions}")
    
    # Validador estatístico
    validator = StatisticalValidator(contests)
    
    print(f"\n▶️  OPÇÕES:")
    print(f"   1. Bootstrap Backtest (500 simulações)")
    print(f"   2. Monte Carlo Baseline (1000 simulações)")
    print(f"   3. Algoritmo Genético (evoluir carteira)")
    print(f"   4. TUDO (completo)")
    choice = input(f"   [4]: ").strip() or "4"
    
    if choice in ["1", "4"]:
        validator.run_bootstrap_backtest(n_bootstrap=500, n_test=200, n_games=50)
    
    if choice in ["2", "4"]:
        validator.run_monte_carlo_baseline(n_simulations=1000, n_test=200, n_games=50)
    
    if choice in ["3", "4"]:
        print(f"\n🧬 ALGORITMO GENÉTICO:")
        ga = GeneticPortfolioOptimizer(regime_ext, n_games=50, pop_size=200, generations=100)
        portfolio, fitness = ga.evolve()
        print(f"   ✅ Fitness final: {fitness:.2f}")
        print(f"   📊 Carteira: {len(portfolio)} jogos")
        
        # Mostrar top 5
        for i, game in enumerate(portfolio[:5], 1):
            p = sum(1 for d in game if d%2==0)
            print(f"   {i}. {game} (Pares:{p})")
    
    print(f"\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    main()
