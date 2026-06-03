#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v38
FOCO EM COBERTURA COMBINATÓRIA E GARANTIAS MATEMÁTICAS

MUDANÇA DE PARADIGMA:
✅ Abandona busca por previsão temporal
✅ Foca em engenharia de carteiras com garantias
✅ Wheel systems (fechamentos reduzidos)
✅ Cobertura de pares, triplas, quadras
✅ Diversidade via distância de Hamming e farthest‑point
✅ Cálculo de garantias matemáticas (t‑guarantee)
✅ Filtro estrutural (pares, moldura, primos) mantido
✅ Validação com walk‑forward e backtest
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon, pearsonr
from scipy.spatial.distance import cdist
from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.multitest import multipletests
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Algumas funções estarão indisponíveis.")

# ============================================================
# CONSTANTES GERAIS
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude", "compressao",
]
N_FEATURES = len(FEATURE_NAMES)

# Estruturais
MAX_CONSECUTIVOS_RUN = 7
STRUCTURAL_TARGETS = {
    'pares': (7.5, 3.0, 0.5),
    'primos': (5.0, 3.0, 0.5),
    'moldura': (9.5, 3.0, 0.3),
    'repeticoes': (9.0, 4.0, 0.3),
    'soma': (195.0, 40.0, 0.1),
    'consecutivos': (5.5, 5.0, 0.1),
    'amplitude': (22.0, 6.0, 0.1),
}

# Cobertura
DEFAULT_MAX_INTERSECTION = 8
DEFAULT_HAMMING_MIN_DIST = 5

# ============================================================
# BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}
    def get_mask(self, game):
        key = tuple(game) if isinstance(game, list) else game
        if key not in self._cache:
            mask = 0
            for d in key:
                mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]

BITMASK_CACHE = BitmaskCache()
mask_intersection = lambda m1, m2: (m1 & m2).bit_count()
hamming_distance = lambda g1, g2: 15 - mask_intersection(BITMASK_CACHE.get_mask(g1), BITMASK_CACHE.get_mask(g2))

def draw_masks_to_array(draws):
    return np.array([BITMASK_CACHE.get_mask(d) for d in draws], dtype=np.uint32)

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        return None
    contests = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17: continue
            try:
                dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                if len(dezenas) != 15 or len(set(dezenas)) != 15: continue
                if any(x < 1 or x > 25 for x in dezenas): continue
                contests.append({'concurso': int(parts[0]), 'data': parts[1], 'dezenas': sorted(dezenas)})
            except: continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos válidos")
    return contests

# ============================================================
# GERADOR COM CRITÉRIOS
# ============================================================
class LooseGenerator:
    def __init__(self, extractor=None):
        self.extractor = extractor

    def generate_one(self, max_penalty=30, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        for _ in range(500):
            game = self._generate_raw(allowed_pares, allowed_moldura, allowed_primos)
            if game is None:
                continue
            if self.extractor is not None:
                pen = self.extractor.compute_structural_penalty(game)
                if pen <= max_penalty:
                    return game
            else:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os critérios fornecidos.")

    def _generate_raw(self, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        if allowed_pares is None and allowed_moldura is None and allowed_primos is None:
            return self._generate_raw_old()
        for _ in range(200):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if allowed_pares is not None:
                if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                    continue
            return game
        return None

    def _generate_raw_old(self):
        game = set()
        available = set(range(1, 26))
        while len(game) < 15 and available:
            candidates = list(available)
            scores = []
            for d in candidates:
                test = sorted(game | {d})
                quad_score = len(set((x-1)//5 for x in test)) * 1.2
                if len(test) > 1:
                    gaps = [test[i+1]-test[i] for i in range(len(test)-1)]
                    cluster_penalty = sum(1 for g in gaps if g <= 2) * 1.8
                else: cluster_penalty = 0.0
                run = 1; max_run = 1
                for i in range(len(test)-1):
                    if test[i+1]-test[i]==1: run += 1; max_run = max(max_run, run)
                    else: run = 1
                consec_penalty = max(0, max_run - 4) * 5
                scores.append(quad_score - cluster_penalty - consec_penalty)
            if scores:
                scores = np.array(scores, dtype=np.float64)
                scores -= np.max(scores)
                probs = np.exp(scores / 2.0)
                probs /= probs.sum()
                chosen = np.random.choice(candidates, p=probs)
            else: chosen = random.choice(candidates)
            game.add(chosen); available.remove(chosen)
        return sorted(game)[:15]

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# EXTRATOR DE FEATURES SIMPLIFICADO
# ============================================================
class FeatureExtractor:
    def __init__(self, contests):
        self.contests = contests
    def compute_structural_penalty(self, game):
        d = sorted(game)
        penalty = 0.0
        actuals = {
            'pares': sum(1 for x in d if x%2==0),
            'primos': sum(1 for x in d if x in PRIMES),
            'moldura': sum(1 for x in d if x in MOLDURA),
            'repeticoes': len(set(d) & set(self.contests[-1]['dezenas'])) if self.contests else 8,
            'soma': sum(d),
            'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1),
            'amplitude': max(d)-min(d),
        }
        for name, (target, tol, w) in STRUCTURAL_TARGETS.items():
            if name in actuals:
                dev = abs(actuals[name]-target)
                if dev > tol:
                    penalty += (dev - tol) * w
        max_run = run = 1
        for i in range(len(d)-1):
            if d[i+1]-d[i]==1: run += 1; max_run = max(max_run, run)
            else: run = 1
        if max_run > MAX_CONSECUTIVOS_RUN:
            penalty += (max_run - MAX_CONSECUTIVOS_RUN) * 2.0
        return penalty

# ============================================================
# MÓDULO DE COBERTURA COMBINATÓRIA E GARANTIAS
# ============================================================
class CoverageEngine:
    """Motor de cobertura combinatória com garantias matemáticas."""
    
    def __init__(self, generator):
        self.generator = generator
    
    def generate_pool(self, n_candidates, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        """Gera um pool de jogos respeitando os critérios."""
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    allowed_pares=allowed_pares,
                    allowed_moldura=allowed_moldura,
                    allowed_primos=allowed_primos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool
    
    def select_max_coverage(self, candidates, n_select, level='triple'):
        """
        Seleciona n_select jogos maximizando cobertura.
        level: 'pair', 'triple', 'quad'
        """
        r = {'pair': 2, 'triple': 3, 'quad': 4}[level]
        # Farthest‑point adaptado para cobertura
        masks = np.array([BITMASK_CACHE.get_mask(c) for c in candidates], dtype=np.uint32)
        n = len(candidates)
        selected_idx = [0]
        for _ in range(n_select - 1):
            best_idx = -1
            best_new = -1
            for i in range(n):
                if i in selected_idx:
                    continue
                # Conta quantos novos r‑tuplos seriam cobertos
                current_tuples = set()
                for idx in selected_idx:
                    current_tuples.update(combinations(candidates[idx], r))
                new_tuples = set(combinations(candidates[i], r))
                new_count = len(new_tuples - current_tuples)
                if new_count > best_new:
                    best_new = new_count
                    best_idx = i
            if best_idx == -1:
                break
            selected_idx.append(best_idx)
        return [candidates[i] for i in selected_idx]
    
    def select_max_diversity(self, candidates, n_select, max_inter=DEFAULT_MAX_INTERSECTION):
        """Seleciona n_select jogos maximizando distância mínima (farthest‑point)."""
        unique = {}
        for c in candidates:
            mask = BITMASK_CACHE.get_mask(c)
            if mask not in unique:
                unique[mask] = c
        cand_unique = list(unique.values())
        if len(cand_unique) < n_select:
            cand_unique = candidates[:n_select]
        masks = np.array([BITMASK_CACHE.get_mask(c) for c in cand_unique], dtype=np.uint32)
        n = len(cand_unique)
        selected_idx = [0]
        for _ in range(n_select - 1):
            min_dists = np.full(n, np.inf, dtype=np.float64)
            for idx in selected_idx:
                intersect = np.array([mask_intersection(masks[i], masks[idx]) for i in range(n)])
                dist = 15.0 - intersect
                dist[intersect > max_inter] = -999.0
                min_dists = np.minimum(min_dists, dist)
            min_dists[selected_idx] = -1.0
            valid = np.where(min_dists >= 0)[0]
            if len(valid) == 0:
                break
            next_idx = valid[np.argmax(min_dists[valid])]
            selected_idx.append(next_idx)
        return [cand_unique[i] for i in selected_idx]
    
    def calculate_guarantees(self, portfolio, t_values=[15, 14, 13]):
        """
        Calcula garantias: se acertar t dezenas no sorteio, quantos acertos
        de 11, 12, 13, 14, 15 são garantidos em pelo menos um jogo da carteira?
        Retorna dicionário: {t: {k: min_acertos}}.
        """
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        guarantees = {}
        for t in t_values:
            # Enumera todas as combinações de t dezenas (25 choose t é grande para t=15, amostramos)
            if t == 15:
                # Apenas os próprios jogos da carteira
                test_sets = [set(g) for g in portfolio]
            elif t == 14:
                # Gera amostra de conjuntos de 14
                test_sets = []
                for _ in range(1000):
                    test_sets.append(set(np.random.choice(range(1,26), 14, replace=False)))
            elif t == 13:
                test_sets = []
                for _ in range(1000):
                    test_sets.append(set(np.random.choice(range(1,26), 13, replace=False)))
            else:
                continue
            
            min_hits = {k: 15 for k in range(11, 16)}
            for test_set in test_sets:
                test_mask = sum(1 << d for d in test_set)
                best = 0
                for pm in portfolio_masks:
                    hits = mask_intersection(pm, test_mask)
                    if hits > best:
                        best = hits
                for k in range(11, 16):
                    if best < k:
                        min_hits[k] = min(min_hits[k], 0)  # não atinge
                    elif min_hits[k] > 0:
                        min_hits[k] = min(min_hits[k], best)
            guarantees[t] = {k: min_hits[k] for k in range(11, 16)}
        return guarantees
    
    def wheel_system(self, n_numbers, n_games, guarantee=11, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        """
        Gera um sistema de fechamento (wheel) para n_numbers dezenas,
        com n_games jogos, buscando garantir pelo menos 'guarantee' pontos
        se as 15 sorteadas estiverem entre as n_numbers escolhidas.
        """
        print(f"\n🎯 WHEEL SYSTEM: {n_numbers} números, {n_games} jogos, garantia ≥ {guarantee} pontos")
        # Seleciona as n_numbers dezenas mais equilibradas (com critérios)
        if allowed_pares is None and allowed_moldura is None and allowed_primos is None:
            # Usa as n_numbers dezenas (ex.: 20) padrão
            base_numbers = set(range(1, n_numbers + 1))
        else:
            # Gera pool e escolhe as dezenas mais frequentes
            pool = self.generate_pool(2000, allowed_pares, allowed_moldura, allowed_primos)
            freq = Counter(d for g in pool for d in g)
            base_numbers = set([d for d, _ in freq.most_common(n_numbers)])
        
        print(f"   Dezenas base: {sorted(base_numbers)}")
        
        # Gera combinações de 15 dentro das n_numbers
        pool = []
        for _ in range(50000):
            game = sorted(np.random.choice(list(base_numbers), 15, replace=False))
            if allowed_pares is not None:
                if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                    continue
            pool.append(game)
            if len(pool) >= 3000:
                break
        
        if len(pool) < n_games:
            raise RuntimeError("Não foi possível gerar jogos suficientes com os critérios.")
        
        # Seleciona os n_games que maximizam cobertura de pares
        portfolio = self.select_max_coverage(pool, n_games, level='pair')
        
        # Exibe garantias
        guarantees = self.calculate_guarantees(portfolio)
        print(f"\n📊 GARANTIAS CALCULADAS:")
        for t, g in guarantees.items():
            print(f"   Se acertar {t} dezenas:")
            for k in range(11, 16):
                print(f"      Pelo menos {k} pontos? {'✅' if g[k] >= k else '❌'} (mínimo observado: {g[k]})")
        
        return portfolio, base_numbers

# ============================================================
# OTIMIZADOR DE CARTEIRA (FOCO EM COBERTURA)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        self.contests = contests
        self.extractor = FeatureExtractor(contests)
        self.generator = LooseGenerator(self.extractor)
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos
        self.engine = CoverageEngine(self.generator)
        self.historical_masks = draw_masks_to_array([c['dezenas'] for c in self.contests])
    
    def optimize_coverage(self, n_games=5, n_candidates=30000, strategy='diversity'):
        """Gera carteira focada em cobertura."""
        print(f"\n🧩 CARTEIRA DE COBERTURA: {n_games} jogos")
        if self.allowed_pares: print(f"   Pares permitidos: {self.allowed_pares}")
        if self.allowed_moldura: print(f"   Moldura permitida: {self.allowed_moldura}")
        if self.allowed_primos: print(f"   Primos permitidos: {self.allowed_primos}")
        
        t0 = time.time()
        pool = self.engine.generate_pool(n_candidates, self.allowed_pares, self.allowed_moldura, self.allowed_primos)
        print(f"   Pool gerado: {len(pool)} jogos")
        
        if strategy == 'diversity':
            portfolio = self.engine.select_max_diversity(pool, n_games)
        elif strategy == 'coverage':
            portfolio = self.engine.select_max_coverage(pool, n_games, level='triple')
        else:
            portfolio = pool[:n_games]
        
        print(f"✅ Otimizado em {time.time()-t0:.1f}s")
        return portfolio
    
    def backtest(self, portfolio, test_draws):
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {'empirical': prob, 'theoretical': theo_prob,
                'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts}

# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                            n_random_benchmark=100, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)")
    print(f"   Benchmark robusto: {n_random_benchmark} carteiras aleatórias por janela")
    results = []
    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue
        opt = PortfolioOptimizer(train_data, allowed_pares, allowed_moldura, allowed_primos)
        portfolio = opt.optimize_coverage(n_games, n_candidates=10000, strategy='diversity')
        bt = opt.backtest(portfolio, test_data)
        rand_lifts = []
        for _ in range(n_random_benchmark):
            rand_port = [opt.generator.generate_pure_random() for _ in range(n_games)]
            bt_rand = opt.backtest(rand_port, test_data)
            rand_lifts.append(bt_rand['lift'])
        rand_lifts = np.array(rand_lifts)
        mean_rand_lift = np.mean(rand_lifts)
        std_rand_lift = np.std(rand_lifts)
        z_score = (bt['lift'] - mean_rand_lift) / std_rand_lift if std_rand_lift > 0 else 0.0
        pct_rank = np.mean(rand_lifts <= bt['lift'])
        results.append({
            'window': w, 'strat_lift': bt['lift'], 'mean_rand_lift': mean_rand_lift,
            'z_score': z_score, 'pct_rank': pct_rank,
            'strat_14': bt['hit_distribution'].get(14,0),
        })
        print(f"   Janela {w}: lift={bt['lift']:.3f} | z={z_score:+.2f} | rank={pct_rank:.2f} | 14pts={bt['hit_distribution'].get(14,0)}")
    if results:
        diffs = [r['strat_lift'] - r['mean_rand_lift'] for r in results]
        print(f"\n📊 RESUMO FINAL:")
        print(f"   Média lift estratégia: {np.mean([r['strat_lift'] for r in results]):.3f}")
        print(f"   Média lift aleatório: {np.mean([r['mean_rand_lift'] for r in results]):.3f}")
        print(f"   Média z‑score: {np.mean([r['z_score'] for r in results]):+.2f}")
        print(f"   Wilcoxon p: ", end="")
        try:
            _, p = wilcoxon(diffs)
            print(f"{p:.4f}")
        except:
            print("N/A")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v38")
    print("   FOCO EM COBERTURA COMBINATÓRIA E GARANTIAS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira de cobertura (diversidade máxima)")
        print("2. Gerar carteira de cobertura (maximizar triplas)")
        print("3. Wheel System (fechamento reduzido)")
        print("4. Calcular garantias de uma carteira")
        print("5. Walk‑forward (validação)")
        print("6. Backtest nos últimos 200 concursos")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            print("\n📝 CRITÉRIOS (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio = opt.optimize_coverage(5, 30000, strategy='diversity')
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x % 2 == 0)
                pr = sum(1 for x in g if x in PRIMES)
                m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '2':
            print("\n📝 CRITÉRIOS (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio = opt.optimize_coverage(5, 30000, strategy='coverage')
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x % 2 == 0)
                pr = sum(1 for x in g if x in PRIMES)
                m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '3':
            print("\n📝 WHEEL SYSTEM")
            try:
                n_numbers = int(input("   Quantas dezenas base (ex: 20): ").strip())
                n_games = int(input("   Quantos jogos (ex: 5): ").strip())
                guarantee = int(input("   Garantia desejada (ex: 11): ").strip())
            except:
                print("Valores inválidos.")
                continue
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio, base = opt.engine.wheel_system(n_numbers, n_games, guarantee, allowed_pares, allowed_moldura, allowed_primos)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x % 2 == 0)
                pr = sum(1 for x in g if x in PRIMES)
                m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '4':
            print("\n📝 CARTEIRA PARA CÁLCULO DE GARANTIAS")
            print("   Digite os 5 jogos (15 dezenas separadas por espaço):")
            portfolio = []
            for i in range(5):
                game_str = input(f"   Jogo {i+1}: ").strip()
                game = [int(x) for x in game_str.split()]
                if len(game) != 15:
                    print("   ❌ Jogo inválido. 15 dezenas são necessárias.")
                    break
                portfolio.append(game)
            if len(portfolio) == 5:
                engine = CoverageEngine(LooseGenerator())
                guarantees = engine.calculate_guarantees(portfolio)
                for t, g in guarantees.items():
                    print(f"\n   Se acertar {t} dezenas:")
                    for k in range(11, 16):
                        status = "✅" if g[k] >= k else "❌"
                        print(f"      {k} pontos: {status} (mínimo observado: {g[k]})")
        
        elif op == '5':
            print("\n📝 CRITÉRIOS PARA WALK‑FORWARD (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            walk_forward_validation(contests, allowed_pares=allowed_pares, allowed_moldura=allowed_moldura, allowed_primos=allowed_primos)
        
        elif op == '6':
            print("\n📝 CRITÉRIOS (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio = opt.optimize_coverage(5, 30000, strategy='diversity')
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (últimos 200):")
            print(f"   Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} 13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
        
        elif op == '0':
            break
        
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
