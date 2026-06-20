#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEGA‑SENA LAB v2.1
CORREÇÕES CRÍTICAS:
✅ Fixas preservadas no Simulated Annealing e DPP
✅ Validação final: todos os jogos contêm as fixas
✅ Limites de filtros corrigidos (soma até 345, amplitude até 59)
✅ Score ponderado na busca OOS para evitar falsos campeões
✅ Tipos nativos (int) em todos os jogos gerados
✅ Mantém todas as funcionalidades do v2.0
"""

import numpy as np
from scipy.stats import hypergeom, binomtest
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings, json
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES – MEGA‑SENA
# ============================================================
TOTAL_DEZENAS = 60
DEZENAS_SORTEADAS = 6
PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}
FIBONACCI = {1,2,3,5,8,13,21,34,55}

Q1 = list(range(1, 16))
Q2 = list(range(16, 31))
Q3 = list(range(31, 46))
Q4 = list(range(46, 61))

PREMIO_FALLBACK = {4: 900.0, 5: 40000.0, 6: 3500000.0}
CUSTO_APOSTA = 5.0

TRAIN_SIZE = 1000
TEST_SIZE = 100
STEP = 50

# Limites reais dos filtros (evita truncar soma em 60)
FILTER_LIMITS = {
    'pares': (0, 6), 'primos': (0, 6), 'fibonacci': (0, 6),
    'soma': (21, 345), 'amplitude': (5, 59), 'consecutivos': (0, 5),
    'q1': (0, 6), 'q2': (0, 6), 'q3': (0, 6), 'q4': (0, 6),
}

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_megasena(csv_file='resultados_megasena.csv'):
    if not os.path.exists(csv_file):
        print(f"⚠️ Arquivo {csv_file} não encontrado. Gerando dados sintéticos...")
        return generate_synthetic_contests(2000)
    contests = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 8: continue
            try:
                dezenas = [int(x) for x in parts[2:8]]
                if len(dezenas) != 6 or len(set(dezenas)) != 6: continue
                if any(x < 1 or x > 60 for x in dezenas): continue
                contests.append({'concurso': int(parts[0]), 'data': parts[1], 'dezenas': sorted(dezenas)})
            except: continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos da Mega‑Sena carregados")
    return contests

def load_premios(csv_file='premios_megasena.csv'):
    if not os.path.exists(csv_file):
        return {}
    premios = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 4: continue
            try:
                concurso = int(parts[0])
                quadra = float(parts[1])
                quina = float(parts[2])
                sena = float(parts[3])
                premios[concurso] = {4: quadra, 5: quina, 6: sena}
            except: continue
    print(f"✅ Premiação carregada para {len(premios)} concursos")
    return premios

def generate_synthetic_contests(n):
    return [{'concurso': i+1, 'data': '', 'dezenas': sorted(np.random.choice(range(1,61), 6, replace=False))} for i in range(n)]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def extract_filter(dezenas, filter_name):
    d = sorted(dezenas)
    if filter_name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if filter_name == 'primos': return sum(1 for x in d if x in PRIMES)
    if filter_name == 'fibonacci': return sum(1 for x in d if x in FIBONACCI)
    if filter_name == 'soma': return sum(d)
    if filter_name == 'amplitude': return max(d) - min(d)
    if filter_name == 'consecutivos': return sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    if filter_name == 'q1': return sum(1 for x in d if x in Q1)
    if filter_name == 'q2': return sum(1 for x in d if x in Q2)
    if filter_name == 'q3': return sum(1 for x in d if x in Q3)
    if filter_name == 'q4': return sum(1 for x in d if x in Q4)
    return 0

ALL_FILTERS = ['pares', 'primos', 'fibonacci', 'soma', 'amplitude', 'consecutivos', 'q1', 'q2', 'q3', 'q4']

# ============================================================
# BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}
    def get_mask(self, game):
        key = tuple(game)
        if key not in self._cache:
            mask = 0
            for d in key:
                mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]

BITMASK_CACHE = BitmaskCache()
mask_intersection = lambda m1, m2: (m1 & m2).bit_count()

# ============================================================
# GERADOR DE JOGOS (CORRIGIDO)
# ============================================================
class LooseGenerator:
    def generate_random(self):
        return [int(x) for x in np.random.choice(range(1, 61), 6, replace=False)]
    def generate_with_fixed(self, fixed):
        fixed_set = set(fixed)
        restantes = list(set(range(1, 61)) - fixed_set)
        complemento = np.random.choice(restantes, 6 - len(fixed_set), replace=False)
        return sorted(fixed_set | set(int(x) for x in complemento))
    def generate_filtered(self, fixed=None, ranges=None):
        for _ in range(500):
            g = self.generate_with_fixed(fixed) if fixed else self.generate_random()
            if ranges is None:
                return g
            ok = True
            for filtro, (low, high) in ranges.items():
                val = extract_filter(g, filtro)
                if val < low or val > high:
                    ok = False
                    break
            if ok:
                return g
        return self.generate_with_fixed(fixed) if fixed else self.generate_random()

# ============================================================
# DISTÂNCIA DE MAHALANOBIS
# ============================================================
class MahalanobisDistance:
    def __init__(self, train_data):
        self.train_data = train_data
        self._compute_covariance()
    def _compute_covariance(self):
        n = len(self.train_data)
        if n < 2:
            self.mean = np.zeros(TOTAL_DEZENAS)
            self.inv_cov = np.eye(TOTAL_DEZENAS)
            return
        X = np.zeros((n, TOTAL_DEZENAS))
        for i, c in enumerate(self.train_data):
            for d in c['dezenas']:
                X[i, d-1] = 1
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(TOTAL_DEZENAS) * 1e-6
        self.inv_cov = np.linalg.inv(cov)
    def distance(self, game):
        x = np.zeros(TOTAL_DEZENAS)
        for d in game:
            x[d-1] = 1
        diff = x - self.mean
        return np.sqrt(np.dot(np.dot(diff.T, self.inv_cov), diff))

# ============================================================
# DPP
# ============================================================
def dpp_sample(pool, kernel_func, n_select=5):
    n = len(pool)
    if n <= n_select:
        return pool[:n_select]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = kernel_func(pool[i], pool[j])
            K[i, j] = sim
            K[j, i] = sim
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0)
    probs = eigenvalues / (eigenvalues + 1)
    selected = []
    indices = list(range(n))
    for _ in range(n_select):
        if not indices: break
        p = probs[indices] / np.sum(probs[indices])
        chosen = np.random.choice(indices, p=p)
        selected.append(pool[chosen])
        indices.remove(chosen)
    return selected

def kernel_similarity(g1, g2):
    inter = len(set(g1) & set(g2))
    return inter / DEZENAS_SORTEADAS

# ============================================================
# ANTI‑HUMAN SCORE
# ============================================================
def anti_human_score(game):
    score = 0.0
    dias = sum(1 for x in game if x <= 31)
    if dias >= 4:
        score -= 1.0
    g = sorted(game)
    run = 1
    for i in range(len(g)-1):
        if g[i+1] - g[i] == 1:
            run += 1
        else:
            run = 1
    if run >= 3:
        score -= 1.0
    mult5 = sum(1 for x in g if x % 5 == 0)
    if mult5 >= 3:
        score -= 0.5
    return score

# ============================================================
# FITNESS GLOBAL
# ============================================================
def fitness_global(portfolio):
    if len(portfolio) == 0:
        return -1e9
    covered_pairs = set()
    for g in portfolio:
        covered_pairs.update(combinations(sorted(g), 2))
    coverage = len(covered_pairs) / comb(60, 2)
    masks = [BITMASK_CACHE.get_mask(g) for g in portfolio]
    intersections = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            intersections.append(mask_intersection(masks[i], masks[j]))
    avg_inter = np.mean(intersections) if intersections else 0
    diversity = 1 - avg_inter / DEZENAS_SORTEADAS
    anti_human = np.mean([anti_human_score(g) for g in portfolio])
    freq = np.zeros(TOTAL_DEZENAS)
    for g in portfolio:
        for d in g:
            freq[d-1] += 1
    freq /= len(portfolio)
    freq = freq[freq > 0]
    entropy = -np.sum(freq * np.log2(freq)) / np.log2(60) if len(freq) > 0 else 0
    return coverage * 0.3 + diversity * 0.3 + anti_human * 0.1 + entropy * 0.3

# ============================================================
# OTIMIZADOR DE CARTEIRA (CORRIGIDO)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, premios=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.premios = premios if premios else {}
        self.mahalanobis = None
        if len(contests) >= 100:
            self.mahalanobis = MahalanobisDistance(contests)
        self.fixed = None  # armazena as fixas atuais

    def generate_pool(self, n_candidates, fixed=None, ranges=None):
        pool, seen = [], set()
        for _ in range(n_candidates):
            if ranges:
                g = self.generator.generate_filtered(fixed, ranges)
            else:
                g = self.generator.generate_with_fixed(fixed) if fixed else self.generator.generate_random()
            key = tuple(g)
            if key not in seen:
                seen.add(key)
                pool.append(g)
        return pool

    def select_pair_covering(self, candidates, n_select):
        if len(candidates) < n_select: return candidates[:n_select]
        covered, selected = set(), []
        for _ in range(n_select):
            best_idx, best_new = -1, -1
            for i, c in enumerate(candidates):
                if c in selected: continue
                pairs = set(combinations(sorted(c), 2))
                new_pairs = len(pairs - covered)
                if new_pairs > best_new:
                    best_new, best_idx = new_pairs, i
            if best_idx == -1: break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), 2))
        return selected

    def _enforce_fixed(self, portfolio, fixed):
        """Garante que todos os jogos contenham as fixas."""
        if not fixed:
            return portfolio
        fixed_set = set(fixed)
        result = []
        for g in portfolio:
            missing = fixed_set - set(g)
            if missing:
                # Substitui dezenas aleatórias pelas fixas faltantes
                g_set = set(g)
                # Remove dezenas não-fixas para abrir espaço
                non_fixed = list(g_set - fixed_set)
                for m in missing:
                    if non_fixed:
                        g_set.remove(non_fixed.pop())
                    g_set.add(m)
                g = sorted(int(x) for x in g_set)
            result.append(g)
        return result

    def simulated_annealing(self, initial, fixed=None, steps=500, temp_start=1.0, temp_end=0.01):
        current = initial[:]
        current_score = fitness_global(current)
        best = current[:]
        best_score = current_score
        for s in range(steps):
            t = temp_start * (temp_end / temp_start) ** (s / steps)
            idx = random.randint(0, len(current)-1)
            # Gera novo jogo respeitando as fixas (CORRIGIDO)
            if fixed:
                new_game = self.generator.generate_with_fixed(fixed)
            else:
                new_game = self.generator.generate_random()
            new_port = current[:]
            new_port[idx] = new_game
            new_score = fitness_global(new_port)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / t):
                current = new_port
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
        return best

    def optimize(self, n_games=5, n_candidates=5000, fixed=None, ranges=None, use_dpp=True, use_annealing=True):
        self.fixed = fixed  # armazena para uso interno
        pool = self.generate_pool(n_candidates, fixed, ranges)
        if len(pool) < n_games:
            return pool

        if use_dpp and len(pool) >= n_games:
            portfolio = dpp_sample(pool, kernel_similarity, n_games)
        else:
            portfolio = self.select_pair_covering(pool, n_games)

        if use_annealing:
            portfolio = self.simulated_annealing(portfolio, fixed)

        # VALIDAÇÃO FINAL: garantir que todas as fixas estão presentes
        if fixed:
            portfolio = self._enforce_fixed(portfolio, fixed)

        # Ordenar por Mahalanobis (menor distância = mais típico)
        if self.mahalanobis is not None:
            portfolio = sorted(portfolio, key=lambda g: self.mahalanobis.distance(g))

        return portfolio[:n_games]

    def backtest(self, portfolio, test_draws):
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint64)
        hit_counts = {k:0 for k in range(4,7)}
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            premio_concurso = self.premios.get(draw['concurso'], PREMIO_FALLBACK)
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 4:
                    n_success += 1
                    total_premio += premio_concurso.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(hypergeom.pmf(k, 60, 6, 6) for k in range(4,7))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        lift = prob/theo_prob if theo_prob>0 else 1.0
        roi = (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0
        # Score ponderado para evitar falsos campeões
        weighted_score = roi * np.sqrt(hit_counts.get(4,0) + hit_counts.get(5,0)*5 + hit_counts.get(6,0)*50)
        return {'lift': lift, 'roi': roi, 'hit_distribution': hit_counts, 'weighted_score': weighted_score}

# ============================================================
# RANKING DE PODER PREDITIVO (inalterado)
# ============================================================
class PredictiveRanking:
    def __init__(self, contests):
        self.contests = contests
    def rank_predictive_power(self, block_sizes=None):
        if block_sizes is None: block_sizes = [50, 100, 200, 500]
        all_results = {}
        for block_size in block_sizes:
            print(f"\n📊 BLOCOS DE {block_size} CONCURSOS")
            print(f"{'Filtro':<15} {'Estratégia':<12} {'Precisão':<10} {'Acertos':<10} {'p-value':<10}")
            print("-" * 60)
            for filtro in ALL_FILTERS:
                series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
                n_blocos = len(series)//block_size
                if n_blocos < 3: continue
                blocos = [series[i*block_size:(i+1)*block_size] for i in range(n_blocos)]
                for strategy in ['reversao', 'tendencia']:
                    acertos = total = 0
                    for i in range(1, len(blocos)):
                        mean_prev = np.mean(blocos[i-1])
                        mean_curr = np.mean(blocos[i])
                        hist_mean = np.mean(series[:i*block_size])
                        total += 1
                        if strategy == 'reversao':
                            if (mean_prev > hist_mean and mean_curr < mean_prev) or (mean_prev <= hist_mean and mean_curr > mean_prev):
                                acertos += 1
                        else:
                            if (mean_prev > hist_mean and mean_curr > mean_prev) or (mean_prev < hist_mean and mean_curr < mean_prev):
                                acertos += 1
                    acc = acertos/total*100 if total>0 else 0
                    pv = binomtest(acertos, total, 0.5, alternative='greater').pvalue if total>0 else 1.0
                    all_results[(filtro, strategy, block_size)] = {'accuracy':acc, 'acertos':acertos, 'total':total, 'p_value':pv}
                    sig = "🔍" if pv<0.05 else ("📊" if pv<0.15 else "  ")
                    print(f"{filtro:<15} {strategy:<12} {acc:<10.1f}% {acertos}/{total:<10} {pv:<10.4f} {sig}")
        return all_results

# ============================================================
# CONTROLE MONTE CARLO, TESTE CONCURSO, STRUCTURAL PREDICTOR, ETC.
# (mantidos com pequenas correções de limites)
# ============================================================
def monte_carlo_control(contests, n_simulations=100, block_sizes=None):
    if block_sizes is None: block_sizes = [50, 100, 200]
    n_concursos = len(contests)
    print(f"\n🎲 CONTROLE MONTE CARLO ({n_simulations} simulações)")
    ranker_real = PredictiveRanking(contests)
    real_results = ranker_real.rank_predictive_power(block_sizes)
    real_summary = {}
    for (filtro, strategy, block_size), res in real_results.items():
        real_summary[(filtro, strategy, block_size)] = res['accuracy']
    sim_accuracies = defaultdict(list)
    for sim in tqdm(range(n_simulations), desc="Simulações"):
        sim_contests = [{'dezenas': sorted(np.random.choice(range(1,61), 6, replace=False))} for _ in range(n_concursos)]
        ranker_sim = PredictiveRanking(sim_contests)
        sim_results = ranker_sim.rank_predictive_power(block_sizes)
        for key, res in sim_results.items():
            sim_accuracies[key].append(res['accuracy'])
    print(f"\n📊 COMPARAÇÃO REAL vs MONTE CARLO:")
    for key, real_acc in sorted(real_summary.items(), key=lambda x: x[1], reverse=True):
        sim_accs = sim_accuracies.get(key, [])
        if not sim_accs: continue
        mean_sim = np.mean(sim_accs)
        diff = real_acc - mean_sim
        p_emp = np.mean(np.array(sim_accs) >= real_acc)
        sig = "🔍" if p_emp<0.05 else ""
        print(f"   {key[0]:<12} {key[1]:<10} bloco={key[2]:<5} real={real_acc:.1f}% mc={mean_sim:.1f}% diff={diff:+.1f}% p={p_emp:.4f} {sig}")
    return real_summary, sim_accuracies

def test_concurso_a_concurso(contests, min_history=200):
    print(f"\n🎯 TESTE CONCURSO A CONCURSO (histórico mín: {min_history})")
    for filtro in ALL_FILTERS:
        series = np.array([extract_filter(c['dezenas'], filtro) for c in contests], dtype=float)
        acertos_rev = acertos_tend = total = 0
        for t in range(min_history, len(contests)-1):
            history = series[:t+1]; curr = series[t]; nxt = series[t+1]
            mean_short = np.mean(history[-20:]) if len(history)>=20 else np.mean(history)
            mean_long = np.mean(history)
            total += 1
            if (mean_short > mean_long and nxt < curr) or (mean_short <= mean_long and nxt > curr): acertos_rev += 1
            if (mean_short > mean_long and nxt > curr) or (mean_short < mean_long and nxt < curr): acertos_tend += 1
        acc_rev = acertos_rev/total*100 if total>0 else 0
        acc_tend = acertos_tend/total*100 if total>0 else 0
        print(f"   {filtro:<12}: reversão={acc_rev:.1f}% tendência={acc_tend:.1f}%")

class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests
    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR (método: {method})")
        ranges = {}
        for filtro in ALL_FILTERS:
            series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
            lo, hi = FILTER_LIMITS[filtro]
            if method == 'recent':
                recent = series[-50:] if len(series)>=50 else series
                low = int(np.percentile(recent, 25))
                high = int(np.percentile(recent, 75))
            elif method == 'ipe':
                short = np.mean(series[-20:]) if len(series)>=20 else np.mean(series)
                long = np.mean(series[-200:]) if len(series)>=200 else np.mean(series)
                ipe = (short - long) / long * 100 if long > 0 else 0
                center = int(round(short))
                if ipe > 10:
                    low, high = center, center
                elif ipe < -10:
                    low, high = center, center
                else:
                    low, high = center - 1, center + 1
            else:
                low, high = lo, hi
            ranges[filtro] = (max(lo, low), min(hi, high))
            print(f"   {filtro:<12}: [{ranges[filtro][0]}, {ranges[filtro][1]}]")
        return ranges

def walk_forward_structural(contests, train_size=TRAIN_SIZE, test_size=TEST_SIZE, step=STEP, premios=None):
    print(f"\n🔬 WALK‑FORWARD (treino={train_size}, teste={test_size}, passo={step})")
    results = []
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        try:
            predictor = StructuralPredictor(train_data)
            ranges = predictor.predict_ranges(method='recent')
            opt = PortfolioOptimizer(train_data, premios)
            portfolio = opt.optimize(5, 3000, ranges=ranges, use_dpp=True, use_annealing=True)
            bt = opt.backtest(portfolio, test_data)
            results.append({'start':start, 'lift':bt['lift'], 'roi':bt['roi'],
                           '4pts':bt['hit_distribution'].get(4,0), '5pts':bt['hit_distribution'].get(5,0), '6pts':bt['hit_distribution'].get(6,0)})
            print(f"   Janela {start}: lift={bt['lift']:.3f} ROI={bt['roi']:+.1f}% 4pts={bt['hit_distribution'].get(4,0)} 5pts={bt['hit_distribution'].get(5,0)}")
        except Exception as e:
            print(f"   Janela {start}: ERRO - {e}")
        start += step
    if results:
        print(f"\n📊 RESUMO: Média lift={np.mean([r['lift'] for r in results]):.3f} | Total 5pts={sum(r['5pts'] for r in results)} | Total 6pts={sum(r['6pts'] for r in results)}")
    return results

def search_best_fixed_oos(contests, n_fixed=2, top_n=20, train_size=1500, premios=None):
    print(f"\n🔎 BUSCANDO MELHORES {n_fixed} FIXAS (OUT‑OF‑SAMPLE)")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    candidates = []
    for fixed_tuple in tqdm(combinations(range(1,61), n_fixed), desc="Filtrando"):
        acertos = sum(1 for c in train_data if set(fixed_tuple).issubset(set(c['dezenas'])))
        freq = acertos/len(train_data)
        if freq >= 0.01: candidates.append((fixed_tuple, freq))
    candidates.sort(key=lambda x: x[1], reverse=True)
    results = []
    for fixed_tuple, freq in tqdm(candidates[:200], desc="Backtest OOS"):
        opt = PortfolioOptimizer(train_data, premios)
        portfolio = opt.optimize(5, 3000, fixed=list(fixed_tuple))
        bt = opt.backtest(portfolio, test_data)
        results.append({'fixed':fixed_tuple, 'freq':freq, 'roi':bt['roi'], 'weighted_score':bt['weighted_score']})
    # Ordenar pelo score ponderado (não apenas ROI)
    results.sort(key=lambda x: x['weighted_score'], reverse=True)
    print(f"\n🏆 TOP {top_n} FIXAS POR SCORE PONDERADO:")
    for i, res in enumerate(results[:top_n], 1):
        print(f"   {i}. {res['fixed']} | freq={res['freq']:.2%} | ROI={res['roi']:+.1f}% | Score={res['weighted_score']:.1f}")
    return results

def search_best_pairs_oos(contests, top_n=20, train_size=1500, premios=None):
    print(f"\n🔎 BUSCANDO MELHORES PARES (OUT‑OF‑SAMPLE)")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    freq_pairs = Counter()
    for c in train_data:
        freq_pairs.update(combinations(sorted(c['dezenas']), 2))
    top_pairs = [pair for pair, _ in freq_pairs.most_common(200)]
    results = []
    for pair in tqdm(top_pairs, desc="Testando pares"):
        opt = PortfolioOptimizer(train_data, premios)
        portfolio = opt.optimize(5, 3000, fixed=list(pair))
        bt = opt.backtest(portfolio, test_data)
        results.append({'pair':pair, 'roi':bt['roi'], 'weighted_score':bt['weighted_score']})
    results.sort(key=lambda x: x['weighted_score'], reverse=True)
    print(f"\n🏆 TOP {top_n} PARES POR SCORE PONDERADO:")
    for i, res in enumerate(results[:top_n], 1):
        print(f"   {i}. {res['pair']} | ROI={res['roi']:+.1f}% | Score={res['weighted_score']:.1f}")
    return results

def search_best_triples_oos(contests, top_n=20, train_size=1500, premios=None):
    print(f"\n🔎 BUSCANDO MELHORES TRINCAS (OUT‑OF‑SAMPLE)")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    freq_triples = Counter()
    for c in train_data:
        freq_triples.update(combinations(sorted(c['dezenas']), 3))
    top_triples = [trip for trip, _ in freq_triples.most_common(200)]
    results = []
    for trip in tqdm(top_triples, desc="Testando trincas"):
        opt = PortfolioOptimizer(train_data, premios)
        portfolio = opt.optimize(5, 3000, fixed=list(trip))
        bt = opt.backtest(portfolio, test_data)
        results.append({'triple':trip, 'roi':bt['roi'], 'weighted_score':bt['weighted_score']})
    results.sort(key=lambda x: x['weighted_score'], reverse=True)
    print(f"\n🏆 TOP {top_n} TRINCAS POR SCORE PONDERADO:")
    for i, res in enumerate(results[:top_n], 1):
        print(f"   {i}. {res['triple']} | ROI={res['roi']:+.1f}% | Score={res['weighted_score']:.1f}")
    return results

def analyze_cycles(contests):
    print(f"\n🔄 ANÁLISE DE CICLOS DE COBERTURA")
    todas_dezenas = set(range(1,61))
    ciclo_atual = []
    seen = set()
    for c in reversed(contests):
        seen.update(c['dezenas'])
        ciclo_atual.append(c['concurso'])
        if seen == todas_dezenas:
            break
    faltantes = sorted(todas_dezenas - seen)
    print(f"   Dezenas ainda não vistas no ciclo atual ({len(faltantes)}): {faltantes[:10]}...")
    print(f"   Concursos desde o início do ciclo: {len(ciclo_atual)}")
    ciclos = []
    seen = set()
    count = 0
    for c in contests:
        seen.update(c['dezenas'])
        count += 1
        if seen == todas_dezenas:
            ciclos.append(count)
            seen = set()
            count = 0
    if ciclos:
        c_arr = np.array(ciclos)
        print(f"   Ciclos históricos: média={np.mean(c_arr):.1f}, mediana={np.median(c_arr):.1f}")
        print(f"   Percentis: 25%={np.percentile(c_arr,25):.0f}, 50%={np.percentile(c_arr,50):.0f}, 75%={np.percentile(c_arr,75):.0f}, 90%={np.percentile(c_arr,90):.0f}")
        print(f"   Ciclo atual ({len(ciclo_atual)}): percentil {np.mean(c_arr <= len(ciclo_atual))*100:.0f}%")
    return ciclos

def monte_carlo_portfolio(contests, n_sim=500, premios=None):
    print(f"\n🎲 MONTE CARLO DE CARTEIRAS ({n_sim} simulações)")
    opt = PortfolioOptimizer(contests, premios)
    portfolio_modelo = opt.optimize(5, 5000, use_dpp=True, use_annealing=True)
    bt_modelo = opt.backtest(portfolio_modelo, contests[-200:])
    rois_aleatorios = []
    for _ in tqdm(range(n_sim), desc="Simulando carteiras aleatórias"):
        rand_port = [opt.generator.generate_random() for _ in range(5)]
        bt_rand = opt.backtest(rand_port, contests[-200:])
        rois_aleatorios.append(bt_rand['roi'])
    rois_aleatorios = np.array(rois_aleatorios)
    mean_rand = np.mean(rois_aleatorios)
    std_rand = np.std(rois_aleatorios)
    z_score = (bt_modelo['roi'] - mean_rand) / std_rand if std_rand > 0 else 0
    pct = np.mean(rois_aleatorios <= bt_modelo['roi']) * 100
    print(f"\n📊 RESULTADO:")
    print(f"   ROI modelo: {bt_modelo['roi']:+.1f}%")
    print(f"   ROI médio aleatório: {mean_rand:+.1f}%")
    print(f"   Z‑score: {z_score:+.2f}")
    print(f"   Percentil: {pct:.1f}%")
    if z_score > 2.0:
        print(f"   🔍 Carteira significativamente superior ao acaso!")
    else:
        print(f"   📊 Carteira não supera significativamente o aleatório.")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 MEGA‑SENA LAB v2.1")
    print("   CORRIGIDO: FIXAS PRESERVADAS + LIMITES + SCORE PONDERADO")
    print("="*70)
    contests = load_megasena('resultados_megasena.csv')
    if not contests:
        print("❌ Nenhum concurso disponível.")
        return
    premios = load_premios('premios_megasena.csv')
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\n" + "="*70)
        print("MENU PRINCIPAL")
        print("="*70)
        print("1. Gerar carteira otimizada (Mahalanobis + DPP + Annealing + Fitness)")
        print("2. Backtest nos últimos 200 concursos")
        print("3. Ranking de poder preditivo")
        print("4. Controle Monte Carlo")
        print("5. Teste concurso a concurso")
        print("6. Structural Predictor")
        print("7. Walk‑Forward do Structural Predictor")
        print("8. Buscar melhores fixas (OOS)")
        print("9. Buscar melhores pares (OOS)")
        print("10. Buscar melhores trincas (OOS)")
        print("11. Análise de ciclos de cobertura")
        print("12. Monte Carlo de carteiras")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            fixed_str = input("\n   Dezenas fixas (ex: 10,25,45 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split(',')] if fixed_str else []
            opt = PortfolioOptimizer(contests, premios)
            portfolio = opt.optimize(5, 5000, fixed=fixed if fixed else None, use_dpp=True, use_annealing=True)
            # Verificação rápida
            if fixed:
                for i, g in enumerate(portfolio, 1):
                    assert set(fixed).issubset(set(g)), f"Jogo {i} não contém as fixas!"
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0)
                pr = sum(1 for x in g if x in PRIMES)
                print(f" {i}. {g} | P:{p} Pr:{pr}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}% | Score={bt['weighted_score']:.1f}")

        elif op == '2':
            opt = PortfolioOptimizer(contests, premios)
            portfolio = opt.optimize(5, 5000, use_dpp=True, use_annealing=True)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   4pts={bt['hit_distribution'].get(4,0)} 5pts={bt['hit_distribution'].get(5,0)} 6pts={bt['hit_distribution'].get(6,0)}")

        elif op == '3':
            blocos_str = input("\n   Blocos (ex: 50,100,200) [50,100,200]: ").strip()
            try: block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [50,100,200]
            except: block_sizes = [50,100,200]
            PredictiveRanking(contests).rank_predictive_power(block_sizes)

        elif op == '4':
            try: n_sim = int(input("\n   Simulações [50]: ").strip() or "50")
            except: n_sim = 50
            monte_carlo_control(contests, n_sim)

        elif op == '5':
            try: min_hist = int(input("\n   Histórico mínimo [200]: ").strip() or "200")
            except: min_hist = 200
            test_concurso_a_concurso(contests, min_hist)

        elif op == '6':
            print("\n   Método: 1. Recente  2. IPE")
            metodo = input("   Escolha [1]: ").strip() or "1"
            method = 'recent' if metodo == '1' else 'ipe'
            StructuralPredictor(contests).predict_ranges(method=method)

        elif op == '7':
            walk_forward_structural(contests, premios=premios)

        elif op == '8':
            try: n_fixed = int(input("\n   Quantas fixas (1,2,3): ").strip())
            except: n_fixed = 2
            search_best_fixed_oos(contests, n_fixed, premios=premios)

        elif op == '9':
            search_best_pairs_oos(contests, premios=premios)

        elif op == '10':
            search_best_triples_oos(contests, premios=premios)

        elif op == '11':
            analyze_cycles(contests)

        elif op == '12':
            monte_carlo_portfolio(contests, premios=premios)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
