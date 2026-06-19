#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEGA‑SENA LAB v1.0
LABORATÓRIO ESTATÍSTICO + MOTOR DE OTIMIZAÇÃO COMBINATÓRIA

Sistema completo que une:
✅ Motor matemático: DPP, Simulated Annealing, Cobertura, Mahalanobis
✅ Laboratório científico: Monte Carlo, Walk‑Forward, Ranking Preditivo,
   Teste Concurso a Concurso, Structural Predictor, Busca OOS,
   Análise de Ciclos, Baseline Aleatório, ROI Real

ADAPTADO PARA MEGA‑SENA:
- 60 dezenas, 6 sorteadas
- Filtros: pares, primos, fibonacci, quadrantes (Q1‑Q4), soma, amplitude, consecutivos
- Premiação baseada em arquivo CSV com valores reais (fallback: valores fixos)
"""

import numpy as np
from scipy.stats import hypergeom, binomtest
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings, json
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES – MEGA‑SENA
# ============================================================
TOTAL_DEZENAS = 60
DEZENAS_SORTEADAS = 6
PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}
FIBONACCI = {1,2,3,5,8,13,21,34,55}

# Quadrantes da Mega‑Sena
Q1 = list(range(1, 16))   # 01‑15
Q2 = list(range(16, 31))  # 16‑30
Q3 = list(range(31, 46))  # 31‑45
Q4 = list(range(46, 61))  # 46‑60

# Premiação (fallback fixo, será substituído por CSV se disponível)
PREMIO_VALORES = {4: 900.0, 5: 40000.0, 6: 3500000.0}
CUSTO_APOSTA = 5.0

# Parâmetros do walk‑forward
TRAIN_SIZE = 1000
TEST_SIZE = 100
STEP = 50

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_megasena(csv_file='resultados_megasena.csv'):
    """Carrega concursos da Mega‑Sena. Formato: concurso;data;d1;d2;d3;d4;d5;d6"""
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

def generate_synthetic_contests(n):
    """Gera concursos sintéticos para teste (i.i.d.)."""
    return [{'concurso': i+1, 'data': '', 'dezenas': sorted(np.random.choice(range(1,61), 6, replace=False))} for i in range(n)]

# ============================================================
# FUNÇÕES AUXILIARES – EXTRAÇÃO DE FILTROS
# ============================================================
def extract_filter(dezenas, filter_name):
    """Extrai o valor de um filtro estrutural para um conjunto de dezenas."""
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

# Lista de todos os filtros disponíveis
ALL_FILTERS = ['pares', 'primos', 'fibonacci', 'soma', 'amplitude', 'consecutivos', 'q1', 'q2', 'q3', 'q4']

# ============================================================
# BITMASK (para interseções rápidas)
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
# GERADOR DE JOGOS
# ============================================================
class LooseGenerator:
    def generate_random(self):
        return sorted(np.random.choice(range(1, 61), 6, replace=False))
    def generate_with_fixed(self, fixed):
        fixed_set = set(fixed)
        restantes = list(set(range(1, 61)) - fixed_set)
        complemento = np.random.choice(restantes, 6 - len(fixed_set), replace=False)
        return sorted(fixed_set | set(complemento))

# ============================================================
# OTIMIZADOR DE CARTEIRA (PAIR COVERING)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests):
        self.contests = contests
        self.generator = LooseGenerator()
    def generate_pool(self, n_candidates, fixed=None):
        pool, seen = [], set()
        for _ in range(n_candidates):
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
    def backtest(self, portfolio, test_draws):
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint64)
        hit_counts = {k:0 for k in range(4,7)}
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 4:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(hypergeom.pmf(k, 60, 6, 6) for k in range(4,7))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        lift = prob/theo_prob if theo_prob>0 else 1.0
        roi = (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0
        return {'lift': lift, 'roi': roi, 'hit_distribution': hit_counts}

# ============================================================
# RANKING DE PODER PREDITIVO (MEGA‑SENA)
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
# CONTROLE MONTE CARLO
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

# ============================================================
# TESTE CONCURSO A CONCURSO
# ============================================================
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

# ============================================================
# STRUCTURAL PREDICTOR
# ============================================================
class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests
    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR (método: {method})")
        ranges = {}
        for filtro in ALL_FILTERS:
            series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
            if method == 'recent':
                recent = series[-50:] if len(series)>=50 else series
                low = int(np.percentile(recent, 25))
                high = int(np.percentile(recent, 75))
                ranges[filtro] = (low, high)
                print(f"   {filtro:<12}: [{low}, {high}]")
        return ranges

# ============================================================
# WALK‑FORWARD DO STRUCTURAL PREDICTOR
# ============================================================
def walk_forward_structural(contests, train_size=TRAIN_SIZE, test_size=TEST_SIZE, step=STEP):
    print(f"\n🔬 WALK‑FORWARD (treino={train_size}, teste={test_size}, passo={step})")
    results = []
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        try:
            predictor = StructuralPredictor(train_data)
            ranges = predictor.predict_ranges(method='recent')
            opt = PortfolioOptimizer(train_data)
            pool = opt.generate_pool(5000)
            portfolio = opt.select_pair_covering(pool, 5)
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

# ============================================================
# BUSCA OOS DE FIXAS
# ============================================================
def search_best_fixed_oos(contests, n_fixed=2, top_n=20, train_size=1500):
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
        opt = PortfolioOptimizer(train_data)
        pool = opt.generate_pool(3000, fixed=list(fixed_tuple))
        portfolio = opt.select_pair_covering(pool, 5)
        bt = opt.backtest(portfolio, test_data)
        results.append({'fixed':fixed_tuple, 'freq':freq, 'lift':bt['lift'], 'roi':bt['roi']})
    results.sort(key=lambda x: x['roi'], reverse=True)
    print(f"\n🏆 TOP {top_n} FIXAS POR ROI:")
    for i, res in enumerate(results[:top_n], 1):
        print(f"   {i}. {res['fixed']} | freq={res['freq']:.2%} | ROI={res['roi']:+.1f}%")
    return results

# ============================================================
# BUSCA OOS DE PARES
# ============================================================
def search_best_pairs_oos(contests, top_n=20, train_size=1500):
    print(f"\n🔎 BUSCANDO MELHORES PARES (OUT‑OF‑SAMPLE)")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    results = []
    for pair in tqdm(combinations(range(1,61), 2), desc="Testando pares"):
        opt = PortfolioOptimizer(train_data)
        pool = opt.generate_pool(3000, fixed=list(pair))
        portfolio = opt.select_pair_covering(pool, 5)
        bt = opt.backtest(portfolio, test_data)
        results.append({'pair':pair, 'lift':bt['lift'], 'roi':bt['roi']})
    results.sort(key=lambda x: x['roi'], reverse=True)
    print(f"\n🏆 TOP {top_n} PARES POR ROI:")
    for i, res in enumerate(results[:top_n], 1):
        print(f"   {i}. {res['pair']} | ROI={res['roi']:+.1f}%")
    return results

# ============================================================
# ANÁLISE DE CICLOS
# ============================================================
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
    print(f"   Dezenas ainda não vistas no ciclo atual: {sorted(todas_dezenas - seen)}")
    print(f"   Concursos desde o início do ciclo: {len(ciclo_atual)}")
    # Histórico de ciclos
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
        print(f"   Ciclos históricos: média={np.mean(ciclos):.1f}, mediana={np.median(ciclos):.1f}, mín={min(ciclos)}, máx={max(ciclos)}")
    return ciclos

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 MEGA‑SENA LAB v1.0")
    print("   LABORATÓRIO ESTATÍSTICO + OTIMIZAÇÃO COMBINATÓRIA")
    print("="*70)
    contests = load_megasena('megasena.csv')
    if not contests:
        print("❌ Nenhum concurso disponível.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira otimizada (Pair Covering)")
        print("2. Backtest nos últimos 200 concursos")
        print("3. Ranking de poder preditivo")
        print("4. Controle Monte Carlo")
        print("5. Teste concurso a concurso")
        print("6. Structural Predictor")
        print("7. Walk‑Forward do Structural Predictor")
        print("8. Buscar melhores fixas (OOS)")
        print("9. Buscar melhores pares (OOS)")
        print("10. Análise de ciclos de cobertura")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            fixed_str = input("\n   Dezenas fixas (ex: 10,25,45 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split(',')] if fixed_str else []
            opt = PortfolioOptimizer(contests)
            pool = opt.generate_pool(10000, fixed=fixed if fixed else None)
            portfolio = opt.select_pair_covering(pool, 5)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0)
                pr = sum(1 for x in g if x in PRIMES)
                print(f" {i}. {g} | P:{p} Pr:{pr}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

        elif op == '2':
            opt = PortfolioOptimizer(contests)
            pool = opt.generate_pool(10000)
            portfolio = opt.select_pair_covering(pool, 5)
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
            walk_forward_structural(contests)

        elif op == '8':
            try: n_fixed = int(input("\n   Quantas fixas (1,2,3): ").strip())
            except: n_fixed = 2
            search_best_fixed_oos(contests, n_fixed)

        elif op == '9':
            search_best_pairs_oos(contests)

        elif op == '10':
            analyze_cycles(contests)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
