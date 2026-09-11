#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA +MILIONÁRIA – v1.3
Baseado no v48.3 da Lotofácil, adaptado para:
  - 6 números de 1 a 50
  - 2 trevos de 1 a 6

EVOLUÇÃO:
✅ v1.0: Correção de leakage, controle FWER, walk-forward, filtros estruturais
✅ v1.1: Auditoria do Fator de Escala (F_t = S_{t-1}/média móvel)
✅ v1.2: F_prev e F_next usam referências DIFERENTES (sem denominador compartilhado)
        Nova Opção 14 — Auditoria do Coeficiente de Escala γ
        Split walk-forward 60/20/20 + grid pré-especificado + baseline IID
✅ v1.3: CORREÇÃO do alinhamento de índices na Opção 14
        (err_abs com target explícito; fim do broadcast 227 vs 379)
        IID MC reproduz o PROCEDIMENTO COMPLETO (escolha de γ* + avaliação no holdout)
        Reporta distribuição nula de γ* sob IID
"""

import numpy as np
from scipy.stats import hypergeom, binomtest, pearsonr, spearmanr, wilcoxon
from collections import defaultdict
from itertools import combinations
import os, random, time, warnings
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES DA +MILIONÁRIA
# ============================================================
TOTAL_MAIN = 50
CHOOSE_MAIN = 6
TOTAL_TREVO = 6
CHOOSE_TREVO = 2

PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}
MOLDURA = set(range(1, 11)) | set(range(41, 51))

HYPE_PROBS_MAIN = {k: hypergeom.pmf(k, TOTAL_MAIN, CHOOSE_MAIN, CHOOSE_MAIN) for k in range(0, CHOOSE_MAIN+1)}
HYPE_PROBS_TREVO = {k: hypergeom.pmf(k, TOTAL_TREVO, CHOOSE_TREVO, CHOOSE_TREVO) for k in range(0, CHOOSE_TREVO+1)}

PREMIO_VALORES = {
    (6,2): 10000000.0, (6,1): 50000.0, (6,0): 1500.0,
    (5,2): 2000.0, (5,1): 100.0, (5,0): 10.0,
    (4,2): 50.0, (4,1): 5.0, (4,0): 0.0,
    (3,2): 3.0, (3,1): 0.0, (3,0): 0.0,
}
CUSTO_APOSTA = 4.5

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
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_maismilionaria.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        return None
    contests = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 10:
                continue
            try:
                dezenas = [int(x.strip()) for x in parts[2:8] if x.strip()]
                trevos = [int(x.strip()) for x in parts[8:10] if x.strip()]
                if len(dezenas) != CHOOSE_MAIN or len(set(dezenas)) != CHOOSE_MAIN:
                    continue
                if any(x < 1 or x > TOTAL_MAIN for x in dezenas):
                    continue
                if len(trevos) != CHOOSE_TREVO or len(set(trevos)) != CHOOSE_TREVO:
                    continue
                if any(x < 1 or x > TOTAL_TREVO for x in trevos):
                    continue
                contests.append({
                    'concurso': int(parts[0]),
                    'data': parts[1],
                    'dezenas': sorted(dezenas),
                    'trevos': sorted(trevos)
                })
            except:
                continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos válidos")
    return contests

# ============================================================
# GERADOR
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                     excluded=None,
                     allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                     range_pares=None, range_moldura=None, range_primos=None,
                     range_soma=None, range_amplitude=None, range_consecutivos=None):
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed,
                                      excluded, allowed_pares, allowed_moldura, allowed_primos,
                                      range_pares, range_moldura, range_primos,
                                      range_soma, range_amplitude, range_consecutivos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      excluded, allowed_pares, allowed_moldura, allowed_primos,
                      range_pares, range_moldura, range_primos,
                      range_soma, range_amplitude, range_consecutivos):
        if fixed is None: fixed = []
        if semifixed is None: semifixed = []
        if excluded is None: excluded = []

        excluded_set = set(excluded)
        fixed_set = set(fixed) - excluded_set
        semifixed_set = set(semifixed) - fixed_set - excluded_set

        proibidas = fixed_set | semifixed_set | excluded_set
        todas = set(range(1, TOTAL_MAIN + 1))
        restantes = list(todas - proibidas)

        n_fixas = len(fixed_set)

        if max_semifixed is None:
            max_semi = len(semifixed_set)
        else:
            max_semi = min(max_semifixed, len(semifixed_set))

        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None

        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = CHOOSE_MAIN - n_fixas - n_semifixed_escolher

        if n_restantes < 0 or n_restantes > len(restantes):
            return None

        for _ in range(200):
            if n_semifixed_escolher > 0 and len(semifixed_set) > 0:
                chosen_semi = set(random.sample(list(semifixed_set), min(n_semifixed_escolher, len(semifixed_set))))
            else:
                chosen_semi = set()

            if n_restantes > 0:
                chosen_rest = set(random.sample(restantes, min(n_restantes, len(restantes))))
            else:
                chosen_rest = set()

            main_numbers = sorted(fixed_set | chosen_semi | chosen_rest)
            if len(main_numbers) != CHOOSE_MAIN:
                continue

            if allowed_pares is not None:
                if sum(1 for x in main_numbers if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in main_numbers if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in main_numbers if x in PRIMES) not in allowed_primos:
                    continue

            pares = sum(1 for x in main_numbers if x % 2 == 0)
            mol = sum(1 for x in main_numbers if x in MOLDURA)
            prim = sum(1 for x in main_numbers if x in PRIMES)
            soma = sum(main_numbers)
            amplitude = max(main_numbers) - min(main_numbers)
            consec = sum(1 for i in range(len(main_numbers)-1) if main_numbers[i+1]-main_numbers[i] == 1)

            if range_pares is not None and not (range_pares[0] <= pares <= range_pares[1]):
                continue
            if range_moldura is not None and not (range_moldura[0] <= mol <= range_moldura[1]):
                continue
            if range_primos is not None and not (range_primos[0] <= prim <= range_primos[1]):
                continue
            if range_soma is not None and not (range_soma[0] <= soma <= range_soma[1]):
                continue
            if range_amplitude is not None and not (range_amplitude[0] <= amplitude <= range_amplitude[1]):
                continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= consec <= range_consecutivos[1]):
                continue

            trevos = sorted(random.sample(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO))
            return (main_numbers, trevos)

        return None

    def generate_pure_random(self):
        main = sorted(np.random.choice(range(1, TOTAL_MAIN + 1), CHOOSE_MAIN, replace=False))
        trevos = sorted(np.random.choice(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO, replace=False))
        return (main, trevos)

# ============================================================
# OTIMIZADOR DE CARTEIRA
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 excluded=None, allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                 range_pares=None, range_moldura=None, range_primos=None,
                 range_soma=None, range_amplitude=None, range_consecutivos=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.excluded = excluded if excluded else []

        excl_set = set(self.excluded)
        self.fixed = fixed if fixed else []
        if excl_set & set(self.fixed):
            removidas_fixas = excl_set & set(self.fixed)
            print(f"⚠️ Dezenas fixas também excluídas; removendo das fixas: {sorted(removidas_fixas)}")
            self.fixed = [d for d in self.fixed if d not in excl_set]

        self.semifixed = semifixed if semifixed else []
        if excl_set & set(self.semifixed):
            removidas_semi = excl_set & set(self.semifixed)
            print(f"⚠️ Dezenas semifixas também excluídas; removendo das semifixas: {sorted(removidas_semi)}")
            self.semifixed = [d for d in self.semifixed if d not in excl_set]

        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos
        self.range_pares = range_pares
        self.range_moldura = range_moldura
        self.range_primos = range_primos
        self.range_soma = range_soma
        self.range_amplitude = range_amplitude
        self.range_consecutivos = range_consecutivos

    def generate_pool(self, n_candidates):
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed,
                    excluded=self.excluded,
                    allowed_pares=self.allowed_pares, allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares, range_moldura=self.range_moldura,
                    range_primos=self.range_primos, range_soma=self.range_soma,
                    range_amplitude=self.range_amplitude, range_consecutivos=self.range_consecutivos)
                key = (tuple(g[0]), tuple(g[1]))
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

    def select_covering(self, candidates, n_select, level='pair'):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        r = 2 if level == 'pair' else 3
        covered = set()
        selected = []
        for _ in range(n_select):
            best_idx = -1
            best_new = -1
            for i, c in enumerate(candidates):
                if c in selected:
                    continue
                groups = set(combinations(sorted(c[0]), r))
                new_groups = len(groups - covered)
                if new_groups > best_new:
                    best_new = new_groups
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx][0]), r))
        return selected

    def optimize(self, n_games=5, n_candidates=50000, method='pair_covering'):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | método: {method}")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.semifixed: print(f"   Semifixas: {self.semifixed} (mín={self.min_semifixed}, máx={self.max_semifixed})")
        if self.excluded: print(f"   Excluídas: {self.excluded}")
        if self.range_pares: print(f"   Pares: {self.range_pares}")
        if self.range_moldura: print(f"   Borda: {self.range_moldura}")
        if self.range_primos: print(f"   Primos: {self.range_primos}")
        if self.range_soma: print(f"   Soma: {self.range_soma}")
        if self.range_amplitude: print(f"   Amplitude: {self.range_amplitude}")
        if self.range_consecutivos: print(f"   Consecutivos: {self.range_consecutivos}")

        t0 = time.time()
        pool = self.generate_pool(n_candidates)
        print(f"   Pool: {len(pool)} jogos")
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")

        if method == 'pair_covering':
            portfolio = self.select_covering(pool, n_games, level='pair')
        elif method == 'triple_covering':
            portfolio = self.select_covering(pool, n_games, level='triple')
        else:
            portfolio = pool[:n_games]

        print(f"✅ {len(portfolio)} jogos em {time.time()-t0:.1f}s")
        return portfolio

    def backtest(self, portfolio, test_draws):
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        hit_counts = defaultdict(int)

        for draw in test_draws:
            draw_main = set(draw['dezenas'])
            draw_trevo = set(draw['trevos'])
            for main, trevo in portfolio:
                main_hits = len(draw_main & set(main))
                trevo_hits = len(draw_trevo & set(trevo))
                premio = PREMIO_VALORES.get((main_hits, trevo_hits), 0.0)
                if premio > 0:
                    n_success += 1
                    total_premio += premio
                    hit_counts[(main_hits, trevo_hits)] += 1

        prob = n_success / (len(portfolio) * len(test_draws)) if test_draws else 0
        p_single = 0.0
        for (m,t), _ in PREMIO_VALORES.items():
            p_single += HYPE_PROBS_MAIN[m] * HYPE_PROBS_TREVO[t]
        theo_prob = 1 - (1 - p_single)**len(portfolio)

        return {
            'empirical': prob, 'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws), 'n_success': n_success,
            'total_premio': total_premio, 'total_custo': total_custo,
            'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0,
            'hit_distribution': dict(hit_counts)
        }

# ============================================================
# EXTRAÇÃO DE FILTROS
# ============================================================
def extract_filter(dezenas, filter_name):
    d = sorted(dezenas)
    if filter_name == 'pares':
        return sum(1 for x in d if x % 2 == 0)
    elif filter_name == 'moldura':
        return sum(1 for x in d if x in MOLDURA)
    elif filter_name == 'primos':
        return sum(1 for x in d if x in PRIMES)
    elif filter_name == 'soma':
        return sum(d)
    elif filter_name == 'consecutivos':
        return sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    elif filter_name == 'amplitude':
        return max(d) - min(d)
    return 0

# ============================================================
# RANKING PREDITIVO
# ============================================================
class PredictiveRanking:
    def __init__(self, contests):
        self.contests = contests

    def rank_predictive_power(self, block_sizes=None, verbose=True):
        if block_sizes is None:
            block_sizes = [50, 100, 200, 500]
        filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']
        all_results = {}

        for block_size in block_sizes:
            if verbose:
                print(f"\n📊 BLOCOS DE {block_size} CONCURSOS")
                print(f"{'Filtro':<15} {'Estratégia':<12} {'Precisão':<10} {'Acertos':<10} {'p-value':<10}")
                print("-" * 60)

            for filtro in filters:
                series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
                n_blocos = len(series) // block_size
                if n_blocos < 3:
                    continue
                blocos = [series[i*block_size:(i+1)*block_size] for i in range(n_blocos)]

                for strategy in ['reversao', 'tendencia']:
                    acertos = 0
                    total_testes = 0
                    for i in range(2, len(blocos)):
                        mean_prev = np.mean(blocos[i-1])
                        mean_curr = np.mean(blocos[i])
                        historical_mean = np.mean(series[:(i-1)*block_size]) if (i-1)*block_size > 0 else mean_prev
                        total_testes += 1
                        if strategy == 'reversao':
                            predicted_down = mean_prev > historical_mean
                            if predicted_down and mean_curr < mean_prev:
                                acertos += 1
                            elif not predicted_down and mean_curr > mean_prev:
                                acertos += 1
                        else:
                            if mean_prev > historical_mean and mean_curr > mean_prev:
                                acertos += 1
                            elif mean_prev < historical_mean and mean_curr < mean_prev:
                                acertos += 1

                    accuracy = acertos / total_testes * 100 if total_testes > 0 else 0
                    p_value = binomtest(acertos, total_testes, 0.5, alternative='greater').pvalue if total_testes > 0 else 1.0
                    all_results[(filtro, strategy, block_size)] = {
                        'accuracy': accuracy, 'acertos': acertos, 'total': total_testes, 'p_value': p_value
                    }
                    if verbose:
                        sig = "🔍" if p_value < 0.05 else ("📊" if p_value < 0.15 else "  ")
                        print(f"{filtro:<15} {strategy:<12} {accuracy:<10.1f}% {acertos}/{total_testes:<10} {p_value:<10.4f} {sig}")

        return all_results

    def rank_dezenas_individual(self, block_sizes=None):
        if block_sizes is None:
            block_sizes = [100, 200, 500]
        print(f"\n📊 PODER PREDITIVO DAS {TOTAL_MAIN} DEZENAS")
        for block_size in block_sizes:
            accuracies = []
            for dezena in range(1, TOTAL_MAIN + 1):
                series = np.array([1 if dezena in c['dezenas'] else 0 for c in self.contests], dtype=float)
                n_blocos = len(series) // block_size
                if n_blocos < 3:
                    continue
                blocos = [series[i*block_size:(i+1)*block_size] for i in range(n_blocos)]
                acertos = 0
                total = 0
                for i in range(2, len(blocos)):
                    freq_prev = np.mean(blocos[i-1])
                    freq_curr = np.mean(blocos[i])
                    freq_hist = np.mean(series[:(i-1)*block_size]) if (i-1)*block_size > 0 else freq_prev
                    total += 1
                    if freq_prev > freq_hist and freq_curr < freq_prev:
                        acertos += 1
                    elif freq_prev < freq_hist and freq_curr > freq_prev:
                        acertos += 1
                if total > 0:
                    accuracies.append(acertos / total * 100)
            if accuracies:
                print(f"\n   Bloco {block_size}: média={np.mean(accuracies):.1f}%, "
                      f"melhor={np.argmax(accuracies)+1} ({max(accuracies):.1f}%), "
                      f"pior={np.argmin(accuracies)+1} ({min(accuracies):.1f}%)")
        return None

# ============================================================
# MONTE CARLO FWER
# ============================================================
def monte_carlo_control(contests, n_simulations=1000, block_sizes=None):
    if block_sizes is None:
        block_sizes = [50, 100, 200]
    n_concursos = len(contests)
    print(f"\n🎲 CONTROLE MONTE CARLO (FWER corrigido)")
    print(f"   Simulações: {n_simulations}")
    print(f"   Blocos testados: {block_sizes}\n")

    ranker_real = PredictiveRanking(contests)
    real_results = ranker_real.rank_predictive_power(block_sizes, verbose=False)
    real_max_acc = max(res['accuracy'] for res in real_results.values())
    real_max_key = max(real_results, key=lambda k: real_results[k]['accuracy'])

    sim_max_accs = []
    sim_accuracies_by_key = defaultdict(list)

    for _ in tqdm(range(n_simulations), desc="Simulações Monte Carlo"):
        sim_contests = []
        for _ in range(n_concursos):
            main = sorted(np.random.choice(range(1, TOTAL_MAIN + 1), CHOOSE_MAIN, replace=False))
            trevos = sorted(np.random.choice(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO, replace=False))
            sim_contests.append({'dezenas': main, 'trevos': trevos})
        ranker_sim = PredictiveRanking(sim_contests)
        sim_results = ranker_sim.rank_predictive_power(block_sizes, verbose=False)
        sim_max = max(res['accuracy'] for res in sim_results.values())
        sim_max_accs.append(sim_max)
        for key, res in sim_results.items():
            sim_accuracies_by_key[key].append(res['accuracy'])

    p_global = np.mean(np.array(sim_max_accs) >= real_max_acc)
    p_global_str = f"<{1.0/n_simulations:.4f}" if p_global == 0.0 else f"{p_global:.4f}"

    print(f"\n📊 COMPARAÇÃO REAL vs. MONTE CARLO")
    print(f"{'Filtro':<15} {'Estratégia':<12} {'Bloco':<8} {'Real':<10} {'MC Médio':<10} {'MC Std':<10} {'Diferença':<10} {'p (MC)':<10}")
    print("-" * 90)

    for key, res_dict in sorted(real_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        filtro, strategy, block_size = key
        real_acc = res_dict['accuracy']
        sim_accs = sim_accuracies_by_key.get(key, [])
        if not sim_accs:
            continue
        mean_sim = np.mean(sim_accs)
        std_sim = np.std(sim_accs)
        diff = real_acc - mean_sim
        p_emp = np.mean(np.array(sim_accs) >= real_acc)
        p_emp_str = f"<{1.0/len(sim_accs):.4f}" if p_emp == 0.0 else f"{p_emp:.4f}"
        marker = " 🏆" if key == real_max_key else ""
        print(f"{filtro:<15} {strategy:<12} {block_size:<8} {real_acc:<10.1f}% {mean_sim:<10.1f}% {std_sim:<10.1f} {diff:<10.1f}% {p_emp_str:<10}{marker}")

    print(f"\n🌟 Melhor resultado real: {real_max_key} com acurácia de {real_max_acc:.1f}%")
    print(f"   p‑global (FWER corrigido): {p_global_str}")
    if p_global < 0.05:
        print("   ✅ Evidência de sinal preditivo genuíno (p < 0.05)")
    else:
        print("   ❌ Sem evidência significativa — compatível com aleatoriedade")

    return real_results, sim_accuracies_by_key

# ============================================================
# TESTE CONCURSO A CONCURSO
# ============================================================
def test_concurso_a_concurso(contests, min_history=200):
    print(f"\n🎯 TESTE CONCURSO A CONCURSO")
    print(f"   Histórico mínimo: {min_history}")
    filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']

    for filtro in filters:
        series = np.array([extract_filter(c['dezenas'], filtro) for c in contests], dtype=float)
        acertos_reversao = acertos_tendencia = total = 0

        for t in range(min_history, len(contests) - 1):
            current_val = series[t]
            next_val = series[t+1]
            history = series[:t]
            if len(history) < 20:
                continue
            mean_short = np.mean(history[-20:])
            mean_long = np.mean(history)
            total += 1
            pred_rev_down = mean_short > mean_long
            if pred_rev_down and next_val < current_val:
                acertos_reversao += 1
            elif not pred_rev_down and next_val > current_val:
                acertos_reversao += 1
            if mean_short > mean_long and next_val > current_val:
                acertos_tendencia += 1
            elif mean_short < mean_long and next_val < current_val:
                acertos_tendencia += 1

        acc_rev = acertos_reversao / total * 100 if total > 0 else 0
        acc_tend = acertos_tendencia / total * 100 if total > 0 else 0
        p_rev = binomtest(acertos_reversao, total, 0.5, alternative='greater').pvalue if total > 0 else 1.0
        p_tend = binomtest(acertos_tendencia, total, 0.5, alternative='greater').pvalue if total > 0 else 1.0
        print(f"{filtro:<15}: Reversão={acc_rev:.1f}% ({acertos_reversao}/{total}, p={p_rev:.4f}) | "
              f"Tendência={acc_tend:.1f}% ({acertos_tendencia}/{total}, p={p_tend:.4f})")

# ============================================================
# STRUCTURAL PREDICTOR
# ============================================================
class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests

    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR (método: {method})")
        filters_info = {
            'pares': {'min': 0, 'max': 6},
            'moldura': {'min': 0, 'max': 6},
            'primos': {'min': 0, 'max': 6},
            'soma': {'min': 21, 'max': 285},
            'amplitude': {'min': 5, 'max': 49},
            'consecutivos': {'min': 0, 'max': 5}
        }
        ranges = {}
        for filtro, info in filters_info.items():
            series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
            if method == 'recent':
                recent = series[-50:]
                mean_val = np.mean(recent)
                low = max(info['min'], int(np.percentile(recent, 35)))
                high = min(info['max'], int(np.percentile(recent, 65)))
                ranges[filtro] = (low, high)
                print(f"   {filtro:<15}: [{low}, {high}] (média={mean_val:.1f})")
            elif method == 'ipe':
                freq_short = np.mean(series[-20:])
                freq_long = np.mean(series[-500:]) if len(series) >= 500 else np.mean(series)
                ipe = (freq_short - freq_long) / freq_long * 100 if freq_long > 0 else 0.0
                if ipe > 5:
                    predicted = max(info['min'], int(freq_long))
                    ranges[filtro] = (predicted - 1, predicted + 1)
                elif ipe < -5:
                    predicted = min(info['max'], int(freq_long) + 1)
                    ranges[filtro] = (predicted - 1, predicted + 1)
                else:
                    ranges[filtro] = (int(freq_long) - 1, int(freq_long) + 1)
                print(f"   {filtro:<15}: {ranges[filtro]} (IPE={ipe:+.1f}%)")
        return ranges

# ============================================================
# WALK-FORWARD STRUCTURAL
# ============================================================
def walk_forward_structural(contests, train_size=500, test_size=50, step=50, excluded=None):
    print(f"\n🔬 WALK‑FORWARD DO STRUCTURAL PREDICTOR")
    print(f"   Treino: {train_size}, Teste: {test_size}, Passo: {step}")
    if excluded:
        print(f"   Excluídas: {excluded}")
    results = []
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        predictor = StructuralPredictor(train_data)
        ranges = predictor.predict_ranges(method='recent')
        opt = PortfolioOptimizer(train_data, excluded=excluded,
                                 range_pares=ranges.get('pares'),
                                 range_moldura=ranges.get('moldura'),
                                 range_primos=ranges.get('primos'),
                                 range_soma=ranges.get('soma'),
                                 range_amplitude=ranges.get('amplitude'),
                                 range_consecutivos=ranges.get('consecutivos'))
        try:
            portfolio = opt.optimize(5, 10000, method='pair_covering')
            bt = opt.backtest(portfolio, test_data)
            results.append({'start': start, 'lift': bt['lift'], 'roi': bt['roi'],
                            'premios': sum(bt['hit_distribution'].values())})
            print(f"   Janela {start}: lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}% | "
                  f"Prêmios={sum(bt['hit_distribution'].values())}")
        except Exception as e:
            print(f"   Janela {start}: ERRO - {e}")
        start += step
    if results:
        print(f"\n📊 RESUMO:")
        print(f"   Média lift: {np.mean([r['lift'] for r in results]):.3f}")
        print(f"   Média ROI: {np.mean([r['roi'] for r in results]):.1f}%")
        print(f"   Total de prêmios: {sum(r['premios'] for r in results)}")
    return results

# ============================================================
# BUSCA OOS E COMPARAÇÕES
# ============================================================
def search_best_fixed_oos(contests, n_fixed=3, top_n=20, train_size=3500, n_games=5, n_candidates=10000, method='pair_covering', excluded=None):
    print(f"\n🔎 BUSCANDO MELHORES {n_fixed} FIXAS (OUT-OF-SAMPLE)")
    if excluded:
        print(f"   Excluídas: {excluded}")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    candidates = []
    for fixed_tuple in tqdm(combinations(range(1, TOTAL_MAIN + 1), n_fixed), desc="Filtrando"):
        if excluded and any(x in fixed_tuple for x in excluded):
            continue
        fixed_set = set(fixed_tuple)
        acertos = sum(1 for c in train_data if fixed_set.issubset(set(c['dezenas'])))
        freq = acertos / len(train_data)
        if freq >= 0.001:
            candidates.append((fixed_tuple, freq, acertos))
    candidates.sort(key=lambda x: x[1], reverse=True)
    results = []
    for fixed_tuple, freq, acertos in tqdm(candidates[:200], desc="Backtest OOS"):
        opt = PortfolioOptimizer(train_data, fixed=list(fixed_tuple), excluded=excluded)
        try:
            portfolio = opt.optimize(n_games, n_candidates, method=method)
            bt = opt.backtest(portfolio, test_data)
            results.append({'fixed': fixed_tuple, 'freq_treino': freq,
                            'lift': bt['lift'], 'roi': bt['roi'],
                            'premios': sum(bt['hit_distribution'].values())})
        except:
            continue
    results.sort(key=lambda x: x['roi'], reverse=True)
    print(f"\n🏆 TOP {top_n} FIXAS POR ROI (OUT-OF-SAMPLE):")
    for i, res in enumerate(results[:top_n], 1):
        print(f"{i:<5} {str(res['fixed']):<20} ROI={res['roi']:<10.1f}% Prêmios={res['premios']}")
    return results

def compare_trincas(contests, trinca1, trinca2, n_games=5, n_candidates=50000, method='pair_covering', excluded=None):
    print(f"\n⚔️ COMPARAÇÃO DE TRINCAS")
    if excluded:
        print(f"   Excluídas: {excluded}")
    for i, trinca in enumerate([trinca1, trinca2], 1):
        opt = PortfolioOptimizer(contests, fixed=list(trinca), excluded=excluded)
        portfolio = opt.optimize(n_games, n_candidates, method=method)
        bt = opt.backtest(portfolio, contests[-200:])
        print(f"   Trinca {i} ({trinca}): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

# ============================================================
# OPÇÃO 13 – AUDITORIA DO FATOR DE ESCALA (v1.3)
#   F_prev e F_next usam referências DIFERENTES (sem denominador compartilhado).
# ============================================================
def audit_scale_factor(contests, window=50):
    sums = np.array([sum(c['dezenas']) for c in contests], dtype=float)
    n = len(sums)
    if n < window + 30:
        print(f"⚠️  Histórico insuficiente ({n} < {window+30}).")
        return None

    F_prev_l, F_next_l, S_prev_l, S_next_l, idx_l = [], [], [], [], []
    for t in range(window + 1, n):
        ref_prev = float(np.mean(sums[t - window - 1 : t - 1]))
        ref_next = float(np.mean(sums[t - window     : t    ]))
        if ref_prev <= 0 or ref_next <= 0:
            continue
        F_prev_l.append(sums[t - 1] / ref_prev)
        F_next_l.append(sums[t]     / ref_next)
        S_prev_l.append(sums[t - 1])
        S_next_l.append(sums[t])
        idx_l.append(t)

    F_prev = np.array(F_prev_l)
    F_next = np.array(F_next_l)
    S_prev = np.array(S_prev_l)
    S_next = np.array(S_next_l)
    refs_next = np.array([np.mean(sums[t - window:t]) for t in idx_l])
    N = len(F_prev)

    print(f"\n📐 AUDITORIA DO FATOR DE ESCALA (v1.3 — refs independentes)  janela = {window}")
    print(f"   Observações válidas: {N}")
    print(f"   F_prev: média={np.mean(F_prev):.4f} | dp={np.std(F_prev):.4f} | "
          f"min={np.min(F_prev):.4f} | max={np.max(F_prev):.4f}")

    pear  = pearsonr(F_prev, F_next)
    spear = spearmanr(F_prev, F_next)
    print(f"\n   Correlação  F_prev  vs  F_next:")
    print(f"     Pearson   r = {pear.statistic:+.4f}  (p = {pear.pvalue:.4f})")
    print(f"     Spearman  ρ = {spear.statistic:+.4f}  (p = {spear.pvalue:.4f})")

    above_prev = F_prev > 1
    above_next = F_next > 1
    concord = int(np.sum(above_prev == above_next))
    p_sign  = binomtest(concord, N, 0.5, alternative='two-sided').pvalue
    print(f"\n   Concordância de estado (F>1 ⇔ próximo F>1):")
    print(f"     {concord}/{N} = {concord/N*100:.1f}%  (binomial p = {p_sign:.4f})")

    if np.sum(above_prev) > 0:
        p_below_given_above = float(np.mean(F_next[above_prev] < 1.0))
        print(f"     P(F_next < 1 | F_prev > 1) = {p_below_given_above:.4f}  (esperado se IID: ~0.5)")
    if np.sum(~above_prev) > 0:
        p_above_given_below = float(np.mean(F_next[~above_prev] > 1.0))
        print(f"     P(F_next > 1 | F_prev < 1) = {p_above_given_below:.4f}  (esperado se IID: ~0.5)")

    print(f"\n   Hipóteses concorrentes:")
    if pear.pvalue < 0.05 and pear.statistic > 0:
        print(f"     ✅ H1 PERSISTÊNCIA (r = {pear.statistic:+.3f}, p = {pear.pvalue:.4f})")
    elif pear.pvalue < 0.05 and pear.statistic < 0:
        print(f"     ✅ H2 REVERSÃO       (r = {pear.statistic:+.3f}, p = {pear.pvalue:.4f})")
    else:
        print(f"     ⚪ H3 INDEPENDÊNCIA não rejeitada (p = {pear.pvalue:.4f})")

    pred_mean_hist = np.array([np.mean(sums[:t]) for t in idx_l])
    pred_mean_mov  = refs_next.copy()
    pred_median    = np.array([np.median(sums[t-window:t]) for t in idx_l])
    pred_last      = S_prev.copy()
    pred_rev05     = refs_next + 0.5 * (refs_next - S_prev)
    pred_rev10     = refs_next + 1.0 * (refs_next - S_prev)

    methods = {
        'Média histórica':      pred_mean_hist,
        'Média móvel':          pred_mean_mov,
        'Mediana móvel':        pred_median,
        'Último valor':         pred_last,
        'Reversão 0.5 (vs mov)': pred_rev05,
        'Reversão 1.0 (vs mov)': pred_rev10,
    }

    print(f"\n   📊 COMPARAÇÃO DE PREDITORES DA SOMA S_t (N = {N}):")
    print(f"   {'Método':<24} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Corr':>9}")
    print("   " + "-" * 62)

    results = {}
    for name, pred in methods.items():
        err  = np.abs(pred - S_next)
        mae  = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(err / S_next) * 100)
        corr = float(pearsonr(pred, S_next).statistic) if np.std(pred) > 0 else 0.0
        results[name] = {'err': err, 'mae': mae, 'rmse': rmse, 'mape': mape, 'corr': corr}
        print(f"   {name:<24} {mae:>8.2f} {rmse:>8.2f} {mape:>8.2f} {corr:>+9.3f}")

    ref_key = 'Último valor'
    ref_err = results[ref_key]['err']
    print(f"\n   📊 WILCOXON pareado vs '{ref_key}':")
    for name, r in results.items():
        if name == ref_key:
            continue
        delta = ref_err - r['err']
        if np.all(delta == 0):
            continue
        try:
            _stat, p = wilcoxon(delta, alternative='greater')
            sig = "✅" if p < 0.05 else ("📊" if p < 0.15 else "  ")
            print(f"     {name:<24} Δmédio = {np.mean(delta):+7.2f}   p = {p:.4f}   {sig}")
        except Exception as e:
            print(f"     {name:<24} (erro: {e})")

    return {'F_prev': F_prev, 'F_next': F_next, 'S_next': S_next, 'results': results}

# ============================================================
# OPÇÃO 14 – AUDITORIA DO COEFICIENTE DE ESCALA γ  (v1.3 CORRIGIDA)
#
# Modelo:  Ŝ_t = R_t + γ·(S_(t-1) − R_t),  R_t = mean(S[t-w : t])
#
# Correções v1.3:
#   • err_abs com target explícito (fim do broadcast 227 vs 379)
#   • Sn[i_disc] / Sn[i_conf] / Sn[i_hold] aplicados consistentemente
#   • IID MC reproduz o PROCEDIMENTO COMPLETO:
#       escolhe γ* na descoberta sintética → mede Δ_OOS no holdout sintético
#   • Reporta distribuição nula de γ* sob IID
# ============================================================
GAMMA_GRID = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25]
WINDOWS_DEFAULT = [7, 10, 20, 30, 50, 100]


def _gamma_experiment(sums, window, gamma_grid):
    """
    Executa UM experimento γ completo sobre uma série `sums` (real ou sintética).
    Nenhum vazamento: cada predição em t usa apenas S[t-window : t].
    """
    n = len(sums)
    idx_l, S_prev_l, S_next_l, ref_l = [], [], [], []
    for t in range(window + 1, n):
        ref = float(np.mean(sums[t - window:t]))
        if ref <= 0:
            continue
        idx_l.append(t)
        S_prev_l.append(sums[t - 1])
        S_next_l.append(sums[t])
        ref_l.append(ref)

    idx = np.asarray(idx_l, dtype=int)
    Sp  = np.asarray(S_prev_l, dtype=float)
    Sn  = np.asarray(S_next_l, dtype=float)
    R   = np.asarray(ref_l,   dtype=float)
    N   = len(idx)
    if N < 50:
        return None

    n_disc = int(0.60 * N)
    n_conf = int(0.20 * N)
    i_disc = slice(0, n_disc)
    i_conf = slice(n_disc, n_disc + n_conf)
    i_hold = slice(n_disc + n_conf, N)

    preds = {g: R + g * (Sp - R) for g in gamma_grid}
    baselines = {
        'Média histórica': np.array([np.mean(sums[:t]) for t in idx]),
        'Média móvel':     R.copy(),
        'Mediana móvel':   np.array([np.median(sums[t - window:t]) for t in idx]),
        'Último valor':    Sp.copy(),
    }

    def mae(p, target):
        return float(np.mean(np.abs(np.asarray(p, dtype=float)
                                    - np.asarray(target, dtype=float))))

    # γ* escolhido SÓ com a descoberta
    mae_disc_g = {g: mae(preds[g][i_disc], Sn[i_disc]) for g in gamma_grid}
    best_gamma = min(gamma_grid, key=lambda g: mae_disc_g[g])
    best_key   = f'γ={best_gamma:+.2f}'

    phases = {}
    for phase, sl in [('disc', i_disc), ('conf', i_conf), ('hold', i_hold)]:
        target_ph = Sn[sl]                              # ← ALVO ALINHADO
        phase_mae = {name: mae(v[sl], target_ph) for name, v in baselines.items()}
        for g in gamma_grid:
            phase_mae[f'γ={g:+.2f}'] = mae(preds[g][sl], target_ph)
        phase_abs = {
            'Média histórica': np.abs(baselines['Média histórica'][sl] - target_ph),
            'Média móvel':     np.abs(baselines['Média móvel'][sl]     - target_ph),
            'Mediana móvel':   np.abs(baselines['Mediana móvel'][sl]   - target_ph),
            'Último valor':    np.abs(baselines['Último valor'][sl]    - target_ph),
            'γ*':              np.abs(preds[best_gamma][sl]            - target_ph),
        }
        if 0.0 in gamma_grid:
            phase_abs['γ=0'] = np.abs(preds[0.0][sl] - target_ph)
        phases[phase] = {'mae': phase_mae, 'abs_err': phase_abs}

    return {
        'N': N, 'n_disc': n_disc, 'n_conf': n_conf, 'n_hold': N - n_disc - n_conf,
        'best_gamma': best_gamma, 'best_key': best_key, 'phases': phases,
    }


def _iid_sums(n, rng):
    """Gera n somas IID equivalentes a um sorteio uniforme 6-de-50."""
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = rng.choice(np.arange(1, TOTAL_MAIN + 1), size=CHOOSE_MAIN, replace=False)
        out[i] = float(np.sum(s))
    return out


def audit_gamma_coefficient(contests, windows=None, gamma_grid=None, iid_mc=200):
    if windows is None:    windows = WINDOWS_DEFAULT
    if gamma_grid is None: gamma_grid = GAMMA_GRID

    sums = np.asarray([sum(c['dezenas']) for c in contests], dtype=float)
    n_real = len(sums)

    print(f"\n📐 AUDITORIA DO COEFICIENTE DE ESCALA γ  (v1.3)")
    print(f"   Modelo: Ŝ_t = R_t + γ·(S_(t-1) − R_t),  R_t = mean(S[t-w : t])")
    print(f"   γ grid: {gamma_grid}")
    print(f"   Janelas: {windows}")
    print(f"   Split: 60% descoberta | 20% confirmação | 20% holdout")
    print(f"   IID MC por janela: {iid_mc} réplicas\n")

    print(f"   📌 Estatísticas da soma S_t (real):")
    print(f"      E[S] teórico (uniforme 6/50) = {CHOOSE_MAIN*(TOTAL_MAIN+1)/2:.2f}")
    print(f"      E[S] amostral                = {np.mean(sums):.2f}")
    print(f"      SD  amostral                 = {np.std(sums):.2f}")
    print(f"      MAE esperado IID (média)     ≈ SD·sqrt(2/π) = "
          f"{np.std(sums)*np.sqrt(2/np.pi):.2f}\n")

    # ---------- 1) Execução real ----------
    reais = {}
    for w in windows:
        r = _gamma_experiment(sums, w, gamma_grid)
        if r is not None:
            reais[w] = r
    if not reais:
        print("⚠️  Nenhuma janela produziu amostra válida.")
        return None

    # ---------- 2) Tabela por janela ----------
    print("=" * 100)
    print("   RESULTADOS REAIS  (MAE por fase)")
    print("=" * 100)
    header = (f"{'Jan':<5} {'Fase':<11} {'N':<5} "
              f"{'MHist':<8} {'MMov':<8} {'Med':<8} {'Últ':<8} "
              f"{'γ*':<7} {'MAE(γ*)':<8}")
    print(header)
    print("-" * len(header))
    for w, r in reais.items():
        for phase, label in [('disc', 'Descoberta'),
                             ('conf', 'Confirmação'),
                             ('hold', 'Holdout')]:
            ph = r['phases'][phase]
            n_ph = {'disc': r['n_disc'],
                    'conf': r['n_conf'],
                    'hold': r['n_hold']}[phase]
            mh = ph['mae']['Média histórica']
            mm = ph['mae']['Média móvel']
            md = ph['mae']['Mediana móvel']
            ul = ph['mae']['Último valor']
            gk = r['best_key']
            mg = ph['mae'][gk]
            print(f"{w:<5} {label:<11} {n_ph:<5} "
                  f"{mh:<8.2f} {mm:<8.2f} {md:<8.2f} {ul:<8.2f} "
                  f"{r['best_gamma']:<+7.2f} {mg:<8.2f}")
        print("-" * len(header))

    # ---------- 3) Veredito OOS ----------
    print("\n" + "=" * 100)
    print("   VEREDITO OOS — γ* (congelado na descoberta) vs baselines no HOLDOUT")
    print("=" * 100)

    for w, r in reais.items():
        ph = r['phases']['hold']
        gk = r['best_key']
        mae_g  = ph['mae'][gk]
        mae_0  = ph['mae'].get('γ=0.00', ph['mae']['Média móvel'])
        mae_mm = ph['mae']['Média móvel']
        mae_mh = ph['mae']['Média histórica']
        mae_ul = ph['mae']['Último valor']

        def _wil(a, b):
            d = a - b
            if np.all(d == 0):
                return 1.0
            try:
                return wilcoxon(d, alternative='greater').pvalue
            except Exception:
                return 1.0

        p_vs_0  = _wil(ph['abs_err']['γ=0'],           ph['abs_err']['γ*'])
        p_vs_ul = _wil(ph['abs_err']['Último valor'], ph['abs_err']['γ*'])
        delta_oos = mae_0 - mae_g

        print(f"\n   Janela {w:>3}:  N={r['N']}  γ* = {r['best_gamma']:+.2f}")
        print(f"      MAE holdout — γ*: {mae_g:.2f} | γ=0: {mae_0:.2f} | "
              f"MMov: {mae_mm:.2f} | MHist: {mae_mh:.2f} | Últ: {mae_ul:.2f}")
        print(f"      Δ_OOS (γ=0 − γ*): {delta_oos:+.3f}   "
              f"(Wilcoxon γ* < γ=0: p = {p_vs_0:.4f})")
        print(f"      Δ   (Últ − γ*):   {mae_ul - mae_g:+.3f}   "
              f"(Wilcoxon γ* < Últ:  p = {p_vs_ul:.4f})")

        if r['best_gamma'] == 0.0:
            print(f"      → γ* = 0  ⇒  o vencedor é a própria Média Móvel "
                  f"(nenhuma informação além do baseline).")
        elif delta_oos > 0 and p_vs_0 < 0.05:
            print(f"      → ✅ γ* ≠ 0 com vantagem significativa OOS (p < 0.05).")
        elif delta_oos > 0 and p_vs_0 < 0.15:
            print(f"      → 📊 γ* ≠ 0 com vantagem marginal OOS (p < 0.15).")
        else:
            print(f"      → ⚪ Sem evidência de vantagem OOS de γ* sobre γ=0.")

    # ---------- 4) IID MC reproduzindo o procedimento completo ----------
    print("\n" + "=" * 100)
    print(f"   BASELINE IID MONTE CARLO  ({iid_mc} réplicas/janela, procedimento completo)")
    print("=" * 100)
    print("   Para cada réplica sintética: mesma escolha de γ* na descoberta, "
          "mesma avaliação no holdout.")
    print("   A distribuição nula de Δ_OOS e de γ* é diretamente comparável "
          "ao resultado real.\n")

    rng = np.random.default_rng(20250911)

    for w, r_real in reais.items():
        mae0_real = r_real['phases']['hold']['mae'].get(
            'γ=0.00', r_real['phases']['hold']['mae']['Média móvel'])
        maeg_real = r_real['phases']['hold']['mae'][r_real['best_key']]
        delta_oos_real = mae0_real - maeg_real

        delta_oos_sim = []
        gamma_choosen_sim = defaultdict(int)

        for _ in range(iid_mc):
            sim_sums = _iid_sums(n_real, rng)
            sim_r = _gamma_experiment(sim_sums, w, gamma_grid)
            if sim_r is None:
                continue
            ph_hold = sim_r['phases']['hold']
            mae0 = ph_hold['mae'].get('γ=0.00', ph_hold['mae']['Média móvel'])
            maeg = ph_hold['mae'][sim_r['best_key']]
            delta_oos_sim.append(mae0 - maeg)
            gamma_choosen_sim[sim_r['best_gamma']] += 1

        delta_oos_sim = np.asarray(delta_oos_sim, dtype=float)
        if len(delta_oos_sim) == 0:
            continue

        p_emp = float(np.mean(delta_oos_sim >= delta_oos_real))

        top = sorted(gamma_choosen_sim.items(), key=lambda kv: -kv[1])[:5]
        top_str = " | ".join(f"{g:+.2f}:{c}" for g, c in top)

        print(f"   Janela {w:>3}: Δ_OOS real = {delta_oos_real:+.3f} | "
              f"IID: média={np.mean(delta_oos_sim):+.3f} "
              f"sd={np.std(delta_oos_sim):.3f} | p_emp = {p_emp:.4f}")
        print(f"      γ* mais frequentes sob IID: {top_str}")
        if p_emp < 0.05:
            print(f"      → ✅ Δ_OOS real > IID em {1-p_emp:.1%} das réplicas — sinal genuíno.")
        else:
            print(f"      → ⚪ Δ_OOS real compatível com IID (p_emp = {p_emp:.3f} ≥ 0.05).")

    return reais

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA +MILIONÁRIA – v1.3")
    print("   FWER + WALK-FORWARD + FATOR DE ESCALA + γ (corrigido)")
    print("="*70)
    contests = load_all_contests('resultados_maismilionaria.csv')
    if not contests:
        print("❌ Arquivo 'resultados_maismilionaria.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']} | Trevos: {contests[-1]['trevos']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira personalizada")
        print("2. Walk‑forward condicional simples")
        print("3. Backtest nos últimos 200 concursos")
        print("4. Buscar melhores fixas (out‑of‑sample)")
        print("5. Comparar duas trincas")
        print("6. Ranking de poder preditivo (reversão vs tendência)")
        print("7. Poder preditivo das 50 dezenas")
        print("8. Structural Predictor (previsão de faixas)")
        print("9. Gerar carteira com previsões estruturais")
        print("10. Controle Monte Carlo (FWER corrigido)")
        print("11. Teste preditivo concurso a concurso (corrigido)")
        print("12. Walk‑forward do Structural Predictor")
        print("13. Auditoria do Fator de Escala (v1.3, refs independentes)")
        print("14. Auditoria do Coeficiente γ  (v1.3, 60/20/20 + grid + IID)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            fixed_str = input("\n   Dezenas fixas (ex: 10 20 30 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            semifixed_str = input("   Dezenas semifixas (ex: 5 15 25 35 45 ou ENTER): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            excl_str = input("   Dezenas excluídas (ex: 7 13 27 ou ENTER): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            if semifixed:
                try:
                    min_semifixed = int(input(f"   Mínimo de semifixas [0-{len(semifixed)}]: ").strip() or "0")
                    max_semifixed = int(input(f"   Máximo de semifixas [0-{len(semifixed)}]: ").strip() or str(len(semifixed)))
                except:
                    min_semifixed = 0
                    max_semifixed = len(semifixed)
            else:
                min_semifixed = 0
                max_semifixed = None
            print("   Faixas estruturais (ENTER para pular)")
            try:
                pares_str = input("   Pares min,max (ex: 2,4): ").strip()
                range_pares = tuple(int(x) for x in pares_str.split(',')) if pares_str else None
            except: range_pares = None
            try:
                moldura_str = input("   Borda min,max (ex: 1,3): ").strip()
                range_moldura = tuple(int(x) for x in moldura_str.split(',')) if moldura_str else None
            except: range_moldura = None
            try:
                primos_str = input("   Primos min,max (ex: 1,3): ").strip()
                range_primos = tuple(int(x) for x in primos_str.split(',')) if primos_str else None
            except: range_primos = None
            metodo = input("\n   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            opt = PortfolioOptimizer(contests, fixed=fixed, semifixed=semifixed,
                                     min_semifixed=min_semifixed, max_semifixed=max_semifixed,
                                     excluded=excluded, range_pares=range_pares,
                                     range_moldura=range_moldura, range_primos=range_primos)
            portfolio = opt.optimize(5, 50000, method=method)
            for i, (main, trevo) in enumerate(portfolio, 1):
                p = sum(1 for x in main if x % 2 == 0)
                pr = sum(1 for x in main if x in PRIMES)
                m = sum(1 for x in main if x in MOLDURA)
                print(f" {i}. Main: {main} | Trevos: {trevo} | P:{p} Pr:{pr} Borda:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

        elif op == '2':
            fixed_str = input("\n   Fixas (ex: 10 20 30): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            excl_str = input("   Excluídas (ex: 7 13 27 ou ENTER): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            results = []
            for w in range(8):
                test_end = len(contests) - w * 50
                test_start = test_end - 50
                train_end = test_start
                train_start = max(0, train_end - 400)
                if train_start >= train_end or test_start >= test_end: continue
                opt = PortfolioOptimizer(contests[train_start:train_end], fixed=fixed, excluded=excluded)
                portfolio = opt.optimize(5, 10000, method=method)
                bt = opt.backtest(portfolio, contests[test_start:test_end])
                results.append({'lift': bt['lift'], 'roi': bt['roi']})
                print(f"   Janela {w}: lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}%")
            if results:
                print(f"\n📊 Média lift: {np.mean([r['lift'] for r in results]):.3f}")

        elif op == '3':
            fixed_str = input("\n   Fixas (ENTER para pular): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            excl_str = input("   Excluídas (ENTER para pular): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            opt = PortfolioOptimizer(contests, fixed=fixed, excluded=excluded)
            portfolio = opt.optimize(5, 50000, method=method)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Distribuição de prêmios: {bt['hit_distribution']}")

        elif op == '4':
            try:
                n_fixed = int(input("\n   Quantas fixas (2,3,4): ").strip())
                if n_fixed not in [2,3,4]: continue
            except: continue
            top_n = int(input("   Resultados [20]: ").strip() or "20")
            train_size = int(input("   Tamanho treino [3500]: ").strip() or "3500")
            excl_str = input("   Excluídas (ENTER para pular): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            search_best_fixed_oos(contests, n_fixed, top_n, train_size, method=method, excluded=excluded)

        elif op == '5':
            trinca1_str = input("\n   Trinca 1: ").strip()
            trinca2_str = input("   Trinca 2: ").strip()
            excl_str = input("   Excluídas (ENTER para pular): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            try:
                trinca1 = tuple(int(x) for x in trinca1_str.split())
                trinca2 = tuple(int(x) for x in trinca2_str.split())
                if len(trinca1)!=3 or len(trinca2)!=3: continue
            except: continue
            compare_trincas(contests, trinca1, trinca2, excluded=excluded)

        elif op == '6':
            blocos_str = input("\n   Tamanhos de bloco [50,100,200,500]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [50,100,200,500]
            except: block_sizes = [50,100,200,500]
            ranker = PredictiveRanking(contests)
            ranker.rank_predictive_power(block_sizes)

        elif op == '7':
            blocos_str = input("\n   Tamanhos de bloco [100,200,500]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [100,200,500]
            except: block_sizes = [100,200,500]
            ranker = PredictiveRanking(contests)
            ranker.rank_dezenas_individual(block_sizes)

        elif op == '8':
            print("\n   Método: 1. Recente (50 concursos)  2. IPE")
            metodo = input("   Escolha [1]: ").strip() or "1"
            method = 'recent' if metodo == '1' else 'ipe'
            predictor = StructuralPredictor(contests)
            predictor.predict_ranges(method=method)

        elif op == '9':
            predictor = StructuralPredictor(contests)
            ranges = predictor.predict_ranges(method='recent')
            print("\n   Dezenas fixas sugeridas: 10 20 30")
            fixed_str = input("   Digite as dezenas fixas (ENTER p/ sugestão): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else [10, 20, 30]
            print(f"   Fixas utilizadas: {fixed}")
            excl_str = input("   Excluídas (ENTER para pular): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            gerar = input("   Gerar carteira? (s/n): ").strip().lower()
            if gerar == 's':
                metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
                method = 'pair_covering' if metodo == '1' else 'triple_covering'
                opt = PortfolioOptimizer(contests, fixed=fixed, excluded=excluded,
                                         range_pares=ranges.get('pares'),
                                         range_moldura=ranges.get('moldura'),
                                         range_primos=ranges.get('primos'),
                                         range_soma=ranges.get('soma'),
                                         range_amplitude=ranges.get('amplitude'),
                                         range_consecutivos=ranges.get('consecutivos'))
                portfolio = opt.optimize(5, 50000, method=method)
                for i, (main, trevo) in enumerate(portfolio, 1):
                    p = sum(1 for x in main if x%2==0); pr = sum(1 for x in main if x in PRIMES); m = sum(1 for x in main if x in MOLDURA)
                    print(f" {i}. Main: {main} | Trevos: {trevo} | P:{p} Pr:{pr} Borda:{m}")
                if len(contests) > 200:
                    bt = opt.backtest(portfolio, contests[-200:])
                    print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

        elif op == '10':
            try:
                n_sim = int(input("\n   Número de simulações [1000]: ").strip() or "1000")
            except: n_sim = 1000
            blocos_str = input("   Blocos [50,100,200]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [50,100,200]
            except: block_sizes = [50,100,200]
            monte_carlo_control(contests, n_sim, block_sizes)

        elif op == '11':
            try:
                min_hist = int(input("\n   Histórico mínimo [200]: ").strip() or "200")
            except: min_hist = 200
            test_concurso_a_concurso(contests, min_hist)

        elif op == '12':
            try:
                train_size = int(input("\n   Tamanho do treino [500]: ").strip() or "500")
                test_size = int(input("   Tamanho do teste [50]: ").strip() or "50")
                step = int(input("   Passo [50]: ").strip() or "50")
            except:
                train_size, test_size, step = 500, 50, 50
            excl_str = input("   Excluídas (ENTER para pular): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            walk_forward_structural(contests, train_size, test_size, step, excluded=excluded)

        elif op == '13':
            try:
                win = int(input("\n   Janela da média móvel [50]: ").strip() or "50")
            except ValueError:
                win = 50
            audit_scale_factor(contests, window=win)

        elif op == '14':
            jan_str = input(f"\n   Janelas [{' '.join(map(str, WINDOWS_DEFAULT))}]: ").strip()
            try:
                windows = [int(x) for x in jan_str.split()] if jan_str else WINDOWS_DEFAULT
            except:
                windows = WINDOWS_DEFAULT
            gam_str = input(f"   γ grid [{' '.join(map(str, GAMMA_GRID))}]: ").strip()
            try:
                gamma_grid = [float(x) for x in gam_str.split()] if gam_str else GAMMA_GRID
            except:
                gamma_grid = GAMMA_GRID
            try:
                iid_mc = int(input("   Amostras IID MC [1000]: ").strip() or "1000")
            except:
                iid_mc = 1000
            audit_gamma_coefficient(contests, windows=windows, gamma_grid=gamma_grid, iid_mc=iid_mc)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
