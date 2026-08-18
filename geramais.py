#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA +MILIONÁRIA – v1.0
Baseado no v48.3 da Lotofácil, adaptado para:
  - 6 números de 1 a 50
  - 2 trevos de 1 a 6

EVOLUÇÃO:
✅ Correção de leakage
✅ Controle Monte Carlo com correção FWER (máximo global)
✅ Exclusão de dezenas principais
✅ Walk‑forward do Structural Predictor
✅ Filtros estruturais adaptados (pares, borda, primos, soma, amplitude, consecutivos)
"""

import numpy as np
from scipy.stats import hypergeom, binomtest
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
# "Borda" adaptada: 1-10 e 41-50 (equivalente à moldura da Lotofácil)
MOLDURA = set(range(1, 11)) | set(range(41, 51))

# Probabilidades para acertos nas dezenas principais
HYPE_PROBS_MAIN = {k: hypergeom.pmf(k, TOTAL_MAIN, CHOOSE_MAIN, CHOOSE_MAIN) for k in range(0, CHOOSE_MAIN+1)}
# Probabilidades para acertos nos trevos
HYPE_PROBS_TREVO = {k: hypergeom.pmf(k, TOTAL_TREVO, CHOOSE_TREVO, CHOOSE_TREVO) for k in range(0, CHOOSE_TREVO+1)}

# Valores aproximados (R$) para cada faixa de premiação (ajuste conforme necessário)
PREMIO_VALORES = {
    (6,2): 10000000.0,
    (6,1): 50000.0,
    (6,0): 1500.0,
    (5,2): 2000.0,
    (5,1): 100.0,
    (5,0): 10.0,
    (4,2): 50.0,
    (4,1): 5.0,
    (4,0): 0.0,
    (3,2): 3.0,
    (3,1): 0.0,
    (3,0): 0.0,
}
CUSTO_APOSTA = 4.5  # valor médio da aposta simples

# ============================================================
# BITMASK (apenas para dezenas principais)
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
            if len(parts) < 10:  # concurso;data;6 dezenas;2 trevos
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
# GERADOR COM FIXAS, SEMIFIXAS, EXCLUSÕES E FAIXAS
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
                                      excluded,
                                      allowed_pares, allowed_moldura, allowed_primos,
                                      range_pares, range_moldura, range_primos,
                                      range_soma, range_amplitude, range_consecutivos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      excluded,
                      allowed_pares, allowed_moldura, allowed_primos,
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

            # Filtros aplicados apenas às dezenas principais
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

            # Gerar os dois trevos (sem filtros)
            trevos = sorted(random.sample(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO))

            return (main_numbers, trevos)

        return None

    def generate_pure_random(self):
        main = sorted(np.random.choice(range(1, TOTAL_MAIN + 1), CHOOSE_MAIN, replace=False))
        trevos = sorted(np.random.choice(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO, replace=False))
        return (main, trevos)

# ============================================================
# OTIMIZADOR DE CARTEIRA (COBERTURA)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 excluded=None,
                 allowed_pares=None, allowed_moldura=None, allowed_primos=None,
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
                    fixed=self.fixed,
                    semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed,
                    max_semifixed=self.max_semifixed,
                    excluded=self.excluded,
                    allowed_pares=self.allowed_pares,
                    allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares,
                    range_moldura=self.range_moldura,
                    range_primos=self.range_primos,
                    range_soma=self.range_soma,
                    range_amplitude=self.range_amplitude,
                    range_consecutivos=self.range_consecutivos
                )
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
                main = c[0]
                groups = set(combinations(sorted(main), r))
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

                # Considera apenas combinações com prêmio definido
                premio = PREMIO_VALORES.get((main_hits, trevo_hits), 0.0)
                if premio > 0:
                    n_success += 1
                    total_premio += premio
                    hit_counts[(main_hits, trevo_hits)] += 1

        prob = n_success / (len(portfolio) * len(test_draws)) if test_draws else 0
        # Probabilidade teórica de pelo menos um prêmio em uma aposta simples
        p_single = 0.0
        for (m,t), _ in PREMIO_VALORES.items():
            p_single += HYPE_PROBS_MAIN[m] * HYPE_PROBS_TREVO[t]
        theo_prob = 1 - (1 - p_single)**len(portfolio)

        return {
            'empirical': prob,
            'theoretical': theo_prob,
            'lift': prob / theo_prob if theo_prob > 0 else 1.0,
            'n_test': len(test_draws),
            'n_success': n_success,
            'total_premio': total_premio,
            'total_custo': total_custo,
            'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0,
            'hit_distribution': dict(hit_counts)
        }

# ============================================================
# FUNÇÕES DE EXTRAÇÃO DE FILTROS (apenas dezenas principais)
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
# RANKING DE PODER PREDITIVO
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
# CONTROLE MONTE CARLO COM CORREÇÃO FWER
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

        for (filtro, strategy, block_size), res in sim_results.items():
            key = (filtro, strategy, block_size)
            sim_accuracies_by_key[key].append(res['accuracy'])

    p_global = np.mean(np.array(sim_max_accs) >= real_max_acc)
    if p_global == 0.0:
        p_global_str = f"<{1.0/n_simulations:.4f}"
    else:
        p_global_str = f"{p_global:.4f}"

    print(f"\n📊 COMPARAÇÃO REAL vs. MONTE CARLO (resultados individuais)")
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
        if p_emp == 0.0:
            p_emp_str = f"<{1.0/len(sim_accs):.4f}"
        else:
            p_emp_str = f"{p_emp:.4f}"
        marker = " 🏆" if key == real_max_key else ""
        print(f"{filtro:<15} {strategy:<12} {block_size:<8} {real_acc:<10.1f}% {mean_sim:<10.1f}% {std_sim:<10.1f} {diff:<10.1f}% {p_emp_str:<10}{marker}")

    print(f"\n🌟 Melhor resultado real: {real_max_key} com acurácia de {real_max_acc:.1f}%")
    print(f"   p‑global (corrigido para múltiplas comparações): {p_global_str}")
    if p_global < 0.05:
        print("   ✅ Evidência de sinal preditivo genuíno (p < 0.05)")
    else:
        print("   ❌ Não há evidência significativa — resultado compatível com aleatoriedade")

    return real_results, sim_accuracies_by_key

# ============================================================
# TESTE CONCURSO A CONCURSO (CORRIGIDO)
# ============================================================
def test_concurso_a_concurso(contests, min_history=200):
    print(f"\n🎯 TESTE CONCURSO A CONCURSO (vazamento corrigido)")
    print(f"   Histórico mínimo: {min_history}")
    print(f"   Testes: até {len(contests) - min_history - 1} previsões\n")

    filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']

    for filtro in filters:
        series = np.array([extract_filter(c['dezenas'], filtro) for c in contests], dtype=float)

        acertos_reversao = 0
        acertos_tendencia = 0
        total = 0

        for t in range(min_history, len(contests) - 1):
            current_val = series[t]
            next_val = series[t+1]
            history = series[:t]

            if len(history) < 20:
                continue

            mean_short = np.mean(history[-20:])
            mean_long = np.mean(history)

            total += 1

            if mean_short > mean_long:
                pred_rev_down = True
            else:
                pred_rev_down = False

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
# WALK‑FORWARD DO STRUCTURAL PREDICTOR
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

        opt = PortfolioOptimizer(train_data,
                                 excluded=excluded,
                                 range_pares=ranges.get('pares'),
                                 range_moldura=ranges.get('moldura'),
                                 range_primos=ranges.get('primos'),
                                 range_soma=ranges.get('soma'),
                                 range_amplitude=ranges.get('amplitude'),
                                 range_consecutivos=ranges.get('consecutivos'))

        try:
            portfolio = opt.optimize(5, 10000, method='pair_covering')
            bt = opt.backtest(portfolio, test_data)

            results.append({
                'start': start,
                'lift': bt['lift'],
                'roi': bt['roi'],
                'premios': sum(bt['hit_distribution'].values()),
            })
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
        # Frequência mínima ajustada para o universo 50/6
        if freq >= 0.001:
            candidates.append((fixed_tuple, freq, acertos))

    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for fixed_tuple, freq, acertos in tqdm(candidates[:200], desc="Backtest OOS"):
        opt = PortfolioOptimizer(train_data, fixed=list(fixed_tuple), excluded=excluded)
        try:
            portfolio = opt.optimize(n_games, n_candidates, method=method)
            bt = opt.backtest(portfolio, test_data)
            results.append({
                'fixed': fixed_tuple,
                'freq_treino': freq,
                'lift': bt['lift'],
                'roi': bt['roi'],
                'premios': sum(bt['hit_distribution'].values()),
            })
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
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA +MILIONÁRIA – v1.0")
    print("   MONTE CARLO CORRIGIDO + WALK‑FORWARD ESTRUTURAL + EXCLUSÃO")
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

            opt = PortfolioOptimizer(
                contests,
                fixed=fixed,
                semifixed=semifixed,
                min_semifixed=min_semifixed,
                max_semifixed=max_semifixed,
                excluded=excluded,
                range_pares=range_pares,
                range_moldura=range_moldura,
                range_primos=range_primos)
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
            blocos_str = input("\n   Tamanhos de bloco (ex: 50,100,200,500) [50,100,200,500]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [50,100,200,500]
            except: block_sizes = [50,100,200,500]
            ranker = PredictiveRanking(contests)
            ranker.rank_predictive_power(block_sizes)

        elif op == '7':
            blocos_str = input("\n   Tamanhos de bloco (ex: 100,200,500) [100,200,500]: ").strip()
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
            fixed_str = input("   Digite as dezenas fixas (ex: 10 20 30) ou ENTER para usar a sugestão: ").strip()
            if fixed_str:
                fixed = [int(x) for x in fixed_str.split()]
            else:
                fixed = [10, 20, 30]   # sugestão padrão
            print(f"   Fixas utilizadas: {fixed}")
            excl_str = input("   Excluídas (ENTER para pular): ").strip()
            excluded = [int(x) for x in excl_str.split()] if excl_str else []
            gerar = input("   Gerar carteira? (s/n): ").strip().lower()
            if gerar == 's':
                metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
                method = 'pair_covering' if metodo == '1' else 'triple_covering'
                opt = PortfolioOptimizer(contests, fixed=fixed,
                                         excluded=excluded,
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
            blocos_str = input("   Blocos (ex: 50,100,200) [50,100,200]: ").strip()
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

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
