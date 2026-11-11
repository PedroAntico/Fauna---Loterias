#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA +MILIONÁRIA – v1.5
MÓDULOS 1–5: Gerador validado + Carteiras sem duplicata + Backtest corrigido
             + Estatística com MC + Auditoria de integridade.

EVOLUÇÃO v1.5 (correções cirúrgicas sobre v1.4):
✅ Backtest: p_bet vs p_draw separados; lift usa MC (não independência)
✅ Prêmios: tabela oficial (R$ 6,00; 13 combinações; fixos 3+2/3+1/2+2/2+1)
✅ Semifixas: sempre validadas; _validate_config rejeita impossíveis
✅ Entradas: validação estrita (1–50, sem repetidos, sem conflitos)
✅ Covering: selected_mains evita duplicata; avisa se <n_select
✅ Covering: coverage_stats reportado
✅ γ: 'Média móvel' como γ=0 (sem KeyError); BH entre janelas
✅ Ranking: block permutation opcional
✅ MC FWER: (1+Σ)/(n_sim+1)
✅ Structural IPE: clamp dentro dos limites
✅ Módulo 5: integrity_audit() com 7 verificações automáticas
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

# --- TABELA OFICIAL ---
# 13 combinações premiadas. 3+0 e 2+0 NÃO pagam.
# Valores variáveis marcados como None (não fabricar prêmio).
FAIXAS_PREMIADAS = {
    (6, 2): None, (6, 1): None, (6, 0): None,
    (5, 2): None, (5, 1): None, (5, 0): None,
    (4, 2): None, (4, 1): None, (4, 0): None,
    (3, 2): 50.0, (3, 1): 24.0,
    (2, 2): 12.0, (2, 1):  6.0,
}
FAIXAS_PREMIADAS_SET = set(FAIXAS_PREMIADAS.keys())
CUSTO_APOSTA = 6.0   # CORRIGIDO: R$ 6,00 (v1.4 usava 4.5)

# Probabilidade matemática de uma aposta premiar (qualquer faixa)
P_BET_WIN_THEORETICAL = sum(
    HYPE_PROBS_MAIN[m] * HYPE_PROBS_TREVO[t] for (m, t) in FAIXAS_PREMIADAS
)

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
# VALIDAÇÃO DE ENTRADAS
# ============================================================
def validate_numbers(numbers, name, max_val=TOTAL_MAIN):
    if numbers is None:
        return []
    if not isinstance(numbers, (list, tuple)):
        raise ValueError(f"{name}: deve ser lista ou tupla.")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{name}: valores repetidos: {sorted(numbers)}")
    for x in numbers:
        if not isinstance(x, int) or x < 1 or x > max_val:
            raise ValueError(f"{name}: valores devem ser inteiros 1–{max_val}. Recebido: {x!r}")
    return sorted(numbers)


def validate_inputs(fixed=None, semifixed=None, excluded=None,
                    min_semifixed=0, max_semifixed=None):
    fixed = validate_numbers(fixed, 'fixas', TOTAL_MAIN)
    semifixed = validate_numbers(semifixed, 'semifixas', TOTAL_MAIN)
    excluded = validate_numbers(excluded, 'excluídas', TOTAL_MAIN)

    if len(fixed) > CHOOSE_MAIN:
        raise ValueError(f"Fixas: máximo {CHOOSE_MAIN}, recebido {len(fixed)}.")

    fs, ss, es = set(fixed), set(semifixed), set(excluded)
    if fs & ss:
        raise ValueError(f"Conflito fixas/semifixas: {sorted(fs & ss)}")
    if fs & es:
        raise ValueError(f"Conflito fixas/excluídas: {sorted(fs & es)}")
    if ss & es:
        raise ValueError(f"Conflito semifixas/excluídas: {sorted(ss & es)}")

    if min_semifixed < 0:
        raise ValueError(f"min_semifixed < 0: {min_semifixed}")
    if max_semifixed is not None:
        if max_semifixed < 0:
            raise ValueError(f"max_semifixed < 0: {max_semifixed}")
        if max_semifixed < min_semifixed:
            raise ValueError(
                f"max_semifixed ({max_semifixed}) < min_semifixed ({min_semifixed})"
            )

    return fixed, semifixed, excluded

# ============================================================
# CARREGAMENTO
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
                if (len(dezenas) != CHOOSE_MAIN or len(set(dezenas)) != CHOOSE_MAIN
                    or any(x < 1 or x > TOTAL_MAIN for x in dezenas)):
                    continue
                if (len(trevos) != CHOOSE_TREVO or len(set(trevos)) != CHOOSE_TREVO
                    or any(x < 1 or x > TOTAL_TREVO for x in trevos)):
                    continue
                contests.append({
                    'concurso': int(parts[0]),
                    'data': parts[1],
                    'dezenas': sorted(dezenas),
                    'trevos': sorted(trevos),
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
    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                     excluded=None, allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                     range_pares=None, range_moldura=None, range_primos=None,
                     range_soma=None, range_amplitude=None, range_consecutivos=None):
        for _ in range(500):
            g = self._generate_raw(
                fixed, semifixed, min_semifixed, max_semifixed, excluded,
                allowed_pares, allowed_moldura, allowed_primos,
                range_pares, range_moldura, range_primos,
                range_soma, range_amplitude, range_consecutivos,
            )
            if g is not None:
                return g
        raise RuntimeError("Geração falhou com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      excluded, allowed_pares, allowed_moldura, allowed_primos,
                      range_pares, range_moldura, range_primos,
                      range_soma, range_amplitude, range_consecutivos):
        fixed = fixed or []
        semifixed = semifixed or []
        excluded = excluded or []

        excl_set = set(excluded)
        fixed_set = set(fixed) - excl_set
        semi_set = set(semifixed) - fixed_set - excl_set
        banned = fixed_set | semi_set | excl_set
        rest = list(set(range(1, TOTAL_MAIN + 1)) - banned)
        n_fix = len(fixed_set)

        max_semi = len(semi_set) if max_semifixed is None else min(max_semifixed, len(semi_set))
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None

        n_semi = random.randint(min_semi, max_semi)
        n_rest = CHOOSE_MAIN - n_fix - n_semi
        if n_rest < 0 or n_rest > len(rest):
            return None

        for _ in range(200):
            semi_pick = set(random.sample(list(semi_set), n_semi)) if n_semi > 0 else set()
            rest_pick = set(random.sample(rest, n_rest)) if n_rest > 0 else set()
            main = sorted(fixed_set | semi_pick | rest_pick)
            if len(main) != CHOOSE_MAIN:
                continue

            if allowed_pares is not None and sum(1 for x in main if x % 2 == 0) not in allowed_pares:
                continue
            if allowed_moldura is not None and sum(1 for x in main if x in MOLDURA) not in allowed_moldura:
                continue
            if allowed_primos is not None and sum(1 for x in main if x in PRIMES) not in allowed_primos:
                continue

            p = sum(1 for x in main if x % 2 == 0)
            m = sum(1 for x in main if x in MOLDURA)
            pr = sum(1 for x in main if x in PRIMES)
            s = sum(main)
            a = max(main) - min(main)
            c = sum(1 for i in range(len(main)-1) if main[i+1]-main[i] == 1)

            if range_pares is not None and not (range_pares[0] <= p <= range_pares[1]):
                continue
            if range_moldura is not None and not (range_moldura[0] <= m <= range_moldura[1]):
                continue
            if range_primos is not None and not (range_primos[0] <= pr <= range_primos[1]):
                continue
            if range_soma is not None and not (range_soma[0] <= s <= range_soma[1]):
                continue
            if range_amplitude is not None and not (range_amplitude[0] <= a <= range_amplitude[1]):
                continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= c <= range_consecutivos[1]):
                continue

            trevos = sorted(random.sample(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO))
            return (main, trevos)
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
        fixed, semifixed, excluded = validate_inputs(
            fixed, semifixed, excluded, min_semifixed, max_semifixed
        )
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed
        self.semifixed = semifixed
        self.excluded = excluded
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
        self._validate_config()

    def _validate_config(self):
        """Rejeita configurações impossíveis antes de gerar."""
        semi_set = set(self.semifixed) - set(self.fixed) - set(self.excluded)
        if self.min_semifixed > len(semi_set):
            raise ValueError(
                f"Semifixas efetivas ({len(semi_set)}) < mínimo exigido "
                f"({self.min_semifixed}). Impossível satisfazer."
            )
        n_free = CHOOSE_MAIN - len(self.fixed)
        available = TOTAL_MAIN - len(self.fixed) - len(self.excluded)
        if n_free < 0:
            raise ValueError(f"Fixas ({len(self.fixed)}) > {CHOOSE_MAIN}.")
        if n_free > available:
            raise ValueError(
                f"Livres necessárias ({n_free}) > disponíveis ({available})."
            )

    def _effective_universe_main(self):
        fixed_set = set(self.fixed)
        excl_set = set(self.excluded)
        semi_set = set(self.semifixed) - fixed_set - excl_set
        free = set(range(1, TOTAL_MAIN + 1)) - fixed_set - excl_set
        non_semi_free = free - semi_set
        s = len(semi_set)
        f = len(non_semi_free)
        n_free = CHOOSE_MAIN - len(fixed_set)
        if n_free < 0:
            return 0
        min_k = max(0, self.min_semifixed)
        max_k = s if self.max_semifixed is None else min(self.max_semifixed, s)
        lo = max(min_k, n_free - f, 0)
        hi = min(max_k, n_free, s)
        if lo > hi:
            return 0
        return sum(comb(s, k) * comb(f, n_free - k) for k in range(lo, hi + 1))

    def generate_pool(self, n_candidates):
        pool, seen = [], set()
        attempts = 0
        max_attempts = n_candidates * 20
        while len(pool) < n_candidates and attempts < max_attempts:
            attempts += 1
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed,
                    excluded=self.excluded,
                    allowed_pares=self.allowed_pares, allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares, range_moldura=self.range_moldura,
                    range_primos=self.range_primos, range_soma=self.range_soma,
                    range_amplitude=self.range_amplitude,
                    range_consecutivos=self.range_consecutivos,
                )
            except RuntimeError:
                continue
            key = (tuple(g[0]), tuple(g[1]))
            if key in seen:
                continue
            seen.add(key)
            pool.append(g)
        return pool

    def enumerate_and_sample(self, n_sample, seed=None):
        """
        Enumera universe_main_eff mains. Para cada main válido, percorre
        15 pares de trevos. Reservoir sampling sobre (main, trevo).
        Memória: O(k), k = n_sample.
        """
        rng = np.random.default_rng(seed)
        fixed_set = set(self.fixed)
        excl_set = set(self.excluded)
        semi_set = set(self.semifixed) - fixed_set - excl_set
        free = sorted(set(range(1, TOTAL_MAIN + 1)) - fixed_set - excl_set)
        n_free = CHOOSE_MAIN - len(fixed_set)

        if n_free < 0 or n_free > len(free):
            return [], {'error': 'infeasible'}

        all_trevo_combos = list(combinations(range(1, TOTAL_TREVO + 1), CHOOSE_TREVO))
        n_trevo = len(all_trevo_combos)

        universe_main_raw  = comb(TOTAL_MAIN, CHOOSE_MAIN)
        universe_main_eff  = comb(len(free), n_free)
        universe_main_semi = self._effective_universe_main()

        universe_pairs_raw  = universe_main_raw  * n_trevo
        universe_pairs_eff  = universe_main_eff  * n_trevo
        universe_pairs_semi = universe_main_semi * n_trevo

        min_semi = max(0, self.min_semifixed)
        max_semi = len(semi_set) if self.max_semifixed is None else min(self.max_semifixed, len(semi_set))

        print(f"\n🔎 ENUMERAÇÃO DO UNIVERSO (+Milionária)")
        print(f"   Fixas={sorted(fixed_set)} Excluídas={sorted(excl_set)}")
        print(f"   Semifixas disponíveis={sorted(semi_set)} (min={min_semi}, max={max_semi})")
        print(f"   Universo bruto mains C(50,6):           {universe_main_raw:>18,}")
        print(f"   Efetivas mains (após fixas/excl):       {universe_main_eff:>18,}")
        print(f"   Efetivas mains (após semifixas):        {universe_main_semi:>18,}")
        print(f"   Efetivas (main, trevo):                 {universe_pairs_semi:>18,}")

        n_valid_main = n_valid = 0
        reservoir = []
        rej = {k: 0 for k in [
            'semifixas', 'pares_allowed', 'moldura_allowed', 'primos_allowed',
            'pares_range', 'moldura_range', 'primos_range',
            'soma', 'amplitude', 'consecutivos',
        ]}
        fixed_tuple = tuple(sorted(fixed_set)) if fixed_set else None

        t0 = time.time()
        for combo_free in tqdm(
            combinations(free, n_free),
            total=universe_main_eff, desc="Enumeração", unit=" main",
            mininterval=0.5, miniters=10000,
        ):
            main = tuple(sorted(fixed_tuple + combo_free)) if fixed_tuple else combo_free

            # SEMIFIXAS: sempre verificado (mesmo se semi_set vazio → n=0)
            main_set = set(main)
            n_semi_in = len(main_set & semi_set)
            if n_semi_in < min_semi or n_semi_in > max_semi:
                rej['semifixas'] += 1
                continue

            p = m = pr = s = 0
            prev = main[0]; c = 0
            for i, x in enumerate(main):
                if x % 2 == 0: p += 1
                if x in MOLDURA: m += 1
                if x in PRIMES: pr += 1
                s += x
                if i > 0 and x - prev == 1: c += 1
                prev = x
            a = main[-1] - main[0]

            if self.allowed_pares is not None and p not in self.allowed_pares:
                rej['pares_allowed'] += 1; continue
            if self.allowed_moldura is not None and m not in self.allowed_moldura:
                rej['moldura_allowed'] += 1; continue
            if self.allowed_primos is not None and pr not in self.allowed_primos:
                rej['primos_allowed'] += 1; continue

            if self.range_pares is not None and not (self.range_pares[0] <= p <= self.range_pares[1]):
                rej['pares_range'] += 1; continue
            if self.range_moldura is not None and not (self.range_moldura[0] <= m <= self.range_moldura[1]):
                rej['moldura_range'] += 1; continue
            if self.range_primos is not None and not (self.range_primos[0] <= pr <= self.range_primos[1]):
                rej['primos_range'] += 1; continue
            if self.range_soma is not None and not (self.range_soma[0] <= s <= self.range_soma[1]):
                rej['soma'] += 1; continue
            if self.range_amplitude is not None and not (self.range_amplitude[0] <= a <= self.range_amplitude[1]):
                rej['amplitude'] += 1; continue
            if self.range_consecutivos is not None and not (self.range_consecutivos[0] <= c <= self.range_consecutivos[1]):
                rej['consecutivos'] += 1; continue

            n_valid_main += 1
            main_list = list(main)
            for trevo in all_trevo_combos:
                n_valid += 1
                if len(reservoir) < n_sample:
                    reservoir.append((main_list.copy(), list(trevo)))
                else:
                    j = int(rng.integers(0, n_valid))
                    if j < n_sample:
                        reservoir[j] = (main_list.copy(), list(trevo))

        elapsed = time.time() - t0
        has_filters = any([
            self.allowed_pares is not None, self.allowed_moldura is not None,
            self.allowed_primos is not None,
            self.range_pares is not None, self.range_moldura is not None,
            self.range_primos is not None, self.range_soma is not None,
            self.range_amplitude is not None, self.range_consecutivos is not None,
        ])
        if not has_filters and n_valid_main != universe_main_semi:
            print(f"\n⚠️ INCONSISTÊNCIA: n_valid_main={n_valid_main:,} ≠ semi={universe_main_semi:,}")

        stats = {
            'universe_main_raw': universe_main_raw,
            'universe_main_eff': universe_main_eff,
            'universe_main_semi': universe_main_semi,
            'universe_pairs_raw': universe_pairs_raw,
            'universe_pairs_eff': universe_pairs_eff,
            'universe_pairs_semi': universe_pairs_semi,
            'n_valid_main': n_valid_main,
            'n_valid': n_valid,
            'n_trevo_combos': n_trevo,
            'requested': n_sample,
            'obtained': len(reservoir),
            'coverage_valid_over_eff_semi': (n_valid / universe_pairs_semi) if universe_pairs_semi else 0.0,
            'coverage_sample_over_valid': (len(reservoir) / n_valid) if n_valid else 0.0,
            'rejections': rej,
            'elapsed': elapsed,
        }
        return reservoir, stats

    def select_covering(self, candidates, n_select, level='pair'):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")

        MAX_COVER = 50000 if level == 'pair' else 20000
        total_available = len(candidates)
        print(f"\n🔧 COVERING ({level})")
        print(f"   Entrada: {total_available:,}")
        if total_available > MAX_COVER:
            print(f"   MAX_COVER: {MAX_COVER:,} → subamostra")
            working = random.sample(candidates, MAX_COVER)
        else:
            working = list(candidates)

        t0 = time.time()
        r = 2 if level == 'pair' else 3
        covered = set()
        selected = []
        selected_mains = set()
        used_indices = set()

        for _ in range(n_select):
            best_idx, best_gain = -1, -1
            for i, c in enumerate(working):
                if i in used_indices:
                    continue
                main_key = tuple(sorted(c[0]))
                if main_key in selected_mains:
                    continue
                groups = set(combinations(main_key, r))
                new = len(groups - covered)
                if new > best_gain:
                    best_gain = new
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(working[best_idx])
            selected_mains.add(tuple(sorted(working[best_idx][0])))
            covered.update(combinations(sorted(working[best_idx][0]), r))
            used_indices.add(best_idx)

        print(f"   Seleção: {time.time()-t0:.2f}s")
        if len(selected) < n_select:
            print(f"   ⚠️ Covering retornou {len(selected)}/{n_select} "
                  f"(mains distintos insuficientes)")
        return selected

    def coverage_stats(self, portfolio):
        pairs, triples = set(), set()
        for main, _ in portfolio:
            m = sorted(main)
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    pairs.add((m[i], m[j]))
                    for k in range(j + 1, len(m)):
                        triples.add((m[i], m[j], m[k]))
        return {
            'n_pairs': len(pairs), 'total_pairs': comb(TOTAL_MAIN, 2),
            'n_triples': len(triples), 'total_triples': comb(TOTAL_MAIN, 3),
        }

    def optimize(self, n_games=5, n_candidates=50000, method='pair_covering',
                 exact_universe=False, seed=None):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | {method} | pool={n_candidates:,}")
        print(f"   Modo: {'ENUMERAÇÃO EXATA' if exact_universe else 'TENTATIVAS'}")
        if self.fixed:     print(f"   Fixas: {self.fixed}")
        if self.semifixed: print(f"   Semifixas: {self.semifixed} (min={self.min_semifixed}, max={self.max_semifixed})")
        if self.excluded:  print(f"   Excluídas: {self.excluded}")

        t0 = time.time()
        if exact_universe:
            pool, stats = self.enumerate_and_sample(n_candidates, seed=seed)
            if not pool:
                raise RuntimeError(f"Enumeração falhou: {stats}")

            print(f"\n📊 UNIVERSO E POOL")
            print(f"   Bruto mains:                       {stats['universe_main_raw']:>18,}")
            print(f"   Efetivas mains (fixas/excl):       {stats['universe_main_eff']:>18,}")
            print(f"   Efetivas mains (c/ semifixas):     {stats['universe_main_semi']:>18,}")
            print(f"   Efetivas (main, trevo):            {stats['universe_pairs_semi']:>18,}")
            print(f"   Válidas mains (pós-filtros):       {stats['n_valid_main']:>18,}")
            print(f"   Válidas (main, trevo):             {stats['n_valid']:>18,}")
            print(f"   Pool solicitado / obtido:          {stats['requested']:>18,} / {stats['obtained']:,}")
            print(f"   Cobertura sobre válidos:           {stats['coverage_sample_over_valid']*100:.2f}%")

            print(f"\n📊 REJEIÇÕES (ordem de avaliação; cada main creditado só ao 1º filtro)")
            total_rej = sum(stats['rejections'].values())
            for k, v in stats['rejections'].items():
                if v > 0: print(f"   {k:<18} {v:>18,}")
            print(f"   {'Total rejeitado':<18} {total_rej:>18,}")
            print(f"   {'Aprovado':<18} {stats['n_valid_main']:>18,} mains "
                  f"(× {stats['n_trevo_combos']} trevos = {stats['n_valid']:,} pares)")
        else:
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

        cov = self.coverage_stats(portfolio)
        print(f"   Cobertura: {cov['n_pairs']}/{cov['total_pairs']} pares, "
              f"{cov['n_triples']}/{cov['total_triples']} trincas")
        print(f"✅ {len(portfolio)} jogos em {time.time()-t0:.1f}s")
        return portfolio

    def _estimate_p_draw_mc(self, portfolio, n_sim=10000, seed=42):
        """P(≥1 jogo premiado em um sorteio) via MC usando a carteira real."""
        rng = np.random.default_rng(seed)
        pm_list = [set(m) for m, _ in portfolio]
        pt_list = [set(t) for _, t in portfolio]
        wins = 0
        for _ in range(n_sim):
            main = set(int(x) for x in rng.choice(np.arange(1, TOTAL_MAIN+1), CHOOSE_MAIN, replace=False))
            trevo = set(int(x) for x in rng.choice(np.arange(1, TOTAL_TREVO+1), CHOOSE_TREVO, replace=False))
            for pm, pt in zip(pm_list, pt_list):
                if (len(pm & main), len(pt & trevo)) in FAIXAS_PREMIADAS_SET:
                    wins += 1
                    break
        return wins / n_sim

    def backtest(self, portfolio, test_draws):
        """
        Backtest corrigido: separa métricas por aposta (p_bet) e por concurso (p_draw).
        Lift usa p_draw vs MC teórico (captura dependência entre jogos da carteira).
        A fórmula de independência é reportada apenas como REFERÊNCIA.
        Prêmios: só os fixos (R$ 50/24/12/6). Variáveis NÃO fabricados.
        """
        bets_won = 0
        draws_with_win = 0
        total_premio_fixo = 0.0
        premio_variavel_count = 0
        hit_counts = defaultdict(int)

        for draw in test_draws:
            dm = set(draw['dezenas']); dt = set(draw['trevos'])
            draw_had_win = False
            for main, trevo in portfolio:
                mh = len(dm & set(main))
                th = len(dt & set(trevo))
                key = (mh, th)
                if key in FAIXAS_PREMIADAS_SET:
                    bets_won += 1
                    hit_counts[key] += 1
                    draw_had_win = True
                    v = FAIXAS_PREMIADAS[key]
                    if v is not None:
                        total_premio_fixo += v
                    else:
                        premio_variavel_count += 1
            if draw_had_win:
                draws_with_win += 1

        n_bets_total = len(portfolio) * len(test_draws) if test_draws else 0
        p_bet_emp = bets_won / n_bets_total if n_bets_total else 0.0
        p_draw_emp = draws_with_win / len(test_draws) if test_draws else 0.0
        p_draw_mc = self._estimate_p_draw_mc(portfolio, n_sim=10000, seed=42)
        p_draw_indep = 1 - (1 - P_BET_WIN_THEORETICAL) ** len(portfolio)
        custo_total = n_bets_total * CUSTO_APOSTA

        return {
            # Por aposta
            'p_bet_empirical': p_bet_emp,
            'p_bet_theoretical': P_BET_WIN_THEORETICAL,
            'lift_bet': p_bet_emp / P_BET_WIN_THEORETICAL if P_BET_WIN_THEORETICAL > 0 else 1.0,
            'bets_won': bets_won,
            # Por concurso
            'p_draw_empirical': p_draw_emp,
            'p_draw_theoretical_mc': p_draw_mc,
            'p_draw_indep_reference': p_draw_indep,  # só referência
            'lift_draw': p_draw_emp / p_draw_mc if p_draw_mc > 0 else 1.0,
            'draws_with_win': draws_with_win,
            # Totais
            'n_test': len(test_draws),
            'n_bets_total': n_bets_total,
            'custo_total': custo_total,
            'premio_fixo_total': total_premio_fixo,
            'premio_variavel_count': premio_variavel_count,
            'roi_fixo_apenas': (
                (total_premio_fixo - custo_total) / custo_total * 100 if custo_total > 0 else 0.0
            ),
            'hit_distribution': dict(hit_counts),
        }

# ============================================================
# EXTRAÇÃO DE FILTROS
# ============================================================
def extract_filter(dezenas, filter_name):
    d = sorted(dezenas)
    if filter_name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if filter_name == 'moldura': return sum(1 for x in d if x in MOLDURA)
    if filter_name == 'primos': return sum(1 for x in d if x in PRIMES)
    if filter_name == 'soma': return sum(d)
    if filter_name == 'consecutivos': return sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    if filter_name == 'amplitude': return max(d) - min(d)
    return 0

# ============================================================
# RANKING PREDITIVO (com block permutation opcional)
# ============================================================
class PredictiveRanking:
    def __init__(self, contests):
        self.contests = contests

    def _accuracy(self, series, block_size, strategy):
        n_blocos = len(series) // block_size
        if n_blocos < 3:
            return None, 0
        blocos = [series[i*block_size:(i+1)*block_size] for i in range(n_blocos)]
        ac = tot = 0
        for i in range(2, len(blocos)):
            mp = np.mean(blocos[i-1]); mc = np.mean(blocos[i])
            mh = np.mean(series[:(i-1)*block_size]) if (i-1)*block_size > 0 else mp
            tot += 1
            if strategy == 'reversao':
                d = mp > mh
                if (d and mc < mp) or (not d and mc > mp): ac += 1
            else:
                if (mp > mh and mc > mp) or (mp < mh and mc < mp): ac += 1
        return (ac / tot * 100 if tot else 0.0), tot, ac

    def _block_permutation(self, series, block_size, strategy, n_perm=200, seed=42):
        rng = np.random.default_rng(seed)
        n_blocos = len(series) // block_size
        if n_blocos < 3:
            return None
        blocks = [series[i*block_size:(i+1)*block_size].copy() for i in range(n_blocos)]

        obs_acc, _, _ = self._accuracy(series, block_size, strategy)
        perm_accs = []
        for _ in range(n_perm):
            idx = rng.permutation(n_blocos)
            perm_series = np.concatenate([blocks[i] for i in idx])
            a, _, _ = self._accuracy(perm_series, block_size, strategy)
            perm_accs.append(a)
        perm_accs = np.array(perm_accs)
        p = (1 + np.sum(perm_accs >= obs_acc)) / (n_perm + 1)
        return p

    def rank_predictive_power(self, block_sizes=None, verbose=True, block_permutations=0):
        if block_sizes is None: block_sizes = [50, 100, 200, 500]
        filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']
        out = {}
        for bs in block_sizes:
            if verbose:
                print(f"\n📊 BLOCOS DE {bs} CONCURSOS")
                hdr = f"{'Filtro':<15} {'Estrat':<10} {'Prec':<8} {'Acertos':<10} {'p_binom':<10}"
                if block_permutations > 0: hdr += f" {'p_perm':<10}"
                print(hdr)
                print("-" * len(hdr))
            for f in filters:
                series = np.array([extract_filter(c['dezenas'], f) for c in self.contests], dtype=float)
                for strat in ['reversao', 'tendencia']:
                    acc, tot, ac = self._accuracy(series, bs, strat)
                    if acc is None: continue
                    p = binomtest(ac, tot, 0.5, alternative='greater').pvalue if tot else 1.0
                    entry = {'accuracy': acc, 'acertos': ac, 'total': tot, 'p_value': p}
                    if block_permutations > 0:
                        p_perm = self._block_permutation(series, bs, strat, n_perm=block_permutations)
                        entry['p_permutation'] = p_perm
                    out[(f, strat, bs)] = entry
                    if verbose:
                        sig = "🔍" if p < 0.05 else ("📊" if p < 0.15 else "  ")
                        line = f"{f:<15} {strat:<10} {acc:<8.1f} {ac}/{tot:<8} {p:<10.4f}{sig}"
                        if block_permutations > 0:
                            line += f" {p_perm:<10.4f}"
                        print(line)
        return out

    def rank_dezenas_individual(self, block_sizes=None):
        if block_sizes is None: block_sizes = [100, 200, 500]
        print(f"\n📊 PODER PREDITIVO DAS {TOTAL_MAIN} DEZENAS")
        for bs in block_sizes:
            accs = []
            for dz in range(1, TOTAL_MAIN + 1):
                series = np.array([1 if dz in c['dezenas'] else 0 for c in self.contests], dtype=float)
                acc, tot, _ = self._accuracy(series, bs, 'reversao')
                if acc is not None: accs.append(acc)
            if accs:
                print(f"   Bloco {bs}: média={np.mean(accs):.1f}%, "
                      f"max={max(accs):.1f}%, min={min(accs):.1f}%")

# ============================================================
# MC FWER (+1 corrigido)
# ============================================================
def monte_carlo_control(contests, n_simulations=1000, block_sizes=None):
    if block_sizes is None: block_sizes = [50, 100, 200]
    n_concursos = len(contests)
    print(f"\n🎲 CONTROLE MONTE CARLO (FWER corrigido)")
    ranker_real = PredictiveRanking(contests)
    real_results = ranker_real.rank_predictive_power(block_sizes, verbose=False)
    real_max = max(r['accuracy'] for r in real_results.values())
    real_key = max(real_results, key=lambda k: real_results[k]['accuracy'])

    sim_max = []
    sim_by = defaultdict(list)
    for _ in tqdm(range(n_simulations), desc="MC"):
        sim = [{
            'dezenas': sorted(np.random.choice(range(1, TOTAL_MAIN+1), CHOOSE_MAIN, replace=False)),
            'trevos':  sorted(np.random.choice(range(1, TOTAL_TREVO+1), CHOOSE_TREVO, replace=False)),
        } for _ in range(n_concursos)]
        res = PredictiveRanking(sim).rank_predictive_power(block_sizes, verbose=False)
        sim_max.append(max(r['accuracy'] for r in res.values()))
        for k, v in res.items():
            sim_by[k].append(v['accuracy'])

    # CORRIGIDO: +1 para evitar p=0
    p_global = (1 + float(np.sum(np.array(sim_max) >= real_max))) / (n_simulations + 1)

    print(f"\n🌟 Melhor real: {real_key} = {real_max:.1f}%")
    print(f"   p-global (+1): {p_global:.4f} {'🔍' if p_global < 0.05 else '❌'}")
    print(f"\n⚠️ O MC testa ausência de estrutura temporal — NÃO demonstra ganho preditivo.")
    return real_results, sim_by

# ============================================================
# TESTE CONCURSO A CONCURSO
# ============================================================
def test_concurso_a_concurso(contests, min_history=200):
    print(f"\n🎯 TESTE CONCURSO A CONCURSO (histórico mínimo={min_history})")
    for f in ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']:
        series = np.array([extract_filter(c['dezenas'], f) for c in contests], dtype=float)
        ac_r = ac_t = tot = 0
        for t in range(min_history, len(contests) - 1):
            hist = series[:t]
            if len(hist) < 20: continue
            ms, ml = np.mean(hist[-20:]), np.mean(hist)
            tot += 1
            cur, nxt = series[t], series[t+1]
            d = ms > ml
            if (d and nxt < cur) or (not d and nxt > cur): ac_r += 1
            if (ms > ml and nxt > cur) or (ms < ml and nxt < cur): ac_t += 1
        if tot:
            print(f"{f:<15}: Rev={ac_r/tot*100:.1f}% ({ac_r}/{tot}) | "
                  f"Tend={ac_t/tot*100:.1f}% ({ac_t}/{tot})")

# ============================================================
# STRUCTURAL PREDICTOR (IPE clamp)
# ============================================================
class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests

    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR (método: {method})")
        info = {
            'pares': {'min': 0, 'max': 6}, 'moldura': {'min': 0, 'max': 6},
            'primos': {'min': 0, 'max': 6}, 'soma': {'min': 21, 'max': 285},
            'amplitude': {'min': 5, 'max': 49}, 'consecutivos': {'min': 0, 'max': 5},
        }
        r = {}
        for f, i in info.items():
            s = np.array([extract_filter(c['dezenas'], f) for c in self.contests], dtype=float)
            if method == 'recent':
                rc = s[-50:]
                lo = max(i['min'], int(np.percentile(rc, 35)))
                hi = min(i['max'], int(np.percentile(rc, 65)))
                r[f] = (lo, hi); print(f"   {f:<15}: [{lo}, {hi}]")
            else:
                fs = np.mean(s[-20:]); fl = np.mean(s[-500:]) if len(s) >= 500 else np.mean(s)
                ipe = (fs - fl) / fl * 100 if fl > 0 else 0
                pred = int(fl)
                if ipe > 5:   pred = max(i['min'], pred)
                elif ipe < -5: pred = min(i['max'], pred + 1)
                # CLAMP aos limites
                lo = max(i['min'], pred - 1)
                hi = min(i['max'], pred + 1)
                if lo > hi: lo, hi = i['min'], i['max']
                r[f] = (lo, hi); print(f"   {f:<15}: [{lo}, {hi}] (IPE={ipe:+.1f}%)")
        return r

# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_structural(contests, train_size=500, test_size=50, step=50, excluded=None):
    print(f"\n🔬 WALK-FORWARD ESTRUTURAL")
    print(f"   Treino={train_size} Teste={test_size} Passo={step}")
    results = []
    start = train_size
    while start + test_size <= len(contests):
        train = contests[start-train_size:start]
        test = contests[start:start+test_size]
        ranges = StructuralPredictor(train).predict_ranges(method='recent')
        opt = PortfolioOptimizer(train, excluded=excluded,
                                 range_pares=ranges.get('pares'),
                                 range_moldura=ranges.get('moldura'),
                                 range_primos=ranges.get('primos'),
                                 range_soma=ranges.get('soma'),
                                 range_amplitude=ranges.get('amplitude'),
                                 range_consecutivos=ranges.get('consecutivos'))
        try:
            p = opt.optimize(5, 10000, method='pair_covering')
            bt = opt.backtest(p, test)
            results.append({'start': start, 'lift_draw': bt['lift_draw'], 'roi_fixo': bt['roi_fixo_apenas'],
                            'draws_with_win': bt['draws_with_win']})
            print(f"   Janela {start}: lift_draw={bt['lift_draw']:.3f} | "
                  f"ROI_fixo={bt['roi_fixo_apenas']:+.1f}% | "
                  f"concursos c/ prêmio={bt['draws_with_win']}/{bt['n_test']}")
        except Exception as e:
            print(f"   Janela {start}: ERRO — {e}")
        start += step
    if results:
        print(f"\n📊 RESUMO:")
        print(f"   Média lift_draw: {np.mean([r['lift_draw'] for r in results]):.3f}")
        print(f"   Média ROI_fixo:  {np.mean([r['roi_fixo'] for r in results]):+.1f}%")
    return results

# ============================================================
# BUSCA OOS E TRINCAS
# ============================================================
def search_best_fixed_oos(contests, n_fixed=3, top_n=20, train_size=3500,
                          n_games=5, n_candidates=10000, method='pair_covering', excluded=None):
    print(f"\n🔎 MELHORES {n_fixed} FIXAS (OUT-OF-SAMPLE)")
    train = contests[:train_size]; test = contests[train_size:]
    candidates = []
    for ft in tqdm(combinations(range(1, TOTAL_MAIN+1), n_fixed), desc="Treino"):
        if excluded and any(x in ft for x in excluded): continue
        fset = set(ft)
        ac = sum(1 for c in train if fset.issubset(set(c['dezenas'])))
        freq = ac / len(train)
        if freq >= 0.001: candidates.append((ft, freq))
    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for ft, freq in tqdm(candidates[:200], desc="Backtest"):
        opt = PortfolioOptimizer(train, fixed=list(ft), excluded=excluded)
        try:
            p = opt.optimize(n_games, n_candidates, method=method)
            bt = opt.backtest(p, test)
            results.append({'fixed': ft, 'freq': freq,
                            'lift_draw': bt['lift_draw'], 'roi_fixo': bt['roi_fixo_apenas']})
        except: continue
    results.sort(key=lambda x: x['roi_fixo'], reverse=True)
    print(f"\n🏆 TOP {top_n} FIXAS (ROI fixo OOS):")
    for i, r in enumerate(results[:top_n], 1):
        print(f"{i:<5} {str(r['fixed']):<20} ROI_fixo={r['roi_fixo']:+.1f}% lift={r['lift_draw']:.3f}")
    return results


def compare_trincas(contests, t1, t2, n_games=5, n_candidates=50000,
                    method='pair_covering', excluded=None):
    print(f"\n⚔️ COMPARAÇÃO DE TRINCAS")
    for i, t in enumerate([t1, t2], 1):
        opt = PortfolioOptimizer(contests, fixed=list(t), excluded=excluded)
        p = opt.optimize(n_games, n_candidates, method=method)
        bt = opt.backtest(p, contests[-200:])
        print(f"   Trinca {i} ({t}): lift_draw={bt['lift_draw']:.3f} | "
              f"ROI_fixo={bt['roi_fixo_apenas']:+.1f}% | "
              f"concursos c/ prêmio={bt['draws_with_win']}/{bt['n_test']}")

# ============================================================
# AUDITORIA FATOR DE ESCALA
# ============================================================
def audit_scale_factor(contests, window=50):
    sums = np.array([sum(c['dezenas']) for c in contests], dtype=float)
    n = len(sums)
    if n < window + 30:
        print(f"⚠️ Histórico insuficiente ({n} < {window+30})."); return None
    F_prev_l, F_next_l, S_prev_l, S_next_l, idx_l = [], [], [], [], []
    for t in range(window + 1, n):
        ref_prev = float(np.mean(sums[t-window-1:t-1]))
        ref_next = float(np.mean(sums[t-window:t]))
        if ref_prev <= 0 or ref_next <= 0: continue
        F_prev_l.append(sums[t-1]/ref_prev); F_next_l.append(sums[t]/ref_next)
        S_prev_l.append(sums[t-1]); S_next_l.append(sums[t]); idx_l.append(t)
    F_prev = np.array(F_prev_l); F_next = np.array(F_next_l)
    S_prev = np.array(S_prev_l); S_next = np.array(S_next_l)
    refs_next = np.array([np.mean(sums[t-window:t]) for t in idx_l])
    N = len(F_prev)
    print(f"\n📐 FATOR DE ESCALA  janela = {window}  N = {N}")
    pear  = pearsonr(F_prev, F_next); spear = spearmanr(F_prev, F_next)
    print(f"   Pearson r = {pear.statistic:+.4f} (p = {pear.pvalue:.4f})")
    print(f"   Spearman ρ = {spear.statistic:+.4f} (p = {spear.pvalue:.4f})")
    above_prev = F_prev > 1; above_next = F_next > 1
    concord = int(np.sum(above_prev == above_next))
    p_sign = binomtest(concord, N, 0.5, alternative='two-sided').pvalue
    print(f"   Concordância estado: {concord}/{N} = {concord/N*100:.1f}% (p = {p_sign:.4f})")
    if pear.pvalue < 0.05 and pear.statistic > 0:
        print(f"   ✅ H1 PERSISTÊNCIA")
    elif pear.pvalue < 0.05 and pear.statistic < 0:
        print(f"   ✅ H2 REVERSÃO")
    else:
        print(f"   ⚪ H3 INDEPENDÊNCIA")
    return {'F_prev': F_prev, 'F_next': F_next}

# ============================================================
# AUDITORIA γ (sem KeyError; BH entre janelas)
# ============================================================
GAMMA_GRID = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25]
WINDOWS_DEFAULT = [7, 10, 20, 30, 50, 100]


def _gamma_experiment(sums, window, gamma_grid):
    n = len(sums)
    idx_l, S_prev_l, S_next_l, ref_l = [], [], [], []
    for t in range(window + 1, n):
        ref = float(np.mean(sums[t-window:t]))
        if ref <= 0: continue
        idx_l.append(t); S_prev_l.append(sums[t-1])
        S_next_l.append(sums[t]); ref_l.append(ref)
    idx = np.asarray(idx_l, dtype=int); Sp = np.asarray(S_prev_l, dtype=float)
    Sn  = np.asarray(S_next_l, dtype=float); R = np.asarray(ref_l, dtype=float)
    N = len(idx)
    if N < 50: return None
    n_disc = int(0.60 * N); n_conf = int(0.20 * N)
    i_disc = slice(0, n_disc); i_conf = slice(n_disc, n_disc+n_conf)
    i_hold = slice(n_disc + n_conf, N)
    preds = {g: R + g * (Sp - R) for g in gamma_grid}
    baselines = {
        'Média histórica': np.array([np.mean(sums[:t]) for t in idx]),
        'Média móvel':     R.copy(),   # ← γ=0 POR DEFINIÇÃO
        'Mediana móvel':   np.array([np.median(sums[t-window:t]) for t in idx]),
        'Último valor':    Sp.copy(),
    }
    def mae(p, target):
        return float(np.mean(np.abs(np.asarray(p, dtype=float) - np.asarray(target, dtype=float))))
    mae_disc_g = {g: mae(preds[g][i_disc], Sn[i_disc]) for g in gamma_grid}
    best_gamma = min(gamma_grid, key=lambda g: mae_disc_g[g])
    best_key = f'γ={best_gamma:+.2f}'
    phases = {}
    for phase, sl in [('disc', i_disc), ('conf', i_conf), ('hold', i_hold)]:
        target_ph = Sn[sl]
        phase_mae = {name: mae(v[sl], target_ph) for name, v in baselines.items()}
        for g in gamma_grid: phase_mae[f'γ={g:+.2f}'] = mae(preds[g][sl], target_ph)
        phase_abs = {
            'Média histórica': np.abs(baselines['Média histórica'][sl] - target_ph),
            'Média móvel':     np.abs(baselines['Média móvel'][sl]     - target_ph),
            'Mediana móvel':   np.abs(baselines['Mediana móvel'][sl]   - target_ph),
            'Último valor':    np.abs(baselines['Último valor'][sl]    - target_ph),
            'γ*':              np.abs(preds[best_gamma][sl]            - target_ph),
        }
        phases[phase] = {'mae': phase_mae, 'abs_err': phase_abs}
    return {'N': N, 'n_disc': n_disc, 'n_conf': n_conf, 'n_hold': N-n_disc-n_conf,
            'best_gamma': best_gamma, 'best_key': best_key, 'phases': phases}


def _iid_sums(n, rng):
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = rng.choice(np.arange(1, TOTAL_MAIN+1), size=CHOOSE_MAIN, replace=False)
        out[i] = float(np.sum(s))
    return out


def _bh_adjust(p_values):
    """Benjamini-Hochberg."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0: return p
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n + 1)
    adj = p * n / ranks
    for i in range(n - 2, -1, -1):
        idx_i, idx_next = order[i], order[i + 1]
        adj[idx_i] = min(adj[idx_i], adj[idx_next])
    return np.clip(adj, 0.0, 1.0)


def audit_gamma_coefficient(contests, windows=None, gamma_grid=None, iid_mc=200):
    if windows is None:    windows = WINDOWS_DEFAULT
    if gamma_grid is None: gamma_grid = GAMMA_GRID
    sums = np.asarray([sum(c['dezenas']) for c in contests], dtype=float)
    n_real = len(sums)

    print(f"\n📐 AUDITORIA γ — v1.5")
    print(f"   Modelo: Ŝ_t = R_t + γ·(S_(t-1) − R_t),  R_t = mean(S[t-w : t])")
    print(f"   γ grid: {gamma_grid}")
    print(f"   Split 60/20/20 | IID MC por janela: {iid_mc}")
    print(f"   E[S] teórico = {CHOOSE_MAIN*(TOTAL_MAIN+1)/2:.2f}")
    print(f"   E[S] amostral = {np.mean(sums):.2f} | SD = {np.std(sums):.2f}")

    reais = {}
    for w in windows:
        r = _gamma_experiment(sums, w, gamma_grid)
        if r is not None: reais[w] = r
    if not reais:
        print("⚠️ Nenhuma janela válida."); return None

    # ---------- Veredito OOS (com BH entre janelas) ----------
    print("\n" + "=" * 100)
    print("   VEREDITO OOS — γ* (congelado na descoberta) vs baselines no HOLDOUT")
    print("=" * 100)
    window_pvals = []
    for w, r in reais.items():
        ph = r['phases']['hold']; gk = r['best_key']
        mae_g = ph['mae'][gk]
        mae_mm = ph['mae']['Média móvel']       # γ=0 POR DEFINIÇÃO
        mae_mh = ph['mae']['Média histórica']
        mae_ul = ph['mae']['Último valor']

        abs_0 = ph['abs_err']['Média móvel']
        abs_g = ph['abs_err']['γ*']
        if np.all(abs_0 - abs_g == 0):
            p_vs_0 = 1.0
        else:
            try: p_vs_0 = wilcoxon(abs_0, abs_g, alternative='greater').pvalue
            except: p_vs_0 = 1.0
        window_pvals.append((w, p_vs_0))

        delta_oos = mae_mm - mae_g
        print(f"\n   Janela {w:>3}:  N={r['N']}  γ* = {r['best_gamma']:+.2f}")
        print(f"      MAE holdout — γ*: {mae_g:.2f} | γ=0 (MMov): {mae_mm:.2f} | "
              f"MHist: {mae_mh:.2f} | Últ: {mae_ul:.2f}")
        print(f"      Δ_OOS (γ=0 − γ*): {delta_oos:+.3f}   "
              f"(Wilcoxon: p_raw = {p_vs_0:.4f})")
        if r['best_gamma'] == 0.0:
            print(f"      → γ* = 0 (vencedor é a própria Média Móvel).")
        elif delta_oos > 0:
            print(f"      → Δ_OOS > 0 (ver p_adj abaixo).")

    # BH entre janelas
    if len(window_pvals) > 1:
        p_raw_arr = np.array([p for _, p in window_pvals])
        p_adj_arr = _bh_adjust(p_raw_arr)
        print("\n" + "=" * 100)
        print("   CORREÇÃO BH ENTRE JANELAS")
        print("=" * 100)
        for (w, p_raw), p_adj in zip(window_pvals, p_adj_arr):
            sig = "🔍" if p_adj < 0.05 else ("📊" if p_adj < 0.15 else "  ")
            print(f"   Janela {w:>3}: p_raw = {p_raw:.4f} | p_adj (BH) = {p_adj:.4f} {sig}")

    # ---------- IID MC ----------
    print("\n" + "=" * 100)
    print(f"   BASELINE IID MONTE CARLO ({iid_mc} réplicas/janela)")
    print("=" * 100)
    rng = np.random.default_rng(20250911)
    for w, r_real in reais.items():
        mae0_real = r_real['phases']['hold']['mae']['Média móvel']
        maeg_real = r_real['phases']['hold']['mae'][r_real['best_key']]
        d_real = mae0_real - maeg_real
        d_sims = []
        g_choices = defaultdict(int)
        for _ in range(iid_mc):
            sim = _gamma_experiment(_iid_sums(n_real, rng), w, gamma_grid)
            if sim is None: continue
            mae0 = sim['phases']['hold']['mae']['Média móvel']
            maeg = sim['phases']['hold']['mae'][sim['best_key']]
            d_sims.append(mae0 - maeg)
            g_choices[sim['best_gamma']] += 1
        if not d_sims: continue
        d_sims = np.asarray(d_sims)
        p_emp = (1 + np.sum(d_sims >= d_real)) / (len(d_sims) + 1)
        top = sorted(g_choices.items(), key=lambda kv: -kv[1])[:5]
        top_str = " | ".join(f"{g:+.2f}:{c}" for g, c in top)
        print(f"   Janela {w:>3}: Δ_OOS real = {d_real:+.3f} | IID: "
              f"média={np.mean(d_sims):+.3f} sd={np.std(d_sims):.3f} | p_emp = {p_emp:.4f}")
        print(f"      γ* mais frequentes IID: {top_str}")
        if p_emp < 0.05:
            print(f"      → ✅ Δ_OOS real > IID em {1-p_emp:.1%} das réplicas.")
        else:
            print(f"      → ⚪ Compatível com IID (p = {p_emp:.3f}).")
    return reais

# ============================================================
# MÓDULO 5 — AUDITORIA DE INTEGRIDADE
# ============================================================
def integrity_audit():
    print("=" * 70)
    print("🔬 AUDITORIA DE INTEGRIDADE — v1.5")
    print("=" * 70)
    errors = []

    # [1] Fórmulas de combinatória
    print("\n[1] Fórmulas de universo")
    u6 = comb(50, 6)
    up = u6 * comb(6, 2)
    print(f"   C(50,6)         = {u6:,}      [esperado: 15.890.700]")
    print(f"   C(50,6)·C(6,2)  = {up:,}      [esperado: 238.360.500]")
    if u6 != 15_890_700 or up != 238_360_500:
        errors.append("[1] Fórmulas de combinatória erradas.")
    else:
        print("   ✅ OK")

    # [2] Enumeração com todas fixas (universo=1, pares=15)
    print("\n[2] Enumeração com 6 fixas → universo_main=1, pares=15")
    opt = PortfolioOptimizer([], fixed=[1,2,3,4,5,6])
    pool, stats = opt.enumerate_and_sample(20, seed=42)
    ok = (stats['n_valid_main'] == 1 and stats['n_valid'] == 15 and len(pool) == 15)
    print(f"   n_valid_main = {stats['n_valid_main']} [esperado: 1]")
    print(f"   n_valid      = {stats['n_valid']}    [esperado: 15]")
    print("   ✅ OK" if ok else "   ❌ FALHA")
    if not ok: errors.append("[2] Enumeração com 6 fixas.")

    # [3] Semifixas impossíveis (min > 0, semifixas vazio)
    print("\n[3] Semifixas impossíveis (min=2, lista vazia)")
    try:
        _ = PortfolioOptimizer([], semifixed=[], min_semifixed=2, max_semifixed=2)
        print("   ❌ NÃO detectou impossível")
        errors.append("[3] Semifixas impossíveis não detectadas.")
    except ValueError as e:
        print(f"   ✅ Detectado: {e}")

    # [4] Semifixas com faixa [1,2]
    print("\n[4] Semifixas [1,2,3] com min=1, max=2")
    opt = PortfolioOptimizer([], semifixed=[1,2,3], min_semifixed=1, max_semifixed=2)
    pool, stats = opt.enumerate_and_sample(100, seed=42)
    ok = all(1 <= len(set(g[0]) & {1,2,3}) <= 2 for g in pool)
    print(f"   Todos os jogos do pool respeitam semi ∈ [1,2]: {ok}")
    print("   ✅ OK" if ok else "   ❌ FALHA")
    if not ok: errors.append("[4] Semifixas faixa.")

    # [5] Covering não repete mains
    print("\n[5] Covering sem repetir mains")
    games = [
        ([1,2,3,4,5,6], [1,2]),
        ([1,2,3,4,5,6], [3,4]),   # mesmo main
        ([7,8,9,10,11,12], [1,2]),
        ([13,14,15,16,17,18], [3,4]),
    ]
    opt = PortfolioOptimizer([])
    sel = opt.select_covering(games, 3, level='pair')
    mains = set(tuple(sorted(g[0])) for g in sel)
    ok = (len(sel) == 3 and len(mains) == 3)
    print(f"   Selecionados {len(sel)} jogos, mains únicos: {len(mains)}")
    print("   ✅ OK" if ok else "   ❌ FALHA")
    if not ok: errors.append("[5] Covering duplicata.")

    # [6] Uniformidade do reservoir
    print("\n[6] Uniformidade do reservoir (universo completo)")
    opt = PortfolioOptimizer([], excluded=list(range(9, 51)))   # livres 1..8, C(8,6)=28
    pool, stats = opt.enumerate_and_sample(28 * 15, seed=42)    # universe = 28*15 = 420
    mains = set(tuple(sorted(g[0])) for g in pool)
    ok = (len(mains) == 28 and len(pool) == 420)
    print(f"   Mains distintos: {len(mains)}/28 | pares: {len(pool)}/420")
    print("   ✅ OK" if ok else "   ❌ FALHA")
    if not ok: errors.append("[6] Uniformidade.")

    # [7] Backtest: unidades separadas
    print("\n[7] Backtest separa p_bet e p_draw (3+2 = R$50)")
    opt = PortfolioOptimizer([])
    portfolio = [([1,2,3,4,5,6], [1,2])]
    draws = [{'dezenas': [1,2,3,7,8,9], 'trevos': [1,2]}]   # 3+2
    bt = opt.backtest(portfolio, draws)
    ok = (abs(bt['p_bet_empirical'] - 1.0) < 1e-9
          and abs(bt['p_draw_empirical'] - 1.0) < 1e-9
          and abs(bt['premio_fixo_total'] - 50.0) < 1e-9)
    print(f"   p_bet_emp = {bt['p_bet_empirical']:.4f}   [esperado: 1.0]")
    print(f"   p_draw_emp = {bt['p_draw_empirical']:.4f}  [esperado: 1.0]")
    print(f"   prêmio_fixo = R$ {bt['premio_fixo_total']:.2f}  [esperado: 50.00]")
    print("   ✅ OK" if ok else "   ❌ FALHA")
    if not ok: errors.append("[7] Backtest unidades.")

    # [8] Reprodutibilidade por seed
    print("\n[8] Reprodutibilidade por seed")
    opt = PortfolioOptimizer([], fixed=[10, 20])
    p1, _ = opt.enumerate_and_sample(50, seed=123)
    p2, _ = opt.enumerate_and_sample(50, seed=123)
    ok = (p1 == p2)
    print(f"   Duas execuções (seed=123) idênticas: {ok}")
    print("   ✅ OK" if ok else "   ❌ FALHA")
    if not ok: errors.append("[8] Reprodutibilidade.")

    # ---------- Veredito ----------
    print("\n" + "=" * 70)
    if not errors:
        print("✅ TODAS AS AUDITORIAS PASSARAM")
    else:
        print(f"❌ {len(errors)} FALHA(S):")
        for e in errors: print(f"   • {e}")
    print("=" * 70)
    return len(errors) == 0

# ============================================================
# MENU
# ============================================================
def _ask_structural_ranges():
    r = {}
    print("   Faixas estruturais (ENTER para pular)")
    for key, label in [('range_pares', 'Pares'), ('range_moldura', 'Borda'),
                       ('range_primos', 'Primos'), ('range_soma', 'Soma'),
                       ('range_amplitude', 'Amplitude'), ('range_consecutivos', 'Consec.')]:
        try:
            v = input(f"   {label} min,max: ").strip()
            r[key] = tuple(int(x) for x in v.split(',')) if v else None
        except:
            r[key] = None
    return r


def _ask_inputs_validated():
    try:
        fs = input("\n   Fixas (ex: 10 20 30 ou ENTER): ").strip()
        fixed = [int(x) for x in fs.split()] if fs else []
        ss = input("   Semifixas (ex: 5 15 25 ou ENTER): ").strip()
        semifixed = [int(x) for x in ss.split()] if ss else []
        es = input("   Excluídas (ex: 7 13 27 ou ENTER): ").strip()
        excluded = [int(x) for x in es.split()] if es else []
        if semifixed:
            mn = int(input(f"   Mín semi [0-{len(semifixed)}]: ").strip() or "0")
            mx = int(input(f"   Máx semi [0-{len(semifixed)}]: ").strip() or str(len(semifixed)))
        else:
            mn = int(input("   Mín semi (ENTER=0): ").strip() or "0")
            mx_s = input("   Máx semi (ENTER=0): ").strip()
            mx = int(mx_s) if mx_s else None
        fixed, semifixed, excluded = validate_inputs(fixed, semifixed, excluded, mn, mx)
        return fixed, semifixed, excluded, mn, mx
    except ValueError as e:
        print(f"❌ {e}")
        return None


def main():
    print("=" * 70)
    print("🔬 LABORATÓRIO +MILIONÁRIA – v1.5")
    print("   Módulos 1–5: validação, backtest corrigido, prêmios oficiais,")
    print("   estatística com MC, auditoria de integridade.")
    print("=" * 70)
    contests = load_all_contests('resultados_maismilionaria.csv')
    if not contests:
        print("❌ 'resultados_maismilionaria.csv' não encontrado."); return
    print(f"\n📂 {len(contests)} concursos | último: {contests[-1]['concurso']}")
    print(f"📌 Dezenas: {contests[-1]['dezenas']} | Trevos: {contests[-1]['trevos']}")
    print(f"💰 Custo/aposta: R$ {CUSTO_APOSTA:.2f} | P(aposta premiada): {P_BET_WIN_THEORETICAL:.6f}")

    while True:
        print("\nOpções:")
        print("1.  Gerar carteira personalizada")
        print("2.  Walk-forward condicional")
        print("3.  Backtest (200 concursos)")
        print("4.  Buscar fixas OOS")
        print("5.  Comparar trincas")
        print("6.  Ranking preditivo (binomial + block permutation opcional)")
        print("7.  Poder preditivo das 50 dezenas")
        print("8.  Structural Predictor")
        print("9.  Carteira com previsões estruturais")
        print("10. Monte Carlo FWER")
        print("11. Teste concurso a concurso")
        print("12. Walk-forward estrutural")
        print("13. Auditoria do Fator de Escala")
        print("14. Auditoria do Coeficiente γ")
        print("15. ⚙️  AUDITORIA DE INTEGRIDADE (Módulo 5)")
        print("0.  Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            parsed = _ask_inputs_validated()
            if not parsed: continue
            fixed, semifixed, excluded, mn, mx = parsed
            ranges = _ask_structural_ranges()
            print("\n   Modo: [1] Tentativas [2] Universo exato")
            modo = input("   Escolha [1]: ").strip() or "1"
            exact = (modo == '2')
            try:
                pool = int(input(f"   Tamanho do pool [{'50000' if exact else '50000'}]: ").strip() or "50000")
            except: pool = 50000
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            try:
                opt = PortfolioOptimizer(contests, fixed=fixed, semifixed=semifixed,
                                         min_semifixed=mn, max_semifixed=mx,
                                         excluded=excluded, **ranges)
                portfolio = opt.optimize(5, pool, method=method, exact_universe=exact)
            except (ValueError, RuntimeError) as e:
                print(f"❌ {e}"); continue
            for i, (main, trevo) in enumerate(portfolio, 1):
                p = sum(1 for x in main if x % 2 == 0)
                pr = sum(1 for x in main if x in PRIMES)
                m = sum(1 for x in main if x in MOLDURA)
                print(f" {i}. Main:{main} | Trevos:{trevo} | P:{p} Pr:{pr} Borda:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200) — por aposta:")
                print(f"   p_bet_emp    = {bt['p_bet_empirical']:.6f}  |  "
                      f"p_bet_teo = {bt['p_bet_theoretical']:.6f}  |  "
                      f"lift = {bt['lift_bet']:.3f}")
                print(f"\n🔬 BACKTEST (200) — por concurso:")
                print(f"   p_draw_emp   = {bt['p_draw_empirical']:.4f}  |  "
                      f"p_draw_mc = {bt['p_draw_theoretical_mc']:.4f}  |  "
                      f"lift = {bt['lift_draw']:.3f}")
                print(f"   p_draw_indep = {bt['p_draw_indep_reference']:.4f}  "
                      f"(referência; NÃO usar como lift)")
                print(f"\n💰 Financeiro (só prêmios fixos):")
                print(f"   Prêmios fixos = R$ {bt['premio_fixo_total']:.2f} | "
                      f"Variáveis (contagem) = {bt['premio_variavel_count']}")
                print(f"   Custo total   = R$ {bt['custo_total']:.2f} | "
                      f"ROI_fixo = {bt['roi_fixo_apenas']:+.2f}%")
                print(f"   ⚠️ Prêmios variáveis NÃO contabilizados.")

        elif op == '2':
            fs = input("\n   Fixas (ENTER pular): ").strip()
            try:
                fixed = [int(x) for x in fs.split()] if fs else []
                fixed = validate_numbers(fixed, 'fixas')
            except ValueError as e:
                print(f"❌ {e}"); continue
            es = input("   Excluídas (ENTER pular): ").strip()
            try:
                excluded = [int(x) for x in es.split()] if es else []
                excluded = validate_numbers(excluded, 'excluídas')
            except ValueError as e:
                print(f"❌ {e}"); continue
            metodo = input("   Método [1/2]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            lifts = []
            for w in range(8):
                te = len(contests) - w * 50
                ts = te - 50
                t_start = max(0, ts - 400)
                if t_start >= ts or ts >= te: continue
                try:
                    opt = PortfolioOptimizer(contests[t_start:ts], fixed=fixed, excluded=excluded)
                    p = opt.optimize(5, 10000, method=method)
                    bt = opt.backtest(p, contests[ts:te])
                    lifts.append(bt['lift_draw'])
                    print(f"   Janela {w}: lift_draw={bt['lift_draw']:.3f} | "
                          f"ROI_fixo={bt['roi_fixo_apenas']:+.1f}% | "
                          f"concursos c/ prêmio={bt['draws_with_win']}/{bt['n_test']}")
                except Exception as e:
                    print(f"   Janela {w}: {e}")
            if lifts:
                print(f"\n📊 Média lift_draw: {np.mean(lifts):.3f}")

        elif op == '3':
            fs = input("\n   Fixas (ENTER pular): ").strip()
            try:
                fixed = [int(x) for x in fs.split()] if fs else []
                fixed = validate_numbers(fixed, 'fixas')
                es = input("   Excluídas (ENTER pular): ").strip()
                excluded = [int(x) for x in es.split()] if es else []
                excluded = validate_numbers(excluded, 'excluídas')
            except ValueError as e:
                print(f"❌ {e}"); continue
            metodo = input("   Método [1/2]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            print("\n   Modo: [1] Tentativas [2] Universo exato")
            modo = input("   Escolha [1]: ").strip() or "1"
            exact = (modo == '2')
            try:
                pool = int(input("   Pool [50000]: ").strip() or "50000")
            except: pool = 50000
            try:
                opt = PortfolioOptimizer(contests, fixed=fixed, excluded=excluded)
                portfolio = opt.optimize(5, pool, method=method, exact_universe=exact)
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n📊 p_bet_emp={bt['p_bet_empirical']:.6f} lift_bet={bt['lift_bet']:.3f}")
                print(f"   p_draw_emp={bt['p_draw_empirical']:.4f} "
                      f"p_draw_mc={bt['p_draw_theoretical_mc']:.4f} "
                      f"lift_draw={bt['lift_draw']:.3f}")
                print(f"   p_draw_indep(ref)={bt['p_draw_indep_reference']:.4f}")
                print(f"   ROI_fixo={bt['roi_fixo_apenas']:+.2f}% (R$ {bt['premio_fixo_total']:.2f} em prêmios fixos)")
                print(f"   Hit_dist: {dict(sorted(bt['hit_distribution'].items()))}")
            except Exception as e:
                print(f"❌ {e}")

        elif op == '4':
            try:
                n_fixed = int(input("\n   Fixas (2-4): ").strip())
                if n_fixed not in [2,3,4]: continue
            except: continue
            try:
                top_n = int(input("   Top N [20]: ").strip() or "20")
                train_size = int(input("   Treino [3500]: ").strip() or "3500")
            except: continue
            es = input("   Excluídas (ENTER pular): ").strip()
            try:
                excluded = [int(x) for x in es.split()] if es else []
                excluded = validate_numbers(excluded, 'excluídas')
            except ValueError as e:
                print(f"❌ {e}"); continue
            metodo = input("   Método [1/2]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            search_best_fixed_oos(contests, n_fixed, top_n, train_size,
                                  method=method, excluded=excluded)

        elif op == '5':
            t1s = input("\n   Trinca 1: ").strip()
            t2s = input("   Trinca 2: ").strip()
            try:
                t1 = validate_numbers([int(x) for x in t1s.split()], 'trinca1')
                t2 = validate_numbers([int(x) for x in t2s.split()], 'trinca2')
                if len(t1) != 3 or len(t2) != 3: continue
            except ValueError as e:
                print(f"❌ {e}"); continue
            compare_trincas(contests, tuple(t1), tuple(t2))

        elif op == '6':
            bs = input("\n   Blocos [50,100,200,500]: ").strip()
            try:
                blocks = [int(x) for x in bs.split(',')] if bs else [50,100,200,500]
            except: blocks = [50,100,200,500]
            try:
                bp = int(input("   Block permutations (0=off) [0]: ").strip() or "0")
            except: bp = 0
            PredictiveRanking(contests).rank_predictive_power(blocks, block_permutations=bp)

        elif op == '7':
            bs = input("\n   Blocos [100,200,500]: ").strip()
            try:
                blocks = [int(x) for x in bs.split(',')] if bs else [100,200,500]
            except: blocks = [100,200,500]
            PredictiveRanking(contests).rank_dezenas_individual(blocks)

        elif op == '8':
            m = 'recent' if (input("   [1] Recente [2] IPE: ").strip() or "1") == "1" else 'ipe'
            StructuralPredictor(contests).predict_ranges(method=m)

        elif op == '9':
            ranges = StructuralPredictor(contests).predict_ranges(method='recent')
            fixed = [10, 20, 30]
            print(f"\n   Fixas sugeridas: {fixed}")
            es = input("   Excluídas (ENTER pular): ").strip()
            try:
                excluded = [int(x) for x in es.split()] if es else []
                excluded = validate_numbers(excluded, 'excluídas')
            except ValueError as e:
                print(f"❌ {e}"); continue
            gerar = input("   Gerar? (s/n): ").strip().lower()
            if gerar == 's':
                try:
                    opt = PortfolioOptimizer(
                        contests, fixed=fixed, excluded=excluded,
                        range_pares=ranges.get('pares'),
                        range_moldura=ranges.get('moldura'),
                        range_primos=ranges.get('primos'),
                        range_soma=ranges.get('soma'),
                        range_amplitude=ranges.get('amplitude'),
                        range_consecutivos=ranges.get('consecutivos'),
                    )
                    portfolio = opt.optimize(5, 50000, method='pair_covering')
                    for i, (main, trevo) in enumerate(portfolio, 1):
                        p = sum(1 for x in main if x%2==0)
                        pr = sum(1 for x in main if x in PRIMES)
                        m = sum(1 for x in main if x in MOLDURA)
                        print(f" {i}. Main:{main} | Trevos:{trevo} | P:{p} Pr:{pr} Borda:{m}")
                    if len(contests) > 200:
                        bt = opt.backtest(portfolio, contests[-200:])
                        print(f"\n🔬 p_draw_emp={bt['p_draw_empirical']:.4f} "
                              f"lift={bt['lift_draw']:.3f} ROI_fixo={bt['roi_fixo_apenas']:+.1f}%")
                except Exception as e:
                    print(f"❌ {e}")

        elif op == '10':
            try: ns = int(input("\n   Simulações [1000]: ").strip() or "1000")
            except: ns = 1000
            bs = input("   Blocos [50,100,200]: ").strip()
            try: blocks = [int(x) for x in bs.split(',')] if bs else [50,100,200]
            except: blocks = [50,100,200]
            monte_carlo_control(contests, ns, blocks)

        elif op == '11':
            try: mh = int(input("\n   Histórico mínimo [200]: ").strip() or "200")
            except: mh = 200
            test_concurso_a_concurso(contests, mh)

        elif op == '12':
            try:
                ts = int(input("\n   Treino [500]: ").strip() or "500")
                te = int(input("   Teste [50]: ").strip() or "50")
                st = int(input("   Passo [50]: ").strip() or "50")
            except: ts, te, st = 500, 50, 50
            es = input("   Excluídas (ENTER pular): ").strip()
            try:
                excluded = [int(x) for x in es.split()] if es else []
                excluded = validate_numbers(excluded, 'excluídas')
            except ValueError as e:
                print(f"❌ {e}"); continue
            walk_forward_structural(contests, ts, te, st, excluded=excluded)

        elif op == '13':
            try: win = int(input("\n   Janela [50]: ").strip() or "50")
            except: win = 50
            audit_scale_factor(contests, window=win)

        elif op == '14':
            jan_str = input(f"\n   Janelas [{' '.join(map(str, WINDOWS_DEFAULT))}]: ").strip()
            try: windows = [int(x) for x in jan_str.split()] if jan_str else WINDOWS_DEFAULT
            except: windows = WINDOWS_DEFAULT
            gam_str = input(f"   γ grid [{' '.join(map(str, GAMMA_GRID))}]: ").strip()
            try: gamma_grid = [float(x) for x in gam_str.split()] if gam_str else GAMMA_GRID
            except: gamma_grid = GAMMA_GRID
            try: iid_mc = int(input("   IID MC [200]: ").strip() or "200")
            except: iid_mc = 200
            audit_gamma_coefficient(contests, windows=windows, gamma_grid=gamma_grid, iid_mc=iid_mc)

        elif op == '15':
            integrity_audit()

        elif op == '0':
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
