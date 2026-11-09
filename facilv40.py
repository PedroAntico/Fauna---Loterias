#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.4.3

EVOLUÇÃO DO v49.4.2:
✅ Semifixas respeitadas no modo exato (min/max validados por jogo)
✅ allowed_pares/moldura/primos aplicados no modo exato
✅ Universo efetivo calculado por combinatória exata (considera semifixas)
✅ Relatório de rejeições por filtro (semifixas, allowed_*, ranges, grupos, critérios)
✅ Covering explicita dupla amostragem: enumeração → amostra; amostra → covering (MAX_COVER)
✅ Docstring corrigida: memória O(k) onde k = n_sample (reservoir)
✅ Checagem de consistência quando não há filtros ativos
"""

import numpy as np
from scipy.stats import hypergeom, binomtest, ttest_rel
from collections import defaultdict
from itertools import combinations
import os, random, time, warnings
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

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

# ============================================================
# PRÉ-COMPUTAÇÃO DE PARES E TRINCAS
# ============================================================
PAIR_BIT = [[0] * 25 for _ in range(25)]
_idx = 0
for i in range(25):
    for j in range(i + 1, 25):
        bit = 1 << _idx
        PAIR_BIT[i][j] = bit
        PAIR_BIT[j][i] = bit
        _idx += 1

TRIPLE_BIT = [[[0] * 25 for _ in range(25)] for _ in range(25)]
_idx = 0
for i in range(25):
    for j in range(i + 1, 25):
        for k in range(j + 1, 25):
            bit = 1 << _idx
            TRIPLE_BIT[i][j][k] = bit
            TRIPLE_BIT[i][k][j] = bit
            TRIPLE_BIT[j][i][k] = bit
            TRIPLE_BIT[j][k][i] = bit
            TRIPLE_BIT[k][i][j] = bit
            TRIPLE_BIT[k][j][i] = bit
            _idx += 1


def _pairs_mask(game):
    numbers = [x - 1 for x in game]
    mask = 0
    for i in range(len(numbers)):
        bi = PAIR_BIT[numbers[i]]
        for j in range(i + 1, len(numbers)):
            mask |= bi[numbers[j]]
    return mask


def _triples_mask(game):
    numbers = sorted(x - 1 for x in game)
    mask = 0
    for i in range(len(numbers)):
        bi = TRIPLE_BIT[numbers[i]]
        for j in range(i + 1, len(numbers)):
            bij = bi[numbers[j]]
            for k in range(j + 1, len(numbers)):
                mask |= bij[numbers[k]]
    return mask

# ============================================================
# JOGOS CRITÉRIOS
# ============================================================
CRITERIA_GAMES = []
CRITERIA_MASKS = []


def load_criteria_games(filename='jogos_criterios.txt'):
    global CRITERIA_GAMES, CRITERIA_MASKS
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de critérios não encontrado: {path}")

    games = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                nums = [int(x) for x in line.split()]
            except ValueError:
                continue
            if (len(nums) in (16, 17, 18, 19)
                and len(set(nums)) == len(nums)
                and all(1 <= x <= 25 for x in nums)):
                games.append(sorted(nums))

    if not games:
        raise ValueError("Nenhum jogo critério válido encontrado.")

    CRITERIA_GAMES = games
    CRITERIA_MASKS = []
    for g in games:
        mask = BITMASK_CACHE.get_mask(g)
        CRITERIA_MASKS.append((mask, 14 if len(g) == 19 else 13))

    print("\n📋 JOGOS CRITÉRIOS CARREGADOS")
    print(f"   Total: {len(games)}")
    for sz in (19, 18, 17, 16):
        print(f"   {sz}d: {sum(len(g) == sz for g in games)}")
    return games


def passes_criteria(game):
    if not CRITERIA_MASKS:
        raise RuntimeError("Critérios não carregados. Geração bloqueada.")
    game_mask = BITMASK_CACHE.get_mask(game)
    for cmask, limit in CRITERIA_MASKS:
        if (game_mask & cmask).bit_count() > limit:
            return False
    return True


def passes_groups(game, groups):
    if not groups:
        return True
    gs = set(game)
    for g in groups:
        qty = len(gs & g["dezenas_set"])
        if not g["min"] <= qty <= g["max"]:
            return False
    return True


def validate_groups(groups, fixed, excluded):
    warns = []
    fixed_set = set(fixed or [])
    excl_set = set(excluded or [])

    normalized = []
    for i, g in enumerate(groups):
        dset = g.get("dezenas_set")
        if dset is None:
            dset = set(g.get("dezenas", []))
        normalized.append({
            "idx": i, "dezenas_set": set(dset),
            "min": g["min"], "max": g["max"],
        })

    for g in normalized:
        n_fx = len(fixed_set & g["dezenas_set"])
        if n_fx > g["max"]:
            warns.append(f"Grupo {g['idx']+1}: {n_fx} fixas > máximo {g['max']}.")
        disp = len(g["dezenas_set"] - excl_set - fixed_set)
        if disp + n_fx < g["min"]:
            warns.append(f"Grupo {g['idx']+1}: mínimo {g['min']} > disponíveis ({disp + n_fx}).")

    n = len(normalized)
    for i in range(n):
        for j in range(i + 1, n):
            g1, g2 = normalized[i], normalized[j]
            overlap = g1["dezenas_set"] & g2["dezenas_set"]
            if not overlap:
                continue
            min_forced = max(0, g1["min"] + g2["min"] - (15 - len(overlap)))
            if min_forced > len(overlap):
                warns.append(
                    f"Grupos {g1['idx']+1} e {g2['idx']+1}: mínimos "
                    f"{g1['min']}+{g2['min']} forçam {min_forced} no overlap "
                    f"de {len(overlap)} — impossível satisfazer ambos."
                )
    return warns


def audit_portfolio(portfolio, groups=None, applied_criteria=True):
    all_valid = True

    if CRITERIA_MASKS:
        header = "APLICADOS" if applied_criteria else "NÃO APLICADOS NA GERAÇÃO (só verificação)"
        print(f"\n🔎 AUDITORIA DA CARTEIRA (CRITÉRIOS {header})")
        for i, game in enumerate(portfolio, 1):
            gmask = BITMASK_CACHE.get_mask(game)
            violations = []
            best = (0, None, None)
            for cmask, limit in CRITERIA_MASKS:
                hits = mask_intersection(gmask, cmask)
                sz = cmask.bit_count()
                if hits > limit:
                    violations.append({"size": sz, "hits": hits, "limit": limit})
                if hits > best[0]:
                    best = (hits, sz, limit)
            if violations:
                all_valid = False
                det = " | ".join(f"Critério {v['size']}d: {v['hits']}/{v['limit']}"
                                 for v in violations)
                print(f"Jogo {i}: ❌ REPROVADO | {det}")
            else:
                print(f"Jogo {i}: ✅ APROVADO | maior: {best[0]}/{best[2]} (critério {best[1]}d)")
        print("\n✅ CARTEIRA VALIDADA (critérios)." if all_valid
              else "\n❌ CARTEIRA COM VIOLAÇÕES (critérios).")

    if groups:
        print("\n🔎 AUDITORIA DA CARTEIRA (GRUPOS)")
        gv = True
        for i, game in enumerate(portfolio, 1):
            gs = set(game)
            parts = []
            for gi, g in enumerate(groups, 1):
                qty = len(gs & g["dezenas_set"])
                ok = g["min"] <= qty <= g["max"]
                if not ok:
                    gv = False
                parts.append(f"G{gi}:{qty}[{g['min']}-{g['max']}]{'✓' if ok else '✗'}")
            print(f"Jogo {i}: " + " | ".join(parts))
        print("\n✅ CARTEIRA VALIDADA (grupos)." if gv
              else "\n❌ CARTEIRA COM VIOLAÇÕES (grupos).")
        if not gv:
            all_valid = False

    return all_valid

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, csv_file)
    if not os.path.exists(path):
        return None
    contests = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(';')
            if len(parts) < 17:
                continue
            try:
                dz = [int(x.strip()) for x in parts[2:17] if x.strip()]
                if len(dz) != 15 or len(set(dz)) != 15:
                    continue
                if any(x < 1 or x > 25 for x in dz):
                    continue
                contests.append({
                    'concurso': int(parts[0]),
                    'data': parts[1],
                    'dezenas': sorted(dz),
                })
            except:
                continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos válidos")
    return contests

# ============================================================
# PORTFÓLIOS ALEATÓRIOS
# ============================================================
def random_portfolio_unrestricted(n_games, rng):
    return [sorted(rng.choice(range(1, 26), 15, replace=False).tolist())
            for _ in range(n_games)]


def random_portfolio_conditioned(n_games, fixed, excluded, rng):
    fs = set(fixed or [])
    es = set(excluded or [])
    avail = [x for x in range(1, 26) if x not in fs and x not in es]
    n_free = 15 - len(fs)
    if n_free < 0:
        raise ValueError(f"Fixas ({len(fs)}) > 15.")
    if n_free > len(avail):
        raise ValueError(f"Disponíveis insuficientes.")
    out = []
    for _ in range(n_games):
        if n_free == 0:
            out.append(sorted(fs))
        else:
            picks = rng.choice(avail, n_free, replace=False).tolist()
            out.append(sorted(fs | set(picks)))
    return out

# ============================================================
# BACKTEST
# ============================================================
def backtest_portfolio(portfolio, test_draws):
    if not portfolio or not test_draws:
        return {
            'empirical': 0.0, 'draws_with_win': 0, 'draws_with_win_rate': 0.0,
            'n_test': len(test_draws) if test_draws else 0, 'n_success': 0,
            'total_premio': 0.0, 'total_custo': 0.0, 'roi': 0.0,
            'hit_distribution': {k: 0 for k in range(11, 16)},
        }

    n_success = 0
    draws_with_win = 0
    total_premio = 0.0
    total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
    portfolio_masks = [BITMASK_CACHE.get_mask(g) for g in portfolio]
    hit_counts = {k: 0 for k in range(11, 16)}

    for draw in test_draws:
        dm = BITMASK_CACHE.get_mask(draw['dezenas'])
        hit_this = False
        for pm in portfolio_masks:
            hits = mask_intersection(pm, dm)
            if hits >= 11:
                n_success += 1
                total_premio += PREMIO_VALORES.get(hits, 0)
                hit_counts[hits] += 1
                hit_this = True
        if hit_this:
            draws_with_win += 1

    rate = draws_with_win / len(test_draws)
    return {
        'empirical': rate,
        'draws_with_win': draws_with_win,
        'draws_with_win_rate': rate,
        'n_test': len(test_draws),
        'n_success': n_success,
        'total_premio': total_premio,
        'total_custo': total_custo,
        'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0.0,
        'hit_distribution': hit_counts,
    }

# ============================================================
# GERADOR BRUTO (modo rápido / tentativas)
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
        rest = list(set(range(1, 26)) - banned)
        n_fix = len(fixed_set)

        max_semi = len(semi_set) if max_semifixed is None else min(max_semifixed, len(semi_set))
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None

        n_semi = random.randint(min_semi, max_semi)
        n_rest = 15 - n_fix - n_semi
        if n_rest < 0 or n_rest > len(rest):
            return None

        for _ in range(200):
            semi_pick = set(random.sample(list(semi_set), n_semi)) if n_semi > 0 else set()
            rest_pick = set(random.sample(rest, n_rest)) if n_rest > 0 else set()
            game = sorted(fixed_set | semi_pick | rest_pick)
            if len(game) != 15:
                continue

            if allowed_pares is not None and sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                continue
            if allowed_moldura is not None and sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                continue
            if allowed_primos is not None and sum(1 for x in game if x in PRIMES) not in allowed_primos:
                continue

            pares = sum(1 for x in game if x % 2 == 0)
            mol = sum(1 for x in game if x in MOLDURA)
            prim = sum(1 for x in game if x in PRIMES)
            soma = sum(game)
            amp = max(game) - min(game)
            cons = sum(1 for i in range(len(game) - 1) if game[i+1] - game[i] == 1)

            if range_pares is not None and not (range_pares[0] <= pares <= range_pares[1]):
                continue
            if range_moldura is not None and not (range_moldura[0] <= mol <= range_moldura[1]):
                continue
            if range_primos is not None and not (range_primos[0] <= prim <= range_primos[1]):
                continue
            if range_soma is not None and not (range_soma[0] <= soma <= range_soma[1]):
                continue
            if range_amplitude is not None and not (range_amplitude[0] <= amp <= range_amplitude[1]):
                continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= cons <= range_consecutivos[1]):
                continue

            return game
        return None

# ============================================================
# OTIMIZADOR DE CARTEIRA
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 excluded=None, allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                 range_pares=None, range_moldura=None, range_primos=None,
                 range_soma=None, range_amplitude=None, range_consecutivos=None,
                 groups=None, use_criteria=True):
        self.contests = contests
        self.generator = LooseGenerator()
        self.use_criteria = use_criteria
        self.excluded = excluded or []

        excl_set = set(self.excluded)
        self.fixed = fixed or []
        if excl_set & set(self.fixed):
            print(f"⚠️ Fixas também excluídas; removendo: {sorted(excl_set & set(self.fixed))}")
            self.fixed = [d for d in self.fixed if d not in excl_set]

        self.semifixed = semifixed or []
        if excl_set & set(self.semifixed):
            print(f"⚠️ Semifixas também excluídas; removendo: {sorted(excl_set & set(self.semifixed))}")
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

        self.groups = [
            {"dezenas": g["dezenas"], "dezenas_set": set(g["dezenas"]),
             "min": g["min"], "max": g["max"]}
            for g in (groups or [])
        ]

    # --------------------------------------------------------
    # Universo efetivo considerando semifixas (combinatória exata)
    # --------------------------------------------------------
    def _effective_universe_with_semifixas(self):
        """
        Calcula sum_{k in [lo, hi]} C(s, k) * C(f, n_free - k),
        onde:
          s = |semifixas após remover fixas/excluídas|
          f = |dezenas livres que NÃO são semifixas|
          n_free = 15 - |fixed|
        e k precisa satisfazer min_semifixed <= k <= max_semifixed (ou |semi| se None),
        além de k <= n_free e n_free - k <= f.
        """
        fixed_set = set(self.fixed)
        excl_set = set(self.excluded)
        semi_set = set(self.semifixed) - fixed_set - excl_set
        free = set(range(1, 26)) - fixed_set - excl_set
        non_semi_free = free - semi_set

        s = len(semi_set)
        f = len(non_semi_free)
        n_free = 15 - len(fixed_set)

        if n_free < 0:
            return 0

        min_k = max(0, self.min_semifixed)
        max_k = s if self.max_semifixed is None else min(self.max_semifixed, s)

        lo = max(min_k, n_free - f, 0)
        hi = min(max_k, n_free, s)
        if lo > hi:
            return 0
        return sum(comb(s, k) * comb(f, n_free - k) for k in range(lo, hi + 1))

    # --------------------------------------------------------
    # MODO RÁPIDO: tentativas até completar pool
    # --------------------------------------------------------
    def generate_pool(self, n_candidates):
        pool, seen = [], set()
        rej_crit = rej_grp = rej_dup = falhas = 0
        max_attempts = n_candidates * 20
        attempts = 0

        while len(pool) < n_candidates and attempts < max_attempts:
            attempts += 1
            if attempts % 100000 == 0:
                print(f"   [{attempts}] pool={len(pool)}/{n_candidates} "
                      f"(crit={rej_crit}, grp={rej_grp})")

            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed,
                    excluded=self.excluded,
                    allowed_pares=self.allowed_pares, allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares, range_moldura=self.range_moldura,
                    range_primos=self.range_primos,
                    range_soma=self.range_soma, range_amplitude=self.range_amplitude,
                    range_consecutivos=self.range_consecutivos,
                )
            except RuntimeError:
                falhas += 1
                continue

            if self.use_criteria and not passes_criteria(g):
                rej_crit += 1
                continue
            if not passes_groups(g, self.groups):
                rej_grp += 1
                continue

            key = tuple(g)
            if key in seen:
                rej_dup += 1
                continue
            seen.add(key)
            pool.append(g)

        print(f"\n📊 POOL (tentativas)")
        print(f"   Solicitado: {n_candidates}")
        print(f"   Obtido:     {len(pool)}")
        print(f"   Tentativas: {attempts}")
        print(f"   rej_crit={rej_crit} (aplicado={self.use_criteria}) "
              f"rej_grp={rej_grp} rej_dup={rej_dup} falhas={falhas}")
        if len(pool) < n_candidates:
            print("⚠️ Pool incompleto: limite de tentativas atingido.")
        return pool

    # --------------------------------------------------------
    # MODO EXATO: enumera universo + reservoir sampling O(k)
    # --------------------------------------------------------
    def enumerate_and_sample(self, n_sample, seed=None):
        """
        Enumera TODAS as combinações efetivas (após fixas/excluídas),
        aplica semifixas, allowed_*, ranges, grupos e critérios.
        Amostra n_sample sem reposição via reservoir sampling.

        Memória: O(k) onde k = n_sample (NÃO O(universo)).
        """
        rng = np.random.default_rng(seed)

        fixed_set = set(self.fixed)
        excl_set = set(self.excluded)
        semi_set = set(self.semifixed) - fixed_set - excl_set
        free = sorted(set(range(1, 26)) - fixed_set - excl_set)
        n_free = 15 - len(fixed_set)

        universe_full = comb(25, 15)

        if n_free < 0:
            print(f"❌ Fixas ({len(fixed_set)}) > 15.")
            return [], {'error': 'fixed_gt_15'}
        if n_free > len(free):
            print(f"❌ Dezenas livres ({len(free)}) insuficientes para {n_free}.")
            return [], {'error': 'not_enough_free'}

        universe_effective = comb(len(free), n_free)
        universe_effective_semi = self._effective_universe_with_semifixas()

        min_semi = max(0, self.min_semifixed)
        max_semi = (len(semi_set) if self.max_semifixed is None
                    else min(self.max_semifixed, len(semi_set)))

        print(f"\n🔎 ENUMERAÇÃO DO UNIVERSO")
        print(f"   Fixas={sorted(fixed_set)} Excluídas={sorted(excl_set)}")
        print(f"   Semifixas disponíveis={sorted(semi_set)} "
              f"(min={min_semi}, max={max_semi})")
        print(f"   Dezenas livres: {len(free)} "
              f"(semi: {len(semi_set)}, não-semi: {len(free)-len(semi_set)})")
        print(f"   Livres no jogo: {n_free}")
        print(f"   Universo bruto C(25,15):              {universe_full:>15,}")
        print(f"   Efetivas C({len(free)},{n_free}):                   {universe_effective:>15,}")
        print(f"   Efetivas considerando semifixas:      {universe_effective_semi:>15,}")
        print(f"   Tempo pode variar de segundos a minutos.\n")

        n_valid = 0
        reservoir = []

        # Contadores por filtro (ordem de avaliação)
        rej = {
            'semifixas': 0,
            'pares_allowed': 0, 'moldura_allowed': 0, 'primos_allowed': 0,
            'pares_range': 0, 'moldura_range': 0, 'primos_range': 0,
            'soma': 0, 'amplitude': 0, 'consecutivos': 0,
            'grupos': 0, 'criteria': 0,
        }

        fixed_tuple = tuple(sorted(fixed_set)) if fixed_set else None
        need_set_for_semi = bool(semi_set)
        need_set_for_groups = bool(self.groups)

        t0 = time.time()
        for combo_free in tqdm(
            combinations(free, n_free),
            total=universe_effective,
            desc="Enumeração",
            unit=" combo",
            mininterval=0.5,
            miniters=10000,
        ):
            if fixed_tuple is not None:
                game = tuple(sorted(fixed_tuple + combo_free))
            else:
                game = combo_free

            # 1) Semifixas
            if need_set_for_semi:
                game_set = set(game)
                n_semi_in = len(game_set & semi_set)
                if n_semi_in < min_semi or n_semi_in > max_semi:
                    rej['semifixas'] += 1
                    continue
            elif not need_set_for_groups:
                game_set = None
            else:
                game_set = set(game)

            # Estatísticas (uma passada)
            p = m = pr = s = 0
            prev = game[0]
            c = 0
            for i, x in enumerate(game):
                if x % 2 == 0: p += 1
                if x in MOLDURA: m += 1
                if x in PRIMES: pr += 1
                s += x
                if i > 0 and x - prev == 1: c += 1
                prev = x
            a = game[-1] - game[0]

            # 2) allowed_*
            if self.allowed_pares is not None and p not in self.allowed_pares:
                rej['pares_allowed'] += 1; continue
            if self.allowed_moldura is not None and m not in self.allowed_moldura:
                rej['moldura_allowed'] += 1; continue
            if self.allowed_primos is not None and pr not in self.allowed_primos:
                rej['primos_allowed'] += 1; continue

            # 3) ranges
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

            # 4) grupos
            if need_set_for_groups:
                skip_g = False
                for g in self.groups:
                    qty = len(game_set & g["dezenas_set"])
                    if qty < g["min"] or qty > g["max"]:
                        skip_g = True
                        break
                if skip_g:
                    rej['grupos'] += 1
                    continue

            # 5) critérios
            if self.use_criteria:
                gm = BITMASK_CACHE.get_mask(game)
                skip_c = False
                for cmask, limit in CRITERIA_MASKS:
                    if (gm & cmask).bit_count() > limit:
                        skip_c = True
                        break
                if skip_c:
                    rej['criteria'] += 1
                    continue

            # válido
            n_valid += 1
            if len(reservoir) < n_sample:
                reservoir.append(list(game))
            else:
                j = int(rng.integers(0, n_valid))
                if j < n_sample:
                    reservoir[j] = list(game)

        elapsed = time.time() - t0

        # Checagem de consistência (só quando não há outro filtro ativo)
        has_other_filters = (
            self.allowed_pares is not None or self.allowed_moldura is not None
            or self.allowed_primos is not None
            or self.range_pares is not None or self.range_moldura is not None
            or self.range_primos is not None or self.range_soma is not None
            or self.range_amplitude is not None or self.range_consecutivos is not None
            or bool(self.groups) or self.use_criteria
        )
        if not has_other_filters and n_valid != universe_effective_semi:
            print(f"\n⚠️ INCONSISTÊNCIA INTERNA: n_valid={n_valid:,} "
                  f"≠ universo_semi={universe_effective_semi:,}")

        stats = {
            'universe_full': universe_full,
            'universe_effective': universe_effective,
            'universe_effective_semi': universe_effective_semi,
            'n_valid': n_valid,
            'requested': n_sample,
            'obtained': len(reservoir),
            'coverage_valid_over_full': n_valid / universe_full if universe_full else 0.0,
            'coverage_valid_over_effective_semi': (
                n_valid / universe_effective_semi if universe_effective_semi else 0.0
            ),
            'coverage_sample_over_valid': len(reservoir) / n_valid if n_valid else 0.0,
            'rejections': rej,
            'elapsed': elapsed,
        }
        return reservoir, stats

    # --------------------------------------------------------
    # COVERING (com clareza sobre dois estágios)
    # --------------------------------------------------------
    def select_covering(self, candidates, n_select, level='pair'):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")

        MAX_COVER = 50000 if level == 'pair' else 20000
        total_available = len(candidates)

        print(f"\n🔧 COVERING ({level})")
        print(f"   Entrada (pool): {total_available:,}")
        if total_available > MAX_COVER:
            print(f"   MAX_COVER: {MAX_COVER:,} → truncando (subamostra do pool)")
            working = random.sample(candidates, MAX_COVER)
        else:
            print(f"   Sem truncar (pool ≤ MAX_COVER)")
            working = list(candidates)

        t0 = time.time()
        masks = [_pairs_mask(c) if level == 'pair' else _triples_mask(c) for c in working]
        print(f"   Máscaras: {time.time()-t0:.2f}s")

        t1 = time.time()
        covered = 0
        selected = []
        remaining = set(range(len(masks)))
        for _ in range(n_select):
            if not remaining:
                break
            best_i, best_gain = -1, -1
            for i in remaining:
                gain = (masks[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i == -1:
                break
            selected.append(best_i)
            covered |= masks[best_i]
            remaining.discard(best_i)
        print(f"   Seleção: {time.time()-t1:.2f}s")
        return [working[i] for i in selected]

    # --------------------------------------------------------
    # OPTIMIZE
    # --------------------------------------------------------
    def optimize(self, n_games=5, n_candidates=600000, method='pair_covering',
                 exact_universe=False, seed=None):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | {method} | "
              f"pool={n_candidates:,} | crit={self.use_criteria} | grp={len(self.groups)}")
        print(f"   Modo: {'ENUMERAÇÃO EXATA' if exact_universe else 'TENTATIVAS'}")

        if exact_universe:
            pool, stats = self.enumerate_and_sample(n_candidates, seed=seed)
            if not pool:
                raise RuntimeError(f"Enumeração falhou: {stats}")

            print(f"\n📊 UNIVERSO E POOL")
            print(f"   Universo bruto (C(25,15)):         {stats['universe_full']:>15,}")
            print(f"   Efetivas (fixas/excluídas):        {stats['universe_effective']:>15,}")
            print(f"   Efetivas considerando semifixas:   {stats['universe_effective_semi']:>15,}")
            print(f"   Válidas (todos os filtros):        {stats['n_valid']:>15,}  "
                  f"({stats['coverage_valid_over_effective_semi']*100:.2f}% das efetivas c/ semi)")
            print(f"   Pool solicitado:                   {stats['requested']:>15,}")
            print(f"   Pool obtido:                       {stats['obtained']:>15,}")
            if stats['requested'] > stats['n_valid']:
                print(f"   ⚠️ Pool limitado ao universo válido ({stats['n_valid']:,})")
            print(f"   Cobertura sobre universo válido:   "
                  f"{stats['coverage_sample_over_valid']*100:.2f}%")
            print(f"   Tempo de enumeração:               {stats['elapsed']:.1f}s")

            print(f"\n📊 REJEIÇÕES POR FILTRO (ordem de avaliação)")
            rej = stats['rejections']
            total_rej = sum(rej.values())
            for k, v in rej.items():
                if v > 0:
                    print(f"   {k:<18} {v:>15,}")
            print(f"   {'Total rejeitado':<18} {total_rej:>15,}")
            print(f"   {'Aprovado':<18} {stats['n_valid']:>15,}")
            print(f"   ⚠️ Cada jogo é creditado apenas ao PRIMEIRO filtro que falhar.")
        else:
            pool = self.generate_pool(n_candidates)

        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")

        if method == 'pair_covering':
            portfolio = self.select_covering(pool, n_games, level='pair')
        elif method == 'triple_covering':
            portfolio = self.select_covering(pool, n_games, level='triple')
        else:
            portfolio = pool[:n_games]
        return portfolio

    def backtest(self, portfolio, test_draws):
        return backtest_portfolio(portfolio, test_draws)

    def backtest_vs_null(self, portfolio, test_draws, n_null=1000, seed=42,
                         metric='draws_with_win_rate',
                         fixed=None, excluded=None):
        rng = np.random.default_rng(seed)
        real_bt = self.backtest(portfolio, test_draws)
        real_val = real_bt[metric]
        conditioned = (fixed is not None) or (excluded is not None)
        null_vals = []

        for _ in range(n_null):
            try:
                if conditioned:
                    null_port = random_portfolio_conditioned(
                        len(portfolio), fixed, excluded, rng
                    )
                else:
                    null_port = random_portfolio_unrestricted(len(portfolio), rng)
                bt = self.backtest(null_port, test_draws)
                null_vals.append(bt[metric])
            except Exception:
                continue

        if not null_vals:
            print("⚠️ Não foi possível gerar nulos válidos.")
            return None

        null_arr = np.array(null_vals)
        p_value = (1.0 + float(np.sum(null_arr >= real_val))) / (len(null_arr) + 1.0)
        percentile = float(np.mean(null_arr < real_val) * 100.0)

        return {
            'real': real_bt,
            'real_metric': real_val,
            'null_mean': float(null_arr.mean()),
            'null_std': float(null_arr.std()),
            'null_p95': float(np.percentile(null_arr, 95)),
            'p_value': p_value,
            'percentile': percentile,
            'n_null': len(null_arr),
            'metric': metric,
            'conditioned': conditioned,
        }


def print_null_comparison(r, label="Carteira"):
    if r is None:
        return
    tag = "CONDICIONADO" if r['conditioned'] else "IRRESTRITO"
    print(f"\n🔬 BACKTEST vs. NULO ({tag}) — {label}")
    print(f"   Real: {r['real_metric']:.4f}")
    print(f"   Nulo (n={r['n_null']}): média={r['null_mean']:.4f}, "
          f"std={r['null_std']:.4f}, P95={r['null_p95']:.4f}")
    print(f"   Percentil: {r['percentile']:.1f}%")
    sig = "🔍" if r['p_value'] < 0.05 else "  "
    print(f"   p-valor: {r['p_value']:.4f} {sig}")

# ============================================================
# INPUT E VALIDAÇÃO DE GRUPOS
# ============================================================
def input_groups(fixed=None, semifixed=None, excluded=None):
    s = input("\n   Quantidade de grupos (ENTER para nenhum): ").strip()
    if not s:
        return []
    try:
        n = int(s)
        if n <= 0:
            return []
    except:
        print("   ⚠️ Inválido, nenhum grupo.")
        return []

    fixed_set = set(fixed or [])
    excl_set = set(excluded or [])
    groups = []

    for i in range(n):
        print(f"\n   GRUPO {i+1}")
        dz_s = input("   Dezenas: ").strip()
        try:
            dz = [int(x) for x in dz_s.split()]
            if not dz or any(x < 1 or x > 25 for x in dz):
                print(f"   ⚠️ Grupo {i+1} inválido.")
                continue
        except:
            print(f"   ⚠️ Grupo {i+1} inválido.")
            continue

        dset = set(dz)
        n_grp = len(dset)
        try:
            mn = int(input("   Mínimo no jogo: ").strip())
            mx = int(input("   Máximo no jogo: ").strip())
        except:
            print(f"   ⚠️ Limites inválidos.")
            continue

        if not 0 <= mn <= mx <= 15:
            print(f"   ⚠️ 0 ≤ min ≤ max ≤ 15. Ignorado.")
            continue
        if mx > n_grp:
            print(f"   ⚠️ Máximo {mx} > tamanho {n_grp}. Ignorado.")
            continue

        n_fx = len(fixed_set & dset)
        if n_fx > mx:
            print(f"   ⚠️ {n_fx} fixas > máximo {mx}. Ignorado.")
            continue
        if n_fx > mn:
            print(f"   ℹ️ Ajuste do mínimo {mn}→{n_fx} (fixas).")
            mn = n_fx

        n_ex = len(excl_set & dset)
        disp = n_grp - n_ex
        if mn > disp:
            print(f"   ⚠️ Mínimo {mn} > disponíveis {disp} (após excluídas). Ignorado.")
            continue
        if mx > disp:
            print(f"   ℹ️ Ajuste do máximo {mx}→{disp} (excluídas).")
            mx = disp

        groups.append({
            "dezenas": sorted(dset),
            "dezenas_set": set(dset),
            "min": mn,
            "max": mx,
        })
        print(f"   ✅ Grupo {i+1}: {n_grp}d, [{mn}-{mx}]")

    warns = validate_groups(groups, fixed or [], excluded or [])
    if warns:
        print("\n   ⚠️ CONFLITOS POTENCIAIS ENTRE GRUPOS:")
        for w in warns:
            print(f"      • {w}")
        cont = input("   Continuar mesmo assim? (s/n) [s]: ").strip().lower()
        if cont == 'n':
            return []

    return groups


def input_pool_size(default=600000):
    s = input(f"\n   Tamanho do pool [{default}]: ").strip()
    if not s:
        return default
    try:
        return max(5, int(s))
    except:
        return default

# ============================================================
# FILTROS PARA ANÁLISE
# ============================================================
def extract_filter(dz, name):
    d = sorted(dz)
    if name == 'pares':
        return sum(1 for x in d if x % 2 == 0)
    if name == 'moldura':
        return sum(1 for x in d if x in MOLDURA)
    if name == 'primos':
        return sum(1 for x in d if x in PRIMES)
    if name == 'soma':
        return sum(d)
    if name == 'consecutivos':
        return sum(1 for i in range(len(d) - 1) if d[i+1] - d[i] == 1)
    if name == 'amplitude':
        return max(d) - min(d)
    return 0


class PredictiveRanking:
    def __init__(self, contests):
        self.contests = contests

    def rank_predictive_power(self, block_sizes=None, verbose=True):
        block_sizes = block_sizes or [50, 100, 200, 500]
        filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']
        out = {}
        for bs in block_sizes:
            for f in filters:
                s = np.array([extract_filter(c['dezenas'], f) for c in self.contests], dtype=float)
                nb = len(s) // bs
                if nb < 3:
                    continue
                blocks = [s[i*bs:(i+1)*bs] for i in range(nb)]
                for strat in ['reversao', 'tendencia']:
                    ac = tot = 0
                    for i in range(2, len(blocks)):
                        mp = np.mean(blocks[i-1]); mc = np.mean(blocks[i])
                        mh = np.mean(s[:(i-1)*bs]) if (i-1)*bs > 0 else mp
                        tot += 1
                        if strat == 'reversao':
                            d = mp > mh
                            if (d and mc < mp) or (not d and mc > mp):
                                ac += 1
                        else:
                            if (mp > mh and mc > mp) or (mp < mh and mc < mp):
                                ac += 1
                    acc = ac / tot * 100 if tot else 0
                    p = binomtest(ac, tot, 0.5, alternative='greater').pvalue if tot else 1.0
                    out[(f, strat, bs)] = {'accuracy': acc, 'acertos': ac, 'total': tot, 'p_value': p}
                    if verbose:
                        sig = "🔍" if p < 0.05 else ("📊" if p < 0.15 else "  ")
                        print(f"{f:<15} {strat:<12} {acc:<10.1f}% {ac}/{tot:<10} {p:<10.4f} {sig}")
        return out

    def rank_dezenas_individual(self, block_sizes=None):
        block_sizes = block_sizes or [100, 200, 500]
        print(f"\n📊 PODER PREDITIVO DAS 25 DEZENAS")
        for bs in block_sizes:
            accs = []
            for dz in range(1, 26):
                s = np.array([1 if dz in c['dezenas'] else 0 for c in self.contests], dtype=float)
                nb = len(s) // bs
                if nb < 3:
                    continue
                blocks = [s[i*bs:(i+1)*bs] for i in range(nb)]
                ac = tot = 0
                for i in range(2, len(blocks)):
                    fp = np.mean(blocks[i-1]); fc = np.mean(blocks[i])
                    fh = np.mean(s[:(i-1)*bs]) if (i-1)*bs > 0 else fp
                    tot += 1
                    if (fp > fh and fc < fp) or (fp < fh and fc > fp):
                        ac += 1
                if tot:
                    accs.append(ac / tot * 100)
            if accs:
                print(f"   Bloco {bs}: média={np.mean(accs):.1f}%, max={max(accs):.1f}%")


def monte_carlo_control(contests, n_simulations=1000, block_sizes=None):
    block_sizes = block_sizes or [50, 100, 200]
    print(f"\n🎲 MONTE CARLO")
    real = PredictiveRanking(contests).rank_predictive_power(block_sizes, verbose=False)
    real_max = max(v['accuracy'] for v in real.values())
    real_key = max(real, key=lambda k: real[k]['accuracy'])

    sim_max = []
    for _ in tqdm(range(n_simulations), desc="MC"):
        sim = [{'dezenas': sorted(np.random.choice(range(1, 26), 15, replace=False))}
               for _ in range(len(contests))]
        res = PredictiveRanking(sim).rank_predictive_power(block_sizes, verbose=False)
        sim_max.append(max(v['accuracy'] for v in res.values()))

    p_glob = np.mean(np.array(sim_max) >= real_max)
    print(f"\n🌟 Melhor real: {real_key} = {real_max:.1f}%")
    print(f"   p-global: {p_glob:.4f} {'🔍' if p_glob < 0.05 else '❌'}")


def test_concurso_a_concurso(contests, min_hist=200):
    print(f"\n🎯 TESTE CONCURSO A CONCURSO")
    for f in ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']:
        s = np.array([extract_filter(c['dezenas'], f) for c in contests], dtype=float)
        ac_r = ac_t = tot = 0
        for t in range(min_hist, len(contests) - 1):
            hist = s[:t]
            if len(hist) < 20:
                continue
            ms, ml = np.mean(hist[-20:]), np.mean(hist)
            tot += 1
            cur, nxt = s[t], s[t+1]
            d = ms > ml
            if (d and nxt < cur) or (not d and nxt > cur):
                ac_r += 1
            if (ms > ml and nxt > cur) or (ms < ml and nxt < cur):
                ac_t += 1
        if tot:
            print(f"{f:<15}: Rev={ac_r/tot*100:.1f}% ({ac_r}/{tot}) | "
                  f"Tend={ac_t/tot*100:.1f}% ({ac_t}/{tot})")


class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests

    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR ({method})")
        info = {
            'pares': {'min': 3, 'max': 12}, 'moldura': {'min': 6, 'max': 15},
            'primos': {'min': 2, 'max': 9}, 'soma': {'min': 120, 'max': 270},
            'amplitude': {'min': 10, 'max': 24}, 'consecutivos': {'min': 0, 'max': 12},
        }
        r = {}
        for f, i in info.items():
            s = np.array([extract_filter(c['dezenas'], f) for c in self.contests], dtype=float)
            if method == 'recent':
                rc = s[-50:]
                lo = max(i['min'], int(np.percentile(rc, 35)))
                hi = min(i['max'], int(np.percentile(rc, 65)))
                r[f] = (lo, hi)
                print(f"   {f:<15}: [{lo}, {hi}]")
            else:
                fs = np.mean(s[-20:])
                fl = np.mean(s[-500:]) if len(s) >= 500 else np.mean(s)
                ipe = (fs - fl) / fl * 100 if fl > 0 else 0
                p = max(i['min'], int(fl)) if ipe > 5 else (min(i['max'], int(fl)+1) if ipe < -5 else int(fl))
                r[f] = (p - 1, p + 1)
                print(f"   {f:<15}: {r[f]} (IPE={ipe:+.1f}%)")
        return r

# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_structural(contests, train_size=500, test_size=50, step=50,
                            excluded=None, use_criteria=False):
    print(f"\n🔬 WALK-FORWARD ESTRUTURAL")
    print(f"   Critérios: {'ATIVOS' if use_criteria else 'DESATIVADOS (seguro)'}")
    rates = []
    start = train_size
    while start + test_size <= len(contests):
        train = contests[start-train_size:start]
        test = contests[start:start+test_size]
        ranges = StructuralPredictor(train).predict_ranges(method='recent')
        opt = PortfolioOptimizer(
            train, excluded=excluded, use_criteria=use_criteria,
            range_pares=ranges.get('pares'), range_moldura=ranges.get('moldura'),
            range_primos=ranges.get('primos'), range_soma=ranges.get('soma'),
            range_amplitude=ranges.get('amplitude'),
            range_consecutivos=ranges.get('consecutivos'),
        )
        try:
            p = opt.optimize(5, 10000, method='pair_covering')
            bt = opt.backtest(p, test)
            rates.append(bt['draws_with_win_rate'])
            print(f"   Janela {start}: rate={bt['draws_with_win_rate']:.3f}")
        except Exception as e:
            print(f"   Janela {start}: ERRO — {e}")
        start += step
    if rates:
        print(f"\n📊 Média rate: {np.mean(rates):.4f} ± {np.std(rates):.4f}")
    return rates

# ============================================================
# EXPERIMENTO DE TRÊS BRAÇOS
# ============================================================
def three_arm_experiment(
    contests, n_games=5, n_candidates=10000,
    n_replicates=100, test_size=200,
    fixed=None, semifixed=None, excluded=None,
    structural_ranges=None, groups=None,
    use_criteria_in_A=True,
):
    fixed = fixed or []
    semifixed = semifixed or []
    excluded = excluded or []
    groups = groups or []
    ranges = structural_ranges or {}

    if test_size >= len(contests):
        print("❌ test_size > histórico"); return None
    train = contests[:-test_size]
    test = contests[-test_size:]

    print(f"\n⚖️ EXPERIMENTO 3 BRAÇOS | n_replic={n_replicates} test={test_size}")
    rates = {'A': [], 'B': [], 'C': []}
    for rep in range(n_replicates):
        if (rep + 1) % 10 == 0:
            print(f"   réplica {rep+1}/{n_replicates}")

        try:
            random.seed(rep * 997 + 1); np.random.seed(rep * 997 + 1)
            opt = PortfolioOptimizer(
                train, fixed=fixed, semifixed=semifixed, excluded=excluded,
                range_pares=ranges.get('range_pares'),
                range_moldura=ranges.get('range_moldura'),
                range_primos=ranges.get('range_primos'),
                range_soma=ranges.get('range_soma'),
                range_amplitude=ranges.get('range_amplitude'),
                range_consecutivos=ranges.get('range_consecutivos'),
                groups=groups, use_criteria=use_criteria_in_A,
            )
            p = opt.optimize(n_games, n_candidates)
            bt = opt.backtest(p, test)
            rates['A'].append(bt['draws_with_win_rate'])
        except Exception as e:
            print(f"   A rep {rep+1}: {e}"); rates['A'].append(np.nan)

        try:
            random.seed(rep * 997 + 500000); np.random.seed(rep * 997 + 500000)
            opt = PortfolioOptimizer(
                train, fixed=fixed, semifixed=semifixed, excluded=excluded,
                range_pares=ranges.get('range_pares'),
                range_moldura=ranges.get('range_moldura'),
                range_primos=ranges.get('range_primos'),
                range_soma=ranges.get('range_soma'),
                range_amplitude=ranges.get('range_amplitude'),
                range_consecutivos=ranges.get('range_consecutivos'),
                groups=None, use_criteria=False,
            )
            p = opt.optimize(n_games, n_candidates)
            bt = opt.backtest(p, test)
            rates['B'].append(bt['draws_with_win_rate'])
        except Exception as e:
            print(f"   B rep {rep+1}: {e}"); rates['B'].append(np.nan)

        try:
            rng = np.random.default_rng(rep * 997 + 1000000)
            p = random_portfolio_conditioned(n_games, fixed, excluded, rng)
            bt = backtest_portfolio(p, test)
            rates['C'].append(bt['draws_with_win_rate'])
        except Exception as e:
            print(f"   C rep {rep+1}: {e}"); rates['C'].append(np.nan)

    print(f"\n📊 RESULTADOS (n={n_replicates})")
    labels = {'A': 'A: crit+grp+fix/estrut',
              'B': 'B: fix+estrut (sem crit/grp)',
              'C': 'C: fix/excl apenas (aleat. condic.)'}
    for arm in ('A', 'B', 'C'):
        arr = np.array([x for x in rates[arm] if not np.isnan(x)])
        if len(arr):
            print(f"{labels[arm]:<38} {arr.mean():.4f} ± {arr.std():.4f}")

    print(f"\n🔬 COMPARAÇÕES PAREADAS (H1: maior):")
    def paired(a, b):
        pairs = [(x, y) for x, y in zip(rates[a], rates[b])
                 if not (np.isnan(x) or np.isnan(y))]
        if len(pairs) < 3:
            print(f"   {a} > {b}: pares insuficientes"); return
        ra = np.array([p[0] for p in pairs]); rb = np.array([p[1] for p in pairs])
        t, p = ttest_rel(ra, rb, alternative='greater')
        sig = "🔍" if p < 0.05 else "  "
        print(f"   {a}>{b}: diff={np.mean(ra-rb):+.4f} t={t:+.2f} p={p:.4f} {sig}")
    paired('A', 'B'); paired('B', 'C'); paired('A', 'C')

    return {'rates': rates, 'labels': labels}

# ============================================================
# BUSCA OOS EM TRÊS FASES
# ============================================================
def search_best_fixed_oos(contests, n_fixed=3, top_n=20,
                          train_size=2500, val_size=500,
                          n_games=5, n_candidates=10000,
                          method='pair_covering', excluded=None):
    if train_size + val_size >= len(contests):
        print("❌ Histórico insuficiente."); return []
    train = contests[:train_size]
    val = contests[train_size:train_size + val_size]
    test = contests[train_size + val_size:]

    print(f"\n🔎 BUSCA DE FIXAS (Treino={train_size} Val={val_size} Teste={len(test)})")

    candidates = []
    for ft in tqdm(combinations(range(1, 26), n_fixed), desc="Treino"):
        if excluded and any(x in ft for x in excluded):
            continue
        fset = set(ft)
        ac = sum(1 for c in train if fset.issubset(set(c['dezenas'])))
        freq = ac / len(train)
        if freq >= 0.05:
            candidates.append((ft, freq))
    candidates.sort(key=lambda x: x[1], reverse=True)

    train_plus_val = train + val
    val_results = []
    for ft, freq in tqdm(candidates[:200], desc="Validação"):
        opt = PortfolioOptimizer(train_plus_val, fixed=list(ft),
                                 excluded=excluded, use_criteria=False)
        try:
            p = opt.optimize(n_games, n_candidates, method=method)
            bt = opt.backtest(p, val)
            val_results.append({'fixed': ft, 'val_roi': bt['roi'],
                                'val_rate': bt['draws_with_win_rate']})
        except:
            continue
    val_results.sort(key=lambda x: x['val_roi'], reverse=True)
    top = val_results[:top_n]

    final = []
    for r in top:
        opt = PortfolioOptimizer(train_plus_val, fixed=list(r['fixed']),
                                 excluded=excluded, use_criteria=False)
        try:
            p = opt.optimize(n_games, n_candidates, method=method)
            bt = opt.backtest(p, test)
            final.append({'fixed': r['fixed'], 'val_roi': r['val_roi'],
                          'test_roi': bt['roi'], 'test_rate': bt['draws_with_win_rate']})
        except:
            continue

    if final:
        print(f"\n📊 MÉDIA NO TESTE: rate={np.mean([r['test_rate'] for r in final]):.4f} "
              f"roi={np.mean([r['test_roi'] for r in final]):+.1f}%")
    return final

# ============================================================
# MENU
# ============================================================
def _ask_structural_ranges():
    r = {}
    print("   Faixas estruturais (ENTER para pular cada)")
    for key, label in [
        ('range_pares', 'Pares'), ('range_moldura', 'Moldura'),
        ('range_primos', 'Primos'), ('range_soma', 'Soma'),
        ('range_amplitude', 'Amplitude'), ('range_consecutivos', 'Consecutivos'),
    ]:
        try:
            v = input(f"   {label} min,max: ").strip()
            r[key] = tuple(int(x) for x in v.split(',')) if v else None
        except:
            r[key] = None
    return r


def main():
    print("=" * 70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.4.3")
    print("=" * 70)

    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ 'resultados_lotofacil.csv' não encontrado."); return
    print(f"\n📂 {len(contests)} concursos | último: {contests[-1]['concurso']}")

    try:
        load_criteria_games('jogos_criterios.txt')
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERRO: {e}\n   Geração cancelada."); return

    while True:
        print("\nOpções:")
        print("1.  Gerar carteira personalizada")
        print("2.  Walk-forward condicional")
        print("3.  Backtest nos últimos 200 (+ nulo)")
        print("4.  Buscar fixas OOS")
        print("5.  Comparar trincas")
        print("6.  Ranking preditivo")
        print("7.  Poder preditivo das 25 dezenas")
        print("8.  Structural Predictor")
        print("9.  Carteira estrutural")
        print("10. Monte Carlo")
        print("11. Teste concurso a concurso")
        print("12. Walk-forward estrutural")
        print("13. EXPERIMENTO 3 BRAÇOS")
        print("0.  Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            fs = input("\n   Fixas: ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            ss = input("   Semifixas: ").strip()
            semifixed = [int(x) for x in ss.split()] if ss else []
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []

            if semifixed:
                try:
                    mn = int(input(f"   Mín semi [0-{len(semifixed)}]: ").strip() or "0")
                    mx = int(input(f"   Máx semi [0-{len(semifixed)}]: ").strip() or str(len(semifixed)))
                except:
                    mn, mx = 0, len(semifixed)
            else:
                mn, mx = 0, None

            ranges = _ask_structural_ranges()
            groups = input_groups(fixed, semifixed, excluded)

            print("\n   Modo de amostragem:")
            print("     [1] Tentativas (rápido)")
            print("     [2] Universo exato (enumera todas as combinações efetivas,")
            print("         conta válidas, amostra SEM reposição via reservoir)")
            modo = input("   Escolha [1]: ").strip() or "1"
            exact = (modo == '2')

            default_pool = 50000 if exact else 600000
            pool = input_pool_size(default_pool)

            metodo = input("\n   Método [1.Pair 2.Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'

            opt = PortfolioOptimizer(
                contests, fixed=fixed, semifixed=semifixed,
                min_semifixed=mn, max_semifixed=mx,
                excluded=excluded, groups=groups, use_criteria=True,
                **ranges,
            )
            portfolio = opt.optimize(5, pool, method=method, exact_universe=exact)

            for i, g in enumerate(portfolio, 1):
                print(f" {i}. {g}")
            audit_portfolio(portfolio, groups, applied_criteria=True)

            if len(contests) > 200:
                null_r = opt.backtest_vs_null(
                    portfolio, contests[-200:],
                    n_null=500, fixed=fixed, excluded=excluded,
                )
                print_null_comparison(null_r, "Gerada")

        elif op == '2':
            fs = input("\n   Fixas: ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            metodo = input("   Método [1/2]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            rates = []
            for w in range(8):
                te = len(contests) - w * 50
                ts = te - 50
                t_end = ts; t_start = max(0, t_end - 400)
                if t_start >= t_end or ts >= te:
                    continue
                opt = PortfolioOptimizer(contests[t_start:t_end], fixed=fixed,
                                         excluded=excluded, use_criteria=False)
                p = opt.optimize(5, 10000, method=method)
                bt = opt.backtest(p, contests[ts:te])
                rates.append(bt['draws_with_win_rate'])
                print(f"   Janela {w}: rate={bt['draws_with_win_rate']:.3f}")
            if rates:
                print(f"\n📊 Média: {np.mean(rates):.4f}")

        elif op == '3':
            fs = input("\n   Fixas: ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            ranges = _ask_structural_ranges()
            groups = input_groups(fixed, [], excluded)

            print("\n   Modo: [1] Tentativas [2] Universo exato")
            modo = input("   Escolha [1]: ").strip() or "1"
            exact = (modo == '2')
            pool = input_pool_size(50000 if exact else 600000)

            metodo = input("   Método [1/2]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'

            opt = PortfolioOptimizer(
                contests, fixed=fixed, excluded=excluded,
                groups=groups, use_criteria=True, **ranges,
            )
            portfolio = opt.optimize(5, pool, method=method, exact_universe=exact)
            audit_portfolio(portfolio, groups, applied_criteria=True)
            null_r = opt.backtest_vs_null(
                portfolio, contests[-200:],
                n_null=500, fixed=fixed, excluded=excluded,
            )
            print_null_comparison(null_r, "Backtest 200")

        elif op == '4':
            try:
                nf = int(input("\n   Fixas (2-4): ").strip())
                if nf not in (2, 3, 4): continue
            except: continue
            top_n = int(input("   Top N [20]: ").strip() or "20")
            ts = int(input("   Treino [2500]: ").strip() or "2500")
            vs = int(input("   Validação [500]: ").strip() or "500")
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            search_best_fixed_oos(contests, nf, top_n, ts, vs, excluded=excluded)

        elif op == '5':
            s1 = input("\n   Trinca 1: ").strip()
            s2 = input("   Trinca 2: ").strip()
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            try:
                t1 = tuple(int(x) for x in s1.split())
                t2 = tuple(int(x) for x in s2.split())
                if len(t1) != 3 or len(t2) != 3: continue
            except: continue
            print(f"\n⚔️ COMPARAÇÃO DE TRINCAS (critérios desativados)")
            for i, t in enumerate([t1, t2], 1):
                opt = PortfolioOptimizer(contests, fixed=list(t),
                                         excluded=excluded, use_criteria=False)
                p = opt.optimize(5, 600000)
                bt = opt.backtest(p, contests[-200:])
                print(f"   Trinca {i} ({t}): rate={bt['draws_with_win_rate']:.4f}")

        elif op == '6':
            bs = input("\n   Blocos [50,100,200,500]: ").strip()
            blocks = [int(x) for x in bs.split(',')] if bs else [50, 100, 200, 500]
            PredictiveRanking(contests).rank_predictive_power(blocks)

        elif op == '7':
            bs = input("\n   Blocos [100,200,500]: ").strip()
            blocks = [int(x) for x in bs.split(',')] if bs else [100, 200, 500]
            PredictiveRanking(contests).rank_dezenas_individual(blocks)

        elif op == '8':
            m = 'recent' if (input("   [1] Recente [2] IPE: ").strip() or "1") == "1" else 'ipe'
            StructuralPredictor(contests).predict_ranges(method=m)

        elif op == '9':
            ranges = StructuralPredictor(contests).predict_ranges(method='recent')
            fixed = [15, 16, 20]
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            groups = input_groups(fixed, [], excluded)

            print("\n   Modo: [1] Tentativas [2] Universo exato")
            modo = input("   Escolha [1]: ").strip() or "1"
            exact = (modo == '2')
            pool = input_pool_size(50000 if exact else 600000)

            metodo = input("   Método [1/2]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'

            opt = PortfolioOptimizer(
                contests, fixed=fixed, excluded=excluded,
                range_pares=ranges.get('pares'), range_moldura=ranges.get('moldura'),
                range_primos=ranges.get('primos'), range_soma=ranges.get('soma'),
                range_amplitude=ranges.get('amplitude'),
                range_consecutivos=ranges.get('consecutivos'),
                groups=groups, use_criteria=True,
            )
            portfolio = opt.optimize(5, pool, method=method, exact_universe=exact)
            audit_portfolio(portfolio, groups, applied_criteria=True)
            if len(contests) > 200:
                null_r = opt.backtest_vs_null(
                    portfolio, contests[-200:],
                    n_null=500, fixed=fixed, excluded=excluded,
                )
                print_null_comparison(null_r, "Estrutural")

        elif op == '10':
            try: ns = int(input("\n   Simulações [1000]: ").strip() or "1000")
            except: ns = 1000
            bs = input("   Blocos [50,100,200]: ").strip()
            blocks = [int(x) for x in bs.split(',')] if bs else [50, 100, 200]
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
            except:
                ts, te, st = 500, 50, 50
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            walk_forward_structural(contests, ts, te, st, excluded=excluded, use_criteria=False)

        elif op == '13':
            print("\n⚖️ EXPERIMENTO 3 BRAÇOS")
            try:
                ng = int(input("   n_games [5]: ").strip() or "5")
                nc = int(input("   n_candidates [10000]: ").strip() or "10000")
                nr = int(input("   n_replicates [100]: ").strip() or "100")
                tsz = int(input("   test_size [200]: ").strip() or "200")
            except:
                ng, nc, nr, tsz = 5, 10000, 100, 200

            fs = input("   Fixas (todos os braços): ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            es = input("   Excluídas (todos os braços): ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            ranges = _ask_structural_ranges()
            groups = input_groups(fixed, [], excluded)
            use_crit = input("\n   Braço A usa critérios? (s/n) [s]: ").strip().lower() != 'n'

            three_arm_experiment(
                contests, n_games=ng, n_candidates=nc,
                n_replicates=nr, test_size=tsz,
                fixed=fixed, excluded=excluded,
                structural_ranges=ranges, groups=groups,
                use_criteria_in_A=use_crit,
            )

        elif op == '0':
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
