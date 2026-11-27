#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.8.0

EVOLUÇÃO DO v49.7.0:
✅ Módulo de ATRASO DE COMBINAÇÕES (k=3..6) — menu 16
✅ CombinationDelayAnalyzer: z-score, %ile normal, tendência 30/100
✅ Welford incremental para mean/var (sem armazenar gaps)
✅ rank_combination_delays: ranking por z_score / delay / ratio / max_ratio
✅ Exportação CSV opcional
✅ Aviso estatístico explícito: atraso NÃO é previsão

EVOLUÇÃO DO v49.6.0:
✅ Joint Pair+Triple: score marginal log-utility para pares E trincas
✅ Concentração de pares e trincas calibrada (95º percentil do pool)
✅ Exposição normalizada: −2·Σ(c_d − t_d) / (18·n_games) → [−1,+1]
✅ Índices compactos (PAIR_INDEX/TRIPLE_INDEX + dict) pré-computados
✅ Busca local CONJUNTA por substituição, aceita só se score global melhora
✅ Experimento 4 braços (A clássico, B adaptive pair, C joint, D full)

EVOLUÇÃO DO v49.5.0:
✅ Amostragem de semifixas corrigida (weighted k DENTRO do retry loop)
✅ HistoricalProfile: regularização POR JANELA, depois mistura
✅ Score adaptativo NORMALIZADO
✅ Pair covering adaptativo usa log utility Σ log(1+c_p)
✅ target_counts extraído do pool válido
✅ backtest_vs_null aceita candidate_pool
"""

import numpy as np
from scipy.stats import hypergeom, binomtest, ttest_rel, norm as _norm
from collections import defaultdict
from itertools import combinations, permutations
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
LOG2 = np.log(2.0)

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

PAIR_DICT = {}
_idx = 0
for i in range(25):
    for j in range(i + 1, 25):
        PAIR_DICT[(i, j)] = _idx
        PAIR_DICT[(j, i)] = _idx
        _idx += 1
N_PAIRS = _idx

TRIPLE_DICT = {}
_idx = 0
for i, j, k in combinations(range(25), 3):
    for perm in permutations((i, j, k)):
        TRIPLE_DICT[perm] = _idx
    _idx += 1
N_TRIPLES = _idx

PAIR_UTIL_MAX_PER_GAME = 105 * LOG2
TRIPLE_UTIL_MAX_PER_GAME = 455 * LOG2
STRUCTURE_SCALE = 105 * 0.1


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


def _precompute_cand_indices(pool):
    n_cand = len(pool)
    cp = np.zeros((n_cand, 105), dtype=np.int16)
    ct = np.zeros((n_cand, 455), dtype=np.int16)
    cz = np.zeros((n_cand, 15), dtype=np.int8)
    for i, g in enumerate(pool):
        gs = sorted(x - 1 for x in g)
        p = 0
        for a in range(15):
            for b in range(a + 1, 15):
                cp[i, p] = PAIR_DICT[(gs[a], gs[b])]
                p += 1
        t = 0
        for a in range(15):
            for b in range(a + 1, 15):
                for c in range(b + 1, 15):
                    ct[i, t] = TRIPLE_DICT[(gs[a], gs[b], gs[c])]
                    t += 1
        cz[i] = gs
    return cp, ct, cz


def _compute_target_counts(pool, n_games):
    counts = np.zeros(25)
    for g in pool:
        for d in g:
            counts[d - 1] += 1
    return counts / len(pool) * n_games

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
        dset = g.get("dezenas_set") or set(g.get("dezenas", []))
        normalized.append({"idx": i, "dezenas_set": set(dset),
                           "min": g["min"], "max": g["max"]})
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
# MÉTRICAS DA CARTEIRA
# ============================================================
def portfolio_metrics(portfolio):
    if not portfolio:
        return {}
    pair_counts = defaultdict(int)
    triple_counts = defaultdict(int)
    dezena_counts = np.zeros(25)
    for g in portfolio:
        gs = sorted(g)
        for x in gs:
            dezena_counts[x - 1] += 1
        for a, b in combinations(gs, 2):
            pair_counts[(a, b)] += 1
        for t in combinations(gs, 3):
            triple_counts[t] += 1
    n_distinct_pairs = len(pair_counts)
    n_distinct_triples = len(triple_counts)
    total_pair_inc = sum(pair_counts.values())
    total_triple_inc = sum(triple_counts.values())
    overlaps = []
    for i in range(len(portfolio)):
        for j in range(i + 1, len(portfolio)):
            overlaps.append(len(set(portfolio[i]) & set(portfolio[j])))
    return {
        'n_distinct_pairs': n_distinct_pairs,
        'n_distinct_triples': n_distinct_triples,
        'avg_pair_repetition': total_pair_inc / max(1, n_distinct_pairs),
        'avg_triple_repetition': total_triple_inc / max(1, n_distinct_triples),
        'avg_game_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'max_game_overlap': int(max(overlaps)) if overlaps else 0,
        'dezena_counts': dezena_counts.tolist(),
        'dezena_std': float(dezena_counts.std()),
    }


def print_portfolio_metrics(portfolio, label=""):
    m = portfolio_metrics(portfolio)
    if not m:
        return
    hdr = f"📊 MÉTRICAS DA CARTEIRA{' (' + label + ')' if label else ''}"
    print(f"\n{hdr}")
    print(f"   Pares distintos:     {m['n_distinct_pairs']:>5} / 300")
    print(f"   Trincas distintas:   {m['n_distinct_triples']:>5} / 2300")
    print(f"   Repetição média de pares:   {m['avg_pair_repetition']:.2f}")
    print(f"   Repetição média de trincas: {m['avg_triple_repetition']:.2f}")
    print(f"   Sobreposição média:  {m['avg_game_overlap']:.2f}")
    print(f"   Sobreposição máxima: {m['max_game_overlap']}")
    print(f"   Desvio-padrão de exposição: {m['dezena_std']:.2f}")

# ============================================================
# CARREGAMENTO / PORTFÓLIOS ALEATÓRIOS / BACKTEST
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
                contests.append({'concurso': int(parts[0]), 'data': parts[1],
                                 'dezenas': sorted(dz)})
            except:
                continue
    contests.sort(key=lambda x: x['concurso'])
    print(f"✅ {len(contests)} concursos válidos")
    return contests


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


def backtest_portfolio(portfolio, test_draws):
    if not portfolio or not test_draws:
        return {'empirical': 0.0, 'draws_with_win': 0, 'draws_with_win_rate': 0.0,
                'n_test': len(test_draws) if test_draws else 0, 'n_success': 0,
                'total_premio': 0.0, 'total_custo': 0.0, 'roi': 0.0,
                'hit_distribution': {k: 0 for k in range(11, 16)}}
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
    return {'empirical': rate, 'draws_with_win': draws_with_win,
            'draws_with_win_rate': rate, 'n_test': len(test_draws),
            'n_success': n_success, 'total_premio': total_premio,
            'total_custo': total_custo,
            'roi': (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0.0,
            'hit_distribution': hit_counts}

# ============================================================
# GERADOR BRUTO
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
                range_soma, range_amplitude, range_consecutivos)
            if g is not None:
                return g
        raise RuntimeError("Geração falhou com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      excluded, allowed_pares, allowed_moldura, allowed_primos,
                      range_pares, range_moldura, range_primos,
                      range_soma, range_amplitude, range_consecutivos):
        fixed = fixed or []; semifixed = semifixed or []; excluded = excluded or []
        excl_set = set(excluded)
        fixed_set = set(fixed) - excl_set
        semi_set = set(semifixed) - fixed_set - excl_set
        rest = list(set(range(1, 26)) - fixed_set - semi_set - excl_set)
        n_fix = len(fixed_set)
        n_free = 15 - n_fix
        if n_free < 0:
            return None
        max_semi = len(semi_set) if max_semifixed is None else min(max_semifixed, len(semi_set))
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None
        s = len(semi_set); f = len(rest)
        ks_list = list(range(min_semi, max_semi + 1))
        weights = []
        for k in ks_list:
            if 0 <= k <= s and 0 <= n_free - k <= f:
                weights.append(comb(s, k) * comb(f, n_free - k))
            else:
                weights.append(0)
        if sum(weights) == 0:
            return None
        for _ in range(200):
            n_semi = random.choices(ks_list, weights=weights, k=1)[0]
            n_rest = n_free - n_semi
            semi_pick = set(random.sample(list(semi_set), n_semi)) if n_semi > 0 else set()
            rest_pick = set(random.sample(rest, n_rest)) if n_rest > 0 else set()
            game = sorted(fixed_set | semi_pick | rest_pick)
            if len(game) != 15:
                continue
            pares = sum(1 for x in game if x % 2 == 0)
            mol = sum(1 for x in game if x in MOLDURA)
            prim = sum(1 for x in game if x in PRIMES)
            if allowed_pares is not None and pares not in allowed_pares: continue
            if allowed_moldura is not None and mol not in allowed_moldura: continue
            if allowed_primos is not None and prim not in allowed_primos: continue
            soma = sum(game); amp = max(game) - min(game)
            cons = sum(1 for i in range(len(game) - 1) if game[i+1] - game[i] == 1)
            if range_pares is not None and not (range_pares[0] <= pares <= range_pares[1]): continue
            if range_moldura is not None and not (range_moldura[0] <= mol <= range_moldura[1]): continue
            if range_primos is not None and not (range_primos[0] <= prim <= range_primos[1]): continue
            if range_soma is not None and not (range_soma[0] <= soma <= range_soma[1]): continue
            if range_amplitude is not None and not (range_amplitude[0] <= amp <= range_amplitude[1]): continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= cons <= range_consecutivos[1]): continue
            return game
        return None

# ============================================================
# PERFIL HISTÓRICO REGULARIZADO
# ============================================================
class HistoricalProfile:
    WINDOWS = (30, 100, 300)
    WEIGHTS = (0.5, 0.3, 0.2)
    DEZENA_THEORETICAL = 15 / 25
    PAIR_THEORETICAL = (15/25) * (14/24)

    def __init__(self, contests, alpha=50.0):
        self.alpha = float(alpha)
        self.n_contests = len(contests)
        self.dezena_freq = np.full(25, self.DEZENA_THEORETICAL)
        self.pair_freq = np.full((25, 25), self.PAIR_THEORETICAL)
        np.fill_diagonal(self.pair_freq, 0.0)
        self._fit(contests)

    def _fit(self, contests):
        n = len(contests)
        if n == 0:
            return
        dez_sum = np.zeros(25); pair_sum = np.zeros((25, 25))
        weight_sum = 0.0; a = self.alpha
        for w, weight in zip(self.WINDOWS, self.WEIGHTS):
            n_w = min(w, n)
            if n_w == 0:
                continue
            recent = contests[-n_w:]
            counts = np.zeros(25); pair_counts = np.zeros((25, 25))
            for c in recent:
                dz = sorted(x - 1 for x in c['dezenas'])
                for x in dz:
                    counts[x] += 1
                for i in range(len(dz)):
                    for j in range(i + 1, len(dz)):
                        pair_counts[dz[i], dz[j]] += 1
                        pair_counts[dz[j], dz[i]] += 1
            dez_sum += weight * (counts + a * self.DEZENA_THEORETICAL) / (n_w + a)
            pair_sum += weight * (pair_counts + a * self.PAIR_THEORETICAL) / (n_w + a)
            weight_sum += weight
        if weight_sum > 0:
            self.dezena_freq = dez_sum / weight_sum
            self.pair_freq = pair_sum / weight_sum
            self.pair_freq = (self.pair_freq + self.pair_freq.T) / 2
            np.fill_diagonal(self.pair_freq, 0.0)

    def summary(self):
        dev = self.dezena_freq - self.DEZENA_THEORETICAL
        order = np.argsort(dev)
        print(f"\n📊 HISTORICAL PROFILE (n={self.n_contests}, α={self.alpha:.0f})")
        print("   + frequentes vs teórico:")
        for idx in order[::-1][:5]:
            print(f"     {idx+1:2d}: {self.dezena_freq[idx]:.3f} ({dev[idx]:+.3f})")
        print("   − frequentes vs teórico:")
        for idx in order[:5]:
            print(f"     {idx+1:2d}: {self.dezena_freq[idx]:.3f} ({dev[idx]:+.3f})")

# ============================================================
# SCORER CLASSES
# ============================================================
class AdaptivePairScorer:
    def __init__(self, lambda_conc=0.05, c_max=None):
        self.lambda_conc = lambda_conc
        self.c_max = c_max

    def total_gain(self, pair_idx, pair_counts):
        vals = pair_counts[pair_idx]
        log_gain = np.log((vals + 2) / (vals + 1)).sum()
        if self.c_max is not None:
            after = np.maximum(0, vals + 1 - self.c_max) ** 2
            before = np.maximum(0, vals - self.c_max) ** 2
            log_gain -= self.lambda_conc * (after - before).sum()
        return log_gain


class AdaptiveTripleScorer:
    def __init__(self, lambda_conc=0.01, c_max=None):
        self.lambda_conc = lambda_conc
        self.c_max = c_max

    def total_gain(self, triple_idx, triple_counts):
        vals = triple_counts[triple_idx]
        log_gain = np.log((vals + 2) / (vals + 1)).sum()
        if self.c_max is not None:
            after = np.maximum(0, vals + 1 - self.c_max) ** 2
            before = np.maximum(0, vals - self.c_max) ** 2
            log_gain -= self.lambda_conc * (after - before).sum()
        return log_gain


class ExposureScorer:
    def __init__(self, target_counts, n_games):
        self.target = target_counts
        self.scale = 18.0 * n_games

    def marginal_gain(self, dezena_idx, dezena_counts):
        delta = dezena_counts[dezena_idx] - self.target[dezena_idx]
        return float(-2.0 * delta.sum() / self.scale)


class StructureScorer:
    def __init__(self, profile, n_games):
        self.scale = STRUCTURE_SCALE * n_games
        if profile is not None:
            self.bonus = profile.pair_freq - profile.PAIR_THEORETICAL
            np.fill_diagonal(self.bonus, 0.0)
        else:
            self.bonus = None

    def static_gain(self, game):
        if self.bonus is None:
            return 0.0
        gs = [x - 1 for x in game]
        s = 0.0
        for a in range(15):
            for b in range(a + 1, 15):
                s += self.bonus[gs[a], gs[b]]
        return s / self.scale

# ============================================================
# JOINT PORTFOLIO OPTIMIZER
# ============================================================
class JointPortfolioOptimizer:
    def __init__(self, pool, n_games, profile=None,
                 precomputed_base=None, seed=None,
                 lambda_pair_conc=0.05, lambda_triple_conc=0.01,
                 c_max_pair=None, c_max_triple=None):
        self.pool = pool
        self.n_games = n_games
        self.n_cand = len(pool)
        self.rng = random.Random(seed)

        t0 = time.time()
        if precomputed_base is not None:
            self.cand_pair_idx = precomputed_base['cand_pair_idx']
            self.cand_triple_idx = precomputed_base['cand_triple_idx']
            self.cand_dezena_idx = precomputed_base['cand_dezena_idx']
            self.target_counts = precomputed_base['target_counts']
        else:
            (self.cand_pair_idx, self.cand_triple_idx,
             self.cand_dezena_idx) = _precompute_cand_indices(pool)
            self.target_counts = _compute_target_counts(pool, n_games)
        print(f"   [joint] índices: {time.time()-t0:.2f}s")

        self.c_max_pair = c_max_pair if c_max_pair is not None else self._calib_cmax_pair()
        self.c_max_triple = c_max_triple if c_max_triple is not None else self._calib_cmax_triple()
        print(f"   [joint] c_max_pair={self.c_max_pair} c_max_triple={self.c_max_triple}")

        self.pair_scorer = AdaptivePairScorer(lambda_pair_conc, self.c_max_pair)
        self.triple_scorer = AdaptiveTripleScorer(lambda_triple_conc, self.c_max_triple)
        self.exposure_scorer = ExposureScorer(self.target_counts, n_games)
        self.structure_scorer = StructureScorer(profile, n_games)

        t0 = time.time()
        self.structure_gains = np.array(
            [self.structure_scorer.static_gain(g) for g in pool])
        print(f"   [joint] structure gains: {time.time()-t0:.2f}s")

    def _calib_cmax_pair(self):
        n_s = min(100, self.n_cand)
        maxs = []
        for _ in range(n_s):
            sample = self.rng.sample(self.pool, self.n_games)
            pc = np.zeros(N_PAIRS, dtype=int)
            for g in sample:
                gs = sorted(x - 1 for x in g)
                for a, b in combinations(gs, 2):
                    pc[PAIR_DICT[(a, b)]] += 1
            maxs.append(pc.max())
        return int(np.percentile(maxs, 95))

    def _calib_cmax_triple(self):
        n_s = min(50, self.n_cand)
        maxs = []
        for _ in range(n_s):
            sample = self.rng.sample(self.pool, self.n_games)
            tc = np.zeros(N_TRIPLES, dtype=int)
            for g in sample:
                gs = sorted(x - 1 for x in g)
                for a, b, c in combinations(gs, 3):
                    tc[TRIPLE_DICT[(a, b, c)]] += 1
            maxs.append(tc.max())
        return int(np.percentile(maxs, 95))

    def _gain(self, i, pair_counts, triple_counts, dezena_counts, weights):
        pg = self.pair_scorer.total_gain(self.cand_pair_idx[i], pair_counts)
        pg /= PAIR_UTIL_MAX_PER_GAME * self.n_games

        tg = self.triple_scorer.total_gain(self.cand_triple_idx[i], triple_counts)
        tg /= TRIPLE_UTIL_MAX_PER_GAME * self.n_games

        eg = self.exposure_scorer.marginal_gain(self.cand_dezena_idx[i], dezena_counts)
        sg = self.structure_gains[i]

        return (weights.get('pair', 0.0) * pg
                + weights.get('triple', 0.0) * tg
                + weights.get('exposure', 0.0) * eg
                + weights.get('structure', 0.0) * sg)

    def select(self, weights):
        pair_counts = np.zeros(N_PAIRS, dtype=int)
        triple_counts = np.zeros(N_TRIPLES, dtype=int)
        dezena_counts = np.zeros(25)
        selected = []
        remaining = set(range(self.n_cand))
        t0 = time.time()
        for _step in range(self.n_games):
            if not remaining:
                break
            best_i, best_score = -1, -np.inf
            for i in remaining:
                score = self._gain(i, pair_counts, triple_counts,
                                   dezena_counts, weights)
                if score > best_score:
                    best_score, best_i = score, i
            if best_i == -1:
                break
            selected.append(self.pool[best_i])
            pair_counts[self.cand_pair_idx[best_i]] += 1
            triple_counts[self.cand_triple_idx[best_i]] += 1
            dezena_counts[self.cand_dezena_idx[best_i]] += 1
            remaining.discard(best_i)
        print(f"   [joint] select: {time.time()-t0:.2f}s")
        return selected

    def score_portfolio(self, portfolio, weights):
        pc = np.zeros(N_PAIRS, dtype=int)
        tc = np.zeros(N_TRIPLES, dtype=int)
        dc = np.zeros(25)
        for g in portfolio:
            gs = sorted(x - 1 for x in g)
            for a, b in combinations(gs, 2):
                pc[PAIR_DICT[(a, b)]] += 1
            for a, b, c in combinations(gs, 3):
                tc[TRIPLE_DICT[(a, b, c)]] += 1
            for d in gs:
                dc[d] += 1

        pu = np.log(1.0 + pc).sum()
        if self.c_max_pair is not None:
            pu -= self.pair_scorer.lambda_conc * (
                np.maximum(0, pc - self.c_max_pair).astype(float) ** 2).sum()
        pu /= PAIR_UTIL_MAX_PER_GAME * self.n_games

        tu = np.log(1.0 + tc).sum()
        if self.c_max_triple is not None:
            tu -= self.triple_scorer.lambda_conc * (
                np.maximum(0, tc - self.c_max_triple).astype(float) ** 2).sum()
        tu /= TRIPLE_UTIL_MAX_PER_GAME * self.n_games

        eu = -np.sum((dc - self.target_counts) ** 2) / (
            2.0 * 15.0 * 0.6 * self.n_games)
        su = sum(self.structure_scorer.static_gain(g) for g in portfolio)

        return (weights.get('pair', 0.0) * pu
                + weights.get('triple', 0.0) * tu
                + weights.get('exposure', 0.0) * eu
                + weights.get('structure', 0.0) * su)

    def local_search(self, portfolio, weights, n_iter=2, max_candidates=50):
        if self.n_cand <= self.n_games:
            return portfolio
        sample_size = min(1500, self.n_cand)
        sample_idx = self.rng.sample(range(self.n_cand), sample_size)
        best = list(portfolio)
        best_q = self.score_portfolio(best, weights)
        print(f"\n   🔧 LOCAL SEARCH (joint, {n_iter} iters × {max_candidates} cand/jogo)")
        print(f"   Score inicial: {best_q:.5f}")
        for iteration in range(n_iter):
            improved = False
            for pos in range(self.n_games):
                trial_pos = self.rng.sample(sample_idx,
                                            min(max_candidates, len(sample_idx)))
                best_ri, best_q_t = -1, best_q
                for ri in trial_pos:
                    trial = best.copy()
                    trial[pos] = self.pool[ri]
                    q = self.score_portfolio(trial, weights)
                    if q > best_q_t:
                        best_q_t = q
                        best_ri = ri
                if best_ri >= 0:
                    best[pos] = self.pool[best_ri]
                    best_q = best_q_t
                    improved = True
            print(f"   iter {iteration+1}: score={best_q:.5f}"
                  + (" (melhorou)" if improved else " (sem melhora)"))
            if not improved:
                break
        return best

# ============================================================
# PORTFOLIO OPTIMIZER
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None,
                 min_semifixed=0, max_semifixed=None, excluded=None,
                 allowed_pares=None, allowed_moldura=None, allowed_primos=None,
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
            self.fixed = [d for d in self.fixed if d not in excl_set]
        self.semifixed = semifixed or []
        if excl_set & set(self.semifixed):
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

    def generate_pool(self, n_candidates):
        pool, seen = [], set()
        rej_crit = rej_grp = rej_dup = falhas = 0
        max_attempts = n_candidates * 20
        attempts = 0
        while len(pool) < n_candidates and attempts < max_attempts:
            attempts += 1
            if attempts % 100000 == 0:
                print(f"   [{attempts}] pool={len(pool)}/{n_candidates}")
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
                    range_consecutivos=self.range_consecutivos)
            except RuntimeError:
                falhas += 1
                continue
            if self.use_criteria and not passes_criteria(g):
                rej_crit += 1; continue
            if not passes_groups(g, self.groups):
                rej_grp += 1; continue
            key = tuple(g)
            if key in seen:
                rej_dup += 1; continue
            seen.add(key); pool.append(g)
        print(f"\n📊 POOL | Solicitado={n_candidates} Obtido={len(pool)} "
              f"Tentativas={attempts} rej_crit={rej_crit} rej_grp={rej_grp} "
              f"rej_dup={rej_dup} falhas={falhas}")
        return pool

    def select_covering(self, candidates, n_select, level='pair', max_cover=None):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        if max_cover is None:
            max_cover = 50000 if level == 'pair' else 20000
        if len(candidates) > max_cover:
            working = random.sample(candidates, max_cover)
        else:
            working = list(candidates)
        masks = [_pairs_mask(c) if level == 'pair' else _triples_mask(c) for c in working]
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
                    best_gain = gain; best_i = i
            if best_i == -1:
                break
            selected.append(best_i)
            covered |= masks[best_i]
            remaining.discard(best_i)
        return [working[i] for i in selected]

    def select_covering_adaptive(self, candidates, n_select, profile, level='pair',
                                 w_gain=1.0, w_hist=0.05,
                                 w_balance=0.15, w_overlap=0.10, max_cover=None):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        if max_cover is None:
            max_cover = 50000 if level == 'pair' else 20000
        if len(candidates) > max_cover:
            working = random.sample(candidates, max_cover)
        else:
            working = list(candidates)
        n_cand = len(working)

        target_counts = np.zeros(25)
        for g in working:
            for d in g:
                target_counts[d - 1] += 1
        target_counts = target_counts / n_cand * n_select

        bonus_mat = profile.pair_freq - profile.PAIR_THEORETICAL
        np.fill_diagonal(bonus_mat, 0.0)
        hist_bonus = np.zeros(n_cand)
        for idx, g in enumerate(working):
            hb = 0.0
            for a_idx in range(15):
                a = g[a_idx] - 1
                row = bonus_mat[a]
                for b_idx in range(a_idx + 1, 15):
                    hb += row[g[b_idx] - 1]
            hist_bonus[idx] = hb

        if level == 'pair':
            masks = None
            cand_pairs = [[(a - 1, b - 1) for a, b in combinations(sorted(g), 2)]
                          for g in working]
            pair_counts_sel = defaultdict(int)
        else:
            masks = [_triples_mask(c) for c in working]
            cand_pairs = None

        covered = 0
        selected_idx = []
        selected_sets = []
        dezena_counts = np.zeros(25)
        remaining = set(range(n_cand))
        PAIR_MAX_GAIN = 105 * LOG2
        TRIPLE_MAX_GAIN = 455.0

        for _step in range(n_select):
            if not remaining:
                break
            best_i, best_score = -1, -np.inf
            n_sel = len(selected_sets)
            for i in remaining:
                game = working[i]
                if level == 'pair':
                    log_gain = 0.0
                    for p in cand_pairs[i]:
                        c = pair_counts_sel[p]
                        log_gain += np.log((2 + c) / (1 + c))
                    gain_norm = log_gain / PAIR_MAX_GAIN
                else:
                    gain_norm = (masks[i] & ~covered).bit_count() / TRIPLE_MAX_GAIN
                hist_norm = hist_bonus[i] / 10.5
                delta_cost = 15.0
                for x in game:
                    delta_cost += 2.0 * (dezena_counts[x - 1] - target_counts[x - 1])
                balance_norm = delta_cost / 15.0
                gs = set(game)
                overlap_pen = 0.0
                for sg in selected_sets:
                    ov = len(gs & sg)
                    if ov > 8:
                        overlap_pen += (ov - 8) ** 2
                overlap_norm = overlap_pen / max(1, n_sel)
                score = (w_gain * gain_norm + w_hist * hist_norm
                         - w_balance * balance_norm - w_overlap * overlap_norm)
                if score > best_score:
                    best_score, best_i = score, i
            if best_i == -1:
                break
            selected_idx.append(best_i)
            selected_sets.append(set(working[best_i]))
            for x in working[best_i]:
                dezena_counts[x - 1] += 1
            if level == 'pair':
                for p in cand_pairs[best_i]:
                    pair_counts_sel[p] += 1
            else:
                covered |= masks[best_i]
            remaining.discard(best_i)
        return [working[i] for i in selected_idx]

    def optimize(self, n_games=5, n_candidates=600000, method='pair_covering',
                 exact_universe=False, seed=None,
                 use_local_search=True, ls_iter=3, ls_max_candidates=80,
                 joint_weights=None):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | {method} | pool={n_candidates:,}")
        if exact_universe:
            raise NotImplementedError("Enumeração exata requer v49.6")
        pool = self.generate_pool(n_candidates)
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")

        if method == 'pair_covering':
            portfolio = self.select_covering(pool, n_games, level='pair')
        elif method == 'pair_covering_adaptive':
            profile = HistoricalProfile(self.contests)
            profile.summary()
            portfolio = self.select_covering_adaptive(pool, n_games, profile, level='pair')
        elif method == 'triple_covering':
            portfolio = self.select_covering(pool, n_games, level='triple')
        elif method == 'joint':
            weights = joint_weights or {'pair': 0.40, 'triple': 0.35,
                                        'exposure': 0.15, 'structure': 0.10}
            profile = HistoricalProfile(self.contests) if weights.get('structure', 0) > 0 else None
            joint = JointPortfolioOptimizer(
                pool, n_games, profile=profile, seed=seed)
            portfolio = joint.select(weights)
            if use_local_search:
                portfolio = joint.local_search(
                    portfolio, weights, n_iter=ls_iter,
                    max_candidates=ls_max_candidates)
            return portfolio
        else:
            portfolio = pool[:n_games]
        return portfolio

    def backtest(self, portfolio, test_draws):
        return backtest_portfolio(portfolio, test_draws)

    def backtest_vs_null(self, portfolio, test_draws, n_null=1000, seed=42,
                         metric='draws_with_win_rate',
                         fixed=None, excluded=None, candidate_pool=None):
        rng = np.random.default_rng(seed)
        real_bt = self.backtest(portfolio, test_draws)
        real_val = real_bt[metric]
        conditioned = (candidate_pool is not None
                       or (fixed is not None) or (excluded is not None))
        null_vals = []
        for _ in range(n_null):
            try:
                if candidate_pool is not None:
                    if len(candidate_pool) < len(portfolio):
                        continue
                    null_port = random.sample(candidate_pool, len(portfolio))
                elif (fixed is not None) or (excluded is not None):
                    null_port = random_portfolio_conditioned(
                        len(portfolio), fixed, excluded, rng)
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
        return {'real': real_bt, 'real_metric': real_val,
                'null_mean': float(null_arr.mean()), 'null_std': float(null_arr.std()),
                'null_p95': float(np.percentile(null_arr, 95)),
                'p_value': p_value, 'percentile': percentile,
                'n_null': len(null_arr), 'metric': metric,
                'conditioned': conditioned, 'from_pool': candidate_pool is not None}


def print_null_comparison(r, label="Carteira"):
    if r is None:
        return
    if r.get('from_pool'):
        tag = "NULO DO MESMO POOL"
    elif r['conditioned']:
        tag = "CONDICIONADO (fix/excl)"
    else:
        tag = "IRRESTRITO"
    print(f"\n🔬 BACKTEST vs. NULO ({tag}) — {label}")
    print(f"   Real: {r['real_metric']:.4f}")
    print(f"   Nulo (n={r['n_null']}): média={r['null_mean']:.4f}, "
          f"std={r['null_std']:.4f}, P95={r['null_p95']:.4f}")
    print(f"   Percentil: {r['percentile']:.1f}%")
    sig = "🔍" if r['p_value'] < 0.05 else "  "
    print(f"   p-valor: {r['p_value']:.4f} {sig}")

# ============================================================
# INPUT HELPERS
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
        return []
    fixed_set = set(fixed or []); excl_set = set(excluded or [])
    groups = []
    for i in range(n):
        print(f"\n   GRUPO {i+1}")
        dz_s = input("   Dezenas: ").strip()
        try:
            dz = [int(x) for x in dz_s.split()]
            if not dz or any(x < 1 or x > 25 for x in dz):
                continue
        except:
            continue
        dset = set(dz); n_grp = len(dset)
        try:
            mn = int(input("   Mínimo no jogo: ").strip())
            mx = int(input("   Máximo no jogo: ").strip())
        except:
            continue
        if not 0 <= mn <= mx <= 15 or mx > n_grp:
            continue
        n_fx = len(fixed_set & dset)
        if n_fx > mx:
            continue
        if n_fx > mn:
            mn = n_fx
        n_ex = len(excl_set & dset)
        disp = n_grp - n_ex
        if mn > disp:
            continue
        if mx > disp:
            mx = disp
        groups.append({"dezenas": sorted(dset), "dezenas_set": set(dset),
                       "min": mn, "max": mx})
    warns = validate_groups(groups, fixed or [], excluded or [])
    if warns:
        print("\n   ⚠️ CONFLITOS:")
        for w in warns:
            print(f"      • {w}")
        if input("   Continuar? (s/n) [s]: ").strip().lower() == 'n':
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
# ANÁLISE ESTRUTURAL
# ============================================================
def extract_filter(dz, name):
    d = sorted(dz)
    if name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if name == 'moldura': return sum(1 for x in d if x in MOLDURA)
    if name == 'primos': return sum(1 for x in d if x in PRIMES)
    if name == 'soma': return sum(d)
    if name == 'consecutivos': return sum(1 for i in range(len(d) - 1) if d[i+1] - d[i] == 1)
    if name == 'amplitude': return max(d) - min(d)
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
                    out[(f, strat, bs)] = {'accuracy': acc, 'p_value': p, 'total': tot}
                    if verbose:
                        sig = "🔍" if p < 0.05 else ("📊" if p < 0.15 else "  ")
                        print(f"{f:<15} {strat:<12} {acc:<10.1f}% {ac}/{tot:<10} {p:<10.4f} {sig}")
        return out


class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests

    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR ({method})")
        info = {
            'pares': {'min': 3, 'max': 12}, 'moldura': {'min': 6, 'max': 15},
            'primos': {'min': 2, 'max': 9}, 'soma': {'min': 120, 'max': 270},
            'amplitude': {'min': 10, 'max': 24}, 'consecutivos': {'min': 0, 'max': 12}}
        r = {}
        for f, i in info.items():
            s = np.array([extract_filter(c['dezenas'], f) for c in self.contests], dtype=float)
            if method == 'recent':
                rc = s[-50:]
                lo = max(i['min'], int(np.percentile(rc, 35)))
                hi = min(i['max'], int(np.percentile(rc, 65)))
                r[f] = (lo, hi)
            else:
                fs = np.mean(s[-20:]); fl = np.mean(s[-500:]) if len(s) >= 500 else np.mean(s)
                ipe = (fs - fl) / fl * 100 if fl > 0 else 0
                p = max(i['min'], int(fl)) if ipe > 5 else (min(i['max'], int(fl)+1) if ipe < -5 else int(fl))
                r[f] = (p - 1, p + 1)
            print(f"   {f:<15}: {r[f]}")
        return r

# ============================================================
# v49.8 — RANKING DE ATRASO DE COMBINAÇÕES (k=3..6)
# ============================================================
class CombinationDelayAnalyzer:
    """
    Analisa atraso de combinações de k dezenas (k ∈ {3,4,5,6}).

    Para cada combinação que apareceu ao menos 1x:
      count, first_seen, last_seen, current_delay
      mean_gap (média empírica dos intervalos entre aparições)
      std_gap  (desvio-padrão empírico dos intervalos)
      max_gap  (maior intervalo já observado)
      z_score  = (current_delay - mean_gap) / std_gap
      percentile_approx via CDF normal

    ⚠️  Atraso NÃO é previsão. A probabilidade de uma combinação
        aparecer em um concurso justo é SEMPRE P_k, independente do atraso.
        O sorteio não compensa atraso.
    """

    NAMES = {3: 'TRINCAS', 4: 'QUADRAS', 5: 'QUINAS', 6: 'SENAS'}

    def __init__(self, contests, k=3, trend_windows=(30, 100)):
        assert 3 <= k <= 6, "k deve ser 3, 4, 5 ou 6"
        self.k = k
        self.contests = contests
        self.n_contests = len(contests)
        self.trend_windows = trend_windows
        self.contests_sets = [set(c['dezenas']) for c in contests]
        self.p_theoretical = comb(25 - k, 15 - k) / comb(25, 15)
        self.theoretical_gap = 1.0 / self.p_theoretical
        self.stats = {}
        t0 = time.time()
        self._fit()
        self.fit_time = time.time() - t0

    def _fit(self):
        stats = self.stats
        k = self.k
        for idx, c in enumerate(self.contests):
            dz = sorted(c['dezenas'])
            for combo in combinations(dz, k):
                s = stats.get(combo)
                if s is None:
                    stats[combo] = {
                        'count': 1, 'first_seen': idx, 'last_seen': idx,
                        'max_gap': 0, 'gap_count': 0,
                        'mean_gap': 0.0, 'M2': 0.0,
                    }
                else:
                    gap = idx - s['last_seen']
                    s['count'] += 1
                    s['gap_count'] += 1
                    gc = s['gap_count']
                    delta = gap - s['mean_gap']
                    s['mean_gap'] += delta / gc
                    s['M2'] += delta * (gap - s['mean_gap'])
                    if gap > s['max_gap']:
                        s['max_gap'] = gap
                    s['last_seen'] = idx

    def current_delay(self, s):
        return self.n_contests - 1 - s['last_seen']

    def std_gap(self, s):
        if s['gap_count'] < 2:
            return None
        return (s['M2'] / (s['gap_count'] - 1)) ** 0.5

    def z_score(self, s):
        sd = self.std_gap(s)
        if sd is None or sd == 0:
            return None
        return (self.current_delay(s) - s['mean_gap']) / sd

    def percentile_approx(self, s):
        z = self.z_score(s)
        if z is None:
            return None
        return float(_norm.cdf(z) * 100)

    def trend(self, combo):
        combo_set = set(combo)
        out = {}
        for w in self.trend_windows:
            if w > self.n_contests:
                continue
            n_w = min(w, self.n_contests)
            start = self.n_contests - n_w
            obs = sum(1 for cs in self.contests_sets[start:] if combo_set <= cs)
            exp = self.p_theoretical * n_w
            sd = (exp * (1 - self.p_theoretical)) ** 0.5
            z = (obs - exp) / sd if sd > 0 else 0.0
            arrow = '↑' if z >= 2 else ('↓' if z <= -2 else '→')
            out[w] = {'observed': obs, 'expected': exp, 'z': z, 'arrow': arrow}
        return out

    def rank(self, method='z_score', top_n=20, min_count=2):
        rows = []
        for combo, s in self.stats.items():
            if s['count'] < min_count:
                continue
            d = self.current_delay(s)
            if method == 'z_score':
                score = self.z_score(s)
            elif method == 'delay':
                score = d
            elif method == 'ratio':
                score = d / self.theoretical_gap
            elif method == 'max_ratio':
                score = d / s['max_gap'] if s['max_gap'] > 0 else None
            else:
                raise ValueError(f"method inválido: {method}")
            if score is None:
                continue
            rows.append((score, combo, s, d))
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows[:top_n]

    def summary(self):
        total = comb(25, self.k)
        return {
            'k': self.k, 'total_possible': total,
            'appeared_at_least_once': len(self.stats),
            'never_appeared': total - len(self.stats),
            'p_theoretical': self.p_theoretical,
            'theoretical_gap': self.theoretical_gap,
            'n_contests': self.n_contests, 'fit_time': self.fit_time,
        }

    def distribution_summary(self):
        if not self.stats:
            return None
        arr = np.array([self.current_delay(s) for s in self.stats.values()])
        return {
            'n': len(arr), 'mean': float(arr.mean()), 'std': float(arr.std()),
            'min': int(arr.min()), 'max': int(arr.max()),
            'p25': int(np.percentile(arr, 25)), 'p50': int(np.percentile(arr, 50)),
            'p75': int(np.percentile(arr, 75)), 'p90': int(np.percentile(arr, 90)),
            'p95': int(np.percentile(arr, 95)),
        }


def rank_combination_delays(contests, ks=(3, 4, 5, 6), top_n=20,
                            method='z_score', min_count=2, save_csv=False):
    if not contests:
        print("❌ Sem concursos.")
        return []

    method_names = {
        'z_score': 'Z-SCORE (atraso vs própria média histórica)',
        'delay': 'ATRASO ABSOLUTO (concursos desde a última aparição)',
        'ratio': 'RAZÃO vs INTERVALO TEÓRICO',
        'max_ratio': 'RAZÃO vs ATRASO MÁXIMO HISTÓRICO',
    }

    all_results = []
    for k in ks:
        print(f"\n{'='*78}")
        print(f"📊 RANKING DE ATRASO — {CombinationDelayAnalyzer.NAMES[k]} (k={k})")
        print(f"   Método: {method_names[method]} | min_count={min_count}")
        print(f"{'='*78}")

        analyzer = CombinationDelayAnalyzer(contests, k=k)
        print(f"⏱️  Indexação: {analyzer.fit_time:.1f}s")

        summ = analyzer.summary()
        print(f"\n📈 SUMÁRIO")
        print(f"   Universo total:       {summ['total_possible']:>10,}")
        print(f"   Apareceram ≥ 1x:      {summ['appeared_at_least_once']:>10,}")
        print(f"   Nunca apareceram:     {summ['never_appeared']:>10,}")
        print(f"   P(aparecer/concurso)  {summ['p_theoretical']*100:>10.3f}%")
        print(f"   Intervalo teórico:    {summ['theoretical_gap']:>10.2f} concursos")
        print(f"   Concursos analisados: {summ['n_contests']:>10,}")

        dist = analyzer.distribution_summary()
        if dist:
            print(f"\n📊 DISTRIBUIÇÃO DO ATRASO ATUAL (entre as que apareceram)")
            print(f"   Média={dist['mean']:.2f}  σ={dist['std']:.2f}  "
                  f"Min={dist['min']}  Máx={dist['max']}")
            print(f"   P25={dist['p25']}  P50={dist['p50']}  P75={dist['p75']}  "
                  f"P90={dist['p90']}  P95={dist['p95']}")

        rows = analyzer.rank(method=method, top_n=top_n, min_count=min_count)
        if not rows:
            print(f"\n⚠️  Nenhuma combinação com count ≥ {min_count}.")
            continue

        print(f"\n🏆 TOP {top_n} MAIS ATRASADAS")
        header = (f"{'#':>3} | {'Combinação':<20} | {'Última':>7} | {'Atraso':>6} | "
                  f"{'Média':>6} | {'σ':>6} | {'Máx':>4} | {'N':>5} | "
                  f"{'Score':>7} | {'%ile':>5} | {'T30':>3} | {'T100':>4}")
        print(header)
        print('-' * len(header))

        for i, (score, combo, s, d) in enumerate(rows, 1):
            combo_str = '-'.join(f"{x:02d}" for x in combo)
            ultima = contests[s['last_seen']]['concurso']
            mean_str = f"{s['mean_gap']:.2f}" if s['gap_count'] > 0 else "—"
            sd = analyzer.std_gap(s)
            sd_str = f"{sd:.2f}" if sd is not None else "—"
            pct = analyzer.percentile_approx(s)
            pct_str = f"{pct:.0f}" if pct is not None else "—"
            trend = analyzer.trend(combo)
            t30 = trend.get(30, {}).get('arrow', '—')
            t100 = trend.get(100, {}).get('arrow', '—')
            print(f"{i:>3} | {combo_str:<20} | {ultima:>7} | {d:>6} | "
                  f"{mean_str:>6} | {sd_str:>6} | {s['max_gap']:>4} | "
                  f"{s['count']:>5} | {score:>7.2f} | {pct_str:>5} | "
                  f"{t30:>3} | {t100:>4}")
            all_results.append({
                'k': k, 'rank': i, 'combo': combo_str, 'method': method,
                'last_concurso': ultima, 'current_delay': d,
                'mean_gap': s['mean_gap'], 'std_gap': sd, 'max_gap': s['max_gap'],
                'count': s['count'], 'score': float(score),
                'percentile_approx': pct, 'trend_30': t30, 'trend_100': t100,
            })

    if save_csv and all_results:
        try:
            import csv as _csv
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, f"ranking_atraso_{method}.csv")
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = _csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
                w.writeheader()
                for row in all_results:
                    w.writerow(row)
            print(f"\n💾 Salvo em: {path}")
        except Exception as e:
            print(f"\n⚠️  Erro ao salvar CSV: {e}")

    return all_results

# ============================================================
# WALK-FORWARD E EXPERIMENTOS
# ============================================================
def walk_forward_structural(contests, train_size=500, test_size=50, step=50,
                            excluded=None, use_criteria=False):
    print(f"\n🔬 WALK-FORWARD ESTRUTURAL")
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
            range_consecutivos=ranges.get('consecutivos'))
        try:
            p = opt.optimize(5, 10000, method='pair_covering')
            bt = opt.backtest(p, test)
            rates.append(bt['draws_with_win_rate'])
            print(f"   Janela {start}: rate={bt['draws_with_win_rate']:.3f}")
        except Exception as e:
            print(f"   Janela {start}: ERRO — {e}")
        start += step
    if rates:
        print(f"\n📊 Média: {np.mean(rates):.4f} ± {np.std(rates):.4f}")
    return rates


def four_arm_experiment(contests, n_games=5, n_candidates=10000,
                        n_replicates=10, test_size=200,
                        fixed=None, excluded=None, use_local_search=True):
    if test_size >= len(contests):
        print("❌ Histórico insuficiente."); return None
    train = contests[:-test_size]
    test = contests[-test_size:]
    fixed = fixed or []; excluded = excluded or []

    print(f"\n⚖️ EXPERIMENTO 4 BRAÇOS (v49.7)")
    print(f"   A: Pair Covering clássico")
    print(f"   B: Adaptive Pair (v49.6)")
    print(f"   C: Joint Pair+Triple (sem histórico)")
    print(f"   D: Joint Pair+Triple+Exposure+Structure")
    print(f"   n_rep={n_replicates}, pool={n_candidates:,}, test={test_size}, "
          f"LS={'on' if use_local_search else 'off'}")

    weights_C = {'pair': 0.50, 'triple': 0.50, 'exposure': 0.0, 'structure': 0.0}
    weights_D = {'pair': 0.40, 'triple': 0.35, 'exposure': 0.15, 'structure': 0.10}
    results = {'A': [], 'B': [], 'C': [], 'D': []}
    metrics = {'A': [], 'B': [], 'C': [], 'D': []}

    t_start = time.time()
    for rep in range(n_replicates):
        if (rep + 1) % max(1, n_replicates // 5) == 0:
            print(f"   réplica {rep+1}/{n_replicates} (elapsed={time.time()-t_start:.0f}s)")
        random.seed(rep * 997 + 1)
        np.random.seed(rep * 997 + 1)

        opt = PortfolioOptimizer(train, fixed=fixed, excluded=excluded, use_criteria=False)
        pool = opt.generate_pool(n_candidates)
        profile = HistoricalProfile(train)

        pA = opt.select_covering(pool, n_games, level='pair')
        pB = opt.select_covering_adaptive(pool, n_games, profile, level='pair')

        (cp, ct, cz) = _precompute_cand_indices(pool)
        tgt = _compute_target_counts(pool, n_games)
        shared_base = {'cand_pair_idx': cp, 'cand_triple_idx': ct,
                       'cand_dezena_idx': cz, 'target_counts': tgt}

        joint_C = JointPortfolioOptimizer(pool, n_games, profile=None,
                                          precomputed_base=shared_base,
                                          seed=rep * 997 + 7)
        pC = joint_C.select(weights_C)
        if use_local_search:
            pC = joint_C.local_search(pC, weights_C, n_iter=2, max_candidates=50)

        joint_D = JointPortfolioOptimizer(pool, n_games, profile=profile,
                                          precomputed_base=shared_base,
                                          seed=rep * 997 + 11)
        pD = joint_D.select(weights_D)
        if use_local_search:
            pD = joint_D.local_search(pD, weights_D, n_iter=2, max_candidates=50)

        for k, p in [('A', pA), ('B', pB), ('C', pC), ('D', pD)]:
            bt = opt.backtest(p, test)
            results[k].append(bt['draws_with_win_rate'])
            metrics[k].append(portfolio_metrics(p))

    print(f"\n📊 RESULTADOS OOS (draws_with_win_rate)")
    for k in 'ABCD':
        arr = np.array(results[k])
        print(f"   {k}: {arr.mean():.4f} ± {arr.std():.4f}")

    print(f"\n📊 MÉTRICAS COMBINATÓRIAS MÉDIAS")
    mkeys = ['n_distinct_pairs', 'n_distinct_triples',
             'avg_pair_repetition', 'avg_triple_repetition',
             'avg_game_overlap', 'dezena_std']
    for mk in mkeys:
        parts = []
        for k in 'ABCD':
            vals = [m[mk] for m in metrics[k] if m]
            if vals:
                parts.append(f"{k}={np.mean(vals):6.2f}")
        print(f"   {mk:<24} " + "  ".join(parts))

    print(f"\n🔬 COMPARAÇÕES PAREADAS (H1: maior)")
    def paired(a, b, label):
        if len(results[a]) < 3:
            return
        ra = np.array(results[a]); rb = np.array(results[b])
        t, p = ttest_rel(ra, rb, alternative='greater')
        sig = "🔍" if p < 0.05 else "  "
        print(f"   {label}: diff={np.mean(ra-rb):+.4f} t={t:+.2f} p={p:.4f} {sig}")
    paired('B', 'A', 'B (Adaptive Pair) > A (Classic)')
    paired('C', 'A', 'C (Joint) > A (Classic)')
    paired('D', 'C', 'D (Full) > C (Joint no history)')
    paired('D', 'B', 'D (Full) > B (Adaptive Pair)')
    print("   ⚠️ Réplicas compartilham o mesmo teste OOS.")
    print("      O p-valor mede robustez frente à semente, NÃO generalização temporal.")

    return {'results': results, 'metrics': metrics}


def three_arm_experiment(contests, n_games=5, n_candidates=10000,
                         n_replicates=100, test_size=200,
                         fixed=None, semifixed=None, excluded=None,
                         structural_ranges=None, groups=None,
                         use_criteria_in_A=True):
    fixed = fixed or []; semifixed = semifixed or []; excluded = excluded or []
    groups = groups or []; ranges = structural_ranges or {}
    if test_size >= len(contests):
        print("❌ test_size > histórico"); return None
    train = contests[:-test_size]; test = contests[-test_size:]
    rates = {'A': [], 'B': [], 'C': []}
    for rep in range(n_replicates):
        if (rep + 1) % 10 == 0:
            print(f"   réplica {rep+1}/{n_replicates}")
        try:
            random.seed(rep * 997 + 1); np.random.seed(rep * 997 + 1)
            opt = PortfolioOptimizer(train, fixed=fixed, semifixed=semifixed,
                                     excluded=excluded, **ranges, groups=groups,
                                     use_criteria=use_criteria_in_A)
            p = opt.optimize(n_games, n_candidates)
            rates['A'].append(opt.backtest(p, test)['draws_with_win_rate'])
        except Exception as e:
            rates['A'].append(np.nan)
        try:
            random.seed(rep * 997 + 500000); np.random.seed(rep * 997 + 500000)
            opt = PortfolioOptimizer(train, fixed=fixed, semifixed=semifixed,
                                     excluded=excluded, **ranges,
                                     groups=None, use_criteria=False)
            p = opt.optimize(n_games, n_candidates)
            rates['B'].append(opt.backtest(p, test)['draws_with_win_rate'])
        except Exception as e:
            rates['B'].append(np.nan)
        try:
            rng = np.random.default_rng(rep * 997 + 1000000)
            p = random_portfolio_conditioned(n_games, fixed, excluded, rng)
            rates['C'].append(backtest_portfolio(p, test)['draws_with_win_rate'])
        except Exception as e:
            rates['C'].append(np.nan)
    print(f"\n📊 RESULTADOS (n={n_replicates})")
    labels = {'A': 'A: crit+grp', 'B': 'B: fix+estrut', 'C': 'C: aleat. condic.'}
    for arm in ('A', 'B', 'C'):
        arr = np.array([x for x in rates[arm] if not np.isnan(x)])
        if len(arr):
            print(f"{labels[arm]:<20} {arr.mean():.4f} ± {arr.std():.4f}")
    def paired(a, b):
        pairs = [(x, y) for x, y in zip(rates[a], rates[b])
                 if not (np.isnan(x) or np.isnan(y))]
        if len(pairs) < 3:
            return
        ra = np.array([p[0] for p in pairs]); rb = np.array([p[1] for p in pairs])
        t, p = ttest_rel(ra, rb, alternative='greater')
        sig = "🔍" if p < 0.05 else "  "
        print(f"   {a}>{b}: diff={np.mean(ra-rb):+.4f} t={t:+.2f} p={p:.4f} {sig}")
    paired('A', 'B'); paired('B', 'C'); paired('A', 'C')
    return {'rates': rates, 'labels': labels}

# ============================================================
# MENU
# ============================================================
def _ask_structural_ranges():
    r = {}
    for key, label in [
        ('range_pares', 'Pares'), ('range_moldura', 'Moldura'),
        ('range_primos', 'Primos'), ('range_soma', 'Soma'),
        ('range_amplitude', 'Amplitude'), ('range_consecutivos', 'Consecutivos')]:
        try:
            v = input(f"   {label} min,max: ").strip()
            r[key] = tuple(int(x) for x in v.split(',')) if v else None
        except:
            r[key] = None
    return r


def _ask_covering_method():
    print("\n   Método:")
    print("     [1] Pair covering clássico")
    print("     [2] Pair covering ADAPTATIVO (v49.6)")
    print("     [3] Triple covering clássico")
    print("     [4] JOINT Pair+Triple+Exposure+Structure (v49.7)")
    s = input("   Escolha [1]: ").strip() or "1"
    return {'1': 'pair_covering', '2': 'pair_covering_adaptive',
            '3': 'triple_covering', '4': 'joint'}.get(s, 'pair_covering')


def _ask_joint_weights():
    print("\n   Pesos do joint (ENTER = 0.40/0.35/0.15/0.10):")
    s = input("   pair,triple,exposure,structure: ").strip()
    if not s:
        return {'pair': 0.40, 'triple': 0.35, 'exposure': 0.15, 'structure': 0.10}
    try:
        p, t, e, st = (float(x) for x in s.split(','))
        return {'pair': p, 'triple': t, 'exposure': e, 'structure': st}
    except:
        return {'pair': 0.40, 'triple': 0.35, 'exposure': 0.15, 'structure': 0.10}


def main():
    print("=" * 70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.8.0")
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
        print("9.  Carteira estrutural (SEM vazamento temporal)")
        print("10. Monte Carlo")
        print("11. Teste concurso a concurso")
        print("12. Walk-forward estrutural")
        print("13. EXPERIMENTO 3 BRAÇOS")
        print("14. COMPARAR Pair Clássico vs Adaptive (v49.6)")
        print("15. EXPERIMENTO 4 BRAÇOS (v49.7)")
        print("16. RANKING DE ATRASO (trincas/quadras/quinas/senas) v49.8")
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
            pool = input_pool_size(50000)
            method = _ask_covering_method()
            weights = _ask_joint_weights() if method == 'joint' else None
            ls = (input("   Local search? (s/n) [s]: ").strip().lower() != 'n')

            opt = PortfolioOptimizer(contests, fixed=fixed, semifixed=semifixed,
                                     min_semifixed=mn, max_semifixed=mx,
                                     excluded=excluded, groups=groups,
                                     use_criteria=True, **ranges)
            portfolio = opt.optimize(5, pool, method=method,
                                     use_local_search=ls, joint_weights=weights)
            for i, g in enumerate(portfolio, 1):
                print(f" {i}. {g}")
            audit_portfolio(portfolio, groups, applied_criteria=True)
            print_portfolio_metrics(portfolio, "gerada")
            if len(contests) > 200:
                pool_bt = opt.generate_pool(5000)
                null_r = opt.backtest_vs_null(
                    portfolio, contests[-200:], n_null=500,
                    fixed=fixed, excluded=excluded, candidate_pool=pool_bt)
                print_null_comparison(null_r, "Gerada")

        elif op == '9':
            print("\n   ⚠️ v49.8: treino/teste estanques")
            train = contests[:-200]; test = contests[-200:]
            ranges = StructuralPredictor(train).predict_ranges(method='recent')
            fixed = [15, 16, 20]
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            groups = input_groups(fixed, [], excluded)
            pool = input_pool_size(50000)
            method = _ask_covering_method()
            weights = _ask_joint_weights() if method == 'joint' else None
            ls = (input("   Local search? (s/n) [s]: ").strip().lower() != 'n')

            opt = PortfolioOptimizer(
                train, fixed=fixed, excluded=excluded,
                range_pares=ranges.get('pares'), range_moldura=ranges.get('moldura'),
                range_primos=ranges.get('primos'), range_soma=ranges.get('soma'),
                range_amplitude=ranges.get('amplitude'),
                range_consecutivos=ranges.get('consecutivos'),
                groups=groups, use_criteria=False)
            portfolio = opt.optimize(5, pool, method=method,
                                     use_local_search=ls, joint_weights=weights)
            audit_portfolio(portfolio, groups, applied_criteria=False)
            print_portfolio_metrics(portfolio, "estrutural OOS")
            bt = opt.backtest(portfolio, test)
            print(f"\n📊 BACKTEST OOS: rate={bt['draws_with_win_rate']:.4f} "
                  f"roi={bt['roi']:+.1f}%")

        elif op == '12':
            try:
                ts = int(input("\n   Treino [500]: ").strip() or "500")
                te = int(input("   Teste [50]: ").strip() or "50")
                st = int(input("   Passo [50]: ").strip() or "50")
            except:
                ts, te, st = 500, 50, 50
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            walk_forward_structural(contests, ts, te, st, excluded=excluded)

        elif op == '13':
            try:
                ng = int(input("\n   n_games [5]: ").strip() or "5")
                nc = int(input("   n_candidates [10000]: ").strip() or "10000")
                nr = int(input("   n_replicates [100]: ").strip() or "100")
                tsz = int(input("   test_size [200]: ").strip() or "200")
            except:
                ng, nc, nr, tsz = 5, 10000, 100, 200
            fs = input("   Fixas: ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            ranges = _ask_structural_ranges()
            groups = input_groups(fixed, [], excluded)
            use_crit = input("   Braço A usa critérios? (s/n) [s]: ").strip().lower() != 'n'
            three_arm_experiment(contests, n_games=ng, n_candidates=nc,
                                 n_replicates=nr, test_size=tsz,
                                 fixed=fixed, excluded=excluded,
                                 structural_ranges=ranges, groups=groups,
                                 use_criteria_in_A=use_crit)

        elif op == '14':
            print("\n⚖️ COMPARAÇÃO v49.6: Pair Clássico vs Adaptive")
            if len(contests) < 400:
                print("   ❌ Histórico insuficiente."); continue
            train = contests[:-200]; test = contests[-200:]
            fs = input("   Fixas: ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            es = input("   Excluídas: ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            pool_size = min(int(input("   Pool [20000]: ").strip() or "20000"), 50000)
            n_rep = int(input("   Réplicas [20]: ").strip() or "20")
            results = {'classic': [], 'adaptive': []}
            for rep in range(n_rep):
                random.seed(rep * 997 + 1); np.random.seed(rep * 997 + 1)
                opt = PortfolioOptimizer(train, fixed=fixed, excluded=excluded,
                                         use_criteria=False)
                pool = opt.generate_pool(pool_size)
                common = pool if len(pool) <= 50000 else random.sample(pool, 50000)
                p_c = opt.select_covering(common, 5, level='pair')
                p_a = opt.select_covering_adaptive(common, 5,
                                                    HistoricalProfile(train), level='pair')
                results['classic'].append(opt.backtest(p_c, test)['draws_with_win_rate'])
                results['adaptive'].append(opt.backtest(p_a, test)['draws_with_win_rate'])
            for k, arr in results.items():
                arr = np.array(arr)
                print(f"   {k:<10} {arr.mean():.4f} ± {arr.std():.4f}")
            ca, ad = np.array(results['classic']), np.array(results['adaptive'])
            t, p = ttest_rel(ad, ca, alternative='greater')
            print(f"   Adaptive>Classic: diff={np.mean(ad-ca):+.4f} "
                  f"t={t:+.2f} p={p:.4f} {'🔍' if p < 0.05 else '  '}")

        elif op == '15':
            print("\n⚖️ EXPERIMENTO 4 BRAÇOS (v49.7)")
            try:
                ng = int(input("   n_games [5]: ").strip() or "5")
                nc = int(input("   n_candidates [10000]: ").strip() or "10000")
                nr = int(input("   n_replicates [10]: ").strip() or "10")
                tsz = int(input("   test_size [200]: ").strip() or "200")
            except:
                ng, nc, nr, tsz = 5, 10000, 10, 200
            fs = input("   Fixas (todos): ").strip()
            fixed = [int(x) for x in fs.split()] if fs else []
            es = input("   Excluídas (todos): ").strip()
            excluded = [int(x) for x in es.split()] if es else []
            ls = (input("   Local search em C/D? (s/n) [s]: ").strip().lower() != 'n')
            four_arm_experiment(contests, n_games=ng, n_candidates=nc,
                                n_replicates=nr, test_size=tsz,
                                fixed=fixed, excluded=excluded,
                                use_local_search=ls)

        elif op == '16':
            print("\n📊 RANKING DE ATRASO DE COMBINAÇÕES (v49.8)")
            print("   ⚠️  Atraso NÃO é previsão — P(aparecer) = P_k sempre.")
            print("   [1] Trincas (k=3)   [2] Quadras (k=4)")
            print("   [3] Quinas  (k=5)   [4] Senas   (k=6)")
            print("   [5] Todas")
            choice = input("   Escolha [5]: ").strip() or "5"
            ks = {'1': (3,), '2': (4,), '3': (5,), '4': (6,),
                  '5': (3, 4, 5, 6)}.get(choice, (3, 4, 5, 6))

            print("\n   Método de ranking:")
            print("     [1] Z-score (atraso vs própria média histórica)")
            print("     [2] Atraso absoluto")
            print("     [3] Razão vs intervalo teórico")
            print("     [4] Razão vs atraso máximo histórico")
            mc = input("   Escolha [1]: ").strip() or "1"
            method = {'1': 'z_score', '2': 'delay',
                      '3': 'ratio', '4': 'max_ratio'}.get(mc, 'z_score')

            try:
                top_n = int(input("   Top N [20]: ").strip() or "20")
            except:
                top_n = 20
            try:
                min_count = int(input("   Mínimo de aparições [2]: ").strip() or "2")
            except:
                min_count = 2
            save = input("   Salvar CSV? (s/n) [n]: ").strip().lower() == 's'

            rank_combination_delays(contests, ks=ks, top_n=top_n,
                                    method=method, min_count=min_count,
                                    save_csv=save)

        elif op == '6':
            bs = input("\n   Blocos [50,100,200,500]: ").strip()
            blocks = [int(x) for x in bs.split(',')] if bs else [50, 100, 200, 500]
            PredictiveRanking(contests).rank_predictive_power(blocks)

        elif op == '8':
            m = 'recent' if (input("   [1] Recente [2] IPE: ").strip() or "1") == "1" else 'ipe'
            StructuralPredictor(contests).predict_ranges(method=m)

        elif op == '0':
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
