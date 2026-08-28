#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49.7
======================================================================

OPÇÕES:
1  - GERAR CARTEIRA PERSONALIZADA
13 - ANÁLISE AVANÇADA DE FREQUÊNCIA + ATRASO
14 - ANÁLISE NUMEROLÓGICA / TEMPORAL
0  - SAIR

v49.7
----------------------------------------------------------------------
Mantém a estrutura da v49.5 e acrescenta a OPÇÃO 14.

OPÇÃO 14:
- Data do sorteio
- Dia
- Mês
- Ano
- Soma dos dígitos
- Raiz digital
- Dia + mês
- Dia + mês + ano
- Soma das dezenas do concurso anterior
- Raiz digital da soma anterior
- Divisão soma anterior / valor temporal
- Operações módulo 25
- Floor / round / ceil
- Defasagens
- Teste contra baseline aleatória
- Binomial
- IC95%
- Cohen's h
- Monte Carlo
- Treino / validação / OOS
- FDR
- Ranking das regras
- Estabilidade temporal

ATENÇÃO:
A análise NÃO assume causalidade.
Ela procura apenas evidência estatística de associação.
"""

import numpy as np
from scipy.stats import (
    hypergeom,
    binomtest,
    wilcoxon,
    mannwhitneyu,
    norm
)

from collections import Counter
from itertools import combinations
import os
import random
import time
import warnings
import math

from tqdm import tqdm

warnings.filterwarnings("ignore")


# ======================================================================
# CONSTANTES
# ======================================================================

VERSION = "v49.7"

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

MOLDURA = {
    1, 2, 3, 4, 5,
    6, 10, 11, 15, 16,
    20, 21, 22, 23, 24, 25
}

HYPE_PROBS = {
    k: hypergeom.pmf(k, 25, 15, 15)
    for k in range(0, 16)
}

PREMIO_VALORES = {
    11: 6.0,
    12: 12.0,
    13: 30.0,
    14: 1500.0,
    15: 1800000.0
}

CUSTO_APOSTA = 3.5

BASELINE_PROB_DEZENA = 15 / 25


# ======================================================================
# BITMASK
# ======================================================================

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


# ======================================================================
# CARREGAMENTO
# ======================================================================

def load_all_contests(csv_file='resultados_lotofacil.csv'):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)

    if not os.path.exists(csv_path):
        return None

    contests = []

    with open(csv_path, 'r', encoding='utf-8') as f:

        for line in f.readlines()[1:]:

            parts = line.strip().split(';')

            if len(parts) < 17:
                continue

            try:

                dezenas = [
                    int(x.strip())
                    for x in parts[2:17]
                    if x.strip()
                ]

                if len(dezenas) != 15:
                    continue

                if len(set(dezenas)) != 15:
                    continue

                if any(x < 1 or x > 25 for x in dezenas):
                    continue

                concursos_num = int(parts[0])

                contests.append({
                    'concurso': concursos_num,
                    'data': parts[1],
                    'dezenas': sorted(dezenas)
                })

            except Exception:
                continue

    contests.sort(key=lambda x: x['concurso'])

    print(f"✅ {len(contests)} concursos válidos")

    return contests


# ======================================================================
# GERADOR
# ======================================================================

class LooseGenerator:

    def __init__(self):
        pass

    def generate_one(
        self,
        fixed=None,
        semifixed=None,
        min_semifixed=0,
        max_semifixed=None,
        excluded=None,
        allowed_pares=None,
        allowed_moldura=None,
        allowed_primos=None,
        range_pares=None,
        range_moldura=None,
        range_primos=None,
        range_soma=None,
        range_amplitude=None,
        range_consecutivos=None
    ):

        for _ in range(500):

            game = self._generate_raw(
                fixed,
                semifixed,
                min_semifixed,
                max_semifixed,
                excluded,
                allowed_pares,
                allowed_moldura,
                allowed_primos,
                range_pares,
                range_moldura,
                range_primos,
                range_soma,
                range_amplitude,
                range_consecutivos
            )

            if game is not None:
                return game

        raise RuntimeError(
            "Não foi possível gerar jogo com os parâmetros fornecidos."
        )

    def _generate_raw(
        self,
        fixed,
        semifixed,
        min_semifixed,
        max_semifixed,
        excluded,
        allowed_pares,
        allowed_moldura,
        allowed_primos,
        range_pares,
        range_moldura,
        range_primos,
        range_soma,
        range_amplitude,
        range_consecutivos
    ):

        if fixed is None:
            fixed = []

        if semifixed is None:
            semifixed = []

        if excluded is None:
            excluded = []

        excluded_set = set(excluded)

        fixed_set = set(fixed) - excluded_set

        semifixed_set = (
            set(semifixed)
            - fixed_set
            - excluded_set
        )

        proibidas = (
            fixed_set |
            semifixed_set |
            excluded_set
        )

        todas = set(range(1, 26))

        restantes = list(todas - proibidas)

        n_fixas = len(fixed_set)

        if max_semifixed is None:
            max_semi = len(semifixed_set)
        else:
            max_semi = min(
                max_semifixed,
                len(semifixed_set)
            )

        min_semi = max(min_semifixed, 0)

        if min_semi > max_semi:
            return None

        n_semifixed_escolher = random.randint(
            min_semi,
            max_semi
        )

        n_restantes = (
            15
            - n_fixas
            - n_semifixed_escolher
        )

        if n_restantes < 0:
            return None

        if n_restantes > len(restantes):
            return None

        for _ in range(200):

            if (
                n_semifixed_escolher > 0
                and len(semifixed_set) > 0
            ):

                chosen_semi = set(
                    random.sample(
                        list(semifixed_set),
                        min(
                            n_semifixed_escolher,
                            len(semifixed_set)
                        )
                    )
                )

            else:
                chosen_semi = set()

            if n_restantes > 0:

                chosen_rest = set(
                    random.sample(
                        restantes,
                        min(
                            n_restantes,
                            len(restantes)
                        )
                    )
                )

            else:
                chosen_rest = set()

            game = sorted(
                fixed_set |
                chosen_semi |
                chosen_rest
            )

            if len(game) != 15:
                continue

            pares = sum(
                1 for x in game
                if x % 2 == 0
            )

            mol = sum(
                1 for x in game
                if x in MOLDURA
            )

            prim = sum(
                1 for x in game
                if x in PRIMES
            )

            soma = sum(game)

            amplitude = (
                max(game) -
                min(game)
            )

            consec = sum(
                1
                for i in range(len(game) - 1)
                if game[i + 1] - game[i] == 1
            )

            if allowed_pares is not None:

                if pares not in allowed_pares:
                    continue

            if allowed_moldura is not None:

                if mol not in allowed_moldura:
                    continue

            if allowed_primos is not None:

                if prim not in allowed_primos:
                    continue

            if range_pares is not None:

                if not (
                    range_pares[0]
                    <= pares
                    <= range_pares[1]
                ):
                    continue

            if range_moldura is not None:

                if not (
                    range_moldura[0]
                    <= mol
                    <= range_moldura[1]
                ):
                    continue

            if range_primos is not None:

                if not (
                    range_primos[0]
                    <= prim
                    <= range_primos[1]
                ):
                    continue

            if range_soma is not None:

                if not (
                    range_soma[0]
                    <= soma
                    <= range_soma[1]
                ):
                    continue

            if range_amplitude is not None:

                if not (
                    range_amplitude[0]
                    <= amplitude
                    <= range_amplitude[1]
                ):
                    continue

            if range_consecutivos is not None:

                if not (
                    range_consecutivos[0]
                    <= consec
                    <= range_consecutivos[1]
                ):
                    continue

            return game

        return None

    def generate_pure_random(self):

        return sorted(
            np.random.choice(
                range(1, 26),
                15,
                replace=False
            )
        )


# ======================================================================
# OTIMIZADOR
# ======================================================================

class PortfolioOptimizer:

    def __init__(
        self,
        contests,
        fixed=None,
        semifixed=None,
        min_semifixed=0,
        max_semifixed=None,
        excluded=None,
        allowed_pares=None,
        allowed_moldura=None,
        allowed_primos=None,
        range_pares=None,
        range_moldura=None,
        range_primos=None,
        range_soma=None,
        range_amplitude=None,
        range_consecutivos=None
    ):

        self.contests = contests

        self.generator = LooseGenerator()

        self.excluded = (
            excluded
            if excluded
            else []
        )

        excl_set = set(self.excluded)

        self.fixed = (
            fixed
            if fixed
            else []
        )

        if excl_set & set(self.fixed):

            removidas_fixas = (
                excl_set &
                set(self.fixed)
            )

            print(
                "⚠️ Dezenas fixas também excluídas; "
                f"removendo das fixas: "
                f"{sorted(removidas_fixas)}"
            )

            self.fixed = [
                d
                for d in self.fixed
                if d not in excl_set
            ]

        self.semifixed = (
            semifixed
            if semifixed
            else []
        )

        if excl_set & set(self.semifixed):

            removidas_semi = (
                excl_set &
                set(self.semifixed)
            )

            print(
                "⚠️ Dezenas semifixas também excluídas; "
                f"removendo das semifixas: "
                f"{sorted(removidas_semi)}"
            )

            self.semifixed = [
                d
                for d in self.semifixed
                if d not in excl_set
            ]

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

        for _ in tqdm(
            range(n_candidates),
            desc="Gerando pool"
        ):

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

                key = tuple(g)

                if key not in seen:

                    seen.add(key)
                    pool.append(g)

            except RuntimeError:
                break

        return pool

    def select_covering(
        self,
        candidates,
        n_select,
        level='pair'
    ):

        if len(candidates) < n_select:

            raise ValueError(
                f"Pool insuficiente: "
                f"{len(candidates)} < {n_select}"
            )

        r = 2 if level == 'pair' else 3

        covered = set()
        selected = []

        for _ in range(n_select):

            best_idx = -1
            best_new = -1

            for i, c in enumerate(candidates):

                if c in selected:
                    continue

                groups = set(
                    combinations(
                        sorted(c),
                        r
                    )
                )

                new_groups = len(
                    groups - covered
                )

                if new_groups > best_new:

                    best_new = new_groups
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(
                candidates[best_idx]
            )

            covered.update(
                combinations(
                    sorted(
                        candidates[best_idx]
                    ),
                    r
                )
            )

        return selected

    def optimize(
        self,
        n_games=5,
        n_candidates=100000,
        method='pair_covering'
    ):

        print(
            f"\n🧩 CARTEIRA: "
            f"{n_games} jogos | método: {method}"
        )

        if self.fixed:
            print(
                f"   Fixas: {self.fixed}"
            )

        if self.semifixed:

            print(
                f"   Semifixas: "
                f"{self.semifixed} "
                f"(mín={self.min_semifixed}, "
                f"máx={self.max_semifixed})"
            )

        if self.excluded:
            print(
                f"   Excluídas: {self.excluded}"
            )

        if self.range_pares:
            print(
                f"   Pares: {self.range_pares}"
            )

        if self.range_moldura:
            print(
                f"   Moldura: {self.range_moldura}"
            )

        if self.range_primos:
            print(
                f"   Primos: {self.range_primos}"
            )

        t0 = time.time()

        pool = self.generate_pool(
            n_candidates
        )

        print(
            f"   Pool: {len(pool)} jogos"
        )

        if len(pool) < n_games:

            raise RuntimeError(
                f"Pool insuficiente: "
                f"{len(pool)} < {n_games}."
            )

        if method == 'pair_covering':

            portfolio = self.select_covering(
                pool,
                n_games,
                level='pair'
            )

        elif method == 'triple_covering':

            portfolio = self.select_covering(
                pool,
                n_games,
                level='triple'
            )

        else:

            portfolio = pool[:n_games]

        print(
            f"✅ {len(portfolio)} jogos "
            f"em {time.time() - t0:.1f}s"
        )

        return portfolio

    def backtest(
        self,
        portfolio,
        test_draws
    ):

        n_draw_success = 0

        total_premio = 0

        n_apostas = len(portfolio)

        n_test = len(test_draws)

        total_custo = (
            n_apostas *
            n_test *
            CUSTO_APOSTA
        )

        portfolio_masks = np.array(
            [
                BITMASK_CACHE.get_mask(g)
                for g in portfolio
            ],
            dtype=np.uint32
        )

        hit_counts = {
            k: 0
            for k in range(11, 16)
        }

        for draw in test_draws:

            dm = BITMASK_CACHE.get_mask(
                draw['dezenas']
            )

            draw_success = False

            for pm in portfolio_masks:

                hits = mask_intersection(
                    pm,
                    dm
                )

                if hits >= 11:

                    draw_success = True

                    total_premio += (
                        PREMIO_VALORES.get(
                            hits,
                            0
                        )
                    )

                    hit_counts[hits] += 1

            if draw_success:
                n_draw_success += 1

        empirical = (
            n_draw_success / n_test
            if n_test > 0
            else 0
        )

        p_single = sum(
            HYPE_PROBS[k]
            for k in range(11, 16)
        )

        theo_prob = (
            1 -
            (1 - p_single) ** n_apostas
        )

        return {

            'empirical': empirical,

            'theoretical': theo_prob,

            'lift': (
                empirical / theo_prob
                if theo_prob > 0
                else 1.0
            ),

            'n_test': n_test,

            'n_success': n_draw_success,

            'total_premio': total_premio,

            'total_custo': total_custo,

            'roi': (
                (total_premio - total_custo)
                / total_custo * 100
                if total_custo > 0
                else 0
            ),

            'hit_distribution': hit_counts
        }


# ======================================================================
# FREQUÊNCIA + ATRASO
# ======================================================================

def freq_janela(
    contests,
    inicio,
    fim,
    dezenas=range(1, 26)
):

    freq = Counter()

    inicio = max(0, inicio)

    for c in contests[inicio:fim]:
        freq.update(
            c['dezenas']
        )

    return {
        d: freq.get(d, 0)
        for d in dezenas
    }


def calcular_atrasos(
    contests,
    indice=None
):

    if indice is None:
        indice = len(contests)

    atrasos = {}

    for d in range(1, 26):

        atraso = 0

        for j in range(
            indice - 1,
            -1,
            -1
        ):

            if d in contests[j]['dezenas']:
                break

            atraso += 1

        atrasos[d] = atraso

    return atrasos


def score_dezena(
    freq_recente,
    freq_historica,
    atraso_z,
    janela_recente=10,
    janela_historica=100,
    pesos=(0.5, 0.2, 0.3),
    metodo='linear'
):

    p = 15 / 25

    media_recente = (
        p * janela_recente
    )

    desvio_recente = np.sqrt(
        janela_recente *
        p *
        (1 - p)
    )

    z_recente = (
        freq_recente -
        media_recente
    ) / desvio_recente

    media_hist = (
        p * janela_historica
    )

    desvio_hist = np.sqrt(
        janela_historica *
        p *
        (1 - p)
    )

    z_hist = (
        freq_historica -
        media_hist
    ) / desvio_hist

    if metodo == 'sqrt':

        bonus_atraso = (
            np.sqrt(
                max(atraso_z, 0)
            )
            *
            np.sign(atraso_z)
        )

    elif metodo == 'log':

        bonus_atraso = (
            np.log1p(
                max(atraso_z, 0)
            )
            *
            np.sign(atraso_z)
        )

    else:

        bonus_atraso = atraso_z

    return (
        pesos[0] * z_recente
        +
        pesos[1] * z_hist
        +
        pesos[2] * bonus_atraso
    )


# ======================================================================
# OPÇÃO 13
# ======================================================================

def analise_frequentes_atraso_v3(
    contests,
    top_n_list=[5, 10, 15, 20],
    janelas_recentes=[
        3, 5, 7, 10, 15,
        20, 30, 50, 100
    ],
    janela_historica=100,
    min_history=500,
    pesos_grid=None,
    n_sim_mc=1000,
    alpha=0.05
):

    print(
        f"\n🔬 ANÁLISE AVANÇADA "
        f"DE FREQUÊNCIA + ATRASO ({VERSION})"
    )

    print(
        f"   Top_n avaliados: {top_n_list} "
        f"| Janelas recentes: {janelas_recentes} "
        f"| Janela histórica: {janela_historica}"
    )

    print(
        f"   Histórico mínimo: {min_history} "
        f"| Simulações MC: {n_sim_mc}"
    )

    n = len(contests)

    train_end = int(n * 0.6)

    val_end = int(n * 0.8)

    treino = contests[:train_end]

    validacao = contests[
        train_end:val_end
    ]

    teste_oos = contests[val_end:]

    if pesos_grid is None:

        pesos_grid = [

            (0.5, 0.2, 0.3),

            (0.6, 0.2, 0.2),

            (0.4, 0.3, 0.3),

            (0.7, 0.1, 0.2),

            (0.3, 0.3, 0.4)
        ]

    metodos = [
        'linear',
        'sqrt',
        'log'
    ]

    resultados_topn = {}

    for top_n in top_n_list:

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"   🎯 TOP {top_n} DEZENAS"
        )

        print(
            f"{'=' * 60}"
        )

        melhor_score_treino = -1

        melhor_config = None

        pvals_configs = []

        for janela in janelas_recentes:

            for pesos in pesos_grid:

                for metodo in metodos:

                    acertos = []

                    for i in range(
                        min_history,
                        len(treino)
                    ):

                        passado = treino[:i]

                        alvo = set(
                            treino[i]['dezenas']
                        )

                        inicio_jan = max(
                            0,
                            len(passado) -
                            janela
                        )

                        freq = freq_janela(
                            passado,
                            inicio_jan,
                            len(passado)
                        )

                        hist = freq_janela(
                            passado,
                            0,
                            len(passado)
                        )

                        atr = calcular_atrasos(
                            passado,
                            indice=len(passado)
                        )

                        atr_vals = np.array(
                            list(atr.values())
                        )

                        atr_mean = np.mean(
                            atr_vals
                        )

                        atr_std = np.std(
                            atr_vals
                        )

                        atr_z = {

                            d:
                            (
                                atr[d] -
                                atr_mean
                            )
                            /
                            (
                                atr_std +
                                1e-9
                            )

                            for d in range(
                                1, 26
                            )
                        }

                        scores = {

                            d:
                            score_dezena(
                                freq[d],
                                hist[d],
                                atr_z[d],
                                janela,
                                janela_historica,
                                pesos,
                                metodo
                            )

                            for d in range(
                                1, 26
                            )
                        }

                        top = set(
                            sorted(
                                range(1, 26),
                                key=lambda d:
                                scores[d],
                                reverse=True
                            )[:top_n]
                        )

                        acertos.append(
                            len(top & alvo)
                        )

                    media = (
                        np.mean(acertos)
                        if acertos
                        else 0
                    )

                    try:

                        p = binomtest(
                            int(
                                media *
                                len(acertos)
                            ),
                            len(acertos),
                            6 / top_n
                        ).pvalue

                    except Exception:

                        p = 1.0

                    pvals_configs.append(p)

                    if media > melhor_score_treino:

                        melhor_score_treino = media

                        melhor_config = (
                            janela,
                            pesos,
                            metodo
                        )

        # --------------------------------------------------------------
        # FDR
        # --------------------------------------------------------------

        pvals = np.array(
            pvals_configs
        )

        sorted_idx = np.argsort(
            pvals
        )

        m = len(pvals)

        qvals = np.ones(m)

        for i in range(
            m - 1,
            -1,
            -1
        ):

            rank = i + 1

            q = (
                pvals[
                    sorted_idx[i]
                ]
                *
                m
                /
                rank
            )

            if i < m - 1:

                q = min(
                    q,
                    qvals[
                        sorted_idx[i + 1]
                    ]
                )

            qvals[
                sorted_idx[i]
            ] = min(q, 1.0)

        best_config_idx = (
            list(pvals).index(
                min(pvals)
            )
        )

        if (
            qvals[best_config_idx]
            < 0.2
        ):

            print(
                "   🔍 Melhor configuração "
                "significativa após FDR "
                f"(q={qvals[best_config_idx]:.4f})."
            )

        else:

            print(
                "   ⚠️ Nenhuma configuração "
                "significativa após FDR; "
                "usando melhor média absoluta."
            )

        janela_opt, pesos_opt, metodo_opt = (
            melhor_config
        )

        # --------------------------------------------------------------
        # VALIDAÇÃO
        # --------------------------------------------------------------

        acertos_val = []

        for i in range(
            len(validacao)
        ):

            passado = (
                treino +
                validacao[:i]
            )

            alvo = set(
                validacao[i]['dezenas']
            )

            inicio_jan = max(
                0,
                len(passado) -
                janela_opt
            )

            freq = freq_janela(
                passado,
                inicio_jan,
                len(passado)
            )

            hist = freq_janela(
                passado,
                0,
                len(passado)
            )

            atr = calcular_atrasos(
                passado,
                indice=len(passado)
            )

            atr_vals = np.array(
                list(atr.values())
            )

            atr_mean = np.mean(
                atr_vals
            )

            atr_std = np.std(
                atr_vals
            )

            atr_z = {

                d:
                (
                    atr[d] -
                    atr_mean
                )
                /
                (
                    atr_std +
                    1e-9
                )

                for d in range(
                    1, 26
                )
            }

            scores = {

                d:
                score_dezena(
                    freq[d],
                    hist[d],
                    atr_z[d],
                    janela_opt,
                    janela_historica,
                    pesos_opt,
                    metodo_opt
                )

                for d in range(
                    1, 26
                )
            }

            top = set(
                sorted(
                    range(1, 26),
                    key=lambda d:
                    scores[d],
                    reverse=True
                )[:top_n]
            )

            acertos_val.append(
                len(top & alvo)
            )

        media_val = (
            np.mean(acertos_val)
            if acertos_val
            else 0
        )

        print(
            f"   Melhor configuração: "
            f"janela={janela_opt}, "
            f"pesos={pesos_opt}, "
            f"método={metodo_opt}"
        )

        print(
            f"   Média validação: "
            f"{media_val:.3f}"
        )

        if (
            media_val <
            melhor_score_treino - 0.5
        ):

            print(
                "   ⚠️ Degradação na validação; "
                "usando configuração padrão."
            )

            janela_opt = 10

            pesos_opt = (
                0.5,
                0.2,
                0.3
            )

            metodo_opt = 'linear'

        # --------------------------------------------------------------
        # OOS
        # --------------------------------------------------------------

        estrategias = [

            'aleatorio',

            'frequencia',

            'atraso',

            'freq+atraso_linear',

            'freq+atraso_sqrt',

            'freq+atraso_log',

            'modelo_otimizado'
        ]

        resultados_oos = {

            est: []
            for est in estrategias
        }

        for i in range(
            len(teste_oos)
        ):

            passado = (
                treino +
                validacao +
                teste_oos[:i]
            )

            alvo = set(
                teste_oos[i]['dezenas']
            )

            inicio_jan10 = max(
                0,
                len(passado) - 10
            )

            freq10 = freq_janela(
                passado,
                inicio_jan10,
                len(passado)
            )

            hist = freq_janela(
                passado,
                0,
                len(passado)
            )

            atr = calcular_atrasos(
                passado,
                indice=len(passado)
            )

            atr_vals = np.array(
                list(atr.values())
            )

            atr_mean = np.mean(
                atr_vals
            )

            atr_std = np.std(
                atr_vals
            )

            atr_z = {

                d:
                (
                    atr[d] -
                    atr_mean
                )
                /
                (
                    atr_std +
                    1e-9
                )

                for d in range(
                    1, 26
                )
            }

            selecionadas = set(
                np.random.choice(
                    range(1, 26),
                    top_n,
                    replace=False
                )
            )

            resultados_oos[
                'aleatorio'
            ].append(
                len(
                    selecionadas & alvo
                )
            )

            top_freq = set(
                sorted(
                    range(1, 26),
                    key=lambda d:
                    (
                        freq10[d],
                        hist[d]
                    ),
                    reverse=True
                )[:top_n]
            )

            resultados_oos[
                'frequencia'
            ].append(
                len(
                    top_freq & alvo
                )
            )

            top_atr = set(
                sorted(
                    range(1, 26),
                    key=lambda d:
                    atr[d],
                    reverse=True
                )[:top_n]
            )

            resultados_oos[
                'atraso'
            ].append(
                len(
                    top_atr & alvo
                )
            )

            for metodo, nome in [

                (
                    'linear',
                    'freq+atraso_linear'
                ),

                (
                    'sqrt',
                    'freq+atraso_sqrt'
                ),

                (
                    'log',
                    'freq+atraso_log'
                )
            ]:

                scores = {

                    d:
                    score_dezena(
                        freq10[d],
                        hist[d],
                        atr_z[d],
                        10,
                        janela_historica,
                        (
                            0.5,
                            0.2,
                            0.3
                        ),
                        metodo
                    )

                    for d in range(
                        1, 26
                    )
                }

                top = set(
                    sorted(
                        range(1, 26),
                        key=lambda d:
                        scores[d],
                        reverse=True
                    )[:top_n]
                )

                resultados_oos[
                    nome
                ].append(
                    len(
                        top & alvo
                    )
                )

            inicio_jan_opt = max(
                0,
                len(passado) -
                janela_opt
            )

            freq_jan = freq_janela(
                passado,
                inicio_jan_opt,
                len(passado)
            )

            scores = {

                d:
                score_dezena(
                    freq_jan[d],
                    hist[d],
                    atr_z[d],
                    janela_opt,
                    janela_historica,
                    pesos_opt,
                    metodo_opt
                )

                for d in range(
                    1, 26
                )
            }

            top = set(
                sorted(
                    range(1, 26),
                    key=lambda d:
                    scores[d],
                    reverse=True
                )[:top_n]
            )

            resultados_oos[
                'modelo_otimizado'
            ].append(
                len(
                    top & alvo
                )
            )

        print(
            f"\n📊 Comparação OOS "
            f"(top {top_n}):"
        )

        print(
            f"{'Estratégia':<20} "
            f"{'Média':<8} "
            f"{'≥7':<8} "
            f"{'≥8':<8} "
            f"{'≥9':<8} "
            f"{'≥10':<8}"
        )

        print("-" * 60)

        arrs = {}

        for est in estrategias:

            arr = np.array(
                resultados_oos[est]
            )

            arrs[est] = arr

            media = np.mean(arr)

            p7 = np.mean(
                arr >= 7
            ) * 100

            p8 = np.mean(
                arr >= 8
            ) * 100

            p9 = np.mean(
                arr >= 9
            ) * 100

            p10 = np.mean(
                arr >= 10
            ) * 100

            print(
                f"{est:<20} "
                f"{media:<8.3f} "
                f"{p7:<8.1f} "
                f"{p8:<8.1f} "
                f"{p9:<8.1f} "
                f"{p10:<8.1f}"
            )

        dif = (
            arrs['modelo_otimizado']
            -
            arrs['aleatorio']
        )

        media_dif = np.mean(dif)

        rng = np.random.default_rng(42)

        boot_means = [

            np.mean(
                rng.choice(
                    dif,
                    size=len(dif),
                    replace=True
                )
            )

            for _ in range(1000)
        ]

        ci_low, ci_high = np.percentile(
            boot_means,
            [2.5, 97.5]
        )

        pooled_std = np.sqrt(
            (
                np.std(
                    arrs[
                        'modelo_otimizado'
                    ]
                ) ** 2
                +
                np.std(
                    arrs['aleatorio']
                ) ** 2
            ) / 2
        )

        cohens_d = (
            media_dif /
            pooled_std
            if pooled_std > 0
            else 0
        )

        perm_diffs = []

        combined = np.concatenate(
            [
                arrs[
                    'modelo_otimizado'
                ],
                arrs['aleatorio']
            ]
        )

        for _ in range(1000):

            perm = rng.permutation(
                combined
            )

            perm_diffs.append(

                np.mean(
                    perm[
                        :len(
                            arrs[
                                'modelo_otimizado'
                            ]
                        )
                    ]
                )
                -
                np.mean(
                    perm[
                        len(
                            arrs[
                                'modelo_otimizado'
                            ]
                        ):
                    ]
                )
            )

        p_perm = np.mean(
            np.abs(perm_diffs)
            >= abs(media_dif)
        )

        try:

            w, p_wilc = wilcoxon(
                arrs[
                    'modelo_otimizado'
                ],
                arrs['aleatorio']
            )

        except Exception:

            w = np.nan
            p_wilc = 1.0

        print(
            "\n🔍 Comparação "
            "modelo vs aleatório:"
        )

        print(
            f"   Diferença média: "
            f"{media_dif:.3f} "
            f"(IC95%: "
            f"[{ci_low:.3f}, "
            f"{ci_high:.3f}])"
        )

        print(
            f"   Cohen's d: "
            f"{cohens_d:.3f}"
        )

        print(
            f"   p-valor "
            f"(permutação): "
            f"{p_perm:.4f}"
        )

        print(
            f"   Wilcoxon: "
            f"W={w}, "
            f"p={p_wilc:.4f}"
        )

        resultados_topn[top_n] = {

            'arrs': arrs,

            'media_dif': media_dif,

            'cohens_d': cohens_d,

            'p_perm': p_perm,

            'p_wilc': p_wilc,

            'melhor_config':
                melhor_config
        }

    # ------------------------------------------------------------------
    # FDR SOBRE TOP_N
    # ------------------------------------------------------------------

    print(
        "\n🔍 Correção FDR sobre "
        "os top_n:"
    )

    topn_pvals = []

    for tn in top_n_list:

        topn_pvals.append(
            resultados_topn[
                tn
            ]['p_wilc']
        )

    order = np.argsort(
        topn_pvals
    )

    qvals_topn = np.ones(
        len(topn_pvals)
    )

    for i in range(
        len(order) - 1,
        -1,
        -1
    ):

        idx = order[i]

        rank = i + 1

        q = (
            topn_pvals[idx]
            *
            len(topn_pvals)
            /
            rank
        )

        if i < len(order) - 1:

            q = min(
                q,
                qvals_topn[
                    order[i + 1]
                ]
            )

        qvals_topn[idx] = min(
            q,
            1.0
        )

        print(
            f"   Top {top_n_list[idx]}: "
            f"p-valor Wilcoxon = "
            f"{topn_pvals[idx]:.4f} "
            f"| q = "
            f"{qvals_topn[idx]:.4f}"
        )

    significativos = [

        top_n_list[i]

        for i, q in enumerate(
            qvals_topn
        )

        if q < alpha
    ]

    if significativos:

        melhor_topn = min(
            significativos,
            key=lambda tn:
            resultados_topn[
                tn
            ]['p_wilc']
        )

        print(
            f"   🔍 Top_n significativo: "
            f"{melhor_topn}"
        )

    else:

        melhor_topn = min(
            top_n_list,
            key=lambda tn:
            resultados_topn[
                tn
            ]['p_wilc']
        )

        print(
            f"   Nenhum top_n significativo; "
            f"usando top {melhor_topn}."
        )

    # ------------------------------------------------------------------
    # MONTE CARLO
    # ------------------------------------------------------------------

    print(
        f"\n🎲 Monte Carlo "
        f"({n_sim_mc} simulações) "
        f"para top {melhor_topn}"
    )

    res_mc = resultados_topn[
        melhor_topn
    ]

    obs_media = np.mean(
        res_mc[
            'arrs'
            ]['modelo_otimizado']
    )

    medias_sim = np.empty(
        n_sim_mc,
        dtype=np.float64
    )

    n = len(contests)

    train_end = int(n * 0.6)

    val_end = int(n * 0.8)

    janela_mc, pesos_mc, metodo_mc = (
        res_mc['melhor_config']
    )

    rng = np.random.default_rng(
        20260828
    )

    for sim in range(
        n_sim_mc
    ):

        # --------------------------------------------------------------
        # Geração independente
        # --------------------------------------------------------------

        matriz_sim = np.empty(
            (n, 15),
            dtype=np.int8
        )

        for i in range(n):

            matriz_sim[i] = rng.choice(
                np.arange(1, 26),
                15,
                replace=False
            )

        # --------------------------------------------------------------
        # Modelo simplificado e
        # procedimento walk-forward
        # --------------------------------------------------------------

        acertos = []

        for i in range(
            val_end,
            n
        ):

            passado = matriz_sim[:i]

            if len(passado) < min_history:
                continue

            janela = janela_mc

            recente = passado[
                max(
                    0,
                    len(passado) -
                    janela
                ):
            ]

            freq = np.bincount(
                recente.ravel(),
                minlength=26
            )

            hist = np.bincount(
                passado.ravel(),
                minlength=26
            )

            atrasos = np.zeros(
                26,
                dtype=float
            )

            for d in range(
                1,
                26
            ):

                atraso = 0

                for j in range(
                    i - 1,
                    -1,
                    -1
                ):

                    if d in passado[j]:
                        break

                    atraso += 1

                atrasos[d] = atraso

            atr_mean = np.mean(
                atrasos[1:]
            )

            atr_std = np.std(
                atrasos[1:]
            )

            scores = np.zeros(
                26,
                dtype=float
            )

            p = 15 / 25

            media_rec = (
                p * janela
            )

            desvio_rec = np.sqrt(
                janela *
                p *
                (1 - p)
            )

            media_hist = (
                p * 100
            )

            desvio_hist = np.sqrt(
                100 *
                p *
                (1 - p)
            )

            for d in range(
                1,
                26
            ):

                z_rec = (
                    freq[d] -
                    media_rec
                ) / desvio_rec

                z_hist = (
                    hist[d] -
                    media_hist
                ) / desvio_hist

                atr_z = (
                    atrasos[d] -
                    atr_mean
                ) / (
                    atr_std +
                    1e-9
                )

                if metodo_mc == 'sqrt':

                    bonus = (
                        np.sqrt(
                            max(
                                atr_z,
                                0
                            )
                        )
                        *
                        np.sign(
                            atr_z
                        )
                    )

                elif metodo_mc == 'log':

                    bonus = (
                        np.log1p(
                            max(
                                atr_z,
                                0
                            )
                        )
                        *
                        np.sign(
                            atr_z
                        )
                    )

                else:

                    bonus = atr_z

                scores[d] = (
                    pesos_mc[0] * z_rec
                    +
                    pesos_mc[1] * z_hist
                    +
                    pesos_mc[2] * bonus
                )

            top = np.argsort(
                scores[1:]
            )[-melhor_topn:] + 1

            alvo = matriz_sim[i]

            acertos.append(
                np.isin(
                    top,
                    alvo
                ).sum()
            )

        medias_sim[sim] = (
            np.mean(acertos)
            if acertos
            else 6.0
        )

        if (
            (sim + 1) %
            max(
                1,
                n_sim_mc // 10
            ) == 0
            or sim == 0
        ):

            print(
                f"   Simulação "
                f"{sim + 1}/{n_sim_mc} "
                f"| Média atual: "
                f"{np.mean(medias_sim[:sim+1]):.4f}"
            )

    p_mc = np.mean(
        medias_sim >= obs_media
    )

    q025, q975 = np.percentile(
        medias_sim,
        [2.5, 97.5]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "🎲 RESULTADO DO MONTE CARLO"
    )

    print(
        "=" * 60
    )

    print(
        f"Média observada "
        f"(modelo otimizado): "
        f"{obs_media:.3f}"
    )

    print(
        f"Média sob sorteios "
        f"independentes: "
        f"{np.mean(medias_sim):.3f} "
        f"± "
        f"{np.std(medias_sim):.3f}"
    )

    print(
        f"Mediana das simulações: "
        f"{np.median(medias_sim):.3f}"
    )

    print(
        f"Melhor simulação: "
        f"{np.max(medias_sim):.3f}"
    )

    print(
        f"p-valor empírico: "
        f"{p_mc:.4f}"
    )

    print(
        f"IC empírico 95%: "
        f"[{q025:.3f}, {q975:.3f}]"
    )

    # ------------------------------------------------------------------
    # CARTEIRA FINAL
    # ------------------------------------------------------------------

    print(
        f"\n🧩 Gerando carteira "
        f"otimizada com as top "
        f"{melhor_topn} dezenas..."
    )

    janela_final, pesos_final, metodo_final = (
        res_mc['melhor_config']
    )

    passado_total = contests

    inicio_jan = max(
        0,
        len(passado_total) -
        janela_final
    )

    freq = freq_janela(
        passado_total,
        inicio_jan,
        len(passado_total)
    )

    hist = freq_janela(
        passado_total,
        0,
        len(passado_total)
    )

    atr = calcular_atrasos(
        passado_total
    )

    atr_vals = np.array(
        list(atr.values())
    )

    atr_mean = np.mean(
        atr_vals
    )

    atr_std = np.std(
        atr_vals
    )

    atr_z = {

        d:
        (
            atr[d] -
            atr_mean
        )
        /
        (
            atr_std +
            1e-9
        )

        for d in range(
            1,
            26
        )
    }

    scores = {

        d:
        score_dezena(
            freq[d],
            hist[d],
            atr_z[d],
            janela_final,
            100,
            pesos_final,
            metodo_final
        )

        for d in range(
            1,
            26
        )
    }

    top_final = sorted(
        range(1, 26),
        key=lambda d:
        scores[d],
        reverse=True
    )[:melhor_topn]

    print(
        f"   Top {melhor_topn} dezenas: "
        f"{top_final}"
    )

    try:

        opt = PortfolioOptimizer(
            contests,
            fixed=top_final
        )

        portfolio = opt.optimize(
            5,
            50000,
            method='pair_covering'
        )

        for i, g in enumerate(
            portfolio,
            1
        ):

            p = sum(
                1
                for x in g
                if x % 2 == 0
            )

            pr = sum(
                1
                for x in g
                if x in PRIMES
            )

            m = sum(
                1
                for x in g
                if x in MOLDURA
            )

            print(
                f" {i}. {g} "
                f"| P:{p} "
                f"Pr:{pr} "
                f"M:{m}"
            )

        if len(contests) > 200:

            bt = opt.backtest(
                portfolio,
                contests[-200:]
            )

            print(
                f"\n🔬 BACKTEST (200): "
                f"Lift={bt['lift']:.2f}x "
                f"| ROI={bt['roi']:+.1f}%"
            )

    except Exception as e:

        print(
            f"   Erro ao gerar carteira: "
            f"{e}"
        )

    return (
        resultados_topn,
        p_mc
    )


# ======================================================================
# OPÇÃO 14
# ======================================================================

def raiz_digital(n):

    n = abs(int(n))

    if n == 0:
        return 0

    return 1 + ((n - 1) % 9)


def soma_digitos(n):

    return sum(
        int(c)
        for c in str(abs(int(n)))
        if c.isdigit()
    )


def parse_data(data):

    """
    Aceita principalmente:
        DD/MM/YYYY
        DD-MM-YYYY
        YYYY-MM-DD

    Retorna:
        dia, mês, ano
    """

    if data is None:
        return None

    s = str(data).strip()

    for sep in [
        '/',
        '-',
        '.'
    ]:

        partes = s.split(sep)

        if len(partes) != 3:
            continue

        try:

            a, b, c = [
                int(x)
                for x in partes
            ]

            # YYYY-MM-DD

            if a > 1900:

                ano = a
                mes = b
                dia = c

            else:

                dia = a
                mes = b
                ano = c

            if not (
                1 <= dia <= 31
                and 1 <= mes <= 12
            ):
                continue

            return (
                dia,
                mes,
                ano
            )

        except Exception:
            continue

    return None


def reduzir_para_1_25(x):

    """
    Reduz qualquer inteiro para 1..25
    usando módulo 25.

    Exemplo:
    25 -> 25
    26 -> 1
    50 -> 25
    51 -> 1
    """

    x = int(x)

    return ((x - 1) % 25) + 1


def valor_divisao(
    numerador,
    divisor,
    modo
):

    if divisor == 0:
        return None

    x = numerador / divisor

    if modo == 'floor':
        x = math.floor(x)

    elif modo == 'ceil':
        x = math.ceil(x)

    elif modo == 'round':
        x = round(x)

    else:
        return None

    return reduzir_para_1_25(x)


# ======================================================================
# CONSTRUÇÃO DAS REGRAS TEMPORAIS
# ======================================================================

def construir_regras_temporais(
    contest,
    anterior=None
):

    data = parse_data(
        contest.get('data')
    )

    if data is None:
        return {}

    dia, mes, ano = data

    soma_data = (
        soma_digitos(dia)
        +
        soma_digitos(mes)
        +
        soma_digitos(ano)
    )

    soma_dia = soma_digitos(
        dia
    )

    soma_mes = soma_digitos(
        mes
    )

    soma_ano = soma_digitos(
        ano
    )

    rd_dia = raiz_digital(
        dia
    )

    rd_data = raiz_digital(
        soma_data
    )

    dia_mes = (
        dia + mes
    )

    rd_dia_mes = raiz_digital(
        dia_mes
    )

    regras = {}

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------

    regras[
        'dia'
    ] = reduzir_para_1_25(
        dia
    )

    regras[
        'raiz_dia'
    ] = reduzir_para_1_25(
        rd_dia
    )

    regras[
        'mes'
    ] = reduzir_para_1_25(
        mes
    )

    regras[
        'ano_mod25'
    ] = reduzir_para_1_25(
        ano
    )

    regras[
        'soma_digitos_data'
    ] = reduzir_para_1_25(
        soma_data
    )

    regras[
        'raiz_digital_data'
    ] = reduzir_para_1_25(
        rd_data
    )

    regras[
        'soma_dia'
    ] = reduzir_para_1_25(
        soma_dia
    )

    regras[
        'soma_mes'
    ] = reduzir_para_1_25(
        soma_mes
    )

    regras[
        'soma_ano'
    ] = reduzir_para_1_25(
        soma_ano
    )

    regras[
        'dia_mais_mes'
    ] = reduzir_para_1_25(
        dia_mes
    )

    regras[
        'raiz_dia_mais_mes'
    ] = reduzir_para_1_25(
        rd_dia_mes
    )

    # --------------------------------------------------------------
    # CONCURSO ANTERIOR
    # --------------------------------------------------------------

    if anterior is not None:

        dezenas_ant = (
            anterior['dezenas']
        )

        soma_ant = sum(
            dezenas_ant
        )

        regras[
            'soma_concurso_anterior_mod25'
        ] = reduzir_para_1_25(
            soma_ant
        )

        regras[
            'raiz_soma_anterior'
        ] = reduzir_para_1_25(
            raiz_digital(
                soma_ant
            )
        )

        regras[
            'soma_anterior_mais_rd_data'
        ] = reduzir_para_1_25(
            soma_ant +
            rd_data
        )

        regras[
            'soma_anterior_menos_rd_data'
        ] = reduzir_para_1_25(
            soma_ant -
            rd_data
        )

        # ----------------------------------------------------------
        # DIVISÕES
        # ----------------------------------------------------------

        divisores = {

            'rd_dia':
                rd_dia,

            'rd_data':
                rd_data,

            'dia':
                dia,

            'mes':
                mes,

            'dia_mais_mes':
                dia_mes,

            'soma_dia':
                soma_dia,

            'soma_data':
                soma_data
        }

        for nome, divisor in (
            divisores.items()
        ):

            for modo in [
                'floor',
                'round',
                'ceil'
            ]:

                valor = valor_divisao(
                    soma_ant,
                    divisor,
                    modo
                )

                if valor is not None:

                    regras[
                        f'soma_anterior_div_{nome}_{modo}'
                    ] = valor

        # ----------------------------------------------------------
        # RESTOS / MÓDULOS
        # ----------------------------------------------------------

        valores_mod = {

            'soma_anterior_mod25':
                soma_ant % 25,

            'soma_anterior_mod_dia':
                soma_ant % dia,

            'soma_anterior_mod_mes':
                soma_ant % mes,

            'soma_anterior_mod_rd_data':
                soma_ant % rd_data
        }

        for nome, valor in (
            valores_mod.items()
        ):

            regras[nome] = (
                reduzir_para_1_25(
                    valor
                )
            )

        # ----------------------------------------------------------
        # PRODUTOS
        # ----------------------------------------------------------

        regras[
            'dia_x_rd_data'
        ] = reduzir_para_1_25(
            dia * rd_data
        )

        regras[
            'mes_x_rd_data'
        ] = reduzir_para_1_25(
            mes * rd_data
        )

        regras[
            'soma_anterior_x_rd_data'
        ] = reduzir_para_1_25(
            soma_ant * rd_data
        )

    return regras


# ======================================================================
# ESTATÍSTICA DA OPÇÃO 14
# ======================================================================

def wilson_interval(
    successes,
    n,
    z=1.959963984540054
):

    if n == 0:
        return (
            0.0,
            0.0
        )

    p = successes / n

    denom = (
        1 +
        z**2 / n
    )

    center = (
        p +
        z**2 / (2*n)
    ) / denom

    half = (
        z *
        np.sqrt(
            (
                p * (1-p) / n
            )
            +
            (
                z**2 / (4*n**2)
            )
        )
        /
        denom
    )

    return (
        max(
            0,
            center - half
        ),
        min(
            1,
            center + half
        )
    )


def cohens_h(
    p1,
    p0
):

    p1 = np.clip(
        p1,
        1e-12,
        1 - 1e-12
    )

    p0 = np.clip(
        p0,
        1e-12,
        1 - 1e-12
    )

    return (
        2 * np.arcsin(
            np.sqrt(p1)
        )
        -
        2 * np.arcsin(
            np.sqrt(p0)
        )
    )


def avaliar_regra(
    nome,
    valores,
    alvos
):

    n = len(valores)

    if n == 0:
        return None

    acertos = 0

    hits = []

    for valor, alvo in zip(
        valores,
        alvos
    ):

        hit = (
            int(valor)
            in alvo
        )

        hits.append(
            int(hit)
        )

        acertos += int(hit)

    p_obs = (
        acertos / n
    )

    p0 = 15 / 25

    try:

        bt = binomtest(
            acertos,
            n,
            p0
        )

        p_binom = bt.pvalue

    except Exception:

        p_binom = 1.0

    ci_low, ci_high = (
        wilson_interval(
            acertos,
            n
        )
    )

    h = cohens_h(
        p_obs,
        p0
    )

    return {

        'regra': nome,

        'n': n,

        'acertos': acertos,

        'taxa': p_obs,

        'baseline': p0,

        'excesso': (
            p_obs - p0
        ),

        'p_binom': p_binom,

        'ic_low': ci_low,

        'ic_high': ci_high,

        'cohens_h': h,

        'hits': np.array(
            hits,
            dtype=np.int8
        )
    }


# ======================================================================
# MONTE CARLO DA REGRA
# ======================================================================

def monte_carlo_regra(
    valores,
    n_sim=10000,
    seed=20260828
):

    n = len(valores)

    if n == 0:
        return None

    rng = np.random.default_rng(
        seed
    )

    simulacoes = rng.binomial(
        n,
        15 / 25,
        size=n_sim
    )

    medias = (
        simulacoes / n
    )

    observado = (
        np.sum(
            [
                1
                for v in valores
                if v is not None
            ]
        )
    )

    # Aqui o vetor 'valores' representa uma
    # previsão de uma dezena por concurso.
    # O número de acertos será calculado
    # externamente no caso OOS.

    return medias


# ======================================================================
# MONTE CARLO CORRETO PARA UMA REGRA
# ======================================================================

def monte_carlo_hits(
    n,
    observado,
    n_sim=10000,
    seed=20260828
):

    rng = np.random.default_rng(
        seed
    )

    simulados = rng.binomial(
        n,
        15 / 25,
        size=n_sim
    )

    p_emp = np.mean(
        simulados >= observado
    )

    return {

        'media':
            np.mean(
                simulados / n
            ),

        'std':
            np.std(
                simulados / n
            ),

        'mediana':
            np.median(
                simulados / n
            ),

        'melhor':
            np.max(
                simulados / n
            ),

        'p':
            p_emp,

        'ic_low':
            np.percentile(
                simulados / n,
                2.5
            ),

        'ic_high':
            np.percentile(
                simulados / n,
                97.5
            )
    }


# ======================================================================
# PREPARAÇÃO DOS DADOS TEMPORAIS
# ======================================================================

def preparar_temporal(
    contests
):

    registros = []

    regras_nomes = set()

    for i, contest in enumerate(
        contests
    ):

        anterior = (
            contests[i - 1]
            if i > 0
            else None
        )

        regras = construir_regras_temporais(
            contest,
            anterior
        )

        alvo = set(
            contest['dezenas']
        )

        registros.append({

            'indice': i,

            'concurso':
                contest['concurso'],

            'data':
                contest['data'],

            'regras':
                regras,

            'alvo':
                alvo
        })

        regras_nomes.update(
            regras.keys()
        )

    return (
        registros,
        sorted(regras_nomes)
    )


# ======================================================================
# AVALIAÇÃO WALK-FORWARD DA OPÇÃO 14
# ======================================================================

def avaliar_temporal_oos(
    registros,
    regras_nomes,
    min_history=500
):

    n = len(registros)

    train_end = int(
        n * 0.60
    )

    val_end = int(
        n * 0.80
    )

    print(
        "\n📐 DIVISÃO TEMPORAL"
    )

    print(
        f"   Treino: "
        f"1 – {train_end}"
    )

    print(
        f"   Validação: "
        f"{train_end + 1} – {val_end}"
    )

    print(
        f"   Teste OOS: "
        f"{val_end + 1} – {n}"
    )

    # --------------------------------------------------------------
    # TREINO
    # --------------------------------------------------------------

    resultados_treino = {}

    for regra in regras_nomes:

        valores = []
        alvos = []

        for i in range(
            min_history,
            train_end
        ):

            valor = registros[i][
                'regras'
            ].get(regra)

            if valor is None:
                continue

            valores.append(
                valor
            )

            alvos.append(
                registros[i]['alvo']
            )

        res = avaliar_regra(
            regra,
            valores,
            alvos
        )

        if res is not None:
            resultados_treino[
                regra
            ] = res

    # --------------------------------------------------------------
    # RANKING TREINO
    # --------------------------------------------------------------

    ranking_treino = sorted(

        resultados_treino.values(),

        key=lambda x:
        (
            x['taxa'],
            abs(x['excesso'])
        ),

        reverse=True
    )

    print(
        "\n🏆 TOP 15 REGRAS NO TREINO"
    )

    print(
        f"{'Regra':<42} "
        f"{'Taxa':<8} "
        f"{'Excesso':<9} "
        f"{'p':<10}"
    )

    print("-" * 75)

    for r in ranking_treino[:15]:

        print(
            f"{r['regra'][:41]:<42} "
            f"{r['taxa']*100:>6.2f}% "
            f"{r['excesso']*100:>+7.2f}% "
            f"{r['p_binom']:<10.4g}"
        )

    # --------------------------------------------------------------
    # FDR
    # --------------------------------------------------------------

    pvals = np.array([
        r['p_binom']
        for r in ranking_treino
    ])

    nomes = [
        r['regra']
        for r in ranking_treino
    ]

    if len(pvals) > 0:

        order = np.argsort(
            pvals
        )

        qvals = np.ones(
            len(pvals)
        )

        for i in range(
            len(order) - 1,
            -1,
            -1
        ):

            idx = order[i]

            rank = i + 1

            q = (
                pvals[idx]
                *
                len(pvals)
                /
                rank
            )

            if i < len(order) - 1:

                q = min(
                    q,
                    qvals[
                        order[i + 1]
                    ]
                )

            qvals[idx] = min(
                q,
                1.0
            )

    else:

        qvals = np.array([])

    # --------------------------------------------------------------
    # SELEÇÃO CONTROLADA
    # --------------------------------------------------------------

    candidatos = []

    for i, r in enumerate(
        ranking_treino
    ):

        r2 = dict(r)

        r2['q_fdr'] = (
            qvals[i]
            if i < len(qvals)
            else 1.0
        )

        candidatos.append(
            r2
        )

    candidatos.sort(
        key=lambda x:
        x['taxa'],
        reverse=True
    )

    # Selecionamos até 10 regras,
    # mas NÃO usamos isso para declarar
    # descoberta definitiva.
    regras_selecionadas = [
        r['regra']
        for r in candidatos[:10]
    ]

    print(
        "\n🔎 REGRAS SELECIONADAS "
        "PARA VALIDAÇÃO:"
    )

    for r in candidatos[:10]:

        print(
            f"   {r['regra']}: "
            f"taxa={r['taxa']*100:.2f}% "
            f"p={r['p_binom']:.4g} "
            f"q={r['q_fdr']:.4g}"
        )

    # --------------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------------

    resultados_val = {}

    for regra in regras_selecionadas:

        valores = []
        alvos = []

        for i in range(
            train_end,
            val_end
        ):

            valor = registros[i][
                'regras'
            ].get(regra)

            if valor is None:
                continue

            valores.append(
                valor
            )

            alvos.append(
                registros[i]['alvo']
            )

        res = avaliar_regra(
            regra,
            valores,
            alvos
        )

        if res is not None:
            resultados_val[
                regra
            ] = res

    print(
        "\n📊 VALIDAÇÃO"
    )

    print(
        f"{'Regra':<42} "
        f"{'Taxa':<8} "
        f"{'Excesso':<9} "
        f"{'p':<10}"
    )

    print("-" * 75)

    for regra, r in sorted(
        resultados_val.items(),
        key=lambda x:
        x[1]['taxa'],
        reverse=True
    ):

        print(
            f"{regra[:41]:<42} "
            f"{r['taxa']*100:>6.2f}% "
            f"{r['excesso']*100:>+7.2f}% "
            f"{r['p_binom']:<10.4g}"
        )

    # --------------------------------------------------------------
    # OOS
    # --------------------------------------------------------------

    resultados_oos = {}

    for regra in regras_selecionadas:

        valores = []
        alvos = []

        for i in range(
            val_end,
            n
        ):

            valor = registros[i][
                'regras'
            ].get(regra)

            if valor is None:
                continue

            valores.append(
                valor
            )

            alvos.append(
                registros[i]['alvo']
            )

        res = avaliar_regra(
            regra,
            valores,
            alvos
        )

        if res is not None:

            resultados_oos[
                regra
            ] = res

    print(
        "\n🎯 TESTE OOS — RESULTADO FINAL"
    )

    print(
        f"{'Regra':<42} "
        f"{'Taxa':<8} "
        f"{'Excesso':<9} "
        f"{'p':<10}"
    )

    print("-" * 75)

    ranking_oos = sorted(

        resultados_oos.values(),

        key=lambda x:
        x['taxa'],

        reverse=True
    )

    for r in ranking_oos:

        print(
            f"{r['regra'][:41]:<42} "
            f"{r['taxa']*100:>6.2f}% "
            f"{r['excesso']*100:>+7.2f}% "
            f"{r['p_binom']:<10.4g}"
        )

    return {

        'train':
            resultados_treino,

        'validation':
            resultados_val,

        'oos':
            resultados_oos,

        'selected':
            regras_selecionadas,

        'train_end':
            train_end,

        'val_end':
            val_end
    }


# ======================================================================
# OPÇÃO 14 COMPLETA
# ======================================================================

def analise_numerologica_temporal(
    contests,
    min_history=500,
    n_sim_mc=10000
):

    print(
        "\n" + "=" * 70
    )

    print(
        "🔢 ANÁLISE NUMEROLÓGICA / TEMPORAL "
        f"({VERSION})"
    )

    print(
        "=" * 70
    )

    print(
        "Hipótese:"
    )

    print(
        "   A data do sorteio e/ou "
        "o concurso anterior podem "
        "conter uma associação "
        "estatística com uma dezena "
        "do concurso seguinte."
    )

    print(
        "\n⚠️ A análise não presume "
        "causalidade."
    )

    print(
        "   A hipótese será comparada "
        "contra a probabilidade-base "
        "de uma dezena específica "
        "aparecer: 15/25 = 60%."
    )

    registros, regras_nomes = (
        preparar_temporal(
            contests
        )
    )

    print(
        f"\n📂 Registros analisados: "
        f"{len(registros)}"
    )

    print(
        f"🔢 Regras geradas: "
        f"{len(regras_nomes)}"
    )

    # --------------------------------------------------------------
    # EXEMPLO DO CASO DO USUÁRIO
    # --------------------------------------------------------------

    print(
        "\n🔎 EXEMPLO DA HIPÓTESE"
    )

    for i in range(
        max(
            1,
            len(contests) - 5
        ),
        len(contests)
    ):

        atual = contests[i]

        anterior = (
            contests[i - 1]
            if i > 0
            else None
        )

        regras = construir_regras_temporais(
            atual,
            anterior
        )

        print(
            f"\n   Concurso "
            f"{atual['concurso']} "
            f"| Data: "
            f"{atual['data']}"
        )

        data = parse_data(
            atual['data']
        )

        if data:

            dia, mes, ano = data

            print(
                f"   Dia={dia} "
                f"| Mês={mes} "
                f"| Ano={ano}"
            )

            print(
                f"   Soma dígitos "
                f"do dia = "
                f"{soma_digitos(dia)}"
            )

            print(
                f"   Raiz digital "
                f"do dia = "
                f"{raiz_digital(dia)}"
            )

        if anterior:

            soma_ant = sum(
                anterior['dezenas']
            )

            print(
                f"   Soma concurso "
                f"anterior = "
                f"{soma_ant}"
            )

            if data:

                rd_dia = raiz_digital(
                    dia
                )

                if rd_dia:

                    print(
                        f"   {soma_ant}/"
                        f"{rd_dia} = "
                        f"{soma_ant / rd_dia:.4f}"
                    )

        # Mostrar quais regras
        # efetivamente acertaram.

        acertos_exemplo = []

        alvo = set(
            atual['dezenas']
        )

        for nome, valor in (
            regras.items()
        ):

            if value_in_target(
                valor,
                alvo
            ):
                acertos_exemplo.append(
                    (
                        nome,
                        valor
                    )
                )

        if acertos_exemplo:

            print(
                "   🎯 Regras que "
                "acertaram uma dezena:"
            )

            for nome, valor in (
                acertos_exemplo
            ):

                print(
                    f"      {nome} "
                    f"→ {valor}"
                )

    # --------------------------------------------------------------
    # WALK-FORWARD
    # --------------------------------------------------------------

    resultados = avaliar_temporal_oos(
        registros,
        regras_nomes,
        min_history=min_history
    )

    # --------------------------------------------------------------
    # MONTE CARLO DAS MELHORES REGRAS OOS
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "🎲 MONTE CARLO DAS REGRAS OOS"
    )

    print(
        "=" * 70
    )

    ranking_oos = sorted(

        resultados[
            'oos'
        ].values(),

        key=lambda x:
        x['taxa'],

        reverse=True
    )

    for r in ranking_oos[:10]:

        mc = monte_carlo_hits(
            r['n'],
            r['acertos'],
            n_sim=n_sim_mc
        )

        print(
            f"\n🔹 {r['regra']}"
        )

        print(
            f"   Observado: "
            f"{r['taxa']*100:.2f}%"
        )

        print(
            f"   Esperado: "
            f"60.00%"
        )

        print(
            f"   Diferença: "
            f"{r['excesso']*100:+.2f} pp"
        )

        print(
            f"   Cohen's h: "
            f"{r['cohens_h']:.4f}"
        )

        print(
            f"   Binomial p: "
            f"{r['p_binom']:.6f}"
        )

        print(
            f"   MC média: "
            f"{mc['media']*100:.2f}%"
        )

        print(
            f"   MC IC95%: "
            f"[{mc['ic_low']*100:.2f}%, "
            f"{mc['ic_high']*100:.2f}%]"
        )

        print(
            f"   MC p: "
            f"{mc['p']:.6f}"
        )

    # --------------------------------------------------------------
    # RANKING FINAL
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "🏆 RANKING FINAL — OOS"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Regra':<42} "
        f"{'Taxa':<8} "
        f"{'Excesso':<9} "
        f"{'p':<10}"
    )

    print("-" * 75)

    for r in ranking_oos:

        print(
            f"{r['regra'][:41]:<42} "
            f"{r['taxa']*100:>6.2f}% "
            f"{r['excesso']*100:>+7.2f}% "
            f"{r['p_binom']:<10.4g}"
        )

    # --------------------------------------------------------------
    # INTERPRETAÇÃO AUTOMÁTICA
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "🧠 INTERPRETAÇÃO"
    )

    print(
        "=" * 70
    )

    if not ranking_oos:

        print(
            "❌ Nenhuma regra pôde "
            "ser avaliada."
        )

        return resultados

    melhor = ranking_oos[0]

    print(
        f"Melhor regra OOS: "
        f"{melhor['regra']}"
    )

    print(
        f"Taxa observada: "
        f"{melhor['taxa']*100:.2f}%"
    )

    print(
        f"Baseline: 60.00%"
    )

    print(
        f"Excesso: "
        f"{melhor['excesso']*100:+.2f} "
        f"pontos percentuais"
    )

    print(
        f"p-valor binomial: "
        f"{melhor['p_binom']:.6f}"
    )

    if (
        melhor['p_binom'] < 0.05
        and
        melhor['excesso'] > 0
    ):

        print(
            "\n🔍 Existe uma associação "
            "estatisticamente interessante "
            "para investigação."
        )

        print(
            "⚠️ Isso ainda NÃO demonstra "
            "que a data influencia "
            "o sorteio."
        )

    elif (
        melhor['excesso'] > 0
    ):

        print(
            "\n📈 A melhor regra ficou "
            "acima da baseline, "
            "mas sem evidência "
            "estatística suficiente."
        )

    else:

        print(
            "\n⚠️ A melhor regra OOS "
            "não superou a baseline."
        )

    print(
        "\n🔬 Critério principal:"
    )

    print(
        "   A regra precisa sobreviver "
        "ao teste OOS. Resultados "
        "encontrados somente no "
        "histórico de treino não "
        "serão considerados evidência "
        "forte."
    )

    return resultados


# ======================================================================
# AUXILIAR
# ======================================================================

def value_in_target(
    value,
    alvo
):

    if value is None:
        return False

    try:
        return int(value) in alvo
    except Exception:
        return False


# ======================================================================
# INTERFACE
# ======================================================================

def main():

    print(
        "=" * 70
    )

    print(
        f"🔬 LABORATÓRIO DE ANÁLISE "
        f"ESTRUTURAL DA LOTOFÁCIL – "
        f"{VERSION}"
    )

    print(
        "   OPÇÕES 1, 13 E 14"
    )

    print(
        "=" * 70
    )

    contests = load_all_contests(
        'resultados_lotofacil.csv'
    )

    if not contests:

        print(
            "❌ Arquivo "
            "'resultados_lotofacil.csv' "
            "não encontrado."
        )

        return

    print(
        f"\n📂 {len(contests)} concursos"
    )

    print(
        f"📌 Último: "
        f"{contests[-1]['concurso']} "
        f"- "
        f"{contests[-1]['dezenas']}"
    )

    while True:

        print(
            "\nOpções:"
        )

        print(
            "1. Gerar carteira personalizada"
        )

        print(
            "13. Análise avançada "
            "de frequência + atraso "
            f"({VERSION})"
        )

        print(
            "14. Análise Numerológica/"
            "Temporal da Data + "
            "Concurso Anterior"
        )

        print(
            "0. Sair"
        )

        op = input(
            "Escolha: "
        ).strip()

        # ==============================================================
        # OPÇÃO 1
        # ==============================================================

        if op == '1':

            fixed_str = input(
                "\n   Dezenas fixas "
                "(ex: 15 16 20 ou ENTER): "
            ).strip()

            fixed = (
                [
                    int(x)
                    for x in fixed_str.split()
                ]
                if fixed_str
                else []
            )

            semifixed_str = input(
                "   Dezenas semifixas "
                "(ex: 03 07 14 25 ou ENTER): "
            ).strip()

            semifixed = (
                [
                    int(x)
                    for x in semifixed_str.split()
                ]
                if semifixed_str
                else []
            )

            excl_str = input(
                "   Dezenas excluídas "
                "(ex: 04 18 22 ou ENTER): "
            ).strip()

            excluded = (
                [
                    int(x)
                    for x in excl_str.split()
                ]
                if excl_str
                else []
            )

            if semifixed:

                try:

                    min_semifixed = int(
                        input(
                            f"   Mínimo de "
                            f"semifixas "
                            f"[0-{len(semifixed)}]: "
                        ).strip()
                        or "0"
                    )

                    max_semifixed = int(
                        input(
                            f"   Máximo de "
                            f"semifixas "
                            f"[0-{len(semifixed)}]: "
                        ).strip()
                        or str(
                            len(semifixed)
                        )
                    )

                except Exception:

                    min_semifixed = 0

                    max_semifixed = len(
                        semifixed
                    )

            else:

                min_semifixed = 0

                max_semifixed = None

            print(
                "   Faixas estruturais "
                "(ENTER para pular)"
            )

            try:

                pares_str = input(
                    "   Pares min,max "
                    "(ex: 7,9): "
                ).strip()

                range_pares = (
                    tuple(
                        int(x)
                        for x in
                        pares_str.split(',')
                    )
                    if pares_str
                    else None
                )

            except Exception:

                range_pares = None

            try:

                moldura_str = input(
                    "   Moldura min,max: "
                ).strip()

                range_moldura = (
                    tuple(
                        int(x)
                        for x in
                        moldura_str.split(',')
                    )
                    if moldura_str
                    else None
                )

            except Exception:

                range_moldura = None

            try:

                primos_str = input(
                    "   Primos min,max: "
                ).strip()

                range_primos = (
                    tuple(
                        int(x)
                        for x in
                        primos_str.split(',')
                    )
                    if primos_str
                    else None
                )

            except Exception:

                range_primos = None

            metodo = input(
                "\n   Método "
                "[1. Pair, 2. Triple]: "
            ).strip() or "1"

            method = (
                'pair_covering'
                if metodo == '1'
                else 'triple_covering'
            )

            opt = PortfolioOptimizer(

                contests,

                fixed=fixed,

                semifixed=semifixed,

                min_semifixed=min_semifixed,

                max_semifixed=max_semifixed,

                excluded=excluded,

                range_pares=range_pares,

                range_moldura=range_moldura,

                range_primos=range_primos
            )

            portfolio = opt.optimize(
                5,
                100000,
                method=method
            )

            for i, g in enumerate(
                portfolio,
                1
            ):

                p = sum(
                    1
                    for x in g
                    if x % 2 == 0
                )

                pr = sum(
                    1
                    for x in g
                    if x in PRIMES
                )

                m = sum(
                    1
                    for x in g
                    if x in MOLDURA
                )

                print(
                    f" {i}. {g} "
                    f"| P:{p} "
                    f"Pr:{pr} "
                    f"M:{m}"
                )

            if len(contests) > 200:

                bt = opt.backtest(
                    portfolio,
                    contests[-200:]
                )

                print(
                    f"\n🔬 BACKTEST (200): "
                    f"Lift={bt['lift']:.2f}x "
                    f"| ROI={bt['roi']:+.1f}%"
                )

        # ==============================================================
        # OPÇÃO 13
        # ==============================================================

        elif op == '13':

            try:

                top_n_str = input(
                    "\n   Top_ns a avaliar "
                    "(ex: 5,10,15,20) "
                    "[5,10,15,20]: "
                ).strip()

                top_n_list = (
                    [
                        int(x)
                        for x in
                        top_n_str.split(',')
                    ]
                    if top_n_str
                    else [
                        5,
                        10,
                        15,
                        20
                    ]
                )

                min_history = int(
                    input(
                        "   Histórico mínimo "
                        "[500]: "
                    ).strip()
                    or "500"
                )

                n_sim = int(
                    input(
                        "   Simulações Monte Carlo "
                        "[1000]: "
                    ).strip()
                    or "1000"
                )

            except Exception:

                top_n_list = [
                    5,
                    10,
                    15,
                    20
                ]

                min_history = 500

                n_sim = 1000

            analise_frequentes_atraso_v3(
                contests,
                top_n_list=top_n_list,
                min_history=min_history,
                n_sim_mc=n_sim
            )

        # ==============================================================
        # OPÇÃO 14
        # ==============================================================

        elif op == '14':

            try:

                min_history = int(
                    input(
                        "\n   Histórico mínimo "
                        "para OOS [500]: "
                    ).strip()
                    or "500"
                )

                n_sim = int(
                    input(
                        "   Simulações Monte Carlo "
                        "[10000]: "
                    ).strip()
                    or "10000"
                )

            except Exception:

                min_history = 500

                n_sim = 10000

            analise_numerologica_temporal(
                contests,
                min_history=min_history,
                n_sim_mc=n_sim
            )

        # ==============================================================
        # SAIR
        # ==============================================================

        elif op == '0':

            print(
                "\nEncerrando laboratório."
            )

            break

        else:

            print(
                "Opção inválida."
            )


# ======================================================================
# EXECUÇÃO
# ======================================================================

if __name__ == "__main__":

    main()
