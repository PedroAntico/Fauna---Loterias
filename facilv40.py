#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v52
BASELINE JUSTO + Z-SCORE + FILTRO PRÉVIO DAS 2300 TRINCAS

CORREÇÕES METODOLÓGICAS:
✅ Baseline com mesmo número de jogos que a trinca (justo)
✅ Z-score = (score - média_aleatória) / desvio_aleatório
✅ Otimização: única passada para argmax (sem recalcular)
✅ Pool global de 100.000 jogos
✅ Teste das 2300 trincas: filtra top 200 por frequência histórica primeiro
✅ Mantém distribuição da trinca nos sorteios (0,1,2,3)
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
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
# GERADOR DE POOL GLOBAL
# ============================================================
def generate_global_pool(n_games=100000):
    """Gera um pool global de jogos aleatórios (sem restrições)."""
    print(f"Gerando pool global de {n_games} jogos...")
    pool = []
    seen = set()
    while len(pool) < n_games:
        game = sorted(np.random.choice(range(1, 26), 15, replace=False))
        key = tuple(game)
        if key not in seen:
            seen.add(key)
            pool.append(game)
    return pool

# ============================================================
# AVALIADOR DE TRINCA COM BASELINE JUSTO (v52)
# ============================================================
def evaluate_trinca_v52(contests, trinca, global_pool, global_masks, test_draws=None):
    """
    Avalia uma trinca com baseline justo (mesmo número de jogos).
    Retorna score, contagens e z-score (calculado externamente).
    """
    if test_draws is None:
        test_draws = contests[-500:] if len(contests) >= 500 else contests

    trinca_set = set(trinca)
    # Filtra jogos do pool que contêm a trinca
    pool_indices = [i for i, g in enumerate(global_pool) if trinca_set.issubset(g)]
    n_trinca = len(pool_indices)
    
    if n_trinca == 0:
        return {'14_count': 0, '13_count': 0, 'score': 0, 'trinca_dist': Counter(), 'n_jogos': 0}

    pool_masks_trinca = global_masks[pool_indices]

    # Baseline justo: mesma quantidade de jogos, mas escolhidos aleatoriamente do pool global
    baseline_indices = np.random.choice(len(global_pool), n_trinca, replace=False)
    baseline_masks = global_masks[baseline_indices]

    hit_counts = {k: 0 for k in range(11, 16)}
    baseline_counts = {k: 0 for k in range(11, 16)}
    trinca_dist = Counter()

    for draw in tqdm(test_draws, desc=f"  Avaliando {trinca}", leave=False):
        dm = BITMASK_CACHE.get_mask(draw['dezenas'])
        drawn_set = set(draw['dezenas'])

        # Distribuição da trinca no sorteio
        trinca_hit = len(trinca_set & drawn_set)
        trinca_dist[trinca_hit] += 1

        # Melhor jogo COM a trinca (otimizado: única passada)
        best_com = 0
        for pm in pool_masks_trinca:
            hits = mask_intersection(pm, dm)
            if hits > best_com:
                best_com = hits
        if best_com >= 11:
            hit_counts[best_com] += 1

        # Baseline (sem trinca)
        best_base = 0
        for pm in baseline_masks:
            hits = mask_intersection(pm, dm)
            if hits > best_base:
                best_base = hits
        if best_base >= 11:
            baseline_counts[best_base] += 1

    # Score
    score = (
        hit_counts.get(13, 0) * 1 +
        hit_counts.get(14, 0) * 20 +
        hit_counts.get(15, 0) * 1000
    )
    score_baseline = (
        baseline_counts.get(13, 0) * 1 +
        baseline_counts.get(14, 0) * 20 +
        baseline_counts.get(15, 0) * 1000
    )

    return {
        'hit_counts': hit_counts,
        'baseline_counts': baseline_counts,
        '14_count': hit_counts.get(14, 0),
        '13_count': hit_counts.get(13, 0),
        '15_count': hit_counts.get(15, 0),
        'score': score,
        'score_baseline': score_baseline,
        'trinca_dist': trinca_dist,
        'n_jogos': n_trinca
    }

# ============================================================
# COMPARAÇÃO DE TRINCAS (COM Z-SCORE)
# ============================================================
def compare_trincas_v52(contests, trincas_interesse, n_random=100, pool_size=100000):
    """
    Compara as trincas de interesse com trincas aleatórias e baseline justo.
    Calcula z-score.
    """
    print("\n" + "="*70)
    print("🔬 AVALIAÇÃO DE TRINCAS COM BASELINE JUSTO (v52)")
    print("="*70)
    print(f"   Pool global: {pool_size} jogos")
    print(f"   Concursos: últimos 500\n")

    # 1. Gerar pool global
    t0 = time.time()
    global_pool = generate_global_pool(pool_size)
    global_masks = np.array([BITMASK_CACHE.get_mask(g) for g in global_pool], dtype=np.uint32)
    print(f"   Pool gerado em {time.time()-t0:.1f}s")

    # 2. Avaliar trincas de interesse
    print("\n📊 AVALIANDO TRINCAS DE INTERESSE...")
    results = {}
    for trinca in trincas_interesse:
        print(f"\n   Trinca {trinca}:")
        res = evaluate_trinca_v52(contests, trinca, global_pool, global_masks, test_draws=contests[-500:])
        results[trinca] = res
        print(f"      Jogos com a trinca: {res['n_jogos']}")
        print(f"      Score trinca: {res['score']} | Score baseline: {res['score_baseline']}")
        print(f"      Distribuição nos sorteios: 0={res['trinca_dist'][0]}, 1={res['trinca_dist'][1]}, 2={res['trinca_dist'][2]}, 3={res['trinca_dist'][3]}")

    # 3. Gerar e avaliar trincas aleatórias
    print(f"\n📊 AVALIANDO {n_random} TRINCAS ALEATÓRIAS...")
    todas_trincas = list(combinations(range(1, 26), 3))
    for t in trincas_interesse:
        if t in todas_trincas:
            todas_trincas.remove(t)
    random.shuffle(todas_trincas)
    random_trincas = todas_trincas[:n_random]

    random_scores = []
    for trinca in tqdm(random_trincas, desc="Progresso"):
        res = evaluate_trinca_v52(contests, trinca, global_pool, global_masks, test_draws=contests[-500:])
        random_scores.append(res['score'])

    random_scores = np.array(random_scores)
    mean_random = np.mean(random_scores)
    std_random = np.std(random_scores)

    # 4. Exibir resultados com z-score
    print("\n" + "="*70)
    print("📊 RESULTADOS COMPARATIVOS")
    print("="*70)
    print(f"{'Trinca':<20} {'Score':<8} {'Base':<8} {'14pts':<8} {'Z-score':<10} {'% Base':<10}")
    print("-" * 80)

    for trinca, res in results.items():
        z = (res['score'] - mean_random) / std_random if std_random > 0 else 0.0
        pct = np.mean(random_scores <= res['score']) * 100
        print(f"{str(trinca):<20} {res['score']:<8} {res['score_baseline']:<8} {res['14_count']:<8} {z:<10.2f} {pct:<10.1f}%")

    print(f"\n📊 BASELINE ALEATÓRIO ({n_random} trincas):")
    print(f"   Média de score: {mean_random:.1f} | Desvio: {std_random:.1f}")

    for trinca, res in results.items():
        z = (res['score'] - mean_random) / std_random if std_random > 0 else 0.0
        if z > 2.0:
            print(f"\n🔍 {trinca} com z={z:.2f} — sinal forte (>2 desvios).")
        elif z > 1.0:
            print(f"\n📊 {trinca} com z={z:.2f} — acima da média.")

    return results

# ============================================================
# TESTE DAS 2300 TRINCAS (FILTRADO POR FREQUÊNCIA)
# ============================================================
def test_all_trincas_filtered(contests, pool_size=100000, top_n=50):
    """
    Filtra as 200 trincas mais frequentes no histórico e testa apenas essas.
    """
    print("\n" + "="*70)
    print("🔬 TESTANDO TOP 200 TRINCAS POR FREQUÊNCIA HISTÓRICA")
    print("="*70)

    # Fase 1: frequência histórica
    print("Fase 1: calculando frequência de todas as 2300 trincas...")
    trinca_freq = Counter()
    for c in tqdm(contests, desc="Histórico"):
        d = c['dezenas']
        for trinca in combinations(sorted(d), 3):
            trinca_freq[trinca] += 1

    top200 = trinca_freq.most_common(200)
    print(f"   Top 200 trincas mais frequentes selecionadas.")

    # Fase 2: pool global
    t0 = time.time()
    global_pool = generate_global_pool(pool_size)
    global_masks = np.array([BITMASK_CACHE.get_mask(g) for g in global_pool], dtype=np.uint32)
    print(f"   Pool gerado em {time.time()-t0:.1f}s")

    # Fase 3: avaliar as 200 trincas
    print("Fase 2: avaliando as 200 trincas com score...")
    results = []
    for trinca, freq in tqdm(top200, desc="Avaliando"):
        res = evaluate_trinca_v52(contests, trinca, global_pool, global_masks, test_draws=contests[-500:])
        results.append((trinca, freq, res))

    # Ordenar por score
    results.sort(key=lambda x: x[2]['score'], reverse=True)

    print(f"\n🏆 TOP {top_n} TRINCAS POR SCORE:")
    print(f"{'Rank':<5} {'Trinca':<20} {'Score':<8} {'14pts':<8} {'13pts':<8} {'Freq Hist':<12}")
    print("-" * 65)
    for i, (trinca, freq, res) in enumerate(results[:top_n], 1):
        print(f"{i:<5} {str(trinca):<20} {res['score']:<8} {res['14_count']:<8} {res['13_count']:<8} {freq:<12}")

    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v52")
    print("   BASELINE JUSTO + Z-SCORE + FILTRO PRÉVIO")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    while True:
        print("\nOpções:")
        print("1. Comparar trincas específicas (com baseline justo e z-score)")
        print("2. Testar top 200 trincas (filtradas por frequência histórica)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            trincas_interesse = [
                (15, 16, 20),
                (4, 6, 11),
                (1, 2, 3),
                (5, 10, 15),
                (7, 11, 23)
            ]
            extra = input("   Adicionar trinca extra (ex: 4,13,25 ou ENTER): ").strip()
            if extra:
                try:
                    t = tuple(int(x) for x in extra.split(','))
                    if len(t) == 3:
                        trincas_interesse.append(t)
                except:
                    pass
            compare_trincas_v52(contests, trincas_interesse, n_random=100, pool_size=100000)

        elif op == '2':
            print("\n⚠️ Esta operação pode levar alguns minutos.")
            confirm = input("   Continuar? (s/n): ").strip().lower()
            if confirm == 's':
                test_all_trincas_filtered(contests, pool_size=100000, top_n=50)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
