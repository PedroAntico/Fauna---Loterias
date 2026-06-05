#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v53
TESTE FORA DA AMOSTRA DAS MELHORES TRINCAS + RANKING DE DEZENAS

OBJETIVO:
✅ Dividir histórico em duas janelas independentes
✅ Ranquear trincas na janela 1 (treino)
✅ Testar as top 50 trincas na janela 2 (teste)
✅ Verificar se mantêm desempenho acima da média
✅ Ranking de dezenas por aparições nas melhores trincas
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
# AVALIADOR DE TRINCA (VERSÃO OTIMIZADA)
# ============================================================
def evaluate_trinca_v53(contests, trinca, global_pool, global_masks):
    """
    Avalia uma trinca usando o pool global (com baseline justo).
    Retorna score e contagens.
    """
    trinca_set = set(trinca)
    pool_indices = [i for i, g in enumerate(global_pool) if trinca_set.issubset(g)]
    n_trinca = len(pool_indices)
    
    if n_trinca == 0:
        return {'14_count': 0, '13_count': 0, 'score': 0, 'n_jogos': 0}

    pool_masks_trinca = global_masks[pool_indices]
    baseline_indices = np.random.choice(len(global_pool), n_trinca, replace=False)
    baseline_masks = global_masks[baseline_indices]

    hit_counts = {k: 0 for k in range(11, 16)}
    baseline_counts = {k: 0 for k in range(11, 16)}

    for draw in contests:
        dm = BITMASK_CACHE.get_mask(draw['dezenas'])
        best_com = max((mask_intersection(pm, dm) for pm in pool_masks_trinca), default=0)
        if best_com >= 11:
            hit_counts[best_com] += 1

        best_base = max((mask_intersection(pm, dm) for pm in baseline_masks), default=0)
        if best_base >= 11:
            baseline_counts[best_base] += 1

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
        '14_count': hit_counts.get(14, 0),
        '13_count': hit_counts.get(13, 0),
        'score': score,
        'score_baseline': score_baseline,
        'n_jogos': n_trinca
    }

# ============================================================
# RANKING DE DEZENAS POR APARIÇÕES NAS MELHORES TRINCAS
# ============================================================
def rank_dezenas_from_top_trincas(top_trincas):
    """
    Dado uma lista de trincas (ex.: top 50), conta quantas vezes cada dezena aparece.
    """
    dezena_count = Counter()
    for trinca in top_trincas:
        for d in trinca:
            dezena_count[d] += 1
    return dezena_count.most_common()

# ============================================================
# TESTE PRINCIPAL: JANELA 1 (TREINO) → JANELA 2 (TESTE)
# ============================================================
def test_out_of_sample(contests, train_start=2700, train_end=3200, test_start=3201, test_end=3701, pool_size=100000):
    """
    Fase 1: Ranqueia trincas nos concursos train_start..train_end
    Fase 2: Testa as top 50 nos concursos test_start..test_end
    """
    print("\n" + "="*70)
    print("🔬 TESTE FORA DA AMOSTRA DAS MELHORES TRINCAS")
    print("="*70)
    print(f"   Treino: concursos {train_start} a {train_end} ({train_end-train_start+1} concursos)")
    print(f"   Teste:  concursos {test_start} a {test_end} ({test_end-test_start+1} concursos)\n")

    # Dados de treino e teste
    train_contests = [c for c in contests if train_start <= c['concurso'] <= train_end]
    test_contests = [c for c in contests if test_start <= c['concurso'] <= test_end]

    if len(train_contests) < 100 or len(test_contests) < 100:
        print("❌ Janelas muito pequenas. Ajuste os intervalos.")
        return

    # Pool global (comum a treino e teste)
    global_pool = generate_global_pool(pool_size)
    global_masks = np.array([BITMASK_CACHE.get_mask(g) for g in global_pool], dtype=np.uint32)

    # FASE 1: Ranquear as 200 trincas mais frequentes no treino
    print("FASE 1: Ranqueando trincas no período de TREINO...")
    trinca_freq = Counter()
    for c in tqdm(train_contests, desc="Frequência histórica"):
        d = c['dezenas']
        for trinca in combinations(sorted(d), 3):
            trinca_freq[trinca] += 1

    top200 = trinca_freq.most_common(200)
    print(f"   Top 200 trincas mais frequentes selecionadas.")

    results_train = []
    for trinca, freq in tqdm(top200, desc="Avaliando no treino"):
        res = evaluate_trinca_v53(train_contests, trinca, global_pool, global_masks)
        results_train.append((trinca, freq, res))

    results_train.sort(key=lambda x: x[2]['score'], reverse=True)
    top50_train = results_train[:50]

    print(f"\n🏆 TOP 10 TRINCAS NO TREINO:")
    for i, (trinca, freq, res) in enumerate(top50_train[:10], 1):
        print(f"   {i}. {trinca} | Score={res['score']} | 14pts={res['14_count']} | Freq={freq}")

    # FASE 2: Testar as top 50 no teste
    print("\nFASE 2: Testando as top 50 trincas no período de TESTE...")
    results_test = []
    for trinca, freq, _ in tqdm(top50_train, desc="Testando fora da amostra"):
        res = evaluate_trinca_v53(test_contests, trinca, global_pool, global_masks)
        results_test.append((trinca, freq, res))

    # Calcular baseline para o teste (usando 50 trincas aleatórias)
    todas_trincas = list(combinations(range(1, 26), 3))
    top50_set = set(t[0] for t in top50_train)
    restantes = [t for t in todas_trincas if t not in top50_set]
    random.shuffle(restantes)
    random50 = restantes[:50]

    random_scores = []
    for trinca in tqdm(random50, desc="Baseline aleatório"):
        res = evaluate_trinca_v53(test_contests, trinca, global_pool, global_masks)
        random_scores.append(res['score'])

    random_scores = np.array(random_scores)
    mean_random = np.mean(random_scores)
    std_random = np.std(random_scores)

    # Exibir resultados do teste
    print(f"\n📊 RESULTADOS NO TESTE (concursos {test_start}–{test_end}):")
    print(f"{'Trinca':<20} {'Score':<8} {'14pts':<8} {'Z-score':<10} {'% Base':<10}")
    print("-" * 65)

    results_test.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (trinca, freq, res) in enumerate(results_test[:20], 1):
        z = (res['score'] - mean_random) / std_random if std_random > 0 else 0.0
        pct = np.mean(random_scores <= res['score']) * 100
        print(f"{i:<3} {str(trinca):<15} {res['score']:<8} {res['14_count']:<8} {z:<10.2f} {pct:<10.1f}%")

    # Quantas das top 50 estão acima da média?
    above_mean = sum(1 for _, _, res in results_test if res['score'] > mean_random)
    print(f"\n📊 RESUMO:")
    print(f"   Média baseline aleatório: {mean_random:.1f}")
    print(f"   Trincas do treino acima da média no teste: {above_mean}/50 ({above_mean/50*100:.0f}%)")
    print(f"   Z-score médio das top 50: {np.mean([(r[2]['score']-mean_random)/std_random for r in results_test]):.2f}")

    # Ranking de dezenas
    print(f"\n📊 DEZENAS MAIS FREQUENTES NAS TOP 50 DO TREINO:")
    dezena_rank = rank_dezenas_from_top_trincas([t[0] for t in top50_train])
    for i, (dezena, count) in enumerate(dezena_rank[:15], 1):
        bar = "█" * count
        print(f"   {i:2}. Dezena {dezena:2}: {count} aparições {bar}")

    return results_test, dezena_rank

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v53")
    print("   TESTE FORA DA AMOSTRA DAS MELHORES TRINCAS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # Intervalos padrão
    train_start = 2700
    train_end = 3200
    test_start = 3201
    test_end = len(contests)

    print(f"\nConfiguração padrão:")
    print(f"   Treino: concursos {train_start}–{train_end} ({train_end-train_start+1} concursos)")
    print(f"   Teste:  concursos {test_start}–{test_end} ({test_end-test_start+1} concursos)")

    alterar = input("\nAlterar intervalos? (s/n): ").strip().lower()
    if alterar == 's':
        try:
            train_start = int(input("   Início treino: "))
            train_end = int(input("   Fim treino: "))
            test_start = int(input("   Início teste: "))
            test_end = int(input("   Fim teste: "))
        except:
            print("Valores inválidos. Usando padrão.")

    test_out_of_sample(contests, train_start, train_end, test_start, test_end)

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
