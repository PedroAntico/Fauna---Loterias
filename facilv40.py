#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49
TESTE DE SIGNIFICÂNCIA DA TRINCA (15,16,20) vs. TRINCAS ALEATÓRIAS

OBJETIVO:
✅ Comparar a trinca 15‑16‑20 com 100 trincas aleatórias em walk‑forward
✅ Medir quantos acertos de 14 pontos cada trinca produz
✅ Análise detalhada dos dois concursos onde 15‑16‑20 fez 14 pontos
✅ Determinar se a trinca é especial ou apenas teve sorte
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
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

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
# GERADOR
# ============================================================
class LooseGenerator:
    def generate_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))
    def generate_with_fixed(self, fixed):
        fixed_set = set(fixed)
        restantes = list(set(range(1, 26)) - fixed_set)
        complemento = np.random.choice(restantes, 15 - len(fixed_set), replace=False)
        return sorted(fixed_set | set(complemento))

# ============================================================
# OTIMIZADOR DE CARTEIRA (PAIR COVERING)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests):
        self.contests = contests
        self.generator = LooseGenerator()

    def generate_pool(self, n_candidates, fixed=None):
        pool = []
        seen = set()
        for _ in range(n_candidates):
            if fixed:
                g = self.generator.generate_with_fixed(fixed)
            else:
                g = self.generator.generate_random()
            key = tuple(g)
            if key not in seen:
                seen.add(key)
                pool.append(g)
        return pool

    def select_pair_covering(self, candidates, n_select):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        covered = set()
        selected = []
        for _ in range(n_select):
            best_idx = -1
            best_new = -1
            for i, c in enumerate(candidates):
                if c in selected:
                    continue
                pairs = set(combinations(sorted(c), 2))
                new_pairs = len(pairs - covered)
                if new_pairs > best_new:
                    best_new = new_pairs
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), 2))
        return selected

    def backtest(self, portfolio, test_draws):
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        best_per_draw = {}  # guarda o melhor jogo para cada concurso
        for idx, draw in enumerate(test_draws):
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            best_hit = 0
            best_game = None
            for i, pm in enumerate(portfolio_masks):
                hits = mask_intersection(pm, dm)
                if hits > best_hit:
                    best_hit = hits
                    best_game = portfolio[i]
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
            if best_hit >= 14:
                best_per_draw[draw['concurso']] = (best_hit, best_game)
        prob = n_success / (len(portfolio) * len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1 - p_single) ** len(portfolio)
        lift = prob / theo_prob if theo_prob > 0 else 1.0
        roi = (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0
        return {'lift': lift, 'roi': roi, 'hit_counts': hit_counts, 'best_per_draw': best_per_draw}

# ============================================================
# WALK‑FORWARD PARA UMA TRINCA
# ============================================================
def walk_forward_single_trinca(contests, trinca, train_size=500, test_size=50, step=50, n_games=5):
    """Retorna total de 14pts, lista de concursos com 14pts e detalhes."""
    total_14 = 0
    detalhes_14 = []  # (concurso, jogo vencedor, dezenas sorteadas)
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        opt = PortfolioOptimizer(train_data)
        try:
            pool = opt.generate_pool(2000, fixed=list(trinca))
            portfolio = opt.select_pair_covering(pool, n_games)
            bt = opt.backtest(portfolio, test_data)
            count_14 = bt['hit_counts'].get(14, 0)
            total_14 += count_14
            # Guardar detalhes dos concursos com 14 pontos
            for concurso, (hits, game) in bt['best_per_draw'].items():
                # Encontrar as dezenas sorteadas
                draw = next(d for d in test_data if d['concurso'] == concurso)
                detalhes_14.append({
                    'concurso': concurso,
                    'dezenas_sorteadas': draw['dezenas'],
                    'jogo_vencedor': game,
                    'hits': hits,
                    'trinca_presente': sum(1 for d in trinca if d in game)
                })
        except Exception as e:
            pass
        start += step
    return total_14, detalhes_14

# ============================================================
# TESTE PRINCIPAL
# ============================================================
def test_trinca_significance(contests, n_random_trincas=100):
    print("\n" + "="*70)
    print("🔬 TESTE DE SIGNIFICÂNCIA DA TRINCA (15,16,20)")
    print("="*70)
    print(f"   Comparando contra {n_random_trincas} trincas aleatórias em walk‑forward")
    print("   (Treino 500, Teste 50, Passo 50)\n")

    # 1. Testar a trinca 15-16-20
    print("Testando trinca (15,16,20)...")
    t0 = time.time()
    count_15_16_20, detalhes_15_16_20 = walk_forward_single_trinca(contests, (15,16,20))
    print(f"   ✅ 14 pontos obtidos: {count_15_16_20} (tempo: {time.time()-t0:.1f}s)")

    # 2. Gerar n_random_trincas trincas aleatórias (sem repetir a original)
    todas_trincas = list(combinations(range(1, 26), 3))
    # Remover a trinca original se estiver presente
    if (15,16,20) in todas_trincas:
        todas_trincas.remove((15,16,20))
    random.shuffle(todas_trincas)
    trincas_aleatorias = todas_trincas[:n_random_trincas]

    resultados_aleatorios = []
    print(f"\nTestando {n_random_trincas} trincas aleatórias...")
    for i, trinca in enumerate(tqdm(trincas_aleatorias, desc="Progresso"), 1):
        total_14, _ = walk_forward_single_trinca(contests, trinca)
        resultados_aleatorios.append(total_14)

    # 3. Análise estatística
    resultados_aleatorios = np.array(resultados_aleatorios)
    media_aleatoria = np.mean(resultados_aleatorios)
    std_aleatoria = np.std(resultados_aleatorios)
    pct_rank = np.mean(resultados_aleatorios <= count_15_16_20) * 100  # percentil

    print("\n" + "="*70)
    print("📊 RESULTADOS DA COMPARAÇÃO")
    print("="*70)
    print(f"   Trinca (15,16,20): {count_15_16_20} acertos de 14 pontos")
    print(f"   Média das aleatórias: {media_aleatoria:.2f}")
    print(f"   Desvio padrão das aleatórias: {std_aleatoria:.2f}")
    print(f"   Mínimo / Máximo: {resultados_aleatorios.min()} / {resultados_aleatorios.max()}")
    print(f"   Percentil da trinca (15,16,20): {pct_rank:.1f}%")
    if pct_rank >= 95:
        print("   🔍 Resultado no top 5% — possível sinal.")
    elif pct_rank >= 80:
        print("   📊 Acima da média, mas não extremo.")
    else:
        print("   📊 Desempenho dentro do esperado ao acaso.")

    # 4. Análise detalhada dos concursos com 14 pontos
    if detalhes_15_16_20:
        print("\n" + "="*70)
        print("🔎 ANÁLISE DOS CONCURSOS COM 14 PONTOS (TRINCA 15,16,20)")
        print("="*70)
        for d in detalhes_15_16_20:
            print(f"\n   Concurso {d['concurso']}:")
            print(f"   Dezenas sorteadas: {d['dezenas_sorteadas']}")
            print(f"   Jogo vencedor:    {d['jogo_vencedor']}")
            print(f"   Acertos: {d['hits']}")
            print(f"   Dezenas da trinca no jogo: {d['trinca_presente']}/3")
            # Verificar quantas da trinca estavam no sorteio
            trinca_set = set((15,16,20))
            acertou_trinca = trinca_set.issubset(set(d['dezenas_sorteadas']))
            print(f"   A trinca inteira estava no sorteio? {'Sim' if acertou_trinca else 'Não'}")
            # Jogo sem a trinca: quantas seriam substituídas?
            jogo_sem_trinca = [x for x in d['jogo_vencedor'] if x not in trinca_set]
            print(f"   Jogo sem as fixas: {len(jogo_sem_trinca)} dezenas restantes: {jogo_sem_trinca[:5]}...")

    print("\n✅ Teste concluído.")
    return count_15_16_20, media_aleatoria, pct_rank, detalhes_15_16_20

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v49")
    print("   TESTE DE SIGNIFICÂNCIA DA TRINCA (15,16,20)")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    # Número de trincas aleatórias (pode ser ajustado)
    n_random = 100
    test_trinca_significance(contests, n_random)

if __name__ == "__main__":
    main()
