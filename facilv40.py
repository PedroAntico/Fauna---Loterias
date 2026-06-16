#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v65
TESTE DE HEURÍSTICA DINÂMICA PARA FIXAS E SEMIFIXAS

OBJETIVO:
✅ Simular uma regra automatizada para escolha de fixas e semifixas
   usando apenas informações disponíveis até o concurso anterior.
✅ Para cada concurso (walk‑forward):
   - Seleciona top K dezenas por score = frequência recente + bônus de atraso
   - As primeiras F viram fixas; as S seguintes viram semifixas
   - Gera carteira com Pair Covering respeitando fixas e semifixas
✅ Compara acertos com centenas de seleções aleatórias de fixas/semifixas
✅ Métricas: média de acertos, z‑score, total 13/14 pontos, ROI
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings
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
# GERADOR DE JOGOS (COM FIXAS E SEMIFIXAS)
# ============================================================
class LooseGenerator:
    def generate_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

    def generate_with_fixed_and_semi(self, fixed, semifixed, min_semi, max_semi):
        """
        Gera um jogo contendo todas as fixas e entre min_semi e max_semi semifixas.
        """
        fixed_set = set(fixed)
        semi_set = set(semifixed) - fixed_set
        proibidas = fixed_set | semi_set
        restantes = list(set(range(1, 26)) - proibidas)
        n_fixas = len(fixed_set)
        max_s = min(max_semi, len(semi_set))
        min_s = max(min_semi, 0)
        if min_s > max_s:
            return None
        for _ in range(200):
            n_semi = random.randint(min_s, max_s)
            n_rest = 15 - n_fixas - n_semi
            if n_rest < 0 or n_rest > len(restantes):
                continue
            chosen_semi = set(random.sample(list(semi_set), n_semi)) if n_semi > 0 and len(semi_set) > 0 else set()
            chosen_rest = set(random.sample(restantes, n_rest)) if n_rest > 0 else set()
            game = sorted(fixed_set | chosen_semi | chosen_rest)
            if len(game) == 15:
                return game
        return None

# ============================================================
# HEURÍSTICA PARA SELEÇÃO DE FIXAS E SEMIFIXAS
# ============================================================
def selecionar_fixas_automaticas(contests, n_fixas=4, n_semifixas=7, 
                                 janela_freq=50, peso_atraso=0.3):
    """
    Dado o histórico até o momento, ranqueia as 25 dezenas por:
        score[d] = freq_recente[d] + peso_atraso * (1 / (atraso[d] + 1))
    onde freq_recente é a proporção de aparições nos últimos janela_freq concursos,
    e atraso é o número de concursos desde a última aparição.
    As top n_fixas viram fixas; as n_semifixas seguintes viram semifixas.
    """
    if len(contests) == 0:
        return [], []
    
    total = len(contests)
    recent = contests[-janela_freq:] if janela_freq < total else contests
    
    freq_recente = Counter()
    for c in recent:
        freq_recente.update(c['dezenas'])
    for d in range(1, 26):
        freq_recente[d] = freq_recente.get(d, 0) / len(recent)
    
    atraso = {}
    last_seen = {d: -1 for d in range(1, 26)}
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            last_seen[d] = i
    for d in range(1, 26):
        atraso[d] = (total - 1) - last_seen[d]  # concursos desde a última aparição
    
    # Score combinado
    scores = {}
    for d in range(1, 26):
        score = freq_recente[d] + peso_atraso / (atraso[d] + 1)
        scores[d] = score
    
    # Ordenar dezenas por score decrescente
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    fixas = [d for d, _ in ranked[:n_fixas]]
    semifixas = [d for d, _ in ranked[n_fixas:n_fixas+n_semifixas]]
    return fixas, semifixas

# ============================================================
# OTIMIZADOR DE CARTEIRA (PAIR COVERING COM RESTRIÇÕES)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests):
        self.contests = contests
        self.generator = LooseGenerator()

    def generate_pool(self, n_candidates, fixed, semifixed, min_semi, max_semi):
        pool = []
        seen = set()
        for _ in range(n_candidates):
            g = self.generator.generate_with_fixed_and_semi(fixed, semifixed, min_semi, max_semi)
            if g is None:
                continue
            key = tuple(g)
            if key not in seen:
                seen.add(key)
                pool.append(g)
        return pool

    def select_pair_covering(self, candidates, n_select):
        if len(candidates) < n_select:
            return candidates[:n_select] if candidates else []
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
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success / (len(portfolio) * len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1 - p_single) ** len(portfolio)
        lift = prob / theo_prob if theo_prob > 0 else 1.0
        roi = (total_premio - total_custo) / total_custo * 100 if total_custo > 0 else 0
        return {'acertos': sum(hit_counts.values()), 'lift': lift, 'roi': roi, 
                'hit_counts': hit_counts}

# ============================================================
# TESTE DE HEURÍSTICA DINÂMICA
# ============================================================
def testar_heuristica_dinamica(contests, n_fixas=4, n_semifixas=7, 
                               min_semi=1, max_semi=3, n_random=200,
                               janela_freq=50, peso_atraso=0.3,
                               start=500, n_games=5, pool_size=2000):
    """
    Walk‑forward: para cada concurso a partir de 'start', 
    seleciona fixas e semifixas com a heurística, gera carteira e
    compara com n_random seleções aleatórias de fixas/semifixas.
    """
    print(f"\n{'='*70}")
    print(f"🔬 TESTE DE HEURÍSTICA DINÂMICA PARA FIXAS")
    print(f"{'='*70}")
    print(f"   Fixas: {n_fixas} | Semifixas: {n_semifixas} (usadas {min_semi} a {max_semi} por jogo)")
    print(f"   Heurística: frequência ({janela_freq} conc.) + atraso (peso {peso_atraso})")
    print(f"   Comparação: {n_random} seleções aleatórias por concurso\n")

    resultados_heuristicos = []
    resultados_aleatorios = defaultdict(list)

    for t in tqdm(range(start, len(contests)), desc="Walk‑forward"):
        treino = contests[:t]  # dados disponíveis até t-1
        teste = [contests[t]]
        
        # Selecionar fixas/semifixas com a heurística
        fixas, semifixas = selecionar_fixas_automaticas(
            treino, n_fixas, n_semifixas, janela_freq, peso_atraso
        )
        
        # Carteira heurística
        opt = PortfolioOptimizer(treino)
        try:
            pool_heu = opt.generate_pool(pool_size, fixas, semifixas, min_semi, max_semi)
            portfolio_heu = opt.select_pair_covering(pool_heu, n_games)
            bt_heu = opt.backtest(portfolio_heu, teste)
            resultados_heuristicos.append(bt_heu)
        except Exception as e:
            resultados_heuristicos.append({'acertos': 0, 'hit_counts': {}, 'lift': 0, 'roi': 0})
            continue

        # Carteiras aleatórias (mesmo número de fixas e semifixas)
        for _ in range(n_random):
            fixas_rand = random.sample(range(1, 26), n_fixas)
            # semifixas aleatórias (sem interseção com fixas)
            candidatas_semi = [d for d in range(1, 26) if d not in fixas_rand]
            semifixas_rand = random.sample(candidatas_semi, min(n_semifixas, len(candidatas_semi)))
            try:
                pool_rand = opt.generate_pool(pool_size, fixas_rand, semifixas_rand, min_semi, max_semi)
                portfolio_rand = opt.select_pair_covering(pool_rand, n_games)
                bt_rand = opt.backtest(portfolio_rand, teste)
                resultados_aleatorios[t].append(bt_rand['acertos'])
            except:
                resultados_aleatorios[t].append(0)

    # Análise
    acertos_heu = np.array([r['acertos'] for r in resultados_heuristicos])
    medias_rand = []
    for t in range(start, len(contests)):
        if t in resultados_aleatorios and len(resultados_aleatorios[t]) > 0:
            medias_rand.append(np.mean(resultados_aleatorios[t]))
        else:
            medias_rand.append(0)
    medias_rand = np.array(medias_rand)

    dif = acertos_heu - medias_rand
    z_scores = dif / np.std(medias_rand) if np.std(medias_rand) > 0 else np.zeros_like(dif)

    print(f"\n📊 RESULTADOS (heurística vs. aleatório):")
    print(f"   Média acertos (heurística): {np.mean(acertos_heu):.3f}")
    print(f"   Média acertos (aleatório):   {np.mean(medias_rand):.3f}")
    print(f"   Z‑score médio: {np.mean(z_scores):.3f}")
    print(f"   % concursos com z > 0: {np.mean(z_scores > 0)*100:.1f}%")
    print(f"   Total 13pts (heurística): {sum(r['hit_counts'].get(13,0) for r in resultados_heuristicos)}")
    print(f"   Total 14pts (heurística): {sum(r['hit_counts'].get(14,0) for r in resultados_heuristicos)}")
    print(f"   ROI médio (heurística): {np.mean([r['roi'] for r in resultados_heuristicos]):.1f}%")

    return resultados_heuristicos

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v65")
    print("   HEURÍSTICA DINÂMICA PARA FIXAS E SEMIFIXAS")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # Parâmetros configuráveis
    print("\n⚙️ Parâmetros da heurística (ENTER para padrão):")
    try:
        n_fixas = int(input("   Número de fixas [4]: ").strip() or "4")
        n_semifixas = int(input("   Número de semifixas [7]: ").strip() or "7")
        min_semi = int(input("   Mínimo de semifixas por jogo [1]: ").strip() or "1")
        max_semi = int(input("   Máximo de semifixas por jogo [3]: ").strip() or "3")
        janela = int(input("   Janela para frequência recente [50]: ").strip() or "50")
        peso = float(input("   Peso do atraso [0.3]: ").strip() or "0.3")
        n_random = int(input("   Comparações aleatórias por concurso [200]: ").strip() or "200")
        start = int(input("   Início do walk‑forward [500]: ").strip() or "500")
    except ValueError:
        print("   Valor inválido. Usando padrões.")
        n_fixas, n_semifixas, min_semi, max_semi = 4, 7, 1, 3
        janela, peso, n_random, start = 50, 0.3, 200, 500

    testar_heuristica_dinamica(
        contests, n_fixas, n_semifixas, min_semi, max_semi,
        n_random, janela, peso, start
    )

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
