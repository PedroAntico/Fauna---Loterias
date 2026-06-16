#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v64
TESTE DE EFICIÊNCIA DE FIXAS MANUAIS vs. ALEATÓRIAS

OBJETIVO:
✅ Comparar a escolha manual de fixas contra centenas de fixas aleatórias
✅ Walk‑forward: para cada concurso, gerar carteira e medir acertos
✅ Métricas: média de acertos, 13+, 14+, ROI
✅ Z‑score e percentil para avaliar significância
✅ Responde: a escolha intuitiva de fixas agrega valor real?
"""

import numpy as np
from scipy.stats import hypergeom, wilcoxon
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
# GERADOR DE JOGOS
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
        return {'lift': lift, 'roi': roi, 'hit_counts': hit_counts}

# ============================================================
# TESTE DE EFICIÊNCIA DE FIXAS
# ============================================================
def testar_eficiencia_fixas(contests, fixas_manual, n_random_fixas=200, 
                            n_games=5, pool_size=2000, start=3500, step=1):
    """
    Walk‑forward: para cada concurso a partir de 'start', gera uma carteira
    com as fixas manuais e compara com n_random_fixas conjuntos aleatórios
    de fixas (do mesmo tamanho).
    """
    print(f"\n{'='*70}")
    print(f"🔬 TESTE DE EFICIÊNCIA DE FIXAS MANUAIS")
    print(f"{'='*70}")
    print(f"   Fixas manuais: {fixas_manual}")
    print(f"   Fixas aleatórias comparadas: {n_random_fixas}")
    print(f"   Concursos testados: {len(contests) - start}\n")

    n_fixas = len(fixas_manual)
    todas_trincas = list(combinations(range(1, 26), n_fixas))
    
    resultados_manual = []
    resultados_aleatorios = defaultdict(list)  # chave = concurso, valor = lista de acertos

    for t in tqdm(range(start, len(contests)), desc="Walk‑forward"):
        train_data = contests[t-500:t] if t >= 500 else contests[:t]
        test_data = [contests[t]]
        
        # --- Carteira com fixas manuais ---
        opt_manual = PortfolioOptimizer(train_data)
        try:
            pool_manual = opt_manual.generate_pool(pool_size, fixed=fixas_manual)
            portfolio_manual = opt_manual.select_pair_covering(pool_manual, n_games)
            bt_manual = opt_manual.backtest(portfolio_manual, test_data)
            acertos_manual = sum(bt_manual['hit_counts'].values())
            resultados_manual.append({
                'concurso': contests[t]['concurso'],
                'acertos': acertos_manual,
                '13pts': bt_manual['hit_counts'].get(13, 0),
                '14pts': bt_manual['hit_counts'].get(14, 0),
                'roi': bt_manual['roi']
            })
        except Exception as e:
            resultados_manual.append({'concurso': contests[t]['concurso'], 'acertos': 0, '13pts': 0, '14pts': 0, 'roi': 0})
            continue

        # --- Carteiras com fixas aleatórias ---
        random.shuffle(todas_trincas)
        for trinca in todas_trincas[:n_random_fixas]:
            if set(trinca) == set(fixas_manual):
                continue
            try:
                opt_rand = PortfolioOptimizer(train_data)
                pool_rand = opt_rand.generate_pool(pool_size, fixed=list(trinca))
                portfolio_rand = opt_rand.select_pair_covering(pool_rand, n_games)
                bt_rand = opt_rand.backtest(portfolio_rand, test_data)
                acertos_rand = sum(bt_rand['hit_counts'].values())
                resultados_aleatorios[t].append(acertos_rand)
            except:
                resultados_aleatorios[t].append(0)

    # --- Análise estatística ---
    acertos_manual_arr = np.array([r['acertos'] for r in resultados_manual])
    
    # Para cada concurso, média e desvio das aleatórias
    medias_aleatorias = []
    stds_aleatorias = []
    for t in range(start, len(contests)):
        if t in resultados_aleatorios and len(resultados_aleatorios[t]) > 0:
            medias_aleatorias.append(np.mean(resultados_aleatorios[t]))
            stds_aleatorias.append(np.std(resultados_aleatorios[t]))
        else:
            medias_aleatorias.append(0)
            stds_aleatorias.append(0)
    
    medias_aleatorias = np.array(medias_aleatorias)
    stds_aleatorias = np.array(stds_aleatorias)
    
    # Z‑score por concurso (quando std > 0)
    z_scores = np.where(stds_aleatorias > 0, 
                        (acertos_manual_arr - medias_aleatorias) / stds_aleatorias, 
                        0.0)
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Média de acertos (manual): {np.mean(acertos_manual_arr):.3f}")
    print(f"   Média de acertos (aleatório): {np.mean(medias_aleatorias):.3f}")
    print(f"   Z‑score médio: {np.mean(z_scores):.3f}")
    print(f"   Z‑score > 0 em {np.mean(z_scores > 0)*100:.1f}% dos concursos")
    print(f"   Total 13pts (manual): {sum(r['13pts'] for r in resultados_manual)}")
    print(f"   Total 14pts (manual): {sum(r['14pts'] for r in resultados_manual)}")
    print(f"   ROI médio (manual): {np.mean([r['roi'] for r in resultados_manual]):.1f}%")
    
    # Percentil da média manual em relação às médias aleatórias
    if np.std(medias_aleatorias) > 0:
        z_global = (np.mean(acertos_manual_arr) - np.mean(medias_aleatorias)) / (np.std(medias_aleatorias) / np.sqrt(len(medias_aleatorias)))
        print(f"   Z‑score global: {z_global:.3f}")
    
    return resultados_manual, resultados_aleatorios

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v64")
    print("   EFICIÊNCIA DE FIXAS MANUAIS vs. ALEATÓRIAS")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # Definir fixas manuais a testar
    print("\n📝 DEFINIÇÃO DAS FIXAS MANUAIS")
    fixas_str = input("   Digite as fixas (ex: 15,16,20): ").strip()
    try:
        fixas_manual = [int(x) for x in fixas_str.split(',')]
        if len(fixas_manual) < 2:
            print("   É necessário pelo menos 2 fixas.")
            return
    except:
        print("   Formato inválido.")
        return

    # Parâmetros do teste
    try:
        n_random = int(input("   Quantas fixas aleatórias comparar por concurso? [200]: ").strip() or "200")
    except:
        n_random = 200

    testar_eficiencia_fixas(contests, fixas_manual, n_random)

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
