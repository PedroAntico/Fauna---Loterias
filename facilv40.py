#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v48
COMPARAÇÃO LIMPA DE ESTRATÉGIAS PURAS EM WALK‑FORWARD

ESTRATÉGIAS COMPARADAS:
✅ Carteira A: 5 jogos totalmente aleatórios (baseline)
✅ Carteira B: 5 jogos com Pair Covering (sem filtros)
✅ Carteira C: 5 jogos com Triple Covering (sem filtros)
✅ Carteira D: 5 jogos com trinca fixa (15,16,20) + Pair Covering (sem outros filtros)

MÉTRICAS:
- Lift médio por concurso
- ROI médio
- Total de 13 pontos
- Total de 14 pontos

VALIDADE:
- Walk‑forward completo (treino 500, teste 50, passo 50)
- Nenhuma filtragem estrutural adicional
- Comparação direta contra baseline aleatório
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter
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
# GERADOR DE JOGOS (SIMPLES, SEM FILTROS)
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_random(self):
        """Gera um jogo completamente aleatório (15 dezenas distintas)."""
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

    def generate_with_fixed(self, fixed):
        """Gera um jogo que contém as dezenas fixas, completando com aleatórias."""
        fixed_set = set(fixed)
        restantes = list(set(range(1, 26)) - fixed_set)
        complemento = np.random.choice(restantes, 15 - len(fixed_set), replace=False)
        return sorted(fixed_set | set(complemento))

# ============================================================
# OTIMIZADOR DE CARTEIRAS (COBERTURA PURA)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests):
        self.contests = contests
        self.generator = LooseGenerator()

    def generate_pool(self, n_candidates, fixed=None):
        """Gera um pool de jogos aleatórios (com fixas opcionais)."""
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
        """Seleciona n_select jogos maximizando a cobertura de pares distintos."""
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

    def select_triple_covering(self, candidates, n_select):
        """Seleciona n_select jogos maximizando a cobertura de triplas distintas."""
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
                triples = set(combinations(sorted(c), 3))
                new_triples = len(triples - covered)
                if new_triples > best_new:
                    best_new = new_triples
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), 3))
        return selected

    def backtest(self, portfolio, test_draws):
        """Calcula métricas de desempenho da carteira nos concursos de teste."""
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
# WALK‑FORWARD COMPARATIVO
# ============================================================
def walk_forward_comparison(contests, train_size=500, test_size=50, step=50, n_games=5):
    """
    Executa walk‑forward para 4 estratégias:
    A: Aleatório
    B: Pair Covering
    C: Triple Covering
    D: Trinca fixa (15,16,20) + Pair Covering
    """
    strategies = {
        'A (Aleatório)': {'method': 'random', 'fixed': None},
        'B (Pair Covering)': {'method': 'pair_covering', 'fixed': None},
        'C (Triple Covering)': {'method': 'triple_covering', 'fixed': None},
        'D (Fixas 15,16,20 + Pair)': {'method': 'pair_covering', 'fixed': [15, 16, 20]},
    }

    results = {name: {'lift': [], 'roi': [], '13pts': [], '14pts': []} for name in strategies}

    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start - train_size:start]
        test_data = contests[start:start + test_size]

        opt = PortfolioOptimizer(train_data)

        for name, cfg in strategies.items():
            try:
                if cfg['method'] == 'random':
                    # Carteira aleatória pura
                    portfolio = [opt.generator.generate_random() for _ in range(n_games)]
                elif cfg['method'] == 'pair_covering':
                    pool = opt.generate_pool(2000, fixed=cfg['fixed'])
                    portfolio = opt.select_pair_covering(pool, n_games)
                elif cfg['method'] == 'triple_covering':
                    pool = opt.generate_pool(2000, fixed=cfg['fixed'])
                    portfolio = opt.select_triple_covering(pool, n_games)
                else:
                    continue

                bt = opt.backtest(portfolio, test_data)
                results[name]['lift'].append(bt['lift'])
                results[name]['roi'].append(bt['roi'])
                results[name]['13pts'].append(bt['hit_counts'].get(13, 0))
                results[name]['14pts'].append(bt['hit_counts'].get(14, 0))
            except Exception as e:
                # Em caso de erro (ex.: pool insuficiente), registrar como zero
                results[name]['lift'].append(0)
                results[name]['roi'].append(0)
                results[name]['13pts'].append(0)
                results[name]['14pts'].append(0)

        start += step

    # Exibir resultados comparativos
    print("\n" + "="*70)
    print("📊 RESULTADOS DO WALK‑FORWARD (COMPARAÇÃO LIMPA)")
    print("="*70)
    print(f"{'Estratégia':<30} {'Lift Médio':<12} {'ROI Médio':<12} {'13pts Total':<12} {'14pts Total':<12}")
    print("-"*70)
    for name in strategies:
        lifts = results[name]['lift']
        rois = results[name]['roi']
        total_13 = sum(results[name]['13pts'])
        total_14 = sum(results[name]['14pts'])
        avg_lift = np.mean(lifts) if lifts else 0
        avg_roi = np.mean(rois) if rois else 0
        print(f"{name:<30} {avg_lift:<12.3f} {avg_roi:<12.1f}% {total_13:<12} {total_14:<12}")

    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v48")
    print("   COMPARAÇÃO LIMPA DE ESTRATÉGIAS PURAS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    print("\nIniciando walk‑forward comparativo...")
    print("   (Treino 500, Teste 50, Passo 50)")
    results = walk_forward_comparison(contests)

    print("\n✅ Walk‑forward concluído.")
    print("\n🔍 Interpretação:")
    print("   A estratégia com maior lift (acima de 1.0) seria a mais eficiente.")
    print("   Valores de lift abaixo de 1.0 indicam desempenho pior que o aleatório teórico.")
    print("   Observe também o total de 13 e 14 pontos para avaliar capacidade de acertos altos.")

if __name__ == "__main__":
    main()
