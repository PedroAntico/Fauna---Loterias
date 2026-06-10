#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v61
INFORMAÇÃO MÚTUA AVANÇADA: CONCURSOS, DEZENAS E ESTRUTURAS

MELHORIAS:
✅ Permutações aumentadas para 2000 (significância mais precisa)
✅ Lags estendidos: 1, 2, 3, 5, 10, 20, 50
✅ MI para dezenas individuais (25 séries binárias)
✅ MI para o concurso completo (vetor de 25 bits)
✅ Correção de viés por permutação em todos os casos
"""

import numpy as np
from scipy.stats import entropy
from collections import Counter
from itertools import product
import os, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
MOLDURA_SET = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
PRIMES_SET = {2, 3, 5, 7, 11, 13, 17, 19, 23}

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
# EXTRAÇÃO DE DADOS
# ============================================================
def extrair_series_estruturais(contests):
    """Séries de características agregadas (pares, moldura, etc.)."""
    series = {
        'pares': [],
        'moldura': [],
        'primos': [],
        'soma': [],
        'consecutivos': [],
        'amplitude': []
    }
    for c in contests:
        d = c['dezenas']
        series['pares'].append(sum(1 for x in d if x % 2 == 0))
        series['moldura'].append(sum(1 for x in d if x in MOLDURA_SET))
        series['primos'].append(sum(1 for x in d if x in PRIMES_SET))
        series['soma'].append(sum(d))
        series['consecutivos'].append(sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1))
        series['amplitude'].append(max(d) - min(d))
    return {k: np.array(v, dtype=float) for k, v in series.items()}

def extrair_series_dezenas(contests):
    """25 séries binárias (presença/ausência de cada dezena)."""
    n = len(contests)
    dezenas = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            dezenas[i, d-1] = 1
    return dezenas

# ============================================================
# INFORMAÇÃO MÚTUA (COM CORREÇÃO POR PERMUTAÇÃO)
# ============================================================
def mutual_information(x, y, bins=None):
    """MI entre duas séries. bins=None para dados discretos."""
    if bins is not None:
        combined = np.concatenate([x, y])
        bin_edges = np.percentile(combined, np.linspace(0, 100, bins+1))
        x_disc = np.digitize(x, bin_edges[1:-1])
        y_disc = np.digitize(y, bin_edges[1:-1])
    else:
        x_disc = x.astype(int)
        y_disc = y.astype(int)
    joint = Counter(zip(x_disc, y_disc))
    total = len(x_disc)
    mi = 0.0
    for (xi, yi), cnt in joint.items():
        p_xy = cnt / total
        p_x = np.mean(x_disc == xi)
        p_y = np.mean(y_disc == yi)
        if p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi

def mi_significance(series, lag, n_perm=2000, bins=None):
    """MI observada, corrigida, p-valor e média nula."""
    if len(series) <= lag:
        return 0, 0, 1.0, 0
    x = series[:-lag]
    y = series[lag:]
    mi_obs = mutual_information(x, y, bins=bins)
    mi_null = np.zeros(n_perm)
    y_shuffled = y.copy()
    for i in range(n_perm):
        np.random.shuffle(y_shuffled)
        mi_null[i] = mutual_information(x, y_shuffled, bins=bins)
    mean_null = np.mean(mi_null)
    mi_corr = mi_obs - mean_null
    p_val = np.mean(mi_null >= mi_obs)
    return mi_obs, mi_corr, p_val, mean_null

# ============================================================
# ANÁLISE COMPLETA
# ============================================================
def analisar_mi_estrutural(series_dict, lags):
    """MI para características agregadas."""
    bins_cfg = {
        'pares': None, 'moldura': None, 'primos': None,
        'soma': 10, 'consecutivos': None, 'amplitude': 10
    }
    print("\n📊 INFORMAÇÃO MÚTUA – CARACTERÍSTICAS ESTRUTURAIS")
    print(f"   Lags: {lags} | Permutações: 2000\n")
    resultados = {}
    for nome, serie in series_dict.items():
        bins = bins_cfg.get(nome, 10)
        print(f"--- {nome} ---")
        print(f"   {'Lag':<8} {'MI obs':<10} {'MI corr':<10} {'p‑valor':<10}")
        for lag in lags:
            mi_obs, mi_corr, p_val, _ = mi_significance(serie, lag, bins=bins)
            print(f"   {lag:<8} {mi_obs:<10.4f} {mi_corr:<10.4f} {p_val:<10.4f}")
        resultados[nome] = {lag: {'mi_corr': mi_corr, 'p_val': p_val} for lag in lags}
    return resultados

def analisar_mi_dezenas(dezenas, lags):
    """MI para cada uma das 25 dezenas (séries binárias)."""
    print("\n📊 INFORMAÇÃO MÚTUA – DEZENAS INDIVIDUAIS")
    print(f"   Lags: {lags} | Permutações: 2000\n")
    resultados = {}
    for d in range(25):
        serie = dezenas[:, d]
        print(f"--- Dezena {d+1:2d} ---")
        print(f"   {'Lag':<8} {'MI obs':<10} {'MI corr':<10} {'p‑valor':<10}")
        for lag in lags:
            mi_obs, mi_corr, p_val, _ = mi_significance(serie, lag, bins=None)
            print(f"   {lag:<8} {mi_obs:<10.4f} {mi_corr:<10.4f} {p_val:<10.4f}")
        resultados[d+1] = {lag: {'mi_corr': mi_corr, 'p_val': p_val} for lag in lags}
    return resultados

def analisar_mi_concurso(contests, lags):
    """MI entre concursos completos (vetor de 25 bits)."""
    # Representação como máscara de bits (número inteiro)
    masks = np.array([BITMASK_CACHE.get_mask(c['dezenas']) for c in contests], dtype=np.uint32)
    print("\n📊 INFORMAÇÃO MÚTUA – CONCURSO COMPLETO (25 bits)")
    print(f"   Lags: {lags} | Permutações: 2000\n")
    # Discretização: usamos os próprios inteiros (já são discretos)
    print(f"   {'Lag':<8} {'MI obs':<10} {'MI corr':<10} {'p‑valor':<10}")
    for lag in lags:
        mi_obs, mi_corr, p_val, _ = mi_significance(masks, lag, bins=None)
        print(f"   {lag:<8} {mi_obs:<10.4f} {mi_corr:<10.4f} {p_val:<10.4f}")

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

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v61")
    print("   INFORMAÇÃO MÚTUA AVANÇADA")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # Parâmetros
    lags = [1, 2, 3, 5, 10, 20, 50]

    # 1. Características estruturais
    series_est = extrair_series_estruturais(contests)
    resultados_est = analisar_mi_estrutural(series_est, lags)

    # 2. Dezenas individuais
    dezenas = extrair_series_dezenas(contests)
    resultados_dez = analisar_mi_dezenas(dezenas, lags)

    # 3. Concurso completo
    analisar_mi_concurso(contests, lags)

    # Resumo
    print("\n" + "="*70)
    print("📊 RESUMO GERAL")
    print("="*70)
    sinais = []
    # Verificar características
    for nome, res in resultados_est.items():
        for lag, val in res.items():
            if val['p_val'] < 0.05 and val['mi_corr'] > 0.001:
                sinais.append((f"Estrutura {nome}", lag, val['mi_corr'], val['p_val']))
    # Verificar dezenas
    for dezena, res in resultados_dez.items():
        for lag, val in res.items():
            if val['p_val'] < 0.05 and val['mi_corr'] > 0.001:
                sinais.append((f"Dezena {dezena}", lag, val['mi_corr'], val['p_val']))
    if sinais:
        print("   🔍 Possíveis dependências detectadas:")
        for s in sinais[:10]:
            print(f"      {s[0]} lag {s[1]}: MI corr={s[2]:.4f}, p={s[3]:.4f}")
    else:
        print("   ✅ Nenhuma dependência temporal significativa encontrada.")
        print("   As características e dezenas são compatíveis com independência (i.i.d.).")
        print("   I(X_t ; X_{t+lag}) ≈ 0 para todos os lags testados.")

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
