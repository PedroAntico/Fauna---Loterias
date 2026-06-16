#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v62
TESTE DE SINAL INVERTIDO (ANTI‑PREDITIVO) + ESTUDO DA DEZENA 25

HIPÓTESES:
✅ O modelo erra de forma sistemática? (edge invertido)
✅ Comparação das taxas de acerto normal e invertida contra baseline aleatório
✅ Estudo específico da dezena 25: P(25 em t+2 | 25 em t) vs P(25 em t+2 | 25 ∉ t)
✅ Walk‑forward honesto com correção por permutação
"""

import numpy as np
from scipy.stats import binomtest
from collections import Counter
import os, random, warnings
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
# EXTRAÇÃO DE SÉRIES
# ============================================================
def extrair_series_estruturais(contests):
    series = {
        'pares': [], 'moldura': [], 'primos': [],
        'soma': [], 'consecutivos': [], 'amplitude': []
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
    n = len(contests)
    dezenas = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            dezenas[i, d-1] = 1
    return dezenas

# ============================================================
# PREDITOR SIMPLES (REVERSÃO À MÉDIA)
# ============================================================
def prever_direcao(serie, idx, janela=20):
    """Prevê se o próximo valor será maior (+1) ou menor (-1) que o atual."""
    if idx < janela:
        return 0  # sem previsão
    media_recente = np.mean(serie[idx-janela:idx])
    media_historica = np.mean(serie[:idx])
    if media_recente > media_historica:
        return -1  # prevê queda
    else:
        return 1   # prevê alta

def direcao_real(serie, idx):
    """Retorna +1 se o valor subiu em relação ao anterior, -1 se caiu, 0 se igual."""
    if idx < 1:
        return 0
    if serie[idx] > serie[idx-1]:
        return 1
    elif serie[idx] < serie[idx-1]:
        return -1
    else:
        return 0

# ============================================================
# ANÁLISE DE ACERTO NORMAL E INVERTIDO
# ============================================================
def analisar_edge(serie, janela=20, n_perm=500):
    """
    Para cada ponto a partir de janela+1, faz uma previsão de direção.
    Calcula a taxa de acerto normal e a taxa de acerto invertida.
    Compara ambas com um baseline aleatório (permutação da série).
    """
    n = len(serie)
    previsoes = []
    reais = []
    
    for i in range(janela + 1, n):
        pred = prever_direcao(serie, i-1, janela)  # prevê direção de i-1 → i
        real = direcao_real(serie, i)
        if pred != 0 and real != 0:
            previsoes.append(pred)
            reais.append(real)
    
    previsoes = np.array(previsoes)
    reais = np.array(reais)
    
    if len(previsoes) == 0:
        return {'acc_normal': 0.5, 'acc_invertido': 0.5, 'p_normal': 1.0, 'p_invertido': 1.0, 'n': 0}
    
    # Acerto normal
    acertos_normal = np.sum(previsoes == reais)
    acc_normal = acertos_normal / len(previsoes)
    
    # Acerto invertido (se tivéssemos usado a previsão oposta)
    acertos_invertido = np.sum(previsoes != reais)
    acc_invertido = acertos_invertido / len(previsoes)
    
    # Baseline por permutação
    accs_perm = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(serie)
        prev_perm = []
        real_perm = []
        for i in range(janela + 1, n):
            pred = prever_direcao(shuffled, i-1, janela)
            real = direcao_real(shuffled, i)
            if pred != 0 and real != 0:
                prev_perm.append(pred)
                real_perm.append(real)
        if len(prev_perm) > 0:
            prev_perm = np.array(prev_perm)
            real_perm = np.array(real_perm)
            acc_perm = np.mean(prev_perm == real_perm)
            accs_perm.append(acc_perm)
    
    accs_perm = np.array(accs_perm)
    mean_perm = np.mean(accs_perm)
    std_perm = np.std(accs_perm)
    
    # p-valor para acerto normal (unicaudal: acc_normal >= acc_perm)
    p_normal = np.mean(accs_perm >= acc_normal)
    # p-valor para acerto invertido (acc_invertido >= 1 - acc_perm)
    p_invertido = np.mean(accs_perm >= acc_invertido)
    
    return {
        'acc_normal': acc_normal,
        'acc_invertido': acc_invertido,
        'p_normal': p_normal,
        'p_invertido': p_invertido,
        'baseline_mean': mean_perm,
        'baseline_std': std_perm,
        'n': len(previsoes)
    }

# ============================================================
# ESTUDO ESPECÍFICO DA DEZENA 25
# ============================================================
def estudo_dezena_25(dezenas, lag=2, n_perm=2000):
    """P(25 em t+lag | 25 em t) vs P(25 em t+lag | 25 não em t)."""
    serie = dezenas[:, 24]  # índice 24 = dezena 25
    n = len(serie)
    
    # Casos onde 25 apareceu em t
    idx_presente = np.where(serie[:n-lag] == 1)[0]
    # Casos onde 25 NÃO apareceu em t
    idx_ausente = np.where(serie[:n-lag] == 0)[0]
    
    if len(idx_presente) == 0 or len(idx_ausente) == 0:
        return None
    
    prob_dado_presente = np.mean(serie[idx_presente + lag])
    prob_dado_ausente = np.mean(serie[idx_ausente + lag])
    prob_geral = np.mean(serie)
    
    # Teste de permutação: diferença entre as duas condicionais
    dif_obs = prob_dado_presente - prob_dado_ausente
    
    difs_perm = []
    for _ in range(n_perm):
        shuffled = serie.copy()
        np.random.shuffle(shuffled)
        p_pres = np.mean(shuffled[idx_presente + lag]) if len(idx_presente) > 0 else 0
        p_aus = np.mean(shuffled[idx_ausente + lag]) if len(idx_ausente) > 0 else 0
        difs_perm.append(p_pres - p_aus)
    difs_perm = np.array(difs_perm)
    
    p_val = np.mean(np.abs(difs_perm) >= np.abs(dif_obs))
    
    return {
        'P(t+lag | presente)': prob_dado_presente,
        'P(t+lag | ausente)': prob_dado_ausente,
        'P(geral)': prob_geral,
        'diferença': dif_obs,
        'p_valor': p_val,
        'n_presente': len(idx_presente),
        'n_ausente': len(idx_ausente)
    }

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v62")
    print("   TESTE DE SINAL INVERTIDO + ESTUDO DA DEZENA 25")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # 1. Análise de características estruturais
    print("\n📊 ANÁLISE DE EDGE NORMAL / INVERTIDO (CARACTERÍSTICAS ESTRUTURAIS)")
    series_est = extrair_series_estruturais(contests)
    
    for nome, serie in series_est.items():
        res = analisar_edge(serie)
        print(f"\n--- {nome} ---")
        print(f"   Previsões: {res['n']}")
        print(f"   Acerto normal:    {res['acc_normal']*100:.1f}% (p={res['p_normal']:.4f})")
        print(f"   Acerto invertido: {res['acc_invertido']*100:.1f}% (p={res['p_invertido']:.4f})")
        print(f"   Baseline (perm):  média={res['baseline_mean']*100:.1f}%")
        if res['p_normal'] < 0.05:
            print(f"   🔍 Edge NORMAL significativo!")
        if res['p_invertido'] < 0.05:
            print(f"   🔍 Edge INVERTIDO significativo!")
    
    # 2. Análise das 25 dezenas
    print("\n📊 ANÁLISE DE EDGE NORMAL / INVERTIDO (DEZENAS INDIVIDUAIS)")
    dezenas = extrair_series_dezenas(contests)
    for d in range(25):
        serie = dezenas[:, d]
        res = analisar_edge(serie)
        # Só exibe se houver algum sinal
        if res['p_normal'] < 0.05 or res['p_invertido'] < 0.05:
            print(f"   Dezena {d+1:2d}: normal={res['acc_normal']*100:.1f}% invertido={res['acc_invertido']*100:.1f}%")
    
    # 3. Estudo específico da dezena 25
    print("\n📊 ESTUDO DA DEZENA 25 (lag=2)")
    estudo = estudo_dezena_25(dezenas, lag=2)
    if estudo:
        print(f"   P(25 em t+2 | 25 em t)    = {estudo['P(t+lag | presente)']:.4f}")
        print(f"   P(25 em t+2 | 25 não em t) = {estudo['P(t+lag | ausente)']:.4f}")
        print(f"   P(25 geral)                = {estudo['P(geral)']:.4f}")
        print(f"   Diferença = {estudo['diferença']:.4f}")
        print(f"   p‑valor (permutação) = {estudo['p_valor']:.4f}")
        if estudo['p_valor'] < 0.05:
            print(f"   🔍 Diferença significativa! A dezena 25 pode ter dependência temporal.")
        else:
            print(f"   📊 Diferença não significativa. Comportamento compatível com independência.")
    
    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
