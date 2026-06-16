#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v63
ESTUDO DE DEPENDÊNCIA TEMPORAL CONDICIONAL DAS 25 DEZENAS

OBJETIVO:
✅ Para cada dezena e cada lag (1,2,3,5), calcular:
   P(d em t+lag | d em t) e P(d em t+lag | d ausente em t)
✅ Testar significância da diferença via permutação (2000 simulações)
✅ Aplicar correção FDR (Benjamini‑Hochberg) para múltiplos testes
✅ Ranquear os sinais mais promissores
✅ Investigar se a dezena 25 (lag=2) sobrevive à correção
"""

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import os, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

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
# EXTRAÇÃO DAS 25 SÉRIES BINÁRIAS
# ============================================================
def extrair_series_dezenas(contests):
    n = len(contests)
    dezenas = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            dezenas[i, d-1] = 1
    return dezenas

# ============================================================
# TESTE DE DEPENDÊNCIA CONDICIONAL PARA UMA DEZENA
# ============================================================
def teste_dependencia_condicional(serie, lag, n_perm=2000):
    """
    Calcula:
    - P(d em t+lag | d em t)
    - P(d em t+lag | d ausente em t)
    - Diferença e significância via permutação
    """
    n = len(serie)
    if n <= lag:
        return None
    
    # Índices onde a dezena estava presente em t
    idx_presente = np.where(serie[:n-lag] == 1)[0]
    idx_ausente  = np.where(serie[:n-lag] == 0)[0]
    
    if len(idx_presente) < 10 or len(idx_ausente) < 10:
        return None
    
    prob_dado_presente = np.mean(serie[idx_presente + lag])
    prob_dado_ausente  = np.mean(serie[idx_ausente + lag])
    prob_geral = np.mean(serie)
    dif_obs = prob_dado_presente - prob_dado_ausente
    
    # Permutação
    difs_perm = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = serie.copy()
        np.random.shuffle(shuffled)
        p_pres = np.mean(shuffled[idx_presente + lag])
        p_aus  = np.mean(shuffled[idx_ausente + lag])
        difs_perm[i] = p_pres - p_aus
    
    # p-valor bicaudal
    p_val = np.mean(np.abs(difs_perm) >= np.abs(dif_obs))
    
    return {
        'dezena': None,  # será preenchido depois
        'lag': lag,
        'prob_presente': prob_dado_presente,
        'prob_ausente': prob_dado_ausente,
        'prob_geral': prob_geral,
        'dif': dif_obs,
        'p_val': p_val,
        'n_presente': len(idx_presente),
        'n_ausente': len(idx_ausente)
    }

# ============================================================
# ANÁLISE COMPLETA PARA TODAS AS DEZENAS E LAGS
# ============================================================
def analisar_todas_dezenas(dezenas, lags=[1, 2, 3, 5], n_perm=2000):
    """
    Para cada dezena (1..25) e cada lag, executa o teste de dependência condicional.
    Aplica correção FDR ao final.
    """
    resultados = []
    
    print(f"\n🔬 TESTANDO DEPENDÊNCIA CONDICIONAL PARA AS 25 DEZENAS")
    print(f"   Lags: {lags} | Permutações por teste: {n_perm}\n")
    
    for d in tqdm(range(25), desc="Analisando dezenas"):
        serie = dezenas[:, d]
        for lag in lags:
            res = teste_dependencia_condicional(serie, lag, n_perm)
            if res is not None:
                res['dezena'] = d + 1
                resultados.append(res)
    
    # Aplicar correção FDR
    pvals = [r['p_val'] for r in resultados]
    _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
    for i, res in enumerate(resultados):
        res['p_fdr'] = pvals_fdr[i]
    
    # Ordenar por significância (menor p_val primeiro)
    resultados.sort(key=lambda x: x['p_val'])
    
    return resultados

# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================
def exibir_resultados(resultados):
    print(f"\n{'='*90}")
    print(f"🏆 RANKING DE DEPENDÊNCIA TEMPORAL CONDICIONAL (Top 30)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Dezena':<8} {'Lag':<6} {'P(t+lag|presente)':<18} {'P(t+lag|ausente)':<18} {'Dif':<10} {'p‑raw':<10} {'p‑FDR':<10} {'Status'}")
    print(f"{'='*90}")
    
    for i, res in enumerate(resultados[:30], 1):
        status = ""
        if res['p_fdr'] < 0.05:
            status = "🔍 SIGNIFICATIVO"
        elif res['p_val'] < 0.05:
            status = "⚠️  (não sobrevive ao FDR)"
        else:
            status = ""
        
        print(f"{i:<5} {res['dezena']:<8} {res['lag']:<6} "
              f"{res['prob_presente']:<18.4f} {res['prob_ausente']:<18.4f} "
              f"{res['dif']*100:<10.2f}% {res['p_val']:<10.4f} {res['p_fdr']:<10.4f} {status}")
    
    # Destacar os sobreviventes ao FDR
    sobreviventes = [r for r in resultados if r['p_fdr'] < 0.05]
    print(f"\n📊 SOBREVIVENTES À CORREÇÃO FDR (p < 0.05): {len(sobreviventes)}")
    if sobreviventes:
        for res in sobreviventes:
            print(f"   Dezena {res['dezena']:2d} lag {res['lag']}: "
                  f"dif = {res['dif']*100:+.2f}% (p_raw={res['p_val']:.4f}, p_fdr={res['p_fdr']:.4f})")
    else:
        print("   Nenhum sinal sobreviveu à correção para múltiplos testes.")
        print("   Os sinais observados são compatíveis com flutuações aleatórias.")
    
    # Resumo específico para a dezena 25
    res25 = [r for r in resultados if r['dezena'] == 25 and r['lag'] == 2]
    if res25:
        r = res25[0]
        print(f"\n📊 DETALHE DA DEZENA 25 (lag 2):")
        print(f"   P(25 em t+2 | 25 em t)    = {r['prob_presente']:.4f} (n={r['n_presente']})")
        print(f"   P(25 em t+2 | 25 ausente)  = {r['prob_ausente']:.4f} (n={r['n_ausente']})")
        print(f"   Diferença = {r['dif']*100:+.2f}%")
        print(f"   p_raw = {r['p_val']:.4f}  |  p_fdr = {r['p_fdr']:.4f}")
        if r['p_fdr'] < 0.05:
            print(f"   🔍 A dezena 25 (lag 2) SOBREVIVE à correção FDR.")
        else:
            print(f"   ⚠️  A dezena 25 (lag 2) NÃO sobrevive à correção FDR.")
            print(f"       O sinal é promissor, mas pode ser falso positivo.")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v63")
    print("   DEPENDÊNCIA TEMPORAL CONDICIONAL DAS 25 DEZENAS")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    
    # Extrair séries
    dezenas = extrair_series_dezenas(contests)
    
    # Parâmetros
    lags = [1, 2, 3, 5]
    n_perm = 2000
    
    # Executar análise
    resultados = analisar_todas_dezenas(dezenas, lags, n_perm)
    
    # Exibir
    exibir_resultados(resultados)
    
    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
