#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA TEMPORAL – Teste de existência de dependência temporal
Independente de arquitetura de IA.

Correções:
  - Exclui diagonal da matriz 25×25
  - Teste bilateral para autocorrelação (valores negativos também)
  - FDR sobre 250 testes individuais (25 dezenas × 10 lags)
  - Nomenclatura: autocorrelacao() para qualquer série
  - RNG mestre para placebos reproduzíveis
  - Mostra distribuição completa dos 600 pares off-diagonal

Pergunta: Existe informação no passado que distingue a sequência real
de uma sequência temporalmente embaralhada?
"""

import numpy as np
from collections import Counter
import os, time, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2,3,5,7,11,13,17,19,23}
MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
FIBONACCI = {1,2,3,5,8,13,21}

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
# FUNÇÕES AUXILIARES
# ============================================================
def matriz_presenca(contests):
    """Converte lista de concursos em matriz binária (n_concursos x 25)."""
    n = len(contests)
    X = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            X[i, d-1] = 1
    return X

def extrair_estrutura(dezenas):
    d = sorted(dezenas)
    return {
        'pares': sum(1 for x in d if x % 2 == 0),
        'primos': sum(1 for x in d if x in PRIMES),
        'moldura': sum(1 for x in d if x in MOLDURA),
        'fibonacci': sum(1 for x in d if x in FIBONACCI),
        'soma': sum(d),
        'amplitude': max(d) - min(d),
        'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    }

def autocorrelacao(serie, lag):
    """Autocorrelação de qualquer série (binária ou contínua)."""
    if len(serie) <= lag:
        return 0.0
    x = np.asarray(serie[:-lag], dtype=float)
    y = np.asarray(serie[lag:], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

def informacao_mutua_binaria(serie, lag):
    """Informação mútua entre série binária e ela mesma deslocada de lag."""
    if len(serie) <= lag:
        return 0.0
    x = serie[:-lag]
    y = serie[lag:]
    n00 = np.sum((x == 0) & (y == 0))
    n01 = np.sum((x == 0) & (y == 1))
    n10 = np.sum((x == 1) & (y == 0))
    n11 = np.sum((x == 1) & (y == 1))
    total = len(x)
    if total == 0:
        return 0.0
    p00, p01, p10, p11 = n00/total, n01/total, n10/total, n11/total
    px0 = (n00+n01)/total
    px1 = (n10+n11)/total
    py0 = (n00+n10)/total
    py1 = (n01+n11)/total
    mi = 0.0
    if p00 > 0: mi += p00 * np.log2(p00 / (px0*py0))
    if p01 > 0: mi += p01 * np.log2(p01 / (px0*py1))
    if p10 > 0: mi += p10 * np.log2(p10 / (px1*py0))
    if p11 > 0: mi += p11 * np.log2(p11 / (px1*py1))
    return mi

def dependencia_cruzada(X, i, j, lag=1):
    """Retorna P(i em t) e P(i em t | j em t-lag)."""
    n = X.shape[0]
    if n <= lag:
        return 0.0, 0.0
    cond = X[lag:, i]
    pred = X[:-lag, j]
    p_base = np.mean(X[:, i])
    if np.sum(pred) == 0:
        p_cond = p_base
    else:
        p_cond = np.mean(cond[pred == 1])
    return p_base, p_cond

def fdr_bh(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR."""
    p = np.array(p_values)
    n = len(p)
    if n == 0:
        return []
    ordem = np.argsort(p)
    p_sorted = p[ordem]
    q = np.ones(n)
    q[n-1] = p_sorted[n-1]
    for i in range(n-2, -1, -1):
        q[i] = min(p_sorted[i], q[i+1] * (n) / (i+1))
    rejeitados = []
    for i in range(n):
        if q[i] < alpha:
            rejeitados.append(ordem[i])
    return rejeitados

# ============================================================
# AUDITORIA PRINCIPAL
# ============================================================
def auditoria_temporal(contests, n_placebos=100, lag_max=10, alpha=0.05):
    print("\n🔍 AUDITORIA TEMPORAL")
    print(f"   Testando dependência temporal com {n_placebos} placebos")
    print(f"   Lags: 1..{lag_max}\n")

    X = matriz_presenca(contests)
    n_concursos = X.shape[0]

    # Estruturas dos concursos
    estruturas = [extrair_estrutura(c['dezenas']) for c in contests]

    # ---------- 1. Estatísticas reais ----------
    print("Computando estatísticas reais...")

    # Matrizes para armazenar p-values individuais
    p_autocorr = []  # (dezena, lag, p_value)
    p_mi = []

    # Autocorrelação e MI por dezena e lag
    for d in range(25):
        serie = X[:, d]
        for lag in range(1, lag_max+1):
            ac = autocorrelacao(serie, lag)
            mi = informacao_mutua_binaria(serie, lag)
            # Guardamos os valores reais para comparação posterior
            # (p-values serão calculados após gerar os placebos)
            p_autocorr.append((d, lag, ac))
            p_mi.append((d, lag, mi))

    # Dependência cruzada lag-1 (matriz 25×25)
    p_base_mat = np.zeros((25, 25))
    p_cond_mat = np.zeros((25, 25))
    for i in range(25):
        for j in range(25):
            p_base, p_cond = dependencia_cruzada(X, i, j, lag=1)
            p_base_mat[i, j] = p_base
            p_cond_mat[i, j] = p_cond

    # Dependência estrutural lag-1
    chaves_estruturais = ['pares', 'primos', 'moldura', 'fibonacci', 'soma', 'amplitude', 'consecutivos']
    corr_estruturais = {}
    for chave in chaves_estruturais:
        serie = np.array([e[chave] for e in estruturas], dtype=float)
        corr_estruturais[chave] = autocorrelacao(serie, 1)

    # ---------- 2. Distribuições nulas via placebo ----------
    print(f"\nGerando {n_placebos} placebos...")
    rng_master = np.random.default_rng(20260908)

    # Estruturas para armazenar placebos
    autocorr_placebo = {d: {lag: [] for lag in range(1, lag_max+1)} for d in range(25)}
    mi_placebo = {d: {lag: [] for lag in range(1, lag_max+1)} for d in range(25)}
    cruzada_placebo = []
    estrutural_placebo = {chave: [] for chave in chaves_estruturais}

    for _ in tqdm(range(n_placebos), desc="Placebos"):
        indices = rng_master.permutation(n_concursos)
        X_p = X[indices]
        estruturas_p = [estruturas[i] for i in indices]

        # Autocorrelação e MI por dezena e lag
        for d in range(25):
            serie = X_p[:, d]
            for lag in range(1, lag_max+1):
                ac = autocorrelacao(serie, lag)
                mi = informacao_mutua_binaria(serie, lag)
                autocorr_placebo[d][lag].append(ac)
                mi_placebo[d][lag].append(mi)

        # Dependência cruzada máxima off-diagonal lag-1
        p_base_p = np.zeros((25, 25))
        p_cond_p = np.zeros((25, 25))
        mask_offdiag = ~np.eye(25, dtype=bool)
        for i in range(25):
            for j in range(25):
                p_base, p_cond = dependencia_cruzada(X_p, i, j, lag=1)
                p_base_p[i, j] = p_base
                p_cond_p[i, j] = p_cond
        diff_p = np.abs(p_cond_p - p_base_p)
        cruzada_placebo.append(np.max(diff_p[mask_offdiag]))

        # Estrutural lag-1
        for chave in chaves_estruturais:
            serie = np.array([e[chave] for e in estruturas_p], dtype=float)
            corr_estruturais_p = autocorrelacao(serie, 1)
            estrutural_placebo[chave].append(corr_estruturais_p)

    # ---------- 3. Cálculo de p-values individuais e FDR ----------
    print("\nCalculando p-values e correção FDR...")

    # Autocorrelação (teste bilateral)
    pvals_autocorr = []
    for d, lag, real in p_autocorr:
        nula = np.array(autocorr_placebo[d][lag])
        p = np.mean(np.abs(nula) >= abs(real))
        pvals_autocorr.append(p)

    # Informação mútua (teste unilateral, MI ≥ 0)
    pvals_mi = []
    for d, lag, real in p_mi:
        nula = np.array(mi_placebo[d][lag])
        p = np.mean(nula >= real)
        pvals_mi.append(p)

    # FDR
    rejeitados_autocorr = fdr_bh(pvals_autocorr, alpha)
    rejeitados_mi = fdr_bh(pvals_mi, alpha)

    # ---------- 4. Exibição dos resultados ----------
    print("\n📊 RESULTADOS")
    print("="*70)

    # (a) Autocorrelação
    print("\nAUTOCORRELAÇÃO (teste bilateral, FDR sobre 250 testes)")
    print(f"   Total de testes: {len(pvals_autocorr)}")
    print(f"   Rejeitados após FDR: {len(rejeitados_autocorr)}")
    if rejeitados_autocorr:
        for idx in rejeitados_autocorr:
            d = idx // lag_max
            lag = idx % lag_max + 1
            real = p_autocorr[idx][2]
            nula = np.array(autocorr_placebo[d][lag])
            p = pvals_autocorr[idx]
            print(f"      dezena {d+1:2d}, lag {lag:2d}: real={real:+.4f}, p={p:.4f}")

    # (b) Informação mútua
    print("\nINFORMAÇÃO MÚTUA (teste unilateral, FDR sobre 250 testes)")
    print(f"   Total de testes: {len(pvals_mi)}")
    print(f"   Rejeitados após FDR: {len(rejeitados_mi)}")
    if rejeitados_mi:
        for idx in rejeitados_mi:
            d = idx // lag_max
            lag = idx % lag_max + 1
            real = p_mi[idx][2]
            nula = np.array(mi_placebo[d][lag])
            p = pvals_mi[idx]
            print(f"      dezena {d+1:2d}, lag {lag:2d}: real={real:.4f}, p={p:.4f}")

    # (c) Dependência cruzada máxima off-diagonal
    mask_offdiag = ~np.eye(25, dtype=bool)
    real_diff = np.abs(p_cond_mat - p_base_mat)
    real_max_cruzada = np.max(real_diff[mask_offdiag])
    nula_cruzada = np.array(cruzada_placebo)
    p_cruzada = np.mean(nula_cruzada >= real_max_cruzada)
    print(f"\nDEPENDÊNCIA CRUZADA MÁXIMA OFF-DIAGONAL (lag 1)")
    print(f"   real={real_max_cruzada:.4f}  placebo={np.mean(nula_cruzada):.4f}  p={p_cruzada:.4f}")

    # Distribuição das 600 relações
    print(f"\nDISTRIBUIÇÃO DAS 600 RELAÇÕES OFF-DIAGONAL")
    diffs = real_diff[mask_offdiag]
    print(f"   média={np.mean(diffs):.4f}, mediana={np.median(diffs):.4f}")
    print(f"   P5={np.percentile(diffs,5):.4f}, P95={np.percentile(diffs,95):.4f}")
    print(f"   máximo={np.max(diffs):.4f}")

    # (d) Dependência estrutural lag-1 (teste bilateral)
    print("\nDEPENDÊNCIA ESTRUTURAL (lag 1, teste bilateral)")
    for chave in chaves_estruturais:
        real = corr_estruturais[chave]
        nula = np.array(estrutural_placebo[chave])
        p_emp = np.mean(np.abs(nula) >= abs(real))
        print(f"   {chave:<15}: real={real:+.4f}  placebo={np.mean(nula):+.4f}  p={p_emp:.4f}")

    # ---------- 5. Resumo final ----------
    print("\n🔍 RESUMO FINAL")
    n_total_testes = len(pvals_autocorr) + len(pvals_mi) + 7 + 1  # +1 para matriz
    n_rejeitados = len(rejeitados_autocorr) + len(rejeitados_mi)
    print(f"   Total de testes independentes: {n_total_testes}")
    print(f"   Rejeitados após FDR: {n_rejeitados}")
    if n_rejeitados == 0:
        print("   ✅ Nenhuma evidência de dependência temporal significativa.")
    else:
        print("   ⚠️ Existem testes rejeitados; investigar.")

    return None

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔍 AUDITORIA TEMPORAL – Teste de dependência temporal")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria temporal completa")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            try:
                n_placebos = int(input("   Número de placebos [100]: ").strip() or "100")
                lag_max = int(input("   Lag máximo [10]: ").strip() or "10")
            except:
                n_placebos, lag_max = 100, 10
            auditoria_temporal(contests, n_placebos=n_placebos, lag_max=lag_max)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
