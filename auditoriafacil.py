#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA TEMPORAL v2 – Confirmação de sinais de dependência temporal

Correções:
  - FDR sobre os p-values da confirmação
  - Camadas de confirmação: nominal, FDR, consistência de sinal
  - Odds ratio adicionado
  - Saída como "sinais candidatos" e "desempenho na confirmação"
  - Separação descoberta/confirmação preservada
"""

import numpy as np
from collections import Counter
import os, time, warnings
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
# FUNÇÕES AUXILIARES
# ============================================================
def matriz_presenca(contests):
    n = len(contests)
    X = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            X[i, d-1] = 1
    return X

def autocorrelacao(serie, lag):
    if len(serie) <= lag:
        return 0.0
    x = np.asarray(serie[:-lag], dtype=float)
    y = np.asarray(serie[lag:], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

def informacao_mutua_binaria(serie, lag):
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

def fdr_bh(p_values, alpha=0.05):
    p = np.array(p_values)
    n = len(p)
    if n == 0:
        return [], []
    ordem = np.argsort(p)
    p_sorted = p[ordem]
    q = np.ones(n)
    q[n-1] = p_sorted[n-1]
    for i in range(n-2, -1, -1):
        q[i] = min(p_sorted[i], q[i+1] * (n) / (i+1))
    rejeitados = []
    q_vals = np.ones(n)
    for i in range(n):
        q_vals[ordem[i]] = q[i]
        if q[i] < alpha:
            rejeitados.append(ordem[i])
    return rejeitados, q_vals

def odds_ratio(p1, p0):
    if p1 >= 1.0 or p0 >= 1.0 or p1 <= 0.0 or p0 <= 0.0:
        return np.inf
    return (p1/(1-p1)) / (p0/(1-p0))

# ============================================================
# FUNÇÃO PRINCIPAL DA AUDITORIA
# ============================================================
def auditoria_temporal_v2(contests, frac_descoberta=0.6, n_placebos_descoberta=5000,
                          n_placebos_confirmacao=5000, lag_max=10, alpha=0.05):
    print("\n🔍 AUDITORIA TEMPORAL v2 – Confirmação de sinais")
    print(f"   Divisão: {frac_descoberta:.0%} descoberta / {1-frac_descoberta:.0%} confirmação")
    print(f"   Placebos (descoberta): {n_placebos_descoberta}")
    print(f"   Placebos (confirmação): {n_placebos_confirmacao}")
    print(f"   Lags: 1..{lag_max}\n")

    n = len(contests)
    split = int(n * frac_descoberta)
    contests_desc = contests[:split]
    contests_conf = contests[split:]

    X_desc = matriz_presenca(contests_desc)
    X_conf = matriz_presenca(contests_conf)

    rng_desc = np.random.default_rng(20260908)
    rng_conf = np.random.default_rng(20260909)

    # =====================================================
    # FASE DE DESCOBERTA
    # =====================================================
    print("Fase de DESCOBERTA...")
    n_desc = X_desc.shape[0]

    real_acf = np.zeros((25, lag_max))
    real_mi = np.zeros((25, lag_max))
    for d in range(25):
        serie = X_desc[:, d]
        for lag in range(1, lag_max+1):
            real_acf[d, lag-1] = autocorrelacao(serie, lag)
            real_mi[d, lag-1] = informacao_mutua_binaria(serie, lag)

    print(f"Gerando {n_placebos_descoberta} placebos (descoberta)...")
    acf_placebo = {d: {lag: [] for lag in range(1, lag_max+1)} for d in range(25)}
    mi_placebo = {d: {lag: [] for lag in range(1, lag_max+1)} for d in range(25)}

    for _ in tqdm(range(n_placebos_descoberta), desc="Placebos desc."):
        indices = rng_desc.permutation(n_desc)
        X_p = X_desc[indices]
        for d in range(25):
            serie = X_p[:, d]
            for lag in range(1, lag_max+1):
                ac = autocorrelacao(serie, lag)
                mi = informacao_mutua_binaria(serie, lag)
                acf_placebo[d][lag].append(ac)
                mi_placebo[d][lag].append(mi)

    p_acf = np.zeros((25, lag_max))
    p_mi = np.zeros((25, lag_max))
    for d in range(25):
        for lag_idx in range(lag_max):
            lag = lag_idx + 1
            nula_acf = np.array(acf_placebo[d][lag])
            p_acf[d, lag_idx] = np.mean(np.abs(nula_acf) >= abs(real_acf[d, lag_idx]))
            nula_mi = np.array(mi_placebo[d][lag])
            p_mi[d, lag_idx] = np.mean(nula_mi >= real_mi[d, lag_idx])

    p_acf_flat = p_acf.flatten()
    p_mi_flat = p_mi.flatten()
    rej_acf, q_acf = fdr_bh(p_acf_flat, alpha)
    rej_mi, q_mi = fdr_bh(p_mi_flat, alpha)

    sinais_selecionados = set(rej_acf) | set(rej_mi)
    sinais_selecionados = sorted(sinais_selecionados)
    print(f"\nSinais selecionados na descoberta: {len(sinais_selecionados)}")
    for idx in sinais_selecionados:
        d = idx // lag_max
        lag = idx % lag_max + 1
        qa = q_acf[idx]
        qm = q_mi[idx]
        print(f"   dezena {d+1:2d}, lag {lag:2d}: ACF={real_acf[d,lag-1]:+.4f} (q={qa:.4f}), "
              f"MI={real_mi[d,lag-1]:.4f} (q={qm:.4f})")

    # =====================================================
    # FASE DE CONFIRMAÇÃO
    # =====================================================
    print("\nFase de CONFIRMAÇÃO...")
    n_conf = X_conf.shape[0]
    confirmados = []

    for idx in sinais_selecionados:
        d = idx // lag_max
        lag = idx % lag_max + 1

        serie = X_conf[:, d]
        real_acf_conf = autocorrelacao(serie, lag)
        real_mi_conf = informacao_mutua_binaria(serie, lag)

        acf_conf_nula = []
        mi_conf_nula = []
        for _ in range(n_placebos_confirmacao):
            indices = rng_conf.permutation(n_conf)
            serie_p = serie[indices]
            acf_conf_nula.append(autocorrelacao(serie_p, lag))
            mi_conf_nula.append(informacao_mutua_binaria(serie_p, lag))
        acf_conf_nula = np.array(acf_conf_nula)
        mi_conf_nula = np.array(mi_conf_nula)
        p_acf_conf = np.mean(np.abs(acf_conf_nula) >= abs(real_acf_conf))
        p_mi_conf = np.mean(mi_conf_nula >= real_mi_conf)

        # Probabilidades condicionais
        x = serie[:-lag]
        y = serie[lag:]
        mask1 = x == 1
        p_base = np.mean(serie)
        if np.sum(mask1) > 0:
            p_cond1 = np.mean(y[mask1])
        else:
            p_cond1 = p_base
        if np.sum(~mask1) > 0:
            p_cond0 = np.mean(y[~mask1])
        else:
            p_cond0 = p_base
        diff_abs = p_cond1 - p_cond0
        rr = (p_cond1 / p_cond0) if p_cond0 > 0 else np.inf
        or_ = odds_ratio(p_cond1, p_cond0)

        # Persistência em 3 blocos
        blocos = np.array_split(serie, 3)
        acf_blocos = [autocorrelacao(bloco, lag) for bloco in blocos]
        sinal_real = 1 if real_acf_conf >= 0 else -1
        blocos_mesmo_sinal = sum(1 for a in acf_blocos if (a >= 0 and sinal_real == 1) or (a < 0 and sinal_real == -1))

        confirmados.append({
            'dezena': d+1,
            'lag': lag,
            'ACF_desc': real_acf[d, lag-1],
            'MI_desc': real_mi[d, lag-1],
            'q_ACF_desc': q_acf[idx],
            'q_MI_desc': q_mi[idx],
            'ACF_conf': real_acf_conf,
            'MI_conf': real_mi_conf,
            'p_ACF_conf': p_acf_conf,
            'p_MI_conf': p_mi_conf,
            'p_base': p_base,
            'p_cond1': p_cond1,
            'p_cond0': p_cond0,
            'diff_abs': diff_abs,
            'rr': rr,
            'odds_ratio': or_,
            'blocos_mesmo_sinal': blocos_mesmo_sinal,
            'acf_blocos': acf_blocos
        })

    # Aplicar FDR na confirmação
    p_conf_acf = [s['p_ACF_conf'] for s in confirmados]
    p_conf_mi = [s['p_MI_conf'] for s in confirmados]
    rej_conf_acf, q_conf_acf = fdr_bh(p_conf_acf, alpha)
    rej_conf_mi, q_conf_mi = fdr_bh(p_conf_mi, alpha)

    for i, s in enumerate(confirmados):
        s['q_ACF_conf'] = q_conf_acf[i]
        s['q_MI_conf'] = q_conf_mi[i]
        s['nominal'] = s['p_ACF_conf'] < 0.05 or s['p_MI_conf'] < 0.05
        s['confirmado_fdr'] = s['q_ACF_conf'] < 0.05 or s['q_MI_conf'] < 0.05

    # =====================================================
    # EXIBIÇÃO FINAL
    # =====================================================
    print("\n📊 DESEMPENHO NA CONFIRMAÇÃO")
    print(f"{'Dez':<4} {'Lag':<4} {'ACF desc':<10} {'ACF conf':<10} {'p_ACF':<8} {'q_ACF':<8} "
          f"{'MI conf':<10} {'p_MI':<8} {'q_MI':<8} {'P(1|1)':<8} {'P(1|0)':<8} {'Δ':<8} {'RR':<8} {'OR':<8} {'Blocos':<8}")
    print("-" * 130)
    for s in confirmados:
        print(f"{s['dezena']:<4} {s['lag']:<4} {s['ACF_desc']:+.4f}     {s['ACF_conf']:+.4f}     "
              f"{s['p_ACF_conf']:.4f}   {s['q_ACF_conf']:.4f}   {s['MI_conf']:.4f}   {s['p_MI_conf']:.4f}   {s['q_MI_conf']:.4f}   "
              f"{s['p_cond1']:.4f} {s['p_cond0']:.4f} {s['diff_abs']:+.4f} {s['rr']:.3f} {s['odds_ratio']:.3f} "
              f"{s['blocos_mesmo_sinal']}/3")

    # Resumo em camadas
    n_total = len(confirmados)
    n_nominal = sum(1 for s in confirmados if s['nominal'])
    n_fdr = sum(1 for s in confirmados if s['confirmado_fdr'])
    n_blocos2 = sum(1 for s in confirmados if s['blocos_mesmo_sinal'] >= 2)
    n_blocos3 = sum(1 for s in confirmados if s['blocos_mesmo_sinal'] == 3)

    print(f"\n🔍 RESUMO")
    print(f"   Sinais candidatos (descoberta): {n_total}")
    print(f"   Confirmação nominal (p<0,05): {n_nominal}")
    print(f"   Confirmação FDR (q<0,05): {n_fdr}")
    print(f"   Mesmo sinal em ≥2/3 blocos: {n_blocos2}")
    print(f"   Mesmo sinal em 3/3 blocos: {n_blocos3}")

    if n_fdr == 0:
        print("   ✅ Nenhum sinal sobreviveu à confirmação com correção múltipla.")
    else:
        print("   ⚠️ Existem sinais com confirmação FDR; investigar em bloco OOS separado.")

    return confirmados

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔍 AUDITORIA TEMPORAL v2 – Confirmação de sinais")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria temporal v2 (descoberta + confirmação)")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            try:
                frac_desc = float(input("   Fração de descoberta [0.6]: ").strip() or "0.6")
                n_placebos_desc = int(input("   Placebos descoberta [5000]: ").strip() or "5000")
                n_placebos_conf = int(input("   Placebos confirmação [5000]: ").strip() or "5000")
                lag_max = int(input("   Lag máximo [10]: ").strip() or "10")
            except:
                frac_desc, n_placebos_desc, n_placebos_conf, lag_max = 0.6, 5000, 5000, 10
            auditoria_temporal_v2(contests, frac_descoberta=frac_desc,
                                  n_placebos_descoberta=n_placebos_desc,
                                  n_placebos_confirmacao=n_placebos_conf,
                                  lag_max=lag_max)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
