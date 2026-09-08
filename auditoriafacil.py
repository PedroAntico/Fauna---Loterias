#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA TEMPORAL v2.2 – Protocolo de três fases
Descoberta → Confirmação → Holdout Final

Métricas:
  - Autocorrelação (ACF) e Informação Mútua (MI) para cada dezena e lag (1..10)
  - p‑valores empíricos com correção +1
  - FDR (Benjamini‑Hochberg) conjunto
  - Δ = P(X_t=1 | X_{t-lag}=1) - P(X_t=1 | X_{t-lag}=0)
  - RR, OR, IC95% (bootstrap)
  - Teste direcional e bilateral no holdout
  - Estabilidade em 3 blocos temporais

Saída: candidatos selecionados na descoberta, desempenho na confirmação e holdout.
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

def bootstrap_delta(serie, lag, n_boot=2000, seed=123):
    """Calcula Δ = P(1|1) - P(1|0) com IC95% via bootstrap."""
    if len(serie) <= lag:
        return np.nan, np.nan, np.nan
    x = serie[:-lag]
    y = serie[lag:]
    mask1 = x == 1
    mask0 = x == 0
    y1 = y[mask1]
    y0 = y[mask0]
    if len(y1) == 0 or len(y0) == 0:
        return np.nan, np.nan, np.nan

    delta_obs = np.mean(y1) - np.mean(y0)
    rng = np.random.default_rng(seed)
    deltas_boot = []
    for _ in range(n_boot):
        b1 = rng.choice(y1, size=len(y1), replace=True)
        b0 = rng.choice(y0, size=len(y0), replace=True)
        deltas_boot.append(np.mean(b1) - np.mean(b0))
    ic_low = np.percentile(deltas_boot, 2.5)
    ic_high = np.percentile(deltas_boot, 97.5)
    return delta_obs, ic_low, ic_high

# ============================================================
# FUNÇÃO PRINCIPAL DA AUDITORIA
# ============================================================
def auditoria_temporal_v2(contests, frac_descoberta=0.6, frac_confirmacao=0.2,
                          n_placebos_descoberta=5000, n_placebos_confirmacao=5000,
                          n_placebos_holdout=5000, lag_max=10, alpha=0.05):
    print("\n🔍 AUDITORIA TEMPORAL v2.2 – Protocolo de três fases")
    print(f"   Divisão: {frac_descoberta:.0%} descoberta / {frac_confirmacao:.0%} confirmação / "
          f"{1-frac_descoberta-frac_confirmacao:.0%} holdout")
    print(f"   Placebos: descoberta={n_placebos_descoberta}, confirmação={n_placebos_confirmacao}, "
          f"holdout={n_placebos_holdout}")
    print(f"   Lags: 1..{lag_max}\n")

    n = len(contests)
    split_desc = int(n * frac_descoberta)
    split_conf = int(n * (frac_descoberta + frac_confirmacao))
    contests_desc = contests[:split_desc]
    contests_conf = contests[split_desc:split_conf]
    contests_holdout = contests[split_conf:]

    X_desc = matriz_presenca(contests_desc)
    X_conf = matriz_presenca(contests_conf)
    X_holdout = matriz_presenca(contests_holdout)

    rng_desc = np.random.default_rng(20260908)
    rng_conf = np.random.default_rng(20260909)
    rng_holdout = np.random.default_rng(20260910)

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
            p_acf[d, lag_idx] = (1 + np.sum(np.abs(nula_acf) >= abs(real_acf[d, lag_idx]))) / (len(nula_acf) + 1)
            nula_mi = np.array(mi_placebo[d][lag])
            p_mi[d, lag_idx] = (1 + np.sum(nula_mi >= real_mi[d, lag_idx])) / (len(nula_mi) + 1)

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

    if not sinais_selecionados:
        print("   Nenhum sinal selecionado.")
        return None

    # =====================================================
    # FASE DE CONFIRMAÇÃO
    # =====================================================
    print("\nFase de CONFIRMAÇÃO...")
    n_conf = X_conf.shape[0]
    resultados_conf = []

    for idx in sinais_selecionados:
        d = idx // lag_max
        lag = idx % lag_max + 1

        serie = X_conf[:, d]
        real_acf_conf = autocorrelacao(serie, lag)
        real_mi_conf = informacao_mutua_binaria(serie, lag)
        delta_conf, ic_low_conf, ic_high_conf = bootstrap_delta(serie, lag, n_boot=2000, seed=1000+idx)

        acf_conf_nula = []
        mi_conf_nula = []
        for _ in range(n_placebos_confirmacao):
            indices = rng_conf.permutation(n_conf)
            serie_p = serie[indices]
            acf_conf_nula.append(autocorrelacao(serie_p, lag))
            mi_conf_nula.append(informacao_mutua_binaria(serie_p, lag))
        acf_conf_nula = np.array(acf_conf_nula)
        mi_conf_nula = np.array(mi_conf_nula)
        p_acf_conf = (1 + np.sum(np.abs(acf_conf_nula) >= abs(real_acf_conf))) / (len(acf_conf_nula) + 1)
        p_mi_conf = (1 + np.sum(mi_conf_nula >= real_mi_conf)) / (len(mi_conf_nula) + 1)

        # Probabilidades condicionais
        x = serie[:-lag]
        y = serie[lag:]
        mask1 = x == 1
        p_base = np.mean(serie)
        p_cond1 = np.mean(y[mask1]) if np.sum(mask1) > 0 else p_base
        p_cond0 = np.mean(y[~mask1]) if np.sum(~mask1) > 0 else p_base
        diff_abs = p_cond1 - p_cond0
        rr = (p_cond1 / p_cond0) if p_cond0 > 0 else np.inf
        or_ = odds_ratio(p_cond1, p_cond0)

        # Persistência em 3 blocos
        blocos = np.array_split(serie, 3)
        acf_blocos = [autocorrelacao(bloco, lag) for bloco in blocos]
        sinal_real = 1 if real_acf_conf >= 0 else -1
        blocos_mesmo_sinal = sum(1 for a in acf_blocos if (a >= 0 and sinal_real == 1) or (a < 0 and sinal_real == -1))

        resultados_conf.append({
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
            'delta_conf': delta_conf,
            'ic_low_conf': ic_low_conf,
            'ic_high_conf': ic_high_conf,
            'p_base': p_base,
            'p_cond1': p_cond1,
            'p_cond0': p_cond0,
            'diff_abs': diff_abs,
            'rr': rr,
            'odds_ratio': or_,
            'blocos_mesmo_sinal': blocos_mesmo_sinal,
            'acf_blocos': acf_blocos
        })

    # FDR conjunto na confirmação
    p_conf_todos = []
    for s in resultados_conf:
        p_conf_todos.append(s['p_ACF_conf'])
        p_conf_todos.append(s['p_MI_conf'])
    rej_conf, q_conf_todos = fdr_bh(p_conf_todos, alpha)

    for i, s in enumerate(resultados_conf):
        s['q_ACF_conf'] = q_conf_todos[2*i]
        s['q_MI_conf'] = q_conf_todos[2*i+1]
        s['nominal'] = (s['p_ACF_conf'] < alpha or s['p_MI_conf'] < alpha)
        s['confirmado_fdr'] = (s['q_ACF_conf'] < alpha or s['q_MI_conf'] < alpha)

    # =====================================================
    # FASE DE HOLDOUT FINAL
    # =====================================================
    print("\nFase de HOLDOUT FINAL...")
    n_holdout = X_holdout.shape[0]
    resultados_holdout = []

    for s in resultados_conf:
        d = s['dezena'] - 1
        lag = s['lag']
        serie = X_holdout[:, d]
        real_acf_holdout = autocorrelacao(serie, lag)
        real_mi_holdout = informacao_mutua_binaria(serie, lag)
        delta_hold, ic_low_hold, ic_high_hold = bootstrap_delta(serie, lag, n_boot=2000, seed=2000+s['dezena']*100+lag)

        acf_holdout_nula = []
        mi_holdout_nula = []
        for _ in range(n_placebos_holdout):
            indices = rng_holdout.permutation(n_holdout)
            serie_p = serie[indices]
            acf_holdout_nula.append(autocorrelacao(serie_p, lag))
            mi_holdout_nula.append(informacao_mutua_binaria(serie_p, lag))
        acf_holdout_nula = np.array(acf_holdout_nula)
        mi_holdout_nula = np.array(mi_holdout_nula)

        # p bilateral
        p_acf_holdout_bilat = (1 + np.sum(np.abs(acf_holdout_nula) >= abs(real_acf_holdout))) / (len(acf_holdout_nula) + 1)
        p_mi_holdout = (1 + np.sum(mi_holdout_nula >= real_mi_holdout)) / (len(mi_holdout_nula) + 1)

        # p direcional conforme sinal descoberto
        sinal_desc = 1 if s['ACF_desc'] >= 0 else -1
        if sinal_desc >= 0:
            p_acf_holdout_dir = (1 + np.sum(acf_holdout_nula >= real_acf_holdout)) / (len(acf_holdout_nula) + 1)
        else:
            p_acf_holdout_dir = (1 + np.sum(acf_holdout_nula <= real_acf_holdout)) / (len(acf_holdout_nula) + 1)

        # Probabilidades condicionais
        x = serie[:-lag]
        y = serie[lag:]
        mask1 = x == 1
        p_base = np.mean(serie)
        p_cond1 = np.mean(y[mask1]) if np.sum(mask1) > 0 else p_base
        p_cond0 = np.mean(y[~mask1]) if np.sum(~mask1) > 0 else p_base
        diff_abs = p_cond1 - p_cond0
        rr = (p_cond1 / p_cond0) if p_cond0 > 0 else np.inf
        or_ = odds_ratio(p_cond1, p_cond0)

        resultados_holdout.append({
            'dezena': s['dezena'],
            'lag': s['lag'],
            'ACF_conf': s['ACF_conf'],
            'ACF_holdout': real_acf_holdout,
            'MI_holdout': real_mi_holdout,
            'p_ACF_holdout_bilat': p_acf_holdout_bilat,
            'p_ACF_holdout_dir': p_acf_holdout_dir,
            'p_MI_holdout': p_mi_holdout,
            'delta_hold': delta_hold,
            'ic_low_hold': ic_low_hold,
            'ic_high_hold': ic_high_hold,
            'p_base': p_base,
            'p_cond1': p_cond1,
            'p_cond0': p_cond0,
            'diff_abs': diff_abs,
            'rr': rr,
            'odds_ratio': or_,
        })

    # FDR conjunto no holdout (usando p bilateral para ACF e p unilateral para MI)
    p_holdout_todos = []
    for s in resultados_holdout:
        p_holdout_todos.append(s['p_ACF_holdout_bilat'])
        p_holdout_todos.append(s['p_MI_holdout'])
    rej_holdout, q_holdout_todos = fdr_bh(p_holdout_todos, alpha)

    for i, s in enumerate(resultados_holdout):
        s['q_ACF_holdout'] = q_holdout_todos[2*i]
        s['q_MI_holdout'] = q_holdout_todos[2*i+1]
        s['nominal_holdout'] = (s['p_ACF_holdout_bilat'] < alpha or s['p_MI_holdout'] < alpha)
        s['confirmado_holdout'] = (s['q_ACF_holdout'] < alpha or s['q_MI_holdout'] < alpha)

    # =====================================================
    # EXIBIÇÃO FINAL
    # =====================================================
    print("\n📊 RESULTADOS")
    print(f"{'Dez':<4} {'Lag':<4} {'ACF desc':<10} {'ACF conf':<10} {'ACF hold':<10} "
          f"{'p bilat':<8} {'p dir':<8} {'q hold':<8} {'Δ conf [IC95%]':<20} {'Δ hold [IC95%]':<20} "
          f"{'RR':<6} {'OR':<6}")
    print("-" * 130)
    for i, s in enumerate(resultados_conf):
        h = resultados_holdout[i]
        conf_ic = f"{s['delta_conf']:+.3f} [{s['ic_low_conf']:+.3f}, {s['ic_high_conf']:+.3f}]"
        hold_ic = f"{h['delta_hold']:+.3f} [{h['ic_low_hold']:+.3f}, {h['ic_high_hold']:+.3f}]"
        print(f"{s['dezena']:<4} {s['lag']:<4} {s['ACF_desc']:+.4f}     {s['ACF_conf']:+.4f}     {h['ACF_holdout']:+.4f}     "
              f"{h['p_ACF_holdout_bilat']:.4f}   {h['p_ACF_holdout_dir']:.4f}   {min(h['q_ACF_holdout'],h['q_MI_holdout']):.4f}   "
              f"{conf_ic:<20} {hold_ic:<20} {h['rr']:.3f} {h['odds_ratio']:.3f}")

    # Resumo
    n_total = len(resultados_conf)
    n_conf_nominal = sum(1 for s in resultados_conf if s['nominal'])
    n_conf_fdr = sum(1 for s in resultados_conf if s['confirmado_fdr'])
    n_holdout_nominal = sum(1 for s in resultados_holdout if s['nominal_holdout'])
    n_holdout_fdr = sum(1 for s in resultados_holdout if s['confirmado_holdout'])

    print(f"\n🔍 RESUMO")
    print(f"   Candidatos (descoberta): {n_total}")
    print(f"   Confirmação nominal: {n_conf_nominal}")
    print(f"   Confirmação FDR: {n_conf_fdr}")
    print(f"   Holdout nominal: {n_holdout_nominal}")
    print(f"   Holdout FDR: {n_holdout_fdr}")

    if n_holdout_fdr == 0:
        print("\n✅ Nenhum candidato apresentou dependência temporal estatisticamente significativa")
        print("   no holdout após FDR. Não há evidência de persistência temporal estável")
        print("   nos sinais previamente selecionados.")
    else:
        print("\n⚠️ Há candidato(s) com evidência no holdout; não concluir causalidade.")
        print("   Esses sinais merecem investigação com modelo específico.")

    return resultados_conf, resultados_holdout

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔍 AUDITORIA TEMPORAL v2.2 – Descoberta, Confirmação e Holdout")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria temporal v2.2")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            try:
                frac_desc = float(input("   Fração de descoberta [0.6]: ").strip() or "0.6")
                frac_conf = float(input("   Fração de confirmação [0.2]: ").strip() or "0.2")
                n_placebos_desc = int(input("   Placebos descoberta [5000]: ").strip() or "5000")
                n_placebos_conf = int(input("   Placebos confirmação [5000]: ").strip() or "5000")
                n_placebos_hold = int(input("   Placebos holdout [5000]: ").strip() or "5000")
                lag_max = int(input("   Lag máximo [10]: ").strip() or "10")
            except:
                frac_desc, frac_conf = 0.6, 0.2
                n_placebos_desc = n_placebos_conf = n_placebos_hold = 5000
                lag_max = 10
            auditoria_temporal_v2(contests, frac_descoberta=frac_desc, frac_confirmacao=frac_conf,
                                  n_placebos_descoberta=n_placebos_desc,
                                  n_placebos_confirmacao=n_placebos_conf,
                                  n_placebos_holdout=n_placebos_hold,
                                  lag_max=lag_max)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
