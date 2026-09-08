#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA SOMA → DEZENAS v1.2 (corrigida e completa)

Correções:
- Alinhamento temporal: soma do concurso t prediz dezenas do concurso t+1
- AUC calculada com a soma contínua (não discretizada)
- Limiares de discretização e Δ fixados na descoberta
- Contagem de observações por bin para diagnóstico
- Análise Soma → Próxima Soma agora é executada mesmo sem sinais na descoberta
- Análise Soma → Próxima Soma com protocolo 60/20/20, Pearson, Spearman, MI e placebos
- Placebo com permutação (teste de independência global)

Métricas:
  - MI entre soma discretizada (5 quantis) e indicador da dezena no próximo concurso
  - AUC (soma contínua) para prever a dezena
  - Δ = P(dezena | soma ≥ percentil 70) - P(dezena | soma ≤ percentil 30)
  - RR, OR correspondentes (não exibidos aqui, mas disponíveis para extensão)

Protocolo:
  60% descoberta → FDR/BH → 20% confirmação → FDR/BH → 20% holdout final.
Placebo: permutação da ordem das somas (destrói relação soma→próximo).
"""

import numpy as np
from scipy.stats import spearmanr
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
def soma_concurso(dezenas):
    return sum(dezenas)

def matriz_presenca(contests):
    n = len(contests)
    X = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            X[i, d-1] = 1
    return X

def informacao_mutua_discreta(x, y, n_bins_x=5):
    """MI entre x discreto (já discretizado) e y binário."""
    if len(x) < 10:
        return 0.0
    cont = Counter(zip(x, y))
    total = len(x)
    px = Counter(x)
    py = Counter(y)
    mi = 0.0
    for (xi, yi), count in cont.items():
        pxy = count / total
        px_ = px[xi] / total
        py_ = py[yi] / total
        if pxy > 0 and px_ > 0 and py_ > 0:
            mi += pxy * np.log2(pxy / (px_ * py_))
    return mi

def auc_roc(y_true, y_score):
    """AUC ROC com ranks médios (ordenação crescente)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return 0.5
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    ranks = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores_sorted[j] == scores_sorted[i]:
            j += 1
        ranks[i:j] = np.mean(ranks[i:j])
        i = j
    n_pos = np.sum(y_sorted)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos_ranks = np.sum(ranks[y_sorted == 1])
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

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
# FUNÇÃO PRINCIPAL
# ============================================================
def auditoria_soma_dezenas(contests, frac_descoberta=0.6, frac_confirmacao=0.2,
                           n_placebos_descoberta=5000, n_placebos_confirmacao=5000,
                           n_placebos_holdout=5000, alpha=0.05, n_bins=5):
    print("\n🔍 AUDITORIA SOMA → DEZENAS v1.2")
    print(f"   Divisão: {frac_descoberta:.0%} descoberta / {frac_confirmacao:.0%} confirmação / "
          f"{1-frac_descoberta-frac_confirmacao:.0%} holdout")
    print(f"   Discretização da soma: {n_bins} quantis")
    print(f"   Placebos: descoberta={n_placebos_descoberta}, confirmação={n_placebos_confirmacao}, "
          f"holdout={n_placebos_holdout}\n")

    # Construir transições S_t -> X_{t+1}
    n = len(contests)
    somas = np.array([soma_concurso(c['dezenas']) for c in contests])
    presencas = matriz_presenca(contests)
    X_soma = somas[:-1]          # soma no tempo t
    Y_next = presencas[1:]       # presença no tempo t+1

    # Dividir em três blocos temporais
    n_trans = len(X_soma)
    split_desc = int(n_trans * frac_descoberta)
    split_conf = int(n_trans * (frac_descoberta + frac_confirmacao))

    somas_desc = X_soma[:split_desc]
    pres_desc = Y_next[:split_desc]
    somas_conf = X_soma[split_desc:split_conf]
    pres_conf = Y_next[split_desc:split_conf]
    somas_hold = X_soma[split_conf:]
    pres_hold = Y_next[split_conf:]

    # Definir limites de discretização e Δ usando apenas descoberta
    limites = np.percentile(somas_desc, np.linspace(0, 100, n_bins+1))
    limite_alto = np.percentile(somas_desc, 70)
    limite_baixo = np.percentile(somas_desc, 30)

    def discretizar(somas):
        return np.digitize(somas, limites[1:-1])

    somas_desc_disc = discretizar(somas_desc)
    somas_conf_disc = discretizar(somas_conf)
    somas_hold_disc = discretizar(somas_hold)

    # Contagens de bins (diagnóstico)
    print("Bins descoberta:", np.bincount(somas_desc_disc))
    print("Bins confirmação:", np.bincount(somas_conf_disc, minlength=n_bins))
    print("Bins holdout:", np.bincount(somas_hold_disc, minlength=n_bins))
    print()

    rng_desc = np.random.default_rng(20260911)
    rng_conf = np.random.default_rng(20260912)
    rng_hold = np.random.default_rng(20260913)

    # =====================================================
    # FASE DE DESCOBERTA
    # =====================================================
    print("Fase de DESCOBERTA...")
    resultados_desc = []

    for d in range(25):
        y = pres_desc[:, d]
        # MI real (usando soma discretizada)
        mi_real = informacao_mutua_discreta(somas_desc_disc, y)
        # AUC real (usando soma contínua)
        auc_real = auc_roc(y, somas_desc)
        # Δ real (quantis 30/70 fixos)
        mask_alto = somas_desc >= limite_alto
        mask_baixo = somas_desc <= limite_baixo
        p_alto = np.mean(y[mask_alto]) if np.sum(mask_alto) > 0 else 0.0
        p_baixo = np.mean(y[mask_baixo]) if np.sum(mask_baixo) > 0 else 0.0
        delta = p_alto - p_baixo

        # Placebos: permutar somas, manter y
        mi_placebo = []
        auc_placebo = []
        delta_placebo = []
        for _ in range(n_placebos_descoberta):
            idx = rng_desc.permutation(len(somas_desc))
            somas_p = somas_desc[idx]
            somas_p_disc = discretizar(somas_p)
            mi_placebo.append(informacao_mutua_discreta(somas_p_disc, y))
            auc_placebo.append(auc_roc(y, somas_p))
            mask_alto_p = somas_p >= limite_alto
            mask_baixo_p = somas_p <= limite_baixo
            p_alto_p = np.mean(y[mask_alto_p]) if np.sum(mask_alto_p) > 0 else 0.0
            p_baixo_p = np.mean(y[mask_baixo_p]) if np.sum(mask_baixo_p) > 0 else 0.0
            delta_placebo.append(p_alto_p - p_baixo_p)

        mi_placebo = np.array(mi_placebo)
        auc_placebo = np.array(auc_placebo)
        delta_placebo = np.array(delta_placebo)

        p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (len(mi_placebo) + 1)
        auc_dev = abs(auc_real - 0.5)
        auc_dev_placebo = np.abs(auc_placebo - 0.5)
        p_auc = (1 + np.sum(auc_dev_placebo >= auc_dev)) / (len(auc_placebo) + 1)
        p_delta = (1 + np.sum(np.abs(delta_placebo) >= abs(delta))) / (len(delta_placebo) + 1)

        resultados_desc.append({
            'dezena': d+1,
            'mi_real': mi_real,
            'p_mi': p_mi,
            'auc_real': auc_real,
            'p_auc': p_auc,
            'delta': delta,
            'p_delta': p_delta,
        })

    # FDR sobre p-values de MI e AUC (conjunto)
    p_todos = []
    for r in resultados_desc:
        p_todos.append(r['p_mi'])
        p_todos.append(r['p_auc'])
    rej_desc, q_todos = fdr_bh(p_todos, alpha)

    selecionadas = set()
    for i, r in enumerate(resultados_desc):
        r['q_mi'] = q_todos[2*i]
        r['q_auc'] = q_todos[2*i+1]
        if r['q_mi'] < alpha or r['q_auc'] < alpha:
            selecionadas.add(r['dezena'])

    selecionadas = sorted(selecionadas)
    print(f"\nDezenas selecionadas na descoberta: {len(selecionadas)}")
    for d in selecionadas:
        r = resultados_desc[d-1]
        print(f"   Dezena {r['dezena']:2d}: MI={r['mi_real']:.4f} (q={r['q_mi']:.4f}), "
              f"AUC={r['auc_real']:.4f} (q={r['q_auc']:.4f}), Δ={r['delta']:+.4f} (p={r['p_delta']:.4f})")

    if not selecionadas:
        print("   Nenhuma dezena selecionada.")
        print("   A análise Soma → Próxima Soma continuará independentemente.")

    # =====================================================
    # FASE DE CONFIRMAÇÃO (apenas se houver selecionadas)
    # =====================================================
    if selecionadas:
        print("\nFase de CONFIRMAÇÃO...")
        resultados_conf = []

        for d in selecionadas:
            y = pres_conf[:, d-1]
            mi_real = informacao_mutua_discreta(somas_conf_disc, y)
            auc_real = auc_roc(y, somas_conf)
            mask_alto = somas_conf >= limite_alto
            mask_baixo = somas_conf <= limite_baixo
            p_alto = np.mean(y[mask_alto]) if np.sum(mask_alto) > 0 else 0.0
            p_baixo = np.mean(y[mask_baixo]) if np.sum(mask_baixo) > 0 else 0.0
            delta = p_alto - p_baixo

            mi_placebo = []
            auc_placebo = []
            delta_placebo = []
            for _ in range(n_placebos_confirmacao):
                idx = rng_conf.permutation(len(somas_conf))
                somas_p = somas_conf[idx]
                somas_p_disc = discretizar(somas_p)
                mi_placebo.append(informacao_mutua_discreta(somas_p_disc, y))
                auc_placebo.append(auc_roc(y, somas_p))
                mask_alto_p = somas_p >= limite_alto
                mask_baixo_p = somas_p <= limite_baixo
                p_alto_p = np.mean(y[mask_alto_p]) if np.sum(mask_alto_p) > 0 else 0.0
                p_baixo_p = np.mean(y[mask_baixo_p]) if np.sum(mask_baixo_p) > 0 else 0.0
                delta_placebo.append(p_alto_p - p_baixo_p)

            mi_placebo = np.array(mi_placebo)
            auc_placebo = np.array(auc_placebo)
            delta_placebo = np.array(delta_placebo)

            p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (len(mi_placebo) + 1)
            auc_dev = abs(auc_real - 0.5)
            auc_dev_placebo = np.abs(auc_placebo - 0.5)
            p_auc = (1 + np.sum(auc_dev_placebo >= auc_dev)) / (len(auc_placebo) + 1)
            p_delta = (1 + np.sum(np.abs(delta_placebo) >= abs(delta))) / (len(delta_placebo) + 1)

            resultados_conf.append({
                'dezena': d,
                'mi_real': mi_real,
                'p_mi': p_mi,
                'auc_real': auc_real,
                'p_auc': p_auc,
                'delta': delta,
                'p_delta': p_delta,
                'p_cond_alto': p_alto,
                'p_cond_baixo': p_baixo,
            })

        # FDR conjunto
        p_conf_todos = []
        for r in resultados_conf:
            p_conf_todos.append(r['p_mi'])
            p_conf_todos.append(r['p_auc'])
        rej_conf, q_conf_todos = fdr_bh(p_conf_todos, alpha)

        for i, r in enumerate(resultados_conf):
            r['q_mi'] = q_conf_todos[2*i]
            r['q_auc'] = q_conf_todos[2*i+1]
            r['nominal'] = r['p_mi'] < alpha or r['p_auc'] < alpha
            r['confirmado_fdr'] = r['q_mi'] < alpha or r['q_auc'] < alpha

        # =====================================================
        # FASE DE HOLDOUT FINAL (apenas se houver selecionadas)
        # =====================================================
        print("\nFase de HOLDOUT FINAL...")
        resultados_hold = []

        for r in resultados_conf:
            d = r['dezena']
            y = pres_hold[:, d-1]
            mi_real = informacao_mutua_discreta(somas_hold_disc, y)
            auc_real = auc_roc(y, somas_hold)
            mask_alto = somas_hold >= limite_alto
            mask_baixo = somas_hold <= limite_baixo
            p_alto = np.mean(y[mask_alto]) if np.sum(mask_alto) > 0 else 0.0
            p_baixo = np.mean(y[mask_baixo]) if np.sum(mask_baixo) > 0 else 0.0
            delta = p_alto - p_baixo

            mi_placebo = []
            auc_placebo = []
            delta_placebo = []
            for _ in range(n_placebos_holdout):
                idx = rng_hold.permutation(len(somas_hold))
                somas_p = somas_hold[idx]
                somas_p_disc = discretizar(somas_p)
                mi_placebo.append(informacao_mutua_discreta(somas_p_disc, y))
                auc_placebo.append(auc_roc(y, somas_p))
                mask_alto_p = somas_p >= limite_alto
                mask_baixo_p = somas_p <= limite_baixo
                p_alto_p = np.mean(y[mask_alto_p]) if np.sum(mask_alto_p) > 0 else 0.0
                p_baixo_p = np.mean(y[mask_baixo_p]) if np.sum(mask_baixo_p) > 0 else 0.0
                delta_placebo.append(p_alto_p - p_baixo_p)

            mi_placebo = np.array(mi_placebo)
            auc_placebo = np.array(auc_placebo)
            delta_placebo = np.array(delta_placebo)

            p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (len(mi_placebo) + 1)
            auc_dev = abs(auc_real - 0.5)
            auc_dev_placebo = np.abs(auc_placebo - 0.5)
            p_auc = (1 + np.sum(auc_dev_placebo >= auc_dev)) / (len(auc_placebo) + 1)
            p_delta = (1 + np.sum(np.abs(delta_placebo) >= abs(delta))) / (len(delta_placebo) + 1)

            resultados_hold.append({
                'dezena': d,
                'mi_real': mi_real,
                'p_mi': p_mi,
                'auc_real': auc_real,
                'p_auc': p_auc,
                'delta': delta,
                'p_delta': p_delta,
                'p_cond_alto': p_alto,
                'p_cond_baixo': p_baixo,
            })

        # FDR conjunto no holdout
        p_hold_todos = []
        for r in resultados_hold:
            p_hold_todos.append(r['p_mi'])
            p_hold_todos.append(r['p_auc'])
        rej_hold, q_hold_todos = fdr_bh(p_hold_todos, alpha)

        for i, r in enumerate(resultados_hold):
            r['q_mi'] = q_hold_todos[2*i]
            r['q_auc'] = q_hold_todos[2*i+1]
            r['nominal_hold'] = r['p_mi'] < alpha or r['p_auc'] < alpha
            r['confirmado_hold'] = r['q_mi'] < alpha or r['q_auc'] < alpha

        # =====================================================
        # EXIBIÇÃO FINAL (apenas se houver selecionadas)
        # =====================================================
        print("\n📊 RESULTADOS")
        print(f"{'Dez':<4} {'MI desc':<8} {'AUC desc':<8} {'MI conf':<8} {'AUC conf':<8} "
              f"{'MI hold':<8} {'AUC hold':<8} {'Δ hold':<8} {'p MI hold':<10} {'q hold':<8}")
        print("-" * 90)
        for i, r_conf in enumerate(resultados_conf):
            r_hold = resultados_hold[i]
            print(f"{r_conf['dezena']:<4} "
                  f"{resultados_desc[r_conf['dezena']-1]['mi_real']:.4f}   "
                  f"{resultados_desc[r_conf['dezena']-1]['auc_real']:.4f}   "
                  f"{r_conf['mi_real']:.4f}   {r_conf['auc_real']:.4f}   "
                  f"{r_hold['mi_real']:.4f}   {r_hold['auc_real']:.4f}   "
                  f"{r_hold['delta']:+.4f}   {r_hold['p_mi']:.4f}   "
                  f"{min(r_hold['q_mi'], r_hold['q_auc']):.4f}")

        n_total = len(selecionadas)
        n_conf_nominal = sum(1 for r in resultados_conf if r['nominal'])
        n_conf_fdr = sum(1 for r in resultados_conf if r['confirmado_fdr'])
        n_hold_nominal = sum(1 for r in resultados_hold if r['nominal_hold'])
        n_hold_fdr = sum(1 for r in resultados_hold if r['confirmado_hold'])

        print(f"\n🔍 RESUMO")
        print(f"   Dezenas selecionadas: {n_total}")
        print(f"   Confirmação nominal: {n_conf_nominal}")
        print(f"   Confirmação FDR: {n_conf_fdr}")
        print(f"   Holdout nominal: {n_hold_nominal}")
        print(f"   Holdout FDR: {n_hold_fdr}")

        if n_hold_fdr == 0:
            print("\n✅ Nenhuma dezena apresentou dependência significativa com a soma anterior")
            print("   no holdout após FDR. Não há evidência de que a soma do concurso passado")
            print("   contenha informação estável sobre a presença das dezenas no próximo.")
        else:
            print("\n⚠️ Há dezenas com evidência no holdout; investigar com modelos específicos.")

    # =====================================================
    # ANÁLISE ADICIONAL: S_t -> S_{t+1} (sempre executada)
    # =====================================================
    print("\n📊 ANÁLISE SOMA → PRÓXIMA SOMA (60/20/20)")
    # Preparar transições S_t -> S_{t+1}
    soma_t = X_soma[:-1]   # S_t
    soma_t1 = X_soma[1:]   # S_{t+1}
    n_ss = len(soma_t)
    split_ss_desc = int(n_ss * 0.6)
    split_ss_conf = int(n_ss * 0.8)

    soma_t_desc = soma_t[:split_ss_desc]
    soma_t1_desc = soma_t1[:split_ss_desc]
    soma_t_conf = soma_t[split_ss_desc:split_ss_conf]
    soma_t1_conf = soma_t1[split_ss_desc:split_ss_conf]
    soma_t_hold = soma_t[split_ss_conf:]
    soma_t1_hold = soma_t1[split_ss_conf:]

    # Limites de discretização baseados na descoberta
    lim_soma_ss = np.percentile(soma_t_desc, np.linspace(0, 100, 6))

    def discretizar_soma_ss(somas):
        return np.digitize(somas, lim_soma_ss[1:-1])

    def calc_metricas_soma_soma(t, t1):
        pearson = np.corrcoef(t, t1)[0,1]
        rho, p_spearman = spearmanr(t, t1)
        disc_t = discretizar_soma_ss(t)
        disc_t1 = discretizar_soma_ss(t1)
        cont = Counter(zip(disc_t, disc_t1))
        total = len(disc_t)
        px = Counter(disc_t)
        py = Counter(disc_t1)
        mi = 0.0
        for (a,b), n in cont.items():
            pxy = n / total
            px_ = px[a] / total
            py_ = py[b] / total
            if pxy > 0:
                mi += pxy * np.log2(pxy / (px_ * py_))
        return pearson, rho, p_spearman, mi

    # Descoberta
    pear_desc, rho_desc, p_spear_desc, mi_desc = calc_metricas_soma_soma(soma_t_desc, soma_t1_desc)
    print(f"   DESCOBERTA: Pearson={pear_desc:+.4f}, Spearman={rho_desc:+.4f}, MI={mi_desc:.4f}")

    # Placebos para descoberta
    rng_ss = np.random.default_rng(20260915)
    pear_placebo_desc = []
    rho_placebo_desc = []
    mi_placebo_desc = []
    for _ in range(1000):
        idx = rng_ss.permutation(len(soma_t_desc))
        t_p = soma_t_desc[idx]
        pear_p, rho_p, _, mi_p = calc_metricas_soma_soma(t_p, soma_t1_desc)
        pear_placebo_desc.append(pear_p)
        rho_placebo_desc.append(rho_p)
        mi_placebo_desc.append(mi_p)
    p_pear_desc = (1 + np.sum(np.abs(pear_placebo_desc) >= abs(pear_desc))) / (len(pear_placebo_desc)+1)
    p_rho_desc = (1 + np.sum(np.abs(rho_placebo_desc) >= abs(rho_desc))) / (len(rho_placebo_desc)+1)
    p_mi_desc = (1 + np.sum(mi_placebo_desc >= mi_desc)) / (len(mi_placebo_desc)+1)
    print(f"   p-valores descoberta: Pearson={p_pear_desc:.4f}, Spearman={p_rho_desc:.4f}, MI={p_mi_desc:.4f}")

    # Confirmação
    pear_conf, rho_conf, p_spear_conf, mi_conf = calc_metricas_soma_soma(soma_t_conf, soma_t1_conf)
    print(f"   CONFIRMAÇÃO: Pearson={pear_conf:+.4f}, Spearman={rho_conf:+.4f}, MI={mi_conf:.4f}")
    pear_placebo_conf = []
    rho_placebo_conf = []
    mi_placebo_conf = []
    for _ in range(1000):
        idx = rng_ss.permutation(len(soma_t_conf))
        t_p = soma_t_conf[idx]
        pear_p, rho_p, _, mi_p = calc_metricas_soma_soma(t_p, soma_t1_conf)
        pear_placebo_conf.append(pear_p)
        rho_placebo_conf.append(rho_p)
        mi_placebo_conf.append(mi_p)
    p_pear_conf = (1 + np.sum(np.abs(pear_placebo_conf) >= abs(pear_conf))) / (len(pear_placebo_conf)+1)
    p_rho_conf = (1 + np.sum(np.abs(rho_placebo_conf) >= abs(rho_conf))) / (len(rho_placebo_conf)+1)
    p_mi_conf = (1 + np.sum(mi_placebo_conf >= mi_conf)) / (len(mi_placebo_conf)+1)
    print(f"   p-valores confirmação: Pearson={p_pear_conf:.4f}, Spearman={p_rho_conf:.4f}, MI={p_mi_conf:.4f}")

    # Holdout
    pear_hold, rho_hold, p_spear_hold, mi_hold = calc_metricas_soma_soma(soma_t_hold, soma_t1_hold)
    print(f"   HOLDOUT: Pearson={pear_hold:+.4f}, Spearman={rho_hold:+.4f}, MI={mi_hold:.4f}")
    pear_placebo_hold = []
    rho_placebo_hold = []
    mi_placebo_hold = []
    for _ in range(1000):
        idx = rng_ss.permutation(len(soma_t_hold))
        t_p = soma_t_hold[idx]
        pear_p, rho_p, _, mi_p = calc_metricas_soma_soma(t_p, soma_t1_hold)
        pear_placebo_hold.append(pear_p)
        rho_placebo_hold.append(rho_p)
        mi_placebo_hold.append(mi_p)
    p_pear_hold = (1 + np.sum(np.abs(pear_placebo_hold) >= abs(pear_hold))) / (len(pear_placebo_hold)+1)
    p_rho_hold = (1 + np.sum(np.abs(rho_placebo_hold) >= abs(rho_hold))) / (len(rho_placebo_hold)+1)
    p_mi_hold = (1 + np.sum(mi_placebo_hold >= mi_hold)) / (len(mi_placebo_hold)+1)
    print(f"   p-valores holdout: Pearson={p_pear_hold:.4f}, Spearman={p_rho_hold:.4f}, MI={p_mi_hold:.4f}")

    return resultados_desc, (resultados_conf if selecionadas else None), (resultados_hold if selecionadas else None)

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔍 AUDITORIA SOMA → DEZENAS v1.2")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria soma → dezenas")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            try:
                frac_desc = float(input("   Fração de descoberta [0.6]: ").strip() or "0.6")
                frac_conf = float(input("   Fração de confirmação [0.2]: ").strip() or "0.2")
                n_placebos_desc = int(input("   Placebos descoberta [5000]: ").strip() or "5000")
                n_placebos_conf = int(input("   Placebos confirmação [5000]: ").strip() or "5000")
                n_placebos_hold = int(input("   Placebos holdout [5000]: ").strip() or "5000")
                n_bins = int(input("   Número de quantis da soma [5]: ").strip() or "5")
            except:
                frac_desc, frac_conf = 0.6, 0.2
                n_placebos_desc = n_placebos_conf = n_placebos_hold = 5000
                n_bins = 5
            auditoria_soma_dezenas(contests, frac_descoberta=frac_desc, frac_confirmacao=frac_conf,
                                   n_placebos_descoberta=n_placebos_desc,
                                   n_placebos_confirmacao=n_placebos_conf,
                                   n_placebos_holdout=n_placebos_hold,
                                   n_bins=n_bins)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
