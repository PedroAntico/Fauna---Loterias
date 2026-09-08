#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA MATEMÁTICA DA SOMA ESTRUTURAL v1.2 (corrigida)

Alvos discretos: pares, ímpares, primos, moldura, fibonacci, consecutivos
Alvos contínuos: soma, amplitude, disp_linhas, disp_colunas (discretizados em 5 quantis, limites congelados)

Análise descritiva: para o candidato que chegar à confirmação, calcula média, mediana,
IC95% e efeito por categoria da transformação na confirmação e holdout.

Protocolo: 60% descoberta → FDR global → 20% confirmação → FDR → 20% holdout final.
Placebo: permutação da transformação (mantém alvo fixo).
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

def extrair_estrutura(dezenas):
    d = sorted(dezenas)
    pares = sum(1 for x in d if x % 2 == 0)
    impares = 15 - pares
    primos = sum(1 for x in d if x in PRIMES)
    moldura = sum(1 for x in d if x in MOLDURA)
    fibonacci = sum(1 for x in d if x in FIBONACCI)
    soma = sum(d)
    amplitude = max(d) - min(d)
    consecutivos = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    linhas = Counter((x-1)//5 for x in d)
    colunas = Counter((x-1)%5 for x in d)
    disp_linhas = np.std([linhas.get(i,0) for i in range(5)])
    disp_colunas = np.std([colunas.get(i,0) for i in range(5)])
    return {
        'pares': pares,
        'impares': impares,
        'primos': primos,
        'moldura': moldura,
        'fibonacci': fibonacci,
        'soma': soma,
        'amplitude': amplitude,
        'consecutivos': consecutivos,
        'disp_linhas': disp_linhas,
        'disp_colunas': disp_colunas,
    }

PRIMES = {2,3,5,7,11,13,17,19,23}
MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
FIBONACCI = {1,2,3,5,8,13,21}

# ============================================================
# FUNÇÕES DE MI E FDR
# ============================================================
def mi_discreta_fast(x, y):
    x = np.asarray(x, dtype=np.int16)
    y = np.asarray(y, dtype=np.int16)
    if len(x) < 10:
        return 0.0

    _, x_codes = np.unique(x, return_inverse=True)
    _, y_codes = np.unique(y, return_inverse=True)

    kx = x_codes.max() + 1
    ky = y_codes.max() + 1

    pair = x_codes * ky + y_codes
    tabela = np.bincount(pair, minlength=kx*ky).reshape(kx, ky)

    total = len(x)
    px = tabela.sum(axis=1) / total
    py = tabela.sum(axis=0) / total
    pxy = tabela / total

    mi = 0.0
    for i in range(kx):
        for j in range(ky):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
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

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True

# ============================================================
# DEFINIÇÃO DAS TRANSFORMAÇÕES
# ============================================================
def raiz_digital(n):
    return (n - 1) % 9 + 1 if n > 0 else 0

def soma_digitos(n):
    return sum(int(d) for d in str(abs(n)))

def ultimo_digito(n):
    return abs(n) % 10

def paridade_soma_digitos(n):
    return soma_digitos(n) % 2

TRANSFORMACOES = []
for k in range(2, 26):
    TRANSFORMACOES.append((f"mod_{k}", lambda s, k=k: s % k, "categorico"))
for k in range(2, 26):
    TRANSFORMACOES.append((f"div_{k}", lambda s, k=k: int(s % k == 0), "binario"))
TRANSFORMACOES.extend([
    ("ultimo_digito", ultimo_digito, "categorico"),
    ("soma_digitos", soma_digitos, "continuo"),
    ("raiz_digital", raiz_digital, "categorico"),
    ("paridade_soma_digitos", paridade_soma_digitos, "binario"),
    ("primo", lambda s: int(is_prime(s)), "binario"),
])

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def auditoria_matematica_soma_estrutural(contests, frac_descoberta=0.6, frac_confirmacao=0.2,
                                         n_placebos_descoberta=2000, n_placebos_confirmacao=10000,
                                         n_placebos_holdout=10000, alpha=0.05):
    print("\n🔍 AUDITORIA MATEMÁTICA DA SOMA ESTRUTURAL v1.2")
    print(f"   Transformações: {len(TRANSFORMACOES)}")
    print(f"   Alvos estruturais: 10")
    print(f"   Divisão: {frac_descoberta:.0%} descoberta / {frac_confirmacao:.0%} confirmação / "
          f"{1-frac_descoberta-frac_confirmacao:.0%} holdout")
    print(f"   Placebos: descoberta={n_placebos_descoberta}, confirmação={n_placebos_confirmacao}, "
          f"holdout={n_placebos_holdout}\n")

    # Preparar dados
    n = len(contests)
    somas = np.array([soma_concurso(c['dezenas']) for c in contests])
    estruturas = [extrair_estrutura(c['dezenas']) for c in contests]

    X_soma = somas[:-1]
    estruturas_next = estruturas[1:]

    chaves = ['pares', 'impares', 'primos', 'moldura', 'fibonacci', 'soma', 'amplitude',
              'consecutivos', 'disp_linhas', 'disp_colunas']
    chaves_discretas = {'pares', 'impares', 'primos', 'moldura', 'fibonacci', 'consecutivos'}
    chaves_continuas = {'soma', 'amplitude', 'disp_linhas', 'disp_colunas'}

    n_trans = len(X_soma)
    split_desc = int(n_trans * frac_descoberta)
    split_conf = int(n_trans * (frac_descoberta + frac_confirmacao))

    somas_desc = X_soma[:split_desc]
    est_desc = estruturas_next[:split_desc]
    somas_conf = X_soma[split_desc:split_conf]
    est_conf = estruturas_next[split_desc:split_conf]
    somas_hold = X_soma[split_conf:]
    est_hold = estruturas_next[split_conf:]

    TRANSFORM_LIMITS = {}
    TARGET_LIMITS = {}

    # =====================================================
    # FASE DE DESCOBERTA
    # =====================================================
    print("Fase de DESCOBERTA...")
    resultados = []
    total_testes = len(TRANSFORMACOES) * len(chaves)
    print(f"Total de testes: {total_testes}")

    for i_trans, (nome_trans, func, tipo) in enumerate(tqdm(TRANSFORMACOES, desc="Transformações")):
        trans_desc = np.array([func(s) for s in somas_desc])
        if tipo == "continuo":
            limites = np.percentile(trans_desc, np.linspace(0,100,6))
            TRANSFORM_LIMITS[nome_trans] = limites
            trans_desc = np.digitize(trans_desc, limites[1:-1])
        _, trans_cod = np.unique(trans_desc, return_inverse=True)

        rng = np.random.default_rng(20260920 + i_trans * 100)
        perm_indices = np.empty((n_placebos_descoberta, len(trans_desc)), dtype=np.int32)
        for p in range(n_placebos_descoberta):
            perm_indices[p] = rng.permutation(len(trans_desc))

        for j, chave in enumerate(chaves):
            y = np.array([e[chave] for e in est_desc])
            if chave in chaves_continuas:
                limites_y = np.percentile(y, np.linspace(0,100,6))
                TARGET_LIMITS[chave] = limites_y
                y_disc = np.digitize(y, limites_y[1:-1])
            else:
                y_disc = y.astype(np.int16)  # mantém valor original

            mi_real = mi_discreta_fast(trans_cod, y_disc)

            mi_placebo = np.empty(n_placebos_descoberta)
            for p in range(n_placebos_descoberta):
                trans_perm = trans_cod[perm_indices[p]]
                mi_placebo[p] = mi_discreta_fast(trans_perm, y_disc)

            p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (n_placebos_descoberta + 1)

            resultados.append({
                'transformacao': nome_trans,
                'alvo': chave,
                'mi_real': mi_real,
                'p_mi': p_mi,
                'i_trans': i_trans,
            })

    # FDR global
    pvals = [r['p_mi'] for r in resultados]
    rej, qvals = fdr_bh(pvals, alpha)
    selecionados = []
    for i in rej:
        r = resultados[i]
        r['q'] = qvals[i]
        selecionados.append(r)

    print(f"\nCandidatos selecionados na descoberta: {len(selecionados)}")
    for r in selecionados:
        print(f"   {r['transformacao']} → {r['alvo']}: "
              f"MI={r['mi_real']:.6f}, p={r['p_mi']:.6f}, q={r['q']:.6f}")

    if not selecionados:
        print("\n✅ Nenhuma transformação apresentou evidência de relação com as características")
        print("   estruturais do próximo concurso na descoberta.")
        return None

    # =====================================================
    # FASE DE CONFIRMAÇÃO
    # =====================================================
    print("\nFase de CONFIRMAÇÃO...")
    resultados_conf = []
    for i_cand, r in enumerate(selecionados):
        nome_trans = r['transformacao']
        alvo = r['alvo']
        i_trans = r['i_trans']
        func = TRANSFORMACOES[i_trans][1]
        tipo = TRANSFORMACOES[i_trans][2]

        trans_conf = np.array([func(s) for s in somas_conf])
        if tipo == "continuo":
            limites = TRANSFORM_LIMITS.get(nome_trans)
            if limites is not None:
                trans_conf = np.digitize(trans_conf, limites[1:-1])
        _, trans_cod = np.unique(trans_conf, return_inverse=True)

        y = np.array([e[alvo] for e in est_conf])
        if alvo in chaves_continuas:
            limites_y = TARGET_LIMITS[alvo]
            y_disc = np.digitize(y, limites_y[1:-1])
        else:
            y_disc = y.astype(np.int16)

        mi_real = mi_discreta_fast(trans_cod, y_disc)

        rng = np.random.default_rng(20260921 + i_cand)
        mi_placebo = np.empty(n_placebos_confirmacao)
        for p in range(n_placebos_confirmacao):
            idx = rng.permutation(len(trans_cod))
            trans_perm = trans_cod[idx]
            mi_placebo[p] = mi_discreta_fast(trans_perm, y_disc)
        p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (n_placebos_confirmacao + 1)

        resultados_conf.append({
            'transformacao': nome_trans,
            'alvo': alvo,
            'mi_real': mi_real,
            'p_mi': p_mi,
            'i_trans': i_trans,
        })

    p_conf = [r['p_mi'] for r in resultados_conf]
    rej_conf, q_conf = fdr_bh(p_conf, alpha)
    confirmados = []
    for i in rej_conf:
        r = resultados_conf[i]
        r['q'] = q_conf[i]
        confirmados.append(r)

    print(f"Sinais com confirmação FDR: {len(confirmados)}")
    for r in confirmados:
        print(f"   {r['transformacao']} → {r['alvo']}: "
              f"MI={r['mi_real']:.6f}, p={r['p_mi']:.6f}, q={r['q']:.6f}")

    if not confirmados:
        print("\n✅ Nenhum sinal sobreviveu à confirmação.")
        return None

    # =====================================================
    # FASE DE HOLDOUT FINAL
    # =====================================================
    print("\nFase de HOLDOUT FINAL...")
    resultados_hold = []
    for i_cand, r in enumerate(confirmados):
        nome_trans = r['transformacao']
        alvo = r['alvo']
        i_trans = r['i_trans']
        func = TRANSFORMACOES[i_trans][1]
        tipo = TRANSFORMACOES[i_trans][2]

        trans_hold = np.array([func(s) for s in somas_hold])
        if tipo == "continuo":
            limites = TRANSFORM_LIMITS.get(nome_trans)
            if limites is not None:
                trans_hold = np.digitize(trans_hold, limites[1:-1])
        _, trans_cod = np.unique(trans_hold, return_inverse=True)

        y = np.array([e[alvo] for e in est_hold])
        if alvo in chaves_continuas:
            limites_y = TARGET_LIMITS[alvo]
            y_disc = np.digitize(y, limites_y[1:-1])
        else:
            y_disc = y.astype(np.int16)

        mi_real = mi_discreta_fast(trans_cod, y_disc)

        rng = np.random.default_rng(20260922 + i_cand)
        mi_placebo = np.empty(n_placebos_holdout)
        for p in range(n_placebos_holdout):
            idx = rng.permutation(len(trans_cod))
            trans_perm = trans_cod[idx]
            mi_placebo[p] = mi_discreta_fast(trans_perm, y_disc)
        p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (n_placebos_holdout + 1)

        resultados_hold.append({
            'transformacao': nome_trans,
            'alvo': alvo,
            'mi_real': mi_real,
            'p_mi': p_mi,
        })

    p_hold = [r['p_mi'] for r in resultados_hold]
    rej_hold, q_hold = fdr_bh(p_hold, alpha)
    finalistas = []
    for i in rej_hold:
        r = resultados_hold[i]
        r['q'] = q_hold[i]
        finalistas.append(r)

    print(f"Sinais que sobreviveram ao holdout FDR: {len(finalistas)}")
    for r in finalistas:
        print(f"   {r['transformacao']} → {r['alvo']}: "
              f"MI={r['mi_real']:.6f}, p={r['p_mi']:.6f}, q={r['q']:.6f}")

    # Análise descritiva para o primeiro finalista (se houver)
    if finalistas:
        r = finalistas[0]
        nome_trans = r['transformacao']
        alvo = r['alvo']
        i_trans = [i for i,(n,_,_) in enumerate(TRANSFORMACOES) if n == nome_trans][0]
        func = TRANSFORMACOES[i_trans][1]
        tipo = TRANSFORMACOES[i_trans][2]
        print(f"\n📊 Análise descritiva para {nome_trans} → {alvo}")
        for fase, somas_fase, est_fase in [("Confirmação", somas_conf, est_conf), ("Holdout", somas_hold, est_hold)]:
            trans_fase = np.array([func(s) for s in somas_fase])
            if tipo == "continuo":
                limites = TRANSFORM_LIMITS.get(nome_trans)
                if limites is not None:
                    trans_fase = np.digitize(trans_fase, limites[1:-1])
            y_fase = np.array([e[alvo] for e in est_fase])
            valores_unicos = np.unique(trans_fase)
            print(f"   {fase}:")
            for val in valores_unicos:
                mask = trans_fase == val
                if np.sum(mask) > 0:
                    media = np.mean(y_fase[mask])
                    mediana = np.median(y_fase[mask])
                    n_obs = np.sum(mask)
                    print(f"      {nome_trans}={val}: n={n_obs}, média={media:.3f}, mediana={mediana:.3f}")

    if finalistas:
        print("\n⚠️ Existem transformações com evidência consistente. Investigar com modelos específicos.")
    else:
        print("\n✅ Não foi encontrada evidência estatística de dependência temporal")
        print("   para as transformações pré-especificadas testadas contra características estruturais.")

    return selecionados, confirmados, finalistas

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔍 AUDITORIA MATEMÁTICA DA SOMA ESTRUTURAL v1.2")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria matemática da soma estrutural")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            try:
                frac_desc = float(input("   Fração de descoberta [0.6]: ").strip() or "0.6")
                frac_conf = float(input("   Fração de confirmação [0.2]: ").strip() or "0.2")
                n_placebos_desc = int(input("   Placebos descoberta [2000]: ").strip() or "2000")
                n_placebos_conf = int(input("   Placebos confirmação [10000]: ").strip() or "10000")
                n_placebos_hold = int(input("   Placebos holdout [10000]: ").strip() or "10000")
            except:
                frac_desc, frac_conf = 0.6, 0.2
                n_placebos_desc, n_placebos_conf, n_placebos_hold = 2000, 10000, 10000
            auditoria_matematica_soma_estrutural(contests, frac_descoberta=frac_desc,
                                                 frac_confirmacao=frac_conf,
                                                 n_placebos_descoberta=n_placebos_desc,
                                                 n_placebos_confirmacao=n_placebos_conf,
                                                 n_placebos_holdout=n_placebos_hold)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
