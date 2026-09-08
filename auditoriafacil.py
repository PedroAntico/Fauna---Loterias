#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA MATEMÁTICA DA SOMA v1.1 (corrigida)

Correções:
- Placebos reprodutíveis (índice determinístico, sem hash())
- n_placebos_descoberta = 5000
- Remoção de transformações duplicadas (par, mult_3, mult_5, mult_7)
- Exibição correta dos q-values
- Conclusão cautelosa: ausência de evidência, não prova de independência

Famílias:
  A - Resíduos: S_t mod k, para k=2..25
  B - Divisibilidade: S_t é divisível por k?
  C - Dígitos: último dígito, soma dos dígitos, raiz digital, paridade da soma dos dígitos
  D - Relações simples: primo/composto

Protocolo:
  60% descoberta → FDR global → 20% confirmação → FDR → 20% holdout final.
Placebo: permutação da transformação (mantém alvo fixo).

Métricas: Informação Mútua (MI) entre transformação e indicador da dezena.
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
def soma_concurso(dezenas):
    return sum(dezenas)

def matriz_presenca(contests):
    n = len(contests)
    X = np.zeros((n, 25), dtype=np.int8)
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            X[i, d-1] = 1
    return X

def informacao_mutua_discreta(x, y):
    """MI entre x categórico (já discretizado) e y binário."""
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
# DEFINIÇÃO DAS TRANSFORMAÇÕES (sem duplicatas)
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
# A - Resíduos
for k in range(2, 26):
    TRANSFORMACOES.append((f"mod_{k}", lambda s, k=k: s % k, "categorico"))
# B - Divisibilidade
for k in range(2, 26):
    TRANSFORMACOES.append((f"div_{k}", lambda s, k=k: int(s % k == 0), "binario"))
# C - Dígitos
TRANSFORMACOES.extend([
    ("ultimo_digito", ultimo_digito, "categorico"),
    ("soma_digitos", soma_digitos, "continuo"),
    ("raiz_digital", raiz_digital, "categorico"),
    ("paridade_soma_digitos", paridade_soma_digitos, "binario"),
])
# D - Relações simples (apenas primo, sem duplicatas)
TRANSFORMACOES.append(("primo", lambda s: int(is_prime(s)), "binario"))

# Criar dicionário nome -> (função, tipo)
TRANSFORM_DICT = {nome: (func, tipo) for nome, func, tipo in TRANSFORMACOES}

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def auditoria_matematica_soma(contests, frac_descoberta=0.6, frac_confirmacao=0.2,
                              n_placebos_descoberta=5000, n_placebos_confirmacao=5000,
                              n_placebos_holdout=5000, alpha=0.05):
    print("\n🔍 AUDITORIA MATEMÁTICA DA SOMA v1.1")
    print(f"   Divisão: {frac_descoberta:.0%} descoberta / {frac_confirmacao:.0%} confirmação / "
          f"{1-frac_descoberta-frac_confirmacao:.0%} holdout")
    print(f"   Total de transformações: {len(TRANSFORMACOES)}")
    print(f"   Placebos: descoberta={n_placebos_descoberta}, confirmação={n_placebos_confirmacao}, "
          f"holdout={n_placebos_holdout}\n")

    # Construir transições S_t -> X_{t+1}
    n = len(contests)
    somas = np.array([soma_concurso(c['dezenas']) for c in contests])
    presencas = matriz_presenca(contests)
    X_soma = somas[:-1]
    Y_next = presencas[1:]

    n_trans = len(X_soma)
    split_desc = int(n_trans * frac_descoberta)
    split_conf = int(n_trans * (frac_descoberta + frac_confirmacao))

    somas_desc = X_soma[:split_desc]
    pres_desc = Y_next[:split_desc]
    somas_conf = X_soma[split_desc:split_conf]
    pres_conf = Y_next[split_desc:split_conf]
    somas_hold = X_soma[split_conf:]
    pres_hold = Y_next[split_conf:]

    # =====================================================
    # FASE DE DESCOBERTA
    # =====================================================
    print("Fase de DESCOBERTA...")
    resultados = []

    total_testes = len(TRANSFORMACOES) * 25
    print(f"Total de testes (transformações × dezenas): {total_testes}")

    for i_trans, (nome_trans, func, tipo) in enumerate(tqdm(TRANSFORMACOES, desc="Transformações")):
        # Aplicar transformação à descoberta
        trans_desc = np.array([func(s) for s in somas_desc])
        # Discretizar se necessário
        if tipo == "continuo":
            limites = np.percentile(trans_desc, np.linspace(0,100,6))
            trans_desc_disc = np.digitize(trans_desc, limites[1:-1])
            # Armazenar limites para uso posterior
            globals().setdefault('TRANSFORM_LIMITS', {})[nome_trans] = limites
        else:
            trans_desc_disc = trans_desc

        for d in range(25):
            y = pres_desc[:, d]
            mi_real = informacao_mutua_discreta(trans_desc_disc, y)

            # Placebo determinístico
            rng = np.random.default_rng(20260916 + i_trans * 100 + d)
            mi_placebo = []
            for _ in range(n_placebos_descoberta):
                idx = rng.permutation(len(trans_desc_disc))
                trans_p = trans_desc_disc[idx]
                mi_placebo.append(informacao_mutua_discreta(trans_p, y))
            mi_placebo = np.array(mi_placebo)
            p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (len(mi_placebo) + 1)

            resultados.append({
                'transformacao': nome_trans,
                'dezena': d+1,
                'mi_real': mi_real,
                'p_mi': p_mi,
            })

    # FDR global
    pvals = [r['p_mi'] for r in resultados]
    rej, qvals = fdr_bh(pvals, alpha)

    selecionados = []
    for i in rej:
        r = resultados[i]
        r['q'] = qvals[i]
        selecionados.append(r)

    print(f"\nCandidatos selecionados na descoberta (q < {alpha}): {len(selecionados)}")
    for r in selecionados:
        print(f"   {r['transformacao']} → dezena {r['dezena']:2d}: "
              f"MI={r['mi_real']:.6f}, p={r['p_mi']:.6f}, q={r['q']:.6f}")

    if not selecionados:
        print("\n✅ Nenhuma transformação matemática da soma apresentou evidência")
        print("   de dependência com a presença de dezenas no próximo concurso")
        print("   na descoberta após FDR global.")
        return None

    # =====================================================
    # FASE DE CONFIRMAÇÃO
    # =====================================================
    print("\nFase de CONFIRMAÇÃO...")
    resultados_conf = []
    for r in selecionados:
        nome_trans = r['transformacao']
        dezena = r['dezena']
        func, tipo = TRANSFORM_DICT[nome_trans]

        trans_conf = np.array([func(s) for s in somas_conf])
        if tipo == "continuo":
            limites = globals().get('TRANSFORM_LIMITS', {}).get(nome_trans)
            if limites is not None:
                trans_conf_disc = np.digitize(trans_conf, limites[1:-1])
            else:
                trans_conf_disc = trans_conf
        else:
            trans_conf_disc = trans_conf

        y = pres_conf[:, dezena-1]
        mi_real = informacao_mutua_discreta(trans_conf_disc, y)

        # Placebo determinístico
        i_trans = [i for i,(n,_,_) in enumerate(TRANSFORMACOES) if n == nome_trans][0]
        rng = np.random.default_rng(20260917 + i_trans * 100 + dezena)
        mi_placebo = []
        for _ in range(n_placebos_confirmacao):
            idx = rng.permutation(len(trans_conf_disc))
            trans_p = trans_conf_disc[idx]
            mi_placebo.append(informacao_mutua_discreta(trans_p, y))
        mi_placebo = np.array(mi_placebo)
        p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (len(mi_placebo) + 1)

        resultados_conf.append({
            'transformacao': nome_trans,
            'dezena': dezena,
            'mi_real': mi_real,
            'p_mi': p_mi,
        })

    # FDR na confirmação
    p_conf = [r['p_mi'] for r in resultados_conf]
    rej_conf, q_conf = fdr_bh(p_conf, alpha)
    confirmados = []
    for i in rej_conf:
        r = resultados_conf[i]
        r['q'] = q_conf[i]
        confirmados.append(r)

    print(f"Sinais com confirmação FDR: {len(confirmados)}")
    for r in confirmados:
        print(f"   {r['transformacao']} → dezena {r['dezena']}: "
              f"MI={r['mi_real']:.6f}, p={r['p_mi']:.6f}, q={r['q']:.6f}")

    if not confirmados:
        print("\n✅ Nenhum sinal sobreviveu à confirmação.")
        return None

    # =====================================================
    # FASE DE HOLDOUT FINAL
    # =====================================================
    print("\nFase de HOLDOUT FINAL...")
    resultados_hold = []
    for r in confirmados:
        nome_trans = r['transformacao']
        dezena = r['dezena']
        func, tipo = TRANSFORM_DICT[nome_trans]

        trans_hold = np.array([func(s) for s in somas_hold])
        if tipo == "continuo":
            limites = globals().get('TRANSFORM_LIMITS', {}).get(nome_trans)
            if limites is not None:
                trans_hold_disc = np.digitize(trans_hold, limites[1:-1])
            else:
                trans_hold_disc = trans_hold
        else:
            trans_hold_disc = trans_hold

        y = pres_hold[:, dezena-1]
        mi_real = informacao_mutua_discreta(trans_hold_disc, y)

        i_trans = [i for i,(n,_,_) in enumerate(TRANSFORMACOES) if n == nome_trans][0]
        rng = np.random.default_rng(20260918 + i_trans * 100 + dezena)
        mi_placebo = []
        for _ in range(n_placebos_holdout):
            idx = rng.permutation(len(trans_hold_disc))
            trans_p = trans_hold_disc[idx]
            mi_placebo.append(informacao_mutua_discreta(trans_p, y))
        mi_placebo = np.array(mi_placebo)
        p_mi = (1 + np.sum(mi_placebo >= mi_real)) / (len(mi_placebo) + 1)

        resultados_hold.append({
            'transformacao': nome_trans,
            'dezena': dezena,
            'mi_real': mi_real,
            'p_mi': p_mi,
        })

    # FDR no holdout
    p_hold = [r['p_mi'] for r in resultados_hold]
    rej_hold, q_hold = fdr_bh(p_hold, alpha)
    finalistas = []
    for i in rej_hold:
        r = resultados_hold[i]
        r['q'] = q_hold[i]
        finalistas.append(r)

    print(f"Sinais que sobreviveram ao holdout FDR: {len(finalistas)}")
    for r in finalistas:
        print(f"   {r['transformacao']} → dezena {r['dezena']}: "
              f"MI={r['mi_real']:.6f}, p={r['p_mi']:.6f}, q={r['q']:.6f}")

    if finalistas:
        print("\n⚠️ Existem transformações com evidência consistente. Investigar com modelos específicos.")
    else:
        print("\n✅ Não foi encontrada evidência estatística de dependência temporal")
        print("   para as transformações pré-especificadas testadas.")

    return selecionados, confirmados, finalistas

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔍 AUDITORIA MATEMÁTICA DA SOMA v1.1")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria matemática da soma")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            try:
                frac_desc = float(input("   Fração de descoberta [0.6]: ").strip() or "0.6")
                frac_conf = float(input("   Fração de confirmação [0.2]: ").strip() or "0.2")
                n_placebos_desc = int(input("   Placebos descoberta [5000]: ").strip() or "5000")
                n_placebos_conf = int(input("   Placebos confirmação [5000]: ").strip() or "5000")
                n_placebos_hold = int(input("   Placebos holdout [5000]: ").strip() or "5000")
            except:
                frac_desc, frac_conf = 0.6, 0.2
                n_placebos_desc = n_placebos_conf = n_placebos_hold = 5000
            auditoria_matematica_soma(contests, frac_descoberta=frac_desc, frac_confirmacao=frac_conf,
                                      n_placebos_descoberta=n_placebos_desc,
                                      n_placebos_confirmacao=n_placebos_conf,
                                      n_placebos_holdout=n_placebos_hold)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
