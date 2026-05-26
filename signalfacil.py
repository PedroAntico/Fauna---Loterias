#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V37 — SIMULADOR DE ESTRATÉGIAS CONDICIONAIS + OTIMIZADOR DE CARTEIRA

Objetivo:
- Testar hipóteses condicionais (pares, moldura, repetidas) de forma massiva
- Simular 100.000 sorteios sintéticos a partir da distribuição histórica
- Otimizar carteiras para maximizar cobertura de pares e diversidade
- Avaliar estratégias com walk‑forward e Monte Carlo
- Sem modelos preditivos: apenas engenharia de carteiras
"""

import numpy as np
from scipy.stats import wilcoxon
from collections import Counter
from itertools import combinations
import random
import os
from tqdm import tqdm
import time

# ============================================================
# CONFIGURAÇÕES
# ============================================================

N_JOGOS = 5
MAX_INTERSECAO = 10
MAX_CONSECUTIVOS_RUN = 3

MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
PRIMES = {2,3,5,7,11,13,17,19,23}

PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}

# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================

def contar_pares(jogo):
    return sum(1 for x in jogo if x % 2 == 0)

def contar_moldura(jogo):
    return sum(1 for x in jogo if x in MOLDURA)

def contar_primos(jogo):
    return sum(1 for x in jogo if x in PRIMES)

def repetidas(jogo, ultimo):
    return len(set(jogo) & set(ultimo))

def interseccao(a, b):
    return len(set(a) & set(b))

def valido(jogo):
    d = sorted(jogo)
    max_run = 1
    run = 1
    for i in range(len(d)-1):
        if d[i+1] - d[i] == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run > MAX_CONSECUTIVOS_RUN:
        return False
    gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
    cluster = sum(1 for g in gaps if g <= 2) / len(gaps)
    return cluster < 0.9

# ============================================================
# GERADOR CONDICIONAL (V36 melhorado)
# ============================================================

def gerar_jogo_condicionado(
    fixas,
    ultimo_concurso,
    alvo_repetidas,
    alvo_pares,
    alvo_moldura,
    tentativas=100000
):
    fixas = sorted(set(fixas))
    if len(fixas) > 15:
        return None

    universo = set(range(1, 26))
    rep_fixas = len(set(fixas) & set(ultimo_concurso))
    faltam_rep = alvo_repetidas - rep_fixas
    if faltam_rep < 0:
        return None

    dezenas_repetidas = list(set(ultimo_concurso) - set(fixas))
    dezenas_novas = list(universo - set(ultimo_concurso) - set(fixas))

    for _ in range(tentativas):
        jogo = set(fixas)

        if faltam_rep > 0:
            if len(dezenas_repetidas) >= faltam_rep:
                escolhidas = random.sample(dezenas_repetidas, faltam_rep)
                jogo.update(escolhidas)
            else:
                continue

        faltam = 15 - len(jogo)
        if faltam < 0:
            continue

        restantes = list(universo - jogo)
        if len(restantes) < faltam:
            continue

        escolhidas = random.sample(restantes, faltam)
        jogo.update(escolhidas)

        jogo = sorted(jogo)
        if len(jogo) != 15:
            continue
        if contar_pares(jogo) != alvo_pares:
            continue
        if contar_moldura(jogo) != alvo_moldura:
            continue
        if repetidas(jogo, ultimo_concurso) != alvo_repetidas:
            continue
        if not valido(jogo):
            continue

        return jogo

    return None

# ============================================================
# GERADOR DE CARTEIRA RÁPIDO
# ============================================================

def gerar_carteira(condicoes, ultimo):
    fixas = condicoes.get('fixas', [])
    pares = condicoes['alvo_pares']
    moldura = condicoes['alvo_moldura']
    rep = condicoes['alvo_repetidas']

    carteira = []
    tentativas = 0
    while len(carteira) < N_JOGOS and tentativas < 200000:
        jogo = gerar_jogo_condicionado(
            fixas, ultimo, rep, pares, moldura
        )
        tentativas += 1
        if jogo is None:
            continue
        if any(interseccao(jogo, j) > MAX_INTERSECAO for j in carteira):
            continue
        carteira.append(jogo)
    return carteira if len(carteira) == N_JOGOS else None

# ============================================================
# SIMULADOR DE SORTEIOS (BASE IID)
# ============================================================

class LotofacilSimulator:
    def __init__(self, historico):
        freq = np.zeros(26)
        for sorteio in historico:
            for d in sorteio:
                freq[d] += 1
        self.prob_dezenas = freq[1:] / len(historico)

    def gerar_sorteio(self):
        return sorted(np.random.choice(25, 15, replace=False, p=self.prob_dezenas) + 1)

    def simular(self, n=100000):
        return [self.gerar_sorteio() for _ in range(n)]

# ============================================================
# AVALIAÇÃO DE ESTRATÉGIA (MONTE CARLO)
# ============================================================

def avaliar_estrategia(condicoes, ultimo, simulador, n_sim=100000):
    carteira = gerar_carteira(condicoes, ultimo)
    if not carteira:
        return None

    sorteios = simulador.simular(n_sim)
    hits_max = [max(len(set(j) & set(s)) for j in carteira) for s in sorteios]

    premio_total = sum(PREMIO_VALORES.get(h, 0) for h in hits_max)
    custo = len(carteira) * 3.0
    roi = (premio_total / n_sim - custo) / custo

    return {
        'condicoes': condicoes,
        'media_hits': np.mean(hits_max),
        'freq_11': sum(1 for h in hits_max if h >= 11) / n_sim,
        'freq_12': sum(1 for h in hits_max if h >= 12) / n_sim,
        'freq_13': sum(1 for h in hits_max if h >= 13) / n_sim,
        'freq_14': sum(1 for h in hits_max if h == 14) / n_sim,
        'freq_15': sum(1 for h in hits_max if h == 15) / n_sim,
        'roi': roi,
        'hits_max': hits_max
    }

# ============================================================
# OTIMIZADOR DE COBERTURA
# ============================================================

def avaliar_cobertura(carteira):
    if not carteira:
        return -1.0, -1.0

    pares_cobertos = set()
    for j in carteira:
        for p in combinations(j, 2):
            pares_cobertos.add(p)
    cobertura = len(pares_cobertos) / 300.0

    intersecoes = [interseccao(a,b) for a,b in combinations(carteira,2)]
    if intersecoes:
        diversidade = 1.0 - np.mean(intersecoes)/15.0
    else:
        diversidade = 1.0
    return cobertura, diversidade

def otimizar_carteira(condicoes_base, ultimo, n_iter=5000):
    melhor_metrica = -np.inf
    melhor_carteira = None

    for _ in range(n_iter):
        pares = condicoes_base['alvo_pares'] + random.choice([-1, 0, 1])
        moldura = condicoes_base['alvo_moldura'] + random.choice([-1, 0, 1])
        rep = condicoes_base['alvo_repetidas'] + random.choice([-1, 0, 1])

        pares = max(2, min(13, pares))
        moldura = max(3, min(12, moldura))
        rep = max(3, min(12, rep))

        cond = {
            'fixas': condicoes_base.get('fixas', []),
            'alvo_pares': pares,
            'alvo_moldura': moldura,
            'alvo_repetidas': rep
        }
        cart = gerar_carteira(cond, ultimo)
        if not cart:
            continue

        cov, div = avaliar_cobertura(cart)
        metrica = cov * 0.6 + div * 0.4
        if metrica > melhor_metrica:
            melhor_metrica = metrica
            melhor_carteira = cart
            melhor_cond = cond

    return melhor_carteira, melhor_cond

# ============================================================
# WALK-FORWARD PARA ESTRATÉGIAS
# ============================================================

def walk_forward_estrategia(concursos, condicoes_base, window=30, passo=5):
    hits = []
    for inicio in range(0, len(concursos) - window - 1, passo):
        ultimo = concursos[inicio + window - 1]
        teste = concursos[inicio + window]
        carteira = gerar_carteira(condicoes_base, ultimo)
        if carteira:
            max_hits = max(interseccao(j, teste) for j in carteira)
            hits.append(max_hits)
    return hits

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================

def carregar_concursos(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
        print(f"⚠️  Arquivo {csv_file} não encontrado.")
        return None
    contests = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(';')
                if len(parts) < 17:
                    continue
                try:
                    dezenas = [int(x.strip()) for x in parts[2:17] if x.strip()]
                    if len(dezenas) != 15 or len(set(dezenas)) != 15:
                        continue
                    if any(x < 1 or x > 25 for x in dezenas):
                        continue
                    contests.append(sorted(dezenas))
                except:
                    continue
        print(f"✅ {len(contests)} concursos carregados")
        return contests
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return None

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():
    print("="*70)
    print("V37 — SIMULADOR DE ESTRATÉGIAS + OTIMIZADOR DE CARTEIRA")
    print("="*70)

    concursos = carregar_concursos()
    if not concursos:
        return

    simulador = LotofacilSimulator(concursos)

    estrategias = [
        {'alvo_pares': 8, 'alvo_moldura': 10, 'alvo_repetidas': 9, 'fixas': []},
        {'alvo_pares': 7, 'alvo_moldura': 9, 'alvo_repetidas': 8, 'fixas': []},
        {'alvo_pares': 6, 'alvo_moldura': 11, 'alvo_repetidas': 10, 'fixas': []},
        {'alvo_pares': 9, 'alvo_moldura': 10, 'alvo_repetidas': 9, 'fixas': []},
        {'alvo_pares': 7, 'alvo_moldura': 10, 'alvo_repetidas': 8, 'fixas': []},
    ]

    ultimo_real = concursos[-1]

    print("\n🔬 Avaliação de estratégias (Monte Carlo 50k simulações):")
    print("-" * 70)

    for i, cond in enumerate(estrategias, 1):
        res = avaliar_estrategia(cond, ultimo_real, simulador, n_sim=50000)
        if res:
            print(f"Estratégia {i}: pares={cond['alvo_pares']}, "
                  f"moldura={cond['alvo_moldura']}, rep={cond['alvo_repetidas']}")
            print(f"  Média hits: {res['media_hits']:.3f}  |  "
                  f"11+: {res['freq_11']:.3f}  12+: {res['freq_12']:.3f}  "
                  f"13+: {res['freq_13']:.3f}  ROI: {res['roi']:.3f}")

    print("\n🔧 Otimizando cobertura de pares e diversidade...")
    base = {'alvo_pares': 8, 'alvo_moldura': 10, 'alvo_repetidas': 9, 'fixas': []}
    melhor_carteira, melhores_cond = otimizar_carteira(base, ultimo_real, n_iter=2000)

    if melhor_carteira:
        cov, div = avaliar_cobertura(melhor_carteira)
        print(f"\nMelhor carteira encontrada (cobertura={cov:.3f}, diversidade={div:.3f}):")
        for i, jogo in enumerate(melhor_carteira, 1):
            print(f"  Jogo {i}: {jogo} | "
                  f"Pares={contar_pares(jogo)} Moldura={contar_moldura(jogo)} "
                  f"Rep={repetidas(jogo, ultimo_real)}")

    print("\n📊 Walk‑forward (últimas 200 janelas de 30 concursos)...")
    hits_wf = walk_forward_estrategia(concursos[-6000:], base, window=30, passo=5)
    if hits_wf:
        print(f"  Média de acertos máximos: {np.mean(hits_wf):.3f}  |  "
              f"11+: {sum(1 for h in hits_wf if h>=11)/len(hits_wf):.3f}  "
              f"12+: {sum(1 for h in hits_wf if h>=12)/len(hits_wf):.3f}")
    else:
        print("  (sem dados suficientes)")

    print("\n✅ V37 concluído.")

if __name__ == "__main__":
    main()
