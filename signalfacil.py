#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V36 — GERADOR CONDICIONAL DE CARTEIRA

Objetivo:
Gerar 5 jogos condicionados por:
✅ dezenas fixas
✅ repetidas do último concurso
✅ quantidade de pares
✅ quantidade de moldura

Estratégia:
- Geração puramente combinatória
- Sem Mahalanobis
- Sem KDE
- Sem "naturalidade"
- Busca por cobertura e equilíbrio

Critério:
- Jogos com baixa sobreposição entre si
- Maximizar cobertura de dezenas
- Maximizar chance de múltiplas premiações menores
"""

import numpy as np
import random
from itertools import combinations

# ============================================================
# CONFIGURAÇÕES
# ============================================================

N_JOGOS = 5
MAX_INTERSECAO = 10

MOLDURA = {
    1,2,3,4,5,
    6,10,
    11,15,
    16,20,
    21,22,23,24,25
}

# ============================================================
# FUNÇÕES
# ============================================================

def contar_pares(jogo):
    return sum(1 for x in jogo if x % 2 == 0)

def contar_moldura(jogo):
    return sum(1 for x in jogo if x in MOLDURA)

def repetidas(jogo, ultimo):
    return len(set(jogo) & set(ultimo))

def interseccao(a, b):
    return len(set(a) & set(b))

def gerar_jogo_condicionado(
    fixas,
    ultimo_concurso,
    alvo_repetidas,
    alvo_pares,
    alvo_moldura,
    tentativas=50000
):
    """
    Gera um único jogo obedecendo às restrições.
    """

    fixas = sorted(set(fixas))

    if len(fixas) > 15:
        return None

    universo = set(range(1, 26))

    repetidas_fixas = len(set(fixas) & set(ultimo_concurso))

    # Quantas repetidas ainda faltam
    faltam_rep = alvo_repetidas - repetidas_fixas

    if faltam_rep < 0:
        return None

    dezenas_repetidas = list(set(ultimo_concurso) - set(fixas))
    dezenas_novas = list(universo - set(ultimo_concurso) - set(fixas))

    for _ in range(tentativas):

        jogo = set(fixas)

        # Completa repetidas
        if faltam_rep > 0:
            escolhidas_rep = random.sample(
                dezenas_repetidas,
                min(faltam_rep, len(dezenas_repetidas))
            )
            jogo.update(escolhidas_rep)

        # Completa restante
        faltam = 15 - len(jogo)

        if faltam < 0:
            continue

        restantes = list(universo - jogo)

        escolhidas = random.sample(restantes, faltam)

        jogo.update(escolhidas)

        jogo = sorted(jogo)

        # Valida
        if len(jogo) != 15:
            continue

        if contar_pares(jogo) != alvo_pares:
            continue

        if contar_moldura(jogo) != alvo_moldura:
            continue

        if repetidas(jogo, ultimo_concurso) != alvo_repetidas:
            continue

        return jogo

    return None

# ============================================================
# GERADOR DE CARTEIRA
# ============================================================

def gerar_carteira(
    fixas,
    ultimo_concurso,
    alvo_repetidas,
    alvo_pares,
    alvo_moldura
):

    carteira = []

    cobertura = set()

    tentativas = 0

    while len(carteira) < N_JOGOS and tentativas < 200000:

        jogo = gerar_jogo_condicionado(
            fixas,
            ultimo_concurso,
            alvo_repetidas,
            alvo_pares,
            alvo_moldura
        )

        tentativas += 1

        if jogo is None:
            continue

        # Evita jogos muito parecidos
        if any(interseccao(jogo, j) > MAX_INTERSECAO for j in carteira):
            continue

        carteira.append(jogo)

        cobertura.update(jogo)

    return carteira, cobertura

# ============================================================
# MAIN
# ============================================================

def main():

    print("="*60)
    print("V36 — GERADOR CONDICIONAL")
    print("="*60)

    # ============================================
    # INPUTS
    # ============================================

    ultimo = input(
        "\nÚltimo concurso (15 dezenas separadas por espaço):\n> "
    )

    ultimo = sorted([int(x) for x in ultimo.split()])

    fixas = input(
        "\nDezenas fixas (separadas por espaço):\n> "
    )

    if fixas.strip():
        fixas = sorted([int(x) for x in fixas.split()])
    else:
        fixas = []

    alvo_repetidas = int(input("\nQtd repetidas desejada: "))
    alvo_pares = int(input("Qtd pares desejada: "))
    alvo_moldura = int(input("Qtd moldura desejada: "))

    # ============================================
    # GERAÇÃO
    # ============================================

    carteira, cobertura = gerar_carteira(
        fixas,
        ultimo,
        alvo_repetidas,
        alvo_pares,
        alvo_moldura
    )

    # ============================================
    # RESULTADOS
    # ============================================

    print("\n" + "="*60)
    print("CARTEIRA GERADA")
    print("="*60)

    if not carteira:
        print("\n❌ Nenhum jogo encontrado.")
        print("Tente relaxar as restrições.")
        return

    for i, jogo in enumerate(carteira, 1):

        print(f"\nJogo {i}")
        print(jogo)

        print(
            f"Pares={contar_pares(jogo)} | "
            f"Moldura={contar_moldura(jogo)} | "
            f"Repetidas={repetidas(jogo, ultimo)}"
        )

    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)

    print(f"Cobertura total das dezenas: {len(cobertura)} / 25")
    print(f"Dezenas cobertas: {sorted(cobertura)}")

    media_intersecao = []

    for a, b in combinations(carteira, 2):
        media_intersecao.append(interseccao(a, b))

    if media_intersecao:
        print(
            f"Interseção média entre jogos: "
            f"{np.mean(media_intersecao):.2f}"
        )

    print("\n✅ Carteira finalizada.")

# ============================================================

if __name__ == "__main__":
    main()
