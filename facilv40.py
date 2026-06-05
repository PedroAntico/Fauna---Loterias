#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v56
TESTE DEFINITIVO: PREVISÃO DIRETA DAS 15 DEZENAS + REGRESSÃO À MÉDIA

OBJETIVO:
✅ Experimento 1: Ranquear as 25 dezenas por frequência nos últimos N concursos,
   selecionar as 15 mais frequentes como previsão para o próximo concurso.
   Walk‑forward completo (3701 concursos). Comparar com baseline aleatório.
✅ Experimento 2: Testar regressão à média nos padrões estruturais
   (pares, moldura, primos). Se a média recente está acima da teórica,
   prever queda; se está abaixo, prever alta.
✅ Métricas: média de acertos, distribuição 11‑15, z‑score vs. aleatório.
✅ Responder: "A informação histórica melhora os acertos reais?"
"""

import numpy as np
from scipy.stats import hypergeom, binomtest
from collections import Counter
import os, random, time, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES_SET = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA_SET = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}

# Médias teóricas (esperadas pela distribuição hipergeométrica)
MEDIA_TEORICA_PARES = 15 * 12 / 25        # 7.2
MEDIA_TEORICA_MOLDURA = 15 * 15 / 25      # 9.0
MEDIA_TEORICA_PRIMOS = 15 * 9 / 25        # 5.4

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
# EXPERIMENTO 1: PREVISÃO DIRETA DAS 15 DEZENAS
# ============================================================
def experimento_previsao_direta(contests, historico=500, n_baseline=200):
    """
    Para cada concurso a partir de 'historico', usa os últimos 'historico'
    concursos para ranquear as dezenas por frequência, seleciona as 15 mais
    frequentes e compara com o resultado real.
    """
    print("\n" + "="*70)
    print("🔬 EXPERIMENTO 1: PREVISÃO DIRETA DAS 15 DEZENAS")
    print("="*70)
    print(f"   Histórico: {historico} concursos")
    print(f"   Baseline: {n_baseline} seleções aleatórias por concurso\n")

    n_total = len(contests)
    acertos_modelo = []
    acertos_baseline = []

    for t in tqdm(range(historico, n_total), desc="Walk‑forward"):
        # Dados de treino: últimos 'historico' concursos antes de t
        train = contests[t-historico:t]
        resultado_real = set(contests[t]['dezenas'])

        # Contar frequência das dezenas no treino
        freq = Counter()
        for c in train:
            freq.update(c['dezenas'])

        # Top 15 dezenas por frequência
        top15 = [d for d, _ in freq.most_common(15)]
        acertos = len(set(top15) & resultado_real)
        acertos_modelo.append(acertos)

        # Baseline: 15 dezenas aleatórias
        baseline_acertos = []
        for _ in range(n_baseline):
            aleatorias = random.sample(range(1, 26), 15)
            baseline_acertos.append(len(set(aleatorias) & resultado_real))
        acertos_baseline.append(np.mean(baseline_acertos))

    acertos_modelo = np.array(acertos_modelo)
    acertos_baseline = np.array(acertos_baseline)

    # Estatísticas
    media_modelo = np.mean(acertos_modelo)
    media_baseline = np.mean(acertos_baseline)
    std_baseline = np.std(acertos_baseline)
    z_score = (media_modelo - media_baseline) / (std_baseline / np.sqrt(len(acertos_baseline))) if std_baseline > 0 else 0.0

    # Distribuição de acertos (11+)
    dist_modelo = {k: 0 for k in range(11, 16)}
    dist_baseline = {k: 0 for k in range(11, 16)}
    for a in acertos_modelo:
        if a >= 11:
            dist_modelo[a] += 1
    # Para o baseline, usar a média das contagens
    # (aproximação: a média de acertos do baseline é usada para estimar a distribuição)
    for a in acertos_baseline:
        if a >= 11:
            dist_baseline[int(a)] += 1  # aproximação

    print(f"\n📊 RESULTADOS:")
    print(f"   Média de acertos (modelo): {media_modelo:.4f}")
    print(f"   Média de acertos (baseline): {media_baseline:.4f}")
    print(f"   Diferença: {media_modelo - media_baseline:+.4f}")
    print(f"   Z‑score: {z_score:.2f}")
    print(f"\n   Distribuição de acertos altos (modelo):")
    for k in range(11, 16):
        print(f"      {k} pontos: {dist_modelo[k]} vezes")
    print(f"\n   Distribuição de acertos altos (baseline, aprox.):")
    for k in range(11, 16):
        print(f"      {k} pontos: {dist_baseline[k]} vezes")

    if z_score > 2.0:
        print(f"\n🔍 Vantagem estatisticamente significativa (z > 2).")
    elif z_score > 1.0:
        print(f"\n📊 Pequena vantagem, mas não significativa.")
    else:
        print(f"\n📊 Desempenho indistinguível do aleatório.")

    return acertos_modelo, acertos_baseline, dist_modelo

# ============================================================
# EXPERIMENTO 2: REGRESSÃO À MÉDIA NOS PADRÕES ESTRUTURAIS
# ============================================================
def experimento_regressao_media(contests, janela_curta=20):
    """
    Testa se a média recente de pares/moldura/primos prevê a direção
    da média nos próximos concursos (reversão à média).
    """
    print("\n" + "="*70)
    print("🔬 EXPERIMENTO 2: REGRESSÃO À MÉDIA")
    print("="*70)
    print(f"   Janela curta: {janela_curta} concursos")
    print(f"   Hipótese: se média recente > teórica → próximo será menor (e vice‑versa)\n")

    # Extrair séries
    series = {'pares': [], 'moldura': [], 'primos': []}
    for c in contests:
        d = c['dezenas']
        series['pares'].append(sum(1 for x in d if x % 2 == 0))
        series['moldura'].append(sum(1 for x in d if x in MOLDURA_SET))
        series['primos'].append(sum(1 for x in d if x in PRIMES_SET))

    medias_teoricas = {
        'pares': MEDIA_TEORICA_PARES,
        'moldura': MEDIA_TEORICA_MOLDURA,
        'primos': MEDIA_TEORICA_PRIMOS
    }

    for nome, serie in series.items():
        serie = np.array(serie, dtype=float)
        n = len(serie)
        acertos = 0
        total = 0

        for t in range(janela_curta, n - 1):
            media_recente = np.mean(serie[t-janela_curta:t])
            media_teorica = medias_teoricas[nome]
            valor_futuro = serie[t+1]

            # Previsão: se recente > teórica, espera valor futuro < média recente
            if media_recente > media_teorica:
                previsao_queda = True
            else:
                previsao_queda = False

            total += 1
            if previsao_queda and valor_futuro < media_recente:
                acertos += 1
            elif not previsao_queda and valor_futuro > media_recente:
                acertos += 1

        acuracia = acertos / total * 100 if total > 0 else 50.0
        p_value = binomtest(acertos, total, 0.5, alternative='greater').pvalue if total > 0 else 1.0

        print(f"   {nome:<10}: Acurácia={acuracia:.1f}% ({acertos}/{total}), p={p_value:.4f}", end="")
        if p_value < 0.05:
            print(" 🔍 Significativo")
        else:
            print("")

    print(f"\n📊 Interpretação: se p > 0.05 para todos, a regressão à média")
    print(f"   não fornece vantagem preditiva significativa nos padrões testados.")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v56")
    print("   TESTE DEFINITIVO: PREVISÃO DIRETA + REGRESSÃO À MÉDIA")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    while True:
        print("\nOpções:")
        print("1. Experimento 1: Previsão direta das 15 dezenas")
        print("2. Experimento 2: Regressão à média nos padrões estruturais")
        print("3. Ambos os experimentos")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            hist = int(input("   Tamanho do histórico [500]: ").strip() or "500")
            experimento_previsao_direta(contests, historico=hist)

        elif op == '2':
            janela = int(input("   Tamanho da janela curta [20]: ").strip() or "20")
            experimento_regressao_media(contests, janela_curta=janela)

        elif op == '3':
            hist = int(input("   Tamanho do histórico [500]: ").strip() or "500")
            janela = int(input("   Tamanho da janela curta [20]: ").strip() or "20")
            experimento_previsao_direta(contests, historico=hist)
            experimento_regressao_media(contests, janela_curta=janela)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
