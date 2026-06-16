#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v65‑lite
TESTE RÁPIDO DE EFICIÊNCIA DA HEURÍSTICA DE FIXAS

OBJETIVO:
✅ Testar apenas a seleção de fixas (sem pair covering, sem semifixas)
✅ Heurística: score = frequência recente + peso * (1/(atraso+1))
✅ Walk‑forward nos últimos 300 concursos
✅ Comparar acertos das fixas escolhidas com 100 conjuntos aleatórios
✅ Métricas: média de acertos, z‑score, % concursos com vantagem
"""

import numpy as np
from collections import Counter
import os, random, time, warnings
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
# HEURÍSTICA DE SELEÇÃO DE FIXAS
# ============================================================
def selecionar_fixas_heuristicas(contests, n_fixas=4, janela_freq=50, peso_atraso=0.3):
    """
    Retorna as top n_fixas dezenas segundo a heurística:
        score[d] = freq_recente[d] + peso_atraso / (atraso[d] + 1)
    onde freq_recente é a proporção nos últimos janela_freq concursos,
    e atraso é o número de concursos desde a última aparição.
    """
    if len(contests) == 0:
        return []
    total = len(contests)
    recent = contests[-janela_freq:] if janela_freq < total else contests
    
    # Frequência recente
    freq_recente = Counter()
    for c in recent:
        freq_recente.update(c['dezenas'])
    for d in range(1, 26):
        freq_recente[d] = freq_recente.get(d, 0) / len(recent)
    
    # Atraso
    last_seen = {d: -1 for d in range(1, 26)}
    for i, c in enumerate(contests):
        for d in c['dezenas']:
            last_seen[d] = i
    atraso = {}
    for d in range(1, 26):
        atraso[d] = (total - 1) - last_seen[d]  # concursos desde última aparição
    
    # Score
    scores = {}
    for d in range(1, 26):
        scores[d] = freq_recente[d] + peso_atraso / (atraso[d] + 1)
    
    # Top n_fixas
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:n_fixas]]

# ============================================================
# TESTE DE EFICIÊNCIA DAS FIXAS (LITE)
# ============================================================
def testar_eficiencia_fixas_lite(contests, n_fixas=4, janela_freq=50, peso_atraso=0.3,
                                 n_random=100, n_ultimos=300):
    """
    Walk‑forward nos últimos n_ultimos concursos.
    Para cada concurso:
      - Seleciona n_fixas com a heurística
      - Conta acertos (quantas das fixas estão no sorteio real)
      - Compara com a média de acertos de n_random conjuntos aleatórios de n_fixas
    """
    start = len(contests) - n_ultimos
    if start < 0:
        start = 0
    
    print(f"\n{'='*70}")
    print(f"🔬 TESTE RÁPIDO DE EFICIÊNCIA DE FIXAS (v65‑lite)")
    print(f"{'='*70}")
    print(f"   Fixas por concurso: {n_fixas}")
    print(f"   Heurística: freq recente ({janela_freq} conc.) + atraso (peso {peso_atraso})")
    print(f"   Comparações aleatórias: {n_random} por concurso")
    print(f"   Concursos testados: {len(contests) - start} (últimos {n_ultimos})\n")
    
    acertos_heuristicos = []
    acertos_medios_random = []
    z_scores = []
    
    for t in tqdm(range(start, len(contests)), desc="Walk‑forward"):
        treino = contests[:t]
        teste = contests[t]
        
        # Heurística
        fixas_heu = selecionar_fixas_heuristicas(treino, n_fixas, janela_freq, peso_atraso)
        acerto_heu = len(set(fixas_heu) & set(teste['dezenas']))
        acertos_heuristicos.append(acerto_heu)
        
        # Aleatórios
        acertos_rand = []
        for _ in range(n_random):
            fixas_rand = random.sample(range(1, 26), n_fixas)
            acertos_rand.append(len(set(fixas_rand) & set(teste['dezenas'])))
        acertos_rand = np.array(acertos_rand)
        media_rand = np.mean(acertos_rand)
        std_rand = np.std(acertos_rand)
        acertos_medios_random.append(media_rand)
        if std_rand > 0:
            z_scores.append((acerto_heu - media_rand) / std_rand)
        else:
            z_scores.append(0.0)
    
    acertos_heuristicos = np.array(acertos_heuristicos)
    acertos_medios_random = np.array(acertos_medios_random)
    z_scores = np.array(z_scores)
    
    # Estatísticas
    media_heu = np.mean(acertos_heuristicos)
    media_rand = np.mean(acertos_medios_random)
    prop_positivo = np.mean(z_scores > 0) * 100
    
    print(f"\n📊 RESULTADOS ({len(acertos_heuristicos)} concursos):")
    print(f"   Média de acertos (heurística): {media_heu:.3f}")
    print(f"   Média de acertos (aleatório):   {media_rand:.3f}")
    print(f"   Diferença: {media_heu - media_rand:+.3f}")
    print(f"   Z‑score médio: {np.mean(z_scores):.3f}")
    print(f"   % concursos com z > 0: {prop_positivo:.1f}%")
    print(f"   Máximo z‑score: {np.max(z_scores):.3f}, Mínimo: {np.min(z_scores):.3f}")
    
    # Interpretação
    print(f"\n📊 INTERPRETAÇÃO:")
    if np.mean(z_scores) > 2.0:
        print("   🔍 A heurística mostra desempenho significativamente superior ao acaso.")
    elif np.mean(z_scores) > 0.5:
        print("   📊 A heurística parece levemente melhor, mas não de forma robusta.")
    else:
        print("   ✅ Desempenho indistinguível do aleatório.")
    print("   (Lembre‑se: a heurística atual premia dezenas recém‑sorteadas, não atrasadas.)")
    
    return acertos_heuristicos, acertos_medios_random, z_scores

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v65‑lite")
    print("   TESTE RÁPIDO DE FIXAS")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # Parâmetros configuráveis
    print("\n⚙️ Parâmetros (ENTER para padrão):")
    try:
        n_fixas = int(input("   Número de fixas [4]: ").strip() or "4")
        janela = int(input("   Janela de frequência recente [50]: ").strip() or "50")
        peso = float(input("   Peso do atraso [0.3]: ").strip() or "0.3")
        n_random = int(input("   Conjuntos aleatórios por concurso [100]: ").strip() or "100")
        n_ultimos = int(input("   Últimos concursos a testar [300]: ").strip() or "300")
    except ValueError:
        print("   Valor inválido. Usando padrões.")
        n_fixas, janela, peso, n_random, n_ultimos = 4, 50, 0.3, 100, 300

    testar_eficiencia_fixas_lite(contests, n_fixas, janela, peso, n_random, n_ultimos)

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
