#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V34 — DRIFT LOCAL CONDICIONAL (HIPÓTESE FALSIFICÁVEL)

Protocolo congelado (sem tuning posterior):
✅ Janela fixa: 30 concursos
✅ Features locais: pares, moldura, repetição, primos
✅ Fitness: log-likelihood ratio (local vs global)
✅ Geração: 100k jogos aleatórios puros (apenas filtro mínimo de estrutura)
✅ Seleção: top 5 scores com interseção ≤ 10
✅ Teste: walk-forward temporal real (compara com aleatório)
✅ Métrica: Wilcoxon sobre a diferença de hits máximos
"""

import numpy as np
from scipy.stats import wilcoxon
from collections import Counter
from itertools import combinations
import random
import os
from math import comb
from tqdm import tqdm
import time

# ============================================================
# CONFIGURAÇÕES FIXAS (CONGELADAS)
# ============================================================

WINDOW = 30                # tamanho da janela local
N_JOGOS = 5                # número de jogos na carteira
N_CANDIDATOS = 100_000     # jogos aleatórios gerados por janela
MAX_INTERSECAO = 10        # sobreposição máxima permitida entre jogos da carteira
EPS = 1e-6                 # suavização para log

# Features monitoradas e suas funções de extração
def pares_count(game):
    return sum(1 for d in game if d % 2 == 0)

def moldura_count(game):
    MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
    return sum(1 for d in game if d in MOLDURA)

def primos_count(game):
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
    return sum(1 for d in game if d in PRIMES)

def repeticao_count(game, ultimo_sorteio):
    return len(set(game) & set(ultimo_sorteio))

FEATURE_FUNCS = {
    'pares': pares_count,
    'moldura': moldura_count,
    'primos': primos_count,
    # 'repeticao' é tratada separadamente porque depende do último sorteio
}

# ============================================================
# UTILITÁRIOS
# ============================================================

def gerar_jogo_aleatorio():
    """Gera um jogo completamente aleatório, com filtro mínimo de estrutura."""
    while True:
        jogo = sorted(np.random.choice(range(1, 26), 15, replace=False))
        # Filtro leve: no máximo 3 consecutivos em sequência
        max_run = 1
        run = 1
        for i in range(len(jogo)-1):
            if jogo[i+1] - jogo[i] == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run <= 3:
            # Clusterização simples (gaps <= 2)
            gaps = [jogo[i+1]-jogo[i] for i in range(len(jogo)-1)]
            cluster = sum(1 for g in gaps if g <= 2) / len(gaps)
            if cluster < 0.9:
                return jogo

def interseccao(jogo1, jogo2):
    return len(set(jogo1) & set(jogo2))

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================

def carregar_concursos(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    if not os.path.exists(csv_path):
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
        print(f"❌ Erro: {e}")
        return None

# ============================================================
# MODELO DE DRIFT LOCAL
# ============================================================

def calcular_frequencias_locais(historico_local):
    """Calcula a distribuição empírica das features na janela local."""
    freq = {feat: Counter() for feat in FEATURE_FUNCS}
    freq['repeticao'] = Counter()
    
    for i, jogo in enumerate(historico_local):
        for feat, func in FEATURE_FUNCS.items():
            valor = func(jogo)
            freq[feat][valor] += 1
        if i > 0:
            rep = repeticao_count(jogo, historico_local[i-1])
            freq['repeticao'][rep] += 1
    
    # Normaliza para probabilidades
    total = len(historico_local)
    for feat in freq:
        for v in freq[feat]:
            freq[feat][v] /= total
    return freq

def calcular_frequencias_globais(historico_global):
    """Calcula a distribuição empírica das features no histórico global."""
    freq = {feat: Counter() for feat in FEATURE_FUNCS}
    freq['repeticao'] = Counter()
    
    for i, jogo in enumerate(historico_global):
        for feat, func in FEATURE_FUNCS.items():
            valor = func(jogo)
            freq[feat][valor] += 1
        if i > 0:
            rep = repeticao_count(jogo, historico_global[i-1])
            freq['repeticao'][rep] += 1
    
    total = len(historico_global)
    for feat in freq:
        for v in freq[feat]:
            freq[feat][v] /= total
    return freq

def score_jogo(jogo, freq_local, freq_global, ultimo_sorteio=None):
    """Log-likelihood ratio de um jogo em relação às distribuições local/global."""
    score = 0.0
    for feat, func in FEATURE_FUNCS.items():
        valor = func(jogo)
        p_local = freq_local[feat].get(valor, EPS)
        p_global = freq_global[feat].get(valor, EPS)
        score += np.log((p_local + EPS) / (p_global + EPS))
    if ultimo_sorteio:
        rep = repeticao_count(jogo, ultimo_sorteio)
        p_local = freq_local['repeticao'].get(rep, EPS)
        p_global = freq_global['repeticao'].get(rep, EPS)
        score += np.log((p_local + EPS) / (p_global + EPS))
    return score

# ============================================================
# SELEÇÃO DE CARTEIRA
# ============================================================

def selecionar_carteira(freq_local, freq_global, ultimo_sorteio, n_candidatos=100000):
    """Gera N jogos aleatórios e seleciona os 5 melhores com restrição de interseção."""
    candidatos = []
    scores = []
    for _ in range(n_candidatos):
        jogo = gerar_jogo_aleatorio()
        s = score_jogo(jogo, freq_local, freq_global, ultimo_sorteio)
        candidatos.append(jogo)
        scores.append(s)
    
    # Ordena por score decrescente
    indices = np.argsort(scores)[::-1]
    selecionados = []
    for idx in indices:
        jogo = candidatos[idx]
        if all(interseccao(jogo, sel) <= MAX_INTERSECAO for sel in selecionados):
            selecionados.append(jogo)
        if len(selecionados) == N_JOGOS:
            break
    return selecionados

# ============================================================
# BACKTEST WALK-FORWARD
# ============================================================

def walk_forward_test(concursos, n_testes=100):
    """
    Realiza walk-forward:
    - Para cada janela de 30 concursos, gera carteira condicionada e aleatória
    - Testa no concurso seguinte (janela deslizante de 1 em 1)
    - Retorna listas de acertos máximos para estratégia e aleatório
    """
    estrategia_hits = []
    aleatorio_hits = []
    
    # Para cada janela de WINDOW concursos, testa no próximo
    for inicio in tqdm(range(0, len(concursos) - WINDOW - 1, max(1, (len(concursos) - WINDOW - 1) // n_testes)), desc="Walk-forward"):
        if len(estrategia_hits) >= n_testes:
            break
            
        fim = inicio + WINDOW
        janela_local = concursos[inicio:fim]
        historico_global = concursos[:inicio]  # tudo antes da janela
        
        if len(historico_global) < 50:  # precisa de histórico global mínimo
            continue
        
        teste = concursos[fim]  # próximo sorteio
        
        # Calcula frequências
        freq_local = calcular_frequencias_locais(janela_local)
        freq_global = calcular_frequencias_globais(historico_global)
        ultimo = janela_local[-1]
        
        # Carteira da estratégia
        carteira_estrategia = selecionar_carteira(freq_local, freq_global, ultimo, N_CANDIDATOS)
        
        # Carteira aleatória pura (para comparação)
        carteira_aleatoria = [gerar_jogo_aleatorio() for _ in range(N_JOGOS)]
        
        # Calcula acertos máximos
        hits_estrategia = max(interseccao(jogo, teste) for jogo in carteira_estrategia)
        hits_aleatorio = max(interseccao(jogo, teste) for jogo in carteira_aleatoria)
        
        estrategia_hits.append(hits_estrategia)
        aleatorio_hits.append(hits_aleatorio)
    
    return estrategia_hits, aleatorio_hits

# ============================================================
# ANÁLISE ESTATÍSTICA
# ============================================================

def analisar_resultados(estrategia_hits, aleatorio_hits):
    """Compara as distribuições de acertos usando Wilcoxon e métricas descritivas."""
    print("\n" + "="*60)
    print("RESULTADOS DO EXPERIMENTO")
    print("="*60)
    
    # Métricas descritivas
    print(f"\nNúmero de testes: {len(estrategia_hits)}")
    print(f"Média de hits (estratégia): {np.mean(estrategia_hits):.3f}")
    print(f"Média de hits (aleatório): {np.mean(aleatorio_hits):.3f}")
    print(f"Desvio padrão (estratégia): {np.std(estrategia_hits):.3f}")
    print(f"Desvio padrão (aleatório): {np.std(aleatorio_hits):.3f}")
    
    # Distribuições
    for hits in [11, 12, 13, 14, 15]:
        freq_est = sum(1 for h in estrategia_hits if h >= hits) / len(estrategia_hits)
        freq_ale = sum(1 for h in aleatorio_hits if h >= hits) / len(aleatorio_hits)
        print(f"Freq. {hits}+: estratégia={freq_est:.4f} | aleatório={freq_ale:.4f}")
    
    # Wilcoxon signed-rank test
    diffs = [e - a for e, a in zip(estrategia_hits, aleatorio_hits)]
    try:
        stat, p = wilcoxon(diffs)
        print(f"\nWilcoxon signed-rank: estatística={stat:.3f}, p-value={p:.4f}")
        if p < 0.05 and np.mean(diffs) > 0.02:
            print("✅ Hipótese confirmada: estratégia supera aleatório com significância estatística.")
        else:
            print("❌ Hipótese NÃO confirmada: não há evidência suficiente de vantagem.")
    except Exception as e:
        print(f"\nErro no teste Wilcoxon: {e}")
    
    return diffs, p if 'p' in locals() else 1.0

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():
    print("="*60)
    print("V34 — DRIFT LOCAL CONDICIONAL (HIPÓTESE FALSIFICÁVEL)")
    print("="*60)
    print(f"Configurações congeladas:")
    print(f"  Janela: {WINDOW} concursos")
    print(f"  Features: pares, moldura, repetição, primos")
    print(f"  Fitness: log-likelihood ratio")
    print(f"  Candidatos: {N_CANDIDATOS:,} jogos aleatórios")
    print(f"  Carteira: {N_JOGOS} jogos (interseção ≤ {MAX_INTERSECAO})")
    
    # Carrega dados
    concursos = carregar_concursos('resultados_lotofacil.csv')
    if concursos is None:
        print("❌ Arquivo de dados não encontrado.")
        return
    
    # Executa walk-forward
    print(f"\n🔍 Executando walk-forward...")
    t0 = time.time()
    estrategia_hits, aleatorio_hits = walk_forward_test(concursos, n_testes=100)
    print(f"⏱️ Tempo: {time.time()-t0:.1f}s")
    
    # Analisa resultados
    diffs, p_valor = analisar_resultados(estrategia_hits, aleatorio_hits)
    
    # Conclusão final
    print("\n" + "="*60)
    print("CONCLUSÃO")
    print("="*60)
    if p_valor < 0.05 and np.mean(diffs) > 0.02:
        print("Há evidência estatística de drift local explorável.")
        print("Recomendação: investigar com mais features ou janelas adaptativas.")
    else:
        print("NÃO há evidência de drift local explorável com este protocolo.")
        print("As distribuições condicionais recentes não persistem o suficiente.")
        print("Isso sugere que a Lotofácil se comporta como um processo IID,")
        print("ou que os desvios têm meia-vida muito curta para serem aproveitados.")
    print("="*60)

if __name__ == "__main__":
    main()
