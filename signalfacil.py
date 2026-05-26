#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V34.1 — DRIFT LOCAL CONDICIONAL (OTIMIZADO)

Otimizações:
✅ N_CANDIDATOS reduzido para 10.000 (suficiente para baixa dimensionalidade)
✅ Pool de features pré-computadas (pares, primos, moldura)
✅ Repetição como único termo dinâmico
✅ Seleção por amostragem ponderada (softmax) em vez de top-5 duro
✅ Sem logs — usa razão simples para ranking
✅ Execução viável em Pydroid/Colab
"""

import numpy as np
from scipy.stats import wilcoxon
from collections import Counter
import random
import os
from tqdm import tqdm
import time

# ============================================================
# CONFIGURAÇÕES FIXAS (CONGELADAS)
# ============================================================

WINDOW = 30
N_JOGOS = 5
N_CANDIDATOS = 10_000      # reduzido — suficiente para 4 features discretas
MAX_INTERSECAO = 10
EPS = 1e-6

MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

def features_jogo(jogo):
    """Extrai features de um jogo (retorna tupla imutável)."""
    pares = sum(1 for d in jogo if d % 2 == 0)
    moldura = sum(1 for d in jogo if d in MOLDURA)
    primos = sum(1 for d in jogo if d in PRIMES)
    return pares, moldura, primos

def repeticao(jogo, ultimo):
    return len(set(jogo) & set(ultimo))

def gerar_jogo_aleatorio():
    while True:
        jogo = sorted(np.random.choice(range(1, 26), 15, replace=False))
        max_run = 1
        run = 1
        for i in range(len(jogo)-1):
            if jogo[i+1] - jogo[i] == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run <= 3:
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
        print(f"✅ {len(contests)} concursos")
        return contests
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

# ============================================================
# POOL DE JOGOS PRÉ-COMPUTADO
# ============================================================

def criar_pool(tamanho=20000):
    """Gera um pool fixo de jogos com features pré-calculadas."""
    print(f"🔧 Criando pool de {tamanho} jogos...")
    pool = []
    for _ in tqdm(range(tamanho), desc="Pool"):
        jogo = gerar_jogo_aleatorio()
        pares, moldura, primos = features_jogo(jogo)
        pool.append({
            'jogo': jogo,
            'pares': pares,
            'moldura': moldura,
            'primos': primos,
        })
    return pool

# ============================================================
# MODELO DE DRIFT LOCAL
# ============================================================

def calcular_frequencias_locais(janela, ultimo):
    """Frequências das features na janela local."""
    freq = {
        'pares': Counter(),
        'moldura': Counter(),
        'primos': Counter(),
        'repeticao': Counter(),
    }
    for i, jogo in enumerate(janela):
        p, m, pr = features_jogo(jogo)
        freq['pares'][p] += 1
        freq['moldura'][m] += 1
        freq['primos'][pr] += 1
        if i > 0:
            freq['repeticao'][repeticao(jogo, janela[i-1])] += 1
    total = len(janela)
    for k in freq:
        for v in freq[k]:
            freq[k][v] /= total
    return freq

def calcular_frequencias_globais(historico):
    freq = {
        'pares': Counter(),
        'moldura': Counter(),
        'primos': Counter(),
        'repeticao': Counter(),
    }
    for i, jogo in enumerate(historico):
        p, m, pr = features_jogo(jogo)
        freq['pares'][p] += 1
        freq['moldura'][m] += 1
        freq['primos'][pr] += 1
        if i > 0:
            freq['repeticao'][repeticao(jogo, historico[i-1])] += 1
    total = len(historico)
    for k in freq:
        for v in freq[k]:
            freq[k][v] /= total
    return freq

def score_jogo(item_pool, freq_local, freq_global, ultimo):
    """Score = soma das razões local/global (sem log)."""
    score = 0.0
    for feat in ['pares', 'moldura', 'primos']:
        valor = item_pool[feat]
        p_local = freq_local[feat].get(valor, EPS)
        p_global = freq_global[feat].get(valor, EPS)
        score += p_local / (p_global + EPS)
    rep = repeticao(item_pool['jogo'], ultimo)
    p_local = freq_local['repeticao'].get(rep, EPS)
    p_global = freq_global['repeticao'].get(rep, EPS)
    score += p_local / (p_global + EPS)
    return score

# ============================================================
# SELEÇÃO DE CARTEIRA (AMOSTRAGEM PONDERADA)
# ============================================================

def selecionar_carteira(pool, freq_local, freq_global, ultimo, n_candidatos=10000):
    # Amostra aleatória do pool
    indices = np.random.choice(len(pool), size=min(n_candidatos, len(pool)), replace=False)
    candidatos = [(idx, score_jogo(pool[idx], freq_local, freq_global, ultimo)) for idx in indices]
    
    # Ordena por score
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    # Seleciona os N_CANDIDATOS melhores
    top = candidatos[:n_candidatos]
    scores = np.array([s for _, s in top])
    
    # Softmax para amostragem ponderada
    scores_adj = scores - np.max(scores)
    probs = np.exp(scores_adj / 0.5)  # temperatura 0.5
    probs /= probs.sum()
    
    selecionados = []
    selecionados_jogos = []
    tentativas = 0
    while len(selecionados) < N_JOGOS and tentativas < 500:
        idx_pool = np.random.choice(len(top), p=probs)
        jogo = pool[top[idx_pool][0]]['jogo']
        if all(interseccao(jogo, s) <= MAX_INTERSECAO for s in selecionados_jogos):
            selecionados.append(jogo)
            selecionados_jogos.append(jogo)
        tentativas += 1
    
    # Fallback: se não conseguir 5, completa com os melhores disponíveis
    for idx_sorted, _ in top:
        jogo = pool[idx_sorted]['jogo']
        if len(selecionados) >= N_JOGOS:
            break
        if all(interseccao(jogo, s) <= MAX_INTERSECAO for s in selecionados_jogos):
            selecionados.append(jogo)
            selecionados_jogos.append(jogo)
    
    return selecionados

# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward_test(concursos, pool, n_testes=50):
    estrategia_hits = []
    aleatorio_hits = []
    indices_testados = []
    
    passo = max(1, (len(concursos) - WINDOW - 1) // n_testes)
    
    for inicio in tqdm(range(0, len(concursos) - WINDOW - 1, passo), desc="Walk-forward"):
        if len(estrategia_hits) >= n_testes:
            break
        
        fim = inicio + WINDOW
        janela_local = concursos[inicio:fim]
        historico_global = concursos[:inicio]
        
        if len(historico_global) < 50:
            continue
        
        teste = concursos[fim]
        ultimo = janela_local[-1]
        
        freq_local = calcular_frequencias_locais(janela_local, ultimo)
        freq_global = calcular_frequencias_globais(historico_global)
        
        carteira_estrategia = selecionar_carteira(pool, freq_local, freq_global, ultimo)
        carteira_aleatoria = [gerar_jogo_aleatorio() for _ in range(N_JOGOS)]
        
        hits_est = max(interseccao(j, teste) for j in carteira_estrategia)
        hits_ale = max(interseccao(j, teste) for j in carteira_aleatoria)
        
        estrategia_hits.append(hits_est)
        aleatorio_hits.append(hits_ale)
        indices_testados.append(fim)
    
    return estrategia_hits, aleatorio_hits, indices_testados

# ============================================================
# ANÁLISE
# ============================================================

def analisar(est, ale, indices):
    print("\n" + "="*60)
    print("RESULTADOS V34.1")
    print("="*60)
    print(f"Testes realizados: {len(est)}")
    print(f"Média hits estratégia: {np.mean(est):.3f} ± {np.std(est):.3f}")
    print(f"Média hits aleatório:   {np.mean(ale):.3f} ± {np.std(ale):.3f}")
    
    for h in [11, 12, 13, 14, 15]:
        fe = sum(1 for x in est if x >= h) / len(est)
        fa = sum(1 for x in ale if x >= h) / len(ale)
        print(f"  {h}+ pts: estratégia={fe:.4f} | aleatório={fa:.4f}")
    
    diffs = [e - a for e, a in zip(est, ale)]
    try:
        stat, p = wilcoxon(diffs)
        print(f"\nWilcoxon: p={p:.4f}")
        if p < 0.05 and np.mean(diffs) > 0.02:
            print("✅ Hipótese CONFIRMADA")
        else:
            print("❌ Hipótese NÃO confirmada")
    except:
        p = 1.0
        print("Wilcoxon não pôde ser calculado")
    
    return diffs, p

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("V34.1 — DRIFT LOCAL (OTIMIZADO)")
    print("="*60)
    
    concursos = carregar_concursos()
    if concursos is None:
        return
    
    pool = criar_pool(20000)
    
    t0 = time.time()
    est, ale, idx = walk_forward_test(concursos, pool, n_testes=50)
    print(f"\n⏱️ Tempo: {time.time()-t0:.1f}s")
    
    diffs, p = analisar(est, ale, idx)
    
    print("\n" + "="*60)
    print("CONCLUSÃO")
    print("="*60)
    if p < 0.05 and np.mean(diffs) > 0.02:
        print("Evidência de drift local explorável.")
    else:
        print("SEM evidência de drift local explorável.")
        print("A Lotofácil comporta-se como IID nas condições testadas.")
    print("="*60)

if __name__ == "__main__":
    main()
