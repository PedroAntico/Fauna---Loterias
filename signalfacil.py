#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V34.3 — MICRO‑DRIFT LOCAL (JANELAS 5, 10, 20)

Último teste legítimo:
✅ Três janelas testadas: 5, 10, 20
✅ Mesmo protocolo congelado: features fixas, pool, scoring, amostragem
✅ 300 testes por janela
✅ Se todos p > 0.05 → encerra‑se a investigação honestamente
"""

import numpy as np
from scipy.stats import wilcoxon
from collections import Counter
import os
from tqdm import tqdm
import time
import random

# ============================================================
# CONFIGURAÇÕES FIXAS (CONGELADAS)
# ============================================================

N_JOGOS = 5
N_CANDIDATOS = 10_000
MAX_INTERSECAO = 10
EPS = 1e-6
N_TESTES = 300

MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

def features_jogo(jogo):
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
# POOL PRÉ-COMPUTADO
# ============================================================

def criar_pool(tamanho=20000):
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
# FREQUÊNCIAS
# ============================================================

def calcular_frequencias_locais(janela):
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

# ============================================================
# SCORE E SELEÇÃO
# ============================================================

def score_jogo(item_pool, freq_local, freq_global, ultimo):
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

def selecionar_carteira(pool, freq_local, freq_global, ultimo):
    indices = np.random.choice(len(pool), size=N_CANDIDATOS, replace=False)
    candidatos = [(idx, score_jogo(pool[idx], freq_local, freq_global, ultimo)) for idx in indices]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    top = candidatos[:N_CANDIDATOS]
    scores = np.array([s for _, s in top])
    scores_adj = scores - np.max(scores)
    probs = np.exp(scores_adj / 0.5)
    probs /= probs.sum()
    
    selecionados = []
    tentativas = 0
    while len(selecionados) < N_JOGOS and tentativas < 500:
        idx_pool = np.random.choice(len(top), p=probs)
        jogo = pool[top[idx_pool][0]]['jogo']
        if all(interseccao(jogo, s) <= MAX_INTERSECAO for s in selecionados):
            selecionados.append(jogo)
        tentativas += 1
    
    for idx_sorted, _ in top:
        jogo = pool[idx_sorted]['jogo']
        if len(selecionados) >= N_JOGOS:
            break
        if all(interseccao(jogo, s) <= MAX_INTERSECAO for s in selecionados):
            selecionados.append(jogo)
    
    return selecionados

# ============================================================
# WALK-FORWARD COM JANELA VARIÁVEL
# ============================================================

def walk_forward_test(concursos, pool, window):
    estrategia_hits = []
    aleatorio_hits = []
    
    passo = max(1, (len(concursos) - window - 1) // N_TESTES)
    
    for inicio in tqdm(range(0, len(concursos) - window - 1, passo), desc=f"W={window}"):
        if len(estrategia_hits) >= N_TESTES:
            break
        
        fim = inicio + window
        janela_local = concursos[inicio:fim]
        historico_global = concursos[:inicio]
        
        if len(historico_global) < 50:
            continue
        
        teste = concursos[fim]
        ultimo = janela_local[-1]
        
        freq_local = calcular_frequencias_locais(janela_local)
        freq_global = calcular_frequencias_globais(historico_global)
        
        carteira_estrategia = selecionar_carteira(pool, freq_local, freq_global, ultimo)
        carteira_aleatoria = [gerar_jogo_aleatorio() for _ in range(N_JOGOS)]
        
        hits_est = max(interseccao(j, teste) for j in carteira_estrategia)
        hits_ale = max(interseccao(j, teste) for j in carteira_aleatoria)
        
        estrategia_hits.append(hits_est)
        aleatorio_hits.append(hits_ale)
    
    return estrategia_hits, aleatorio_hits

# ============================================================
# ANÁLISE DE UMA JANELA
# ============================================================

def analisar_janela(est, ale, window):
    print(f"\n--- Janela {window} ---")
    print(f"Média hits estratégia: {np.mean(est):.4f} ± {np.std(est):.4f}")
    print(f"Média hits aleatório:   {np.mean(ale):.4f} ± {np.std(ale):.4f}")
    diffs = [e - a for e, a in zip(est, ale)]
    print(f"Diferença média: {np.mean(diffs):.4f}")
    
    for h in [11, 12, 13, 14, 15]:
        fe = sum(1 for x in est if x >= h) / len(est)
        fa = sum(1 for x in ale if x >= h) / len(ale)
        print(f"  {h}+ pts: estratégia={fe:.4f} | aleatório={fa:.4f}")
    
    try:
        stat, p = wilcoxon(diffs)
        print(f"Wilcoxon: p={p:.4f}")
    except:
        p = 1.0
        print("Wilcoxon não calculado")
    
    if p < 0.05 and np.mean(diffs) > 0.02:
        print("✅ Sinal detectado")
    else:
        print("❌ Sem evidência")
    return diffs, p

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("V34.3 — TESTE FINAL DE MICRO‑DRIFT (JANELAS 5,10,20)")
    print("="*60)
    
    concursos = carregar_concursos()
    if concursos is None:
        return
    
    pool = criar_pool(20000)
    
    janelas = [5, 10, 20]
    resultados = {}
    
    for w in janelas:
        t0 = time.time()
        est, ale = walk_forward_test(concursos, pool, w)
        tempo = time.time() - t0
        diffs, p = analisar_janela(est, ale, w)
        resultados[w] = {
            'p': p,
            'diff_media': np.mean(diffs),
            'tempo': tempo
        }
    
    print("\n" + "="*60)
    print("RESUMO FINAL")
    for w in janelas:
        r = resultados[w]
        sig = "✅" if (r['p'] < 0.05 and r['diff_media'] > 0.02) else "❌"
        print(f"Janela {w:2d}: p={r['p']:.4f}, dif={r['diff_media']:.4f} {sig}")
    
    if all(r['p'] > 0.05 or r['diff_media'] <= 0.02 for r in resultados.values()):
        print("\nConclusão: NENHUMA janela apresentou evidência de drift local.")
        print("A Lotofácil se comporta como um processo IID nas condições testadas.")
        print("Encerra-se a investigação estatística de forma honesta.")
    else:
        print("\nAtenção: Pelo menos uma janela indicou sinal — investigar com cautela.")
    print("="*60)

if __name__ == "__main__":
    main()
