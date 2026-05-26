#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V35 — MEAN REVERSION LOCAL (HIPÓTESE DE COMPENSAÇÃO PROBABILÍSTICA)

Conceito:
✅ Para cada feature, calcula z‑score de desvio na janela recente
✅ Features "atrasadas" (abaixo do esperado) → peso ALTO
✅ Features "saturadas" (acima do esperado) → peso BAIXO
✅ Score do jogo = produto dos pesos
✅ Walk‑forward com 300 testes para janelas 5, 10, 20, 30
✅ Comparação direta com v34.3 (momentum) para fechar o ciclo
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
EPS = 1e-8
N_TESTES = 300

MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

# Features para o modelo de reversão à média
FEATURE_NAMES = ['pares', 'moldura', 'primos', 'consecutivos', 'repeticao']

def extrair_features(jogo, ultimo=None):
    """Retorna dicionário com todas as features de um jogo."""
    d = sorted(jogo)
    pares = sum(1 for x in d if x % 2 == 0)
    moldura = sum(1 for x in d if x in MOLDURA)
    primos = sum(1 for x in d if x in PRIMES)
    consecutivos = sum(1 for i in range(len(d)-1) if d[i+1] - d[i] == 1)
    rep = repeticao(jogo, ultimo) if ultimo is not None else 0
    return {
        'pares': pares,
        'moldura': moldura,
        'primos': primos,
        'consecutivos': consecutivos,
        'repeticao': rep
    }

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
# POOL PRÉ-COMPUTADO (COM FEATURES)
# ============================================================

def criar_pool(tamanho=20000):
    print(f"🔧 Criando pool de {tamanho} jogos...")
    pool = []
    for _ in tqdm(range(tamanho), desc="Pool"):
        jogo = gerar_jogo_aleatorio()
        feats = extrair_features(jogo)
        pool.append({
            'jogo': jogo,
            'features': feats
        })
    return pool

# ============================================================
# MODELO DE REVERSÃO À MÉDIA
# ============================================================

class MeanReversionModel:
    """
    Para cada feature, calcula:
    - Probabilidade histórica p (global)
    - Valor esperado na janela: E = n * p
    - Observado na janela: O
    - Z-score: z = (O - E) / sqrt(n * p * (1-p))
    - Peso: exp(-z)  (atrasado → peso alto; saturado → peso baixo)
    """
    
    def __init__(self, historico_global):
        self.probabilidades = {}
        self._calibrar(historico_global)
    
    def _calibrar(self, historico):
        """Calcula distribuição de probabilidade histórica para cada feature."""
        for feat in FEATURE_NAMES:
            valores = []
            for i, jogo in enumerate(historico):
                feats = extrair_features(jogo, historico[i-1] if i > 0 else None)
                valores.append(feats[feat])
            self.probabilidades[feat] = Counter(valores)
            total = len(historico)
            for v in self.probabilidades[feat]:
                self.probabilidades[feat][v] /= total
    
    def calcular_pesos(self, janela):
        """
        Para cada feature, calcula o peso baseado no desvio z.
        Retorna um dicionário: {feature: {valor: peso}}.
        """
        pesos = {}
        n = len(janela)
        
        for feat in FEATURE_NAMES:
            # Observado na janela
            obs = Counter()
            for i, jogo in enumerate(janela):
                feats = extrair_features(jogo, janela[i-1] if i > 0 else None)
                obs[feats[feat]] += 1
            
            pesos[feat] = {}
            # Para cada valor possível da feature (presente no histórico)
            for valor, p in self.probabilidades[feat].items():
                esperado = n * p
                observado = obs.get(valor, 0)
                variancia = n * p * (1 - p)
                if variancia > 0:
                    z = (observado - esperado) / np.sqrt(variancia)
                else:
                    z = 0
                # Peso: exp(-z) → z negativo (atrasado) → peso > 1
                #               z positivo (saturado) → peso < 1
                pesos[feat][valor] = np.exp(-z)
        
        return pesos
    
    def score_jogo(self, jogo_info, pesos, ultimo=None):
        """
        Score do jogo = produto dos pesos de cada feature.
        Quanto mais features atrasadas, maior o score.
        """
        score = 1.0
        feats = jogo_info['features']
        
        for feat in FEATURE_NAMES:
            if feat == 'repeticao' and ultimo is not None:
                valor = repeticao(jogo_info['jogo'], ultimo)
            else:
                valor = feats[feat]
            
            if feat in pesos and valor in pesos[feat]:
                score *= pesos[feat][valor]
            else:
                score *= 1.0  # valor não visto no histórico, peso neutro
        
        return score

# ============================================================
# SELEÇÃO DE CARTEIRA (REVERSÃO À MÉDIA)
# ============================================================

def selecionar_carteira_mean_reversion(pool, modelo, janela, ultimo):
    """Seleciona carteira usando o modelo de reversão à média."""
    pesos = modelo.calcular_pesos(janela)
    
    # Amostra do pool
    indices = np.random.choice(len(pool), size=N_CANDIDATOS, replace=False)
    candidatos = [(idx, modelo.score_jogo(pool[idx], pesos, ultimo)) for idx in indices]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    top = candidatos[:N_CANDIDATOS]
    scores = np.array([s for _, s in top])
    
    # Softmax para amostragem (mesma lógica do v34)
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
# SELEÇÃO DE CARTEIRA (MOMENTUM — MESMA LÓGICA DO V34)
# ============================================================

def calcular_frequencias_locais(janela):
    freq = {feat: Counter() for feat in FEATURE_NAMES}
    for i, jogo in enumerate(janela):
        feats = extrair_features(jogo, janela[i-1] if i > 0 else None)
        for feat in FEATURE_NAMES:
            freq[feat][feats[feat]] += 1
    total = len(janela)
    for feat in freq:
        for v in freq[feat]:
            freq[feat][v] /= total
    return freq

def calcular_frequencias_globais(historico):
    freq = {feat: Counter() for feat in FEATURE_NAMES}
    for i, jogo in enumerate(historico):
        feats = extrair_features(jogo, historico[i-1] if i > 0 else None)
        for feat in FEATURE_NAMES:
            freq[feat][feats[feat]] += 1
    total = len(historico)
    for feat in freq:
        for v in freq[feat]:
            freq[feat][v] /= total
    return freq

def score_jogo_momentum(item_pool, freq_local, freq_global, ultimo):
    score = 0.0
    for feat in FEATURE_NAMES:
        if feat == 'repeticao':
            valor = repeticao(item_pool['jogo'], ultimo)
        else:
            valor = item_pool['features'][feat]
        p_local = freq_local[feat].get(valor, EPS)
        p_global = freq_global[feat].get(valor, EPS)
        score += p_local / (p_global + EPS)
    return score

def selecionar_carteira_momentum(pool, freq_local, freq_global, ultimo):
    indices = np.random.choice(len(pool), size=N_CANDIDATOS, replace=False)
    candidatos = [(idx, score_jogo_momentum(pool[idx], freq_local, freq_global, ultimo)) for idx in indices]
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
# WALK-FORWARD (TESTA AMBAS AS HIPÓTESES)
# ============================================================

def walk_forward_dual(concursos, pool, window):
    """
    Para cada janela, testa tanto mean reversion quanto momentum.
    Retorna hits das duas estratégias + aleatório.
    """
    hits_mr = []    # mean reversion
    hits_mo = []    # momentum
    hits_ale = []   # aleatório
    
    passo = max(1, (len(concursos) - window - 1) // N_TESTES)
    
    for inicio in tqdm(range(0, len(concursos) - window - 1, passo), desc=f"W={window}"):
        if len(hits_mr) >= N_TESTES:
            break
        
        fim = inicio + window
        janela_local = concursos[inicio:fim]
        historico_global = concursos[:inicio]
        
        if len(historico_global) < 50:
            continue
        
        teste = concursos[fim]
        ultimo = janela_local[-1]
        
        # Mean reversion
        modelo_mr = MeanReversionModel(historico_global)
        carteira_mr = selecionar_carteira_mean_reversion(pool, modelo_mr, janela_local, ultimo)
        
        # Momentum (v34.3)
        freq_local = calcular_frequencias_locais(janela_local)
        freq_global = calcular_frequencias_globais(historico_global)
        carteira_mo = selecionar_carteira_momentum(pool, freq_local, freq_global, ultimo)
        
        # Aleatório
        carteira_ale = [gerar_jogo_aleatorio() for _ in range(N_JOGOS)]
        
        hits_mr.append(max(interseccao(j, teste) for j in carteira_mr))
        hits_mo.append(max(interseccao(j, teste) for j in carteira_mo))
        hits_ale.append(max(interseccao(j, teste) for j in carteira_ale))
    
    return hits_mr, hits_mo, hits_ale

# ============================================================
# ANÁLISE DE UMA JANELA
# ============================================================

def analisar_janela_dual(hits_mr, hits_mo, hits_ale, window):
    print(f"\n{'='*60}")
    print(f"Janela {window}")
    print(f"{'='*60}")
    
    for nome, hits in [("Mean Reversion", hits_mr), ("Momentum", hits_mo), ("Aleatório", hits_ale)]:
        print(f"\n--- {nome} ---")
        print(f"Média hits: {np.mean(hits):.4f} ± {np.std(hits):.4f}")
        for h in [11, 12, 13, 14, 15]:
            freq = sum(1 for x in hits if x >= h) / len(hits)
            print(f"  {h}+ pts: {freq:.4f}")
    
    # Testes estatísticos
    diffs_mr = [m - a for m, a in zip(hits_mr, hits_ale)]
    diffs_mo = [m - a for m, a in zip(hits_mo, hits_ale)]
    
    print("\n--- Testes Wilcoxon vs Aleatório ---")
    for nome, diffs in [("Mean Reversion", diffs_mr), ("Momentum", diffs_mo)]:
        try:
            stat, p = wilcoxon(diffs)
            sig = "✅" if (p < 0.05 and np.mean(diffs) > 0.02) else "❌"
            print(f"{nome:15s}: p={p:.4f}, dif média={np.mean(diffs):.4f} {sig}")
        except:
            print(f"{nome:15s}: não calculado")
    
    return diffs_mr, diffs_mo

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("V35 — MEAN REVERSION vs MOMENTUM (TESTE DUPLO)")
    print("="*60)
    
    concursos = carregar_concursos()
    if concursos is None:
        return
    
    pool = criar_pool(20000)
    
    janelas = [5, 10, 20, 30]
    resultados = {}
    
    for w in janelas:
        t0 = time.time()
        hits_mr, hits_mo, hits_ale = walk_forward_dual(concursos, pool, w)
        tempo = time.time() - t0
        diffs_mr, diffs_mo = analisar_janela_dual(hits_mr, hits_mo, hits_ale, w)
        resultados[w] = {
            'mr_diff': np.mean(diffs_mr),
            'mo_diff': np.mean(diffs_mo),
            'mr_p': wilcoxon(diffs_mr)[1] if len(set(diffs_mr)) > 1 else 1.0,
            'mo_p': wilcoxon(diffs_mo)[1] if len(set(diffs_mo)) > 1 else 1.0,
            'tempo': tempo
        }
    
    print("\n" + "="*60)
    print("RESUMO FINAL — TODAS AS JANELAS")
    print("="*60)
    print(f"{'Janela':<8} {'Mean Rev p':<12} {'MR diff':<10} {'Momentum p':<12} {'MO diff':<10}")
    print("-"*52)
    
    todas_falharam = True
    for w in janelas:
        r = resultados[w]
        mr_ok = r['mr_p'] < 0.05 and r['mr_diff'] > 0.02
        mo_ok = r['mo_p'] < 0.05 and r['mo_diff'] > 0.02
        if mr_ok or mo_ok:
            todas_falharam = False
        print(f"{w:<8} {r['mr_p']:<12.4f} {r['mr_diff']:<10.4f} {r['mo_p']:<12.4f} {r['mo_diff']:<10.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSÃO FINAL")
    print("="*60)
    if todas_falharam:
        print("❌ NENHUMA hipótese apresentou evidência significativa.")
        print("   Nem momentum (persistência) nem mean reversion (compensação)")
        print("   geram vantagem explorável na Lotofácil.")
        print("   O processo é compatível com IID.")
        print("\n   Investigação encerrada com dignidade metodológica.")
    else:
        print("⚠️  Pelo menos uma hipótese mostrou sinal — investigar com cautela.")
    print("="*60)

if __name__ == "__main__":
    main()
