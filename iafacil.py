#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IAFACIL v1.2.2 – Laboratório independente de IA Estrutural

Correções da v1.2.1:
- Baseline estrutural com distribuições marginais reais (não ingênuo)
- Estado inclui agregações: média de pares 3/5/10, equilíbrio de quentes, indicador 7-7-7
- Aleatório usa os mesmos candidatos que a IA
- Imprime P(ganha), P(empata), P(perde)
- Documentação do placebo (destrói ordem temporal)
"""

import numpy as np
from scipy.stats import wilcoxon
from collections import Counter
import os, random, time, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2,3,5,7,11,13,17,19,23}
MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
PREMIO_VALORES = {11:6.0, 12:12.0, 13:30.0, 14:1500.0, 15:1800000.0}
CUSTO_APOSTA = 3.5

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
def freq_janela(contests, inicio, fim, dezenas=range(1,26)):
    freq = Counter()
    inicio = max(0, inicio)
    for c in contests[inicio:fim]:
        freq.update(c['dezenas'])
    return {d: freq.get(d, 0) for d in dezenas}

def calcular_atrasos(contests, indice=None):
    if indice is None:
        indice = len(contests)
    atrasos = {}
    for d in range(1, 26):
        atraso = 0
        for j in range(indice-1, -1, -1):
            if d in contests[j]['dezenas']:
                break
            atraso += 1
        atrasos[d] = atraso
    return atrasos

def extrair_estrutura(dezenas):
    d = sorted(dezenas)
    pares = sum(1 for x in d if x % 2 == 0)
    impares = 15 - pares
    primos = sum(1 for x in d if x in PRIMES)
    moldura = sum(1 for x in d if x in MOLDURA)
    soma = sum(d)
    amplitude = max(d) - min(d)
    consecutivos = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    return {
        'pares': pares,
        'impares': impares,
        'primos': primos,
        'moldura': moldura,
        'soma': soma,
        'amplitude': amplitude,
        'consecutivos': consecutivos,
        'dezenas': set(d)
    }

def construir_estado(contests, idx, janela=10):
    """
    Vetor de features do estado passado.
    Inclui:
      - Para cada um dos últimos 'janela' concursos: 8 features básicas.
      - Agregações: média de pares 3/5/10, equilíbrio de quentes atual,
        indicador 7-7-7, proporção de ímpares no Top10.
    """
    if idx == 0:
        return np.zeros(8 * janela + 6)
    inicio = max(0, idx - janela)
    recentes = contests[inicio:idx]

    features_por_concurso = []
    for c in recentes:
        est = extrair_estrutura(c['dezenas'])
        features_por_concurso.extend([
            est['pares'],
            est['impares'],
            est['primos'],
            est['moldura'],
            est['soma'],
            est['amplitude'],
            est['consecutivos'],
            1 if c['dezenas'][-1] % 2 == 0 else 0
        ])

    # Agregações adicionais
    pares_recentes = [extrair_estrutura(c['dezenas'])['pares'] for c in recentes]

    # Média de pares dos últimos 3, 5, 10 (ou disponíveis)
    media_pares_3 = np.mean(pares_recentes[-3:]) if len(pares_recentes) >= 1 else 0
    media_pares_5 = np.mean(pares_recentes[-5:]) if len(pares_recentes) >= 1 else 0
    media_pares_10 = np.mean(pares_recentes[-10:]) if len(pares_recentes) >= 1 else 0

    # Equilíbrio de quentes atual (Top10 ímpares - pares)
    freq10 = freq_janela(contests, max(0, idx-10), idx)
    top10 = [d for d, _ in Counter(freq10).most_common(10)]
    impares_top10 = sum(1 for d in top10 if d % 2 != 0)
    pares_top10 = 10 - impares_top10
    equilibrio_quentes = impares_top10 - pares_top10

    # Indicador de sequência 7-7-7 nos últimos três concursos
    indicador_777 = 1.0 if len(pares_recentes) >= 3 and pares_recentes[-3:] == [7,7,7] else 0.0

    # Proporção de ímpares no Top10
    proporcao_impares_top10 = impares_top10 / 10.0

    features_agregadas = [
        media_pares_3,
        media_pares_5,
        media_pares_10,
        equilibrio_quentes,
        indicador_777,
        proporcao_impares_top10
    ]

    features = features_por_concurso + features_agregadas

    # Garantir tamanho consistente
    expected_len = 8 * janela + 6
    while len(features) < expected_len:
        features.append(0.0)
    return np.array(features[:expected_len], dtype=np.float32)

# ============================================================
# MODELO SOFTMAX REGRESSION
# ============================================================
class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.1, epochs=50, l2=0.001, batch_size=128, random_state=42):
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.random_state = random_state
        self.W = None
        self.b = None

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0, 0.01, (d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        y_onehot = np.eye(self.n_classes)[y]
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = y_onehot[idx]
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                z = X_batch @ self.W + self.b
                pred = self.softmax(z)
                grad_W = (X_batch.T @ (pred - y_batch)) / len(X_batch) + self.l2 * self.W
                grad_b = np.mean(pred - y_batch, axis=0)
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

    def predict_proba(self, X):
        z = X @ self.W + self.b
        return self.softmax(z)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ============================================================
# MAPEAMENTO DE CLASSES EXPLÍCITO
# ============================================================
def mapear_classes(y, classes_possiveis):
    mapa = {c: i for i, c in enumerate(classes_possiveis)}
    y_idx = np.array([mapa.get(int(v), 0) for v in y])
    return y_idx, len(classes_possiveis)

CLASSES_PARES = np.arange(5, 11)        # 5..10
CLASSES_REP = np.arange(5, 16)          # 5..15
CLASSES_QUENTES = np.arange(0, 11)      # 0..10
CLASSES_ATR = np.arange(0, 16)          # 0..15

# ============================================================
# TESTE 1: Previsibilidade de pares (com N e razão)
# ============================================================
def teste_previsibilidade_pares(contests, min_history=500):
    print("\n📊 TESTE 1 – Previsibilidade da estrutura de pares")
    dist_global = Counter()
    dist_cond = Counter()
    total_global = 0
    total_cond = 0

    for idx in range(min_history, len(contests)):
        pares_atual = extrair_estrutura(contests[idx]['dezenas'])['pares']
        dist_global[pares_atual] += 1
        total_global += 1
        if idx >= 3:
            p1 = extrair_estrutura(contests[idx-3]['dezenas'])['pares']
            p2 = extrair_estrutura(contests[idx-2]['dezenas'])['pares']
            p3 = extrair_estrutura(contests[idx-1]['dezenas'])['pares']
            if p1 == p2 == p3 == 7:
                dist_cond[pares_atual] += 1
                total_cond += 1

    print(f"   Distribuição geral (N={total_global}):")
    for k in sorted(dist_global):
        pct = dist_global[k] / total_global * 100
        print(f"      {k}: {pct:.1f}%")

    print(f"   Após 7-7-7 (N={total_cond}):")
    if total_cond > 0:
        for k in sorted(dist_cond):
            p_cond = dist_cond[k] / total_cond
            p_global = dist_global[k] / total_global
            razao = p_cond / p_global if p_global > 0 else np.inf
            print(f"      {k}: {p_cond*100:.1f}%  (global={p_global*100:.1f}%, razão={razao:.2f})")
    else:
        print("      Nenhuma ocorrência de 7-7-7 encontrada.")
    return dist_global, dist_cond

# ============================================================
# TESTE 2: Quentes ímpares -> pares seguintes
# ============================================================
def teste_quentes_impares(contests, min_history=500):
    print("\n📊 TESTE 2 – Relação entre quentes ímpares/pares e pares seguintes")
    relacao = {}
    for idx in range(min_history, len(contests)):
        freq10 = freq_janela(contests, max(0, idx-10), idx)
        top10 = [d for d, _ in Counter(freq10).most_common(10)]
        impares_top10 = sum(1 for d in top10 if d % 2 != 0)
        pares_top10 = 10 - impares_top10
        equilibrio = impares_top10 - pares_top10
        pares_atual = extrair_estrutura(contests[idx]['dezenas'])['pares']
        if equilibrio not in relacao:
            relacao[equilibrio] = {'n': 0, 'soma_pares': 0, 'dist': Counter()}
        relacao[equilibrio]['n'] += 1
        relacao[equilibrio]['soma_pares'] += pares_atual
        relacao[equilibrio]['dist'][pares_atual] += 1

    print(f"   Equilíbrio (impares-pares) -> pares no próximo concurso")
    for eq in sorted(relacao):
        n = relacao[eq]['n']
        media = relacao[eq]['soma_pares'] / n if n > 0 else 0
        print(f"   Eq={eq:+2d} (N={n:4d}): média pares={media:.2f} | distribuição:", end="")
        for pares in sorted(relacao[eq]['dist']):
            pct = relacao[eq]['dist'][pares] / n * 100
            print(f" {pares}:{pct:.0f}%", end="")
        print()
    return relacao

# ============================================================
# TESTE 3: Todas as sequências de três pares (diagnóstico)
# ============================================================
def teste_todas_sequencias_pares(contests, min_history=500):
    print("\n📊 TESTE 3 – Memória estrutural de curto prazo (sequências de 3 pares)")
    sequencias = Counter()
    for idx in range(min_history+2, len(contests)):
        p1 = extrair_estrutura(contests[idx-3]['dezenas'])['pares']
        p2 = extrair_estrutura(contests[idx-2]['dezenas'])['pares']
        p3 = extrair_estrutura(contests[idx-1]['dezenas'])['pares']
        pares_atual = extrair_estrutura(contests[idx]['dezenas'])['pares']
        sequencias[(p1,p2,p3,pares_atual)] += 1

    alvo = (7,7,7)
    print(f"   Sequências após {alvo[0]}-{alvo[1]}-{alvo[2]}:")
    for (p1,p2,p3,p4), n in sorted(sequencias.items()):
        if p1==alvo[0] and p2==alvo[1] and p3==alvo[2]:
            print(f"      → {p4} pares: {n} ocorrências")
    return sequencias

# ============================================================
# FUNÇÕES PARA IA ESTRUTURAL (v1.2.2)
# ============================================================
def preparar_alvos(contests, min_history=500, janela=10):
    n = len(contests)
    X_list = []
    y_pares = []
    y_repetidas = []
    y_quentes = []
    y_atrasadas = []
    idx_list = []

    for idx in range(min_history, n):
        estado = construir_estado(contests, idx, janela=janela)
        alvo_est = extrair_estrutura(contests[idx]['dezenas'])
        atual = alvo_est['dezenas']

        pares = alvo_est['pares']
        y_pares.append(pares)

        if idx > 0:
            anterior = set(contests[idx-1]['dezenas'])
            rep = len(anterior & atual)
        else:
            rep = 9
        y_repetidas.append(rep)

        freq10 = freq_janela(contests, max(0, idx-10), idx)
        top10 = set(d for d, _ in Counter(freq10).most_common(10))
        q = len(top10 & atual)
        y_quentes.append(q)

        atrasos = calcular_atrasos(contests, indice=idx)
        atrasadas_set = set(d for d, atr in atrasos.items() if atr >= 5)
        a = len(atrasadas_set & atual)
        y_atrasadas.append(a)

        X_list.append(estado)
        idx_list.append(idx)

    X = np.array(X_list, dtype=np.float32)
    y_pares = np.array(y_pares)
    y_repetidas = np.array(y_repetidas)
    y_quentes = np.array(y_quentes)
    y_atrasadas = np.array(y_atrasadas)
    return X, y_pares, y_repetidas, y_quentes, y_atrasadas, idx_list

def treinar_modelo(X, y, n_classes, epochs=50, lr=0.1, l2=0.001):
    modelo = SoftmaxRegression(n_classes=n_classes, lr=lr, epochs=epochs, l2=l2,
                               batch_size=128, random_state=42)
    modelo.fit(X, y)
    return modelo

def prever_distribuicao(modelos, estado):
    preds = {}
    for alvo, modelo in modelos.items():
        prob = modelo.predict_proba(estado.reshape(1, -1))[0]
        preds[alvo] = prob
    return preds

def pontuar_conjunto(conjunto, preds, estado_info):
    pares = sum(1 for x in conjunto if x % 2 == 0)
    anterior = estado_info['anterior']
    rep = len(conjunto & anterior)
    top10 = estado_info['top10']
    q = len(conjunto & top10)
    atrasadas_set = estado_info['atrasadas_set']
    a = len(conjunto & atrasadas_set)

    log_lik = 0.0
    idx_pares = int(pares) - 5
    if idx_pares < 0: idx_pares = 0
    elif idx_pares >= len(preds['pares']): idx_pares = len(preds['pares'])-1
    log_lik += np.log(preds['pares'][idx_pares] + 1e-9)

    idx_rep = int(rep) - 5
    if idx_rep < 0: idx_rep = 0
    elif idx_rep >= len(preds['repetidas']): idx_rep = len(preds['repetidas'])-1
    log_lik += np.log(preds['repetidas'][idx_rep] + 1e-9)

    idx_quentes = int(q)
    if idx_quentes < 0: idx_quentes = 0
    elif idx_quentes >= len(preds['quentes']): idx_quentes = len(preds['quentes'])-1
    log_lik += np.log(preds['quentes'][idx_quentes] + 1e-9)

    idx_atr = int(a)
    if idx_atr < 0: idx_atr = 0
    elif idx_atr >= len(preds['atrasadas']): idx_atr = len(preds['atrasadas'])-1
    log_lik += np.log(preds['atrasadas'][idx_atr] + 1e-9)
    return log_lik

def gerar_candidatos(seed, n_candidates=2000):
    rng = np.random.default_rng(seed)
    candidatos = []
    for _ in range(n_candidates):
        conj = set(rng.choice(range(1,26), 20, replace=False))
        candidatos.append(conj)
    return candidatos

def gerar_top20_por_likelihood(preds, estado_info, candidatos):
    melhor_conjunto = None
    melhor_loglik = -np.inf
    for conjunto in candidatos:
        loglik = pontuar_conjunto(conjunto, preds, estado_info)
        if loglik > melhor_loglik:
            melhor_loglik = loglik
            melhor_conjunto = conjunto
    return melhor_conjunto

def baseline_estrutural(estado_info, candidatos, dist_marginais):
    """
    Baseline estrutural usando distribuições marginais históricas.
    dist_marginais é um dicionário com as distribuições de pares, repetidas,
    quentes e atrasadas (estimadas no treino).
    """
    melhor_conjunto = None
    melhor_loglik = -np.inf
    for conjunto in candidatos:
        pares = sum(1 for x in conjunto if x % 2 == 0)
        rep = len(conjunto & estado_info['anterior'])
        q = len(conjunto & estado_info['top10'])
        a = len(conjunto & estado_info['atrasadas_set'])

        loglik = 0.0
        # Pares
        prob_pares = dist_marginais['pares'].get(pares, 1e-6)
        loglik += np.log(prob_pares + 1e-9)
        # Repetidas
        prob_rep = dist_marginais['rep'].get(rep, 1e-6)
        loglik += np.log(prob_rep + 1e-9)
        # Quentes
        prob_quentes = dist_marginais['quentes'].get(q, 1e-6)
        loglik += np.log(prob_quentes + 1e-9)
        # Atrasadas
        prob_atr = dist_marginais['atr'].get(a, 1e-6)
        loglik += np.log(prob_atr + 1e-9)

        if loglik > melhor_loglik:
            melhor_loglik = loglik
            melhor_conjunto = conjunto
    return melhor_conjunto

def analise_ia_estrutural(contests, min_history=500, n_backtest=500, n_placebos=50):
    print(f"\n🧠 IA ESTRUTURAL v1.2.2")
    print(f"   OOS: últimos {n_backtest} concursos")
    print(f"   Placebos: {n_placebos}")
    print(f"   Placebo: destrói ordem temporal, mantém estruturas internas")

    n = len(contests)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    oos_inicio = max(val_end, n - n_backtest)

    # Seleção de janela na validação
    print("Selecionando janela do estado na validação...")
    janelas = [3, 5, 10, 20]
    melhor_janela = 10
    melhor_acc = -1
    for janela in janelas:
        X, y_pares, y_rep, y_quentes, y_atr, idx_list = preparar_alvos(
            contests, min_history=min_history, janela=janela
        )
        y_pares_idx, n_pares = mapear_classes(y_pares, CLASSES_PARES)
        y_rep_idx, n_rep = mapear_classes(y_rep, CLASSES_REP)
        y_quentes_idx, n_quentes = mapear_classes(y_quentes, CLASSES_QUENTES)
        y_atr_idx, n_atr = mapear_classes(y_atr, CLASSES_ATR)

        train_mask = [i for i, iv in enumerate(idx_list) if iv < train_end]
        val_mask = [i for i, iv in enumerate(idx_list) if train_end <= iv < val_end]
        X_train = X[train_mask]
        y_pares_train = y_pares_idx[train_mask]
        X_val = X[val_mask]
        y_pares_val = y_pares_idx[val_mask]

        modelo = SoftmaxRegression(n_classes=n_pares, lr=0.1, epochs=20, l2=0.001,
                                   batch_size=128, random_state=42)
        modelo.fit(X_train, y_pares_train)
        pred = modelo.predict(X_val)
        acc = np.mean(pred == y_pares_val)
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_janela = janela
        print(f"   janela={janela}: acurácia pares na validação = {acc:.3f}")
    print(f"   Melhor janela: {melhor_janela}")

    # Construir dataset com a janela selecionada
    X, y_pares, y_rep, y_quentes, y_atr, idx_list = preparar_alvos(
        contests, min_history=min_history, janela=melhor_janela
    )
    y_pares_idx, n_pares = mapear_classes(y_pares, CLASSES_PARES)
    y_rep_idx, n_rep = mapear_classes(y_rep, CLASSES_REP)
    y_quentes_idx, n_quentes = mapear_classes(y_quentes, CLASSES_QUENTES)
    y_atr_idx, n_atr = mapear_classes(y_atr, CLASSES_ATR)

    train_mask = [i for i, iv in enumerate(idx_list) if iv < train_end]
    test_mask = [i for i, iv in enumerate(idx_list) if iv >= oos_inicio]

    X_train = X[train_mask]
    y_pares_train = y_pares_idx[train_mask]
    y_rep_train = y_rep_idx[train_mask]
    y_quentes_train = y_quentes_idx[train_mask]
    y_atr_train = y_atr_idx[train_mask]

    # Distribuições marginais (para baseline estrutural)
    dist_marginais = {
        'pares': Counter(y_pares[train_mask]),
        'rep': Counter(y_rep[train_mask]),
        'quentes': Counter(y_quentes[train_mask]),
        'atr': Counter(y_atr[train_mask])
    }

    # Treinar modelos finais
    print("Treinando modelos finais...")
    modelos = {}
    modelos['pares'] = treinar_modelo(X_train, y_pares_train, n_pares, epochs=50)
    modelos['repetidas'] = treinar_modelo(X_train, y_rep_train, n_rep, epochs=50)
    modelos['quentes'] = treinar_modelo(X_train, y_quentes_train, n_quentes, epochs=50)
    modelos['atrasadas'] = treinar_modelo(X_train, y_atr_train, n_atr, epochs=50)

    # Avaliar no OOS
    print(f"Avaliando no OOS ({oos_inicio} a {n})...")
    acertos_ia = []
    acertos_estrut = []
    acertos_aleat = []
    acertos_freq = []
    acertos_persist = []

    rng = np.random.default_rng(42)

    for idx in tqdm(range(oos_inicio, n), desc="OOS"):
        estado = construir_estado(contests, idx, janela=melhor_janela)
        preds = prever_distribuicao(modelos, estado)

        freq10 = freq_janela(contests, max(0, idx-10), idx)
        top10 = set(d for d, _ in Counter(freq10).most_common(10))
        atrasos = calcular_atrasos(contests, indice=idx)
        atrasadas_set = set(d for d, atr in atrasos.items() if atr >= 5)
        anterior = set(contests[idx-1]['dezenas']) if idx > 0 else set()

        estado_info = {'anterior': anterior, 'top10': top10, 'atrasadas_set': atrasadas_set}

        # Gerar candidatos compartilhados
        candidatos = gerar_candidatos(seed=idx, n_candidates=2000)

        # IA Estrutural
        top20_ia = gerar_top20_por_likelihood(preds, estado_info, candidatos)

        # Baseline estrutural
        top20_estrut = baseline_estrutural(estado_info, candidatos, dist_marginais)

        # Aleatório: escolher aleatoriamente um dos candidatos
        idx_aleatorio = rng.integers(0, len(candidatos))
        aleatorias = candidatos[idx_aleatorio]

        # Frequência recente
        top20_freq = set(sorted(range(1,26), key=lambda d: freq10.get(d,0), reverse=True)[:20])

        # Persistência
        persist = set(anterior)
        restantes = [d for d in range(1,26) if d not in persist]
        restantes.sort(key=lambda d: -freq10.get(d,0))
        persist.update(restantes[:max(0, 20-len(persist))])
        top20_persist = persist

        alvo = set(contests[idx]['dezenas'])
        acertos_ia.append(len(top20_ia & alvo))
        acertos_estrut.append(len(top20_estrut & alvo))
        acertos_aleat.append(len(aleatorias & alvo))
        acertos_freq.append(len(top20_freq & alvo))
        acertos_persist.append(len(top20_persist & alvo))

    arr_ia = np.array(acertos_ia)
    arr_estrut = np.array(acertos_estrut)
    arr_aleat = np.array(acertos_aleat)
    arr_freq = np.array(acertos_freq)
    arr_persist = np.array(acertos_persist)

    media_ia = np.mean(arr_ia)
    media_estrut = np.mean(arr_estrut)
    media_aleat = np.mean(arr_aleat)
    media_freq = np.mean(arr_freq)
    media_persist = np.mean(arr_persist)

    print("\n📊 RESULTADOS (Top 20)")
    print(f"   IA Estrutural: {media_ia:.3f}")
    print(f"   Baseline estrutural: {media_estrut:.3f}")
    print(f"   Frequência: {media_freq:.3f}")
    print(f"   Persistência: {media_persist:.3f}")
    print(f"   Aleatório: {media_aleat:.3f}")
    print(f"   Teórico: 12.000")

    # Testes estatísticos: IA vs Aleatório
    dif = arr_ia - arr_aleat
    observado = np.mean(dif)
    desvio = np.std(dif, ddof=1) if len(dif) > 1 else 0
    cohens_d = observado / desvio if desvio > 0 else 0
    w_stat, w_p = wilcoxon(dif)

    # Percentuais de vitória/empate/derrota
    p_ganha = np.mean(dif > 0)
    p_empata = np.mean(dif == 0)
    p_perde = np.mean(dif < 0)

    rng_perm = np.random.default_rng(123)
    perm_means = np.empty(10000)
    for k in range(10000):
        sinais = rng_perm.choice([-1, 1], size=len(dif))
        perm_means[k] = np.mean(dif * sinais)
    p_perm = np.mean(np.abs(perm_means) >= abs(observado))
    rng_boot = np.random.default_rng(42)
    n_boot = 5000
    ic_low, ic_high = np.percentile(
        dif[rng_boot.integers(0, len(dif), size=(n_boot, len(dif)))].mean(axis=1),
        [2.5, 97.5]
    )
    print(f"\n🔍 Teste estatístico (IA Estrutural vs Aleatório):")
    print(f"   Diferença média: {observado:+.3f} (IC95%: [{ic_low:.3f}, {ic_high:.3f}])")
    print(f"   Cohen's d: {cohens_d:+.3f}")
    print(f"   Wilcoxon: W={w_stat}, p={w_p:.4f}")
    print(f"   Permutação: p={p_perm:.4f}")
    print(f"   P(ganha)={p_ganha:.3f}, P(empata)={p_empata:.3f}, P(perde)={p_perde:.3f}")

    # Placebo temporal com re-treinamento
    print(f"\n🧪 PLACEBO TEMPORAL ({n_placebos} embaralhamentos)")
    deltas_placebo = []
    rng_master = np.random.default_rng(20260907)
    for _ in tqdm(range(n_placebos), desc="Placebos"):
        contests_placebo = contests.copy()
        seed_placebo = int(rng_master.integers(0, 2**32 - 1))
        rng_placebo = np.random.default_rng(seed_placebo)
        rng_placebo.shuffle(contests_placebo)

        X_p, y_pares_p, y_rep_p, y_quentes_p, y_atr_p, idx_list_p = preparar_alvos(
            contests_placebo, min_history=min_history, janela=melhor_janela
        )
        y_pares_p_idx, n_pares_p = mapear_classes(y_pares_p, CLASSES_PARES)
        y_rep_p_idx, n_rep_p = mapear_classes(y_rep_p, CLASSES_REP)
        y_quentes_p_idx, n_quentes_p = mapear_classes(y_quentes_p, CLASSES_QUENTES)
        y_atr_p_idx, n_atr_p = mapear_classes(y_atr_p, CLASSES_ATR)

        train_mask_p = [i for i, iv in enumerate(idx_list_p) if iv < train_end]
        X_train_p = X_p[train_mask_p]
        y_pares_train_p = y_pares_p_idx[train_mask_p]
        y_rep_train_p = y_rep_p_idx[train_mask_p]
        y_quentes_train_p = y_quentes_p_idx[train_mask_p]
        y_atr_train_p = y_atr_p_idx[train_mask_p]

        modelos_p = {}
        modelos_p['pares'] = treinar_modelo(X_train_p, y_pares_train_p, n_pares_p, epochs=50)
        modelos_p['repetidas'] = treinar_modelo(X_train_p, y_rep_train_p, n_rep_p, epochs=50)
        modelos_p['quentes'] = treinar_modelo(X_train_p, y_quentes_train_p, n_quentes_p, epochs=50)
        modelos_p['atrasadas'] = treinar_modelo(X_train_p, y_atr_train_p, n_atr_p, epochs=50)

        # Distribuições marginais placebo
        dist_marginais_p = {
            'pares': Counter(y_pares_p[train_mask_p]),
            'rep': Counter(y_rep_p[train_mask_p]),
            'quentes': Counter(y_quentes_p[train_mask_p]),
            'atr': Counter(y_atr_p[train_mask_p])
        }

        acertos_ia_p = []
        acertos_aleat_p = []
        for idx in range(oos_inicio, n):
            estado = construir_estado(contests_placebo, idx, janela=melhor_janela)
            preds_p = prever_distribuicao(modelos_p, estado)
            freq10_p = freq_janela(contests_placebo, max(0, idx-10), idx)
            top10_p = set(d for d, _ in Counter(freq10_p).most_common(10))
            atrasos_p = calcular_atrasos(contests_placebo, indice=idx)
            atrasadas_set_p = set(d for d, atr in atrasos_p.items() if atr >= 5)
            anterior_p = set(contests_placebo[idx-1]['dezenas']) if idx > 0 else set()
            estado_info_p = {'anterior': anterior_p, 'top10': top10_p, 'atrasadas_set': atrasadas_set_p}
            candidatos_p = gerar_candidatos(seed=idx, n_candidates=2000)
            top20_ia_p = gerar_top20_por_likelihood(preds_p, estado_info_p, candidatos_p)
            aleatorias_p = candidatos_p[0]  # usa primeiro candidato como aleatório
            alvo_p = set(contests_placebo[idx]['dezenas'])
            acertos_ia_p.append(len(top20_ia_p & alvo_p))
            acertos_aleat_p.append(len(aleatorias_p & alvo_p))

        if acertos_ia_p:
            delta_p = np.mean(acertos_ia_p) - np.mean(acertos_aleat_p)
            deltas_placebo.append(delta_p)

    deltas_placebo = np.array(deltas_placebo)
    p_placebo = np.mean(deltas_placebo >= observado)
    ic_placebo_low, ic_placebo_high = np.percentile(deltas_placebo, [2.5, 97.5])
    print(f"   Δ real (IA - Aleatório): {observado:+.3f}")
    print(f"   Δ placebo médio: {np.mean(deltas_placebo):+.3f} ± {np.std(deltas_placebo):.3f}")
    print(f"   Mediana Δ placebo: {np.median(deltas_placebo):+.3f}")
    print(f"   IC95% placebo: [{ic_placebo_low:+.3f}, {ic_placebo_high:+.3f}]")
    print(f"   Melhor Δ placebo: {np.max(deltas_placebo):+.3f}")
    print(f"   Proporção placebo > 0: {np.mean(deltas_placebo > 0):.3f}")
    print(f"   p-valor empírico (placebos ≥ real): {p_placebo:.4f}")

    return arr_ia, arr_aleat, arr_freq, arr_persist, arr_estrut

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧠 IAFACIL v1.2.2 – Laboratório independente de IA Estrutural")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Teste 1: Previsibilidade de pares")
        print("2. Teste 2: Quentes ímpares -> pares seguintes")
        print("3. Teste 3: Todas as sequências de 3 pares (diagnóstico)")
        print("4. Executar IA Estrutural completa (walk-forward + placebo)")
        print("0. Sair")
        op = input("Escolha: ").strip()
        if op == '1':
            teste_previsibilidade_pares(contests)
        elif op == '2':
            teste_quentes_impares(contests)
        elif op == '3':
            teste_todas_sequencias_pares(contests)
        elif op == '4':
            try:
                min_history = int(input("   Histórico mínimo [500]: ").strip() or "500")
                n_backtest = int(input("   Concursos para backtest [500]: ").strip() or "500")
                n_placebos = int(input("   Número de placebos [50]: ").strip() or "50")
            except:
                min_history, n_backtest, n_placebos = 500, 500, 50
            analise_ia_estrutural(contests, min_history=min_history,
                                  n_backtest=n_backtest, n_placebos=n_placebos)
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
