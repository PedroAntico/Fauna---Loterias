#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v75
FEATURE IMPORTANCE WALK‑FORWARD: QUAIS GEOMETRIAS REALMENTE IMPORTAM?

OBJETIVO:
✅ Auditar cada componente geométrica (linhas, colunas, diagonais, cruz)
   e cada parâmetro estrutural (pares, moldura, etc.) isoladamente.
✅ Medir o ganho real de cada feature sobre o baseline (modelo plano).
✅ Walk‑forward honesto: treino 500, teste 50, passo 50.
✅ Ranking final de importância para guiar a simplificação do modelo.
✅ Manter apenas as features com ganho positivo significativo.
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings, json
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

STRUCTURAL_PARAMS = ['pares', 'moldura', 'primos', 'repeticoes', 'amplitude']

LINHAS = {
    1: [1, 2, 3, 4, 5],
    2: [6, 7, 8, 9, 10],
    3: [11, 12, 13, 14, 15],
    4: [16, 17, 18, 19, 20],
    5: [21, 22, 23, 24, 25]
}

COLUNAS = {
    1: [1, 6, 11, 16, 21],
    2: [2, 7, 12, 17, 22],
    3: [3, 8, 13, 18, 23],
    4: [4, 9, 14, 19, 24],
    5: [5, 10, 15, 20, 25]
}

DIAGONAL_PRINCIPAL = [1, 7, 13, 19, 25]
DIAGONAL_SECUNDARIA = [5, 9, 13, 17, 21]
CRUZ_CENTRAL = [3, 8, 13, 18, 23, 11, 12, 14, 15]

# ============================================================
# BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}
    def get_mask(self, game):
        key = tuple(game) if isinstance(game, list) else game
        if key not in self._cache:
            mask = 0
            for d in key:
                mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]

BITMASK_CACHE = BitmaskCache()
mask_intersection = lambda m1, m2: (m1 & m2).bit_count()

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
# GERADOR DE JOGOS
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None):
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed):
        if fixed is None: fixed = []
        if semifixed is None: semifixed = []
        fixed_set = set(fixed)
        semifixed_set = set(semifixed) - fixed_set
        proibidas = fixed_set | semifixed_set
        restantes = list(set(range(1, 26)) - proibidas)
        n_fixas = len(fixed_set)
        max_semi = min(max_semifixed, len(semifixed_set)) if max_semifixed is not None else len(semifixed_set)
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None
        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = 15 - n_fixas - n_semifixed_escolher
        if n_restantes < 0 or n_restantes > len(restantes):
            return None
        for _ in range(200):
            chosen_semi = set(random.sample(list(semifixed_set), min(n_semifixed_escolher, len(semifixed_set)))) if n_semifixed_escolher > 0 and len(semifixed_set) > 0 else set()
            chosen_rest = set(random.sample(restantes, min(n_restantes, len(restantes)))) if n_restantes > 0 else set()
            game = sorted(fixed_set | chosen_semi | chosen_rest)
            if len(game) == 15:
                return game
        return None

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def extract_filter(dezenas, filter_name, prev_dezenas=None):
    d = sorted(dezenas)
    if filter_name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if filter_name == 'moldura': return sum(1 for x in d if x in MOLDURA)
    if filter_name == 'primos': return sum(1 for x in d if x in PRIMES)
    if filter_name == 'amplitude': return max(d) - min(d)
    if filter_name == 'repeticoes':
        if prev_dezenas is None: return 8
        return len(set(d) & set(prev_dezenas))
    return 0

def count_in_set(dezenas, elementos):
    return len(set(dezenas) & set(elementos))

# ============================================================
# FEATURES A SEREM AUDITADAS
# ============================================================
def get_all_features():
    """Retorna uma lista de (nome_da_feature, função_de_extração)."""
    features = []
    # Parâmetros estruturais
    for param in STRUCTURAL_PARAMS:
        features.append((param, lambda g, prev, p=param: extract_filter(g, p, prev)))
    # Linhas
    for linha_num in range(1, 6):
        features.append((f'linha_{linha_num}', lambda g, prev, ln=linha_num: count_in_set(g, LINHAS[ln])))
    # Colunas
    for col_num in range(1, 6):
        features.append((f'coluna_{col_num}', lambda g, prev, cn=col_num: count_in_set(g, COLUNAS[cn])))
    # Diagonais
    features.append(('diagonal_principal', lambda g, prev: count_in_set(g, DIAGONAL_PRINCIPAL)))
    features.append(('diagonal_secundaria', lambda g, prev: count_in_set(g, DIAGONAL_SECUNDARIA)))
    # Cruz central
    features.append(('cruz_central', lambda g, prev: count_in_set(g, CRUZ_CENTRAL)))
    return features

# ============================================================
# OTIMIZADOR SIMPLES (APENAS CENTRO ESTRUTURAL)
# ============================================================
class SimpleOptimizer:
    def __init__(self, contests):
        self.contests = contests
        self.generator = LooseGenerator()

    def generate_pool(self, n_candidates, prev_dezenas=None):
        pool, seen = [], set()
        for _ in range(n_candidates):
            try:
                g = self.generator.generate_one()
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

    def backtest(self, portfolio, test_draws):
        if len(portfolio) == 0:
            return {'lift': 0, 'roi': 0, 'hit_distribution': {k:0 for k in range(11,16)}}
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {'empirical': prob, 'theoretical': theo_prob,
                'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts}

# ============================================================
# FEATURE IMPORTANCE WALK‑FORWARD
# ============================================================
def feature_importance_walk_forward(contests, train_size=500, test_size=50, step=50, n_games=5):
    """
    Para cada feature (linhas, colunas, etc.), testa se adicionar
    o score de ciclo daquela feature melhora o lift em relação ao baseline.
    """
    features = get_all_features()
    print(f"\n🔬 FEATURE IMPORTANCE WALK‑FORWARD")
    print(f"   Features a auditar: {len(features)}")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}\n")

    # Baseline: modelo plano (sem pressão)
    baseline_lifts = []
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        opt = SimpleOptimizer(train_data)
        pool = opt.generate_pool(2000, None)
        portfolio = random.sample(pool, min(n_games, len(pool))) if len(pool) >= n_games else pool
        bt = opt.backtest(portfolio, test_data)
        baseline_lifts.append(bt['lift'])
        start += step
    baseline_mean = np.mean(baseline_lifts) if baseline_lifts else 0.0

    # Para cada feature, testar o ganho
    results = []
    for feat_name, feat_func in tqdm(features, desc="Auditando features"):
        lifts_with_feat = []
        start = train_size
        while start + test_size <= len(contests):
            train_data = contests[start-train_size:start]
            test_data = contests[start:start+test_size]
            try:
                # Gerar pool
                opt = SimpleOptimizer(train_data)
                pool = opt.generate_pool(2000, None)
                if len(pool) < n_games:
                    start += step
                    continue
                # Extrair série da feature no treino
                feat_series = []
                for i, c in enumerate(train_data):
                    prev = train_data[i-1]['dezenas'] if i > 0 else None
                    feat_series.append(feat_func(c['dezenas'], prev))
                feat_series = np.array(feat_series, dtype=int)
                # Calcular z‑score de atraso para cada valor possível
                z_scores = {}
                for val in set(feat_series):
                    ocorrencias = np.where(feat_series == val)[0]
                    if len(ocorrencias) > 1:
                        intervalos = np.diff(ocorrencias)
                        intervalo_medio = np.mean(intervalos)
                        desvio = np.std(intervalos)
                        atraso = len(feat_series) - 1 - ocorrencias[-1]
                        if desvio > 0:
                            z_scores[val] = (atraso - intervalo_medio) / desvio
                        else:
                            z_scores[val] = 0.0
                    else:
                        z_scores[val] = 0.0
                # Score para cada jogo: valor da feature → z‑score (subtrai para bonificar atrasados)
                scored = []
                for g in pool:
                    val = feat_func(g, train_data[-1]['dezenas'] if train_data else None)
                    z = z_scores.get(val, 0.0)
                    # Score simples: apenas o ciclo desta feature (quanto menor, melhor)
                    score = -z  # z positivo (atrasado) → score negativo (melhor)
                    scored.append((score, g))
                scored.sort(key=lambda x: x[0])
                # Selecionar os melhores (2C+2I+1P)
                n_total = len(scored)
                idx1 = min(2 * n_total // n_games, n_total)
                idx2 = min(4 * n_total // n_games, n_total)
                centrais = [g for _, g in scored[:idx1]][:2]
                intermed = [g for _, g in scored[idx1:idx2]][:2]
                perifs = [g for _, g in scored[idx2:]][:1]
                portfolio = centrais + intermed + perifs
                if len(portfolio) < n_games:
                    portfolio = [g for _, g in scored[:n_games]]
                bt = opt.backtest(portfolio, test_data)
                lifts_with_feat.append(bt['lift'])
            except Exception as e:
                lifts_with_feat.append(0.0)
            start += step
        mean_lift = np.mean(lifts_with_feat) if lifts_with_feat else 0.0
        gain = mean_lift - baseline_mean
        results.append({
            'feature': feat_name,
            'mean_lift': mean_lift,
            'gain': gain,
            'n_windows': len(lifts_with_feat)
        })

    # Ordenar por ganho
    results.sort(key=lambda x: x['gain'], reverse=True)

    # Exibir ranking
    print(f"\n📊 RANKING DE IMPORTÂNCIA (ganho sobre baseline = {baseline_mean:.4f}):")
    print(f"{'Feature':<25} {'Lift Médio':<12} {'Ganho':<10} {'Janelas':<10} {'Status'}")
    print("-" * 65)
    for res in results:
        status = "✅ POSITIVO" if res['gain'] > 0.001 else ("❌ NEGATIVO" if res['gain'] < -0.001 else "➖ NEUTRO")
        print(f"{res['feature']:<25} {res['mean_lift']:<12.4f} {res['gain']:<10.4f} {res['n_windows']:<10} {status}")

    # Resumo
    positivas = [r for r in results if r['gain'] > 0.001]
    negativas = [r for r in results if r['gain'] < -0.001]
    print(f"\n📊 RESUMO:")
    print(f"   Features com ganho positivo: {len(positivas)}")
    print(f"   Features com ganho negativo: {len(negativas)}")
    if positivas:
        print(f"   Top 5 positivas: {[r['feature'] for r in positivas[:5]]}")
    if negativas:
        print(f"   Top 5 negativas: {[r['feature'] for r in negativas[:5]]}")
    print(f"\n💡 Recomendação: manter apenas as features com ganho positivo significativo.")
    print(f"   Isso simplifica o modelo e reduz overfitting.")

    return results, baseline_mean

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v75")
    print("   FEATURE IMPORTANCE WALK‑FORWARD")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar auditoria de feature importance (walk‑forward)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            feature_importance_walk_forward(contests)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
