#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v78
DIAGNÓSTICO DE PARES: QUAIS VALORES GERAM MAIS 13+?

OBJETIVO:
✅ Rodar walk‑forward com o modelo ponderado (v77)
✅ Registrar, para cada jogo da carteira, o valor de pares e quantos 13+/14+ gerou
✅ Agregar por valor de pares: frequência, acertos de 13+, acertos de 14+
✅ Calcular "taxa de sucesso" de cada valor
✅ Sugerir calibragem alternativa para o peso de pares
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

ACTIVE_FEATURES = ['pares', 'linha_4', 'linha_3', 'coluna_2', 'moldura']

FEATURE_WEIGHTS = {
    'pares': 0.38,
    'linha_4': 0.20,
    'linha_3': 0.15,
    'coluna_2': 0.13,
    'moldura': 0.13
}

LINHAS = {
    3: [11, 12, 13, 14, 15],
    4: [16, 17, 18, 19, 20]
}
COLUNAS = {
    2: [2, 7, 12, 17, 22]
}

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
def count_in_set(dezenas, elementos):
    return len(set(dezenas) & set(elementos))

def extract_feature(game, prev_dezenas, feature_name):
    d = sorted(game)
    if feature_name == 'pares':
        return sum(1 for x in d if x % 2 == 0)
    elif feature_name == 'moldura':
        return sum(1 for x in d if x in MOLDURA)
    elif feature_name == 'linha_3':
        return count_in_set(game, LINHAS[3])
    elif feature_name == 'linha_4':
        return count_in_set(game, LINHAS[4])
    elif feature_name == 'coluna_2':
        return count_in_set(game, COLUNAS[2])
    return 0

def compute_z_scores(train_contests, feature_name):
    series = []
    for i, c in enumerate(train_contests):
        prev = train_contests[i-1]['dezenas'] if i > 0 else None
        series.append(extract_feature(c['dezenas'], prev, feature_name))
    series = np.array(series, dtype=int)
    z_scores = {}
    for val in set(series):
        ocorrencias = np.where(series == val)[0]
        if len(ocorrencias) > 1:
            intervalos = np.diff(ocorrencias)
            intervalo_medio = np.mean(intervalos)
            desvio = np.std(intervalos)
            atraso = len(series) - 1 - ocorrencias[-1]
            if desvio > 0:
                z_scores[val] = (atraso - intervalo_medio) / desvio
            else:
                z_scores[val] = 0.0
        else:
            z_scores[val] = 0.0
    return z_scores

# ============================================================
# OTIMIZADOR DE CARTEIRA (MODELO PONDERADO, IGUAL v77)
# ============================================================
class PortfolioOptimizerV77:
    def __init__(self, contests, feature_weights=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.feature_weights = feature_weights if feature_weights else FEATURE_WEIGHTS
        self.active_features = list(self.feature_weights.keys())

    def _score_game(self, game, prev_dezenas, feature_z_scores):
        score = 0.0
        for feat_name in self.active_features:
            val = extract_feature(game, prev_dezenas, feat_name)
            z = feature_z_scores.get(feat_name, {}).get(val, 0.0)
            weight = self.feature_weights.get(feat_name, 0.2)
            score -= z * weight
        return score

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

    def optimize(self, n_games=5, n_candidates=20000, n_central=2, n_intermed=2, n_perif=1):
        feature_z_scores = {}
        for feat_name in self.active_features:
            feature_z_scores[feat_name] = compute_z_scores(self.contests, feat_name)
        
        prev_dezenas = self.contests[-1]['dezenas'] if self.contests else None
        pool = self.generate_pool(n_candidates, prev_dezenas)
        if len(pool) < n_games:
            return []
        
        scored = [(self._score_game(g, prev_dezenas, feature_z_scores), g) for g in pool]
        scored.sort(key=lambda x: x[0])
        
        n_total = len(scored)
        idx1 = min(n_central * n_total // n_games, n_total)
        idx2 = min((n_central + n_intermed) * n_total // n_games, n_total)
        centrais = [g for _, g in scored[:idx1]][:n_central]
        intermed = [g for _, g in scored[idx1:idx2]][:n_intermed]
        perifs = [g for _, g in scored[idx2:]][:n_perif]
        portfolio = centrais + intermed + perifs
        if len(portfolio) < n_games:
            portfolio = [g for _, g in scored[:n_games]]
        return portfolio[:n_games]

    def backtest(self, portfolio, test_draws):
        if len(portfolio) == 0:
            return {'lift': 0, 'roi': 0, 'hit_distribution': {k:0 for k in range(11,16)}, 'details': []}
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        details = []  # (concurso, hits, pares_do_jogo)
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for i, pm in enumerate(portfolio_masks):
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
                    details.append({
                        'concurso': draw['concurso'],
                        'hits': hits,
                        'pares': extract_feature(portfolio[i], None, 'pares'),
                        'jogo': portfolio[i]
                    })
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {'empirical': prob, 'theoretical': theo_prob,
                'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts, 'details': details}

# ============================================================
# DIAGNÓSTICO DE PARES
# ============================================================
def diagnostic_pares(contests, train_size=500, test_size=50, step=50):
    """
    Walk‑forward com o modelo ponderado (v77).
    Para cada acerto de 11+, registra o valor de pares do jogo.
    Agrega por valor de pares e calcula taxa de sucesso.
    """
    print(f"\n🔬 DIAGNÓSTICO DE PARES: QUAIS VALORES GERAM MAIS 13+?")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}\n")

    # Acumuladores por valor de pares (0 a 12)
    pares_stats = {p: {'total_jogos': 0, 'acertos_11': 0, 'acertos_12': 0, 'acertos_13': 0, 'acertos_14': 0, 'acertos_15': 0} for p in range(0, 13)}
    total_jogos_gerados = 0

    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        try:
            opt = PortfolioOptimizerV77(train_data, feature_weights=FEATURE_WEIGHTS)
            portfolio = opt.optimize(5, 15000, 2, 2, 1)
            bt = opt.backtest(portfolio, test_data)
            # Registrar pares de cada jogo da carteira
            for g in portfolio:
                p = extract_feature(g, None, 'pares')
                pares_stats[p]['total_jogos'] += len(test_data)
                total_jogos_gerados += len(test_data)
            # Registrar acertos detalhados
            for det in bt['details']:
                p = det['pares']
                hits = det['hits']
                if hits == 11:
                    pares_stats[p]['acertos_11'] += 1
                elif hits == 12:
                    pares_stats[p]['acertos_12'] += 1
                elif hits == 13:
                    pares_stats[p]['acertos_13'] += 1
                elif hits == 14:
                    pares_stats[p]['acertos_14'] += 1
                elif hits == 15:
                    pares_stats[p]['acertos_15'] += 1
        except Exception as e:
            pass
        start += step

    # Exibir tabela
    print(f"📊 TAXA DE SUCESSO POR VALOR DE PARES:")
    print(f"{'Pares':<8} {'Jogos':<10} {'11pts':<8} {'12pts':<8} {'13pts':<8} {'14pts':<8} {'Taxa 13+':<12} {'Taxa 14+':<12}")
    print("-" * 85)
    for p in range(0, 13):
        stats = pares_stats[p]
        total = stats['total_jogos']
        if total > 0:
            taxa_13 = (stats['acertos_13'] + stats['acertos_14'] + stats['acertos_15']) / total * 100
            taxa_14 = (stats['acertos_14'] + stats['acertos_15']) / total * 100
            print(f"{p:<8} {total:<10} {stats['acertos_11']:<8} {stats['acertos_12']:<8} {stats['acertos_13']:<8} {stats['acertos_14']:<8} {taxa_13:<12.4f}% {taxa_14:<12.4f}%")
        else:
            print(f"{p:<8} {0:<10} -")

    # Resumo
    total_13_plus = sum(stats['acertos_13'] + stats['acertos_14'] + stats['acertos_15'] for stats in pares_stats.values())
    print(f"\n📊 RESUMO:")
    print(f"   Total de 13+ pontos gerados: {total_13_plus}")
    # Valor de pares mais eficiente
    best_pares = max(range(0, 13), key=lambda p: (pares_stats[p]['acertos_13'] + pares_stats[p]['acertos_14'] + pares_stats[p]['acertos_15']) / max(1, pares_stats[p]['total_jogos']))
    print(f"   Valor de pares com maior taxa de 13+: {best_pares} ({ (pares_stats[best_pares]['acertos_13'] + pares_stats[best_pares]['acertos_14'] + pares_stats[best_pares]['acertos_15']) / max(1, pares_stats[best_pares]['total_jogos']) * 100:.2f}%)")
    
    # Distribuição dos valores de pares nos jogos gerados
    print(f"\n📊 DISTRIBUIÇÃO DOS VALORES DE PARES NOS JOGOS GERADOS:")
    for p in range(0, 13):
        stats = pares_stats[p]
        if stats['total_jogos'] > 0:
            bar = "█" * int(stats['total_jogos'] / max(1, total_jogos_gerados) * 50)
            print(f"   {p}: {stats['total_jogos']:6d} {bar}")

    # Sugestão de calibragem
    print(f"\n💡 SUGESTÃO DE CALIBRAGEM:")
    print(f"   O modelo atual força jogos com z‑score de pares mais negativo (favorecendo atrasados).")
    print(f"   Se os valores de pares com maior taxa de 13+ não estão sendo gerados,")
    print(f"   considere ajustar o peso de pares ou adicionar um termo de recompensa")
    print(f"   para valores específicos de pares com alta taxa de sucesso.")

    return pares_stats

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v78")
    print("   DIAGNÓSTICO DE PARES: QUAIS VALORES GERAM MAIS 13+?")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    diagnostic_pares(contests)

if __name__ == "__main__":
    main()
