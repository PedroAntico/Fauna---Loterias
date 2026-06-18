#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v77
MODELO PONDERADO + REMOÇÃO DE REPETICOES

EVOLUÇÃO vs v76:
✅ Remove 'repeticoes' (impacto negativo na ablação)
✅ Pesos das features baseados no impacto real da ablação
✅ Comparação walk‑forward: modelo uniforme (v76) vs modelo ponderado (v77)
✅ Mantém features: pares, linha_4, linha_3, moldura, coluna_2
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter
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

# Features ativas (sem repeticoes)
ACTIVE_FEATURES = ['pares', 'linha_4', 'linha_3', 'coluna_2', 'moldura']

# Pesos baseados no impacto da ablação (v76)
# Impactos: pares=0.0131, linha_4=0.0069, linha_3=0.0051, coluna_2=0.0045, moldura=0.0044
# Normalizados para soma = 1
FEATURE_WEIGHTS = {
    'pares': 0.38,
    'linha_4': 0.20,
    'linha_3': 0.15,
    'coluna_2': 0.13,
    'moldura': 0.13
}

# Pesos uniformes para comparação
UNIFORM_WEIGHTS = {f: 0.20 for f in ACTIVE_FEATURES}

# Geometrias
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
    """Extrai o valor de uma feature específica para um jogo."""
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
    """Calcula z‑score de atraso para cada valor possível da feature."""
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
# OTIMIZADOR DE CARTEIRA v77 (COM PESOS CONFIGURÁVEIS)
# ============================================================
class PortfolioOptimizerV77:
    def __init__(self, contests, feature_weights=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.feature_weights = feature_weights if feature_weights else FEATURE_WEIGHTS
        self.active_features = list(self.feature_weights.keys())

    def _score_game(self, game, prev_dezenas, feature_z_scores):
        """
        Score = soma ponderada dos z‑scores das features ativas.
        Subtrai para bonificar valores atrasados (z positivo).
        """
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
# COMPARAÇÃO WALK‑FORWARD: UNIFORME vs PONDERADO
# ============================================================
def compare_uniform_vs_weighted(contests, train_size=500, test_size=50, step=50):
    print(f"\n🔬 COMPARAÇÃO: MODELO UNIFORME (v76) vs MODELO PONDERADO (v77)")
    print(f"   Features: {ACTIVE_FEATURES}")
    print(f"   Pesos uniformes: {UNIFORM_WEIGHTS}")
    print(f"   Pesos ponderados: {FEATURE_WEIGHTS}")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}\n")

    results = {'uniforme': [], 'ponderado': []}
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        for modelo, pesos in [('uniforme', UNIFORM_WEIGHTS), ('ponderado', FEATURE_WEIGHTS)]:
            try:
                opt = PortfolioOptimizerV77(train_data, feature_weights=pesos)
                portfolio = opt.optimize(5, 15000, 2, 2, 1)
                bt = opt.backtest(portfolio, test_data)
                results[modelo].append({
                    'lift': bt['lift'],
                    'roi': bt['roi'],
                    '13pts': bt['hit_distribution'].get(13, 0),
                    '14pts': bt['hit_distribution'].get(14, 0),
                })
            except Exception as e:
                results[modelo].append({'lift': 0, 'roi': 0, '13pts': 0, '14pts': 0})
        l_unif = results['uniforme'][-1]['lift'] if results['uniforme'] else 0
        l_pond = results['ponderado'][-1]['lift'] if results['ponderado'] else 0
        print(f"   Janela {start}: uniforme(lift={l_unif:.3f}) | ponderado(lift={l_pond:.3f})")
        start += step

    print(f"\n📊 RESULTADO FINAL:")
    for modelo in ['uniforme', 'ponderado']:
        avg_lift = np.mean([r['lift'] for r in results[modelo]]) if results[modelo] else 0
        avg_roi = np.mean([r['roi'] for r in results[modelo]]) if results[modelo] else 0
        total_13 = sum(r['13pts'] for r in results[modelo])
        total_14 = sum(r['14pts'] for r in results[modelo])
        nome = "Modelo uniforme (v76)" if modelo == 'uniforme' else "Modelo ponderado (v77)"
        print(f"   {nome}:")
        print(f"      Média lift: {avg_lift:.4f} | Média ROI: {avg_roi:.1f}%")
        print(f"      Total 13pts: {total_13} | 14pts: {total_14}")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v77")
    print("   MODELO PONDERADO + REMOÇÃO DE REPETICOES")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Walk‑forward comparativo: uniforme vs ponderado")
        print("2. Gerar carteira com modelo ponderado (v77)")
        print("3. Gerar carteira com modelo uniforme (v76)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            compare_uniform_vs_weighted(contests)

        elif op in ('2', '3'):
            pesos = FEATURE_WEIGHTS if op == '2' else UNIFORM_WEIGHTS
            nome = "ponderado (v77)" if op == '2' else "uniforme (v76)"
            print(f"\n📝 CONFIGURAÇÃO DA CARTEIRA ({nome})")
            fixed_str = input("   Dezenas fixas (ex: 15 16 20 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            semifixed_str = input("   Dezenas semifixas (ex: 03 07 14 25 ou ENTER): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            min_semi, max_semi = 0, None
            if semifixed:
                try:
                    min_semi = int(input(f"   Mínimo de semifixas [0-{len(semifixed)}]: ").strip() or "0")
                    max_semi = int(input(f"   Máximo de semifixas [0-{len(semifixed)}]: ").strip() or str(len(semifixed)))
                except:
                    min_semi, max_semi = 0, len(semifixed)
            opt = PortfolioOptimizerV77(contests, feature_weights=pesos)
            portfolio = opt.optimize(5, 20000, 2, 2, 1)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in {2,3,5,7,11,13,17,19,23}); m = sum(1 for x in g if x in MOLDURA)
                rep = len(set(g) & set(contests[-1]['dezenas'])) if contests else 0
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
