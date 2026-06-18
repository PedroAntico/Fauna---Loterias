#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v80
TESTE DE SANIDADE: MODELO v79 vs SELEÇÃO ALEATÓRIA

OBJETIVO:
✅ Walk‑forward comparando o modelo de eficiência (v79) com seleção aleatória
✅ 50 simulações aleatórias por janela para distribuição robusta
✅ Teste de Wilcoxon para verificar significância estatística
✅ Métricas: lift médio, total 13pts, total 14pts, ROI
✅ Responde: o ranking de jogos agrega valor sobre o acaso?
"""

import numpy as np
from scipy.stats import hypergeom, wilcoxon
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

PARES_EFFICIENCY = {
    4: 0.00,
    5: 0.22,
    6: 0.26,
    7: 0.68,
    8: 0.20,
    9: 1.00,
    10: 0.40
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
# OTIMIZADOR DE CARTEIRA v79 (EFICIÊNCIA)
# ============================================================
class PortfolioOptimizerV79:
    def __init__(self, contests, feature_weights=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.feature_weights = feature_weights if feature_weights else FEATURE_WEIGHTS
        self.active_features = list(self.feature_weights.keys())

    def _score_game(self, game, prev_dezenas, feature_z_scores):
        score = 0.0
        for feat_name in self.active_features:
            val = extract_feature(game, prev_dezenas, feat_name)
            weight = self.feature_weights.get(feat_name, 0.2)
            if feat_name == 'pares':
                eff = PARES_EFFICIENCY.get(val, 0.0)
                score -= eff * weight * 2.0
            else:
                z = feature_z_scores.get(feat_name, {}).get(val, 0.0)
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
            if feat_name != 'pares':
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
# TESTE DE SANIDADE: MODELO vs ALEATÓRIO
# ============================================================
def sanity_check(contests, train_size=500, test_size=50, step=50, n_random=50):
    """
    Walk‑forward comparando o modelo v79 com seleção aleatória.
    n_random: quantas carteiras aleatórias são geradas por janela.
    """
    print(f"\n🧪 TESTE DE SANIDADE: MODELO v79 vs SELEÇÃO ALEATÓRIA")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}")
    print(f"   Simulações aleatórias por janela: {n_random}\n")

    results_modelo = []
    results_aleatorio = []

    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]

        # Modelo v79
        try:
            opt = PortfolioOptimizerV79(train_data, feature_weights=FEATURE_WEIGHTS)
            portfolio_modelo = opt.optimize(5, 15000, 2, 2, 1)
            bt_modelo = opt.backtest(portfolio_modelo, test_data)
            results_modelo.append({
                'lift': bt_modelo['lift'],
                'roi': bt_modelo['roi'],
                '11pts': bt_modelo['hit_distribution'].get(11,0),
                '12pts': bt_modelo['hit_distribution'].get(12,0),
                '13pts': bt_modelo['hit_distribution'].get(13,0),
                '14pts': bt_modelo['hit_distribution'].get(14,0),
                '15pts': bt_modelo['hit_distribution'].get(15,0),
            })
        except Exception as e:
            results_modelo.append({'lift': 0, 'roi': 0, '11pts': 0, '12pts': 0, '13pts': 0, '14pts': 0, '15pts': 0})

        # Múltiplas carteiras aleatórias
        aleatorio_lifts = []
        aleatorio_rois = []
        aleatorio_11 = []
        aleatorio_12 = []
        aleatorio_13 = []
        aleatorio_14 = []
        aleatorio_15 = []
        for _ in range(n_random):
            rand_portfolio = [sorted(np.random.choice(range(1,26), 15, replace=False)) for _ in range(5)]
            bt_rand = opt.backtest(rand_portfolio, test_data)
            aleatorio_lifts.append(bt_rand['lift'])
            aleatorio_rois.append(bt_rand['roi'])
            aleatorio_11.append(bt_rand['hit_distribution'].get(11,0))
            aleatorio_12.append(bt_rand['hit_distribution'].get(12,0))
            aleatorio_13.append(bt_rand['hit_distribution'].get(13,0))
            aleatorio_14.append(bt_rand['hit_distribution'].get(14,0))
            aleatorio_15.append(bt_rand['hit_distribution'].get(15,0))

        results_aleatorio.append({
            'lift_mean': np.mean(aleatorio_lifts),
            'lift_std': np.std(aleatorio_lifts),
            'roi_mean': np.mean(aleatorio_rois),
            '11pts_mean': np.mean(aleatorio_11),
            '12pts_mean': np.mean(aleatorio_12),
            '13pts_mean': np.mean(aleatorio_13),
            '14pts_mean': np.mean(aleatorio_14),
            '15pts_mean': np.mean(aleatorio_15),
        })

        l_mod = results_modelo[-1]['lift']
        l_ale = results_aleatorio[-1]['lift_mean']
        print(f"   Janela {start}: modelo(lift={l_mod:.3f}) | aleatório(lift={l_ale:.3f})")
        start += step

    # Consolidação final
    lifts_modelo = [r['lift'] for r in results_modelo]
    lifts_aleatorio = [r['lift_mean'] for r in results_aleatorio]

    print(f"\n📊 RESULTADO FINAL ({len(lifts_modelo)} janelas):")
    print(f"   Modelo v79:")
    print(f"      Lift médio: {np.mean(lifts_modelo):.4f}")
    print(f"      ROI médio: {np.mean([r['roi'] for r in results_modelo]):.1f}%")
    print(f"      Total 13pts: {sum(r['13pts'] for r in results_modelo)}")
    print(f"      Total 14pts: {sum(r['14pts'] for r in results_modelo)}")
    print(f"   Aleatório (média de {n_random} simulações):")
    print(f"      Lift médio: {np.mean(lifts_aleatorio):.4f}")
    print(f"      ROI médio: {np.mean([r['roi_mean'] for r in results_aleatorio]):.1f}%")
    print(f"      Total 13pts: {sum(r['13pts_mean'] for r in results_aleatorio):.1f}")
    print(f"      Total 14pts: {sum(r['14pts_mean'] for r in results_aleatorio):.1f}")

    # Teste de Wilcoxon
    if len(lifts_modelo) >= 5:
        try:
            stat, p_value = wilcoxon(lifts_modelo, lifts_aleatorio)
            print(f"\n📊 TESTE DE WILCOXON (modelo vs aleatório):")
            print(f"   Estatística: {stat:.2f}")
            print(f"   p‑valor: {p_value:.4f}")
            if p_value < 0.05:
                print(f"   🔍 Diferença estatisticamente significativa (p < 0.05).")
                if np.mean(lifts_modelo) > np.mean(lifts_aleatorio):
                    print(f"   ✅ Modelo SUPERIOR ao aleatório.")
                else:
                    print(f"   ❌ Modelo INFERIOR ao aleatório.")
            else:
                print(f"   📊 Sem diferença significativa (p ≥ 0.05).")
                print(f"   O modelo NÃO agrega valor estatístico detectável sobre o acaso.")
        except Exception as e:
            print(f"   Não foi possível calcular Wilcoxon: {e}")

    return results_modelo, results_aleatorio

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v80")
    print("   TESTE DE SANIDADE: MODELO v79 vs SELEÇÃO ALEATÓRIA")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar teste de sanidade (modelo vs aleatório)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            try:
                n_random = int(input("   Quantas simulações aleatórias por janela? [50]: ").strip() or "50")
            except:
                n_random = 50
            sanity_check(contests, n_random=n_random)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
