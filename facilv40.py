#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v76
TESTE DE ABLAÇÃO: QUAIS FEATURES REALMENTE SUSTENTAM O MODELO?

OBJETIVO:
✅ Construir modelo de pressão apenas com as features aprovadas no v75
✅ Remover uma feature por vez e medir o impacto no lift (walk‑forward)
✅ Identificar features essenciais, neutras e prejudiciais
✅ Critério: variação > 0.005 → essencial; < -0.005 → prejudicial
✅ Manter apenas o conjunto enxuto que realmente contribui
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

# Features estruturais (parâmetros clássicos)
STRUCTURAL_PARAMS = ['pares', 'moldura', 'repeticoes']

# Features geométricas aprovadas no v75
LINHAS = {
    3: [11, 12, 13, 14, 15],
    4: [16, 17, 18, 19, 20]
}
COLUNAS = {
    2: [2, 7, 12, 17, 22]
}

# Todas as features do modelo (conjunto enxuto)
MODEL_FEATURES = [
    ('pares', 'estrutural'),
    ('moldura', 'estrutural'),
    ('repeticoes', 'estrutural'),
    ('linha_3', 'linha'),
    ('linha_4', 'linha'),
    ('coluna_2', 'coluna')
]

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
    if filter_name == 'repeticoes':
        if prev_dezenas is None: return 8
        return len(set(d) & set(prev_dezenas))
    return 0

def count_in_set(dezenas, elementos):
    return len(set(dezenas) & set(elementos))

# ============================================================
# FEATURE EXTRACTOR (CONJUNTO ENXUTO)
# ============================================================
def extract_feature(game, prev_dezenas, feature_name):
    """Extrai o valor de uma feature específica para um jogo."""
    if feature_name == 'pares':
        return extract_filter(game, 'pares')
    elif feature_name == 'moldura':
        return extract_filter(game, 'moldura')
    elif feature_name == 'repeticoes':
        return extract_filter(game, 'repeticoes', prev_dezenas)
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
# OTIMIZADOR DE CARTEIRA (MODELO DE PRESSÃO ENXUTO)
# ============================================================
class PortfolioOptimizerV76:
    def __init__(self, contests, active_features=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.active_features = active_features if active_features else [f[0] for f in MODEL_FEATURES]

    def _score_game(self, game, prev_dezenas, feature_z_scores):
        """
        Score = soma dos z‑scores das features ativas (subtrai para bonificar atrasados).
        """
        score = 0.0
        for feat_name in self.active_features:
            val = extract_feature(game, prev_dezenas, feat_name)
            z = feature_z_scores.get(feat_name, {}).get(val, 0.0)
            score -= z * 0.2  # peso uniforme para features selecionadas
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
        # Calcular z‑scores para cada feature ativa no treino
        feature_z_scores = {}
        for feat_name in self.active_features:
            feature_z_scores[feat_name] = compute_z_scores(self.contests, feat_name)
        
        prev_dezenas = self.contests[-1]['dezenas'] if self.contests else None
        pool = self.generate_pool(n_candidates, prev_dezenas)
        if len(pool) < n_games:
            return []
        
        scored = []
        for g in pool:
            s = self._score_game(g, prev_dezenas, feature_z_scores)
            scored.append((s, g))
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
# TESTE DE ABLAÇÃO
# ============================================================
def ablation_test(contests, train_size=500, test_size=50, step=50):
    """
    1. Mede o desempenho do modelo completo (todas as features).
    2. Remove uma feature por vez e mede o impacto no lift.
    """
    all_features = [f[0] for f in MODEL_FEATURES]
    print(f"\n🔬 TESTE DE ABLAÇÃO")
    print(f"   Modelo completo: {all_features}")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}\n")

    # 1. Modelo completo
    print("Avaliando modelo completo...")
    full_lifts = []
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        try:
            opt = PortfolioOptimizerV76(train_data, active_features=all_features)
            portfolio = opt.optimize(5, 15000, 2, 2, 1)
            bt = opt.backtest(portfolio, test_data)
            full_lifts.append(bt['lift'])
        except Exception as e:
            full_lifts.append(0.0)
        start += step
    full_mean = np.mean(full_lifts) if full_lifts else 0.0
    print(f"   Lift médio (completo): {full_mean:.4f} ({len(full_lifts)} janelas)\n")

    # 2. Ablação: remover uma feature por vez
    ablation_results = []
    for feature_to_remove in all_features:
        reduced_features = [f for f in all_features if f != feature_to_remove]
        print(f"Removendo '{feature_to_remove}' → features restantes: {reduced_features}")
        lifts = []
        start = train_size
        while start + test_size <= len(contests):
            train_data = contests[start-train_size:start]
            test_data = contests[start:start+test_size]
            try:
                opt = PortfolioOptimizerV76(train_data, active_features=reduced_features)
                portfolio = opt.optimize(5, 15000, 2, 2, 1)
                bt = opt.backtest(portfolio, test_data)
                lifts.append(bt['lift'])
            except Exception as e:
                lifts.append(0.0)
            start += step
        mean_lift = np.mean(lifts) if lifts else 0.0
        impacto = full_mean - mean_lift  # positivo = feature removida faz falta
        ablation_results.append({
            'feature_removed': feature_to_remove,
            'mean_lift': mean_lift,
            'impacto': impacto,
            'n_windows': len(lifts)
        })
        print(f"   Lift sem '{feature_to_remove}': {mean_lift:.4f} (impacto: {impacto:+.4f})\n")

    # Ordenar por impacto (mais negativo = feature mais importante)
    ablation_results.sort(key=lambda x: x['impacto'], reverse=True)

    # Exibir ranking
    print(f"\n📊 RANKING DE IMPORTÂNCIA POR ABLAÇÃO:")
    print(f"   (Impacto positivo = feature é ESSENCIAL; negativo = feature atrapalha)")
    print(f"{'Feature removida':<20} {'Lift s/ feature':<14} {'Impacto':<10} {'Status'}")
    print("-" * 60)
    for res in ablation_results:
        if res['impacto'] > 0.005:
            status = "🔴 ESSENCIAL"
        elif res['impacto'] < -0.005:
            status = "🟢 PREJUDICIAL"
        else:
            status = "⚪ NEUTRA"
        print(f"{res['feature_removed']:<20} {res['mean_lift']:<14.4f} {res['impacto']:<10.4f} {status}")

    # Recomendação
    essenciais = [r for r in ablation_results if r['impacto'] > 0.005]
    prejudiciais = [r for r in ablation_results if r['impacto'] < -0.005]
    print(f"\n💡 RECOMENDAÇÃO:")
    print(f"   Features essenciais (manter): {[r['feature_removed'] for r in essenciais]}")
    print(f"   Features prejudiciais (remover): {[r['feature_removed'] for r in prejudiciais]}")
    if prejudiciais:
        print(f"   Modelo simplificado sugerido: {[f for f in all_features if f not in [r['feature_removed'] for r in prejudiciais]]}")

    return ablation_results, full_mean

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v76")
    print("   TESTE DE ABLAÇÃO: QUAIS FEATURES REALMENTE IMPORTAM?")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Executar teste de ablação (walk‑forward)")
        print("2. Gerar carteira com modelo enxuto (features essenciais)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            ablation_test(contests)

        elif op == '2':
            # Usa apenas as features que o teste de ablação anterior sugeriu (fallback: todas)
            print("\n📝 CONFIGURAÇÃO DA CARTEIRA (MODELO ENXUTO)")
            print("   Features disponíveis: pares, moldura, repeticoes, linha_3, linha_4, coluna_2")
            features_str = input("   Digite as features a usar (ex: pares,moldura,linha_4) ou ENTER para todas: ").strip()
            if features_str:
                active = [f.strip() for f in features_str.split(',')]
            else:
                active = [f[0] for f in MODEL_FEATURES]
            opt = PortfolioOptimizerV76(contests, active_features=active)
            portfolio = opt.optimize(5, 20000, 2, 2, 1)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
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
