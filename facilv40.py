#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v79
EFICIÊNCIA HISTÓRICA + DIAGNÓSTICO DE INTERAÇÃO

EVOLUÇÃO vs v78:
✅ Score de pares agora usa eficiência (taxa de 13+) em vez de z‑score de atraso
✅ Diagnóstico de interação: pares × linha_4
✅ Walk‑forward comparativo: modelo com z‑score (v77) vs modelo com eficiência (v79)
✅ Mantém features: pares, linha_4, linha_3, coluna_2, moldura (pesos calibrados)
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

# Eficiência dos pares (calculada no v78)
# Taxa de 13+ por valor de pares (normalizada para o máximo = 1.0)
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
    """Z‑score de atraso (mantido para as outras features)."""
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
# OTIMIZADOR DE CARTEIRA v79 (EFICIÊNCIA PARA PARES)
# ============================================================
class PortfolioOptimizerV79:
    def __init__(self, contests, feature_weights=None, use_efficiency=True):
        self.contests = contests
        self.generator = LooseGenerator()
        self.feature_weights = feature_weights if feature_weights else FEATURE_WEIGHTS
        self.active_features = list(self.feature_weights.keys())
        self.use_efficiency = use_efficiency

    def _score_game(self, game, prev_dezenas, feature_z_scores):
        """
        Score combinado:
        - pares: eficiência histórica (se use_efficiency=True) ou z‑score (se False)
        - demais features: z‑score de atraso
        """
        score = 0.0
        for feat_name in self.active_features:
            val = extract_feature(game, prev_dezenas, feat_name)
            weight = self.feature_weights.get(feat_name, 0.2)
            
            if feat_name == 'pares' and self.use_efficiency:
                # Usa eficiência: valor mais alto = melhor (subtrai para score menor)
                eff = PARES_EFFICIENCY.get(val, 0.0)
                score -= eff * weight * 2.0  # fator 2 para equiparar escala com z‑score
            else:
                # Z‑score de atraso (mantido para as outras features)
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
        # Z‑scores para as features (exceto pares, se use_efficiency=True)
        feature_z_scores = {}
        for feat_name in self.active_features:
            if feat_name == 'pares' and self.use_efficiency:
                continue  # pares usa eficiência, não z‑score
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
        details = []
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
                        'linha_4': extract_feature(portfolio[i], None, 'linha_4'),
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
# DIAGNÓSTICO DE INTERAÇÃO: PARES × LINHA_4
# ============================================================
def diagnostic_interaction(contests, train_size=500, test_size=50, step=50):
    """
    Walk‑forward com o modelo de eficiência (v79).
    Para cada acerto de 11+, registra pares e linha_4.
    Agrega por combinação (pares, linha_4) e calcula taxa de 13+.
    """
    print(f"\n🔬 DIAGNÓSTICO DE INTERAÇÃO: PARES × LINHA_4")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}\n")

    # Acumuladores por (pares, linha_4)
    inter_stats = defaultdict(lambda: {'total_jogos': 0, 'acertos_13': 0, 'acertos_14': 0, 'acertos_15': 0})

    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        try:
            opt = PortfolioOptimizerV79(train_data, feature_weights=FEATURE_WEIGHTS, use_efficiency=True)
            portfolio = opt.optimize(5, 15000, 2, 2, 1)
            bt = opt.backtest(portfolio, test_data)
            # Registrar cada jogo da carteira
            for g in portfolio:
                p = extract_feature(g, None, 'pares')
                l4 = extract_feature(g, None, 'linha_4')
                inter_stats[(p, l4)]['total_jogos'] += len(test_data)
            # Registrar acertos detalhados
            for det in bt['details']:
                p = det['pares']
                l4 = det['linha_4']
                hits = det['hits']
                if hits == 13:
                    inter_stats[(p, l4)]['acertos_13'] += 1
                elif hits == 14:
                    inter_stats[(p, l4)]['acertos_14'] += 1
                elif hits == 15:
                    inter_stats[(p, l4)]['acertos_15'] += 1
        except Exception as e:
            pass
        start += step

    # Exibir tabela para combinações mais frequentes
    print(f"📊 TAXA DE 13+ POR COMBINAÇÃO (PARES, LINHA_4):")
    print(f"{'Pares':<8} {'Linha_4':<10} {'Jogos':<10} {'13pts':<8} {'14pts':<8} {'15pts':<8} {'Taxa 13+':<12}")
    print("-" * 70)
    
    # Ordenar por total de jogos
    sorted_items = sorted(inter_stats.items(), key=lambda x: x[1]['total_jogos'], reverse=True)
    for (p, l4), stats in sorted_items:
        total = stats['total_jogos']
        if total > 0:
            taxa_13 = (stats['acertos_13'] + stats['acertos_14'] + stats['acertos_15']) / total * 100
            if taxa_13 > 0.05 or stats['acertos_14'] > 0:  # filtra combinações com algum sinal
                print(f"{p:<8} {l4:<10} {total:<10} {stats['acertos_13']:<8} {stats['acertos_14']:<8} {stats['acertos_15']:<8} {taxa_13:<12.4f}%")

    # Destacar melhores combinações
    print(f"\n💡 MELHORES COMBINAÇÕES (TAXA 13+ > 0.2%):")
    best_combos = [(k, v) for k, v in inter_stats.items() 
                   if v['total_jogos'] > 0 and (v['acertos_13'] + v['acertos_14'] + v['acertos_15']) / v['total_jogos'] * 100 > 0.2]
    best_combos.sort(key=lambda x: (x[1]['acertos_13'] + x[1]['acertos_14'] + x[1]['acertos_15']) / x[1]['total_jogos'], reverse=True)
    for (p, l4), stats in best_combos[:10]:
        taxa = (stats['acertos_13'] + stats['acertos_14'] + stats['acertos_15']) / stats['total_jogos'] * 100
        print(f"   Pares={p}, Linha_4={l4}: taxa={taxa:.4f}% ({stats['acertos_13']}×13, {stats['acertos_14']}×14, {stats['acertos_15']}×15)")

    return inter_stats

# ============================================================
# COMPARAÇÃO WALK‑FORWARD: Z‑SCORE vs EFICIÊNCIA
# ============================================================
def compare_zscore_vs_efficiency(contests, train_size=500, test_size=50, step=50):
    print(f"\n🔬 COMPARAÇÃO: Z‑SCORE (v77) vs EFICIÊNCIA (v79)")
    print(f"   Treino: {train_size} | Teste: {test_size} | Passo: {step}\n")

    results = {'zscore': [], 'efficiency': []}
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        for modelo, use_eff in [('zscore', False), ('efficiency', True)]:
            try:
                opt = PortfolioOptimizerV79(train_data, feature_weights=FEATURE_WEIGHTS, use_efficiency=use_eff)
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
        l_z = results['zscore'][-1]['lift'] if results['zscore'] else 0
        l_eff = results['efficiency'][-1]['lift'] if results['efficiency'] else 0
        print(f"   Janela {start}: z‑score(lift={l_z:.3f}) | eficiência(lift={l_eff:.3f})")
        start += step

    print(f"\n📊 RESULTADO FINAL:")
    for modelo in ['zscore', 'efficiency']:
        avg_lift = np.mean([r['lift'] for r in results[modelo]]) if results[modelo] else 0
        avg_roi = np.mean([r['roi'] for r in results[modelo]]) if results[modelo] else 0
        total_13 = sum(r['13pts'] for r in results[modelo])
        total_14 = sum(r['14pts'] for r in results[modelo])
        nome = "Z‑score (v77)" if modelo == 'zscore' else "Eficiência (v79)"
        print(f"   {nome}:")
        print(f"      Média lift: {avg_lift:.4f} | Média ROI: {avg_roi:.1f}%")
        print(f"      Total 13pts: {total_13} | 14pts: {total_14}")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v79")
    print("   EFICIÊNCIA HISTÓRICA + DIAGNÓSTICO DE INTERAÇÃO")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Walk‑forward comparativo: z‑score vs eficiência")
        print("2. Diagnóstico de interação: pares × linha_4")
        print("3. Gerar carteira com modelo de eficiência (v79)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            compare_zscore_vs_efficiency(contests)

        elif op == '2':
            diagnostic_interaction(contests)

        elif op == '3':
            print("\n📝 CONFIGURAÇÃO DA CARTEIRA (MODELO DE EFICIÊNCIA)")
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
            opt = PortfolioOptimizerV79(contests, feature_weights=FEATURE_WEIGHTS, use_efficiency=True)
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
