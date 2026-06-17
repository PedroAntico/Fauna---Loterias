#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v66
STRUCTURAL PREDICTOR REFINADO + AJUSTES DE SELETIVIDADE

MELHORIAS:
✅ Ajuste 1 – Faixas por percentis (P25–P75) em vez de média±desvio
✅ Ajuste 2 – Peso dos filtros por coeficiente de variação (CV)
✅ Ajuste 3 – Consecutivos removidos (baixo poder preditivo)
✅ Ajuste 4 – Previsão do centro da faixa (alvo único, não intervalo)
✅ Ajuste 5 – Ensemble de múltiplas janelas (20, 50, 100, 200)
✅ Ajuste 6 – Variável "repetidas" adicionada ao predictor
✅ Ajuste 7 – Walk‑forward concurso a concurso com taxa de acerto por filtro
"""

import numpy as np
from scipy.stats import hypergeom, wilcoxon, binomtest
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings
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
# GERADOR (mantido para opções de geração de carteira)
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                     allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                     range_pares=None, range_moldura=None, range_primos=None,
                     range_soma=None, range_amplitude=None, range_consecutivos=None):
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed,
                                      allowed_pares, allowed_moldura, allowed_primos,
                                      range_pares, range_moldura, range_primos,
                                      range_soma, range_amplitude, range_consecutivos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      allowed_pares, allowed_moldura, allowed_primos,
                      range_pares, range_moldura, range_primos,
                      range_soma, range_amplitude, range_consecutivos):
        if fixed is None: fixed = []
        if semifixed is None: semifixed = []
        
        fixed_set = set(fixed)
        semifixed_set = set(semifixed) - fixed_set
        proibidas = fixed_set | semifixed_set
        todas = set(range(1, 26))
        restantes = list(todas - proibidas)
        
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
            if len(game) != 15:
                continue
            
            if allowed_pares is not None and sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                continue
            if allowed_moldura is not None and sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                continue
            if allowed_primos is not None and sum(1 for x in game if x in PRIMES) not in allowed_primos:
                continue
            
            pares = sum(1 for x in game if x % 2 == 0)
            mol = sum(1 for x in game if x in MOLDURA)
            prim = sum(1 for x in game if x in PRIMES)
            soma = sum(game)
            amplitude = max(game) - min(game)
            consec = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)
            
            if range_pares is not None and not (range_pares[0] <= pares <= range_pares[1]): continue
            if range_moldura is not None and not (range_moldura[0] <= mol <= range_moldura[1]): continue
            if range_primos is not None and not (range_primos[0] <= prim <= range_primos[1]): continue
            if range_soma is not None and not (range_soma[0] <= soma <= range_soma[1]): continue
            if range_amplitude is not None and not (range_amplitude[0] <= amplitude <= range_amplitude[1]): continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= consec <= range_consecutivos[1]): continue
            
            return game
        return None

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# OTIMIZADOR DE CARTEIRA (mantido)
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 allowed_pares=None, allowed_moldura=None, allowed_primos=None,
                 range_pares=None, range_moldura=None, range_primos=None,
                 range_soma=None, range_amplitude=None, range_consecutivos=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed if fixed else []
        self.semifixed = semifixed if semifixed else []
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos
        self.range_pares = range_pares
        self.range_moldura = range_moldura
        self.range_primos = range_primos
        self.range_soma = range_soma
        self.range_amplitude = range_amplitude
        self.range_consecutivos = range_consecutivos

    def generate_pool(self, n_candidates):
        pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed,
                    allowed_pares=self.allowed_pares, allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares, range_moldura=self.range_moldura,
                    range_primos=self.range_primos, range_soma=self.range_soma,
                    range_amplitude=self.range_amplitude, range_consecutivos=self.range_consecutivos)
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

    def select_covering(self, candidates, n_select, level='pair'):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        r = 2 if level == 'pair' else 3
        covered, selected = set(), []
        for _ in range(n_select):
            best_idx, best_new = -1, -1
            for i, c in enumerate(candidates):
                if c in selected: continue
                groups = set(combinations(sorted(c), r))
                new_groups = len(groups - covered)
                if new_groups > best_new:
                    best_new, best_idx = new_groups, i
            if best_idx == -1: break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), r))
        return selected

    def optimize(self, n_games=5, n_candidates=50000, method='pair_covering'):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | método: {method}")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.range_pares: print(f"   Pares: {self.range_pares}")
        if self.range_moldura: print(f"   Moldura: {self.range_moldura}")
        if self.range_primos: print(f"   Primos: {self.range_primos}")
        if self.range_soma: print(f"   Soma: {self.range_soma}")
        if self.range_amplitude: print(f"   Amplitude: {self.range_amplitude}")
        t0 = time.time()
        pool = self.generate_pool(n_candidates)
        print(f"   Pool: {len(pool)} jogos")
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")
        portfolio = self.select_covering(pool, n_games, level='pair') if method == 'pair_covering' else (self.select_covering(pool, n_games, level='triple') if method == 'triple_covering' else pool[:n_games])
        print(f"✅ {len(portfolio)} jogos em {time.time()-t0:.1f}s")
        return portfolio

    def backtest(self, portfolio, test_draws):
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
# FUNÇÕES AUXILIARES PARA EXTRAÇÃO DE FILTROS
# ============================================================
def extract_filter(dezenas, filter_name, prev_dezenas=None):
    d = sorted(dezenas)
    if filter_name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if filter_name == 'moldura': return sum(1 for x in d if x in MOLDURA)
    if filter_name == 'primos': return sum(1 for x in d if x in PRIMES)
    if filter_name == 'soma': return sum(d)
    if filter_name == 'consecutivos': return sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
    if filter_name == 'amplitude': return max(d) - min(d)
    if filter_name == 'repeticoes':
        if prev_dezenas is None: return 8
        return len(set(d) & set(prev_dezenas))
    return 0

# ============================================================
# STRUCTURAL PREDICTOR REFINADO (v66)
# ============================================================
class StructuralPredictorV66:
    """
    Predictor estrutural com ensemble de janelas, faixas por percentis,
    peso por CV, inclusão de repetidas e exclusão de consecutivos.
    """
    def __init__(self, contests):
        self.contests = contests
        # Ajuste 2: filtros utilizados (consecutivos removido)
        self.active_filters = ['pares', 'moldura', 'primos', 'soma', 'amplitude', 'repeticoes']
        self.windows = [20, 50, 100, 200]  # Ajuste 5: ensemble de janelas

    def _get_filter_series(self, filter_name):
        """Extrai a série temporal para um filtro."""
        series = []
        for i, c in enumerate(self.contests):
            prev = self.contests[i-1]['dezenas'] if i > 0 else None
            series.append(extract_filter(c['dezenas'], filter_name, prev))
        return np.array(series, dtype=float)

    def predict_ranges(self):
        """
        Retorna ranges previstos usando ensemble de janelas e percentis P25-P75.
        Ajuste 1: percentis em vez de média±desvio
        Ajuste 4: alvo central (P50) + tolerância de ±1
        Ajuste 5: ensemble de janelas
        """
        print(f"\n🔮 STRUCTURAL PREDICTOR v66")
        print(f"   Método: ensemble de janelas ({self.windows}) + percentis P25-P75")
        print(f"   Filtros ativos: {self.active_filters}\n")

        ranges = {}
        for filtro in self.active_filters:
            series = self._get_filter_series(filtro)
            preds = []
            for w in self.windows:
                if len(series) >= w:
                    recent = series[-w:]
                    p25 = np.percentile(recent, 25)
                    p75 = np.percentile(recent, 75)
                    # Centro da faixa como alvo (Ajuste 4)
                    center = int(round(np.median(recent)))
                    preds.append((p25, p75, center))
            
            if not preds:
                continue
            
            # Ensemble: mediana dos centros e dos limites
            centers = [p[2] for p in preds]
            lows = [p[0] for p in preds]
            highs = [p[1] for p in preds]
            
            final_center = int(round(np.median(centers)))
            final_low = int(np.floor(np.median(lows)))
            final_high = int(np.ceil(np.median(highs)))
            
            # Ajuste 2: peso por coeficiente de variação
            cv = np.std(series[-200:]) / (np.mean(series[-200:]) + 1e-10) if len(series) >= 200 else 1.0
            weight = 1.0 / (cv + 0.01)
            
            ranges[filtro] = {
                'low': final_low,
                'high': final_high,
                'center': final_center,
                'cv': cv,
                'weight': weight
            }
            
            print(f"   {filtro:<12}: [{final_low}, {final_high}] (centro={final_center}, cv={cv:.3f}, peso={weight:.1f})")
        
        return ranges

    def predict_single(self, history_upto_t):
        """Prevê faixas usando apenas concursos até t (para walk-forward)."""
        temp_contests = history_upto_t
        # Salva estado original
        original_contests = self.contests
        self.contests = temp_contests
        
        ranges = {}
        for filtro in self.active_filters:
            series = self._get_filter_series(filtro)
            preds = []
            for w in self.windows:
                if len(series) >= w:
                    recent = series[-w:]
                    preds.append(int(round(np.median(recent))))
            if preds:
                center = int(round(np.median(preds)))
                ranges[filtro] = center
            else:
                ranges[filtro] = None
        
        # Restaura
        self.contests = original_contests
        return ranges

# ============================================================
# WALK‑FORWARD CONCURSO A CONCURSO (AJUSTE 7)
# ============================================================
def walk_forward_concurso_a_concurso(contests, min_history=500):
    """
    Para cada concurso a partir de min_history, prevê o valor de cada filtro
    (usando apenas histórico até t) e compara com o valor real em t+1.
    Mede taxa de acerto por filtro.
    """
    print(f"\n🎯 WALK‑FORWARD CONCURSO A CONCURSO (v66)")
    print(f"   Histórico mínimo: {min_history}")
    print(f"   Testes: {len(contests) - min_history - 1} previsões por filtro\n")
    
    predictor = StructuralPredictorV66(contests)
    filters = predictor.active_filters
    
    # Acumuladores
    acertos = {f: 0 for f in filters}
    erros_absolutos = {f: [] for f in filters}
    total = 0
    
    for t in tqdm(range(min_history, len(contests) - 1), desc="Walk‑forward"):
        history = contests[:t+1]
        next_contest = contests[t+1]
        
        # Prever cada filtro
        preds = predictor.predict_single(history)
        
        # Valores reais
        prev_dezenas = contests[t]['dezenas']
        reais = {}
        for f in filters:
            reais[f] = extract_filter(next_contest['dezenas'], f, prev_dezenas)
        
        total += 1
        for f in filters:
            if preds[f] is not None:
                if preds[f] == reais[f]:
                    acertos[f] += 1
                erros_absolutos[f].append(abs(preds[f] - reais[f]))
    
    # Resultados
    print(f"\n📊 TAXA DE ACERTO POR FILTRO (previsão exata do valor):")
    print(f"{'Filtro':<15} {'Acertos':<10} {'Taxa':<10} {'MAE':<10}")
    print("-" * 50)
    for f in filters:
        taxa = acertos[f] / total * 100 if total > 0 else 0
        mae = np.mean(erros_absolutos[f]) if erros_absolutos[f] else 0
        print(f"{f:<15} {acertos[f]:<10} {taxa:<10.1f}% {mae:<10.2f}")
    
    print(f"\n   Total de previsões por filtro: {total}")
    
    # Comparar com baseline ingênuo (prever o valor do concurso anterior)
    print(f"\n📊 BASELINE INGÊNUO (repetir valor do concurso anterior):")
    acertos_base = {f: 0 for f in filters}
    for t in range(min_history, len(contests) - 1):
        curr_vals = {}
        prev_dezenas_curr = contests[t-1]['dezenas'] if t > 0 else None
        for f in filters:
            curr_vals[f] = extract_filter(contests[t]['dezenas'], f, prev_dezenas_curr)
        next_vals = {}
        prev_dezenas_next = contests[t]['dezenas']
        for f in filters:
            next_vals[f] = extract_filter(contests[t+1]['dezenas'], f, prev_dezenas_next)
        for f in filters:
            if curr_vals[f] == next_vals[f]:
                acertos_base[f] += 1
    
    print(f"{'Filtro':<15} {'Acertos':<10} {'Taxa':<10}")
    print("-" * 35)
    for f in filters:
        taxa = acertos_base[f] / total * 100 if total > 0 else 0
        print(f"{f:<15} {acertos_base[f]:<10} {taxa:<10.1f}%")
    
    return acertos, total

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v66")
    print("   STRUCTURAL PREDICTOR REFINADO")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    predictor = StructuralPredictorV66(contests)

    while True:
        print("\nOpções:")
        print("1. Structural Predictor (previsão de faixas)")
        print("2. Walk‑forward concurso a concurso (taxa de acerto)")
        print("3. Gerar carteira com previsões estruturais")
        print("4. Backtest nos últimos 200 concursos")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            predictor.predict_ranges()
        
        elif op == '2':
            try:
                min_hist = int(input("\n   Histórico mínimo [500]: ").strip() or "500")
            except:
                min_hist = 500
            walk_forward_concurso_a_concurso(contests, min_hist)
        
        elif op == '3':
            ranges_dict = predictor.predict_ranges()
            # Converte para tuplas compatíveis com PortfolioOptimizer
            range_pares = (ranges_dict['pares']['low'], ranges_dict['pares']['high']) if 'pares' in ranges_dict else None
            range_moldura = (ranges_dict['moldura']['low'], ranges_dict['moldura']['high']) if 'moldura' in ranges_dict else None
            range_primos = (ranges_dict['primos']['low'], ranges_dict['primos']['high']) if 'primos' in ranges_dict else None
            range_soma = (ranges_dict['soma']['low'], ranges_dict['soma']['high']) if 'soma' in ranges_dict else None
            range_amplitude = (ranges_dict['amplitude']['low'], ranges_dict['amplitude']['high']) if 'amplitude' in ranges_dict else None
            
            print("\n📝 Dezenas fixas (opcional):")
            fixed_str = input("   Fixas (ex: 15 16 20 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            
            gerar = input("   Gerar carteira? (s/n): ").strip().lower()
            if gerar == 's':
                metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
                method = 'pair_covering' if metodo == '1' else 'triple_covering'
                opt = PortfolioOptimizer(contests, fixed=fixed,
                                         range_pares=range_pares,
                                         range_moldura=range_moldura,
                                         range_primos=range_primos,
                                         range_soma=range_soma,
                                         range_amplitude=range_amplitude)
                portfolio = opt.optimize(5, 50000, method=method)
                for i, g in enumerate(portfolio, 1):
                    p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                    print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
                if len(contests) > 200:
                    bt = opt.backtest(portfolio, contests[-200:])
                    print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '4':
            fixed_str = input("\n   Fixas (ENTER para pular): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            opt = PortfolioOptimizer(contests, fixed=fixed)
            portfolio = opt.optimize(5, 50000, method=method)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                  f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)}")
        
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
