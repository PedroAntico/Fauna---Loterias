#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v46
STRUCTURAL PREDICTOR + TESTE DE REVERSÃO/TENDÊNCIA + SIGNIFICÂNCIA

EVOLUÇÃO DO v45:
✅ Teste de duas estratégias: reversão e tendência
✅ Múltiplos tamanhos de bloco (50, 100, 200, 500)
✅ Significância estatística (p-value via binomial)
✅ StructuralPredictor: previsão de faixas para filtros estruturais
✅ Geração de pool condicionada às faixas previstas
✅ Mantém cobertura, walk‑forward, busca OOS
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
# GERADOR COM FIXAS E SEMIFIXAS
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
        
        if max_semifixed is None:
            max_semi = len(semifixed_set)
        else:
            max_semi = min(max_semifixed, len(semifixed_set))
        
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None
        
        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = 15 - n_fixas - n_semifixed_escolher
        
        if n_restantes < 0 or n_restantes > len(restantes):
            return None
        
        for _ in range(200):
            if n_semifixed_escolher > 0 and len(semifixed_set) > 0:
                chosen_semi = set(random.sample(list(semifixed_set), min(n_semifixed_escolher, len(semifixed_set))))
            else:
                chosen_semi = set()
            
            if n_restantes > 0:
                chosen_rest = set(random.sample(restantes, min(n_restantes, len(restantes))))
            else:
                chosen_rest = set()
            
            game = sorted(fixed_set | chosen_semi | chosen_rest)
            
            if len(game) != 15:
                continue
            
            # Filtros por lista (valores exatos)
            if allowed_pares is not None:
                if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                    continue
            
            # Filtros por faixa (range)
            pares = sum(1 for x in game if x % 2 == 0)
            mol = sum(1 for x in game if x in MOLDURA)
            prim = sum(1 for x in game if x in PRIMES)
            soma = sum(game)
            amplitude = max(game) - min(game)
            consec = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)
            
            if range_pares is not None and not (range_pares[0] <= pares <= range_pares[1]):
                continue
            if range_moldura is not None and not (range_moldura[0] <= mol <= range_moldura[1]):
                continue
            if range_primos is not None and not (range_primos[0] <= prim <= range_primos[1]):
                continue
            if range_soma is not None and not (range_soma[0] <= soma <= range_soma[1]):
                continue
            if range_amplitude is not None and not (range_amplitude[0] <= amplitude <= range_amplitude[1]):
                continue
            if range_consecutivos is not None and not (range_consecutivos[0] <= consec <= range_consecutivos[1]):
                continue
            
            return game
        
        return None

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# OTIMIZADOR DE CARTEIRA (COBERTURA)
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
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed,
                    allowed_pares=self.allowed_pares, allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos,
                    range_pares=self.range_pares, range_moldura=self.range_moldura,
                    range_primos=self.range_primos, range_soma=self.range_soma,
                    range_amplitude=self.range_amplitude, range_consecutivos=self.range_consecutivos
                )
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
        covered = set()
        selected = []
        
        for _ in range(n_select):
            best_idx = -1
            best_new = -1
            
            for i, c in enumerate(candidates):
                if c in selected:
                    continue
                groups = set(combinations(sorted(c), r))
                new_groups = len(groups - covered)
                if new_groups > best_new:
                    best_new = new_groups
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), r))
        
        return selected

    def optimize(self, n_games=5, n_candidates=30000, method='pair_covering'):
        print(f"\n🧩 CARTEIRA: {n_games} jogos | método: {method}")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.range_pares: print(f"   Pares: {self.range_pares}")
        if self.range_moldura: print(f"   Moldura: {self.range_moldura}")
        if self.range_primos: print(f"   Primos: {self.range_primos}")
        if self.range_soma: print(f"   Soma: {self.range_soma}")
        if self.range_amplitude: print(f"   Amplitude: {self.range_amplitude}")
        if self.range_consecutivos: print(f"   Consecutivos: {self.range_consecutivos}")
        
        t0 = time.time()
        pool = self.generate_pool(n_candidates)
        print(f"   Pool: {len(pool)} jogos")
        
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")
        
        if method == 'pair_covering':
            portfolio = self.select_covering(pool, n_games, level='pair')
        elif method == 'triple_covering':
            portfolio = self.select_covering(pool, n_games, level='triple')
        else:
            portfolio = pool[:n_games]
        
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
# RANKING DE PODER PREDITIVO (VERSÃO ROBUSTA)
# ============================================================
class PredictiveRanking:
    def __init__(self, contests):
        self.contests = contests
    
    def _extract_filter(self, dezenas, filter_name):
        d = sorted(dezenas)
        if filter_name == 'pares':
            return sum(1 for x in d if x % 2 == 0)
        elif filter_name == 'moldura':
            return sum(1 for x in d if x in MOLDURA)
        elif filter_name == 'primos':
            return sum(1 for x in d if x in PRIMES)
        elif filter_name == 'soma':
            return sum(d)
        elif filter_name == 'consecutivos':
            return sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
        elif filter_name == 'amplitude':
            return max(d) - min(d)
        return 0
    
    def rank_predictive_power(self, block_sizes=None):
        """
        Testa reversão e tendência para múltiplos tamanhos de bloco.
        Calcula significância via teste binomial.
        """
        if block_sizes is None:
            block_sizes = [50, 100, 200, 500]
        
        filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']
        all_results = {}
        
        for block_size in block_sizes:
            print(f"\n📊 BLOCOS DE {block_size} CONCURSOS")
            print(f"{'Filtro':<15} {'Estratégia':<12} {'Precisão':<10} {'Acertos':<10} {'p-value':<10}")
            print("-" * 60)
            
            for filtro in filters:
                # Extrair série
                series = []
                for c in self.contests:
                    series.append(self._extract_filter(c['dezenas'], filtro))
                series = np.array(series, dtype=float)
                
                n_blocos = len(series) // block_size
                if n_blocos < 3:
                    continue
                
                # Dividir em blocos independentes
                blocos = []
                for i in range(n_blocos):
                    start = i * block_size
                    end = start + block_size
                    blocos.append(series[start:end])
                
                # Testar duas estratégias: reversão e tendência
                for strategy in ['reversao', 'tendencia']:
                    acertos = 0
                    total_testes = 0
                    
                    for i in range(1, len(blocos)):
                        mean_prev = np.mean(blocos[i-1])
                        mean_curr = np.mean(blocos[i])
                        historical_mean = np.mean(series[:i*block_size])
                        
                        total_testes += 1
                        
                        if strategy == 'reversao':
                            # Se está acima da média, deve cair
                            predicted_down = mean_prev > historical_mean
                        else:  # tendencia
                            # Se está acima da média, continua acima
                            predicted_down = mean_prev > historical_mean
                            # Na tendência, prevemos que continua na mesma direção
                            predicted_down = not predicted_down
                        
                        if strategy == 'reversao':
                            if predicted_down and mean_curr < mean_prev:
                                acertos += 1
                            elif not predicted_down and mean_curr > mean_prev:
                                acertos += 1
                        else:  # tendencia
                            if mean_prev > historical_mean and mean_curr > mean_prev:
                                acertos += 1
                            elif mean_prev < historical_mean and mean_curr < mean_prev:
                                acertos += 1
                    
                    accuracy = acertos / total_testes * 100 if total_testes > 0 else 0
                    
                    # Significância (teste binomial contra 50%)
                    if total_testes > 0:
                        p_value = binomtest(acertos, total_testes, 0.5, alternative='greater').pvalue
                    else:
                        p_value = 1.0
                    
                    key = (filtro, strategy, block_size)
                    all_results[key] = {
                        'accuracy': accuracy,
                        'acertos': acertos,
                        'total': total_testes,
                        'p_value': p_value
                    }
                    
                    sig = "🔍" if p_value < 0.05 else ("📊" if p_value < 0.15 else "  ")
                    print(f"{filtro:<15} {strategy:<12} {accuracy:<10.1f}% {acertos}/{total_testes:<10} {p_value:<10.4f} {sig}")
        
        # Resumo final: melhor estratégia por filtro
        print(f"\n📊 RESUMO: MELHOR ESTRATÉGIA POR FILTRO")
        print(f"{'Filtro':<15} {'Melhor Est.':<12} {'Precisão':<10} {'Bloco':<10} {'p-value':<10}")
        print("-" * 60)
        for filtro in filters:
            best = None
            best_acc = 0
            for key, res in all_results.items():
                if key[0] == filtro and res['accuracy'] > best_acc:
                    best = key
                    best_acc = res['accuracy']
            if best:
                res = all_results[best]
                sig = "🔍" if res['p_value'] < 0.05 else ""
                print(f"{filtro:<15} {best[1]:<12} {res['accuracy']:<10.1f}% {best[2]:<10} {res['p_value']:<10.4f} {sig}")
        
        return all_results
    
    def rank_dezenas_individual(self, block_sizes=None):
        """Testa poder preditivo das 25 dezenas individuais."""
        if block_sizes is None:
            block_sizes = [100, 200, 500]
        
        print(f"\n📊 PODER PREDITIVO DAS 25 DEZENAS")
        
        results = {}
        for block_size in block_sizes:
            accuracies = []
            for dezena in range(1, 26):
                series = np.array([1 if dezena in c['dezenas'] else 0 for c in self.contests], dtype=float)
                n_blocos = len(series) // block_size
                if n_blocos < 3:
                    continue
                
                blocos = []
                for i in range(n_blocos):
                    blocos.append(series[i*block_size:(i+1)*block_size])
                
                acertos = 0
                total = 0
                for i in range(1, len(blocos)):
                    freq_prev = np.mean(blocos[i-1])
                    freq_curr = np.mean(blocos[i])
                    freq_hist = np.mean(series[:i*block_size])
                    
                    total += 1
                    # Estratégia de reversão (mais comum)
                    if freq_prev > freq_hist and freq_curr < freq_prev:
                        acertos += 1
                    elif freq_prev < freq_hist and freq_curr > freq_prev:
                        acertos += 1
                
                if total > 0:
                    accuracies.append(acertos / total * 100)
            
            if accuracies:
                avg_acc = np.mean(accuracies)
                best_dezena = np.argmax(accuracies) + 1
                worst_dezena = np.argmin(accuracies) + 1
                print(f"\n   Bloco {block_size}: média={avg_acc:.1f}%, melhor={best_dezena} ({max(accuracies):.1f}%), pior={worst_dezena} ({min(accuracies):.1f}%)")
        
        return results

# ============================================================
# STRUCTURAL PREDICTOR (NOVO v46)
# ============================================================
class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests
        self.filters = {
            'pares': {'min': 3, 'max': 12, 'typical': (6, 9)},
            'moldura': {'min': 6, 'max': 15, 'typical': (8, 11)},
            'primos': {'min': 2, 'max': 9, 'typical': (4, 7)},
            'soma': {'min': 120, 'max': 270, 'typical': (170, 220)},
            'amplitude': {'min': 10, 'max': 24, 'typical': (20, 24)},
            'consecutivos': {'min': 0, 'max': 12, 'typical': (4, 8)},
        }
    
    def predict_ranges(self, method='recent'):
        """
        Prediz faixas para os filtros estruturais.
        method: 'recent' (últimos 50 concursos), 'ipe' (tendência via IPE)
        """
        print(f"\n🔮 STRUCTURAL PREDICTOR (método: {method})")
        
        ranges = {}
        
        for filtro, info in self.filters.items():
            # Extrair série
            series = []
            for c in self.contests:
                d = c['dezenas']
                if filtro == 'pares':
                    val = sum(1 for x in d if x % 2 == 0)
                elif filtro == 'moldura':
                    val = sum(1 for x in d if x in MOLDURA)
                elif filtro == 'primos':
                    val = sum(1 for x in d if x in PRIMES)
                elif filtro == 'soma':
                    val = sum(d)
                elif filtro == 'consecutivos':
                    val = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
                elif filtro == 'amplitude':
                    val = max(d) - min(d)
                else:
                    val = 0
                series.append(val)
            series = np.array(series, dtype=float)
            
            if method == 'recent':
                # Usa média e desvio dos últimos 50 concursos
                recent = series[-50:]
                mean_val = np.mean(recent)
                std_val = np.std(recent)
                
                low = max(info['min'], int(mean_val - std_val))
                high = min(info['max'], int(mean_val + std_val) + 1)
                ranges[filtro] = (low, high)
                
                print(f"   {filtro:<15}: [{low}, {high}] (média recente={mean_val:.1f})")
            
            elif method == 'ipe':
                # Usa IPE: frequência curta vs longa
                freq_short = np.mean(series[-20:])
                freq_long = np.mean(series[-500:]) if len(series) >= 500 else np.mean(series)
                
                if freq_long > 0:
                    ipe = (freq_short - freq_long) / freq_long * 100
                else:
                    ipe = 0.0
                
                # Se IPE positivo (aquecido), espera reversão (diminuir)
                if ipe > 5:
                    predicted = max(info['min'], int(freq_long))
                    ranges[filtro] = (predicted - 1, predicted + 1)
                elif ipe < -5:
                    predicted = min(info['max'], int(freq_long) + 1)
                    ranges[filtro] = (predicted - 1, predicted + 1)
                else:
                    # Neutro: usa intervalo típico
                    ranges[filtro] = info['typical']
                
                print(f"   {filtro:<15}: {ranges[filtro]} (IPE={ipe:+.1f}%)")
        
        return ranges

# ============================================================
# SUGESTÃO AUTOMÁTICA (mantida e melhorada)
# ============================================================
def suggest_from_structural_predictor(contests):
    """Sugere fixas e filtros baseado no StructuralPredictor."""
    predictor = StructuralPredictor(contests)
    ranges = predictor.predict_ranges(method='recent')
    
    # Usar trinca fixa que já mostrou bom desempenho
    fixed = [15, 16, 20]
    print(f"\n💡 SUGESTÃO AUTOMÁTICA:")
    print(f"   Fixas: {fixed}")
    print(f"   Faixas estruturais: {ranges}")
    
    return fixed, ranges

# ============================================================
# BUSCA OOS E COMPARAÇÕES (mantidas)
# ============================================================
def search_best_fixed_oos(contests, n_fixed=3, top_n=20, train_size=3500, n_games=5, n_candidates=10000, method='pair_covering'):
    print(f"\n🔎 BUSCANDO MELHORES {n_fixed} FIXAS (OUT-OF-SAMPLE)")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    
    candidates = []
    all_combos = list(combinations(range(1, 26), n_fixed))
    for fixed_tuple in tqdm(all_combos, desc="Filtrando"):
        fixed_set = set(fixed_tuple)
        acertos = sum(1 for c in train_data if fixed_set.issubset(set(c['dezenas'])))
        freq = acertos / len(train_data)
        if freq >= 0.05:
            candidates.append((fixed_tuple, freq, acertos))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:200]
    
    results = []
    for fixed_tuple, freq, acertos in tqdm(top_candidates, desc="Backtest OOS"):
        fixed = list(fixed_tuple)
        opt = PortfolioOptimizer(train_data, fixed=fixed)
        try:
            portfolio = opt.optimize(n_games, n_candidates, method=method)
            bt = opt.backtest(portfolio, test_data)
            results.append({
                'fixed': fixed_tuple, 'freq_treino': freq,
                'lift': bt['lift'], 'roi': bt['roi'],
                '14pts': bt['hit_distribution'].get(14,0),
                '13pts': bt['hit_distribution'].get(13,0),
            })
        except: continue
    
    results.sort(key=lambda x: x['roi'], reverse=True)
    print(f"\n🏆 TOP {top_n} FIXAS POR ROI (OUT-OF-SAMPLE):")
    for i, res in enumerate(results[:top_n], 1):
        print(f"{i:<5} {str(res['fixed']):<20} ROI={res['roi']:<10.1f}% 13pts={res['13pts']} 14pts={res['14pts']}")
    return results

def compare_trincas(contests, trinca1, trinca2, n_games=5, n_candidates=30000, method='pair_covering'):
    print(f"\n⚔️ COMPARAÇÃO DE TRINCAS")
    for i, trinca in enumerate([trinca1, trinca2], 1):
        print(f"\n--- Trinca {i}: {trinca} ---")
        opt = PortfolioOptimizer(contests, fixed=list(trinca))
        portfolio = opt.optimize(n_games, n_candidates, method=method)
        bt = opt.backtest(portfolio, contests[-200:])
        print(f"   Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

def walk_forward_conditional(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                             fixed=None, ranges=None, method='pair_covering'):
    print(f"\n🔬 WALK-FORWARD CONDICIONAL ({n_windows} janelas)")
    results = []
    for w in range(n_windows):
        test_end = len(contests) - w * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = max(0, train_end - train_size)
        if train_start >= train_end or test_start >= test_end: continue
        train_data = contests[train_start:train_end]
        test_data = contests[test_start:test_end]
        if len(train_data) < 100 or len(test_data) < 5: continue
        
        opt = PortfolioOptimizer(train_data, fixed=fixed,
                                 range_pares=ranges.get('pares') if ranges else None,
                                 range_moldura=ranges.get('moldura') if ranges else None,
                                 range_primos=ranges.get('primos') if ranges else None,
                                 range_soma=ranges.get('soma') if ranges else None,
                                 range_amplitude=ranges.get('amplitude') if ranges else None,
                                 range_consecutivos=ranges.get('consecutivos') if ranges else None)
        portfolio = opt.optimize(n_games, n_candidates=10000, method=method)
        bt = opt.backtest(portfolio, test_data)
        results.append({
            'window': w, 'lift': bt['lift'], 'roi': bt['roi'],
            '14pts': bt['hit_distribution'].get(14,0),
        })
        print(f"   Janela {w}: lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}% | 14pts={bt['hit_distribution'].get(14,0)}")
    if results:
        print(f"\n📊 RESUMO: Média lift: {np.mean([r['lift'] for r in results]):.3f}")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v46")
    print("   STRUCTURAL PREDICTOR + REVERSÃO/TENDÊNCIA + SIGNIFICÂNCIA")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    ranker = PredictiveRanking(contests)
    predictor = StructuralPredictor(contests)

    while True:
        print("\nOpções:")
        print("1. Gerar carteira personalizada")
        print("2. Walk‑forward condicional")
        print("3. Backtest nos últimos 200 concursos")
        print("4. Buscar melhores fixas (out‑of‑sample)")
        print("5. Comparar duas trincas")
        print("6. Ranking de poder preditivo (reversão vs tendência)")
        print("7. Poder preditivo das 25 dezenas")
        print("8. Structural Predictor (previsão de faixas)")
        print("9. Gerar carteira com previsões estruturais")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            print("\n📝 CONFIGURAÇÃO DA CARTEIRA")
            fixed_str = input("   Dezenas fixas (ex: 15 16 20 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            
            print("   Faixas estruturais (ENTER para pular)")
            try:
                pares_str = input("   Pares min,max (ex: 7,9): ").strip()
                range_pares = tuple(int(x) for x in pares_str.split(',')) if pares_str else None
            except: range_pares = None
            try:
                moldura_str = input("   Moldura min,max: ").strip()
                range_moldura = tuple(int(x) for x in moldura_str.split(',')) if moldura_str else None
            except: range_moldura = None
            try:
                primos_str = input("   Primos min,max: ").strip()
                range_primos = tuple(int(x) for x in primos_str.split(',')) if primos_str else None
            except: range_primos = None
            try:
                soma_str = input("   Soma min,max: ").strip()
                range_soma = tuple(int(x) for x in soma_str.split(',')) if soma_str else None
            except: range_soma = None
            
            print("\n   Método: 1. Pair Covering  2. Triple Covering")
            metodo = input("   Escolha [1]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            
            opt = PortfolioOptimizer(contests, fixed=fixed,
                                     range_pares=range_pares, range_moldura=range_moldura,
                                     range_primos=range_primos, range_soma=range_soma)
            portfolio = opt.optimize(5, 30000, method=method)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '2':
            fixed_str = input("\n   Fixas (ex: 15 16 20): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            walk_forward_conditional(contests, fixed=fixed, method=method)
        
        elif op == '3':
            fixed_str = input("\n   Fixas (ENTER para pular): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            opt = PortfolioOptimizer(contests, fixed=fixed)
            portfolio = opt.optimize(5, 30000, method=method)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                  f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)}")
        
        elif op == '4':
            try:
                n_fixed = int(input("\n   Quantas fixas (2,3,4): ").strip())
                if n_fixed not in [2,3,4]: continue
            except: continue
            top_n = int(input("   Resultados [20]: ").strip() or "20")
            train_size = int(input("   Tamanho treino [3500]: ").strip() or "3500")
            metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            search_best_fixed_oos(contests, n_fixed, top_n, train_size, method=method)
        
        elif op == '5':
            trinca1_str = input("\n   Trinca 1: ").strip()
            trinca2_str = input("   Trinca 2: ").strip()
            try:
                trinca1 = tuple(int(x) for x in trinca1_str.split())
                trinca2 = tuple(int(x) for x in trinca2_str.split())
                if len(trinca1)!=3 or len(trinca2)!=3: continue
            except: continue
            compare_trincas(contests, trinca1, trinca2)
        
        elif op == '6':
            print("\n📝 TAMANHOS DE BLOCO (ex: 50,100,200,500)")
            blocos_str = input("   [50,100,200,500]: ").strip() or "50,100,200,500"
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')]
            except:
                block_sizes = [50, 100, 200, 500]
            ranker.rank_predictive_power(block_sizes)
        
        elif op == '7':
            print("\n📝 TAMANHOS DE BLOCO (ex: 100,200,500)")
            blocos_str = input("   [100,200,500]: ").strip() or "100,200,500"
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')]
            except:
                block_sizes = [100, 200, 500]
            ranker.rank_dezenas_individual(block_sizes)
        
        elif op == '8':
            print("\n📝 MÉTODO DE PREVISÃO")
            print("   1. Recente (últimos 50 concursos)")
            print("   2. IPE (tendência via Índice de Pressão)")
            metodo = input("   Escolha [1]: ").strip() or "1"
            method = 'recent' if metodo == '1' else 'ipe'
            predictor.predict_ranges(method=method)
        
        elif op == '9':
            fixed, ranges = suggest_from_structural_predictor(contests)
            gerar = input("\n   Gerar carteira com estas previsões? (s/n): ").strip().lower()
            if gerar == 's':
                metodo = input("   Método [1. Pair, 2. Triple]: ").strip() or "1"
                method = 'pair_covering' if metodo == '1' else 'triple_covering'
                opt = PortfolioOptimizer(contests, fixed=fixed,
                                         range_pares=ranges.get('pares'),
                                         range_moldura=ranges.get('moldura'),
                                         range_primos=ranges.get('primos'),
                                         range_soma=ranges.get('soma'),
                                         range_amplitude=ranges.get('amplitude'),
                                         range_consecutivos=ranges.get('consecutivos'))
                portfolio = opt.optimize(5, 30000, method=method)
                for i, g in enumerate(portfolio, 1):
                    p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                    print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
                if len(contests) > 200:
                    bt = opt.backtest(portfolio, contests[-200:])
                    print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
