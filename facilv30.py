#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v47
CONTROLE MONTE CARLO + TESTE CONCURSO A CONCURSO + WALK‑FORWARD DO PREDITOR

EVOLUÇÃO DO v46:
✅ Opção 10: Controle Monte Carlo (baseline aleatório)
✅ Opção 11: Teste preditivo concurso a concurso
✅ Opção 12: Walk‑forward do Structural Predictor
✅ Ranking com comparação real vs. simulado
✅ Mantém cobertura, busca OOS, fixas, semifixas
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
# GERADOR COM FIXAS, SEMIFIXAS E FAIXAS
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
            
            if allowed_pares is not None:
                if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                    continue
            
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
        if self.semifixed: print( f"   Semifixas: {self.semifixed} "
                                 f"(mín={self.min_semifixed}, máx={self.max_semifixed})" )
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
# FUNÇÕES DE EXTRAÇÃO DE FILTROS
# ============================================================
def extract_filter(dezenas, filter_name):
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

# ============================================================
# RANKING DE PODER PREDITIVO (COM BASELINE MONTE CARLO)
# ============================================================
class PredictiveRanking:
    def __init__(self, contests):
        self.contests = contests
    
    def rank_predictive_power(self, block_sizes=None):
        """Testa reversão e tendência para múltiplos tamanhos de bloco."""
        if block_sizes is None:
            block_sizes = [50, 100, 200, 500]
        
        filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']
        all_results = {}
        
        for block_size in block_sizes:
            print(f"\n📊 BLOCOS DE {block_size} CONCURSOS")
            print(f"{'Filtro':<15} {'Estratégia':<12} {'Precisão':<10} {'Acertos':<10} {'p-value':<10}")
            print("-" * 60)
            
            for filtro in filters:
                series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
                
                n_blocos = len(series) // block_size
                if n_blocos < 3:
                    continue
                
                blocos = []
                for i in range(n_blocos):
                    blocos.append(series[i*block_size:(i+1)*block_size])
                
                for strategy in ['reversao', 'tendencia']:
                    acertos = 0
                    total_testes = 0
                    
                    for i in range(1, len(blocos)):
                        mean_prev = np.mean(blocos[i-1])
                        mean_curr = np.mean(blocos[i])
                        historical_mean = np.mean(series[:i*block_size])
                        
                        total_testes += 1
                        
                        if strategy == 'reversao':
                            predicted_down = mean_prev > historical_mean
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
                    p_value = binomtest(acertos, total_testes, 0.5, alternative='greater').pvalue if total_testes > 0 else 1.0
                    
                    all_results[(filtro, strategy, block_size)] = {
                        'accuracy': accuracy, 'acertos': acertos, 'total': total_testes, 'p_value': p_value
                    }
                    
                    sig = "🔍" if p_value < 0.05 else ("📊" if p_value < 0.15 else "  ")
                    print(f"{filtro:<15} {strategy:<12} {accuracy:<10.1f}% {acertos}/{total_testes:<10} {p_value:<10.4f} {sig}")
        
        return all_results
    
    def rank_dezenas_individual(self, block_sizes=None):
        """Testa poder preditivo das 25 dezenas."""
        if block_sizes is None:
            block_sizes = [100, 200, 500]
        
        print(f"\n📊 PODER PREDITIVO DAS 25 DEZENAS")
        for block_size in block_sizes:
            accuracies = []
            for dezena in range(1, 26):
                series = np.array([1 if dezena in c['dezenas'] else 0 for c in self.contests], dtype=float)
                n_blocos = len(series) // block_size
                if n_blocos < 3:
                    continue
                
                blocos = [series[i*block_size:(i+1)*block_size] for i in range(n_blocos)]
                acertos = 0
                total = 0
                for i in range(1, len(blocos)):
                    freq_prev = np.mean(blocos[i-1])
                    freq_curr = np.mean(blocos[i])
                    freq_hist = np.mean(series[:i*block_size])
                    total += 1
                    if freq_prev > freq_hist and freq_curr < freq_prev:
                        acertos += 1
                    elif freq_prev < freq_hist and freq_curr > freq_prev:
                        acertos += 1
                if total > 0:
                    accuracies.append(acertos / total * 100)
            
            if accuracies:
                print(f"\n   Bloco {block_size}: média={np.mean(accuracies):.1f}%, "
                      f"melhor={np.argmax(accuracies)+1} ({max(accuracies):.1f}%), "
                      f"pior={np.argmin(accuracies)+1} ({min(accuracies):.1f}%)")
        
        return None

# ============================================================
# CONTROLE MONTE CARLO (NOVO v47)
# ============================================================
def monte_carlo_control(contests, n_simulations=100, block_sizes=None):
    """
    Gera n_simulations séries aleatórias de mesmo tamanho que contests,
    executa o ranking de poder preditivo e compara com o resultado real.
    """
    if block_sizes is None:
        block_sizes = [50, 100, 200]
    
    n_concursos = len(contests)
    print(f"\n🎲 CONTROLE MONTE CARLO")
    print(f"   Simulações: {n_simulations}")
    print(f"   Cada simulação: {n_concursos} concursos puramente aleatórios")
    print(f"   Blocos testados: {block_sizes}\n")
    
    # Executar no dataset real primeiro
    ranker_real = PredictiveRanking(contests)
    real_results = ranker_real.rank_predictive_power(block_sizes)
    
    # Agregar resultados reais por bloco e estratégia
    real_summary = {}
    for (filtro, strategy, block_size), res in real_results.items():
        key = (filtro, strategy, block_size)
        real_summary[key] = res['accuracy']
    
    # Executar nas simulações
    sim_accuracies = defaultdict(list)
    
    for sim in tqdm(range(n_simulations), desc="Simulações Monte Carlo"):
        # Gerar concursos aleatórios
        sim_contests = []
        for _ in range(n_concursos):
            sim_contests.append({'dezenas': sorted(np.random.choice(range(1, 26), 15, replace=False))})
        
        ranker_sim = PredictiveRanking(sim_contests)
        sim_results = ranker_sim.rank_predictive_power(block_sizes)
        
        for (filtro, strategy, block_size), res in sim_results.items():
            key = (filtro, strategy, block_size)
            sim_accuracies[key].append(res['accuracy'])
    
    # Comparar
    print(f"\n📊 COMPARAÇÃO REAL vs. MONTE CARLO:")
    print(f"{'Filtro':<15} {'Estratégia':<12} {'Bloco':<8} {'Real':<10} {'MC Médio':<10} {'MC Std':<10} {'Diferença':<10} {'p-value':<10}")
    print("-" * 90)
    
    for key, real_acc in sorted(real_summary.items(), key=lambda x: x[1], reverse=True):
        filtro, strategy, block_size = key
        sim_accs = sim_accuracies.get(key, [])
        if not sim_accs:
            continue
        
        mean_sim = np.mean(sim_accs)
        std_sim = np.std(sim_accs)
        diff = real_acc - mean_sim
        
        # p-value empírico: quantas simulações superaram o real?
        p_emp = np.mean(np.array(sim_accs) >= real_acc)
        
        sig = "🔍" if p_emp < 0.05 else ""
        print(f"{filtro:<15} {strategy:<12} {block_size:<8} {real_acc:<10.1f}% {mean_sim:<10.1f}% {std_sim:<10.1f} {diff:<10.1f}% {p_emp:<10.4f} {sig}")
    
    print(f"\n🔍 Interpretação:")
    print(f"   Se 'Diferença' for pequena e p-value > 0.05, o resultado real é compatível com aleatoriedade.")
    print(f"   Se 'Diferença' for grande e p-value < 0.05, há evidência de sinal preditivo genuíno.")
    
    return real_summary, sim_accuracies

# ============================================================
# TESTE CONCURSO A CONCURSO (NOVO v47)
# ============================================================
def test_concurso_a_concurso(contests, min_history=200):
    """
    Para cada concurso a partir de min_history, usa o histórico anterior
    para prever o próximo concurso (apenas 1) e verifica se acertou.
    """
    print(f"\n🎯 TESTE CONCURSO A CONCURSO")
    print(f"   Histórico mínimo: {min_history}")
    print(f"   Testes: {len(contests) - min_history - 1} previsões\n")
    
    filters = ['pares', 'moldura', 'primos', 'consecutivos', 'amplitude']
    
    for filtro in filters:
        series = np.array([extract_filter(c['dezenas'], filtro) for c in contests], dtype=float)
        
        acertos_reversao = 0
        acertos_tendencia = 0
        total = 0
        
        for t in range(min_history, len(contests) - 1):
            history = series[:t+1]
            current_val = series[t]
            next_val = series[t+1]
            
            mean_short = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
            mean_long = np.mean(history)
            
            total += 1
            
            # Previsão por reversão
            if mean_short > mean_long:
                pred_rev_down = True
            else:
                pred_rev_down = False
            
            if pred_rev_down and next_val < current_val:
                acertos_reversao += 1
            elif not pred_rev_down and next_val > current_val:
                acertos_reversao += 1
            
            # Previsão por tendência
            if mean_short > mean_long and next_val > current_val:
                acertos_tendencia += 1
            elif mean_short < mean_long and next_val < current_val:
                acertos_tendencia += 1
        
        acc_rev = acertos_reversao / total * 100 if total > 0 else 0
        acc_tend = acertos_tendencia / total * 100 if total > 0 else 0
        
        p_rev = binomtest(acertos_reversao, total, 0.5, alternative='greater').pvalue if total > 0 else 1.0
        p_tend = binomtest(acertos_tendencia, total, 0.5, alternative='greater').pvalue if total > 0 else 1.0
        
        print(f"{filtro:<15}: Reversão={acc_rev:.1f}% ({acertos_reversao}/{total}, p={p_rev:.4f}) | "
              f"Tendência={acc_tend:.1f}% ({acertos_tendencia}/{total}, p={p_tend:.4f})")

# ============================================================
# STRUCTURAL PREDICTOR (MANTIDO)
# ============================================================
class StructuralPredictor:
    def __init__(self, contests):
        self.contests = contests
    
    def predict_ranges(self, method='recent'):
        print(f"\n🔮 STRUCTURAL PREDICTOR (método: {method})")
        filters_info = {
            'pares': {'min': 3, 'max': 12}, 'moldura': {'min': 6, 'max': 15},
            'primos': {'min': 2, 'max': 9}, 'soma': {'min': 120, 'max': 270},
            'amplitude': {'min': 10, 'max': 24}, 'consecutivos': {'min': 0, 'max': 12}
        }
        ranges = {}
        
        for filtro, info in filters_info.items():
            series = np.array([extract_filter(c['dezenas'], filtro) for c in self.contests], dtype=float)
            
            if method == 'recent':
                recent = series[-50:]
                mean_val = np.mean(recent)
                std_val = np.std(recent)
                low = max(info['min'], int(mean_val - std_val))
                high = min(info['max'], int(mean_val + std_val) + 1)
                ranges[filtro] = (low, high)
                print(f"   {filtro:<15}: [{low}, {high}] (média={mean_val:.1f})")
            elif method == 'ipe':
                freq_short = np.mean(series[-20:])
                freq_long = np.mean(series[-500:]) if len(series) >= 500 else np.mean(series)
                ipe = (freq_short - freq_long) / freq_long * 100 if freq_long > 0 else 0.0
                if ipe > 5:
                    predicted = max(info['min'], int(freq_long))
                    ranges[filtro] = (predicted - 1, predicted + 1)
                elif ipe < -5:
                    predicted = min(info['max'], int(freq_long) + 1)
                    ranges[filtro] = (predicted - 1, predicted + 1)
                else:
                    ranges[filtro] = (int(freq_long) - 1, int(freq_long) + 1)
                print(f"   {filtro:<15}: {ranges[filtro]} (IPE={ipe:+.1f}%)")
        
        return ranges

# ============================================================
# WALK‑FORWARD DO STRUCTURAL PREDICTOR (NOVO v47)
# ============================================================
def walk_forward_structural(contests, train_size=500, test_size=50, step=50):
    """
    Walk‑forward usando o Structural Predictor para gerar faixas
    e otimizar a carteira.
    """
    print(f"\n🔬 WALK‑FORWARD DO STRUCTURAL PREDICTOR")
    print(f"   Treino: {train_size}, Teste: {test_size}, Passo: {step}")
    
    results = []
    start = train_size
    while start + test_size <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+test_size]
        
        # Usar o Structural Predictor no treino
        predictor = StructuralPredictor(train_data)
        ranges = predictor.predict_ranges(method='recent')
        
        # Gerar carteira
        opt = PortfolioOptimizer(train_data,
                                 range_pares=ranges.get('pares'),
                                 range_moldura=ranges.get('moldura'),
                                 range_primos=ranges.get('primos'),
                                 range_soma=ranges.get('soma'),
                                 range_amplitude=ranges.get('amplitude'),
                                 range_consecutivos=ranges.get('consecutivos'))
        
        try:
            portfolio = opt.optimize(5, 10000, method='pair_covering')
            bt = opt.backtest(portfolio, test_data)
            
            results.append({
                'start': start,
                'lift': bt['lift'],
                'roi': bt['roi'],
                '14pts': bt['hit_distribution'].get(14, 0),
                '13pts': bt['hit_distribution'].get(13, 0),
            })
            print(f"   Janela {start}: lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}% | "
                  f"13pts={bt['hit_distribution'].get(13,0)} 14pts={bt['hit_distribution'].get(14,0)}")
        except Exception as e:
            print(f"   Janela {start}: ERRO - {e}")
        
        start += step
    
    if results:
        print(f"\n📊 RESUMO:")
        print(f"   Média lift: {np.mean([r['lift'] for r in results]):.3f}")
        print(f"   Média ROI: {np.mean([r['roi'] for r in results]):.1f}%")
        print(f"   Total 13pts: {sum(r['13pts'] for r in results)}")
        print(f"   Total 14pts: {sum(r['14pts'] for r in results)}")
    
    return results

# ============================================================
# BUSCA OOS E COMPARAÇÕES (mantidas)
# ============================================================
def search_best_fixed_oos(contests, n_fixed=3, top_n=20, train_size=3500, n_games=5, n_candidates=10000, method='pair_covering'):
    print(f"\n🔎 BUSCANDO MELHORES {n_fixed} FIXAS (OUT-OF-SAMPLE)")
    train_data = contests[:train_size]
    test_data = contests[train_size:]
    
    candidates = []
    for fixed_tuple in tqdm(combinations(range(1, 26), n_fixed), desc="Filtrando"):
        fixed_set = set(fixed_tuple)
        acertos = sum(1 for c in train_data if fixed_set.issubset(set(c['dezenas'])))
        freq = acertos / len(train_data)
        if freq >= 0.05:
            candidates.append((fixed_tuple, freq, acertos))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for fixed_tuple, freq, acertos in tqdm(candidates[:200], desc="Backtest OOS"):
        opt = PortfolioOptimizer(train_data, fixed=list(fixed_tuple))
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
        opt = PortfolioOptimizer(contests, fixed=list(trinca))
        portfolio = opt.optimize(n_games, n_candidates, method=method)
        bt = opt.backtest(portfolio, contests[-200:])
        print(f"   Trinca {i} ({trinca}): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v47")
    print("   MONTE CARLO + CONCURSO A CONCURSO + WALK‑FORWARD ESTRUTURAL")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira personalizada")
        print("2. Walk‑forward condicional simples")
        print("3. Backtest nos últimos 200 concursos")
        print("4. Buscar melhores fixas (out‑of‑sample)")
        print("5. Comparar duas trincas")
        print("6. Ranking de poder preditivo (reversão vs tendência)")
        print("7. Poder preditivo das 25 dezenas")
        print("8. Structural Predictor (previsão de faixas)")
        print("9. Gerar carteira com previsões estruturais")
        print("10. Controle Monte Carlo (baseline aleatório)")
        print("11. Teste preditivo concurso a concurso")
        print("12. Walk‑forward do Structural Predictor")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            fixed_str = input( "\n   Dezenas fixas (ex: 15 16 20 ou ENTER): ").strip()  
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
              
            semifixed_str = input(
                "   Dezenas semifixas (ex: 03 07 14 25 ou ENTER): ").strip()
            
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            
            if semifixed:
                try:
                    min_semifixed = int(
                        input( f"   Mínimo de semifixas [0-{len(semifixed)}]: " ).strip() or "0" )
            
                    max_semifixed = int( input( f"   Máximo de semifixas [0-{len(semifixed)}]: " ).strip() or str(len(semifixed)) )
            
                except:
                    min_semifixed = 0
                    max_semifixed = len(semifixed)
            else:
                min_semifixed = 0
                max_semifixed = None
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
            metodo = input("\n   Método [1. Pair, 2. Triple]: ").strip() or "1"
            method = 'pair_covering' if metodo == '1' else 'triple_covering'
            
            opt = PortfolioOptimizer(
                contests,
                fixed=fixed,
                semifixed=semifixed,
                min_semifixed=min_semifixed,
                max_semifixed=max_semifixed,
                range_pares=range_pares,
                range_moldura=range_moldura,
                range_primos=range_primos))
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
            
            # Walk‑forward simples
            results = []
            for w in range(8):
                test_end = len(contests) - w * 50
                test_start = test_end - 50
                train_end = test_start
                train_start = max(0, train_end - 400)
                if train_start >= train_end or test_start >= test_end: continue
                opt = PortfolioOptimizer(contests[train_start:train_end], fixed=fixed)
                portfolio = opt.optimize(5, 10000, method=method)
                bt = opt.backtest(portfolio, contests[test_start:test_end])
                results.append({'lift': bt['lift'], 'roi': bt['roi']})
                print(f"   Janela {w}: lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}%")
            if results:
                print(f"\n📊 Média lift: {np.mean([r['lift'] for r in results]):.3f}")
        
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
            blocos_str = input("\n   Tamanhos de bloco (ex: 50,100,200,500) [50,100,200,500]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [50, 100, 200, 500]
            except:
                block_sizes = [50, 100, 200, 500]
            ranker = PredictiveRanking(contests)
            ranker.rank_predictive_power(block_sizes)
        
        elif op == '7':
            blocos_str = input("\n   Tamanhos de bloco (ex: 100,200,500) [100,200,500]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [100, 200, 500]
            except:
                block_sizes = [100, 200, 500]
            ranker = PredictiveRanking(contests)
            ranker.rank_dezenas_individual(block_sizes)
        
        elif op == '8':
            print("\n   Método: 1. Recente (50 concursos)  2. IPE")
            metodo = input("   Escolha [1]: ").strip() or "1"
            method = 'recent' if metodo == '1' else 'ipe'
            predictor = StructuralPredictor(contests)
            predictor.predict_ranges(method=method)
        
        elif op == '9':
            predictor = StructuralPredictor(contests)
            ranges = predictor.predict_ranges(method='recent')
            fixed = [15, 16, 20]
            print(f"\n   Fixas sugeridas: {fixed}")
            gerar = input("   Gerar carteira? (s/n): ").strip().lower()
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
        
        elif op == '10':
            try:
                n_sim = int(input("\n   Número de simulações [100]: ").strip() or "100")
            except: n_sim = 100
            blocos_str = input("   Blocos (ex: 50,100,200) [50,100,200]: ").strip()
            try:
                block_sizes = [int(x) for x in blocos_str.split(',')] if blocos_str else [50, 100, 200]
            except:
                block_sizes = [50, 100, 200]
            monte_carlo_control(contests, n_sim, block_sizes)
        
        elif op == '11':
            try:
                min_hist = int(input("\n   Histórico mínimo [200]: ").strip() or "200")
            except: min_hist = 200
            test_concurso_a_concurso(contests, min_hist)
        
        elif op == '12':
            try:
                train_size = int(input("\n   Tamanho do treino [500]: ").strip() or "500")
                test_size = int(input("   Tamanho do teste [50]: ").strip() or "50")
                step = int(input("   Passo [50]: ").strip() or "50")
            except:
                train_size, test_size, step = 500, 50, 50
            walk_forward_structural(contests, train_size, test_size, step)
        
        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
