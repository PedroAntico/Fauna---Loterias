#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v41
BUSCA AUTOMÁTICA DE FIXAS + RANKING POR BACKTEST REAL

EVOLUÇÃO DO v40:
✅ Opção 5: busca automática com ranking por backtest real (5 jogos)
✅ Comparação direta entre trincas (15,16,19) vs (15,16,20)
✅ Geração de carteira por cobertura de pares (greedy pair covering)
✅ Semifixas com intervalo (min e max)
✅ Análise de lucro/prejuízo e garantias
"""

import numpy as np
from scipy.stats import hypergeom, wilcoxon
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
# GERADOR COM FIXAS E SEMIFIXAS (INTERVALO)
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                     allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed,
                                      allowed_pares, allowed_moldura, allowed_primos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      allowed_pares, allowed_moldura, allowed_primos):
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
            
            return game
        
        return None

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# OTIMIZADOR DE CARTEIRA COM COBERTURA DE PARES
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed if fixed else []
        self.semifixed = semifixed if semifixed else []
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos

    def generate_pool(self, n_candidates):
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed,
                    allowed_pares=self.allowed_pares, allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

    def select_pair_covering(self, candidates, n_select):
        """
        Seleciona n_select jogos maximizando cobertura de pares.
        Algoritmo guloso: a cada passo, escolhe o jogo que adiciona mais pares novos.
        """
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        
        covered_pairs = set()
        selected = []
        masks = [BITMASK_CACHE.get_mask(c) for c in candidates]
        
        for _ in range(n_select):
            best_idx = -1
            best_new = -1
            
            for i, c in enumerate(candidates):
                if i in [candidates.index(s) for s in selected]:
                    continue
                # Calcula pares deste jogo
                pairs = set(combinations(sorted(c), 2))
                new_pairs = len(pairs - covered_pairs)
                if new_pairs > best_new:
                    best_new = new_pairs
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(candidates[best_idx])
            covered_pairs.update(combinations(sorted(candidates[best_idx]), 2))
        
        return selected

    def optimize(self, n_games=5, n_candidates=30000, method='pair_covering'):
        print(f"\n🧩 CARTEIRA COM FIXAS: {n_games} jogos | método: {method}")
        if self.fixed:
            print(f"   Fixas: {self.fixed}")
        if self.semifixed:
            intervalo = f"{self.min_semifixed} a {self.max_semifixed if self.max_semifixed else len(self.semifixed)}"
            print(f"   Semifixas: {self.semifixed} ({intervalo} por jogo)")
        if self.allowed_pares: print(f"   Pares: {self.allowed_pares}")
        if self.allowed_moldura: print(f"   Moldura: {self.allowed_moldura}")
        if self.allowed_primos: print(f"   Primos: {self.allowed_primos}")
        
        t0 = time.time()
        pool = self.generate_pool(n_candidates)
        print(f"   Pool gerado: {len(pool)} jogos")
        
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")
        
        if method == 'pair_covering':
            portfolio = self.select_pair_covering(pool, n_games)
        else:
            # Diversidade simples
            portfolio = self.select_diverse(pool, n_games)
        
        print(f"   Carteira final: {len(portfolio)} jogos")
        print(f"✅ Otimizado em {time.time()-t0:.1f}s")
        return portfolio

    def select_diverse(self, candidates, n_select):
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        
        masks = np.array([BITMASK_CACHE.get_mask(c) for c in candidates], dtype=np.uint32)
        n = len(candidates)
        selected_idx = [0]
        
        for _ in range(n_select - 1):
            min_dists = np.full(n, np.inf, dtype=np.float64)
            for idx in selected_idx:
                intersect = np.array([mask_intersection(masks[i], masks[idx]) for i in range(n)])
                dist = 15.0 - intersect
                min_dists = np.minimum(min_dists, dist)
            min_dists[selected_idx] = -1.0
            valid = np.where(min_dists >= 0)[0]
            if len(valid) == 0:
                break
            next_idx = valid[np.argmax(min_dists[valid])]
            selected_idx.append(next_idx)
        
        for i in range(n):
            if len(selected_idx) >= n_select:
                break
            if i not in selected_idx:
                selected_idx.append(i)
        
        return [candidates[i] for i in selected_idx[:n_select]]

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

    def analyze_fixed_accuracy(self, test_draws=None):
        if test_draws is None:
            test_draws = self.contests
        
        fixed_set = set(self.fixed)
        semifixed_set = set(self.semifixed)
        
        results = {
            'total': len(test_draws),
            'fixed_hit_all': 0,
            'semifixed_in_range': 0,
            'all_conditions_met': 0,
        }
        
        for draw in test_draws:
            draw_set = set(draw['dezenas'])
            fixed_hit = len(fixed_set & draw_set)
            semifixed_hit = len(semifixed_set & draw_set)
            
            if fixed_hit == len(fixed_set):
                results['fixed_hit_all'] += 1
            
            max_semi = self.max_semifixed if self.max_semifixed else len(semifixed_set)
            if self.min_semifixed <= semifixed_hit <= max_semi:
                results['semifixed_in_range'] += 1
            
            if fixed_hit == len(fixed_set) and self.min_semifixed <= semifixed_hit <= max_semi:
                results['all_conditions_met'] += 1
        
        print(f"\n📊 ANÁLISE DE ACERTO DAS FIXAS ({len(test_draws)} concursos):")
        if self.fixed:
            print(f"   Fixas: {self.fixed}")
            print(f"   Acerto total: {results['fixed_hit_all']} "
                  f"({results['fixed_hit_all']/results['total']*100:.1f}%)")
        if self.semifixed:
            intervalo = f"{self.min_semifixed} a {self.max_semifixed if self.max_semifixed else len(self.semifixed)}"
            print(f"   Semifixas: {self.semifixed} (intervalo: {intervalo})")
            print(f"   Acerto no intervalo: {results['semifixed_in_range']} "
                  f"({results['semifixed_in_range']/results['total']*100:.1f}%)")
        if self.fixed or self.semifixed:
            print(f"   Todas as condições: {results['all_conditions_met']} "
                  f"({results['all_conditions_met']/results['total']*100:.1f}%)")
        
        return results

# ============================================================
# BUSCA AUTOMÁTICA COM RANKING POR BACKTEST REAL (v41)
# ============================================================
def search_best_fixed_real(contests, n_fixed=3, top_n=20, min_freq=0.05, n_games=5, n_candidates=10000):
    """
    Testa as melhores combinações de n_fixed dezenas fixas,
    gerando a carteira real e fazendo backtest.
    """
    print(f"\n🔎 BUSCANDO MELHORES {n_fixed} FIXAS (com backtest real)...")
    total_comb = comb(25, n_fixed)
    print(f"   Total de combinações: {total_comb:,}")
    
    # Pré-filtra por frequência para reduzir o espaço
    print("   Fase 1: filtrando por frequência...")
    candidates = []
    all_combos = list(combinations(range(1, 26), n_fixed))
    
    for fixed_tuple in tqdm(all_combos, desc="Filtrando"):
        fixed_set = set(fixed_tuple)
        acertos = sum(1 for c in contests if fixed_set.issubset(set(c['dezenas'])))
        freq = acertos / len(contests)
        if freq >= min_freq:
            candidates.append((fixed_tuple, freq, acertos))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"   {len(candidates)} combinações passaram o filtro (freq ≥ {min_freq:.0%})")
    
    # Fase 2: backtest real para as top 200
    top_candidates = candidates[:200]
    print(f"   Fase 2: backtest real para as {len(top_candidates)} melhores...")
    
    results = []
    for fixed_tuple, freq, acertos in tqdm(top_candidates, desc="Backtest"):
        fixed = list(fixed_tuple)
        opt = PortfolioOptimizer(contests, fixed=fixed)
        try:
            portfolio = opt.optimize(n_games, n_candidates, method='pair_covering')
            bt = opt.backtest(portfolio, contests[-200:])
            
            results.append({
                'fixed': fixed_tuple,
                'freq': freq,
                'acertos': acertos,
                'lift': bt['lift'],
                'roi': bt['roi'],
                '14pts': bt['hit_distribution'].get(14, 0),
                '13pts': bt['hit_distribution'].get(13, 0),
                '15pts': bt['hit_distribution'].get(15, 0),
                'portfolio': portfolio
            })
        except Exception as e:
            continue
    
    # Ordena por ROI
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"\n🏆 TOP {top_n} FIXAS POR ROI (backtest real):")
    print(f"{'Rank':<5} {'Fixas':<20} {'Freq':<8} {'Lift':<8} {'ROI':<10} {'13pts':<8} {'14pts':<8} {'15pts':<8}")
    print("-" * 85)
    for i, res in enumerate(results[:top_n], 1):
        print(f"{i:<5} {str(res['fixed']):<20} {res['freq']:<8.2%} {res['lift']:<8.2f} "
              f"{res['roi']:<10.1f}% {res['13pts']:<8} {res['14pts']:<8} {res['15pts']:<8}")
    
    # Também ordena por 14pts
    results_by_14 = sorted(results, key=lambda x: x['14pts'], reverse=True)
    print(f"\n🏆 TOP {min(10, top_n)} FIXAS POR 14 PONTOS:")
    print(f"{'Rank':<5} {'Fixas':<20} {'14pts':<8} {'13pts':<8} {'ROI':<10}")
    print("-" * 55)
    for i, res in enumerate(results_by_14[:min(10, top_n)], 1):
        print(f"{i:<5} {str(res['fixed']):<20} {res['14pts']:<8} {res['13pts']:<8} {res['roi']:<10.1f}%")
    
    return results, results_by_14

# ============================================================
# COMPARAÇÃO DIRETA DE TRINCAS
# ============================================================
def compare_trincas(contests, trinca1, trinca2, n_games=5, n_candidates=30000):
    """Compara duas trincas de fixas com backtest completo."""
    print(f"\n⚔️ COMPARAÇÃO DE TRINCAS")
    print(f"   Trinca 1: {trinca1}")
    print(f"   Trinca 2: {trinca2}")
    
    for i, trinca in enumerate([trinca1, trinca2], 1):
        print(f"\n--- Trinca {i}: {trinca} ---")
        opt = PortfolioOptimizer(contests, fixed=list(trinca))
        portfolio = opt.optimize(n_games, n_candidates, method='pair_covering')
        
        # Backtest geral
        bt = opt.backtest(portfolio, contests[-200:])
        print(f"   Backtest (200 concursos): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        print(f"   11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
              f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
        
        # Backtest condicionado
        fixed_set = set(trinca)
        cond_draws = [c for c in contests if fixed_set.issubset(set(c['dezenas']))]
        bt_cond = opt.backtest(portfolio, cond_draws)
        print(f"   Condicionado ({len(cond_draws)} concursos): Lift={bt_cond['lift']:.2f}x | ROI={bt_cond['roi']:+.1f}%")
        print(f"   11={bt_cond['hit_distribution'].get(11,0)} 12={bt_cond['hit_distribution'].get(12,0)} "
              f"13={bt_cond['hit_distribution'].get(13,0)} 14={bt_cond['hit_distribution'].get(14,0)} 15={bt_cond['hit_distribution'].get(15,0)}")

# ============================================================
# WALK-FORWARD CONDICIONAL
# ============================================================
def walk_forward_conditional(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                             fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                             allowed_pares=None, allowed_moldura=None, allowed_primos=None):
    print(f"\n🔬 WALK-FORWARD CONDICIONAL ({n_windows} janelas)")
    print(f"   Fixas: {fixed}")
    if semifixed: print(f"   Semifixas: {semifixed} ({min_semifixed} a {max_semifixed if max_semifixed else len(semifixed)})")
    
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
        
        fixed_set = set(fixed)
        semifixed_set = set(semifixed)
        max_semi = max_semifixed if max_semifixed else len(semifixed_set)
        cond_met = sum(1 for d in test_data 
                       if fixed_set.issubset(set(d['dezenas'])) 
                       and min_semifixed <= len(semifixed_set & set(d['dezenas'])) <= max_semi)
        
        opt = PortfolioOptimizer(train_data, fixed, semifixed, min_semifixed, max_semifixed,
                                 allowed_pares, allowed_moldura, allowed_primos)
        portfolio = opt.optimize(n_games, n_candidates=10000, method='pair_covering')
        bt = opt.backtest(portfolio, test_data)
        
        results.append({
            'window': w, 'cond_met': cond_met, 'total_test': len(test_data),
            'lift': bt['lift'], 'roi': bt['roi'], '14pts': bt['hit_distribution'].get(14,0),
        })
        print(f"   Janela {w}: cond={cond_met}/{len(test_data)} | lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}% | 14pts={bt['hit_distribution'].get(14,0)}")
    
    if results:
        print(f"\n📊 RESUMO:")
        print(f"   Média condições atendidas: {np.mean([r['cond_met'] for r in results]):.1f}/{test_size}")
        print(f"   Média lift: {np.mean([r['lift'] for r in results]):.3f}")
        print(f"   14pts total: {sum(r['14pts'] for r in results)}")
    
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v41")
    print("   RANKING POR BACKTEST REAL + COBERTURA DE PARES")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira com dezenas fixas e semifixas")
        print("2. Analisar frequência histórica das fixas")
        print("3. Walk‑forward condicional")
        print("4. Backtest nos últimos 200 concursos")
        print("5. Buscar melhores fixas (backtest real)")
        print("6. Comparar duas trincas de fixas")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS")
            fixed_str = input("   Dezenas fixas (ex: 15 16 19): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            
            semifixed_str = input("   Dezenas semifixas (ex: 5 13 18): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            
            min_semi = 0
            max_semi = None
            if semifixed:
                try:
                    min_semi = int(input(f"   Mínimo de semifixas por jogo (0-{len(semifixed)}): ").strip())
                    min_semi = max(0, min(min_semi, len(semifixed)))
                except:
                    min_semi = 0
                try:
                    max_str = input(f"   Máximo de semifixas por jogo (ENTER = {len(semifixed)}): ").strip()
                    if max_str:
                        max_semi = int(max_str)
                        max_semi = max(min_semi, min(max_semi, len(semifixed)))
                except:
                    max_semi = len(semifixed)
            
            print("\n📝 FILTROS OPCIONAIS (ENTER para pular)")
            pares_str = input("   Pares (ex: 7 8 9): ").strip()
            moldura_str = input("   Moldura (ex: 9 10): ").strip()
            primos_str = input("   Primos (ex: 5 6): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            
            opt = PortfolioOptimizer(contests, fixed, semifixed, min_semi, max_semi,
                                     allowed_pares, allowed_moldura, allowed_primos)
            portfolio = opt.optimize(5, 30000, method='pair_covering')
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x % 2 == 0)
                pr = sum(1 for x in g if x in PRIMES)
                m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (últimos 200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
                print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                      f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
                
                # ROI condicionado
                fixed_set = set(fixed)
                cond_draws = [c for c in contests if fixed_set.issubset(set(c['dezenas']))]
                if cond_draws:
                    bt_cond = opt.backtest(portfolio, cond_draws)
                    print(f"\n💰 ROI CONDICIONADO ({len(cond_draws)} concursos com fixas certas):")
                    print(f"   Lift={bt_cond['lift']:.2f}x | ROI={bt_cond['roi']:+.1f}%")
                    print(f"   11={bt_cond['hit_distribution'].get(11,0)} 12={bt_cond['hit_distribution'].get(12,0)} "
                          f"13={bt_cond['hit_distribution'].get(13,0)} 14={bt_cond['hit_distribution'].get(14,0)} 15={bt_cond['hit_distribution'].get(15,0)}")
        
        elif op == '2':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS PARA ANÁLISE")
            fixed_str = input("   Dezenas fixas (ex: 15 16 19): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            
            semifixed_str = input("   Dezenas semifixas (ex: 5 13 18): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            
            min_semi = 0
            max_semi = None
            if semifixed:
                try:
                    min_semi = int(input(f"   Mínimo de semifixas (0-{len(semifixed)}): ").strip())
                    min_semi = max(0, min(min_semi, len(semifixed)))
                except:
                    min_semi = 0
                try:
                    max_str = input(f"   Máximo de semifixas (ENTER = {len(semifixed)}): ").strip()
                    if max_str:
                        max_semi = int(max_str)
                        max_semi = max(min_semi, min(max_semi, len(semifixed)))
                except:
                    max_semi = len(semifixed)
            
            opt = PortfolioOptimizer(contests, fixed, semifixed, min_semi, max_semi)
            opt.analyze_fixed_accuracy()
        
        elif op == '3':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS PARA WALK‑FORWARD")
            fixed_str = input("   Dezenas fixas (ex: 15 16 19): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            
            semifixed_str = input("   Dezenas semifixas (ex: 5 13 18): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            
            min_semi = 0
            max_semi = None
            if semifixed:
                try:
                    min_semi = int(input(f"   Mínimo de semifixas (0-{len(semifixed)}): ").strip())
                    min_semi = max(0, min(min_semi, len(semifixed)))
                except:
                    min_semi = 0
                try:
                    max_str = input(f"   Máximo de semifixas (ENTER = {len(semifixed)}): ").strip()
                    if max_str:
                        max_semi = int(max_str)
                        max_semi = max(min_semi, min(max_semi, len(semifixed)))
                except:
                    max_semi = len(semifixed)
            
            walk_forward_conditional(contests, fixed=fixed, semifixed=semifixed,
                                     min_semifixed=min_semi, max_semifixed=max_semi)
        
        elif op == '4':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS")
            fixed_str = input("   Dezenas fixas (ex: 15 16 19): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            
            opt = PortfolioOptimizer(contests, fixed=fixed)
            portfolio = opt.optimize(5, 30000, method='pair_covering')
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (últimos 200):")
            print(f"   Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                  f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
        
        elif op == '5':
            print("\n📝 PARÂMETROS DA BUSCA AUTOMÁTICA")
            try:
                n_fixed = int(input("   Quantas dezenas fixas (2, 3 ou 4): ").strip())
                if n_fixed not in [2, 3, 4]:
                    print("   Valor deve ser 2, 3 ou 4.")
                    continue
            except:
                print("   Valor inválido.")
                continue
            
            try:
                top_n = int(input("   Quantos resultados exibir? [20]: ").strip() or "20")
            except:
                top_n = 20
            
            search_best_fixed_real(contests, n_fixed, top_n)
        
        elif op == '6':
            print("\n📝 COMPARAÇÃO DE DUAS TRINCAS")
            trinca1_str = input("   Trinca 1 (ex: 15 16 19): ").strip()
            trinca2_str = input("   Trinca 2 (ex: 15 16 20): ").strip()
            try:
                trinca1 = tuple(int(x) for x in trinca1_str.split())
                trinca2 = tuple(int(x) for x in trinca2_str.split())
                if len(trinca1) != 3 or len(trinca2) != 3:
                    print("   Cada trinca deve ter exatamente 3 dezenas.")
                    continue
            except:
                print("   Valores inválidos.")
                continue
            
            compare_trincas(contests, trinca1, trinca2)
        
        elif op == '0':
            break
        
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
